package nvme

import (
	"sync/atomic"

	"github.com/zeebo/xxh3"
)

// readReq is one queued device read. The submitter already holds the
// segment read-acquire; the worker does the I/O and the verify. claimed is
// the abandonment handshake: whoever CASes it false→true owns the outcome —
// the worker's win commits the result (and its buffer) to done; the
// caller's win (ctx cancel / shutdown) transfers cleanup of the read-hold
// and any completed buffer to the worker side. One owner, always.
type readReq struct {
	seg     *segment
	loc     Loc
	ns      uint32
	key     [32]byte
	want    uint64
	done    chan readResult
	claimed *atomic.Bool
}

type readResult struct {
	st   ReadStatus
	buf  []byte // the pooled buffer (Put back via release)
	data []byte // payload subslice of buf, valid until release
}

// readerLoop is one pool worker: pread the aligned record span, verify
// header magic + nskey + payload xxh3 against the index's expectation, and
// only then hand bytes out. Failures of the RECORD BYTES map to ReadCorrupt
// (the caller self-heals the index entry; a block is never served
// unverified); a buffer-pool failure is ReadBusy — retryable, see readOne.
// On readStop the workers drain the queue so no caller is ever left blocked
// (readq itself is never closed — a straggler send can therefore never
// panic); a drained request whose caller already abandoned it gets its
// read-hold released here instead of a reply.
func (v *Volume) readerLoop() {
	defer v.readerWG.Done()
	for {
		select {
		case req := <-v.readq:
			v.serveRead(req)
		case <-v.readStop:
			for {
				select {
				case req := <-v.readq:
					if req.claimed.CompareAndSwap(false, true) {
						req.done <- readResult{st: ReadGone}
					} else {
						req.seg.releaseRead()
					}
				default:
					return
				}
			}
		}
	}
}

// serveRead runs one request through the claim handshake (see readReq).
func (v *Volume) serveRead(req readReq) {
	if req.claimed.Load() {
		// Abandoned while queued: skip the device I/O entirely; the caller
		// left the read-hold for us to drop.
		req.seg.releaseRead()
		return
	}
	res := v.readOne(req)
	if req.claimed.CompareAndSwap(false, true) {
		req.done <- res // buffered: the (sole) caller owns buf + hold now
		return
	}
	// The caller abandoned us mid-read: the CAS loser cleans up — put the
	// pooled buffer back and drop the read-hold the caller left behind.
	if res.st == ReadOK {
		v.pool.Put(res.buf)
	}
	req.seg.releaseRead()
}

func (v *Volume) readOne(req readReq) readResult {
	span := recordSpan(req.loc.Len)
	buf, err := v.pool.Get(uint32(span)) //nolint:gosec // G115: span ≤ recordSpan(MaxBlobLen) < 4 GiB
	if err != nil {
		// Busy, never corrupt: the error taxonomy separates "retry me"
		// from "destroy the entry". ReadCorrupt's contract self-heals the
		// index entry — classifying a pool failure (mmap ENOMEM under
		// memory pressure, or an impossible span) that way deleted healthy
		// on-disk blocks exactly when the tier was supposed to absorb the
		// pressure. Corrupt is reserved for failures of the record bytes.
		v.log.Warn("nvme: read buffer", "err", err)
		return readResult{st: ReadBusy}
	}
	chunk := buf[:span]
	if err := req.seg.f.ReadAt(chunk, int64(req.loc.Offset)); err != nil {
		v.pool.Put(buf)
		v.log.Warn("nvme: pread failed", "segment", req.loc.SegmentID, "off", req.loc.Offset, "err", err)
		return readResult{st: ReadCorrupt}
	}
	h, err := parseRecordHeader(chunk, v.p.MaxBlobLen)
	if err != nil || h.NS != req.ns || h.Key != req.key || h.Len != req.loc.Len || h.XXH3 != req.want {
		v.pool.Put(buf)
		return readResult{st: ReadCorrupt}
	}
	payload := chunk[recordHdrSize : recordHdrSize+int(h.Len)]
	if xxh3.Hash(payload) != req.want {
		v.pool.Put(buf)
		return readResult{st: ReadCorrupt}
	}
	return readResult{st: ReadOK, buf: buf, data: payload}
}
