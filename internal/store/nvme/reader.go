package nvme

import (
	"github.com/zeebo/xxh3"
)

// readReq is one queued device read. The submitter already holds the
// segment read-acquire; the worker does the I/O and the verify.
type readReq struct {
	seg  *segment
	loc  Loc
	ns   uint32
	key  [32]byte
	want uint64
	done chan readResult
}

type readResult struct {
	st   ReadStatus
	buf  []byte // the pooled buffer (Put back via release)
	data []byte // payload subslice of buf, valid until release
}

// readerLoop is one pool worker: pread the aligned record span, verify
// header magic + nskey + payload xxh3 against the index's expectation, and
// only then hand bytes out. Every failure class maps to ReadCorrupt — the
// caller self-heals the index entry; a block is never served unverified.
// On readStop the workers drain the queue with ReadGone replies so no
// caller is ever left blocked (readq itself is never closed — a straggler
// send can therefore never panic).
func (v *Volume) readerLoop() {
	defer v.readerWG.Done()
	for {
		select {
		case req := <-v.readq:
			req.done <- v.readOne(req)
		case <-v.readStop:
			for {
				select {
				case req := <-v.readq:
					req.done <- readResult{st: ReadGone}
				default:
					return
				}
			}
		}
	}
}

func (v *Volume) readOne(req readReq) readResult {
	span := recordSpan(req.loc.Len)
	buf, err := v.pool.Get(uint32(span)) //nolint:gosec // G115: span ≤ recordSpan(MaxBlobLen) < 4 GiB
	if err != nil {
		v.log.Warn("nvme: read buffer", "err", err)
		return readResult{st: ReadCorrupt}
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
