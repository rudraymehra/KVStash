package server

import (
	"net"
	"time"

	"github.com/zeebo/xxh3"

	"github.com/kvstash/kvblockd/internal/protocol"
	"github.com/kvstash/kvblockd/internal/transport"
)

// Per-connection PUT_STREAM resource bounds. BEGIN reserves NOTHING (staging
// grows lazily as CHUNKs arrive), and both the live-stream count and the total
// staged bytes are capped — so a client cannot pin server memory with cheap
// BEGIN-only frames (the amplification the review found: 4 KiB of BEGINs pinned
// ~191 MiB). Excess is refused with ERR_BUSY (backpressure), not accepted.
const (
	maxLiveStreams   = 256
	maxStagedPerConn = 256 << 20 // 256 MiB of in-flight PUT staging per connection
)

// writeResp2 queues a multi-slice response (the BATCH_GET one-writev shape).
func (s *session) writeResp2(c *transport.Conn, h protocol.Header, bufs net.Buffers, release func()) {
	resp := protocol.Header{
		Opcode:      h.Opcode,
		Flags:       protocol.FlagResp,
		NamespaceID: h.NamespaceID,
		RequestID:   h.RequestID,
	}
	_ = c.WriteFrames(resp, bufs, release)
}

// handlePutStream drives the two-phase state machine (§3.4, §5). The block key
// is in the header on all four sub-ops; request_id binds chunks to their
// staging extent. Invariant: BEGIN, COMMIT, and ABORT each get exactly one
// response; CHUNK (and tombstone-discarded chunks) get none.
func (s *session) handlePutStream(c *transport.Conn, h protocol.Header, body []byte) {
	switch protocol.SubOp(h.Flags) {
	case protocol.PutBegin:
		s.putBegin(c, h, body)
	case protocol.PutChunk:
		s.putChunk(c, h, body)
	case protocol.PutCommit:
		s.putCommit(c, h, body)
	case protocol.PutAbort:
		s.putAbort(c, h, body)
	default:
		// Unknown sub-op is skippable, connection healthy (§9).
		s.consume(c, h, body)
		s.respondStatus(c, h, protocol.StatusErrUnsupported)
	}
}

// liveStreamCount returns the number of non-tombstoned streams. Caller holds
// streamMu.
func (s *session) liveStreamCount() int {
	n := 0
	for _, st := range s.streams {
		if !st.tombstoned {
			n++
		}
	}
	return n
}

func (s *session) putBegin(c *transport.Conn, h protocol.Header, body []byte) {
	b, err := protocol.DecodePutBegin(body)
	s.consume(c, h, body)
	if err != nil {
		s.respondStatus(c, h, protocol.ErrorStatus(err))
		return
	}

	s.streamMu.Lock()
	defer s.streamMu.Unlock()

	if existing, ok := s.streams[h.RequestID]; ok && !existing.tombstoned {
		// BEGIN on a request_id with a LIVE stream (§5): malformed; the original
		// stream is unaffected. (A lingering TOMBSTONE, by contrast, is a reused
		// id whose prior stream never got a terminal response — start fresh.)
		s.respondStatus(c, h, protocol.StatusErrMalformed)
		return
	}
	// Write-once idempotent hit: the block is already sealed, so tell the
	// client to stop and tombstone the id (optimistic chunks get discarded).
	if s.srv.store.Contains(s.ns, h.Key) {
		s.tombstone(h.RequestID, h.Key)
		s.respondStatus(c, h, protocol.StatusOKExists)
		return
	}
	if b.TotalLen > s.limits.MaxBlobLen {
		s.tombstone(h.RequestID, h.Key)
		s.respondStatus(c, h, protocol.StatusErrTooLarge)
		return
	}
	if s.liveStreamCount() >= maxLiveStreams {
		// Too many concurrent streams on this connection: backpressure, don't
		// allocate. No tombstone (the id was never accepted).
		s.respondStatus(c, h, protocol.StatusErrBusy)
		return
	}
	s.streams[h.RequestID] = &putStream{
		ns:         s.ns,
		key:        h.Key,
		totalLen:   b.TotalLen,
		xxh3Hint:   b.XXH3Hint,
		lastActive: nowFn(),
		// buf is nil: staging grows lazily as CHUNKs arrive (no eager reserve).
	}
	s.respondStatus(c, h, protocol.StatusOK)
}

func (s *session) putChunk(c *transport.Conn, h protocol.Header, body []byte) {
	// Copy the chunk into staging BEFORE consuming (Return recycles/poisons the
	// lent buffer). CHUNK is never answered.
	s.streamMu.Lock()
	st, ok := s.streams[h.RequestID]
	switch {
	case !ok || st.tombstoned:
		// Unknown or tombstoned: discard silently.
	case h.PayloadLen == 0:
		// Zero-length CHUNK is permitted but does NOT reset the inactivity timer
		// (§3.4) — an idle client cannot pin staging with free empty frames.
	case st.received+h.PayloadLen > st.totalLen:
		// Overflow beyond the declared length → tombstone; COMMIT will fail.
		s.stagedBytes -= int64(len(st.buf))
		st.buf = nil
		st.tombstoned = true
		st.tombstonedAt = nowFn()
	case s.stagedBytes+int64(h.PayloadLen) > maxStagedPerConn:
		// Per-connection staging cap reached: tombstone this stream rather than
		// grow unbounded (the COMMIT then fails with ERR_STALE_STREAM).
		s.stagedBytes -= int64(len(st.buf))
		st.buf = nil
		st.tombstoned = true
		st.tombstonedAt = nowFn()
	default:
		st.buf = append(st.buf, body...)
		st.received += h.PayloadLen
		s.stagedBytes += int64(h.PayloadLen)
		st.lastActive = nowFn()
	}
	s.streamMu.Unlock()
	s.consume(c, h, body)
}

func (s *session) putCommit(c *transport.Conn, h protocol.Header, body []byte) {
	sum, err := protocol.DecodePutCommit(body)
	s.consume(c, h, body)
	if err != nil {
		s.respondStatus(c, h, protocol.ErrorStatus(err))
		return
	}

	s.streamMu.Lock()
	st, ok := s.streams[h.RequestID]
	if !ok || st.tombstoned {
		delete(s.streams, h.RequestID) // terminal response clears any tombstone
		s.streamMu.Unlock()
		s.respondStatus(c, h, protocol.StatusErrStaleStream)
		return
	}
	// Take the stream out of the table under the lock; validate/commit outside.
	delete(s.streams, h.RequestID)
	s.stagedBytes -= int64(len(st.buf))
	buf, totalLen := st.buf, st.totalLen
	s.streamMu.Unlock()

	if st.received != totalLen {
		s.respondStatus(c, h, protocol.StatusErrShortStream)
		return
	}
	if xxh3.Hash(buf) != sum {
		s.respondStatus(c, h, protocol.StatusErrChecksum)
		return
	}
	status := s.srv.store.Put(st.ns, st.key, buf, sum)
	s.respondStatus(c, h, status)
}

func (s *session) putAbort(c *transport.Conn, h protocol.Header, body []byte) {
	s.consume(c, h, body)
	s.streamMu.Lock()
	st, ok := s.streams[h.RequestID]
	if ok {
		s.stagedBytes -= int64(len(st.buf))
		delete(s.streams, h.RequestID)
	}
	s.streamMu.Unlock()
	if !ok || st.tombstoned {
		s.respondStatus(c, h, protocol.StatusErrStaleStream)
		return
	}
	s.respondStatus(c, h, protocol.StatusOK)
}

// tombstone records a request_id whose stream must not stage bytes (OK_EXISTS,
// too-large, or over-cap). Caller holds streamMu.
func (s *session) tombstone(id uint64, key [32]byte) {
	s.streams[id] = &putStream{ns: s.ns, key: key, tombstoned: true, tombstonedAt: nowFn()}
}

// nowFn is time.Now, indirected so tests can pin stream timing.
var nowFn = time.Now
