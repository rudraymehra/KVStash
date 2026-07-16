package server

import (
	"sync"
	"time"

	"github.com/zeebo/xxh3"

	"github.com/kvstash/kvblockd/internal/protocol"
	"github.com/kvstash/kvblockd/internal/transport"
)

// session is one connection's server-side state. It is both the
// transport.BufferSource (it lends request-body buffers) and the
// transport.FrameHandler (it dispatches frames). Handlers run on the
// connection's single read goroutine, so per-session fields need locking only
// against the PUT-stream reaper, which runs concurrently.
type session struct {
	srv  *Server
	conn *transport.Conn

	authed bool
	ns     uint32
	limits protocol.Limits
	feats  uint64

	keyScratch [][32]byte // reused across batch decodes on the read goroutine
	lendBuf    []byte     // reusable Lend extent (see Lend for the safety invariant)

	streamMu    sync.Mutex
	streams     map[uint64]*putStream // by request_id
	stagedBytes int64                 // total live PUT staging on this conn (cap: maxStagedPerConn)
}

// putStream is one in-flight PUT_STREAM (§5). buf accumulates CHUNK bytes into
// staging that is invisible to reads until a successful COMMIT. digest hashes
// each chunk as it lands (cache-hot, right after the staging copy) so COMMIT
// verifies with a constant-time Sum64 instead of re-reading the whole block
// from DRAM — one full memory pass saved per PUT.
type putStream struct {
	ns           uint32
	key          [32]byte
	totalLen     uint32
	xxh3Hint     uint64
	buf          []byte
	digest       *xxh3.Hasher
	received     uint32
	tombstoned   bool // BEGIN rejected / OK_EXISTS / reaped: consume chunks, fail COMMIT
	lastActive   time.Time
	tombstonedAt time.Time // when tombstoned, for the reaper's grace-period cleanup
}

func newSession(srv *Server) *session {
	return &session{
		srv:     srv,
		limits:  srv.cfg.WireLimits(),
		streams: make(map[uint64]*putStream),
	}
}

// bind attaches the accepted connection and starts the PUT-stream reaper.
func (s *session) bind(c *transport.Conn) {
	s.conn = c
	go s.reaper(c)
}

// maxLendReuse bounds the buffer session.Lend recycles across frames. A body
// up to this size is served from the recycled extent (recycling kills the
// per-1MiB-chunk allocation that dominated the PUT path); a larger body gets a
// fresh make() that GC reclaims after Return — so a single big frame can never
// pin up to the credit window (256 MiB) for the connection's whole life,
// invisible to every staging cap. 2 MiB covers the 0.4–2.5 MiB block range
// (the recommended chunk ceiling) with headroom.
const maxLendReuse = 2 << 20

// Lend / Return implement transport.BufferSource. Every HandleFrame path
// consumes (Returns) its body synchronously on the connection's one read
// goroutine before the transport reads the next frame, so at most one lent
// buffer is live at a time. The DRAM tier will replace this with arena extents
// so PUT bytes land off-heap.
func (s *session) Lend(n int) []byte {
	if n > maxLendReuse {
		return make([]byte, n) // oversize: transient, GC-reclaimed after Return
	}
	if cap(s.lendBuf) < n {
		s.lendBuf = make([]byte, maxLendReuse)
	}
	return s.lendBuf[:n]
}
func (s *session) Return(b []byte) { transport.Poison(b) }

// HandleFrame is the dispatch entry point (transport.FrameHandler). It enforces
// HELLO-first, then routes by opcode. Every path Returns the lent body and
// Grants its payload bytes exactly once (via consume) — the credit-conservation
// contract the transport relies on.
func (s *session) HandleFrame(c *transport.Conn, h protocol.Header, body []byte) {
	if !s.authed {
		if h.Opcode != protocol.OpHello {
			// Pre-auth traffic is one of the three connection-fatal things (§9).
			s.consume(c, h, body)
			c.Abort(protocol.StatusErrAuthRequired)
			return
		}
		s.handleHello(c, h, body)
		return
	}

	if h.Flags&protocol.FlagResp != 0 {
		// A request must not set F_RESP (§2): skip it, answer ERR_MALFORMED,
		// connection stays healthy.
		s.consume(c, h, body)
		s.respondStatus(c, h, protocol.StatusErrMalformed)
		return
	}

	switch h.Opcode {
	case protocol.OpNop:
		// The transport swallows NOPs, but be defensive: a NOP is control —
		// consume it, never answer it (§3).
		s.consume(c, h, body)
	case protocol.OpHello:
		// A second HELLO on an established connection is malformed (§3.1);
		// limits are fixed for the connection's lifetime.
		s.consume(c, h, body)
		s.respondStatus(c, h, protocol.StatusErrMalformed)
	case protocol.OpBatchExists:
		s.handleBatchExists(c, h, body)
	case protocol.OpBatchGet:
		s.handleBatchGet(c, h, body)
	case protocol.OpPutStream:
		s.handlePutStream(c, h, body)
	case protocol.OpTouchLease:
		s.handleKeyStatusVerb(c, h, body)
	case protocol.OpPin:
		s.handleKeyStatusVerb(c, h, body)
	case protocol.OpDelete:
		s.handleDelete(c, h, body)
	case protocol.OpStats:
		s.handleStats(c, h, body)
	default:
		// Unknown opcode: skippable, connection stays healthy (§9).
		s.consume(c, h, body)
		s.respondStatus(c, h, protocol.StatusErrUnsupported)
	}
}

// consume Returns the lent body and Grants its payload bytes back to the
// client's credit window. Call exactly once per frame, as soon as the body has
// been read into owned structures.
func (s *session) consume(c *transport.Conn, h protocol.Header, body []byte) {
	s.Return(body)
	c.GrantCredit(h.PayloadLen)
}

// respondStatus writes a preamble-only response (status, count=0) — the ack
// shape for PUT sub-ops and error replies (§3).
func (s *session) respondStatus(c *transport.Conn, h protocol.Header, status protocol.Status) {
	body := protocol.AppendPreamble(make([]byte, 0, protocol.PreambleSize), status, 0)
	s.writeResp(c, h, body)
}

// writeResp queues a single-slice response frame echoing the request's
// opcode/namespace/request_id with F_RESP set. No release: the body is a fresh
// buffer the transport owns after the write.
func (s *session) writeResp(c *transport.Conn, h protocol.Header, body []byte) {
	s.writeResp2(c, h, netBuffers(body), nil)
}

// reaper tombstones PUT streams idle longer than the negotiated timeout (§5),
// freeing their staging. It exits when the connection is done.
func (s *session) reaper(c *transport.Conn) {
	timeout := time.Duration(s.srv.cfg.StreamTimeoutMS) * time.Millisecond
	tick := timeout / 4
	if tick <= 0 {
		tick = time.Second
	}
	t := time.NewTicker(tick)
	defer t.Stop()
	for {
		select {
		case <-c.Done():
			return
		case now := <-t.C:
			s.sweepStreams(now, timeout)
		}
	}
}

// sweepStreams is one reaper pass: tombstones idle live streams (freeing their
// staging) and deletes tombstones past the grace period. Factored out of the
// reaper loop so tests can drive it with a pinned clock.
func (s *session) sweepStreams(now time.Time, timeout time.Duration) {
	s.streamMu.Lock()
	defer s.streamMu.Unlock()
	for id, st := range s.streams {
		switch {
		case st.tombstoned:
			// Reclaim tombstones after a grace period so a client reusing many
			// request_ids can't grow the map unbounded (§5 only bounds
			// tombstones at connection close otherwise).
			if now.Sub(st.tombstonedAt) > timeout {
				delete(s.streams, id)
			}
		case now.Sub(st.lastActive) > timeout:
			// Idle live stream: free the staging extent AND the digest, then
			// tombstone the id until it gets a terminal response or is reaped.
			s.stagedBytes -= int64(len(st.buf))
			st.buf = nil
			st.digest = nil
			st.tombstoned = true
			st.tombstonedAt = now
		}
	}
}
