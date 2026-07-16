package transport

import (
	"errors"
	"io"
	"net"
	"sync"
	"sync/atomic"
	"time"

	"github.com/kvstash/kvblockd/internal/protocol"
)

// FrameHandler is the server's entry point. HandleFrame runs on the
// connection's read goroutine: blocking blocks this connection only (the
// documented trade — a worker handoff is a server-layer upgrade, not a
// transport concern). body ownership transfers to the handler per the
// BufferSource protocol; the handler must Return it, directly or via a
// WriteFrames release, and must Grant the frame's payload bytes once drained.
type FrameHandler interface {
	HandleFrame(c *Conn, h protocol.Header, body []byte)
}

// ErrConnClosed is returned by WriteFrames after Close/Abort.
var ErrConnClosed = errors.New("transport: connection closed")

// maxFlushFrames bounds one writev flush: each frame contributes one header
// iovec plus its payload iovecs, so 512 frames stays comfortably under the
// kernel's IOV_MAX (1024) even with multi-slice payloads.
const maxFlushFrames = 512

// writeReq is one queued response frame. release fires after the kernel has
// accepted the bytes (success or error) — the §12 hold-until-written rule for
// arena refcounts, and the exact seam a future MSG_ZEROCOPY writer needs
// (fire on errqueue completion instead).
type writeReq struct {
	hdr     protocol.Header
	bufs    net.Buffers
	release func()
}

// Conn is one accepted connection: a read loop feeding the handler and a
// write loop coalescing responses into large writev flushes.
type Conn struct {
	nc      net.Conn
	bs      BufferSource
	handler FrameHandler
	cfg     Config

	credit     CreditWindow
	maxPayload atomic.Uint32 // ParseHeader cap: PreNegCap until SetLimits

	wq        chan writeReq
	closed    chan struct{}
	closeOnce sync.Once
	done      chan struct{}

	discardBuf []byte // per-conn scratch for skipping unwanted payloads
}

// startConn wires the loops around an accepted (already tuned) net.Conn.
func startConn(nc net.Conn, bs BufferSource, h FrameHandler, cfg Config) *Conn {
	c := &Conn{
		nc:         nc,
		bs:         bs,
		handler:    h,
		cfg:        cfg,
		wq:         make(chan writeReq, 256),
		closed:     make(chan struct{}),
		done:       make(chan struct{}),
		discardBuf: make([]byte, 64<<10),
	}
	c.maxPayload.Store(cfg.PreNegCap)
	c.credit.SetWindow(cfg.PreNegCap, cfg.PreNegCap)

	var wg sync.WaitGroup
	wg.Add(2)
	go func() { defer wg.Done(); c.readLoop() }()
	go func() { defer wg.Done(); c.writeLoop() }()
	go func() { wg.Wait(); close(c.done) }()
	return c
}

// SetLimits installs the post-HELLO negotiated limits: the frame cap the
// parser enforces and the credit window (§8 rule 1). The server calls this
// exactly once, after a successful HELLO.
func (c *Conn) SetLimits(l protocol.Limits) {
	c.maxPayload.Store(l.MaxFrameLen)
	c.credit.SetWindow(l.InitialCredit, l.MaxFrameLen)
}

// GrantCredit returns n drained payload bytes to the client's window
// (handlers call this once they have consumed a frame's bytes).
func (c *Conn) GrantCredit(n uint32) { c.credit.Grant(n) }

// WriteFrames queues one response frame: hdr plus the payload slices, emitted
// together in one writev (with other queued frames when the writer is
// backlogged). hdr.PayloadLen is computed here from bufs; hdr.Credit is
// overwritten by the writer with any pending grant. release (may be nil)
// fires after the kernel accepted the bytes — hold arena refcounts until then.
func (c *Conn) WriteFrames(hdr protocol.Header, bufs net.Buffers, release func()) error {
	var total uint64
	for _, b := range bufs {
		total += uint64(len(b))
	}
	hdr.PayloadLen = uint32(total) //nolint:gosec // G115: callers respect the negotiated max_frame_len (u32); a server-side overflow here would be a server bug, caught by the deterministic PayloadLen mismatch on the client
	// Check closed FIRST: after Close the writer may already have drained its
	// queue for the last time, so winning the send race would strand the
	// request (and its release) in a channel nobody reads.
	select {
	case <-c.closed:
		if release != nil {
			release()
		}
		return ErrConnClosed
	default:
	}
	select {
	case c.wq <- writeReq{hdr: hdr, bufs: bufs, release: release}:
		return nil
	case <-c.closed:
		if release != nil {
			release()
		}
		return ErrConnClosed
	}
}

// Abort sends a best-effort §9 protocol-fatal report frame (opcode 0,
// F_RESP|F_FATAL, request_id 0, preamble payload) and closes the connection.
// Receivers MUST NOT rely on the report arriving — and neither do we: if the
// write queue is full, we just close.
func (c *Conn) Abort(status protocol.Status) {
	body := protocol.AppendErrorResp(make([]byte, 0, protocol.PreambleSize), status)
	hdr := protocol.Header{
		Opcode:     protocol.OpNop,
		Flags:      protocol.FlagResp | protocol.FlagFatal,
		PayloadLen: uint32(len(body)), //nolint:gosec // G115: PreambleSize (8)
	}
	select {
	case c.wq <- writeReq{hdr: hdr, bufs: net.Buffers{body}}:
	default:
	}
	_ = c.Close()
}

// Close tears the connection down: both loops exit (in-flight I/O is
// unblocked via an immediate deadline), the writer flushes what it can and
// closes the socket. Safe to call multiple times, from any goroutine.
func (c *Conn) Close() error {
	c.closeOnce.Do(func() {
		close(c.closed)
		_ = c.nc.SetDeadline(time.Now()) // unblock in-flight ReadFull/WriteTo
	})
	return nil
}

// Done is closed when both loops have exited and the socket is closed.
func (c *Conn) Done() <-chan struct{} { return c.done }

// respondError enqueues a transport-owned canned error response — used for
// frames that structurally cannot reach a handler (oversize, credit breach),
// where not answering would leave the client waiting forever.
func (c *Conn) respondError(h protocol.Header, status protocol.Status) {
	body := protocol.AppendErrorResp(make([]byte, 0, protocol.PreambleSize), status)
	hdr := protocol.Header{
		Opcode:      h.Opcode,
		Flags:       protocol.FlagResp,
		NamespaceID: h.NamespaceID,
		RequestID:   h.RequestID,
		PayloadLen:  uint32(len(body)), //nolint:gosec // G115: PreambleSize (8)
	}
	select {
	case c.wq <- writeReq{hdr: hdr, bufs: net.Buffers{body}}:
	case <-c.closed:
	}
}

// discard reads and drops exactly n payload bytes — the skip path for frames
// we refuse. It never Lends a buffer, so a hostile length can never size an
// allocation (the drain() discipline from the transport rig).
func (c *Conn) discard(n uint32) error {
	_, err := io.CopyBuffer(struct{ io.Writer }{io.Discard}, io.LimitReader(c.nc, int64(n)), c.discardBuf)
	return err
}

// readLoop: ReadFull(64B) → ParseHeader(negotiated cap) → route.
func (c *Conn) readLoop() {
	defer c.Close()
	hdrBuf := make([]byte, protocol.HeaderSize)
	for {
		if c.cfg.IdleReadTimeout > 0 {
			_ = c.nc.SetReadDeadline(time.Now().Add(c.cfg.IdleReadTimeout))
		}
		if _, err := io.ReadFull(c.nc, hdrBuf); err != nil {
			return // peer closed, idle timeout, or Close(); nothing to report
		}

		h, err := protocol.ParseHeader(hdrBuf, c.maxPayload.Load())
		switch {
		case errors.Is(err, protocol.ErrPayloadTooLarge):
			// The populated-header path the codec was designed for: the CRC
			// authenticated PayloadLen, so skip exactly that many bytes,
			// answer ERR_TOO_LARGE echoing the request, and re-grant the
			// bytes (§8 amended rule 4: skipped bytes are still re-granted).
			c.credit.Consume(h.PayloadLen)
			if derr := c.discard(h.PayloadLen); derr != nil {
				return
			}
			c.credit.Grant(h.PayloadLen)
			c.respondError(h, protocol.StatusErrTooLarge)
			continue
		case err != nil:
			// Fatal by classification (magic/version/CRC): report, close, no
			// resynchronization attempt, ever (PROTOCOL.md §1).
			c.Abort(protocol.StatusFatalProtocol)
			return
		}

		switch v := c.credit.Consume(h.PayloadLen); v {
		case ViolationBusy, ViolationFatal:
			if derr := c.discard(h.PayloadLen); derr != nil {
				return
			}
			c.credit.Grant(h.PayloadLen)
			if v == ViolationFatal {
				c.Abort(protocol.StatusErrBusy)
				return
			}
			c.respondError(h, protocol.StatusErrBusy)
			continue
		case ViolationNone:
		}

		if h.Opcode == protocol.OpNop {
			// Keepalive / client-side credit noise: tolerate any payload by
			// skipping it (§3), grant the bytes back, never respond.
			if derr := c.discard(h.PayloadLen); derr != nil {
				return
			}
			c.credit.Grant(h.PayloadLen)
			continue
		}

		body := c.bs.Lend(int(h.PayloadLen))
		if _, err := io.ReadFull(c.nc, body); err != nil {
			c.bs.Return(body)
			return
		}
		// Ownership of body transfers to the handler here (Return + Grant are
		// its responsibility, directly or via a WriteFrames release).
		c.handler.HandleFrame(c, h, body)
	}
}

// writeLoop: block for the first request, then drain the queue non-blocking
// into one coalesced writev flush. Coalescing NEVER waits for more frames —
// a lone sub-millisecond EXISTS reply flushes immediately; 16 MiB batches
// emerge only when the queue is genuinely backlogged, which is exactly when
// throughput matters.
func (c *Conn) writeLoop() {
	hdrArena := make([]byte, protocol.HeaderSize*maxFlushFrames)
	reqs := make([]writeReq, 0, maxFlushFrames)
	iovs := make(net.Buffers, 0, 2*maxFlushFrames)

	grantTick := time.NewTicker(c.cfg.grantTickInterval())
	defer grantTick.Stop()

	for {
		select {
		case first := <-c.wq:
			reqs = append(reqs[:0], first)
			total := first.payloadLen()
			for total < uint64(c.cfg.CoalesceBytes) && len(reqs) < maxFlushFrames { //nolint:gosec // G115: CoalesceBytes is a validated config constant (16 MiB)
				select {
				case r := <-c.wq:
					reqs = append(reqs, r)
					total += r.payloadLen()
				default:
					total = uint64(c.cfg.CoalesceBytes) //nolint:gosec // G115: as above; queue momentarily empty: flush now
				}
			}
			if !c.flush(hdrArena, reqs, &iovs) {
				c.drainReleases()
				_ = c.nc.Close()
				return
			}

		case <-grantTick.C:
			// §8 rule 4: unsolicited NOP/CREDIT when there are pending grants
			// and nothing else to ride on.
			if c.credit.PendingGrant() {
				reqs = append(reqs[:0], writeReq{hdr: protocol.Header{Opcode: protocol.OpNop}})
				if !c.flush(hdrArena, reqs, &iovs) {
					c.drainReleases()
					_ = c.nc.Close()
					return
				}
			}

		case <-c.closed:
			// Best-effort final flush of whatever is already queued (the
			// Abort report frame rides this path), then close the socket.
			_ = c.nc.SetWriteDeadline(time.Now().Add(time.Second))
			for {
				select {
				case r := <-c.wq:
					reqs = append(reqs[:0], r)
					_ = c.flush(hdrArena, reqs, &iovs)
				default:
					c.drainReleases()
					_ = c.nc.Close()
					return
				}
			}
		}
	}
}

func (r writeReq) payloadLen() uint64 { return uint64(r.hdr.PayloadLen) }

// flush emits one writev for the batch: iovecs alternate marshalled headers
// (from the per-flush arena) and payload slices. The first header of the
// flush carries any pending credit grant. Releases fire after WriteTo
// returns, success or error (§12: buffers are held until the kernel accepted
// the bytes). Returns false when the connection is dead.
func (c *Conn) flush(hdrArena []byte, reqs []writeReq, iovs *net.Buffers) bool {
	*iovs = (*iovs)[:0]
	for i := range reqs {
		if i == 0 {
			reqs[i].hdr.Credit = c.credit.TakeGrant()
		} else {
			reqs[i].hdr.Credit = 0
		}
		slot := hdrArena[i*protocol.HeaderSize : (i+1)*protocol.HeaderSize]
		reqs[i].hdr.MarshalTo(slot)
		*iovs = append(*iovs, slot)
		*iovs = append(*iovs, reqs[i].bufs...)
	}

	if c.cfg.WriteStallTimeout > 0 {
		_ = c.nc.SetWriteDeadline(time.Now().Add(c.cfg.WriteStallTimeout))
	}
	// net.Buffers.WriteTo performs short-write resumption internally (it
	// advances the vector across partial writev progress). A deadline error
	// here means a peer that made less than one flush of progress in
	// 2×stream_timeout — §8 rule 5's zero-drain closure, a failure by
	// definition (unlike the benchmark rig, where the deadline was the
	// clean end of the run).
	_, err := iovs.WriteTo(c.nc)

	for i := range reqs {
		if reqs[i].release != nil {
			reqs[i].release()
		}
	}
	return err == nil
}

// drainReleases fires the release of everything still queued at teardown so
// arena refcounts never leak on the error path.
func (c *Conn) drainReleases() {
	for {
		select {
		case r := <-c.wq:
			if r.release != nil {
				r.release()
			}
		default:
			return
		}
	}
}
