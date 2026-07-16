package transport

import (
	"errors"
	"fmt"
	"io"
	"log/slog"
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
//
// WriteFrames MUST be called from this handler goroutine (or before Close).
// That single-producer discipline is what lets the post-close final drain
// guarantee every release fires exactly once; an async worker that calls
// WriteFrames from another goroutine is a server-layer design that must add
// its own synchronization (tracked for the server week).
type FrameHandler interface {
	HandleFrame(c *Conn, h protocol.Header, body []byte)
}

// ErrConnClosed is returned by WriteFrames after Close/Abort.
var ErrConnClosed = errors.New("transport: connection closed")

// maxFlushFrames bounds one writev flush by FRAME count. Total iovecs (1
// header + N payload slices per frame) can exceed the kernel's IOV_MAX (1024);
// Go's net.Buffers.WriteTo chunks the vector at IOV_MAX and loops, so a large
// flush becomes several writev syscalls rather than an error — we bound frames,
// not iovecs, and let the runtime handle the syscall chunking.
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

func (r writeReq) payloadLen() uint64 { return uint64(r.hdr.PayloadLen) }

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

	discardBuf []byte // per-conn reusable 64 KiB skip buffer, read-goroutine-only
}

// startConn wires the loops around an accepted (already tuned) net.Conn.
func startConn(nc net.Conn, bs BufferSource, h FrameHandler, cfg Config) *Conn {
	// Tripwire (golang/go#21676): net.Buffers.WriteTo takes the single-writev
	// fast path only on a bare *net.TCPConn (or a wrapper that EMBEDS one). A
	// wrapped conn silently degrades a 32-buffer flush into 32 Write syscalls.
	// Nothing wraps the conn today; if a future refactor does, this log is the
	// difference between a 5-minute diagnosis and a mystery throughput halving.
	if _, ok := nc.(*net.TCPConn); !ok {
		slog.Warn("transport: conn is not *net.TCPConn — net.Buffers writev fast path is OFF (golang/go#21676)",
			"type", fmt.Sprintf("%T", nc))
	}
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
	go func() {
		wg.Wait()
		// Both loops have exited. The read goroutine is the sole producer of
		// release-bearing writeReqs (handlers call WriteFrames on it), so once
		// it is gone no more can be enqueued: this final drain fires every
		// release exactly once, closing the writer-death strand window.
		c.finalDrain()
		conserveCheck(&c.credit) // no-op unless built with -tags kvbdebug
		close(c.done)
	}()
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

// RemoteAddr is the peer address (for HELLO auth logging and abuse accounting).
func (c *Conn) RemoteAddr() net.Addr { return c.nc.RemoteAddr() }

// WriteFrames queues one response frame: hdr plus the payload slices, emitted
// together in one writev (with other queued frames when the writer is
// backlogged). hdr.PayloadLen is computed here from bufs; hdr.Credit is
// overwritten by the writer with any pending grant. release (may be nil) fires
// after the kernel accepted the bytes — hold arena refcounts until then. If the
// connection is closed, release fires immediately and ErrConnClosed is
// returned. Call from the handler goroutine (see FrameHandler).
func (c *Conn) WriteFrames(hdr protocol.Header, bufs net.Buffers, release func()) error {
	var total uint64
	for _, b := range bufs {
		total += uint64(len(b))
	}
	hdr.PayloadLen = uint32(total) //nolint:gosec // G115: callers respect the negotiated max_frame_len (u32); a server-side overflow would be a server bug the client catches via the deterministic PayloadLen mismatch
	// Fast path: if the connection is already closed, never enqueue — the
	// writer and the post-wg.Wait finalDrain may already have run, so a send
	// here would strand the request. Fire the release and report closed.
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
// write queue is full, we just close. The report body carries no release.
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

// Close tears the connection down: it closes c.closed and pokes an immediate
// I/O deadline so both loops unblock and exit; the socket is closed by the
// writer on its way out. Safe to call multiple times, from any goroutine.
func (c *Conn) Close() error {
	c.closeOnce.Do(func() {
		close(c.closed)
		_ = c.nc.SetDeadline(time.Now()) // unblock in-flight ReadFull/WriteTo
	})
	return nil
}

// Done is closed when both loops have exited, the socket is closed, and every
// release has fired.
func (c *Conn) Done() <-chan struct{} { return c.done }

// respondError enqueues a transport-owned canned error response — used for
// frames that structurally cannot reach a handler (oversize, credit breach),
// where not answering would leave the client waiting forever. No release.
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

// discard reads and drops exactly n payload bytes under a body-read deadline —
// the skip path for frames we refuse. It never Lends a buffer, so a hostile
// length can never size an allocation (the drain() discipline from the rig).
func (c *Conn) discard(n uint32) error {
	if c.cfg.BodyReadTimeout > 0 {
		_ = c.nc.SetReadDeadline(time.Now().Add(c.cfg.BodyReadTimeout))
	}
	// The wrapper hides io.Discard's ReaderFrom so CopyBuffer actually uses the
	// pooled discardBuf (bounding memory) instead of io.Discard's own path.
	_, err := io.CopyBuffer(struct{ io.Writer }{io.Discard}, io.LimitReader(c.nc, int64(n)), c.discardBuf)
	return err
}

// readLoop: ReadFull(64B) → ParseHeader(negotiated cap) → route. Routing order
// is load-bearing: NOP (control, never debits, never answered) is handled
// before the over-cap and credit-enforcement paths so a nonconforming
// keepalive can never draw a response or burn a strike (PROTOCOL.md §3/§8).
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
		overCap := errors.Is(err, protocol.ErrPayloadTooLarge)
		if err != nil && !overCap {
			// Fatal by classification (magic/version/CRC): report, close, no
			// resynchronization attempt, ever (PROTOCOL.md §1).
			c.Abort(protocol.StatusFatalProtocol)
			return
		}

		// NOP is a control frame (§3): tolerate any (even over-cap) payload by
		// skipping it, never debit the window, never answer.
		if h.Opcode == protocol.OpNop {
			if c.discard(h.PayloadLen) != nil {
				return
			}
			continue
		}

		if overCap {
			// Populated-header path: the CRC authenticated PayloadLen, so skip
			// exactly that many bytes, answer ERR_TOO_LARGE echoing the
			// request, and re-grant (§8 amended rule 4). Account, not Consume:
			// an over-cap frame is a size error, not a window violation, so it
			// must not touch the strike counter.
			c.credit.Account(h.PayloadLen)
			derr := c.discard(h.PayloadLen)
			c.credit.Grant(h.PayloadLen) // conserve even on a dying connection
			if derr != nil {
				return
			}
			c.respondError(h, protocol.StatusErrTooLarge)
			continue
		}

		switch c.credit.Consume(h.PayloadLen) {
		case ViolationBusy:
			derr := c.discard(h.PayloadLen)
			c.credit.Grant(h.PayloadLen)
			if derr != nil {
				return
			}
			c.respondError(h, protocol.StatusErrBusy)
			continue
		case ViolationFatal:
			_ = c.discard(h.PayloadLen)
			c.credit.Grant(h.PayloadLen)
			c.Abort(protocol.StatusErrBusy)
			return
		case ViolationNone:
		}

		body := c.bs.Lend(int(h.PayloadLen))
		if c.cfg.BodyReadTimeout > 0 {
			_ = c.nc.SetReadDeadline(time.Now().Add(c.cfg.BodyReadTimeout))
		}
		if _, err := io.ReadFull(c.nc, body); err != nil {
			// Truncated frame: the handler will never run to Grant, so grant
			// here to keep the ledger conserved (the connection is dying).
			c.bs.Return(body)
			c.credit.Grant(h.PayloadLen)
			return
		}
		// Ownership of body transfers to the handler (Return + Grant are its
		// responsibility, directly or via a WriteFrames release).
		c.handler.HandleFrame(c, h, body)
	}
}

// writeLoop: block for the first request, then drain the queue non-blocking
// into one coalesced writev flush. Coalescing NEVER waits for more frames — a
// lone sub-millisecond EXISTS reply flushes immediately; ≥16 MiB batches
// emerge only when the queue is genuinely backlogged. On any flush failure the
// loop closes the whole Conn (not just the socket) so a handler blocked in
// WriteFrames sees c.closed and its release fires; the post-wg.Wait drain mops
// up anything already queued.
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
			for total < uint64(c.cfg.CoalesceBytes) && len(reqs) < maxFlushFrames { //nolint:gosec // G115: CoalesceBytes is a config constant (16 MiB)
				select {
				case r := <-c.wq:
					reqs = append(reqs, r)
					total += r.payloadLen()
				default:
					total = uint64(c.cfg.CoalesceBytes) //nolint:gosec // G115: as above; queue momentarily empty → flush now, never wait
				}
			}
			if !c.flush(hdrArena, reqs, &iovs, c.cfg.WriteStallTimeout) {
				_ = c.Close()
				return
			}

		case <-grantTick.C:
			// §8 rule 4: unsolicited NOP/CREDIT when grants are pending and
			// there is nothing else to ride on.
			if c.credit.PendingGrant() {
				reqs = append(reqs[:0], writeReq{hdr: protocol.Header{Opcode: protocol.OpNop}})
				if !c.flush(hdrArena, reqs, &iovs, c.cfg.WriteStallTimeout) {
					_ = c.Close()
					return
				}
			}

		case <-c.closed:
			// Bounded best-effort final flush of what is already queued (the
			// Abort report frame rides this path). One short deadline for the
			// WHOLE drain — flush must not re-arm it — and we stop at the first
			// failure; the post-wg.Wait drain fires the rest of the releases.
			deadline := time.Now().Add(time.Second)
			for {
				select {
				case r := <-c.wq:
					reqs = append(reqs[:0], r)
					if !c.flushBy(hdrArena, reqs, &iovs, deadline) {
						_ = c.nc.Close()
						return
					}
				default:
					_ = c.nc.Close()
					return
				}
			}
		}
	}
}

// flush emits one writev for the batch with a deadline of now+timeout.
func (c *Conn) flush(hdrArena []byte, reqs []writeReq, iovs *net.Buffers, timeout time.Duration) bool {
	var deadline time.Time
	if timeout > 0 {
		deadline = time.Now().Add(timeout)
	}
	return c.flushBy(hdrArena, reqs, iovs, deadline)
}

// flushBy emits one writev for the batch against an absolute deadline (zero =
// none). iovecs alternate marshalled headers (from the per-flush arena) and
// payload slices; the first header carries any pending credit grant. Releases
// fire after WriteTo returns, success or error (§12). Returns false on write
// failure.
func (c *Conn) flushBy(hdrArena []byte, reqs []writeReq, iovs *net.Buffers, deadline time.Time) bool {
	buf := (*iovs)[:0]
	for i := range reqs {
		if i == 0 {
			reqs[i].hdr.Credit = c.credit.TakeGrant()
		} else {
			reqs[i].hdr.Credit = 0
		}
		slot := hdrArena[i*protocol.HeaderSize : (i+1)*protocol.HeaderSize]
		reqs[i].hdr.MarshalTo(slot)
		buf = append(buf, slot)
		buf = append(buf, reqs[i].bufs...)
	}
	*iovs = buf // keep the grown backing array for reuse next flush

	if !deadline.IsZero() {
		_ = c.nc.SetWriteDeadline(deadline)
	}
	// WriteTo advances its receiver as it consumes iovecs, so hand it a COPY of
	// the slice header — otherwise *iovs would be left pointing at the end of
	// the backing array and the next flush's append would reallocate the whole
	// (up to 1024-entry) vector on the hot path. net.Buffers.WriteTo also does
	// short-write resumption internally; a deadline error means the peer made
	// less than one flush of progress in the window (§8 rule 5, a failure).
	err := c.writeVector(buf)

	for i := range reqs {
		if reqs[i].release != nil {
			reqs[i].release()
		}
	}
	return err == nil
}

// writeVector writes the assembled iovec vector. With WriteChunkBytes set, the
// vector is sliced into consecutive syscall windows: a window accumulates whole
// iovecs until it reaches the chunk target, so a 64 B header always travels
// with the payload it precedes (never a lone tiny write). Frame boundaries and
// wire order are untouched — this only bounds how many bytes one writev copies
// into the kernel at a time, which is what keeps the loopback producer/consumer
// copy pipeline overlapped (A1: 14.1 GB/s at ~1 MiB windows vs 6.9 at 16 MiB).
func (c *Conn) writeVector(buf net.Buffers) error {
	chunk := c.cfg.WriteChunkBytes
	if chunk <= 0 {
		toWrite := buf
		_, err := (&toWrite).WriteTo(c.nc)
		return err
	}
	for start := 0; start < len(buf); {
		end, n := start, 0
		for end < len(buf) && n < chunk {
			n += len(buf[end])
			end++
		}
		w := buf[start:end]
		if _, err := (&w).WriteTo(c.nc); err != nil {
			return err
		}
		start = end
	}
	return nil
}

// finalDrain fires the release of everything still queued once both loops have
// exited — the authoritative, race-free release point (no producer remains).
func (c *Conn) finalDrain() {
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
