package transport

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net"
	"testing"
	"time"

	"go.uber.org/goleak"

	"github.com/kvstash/kvblockd/internal/protocol"
)

func TestMain(m *testing.M) {
	goleak.VerifyTestMain(m)
}

// testConfig returns tight timeouts suitable for loopback tests.
func testConfig() Config {
	cfg := DefaultConfig("127.0.0.1:0", 30000)
	cfg.WriteStallTimeout = 2 * time.Second
	cfg.IdleReadTimeout = 5 * time.Second
	cfg.GrantTick = 20 * time.Millisecond
	return cfg
}

// echoHandler is the loopback harness: it echoes every frame's body back
// under the same opcode/request_id, releasing the lent buffer after the
// kernel accepts the response, and granting the frame's bytes.
type echoHandler struct{ bs BufferSource }

func (e *echoHandler) HandleFrame(c *Conn, h protocol.Header, body []byte) {
	n := h.PayloadLen
	resp := protocol.Header{
		Opcode:      h.Opcode,
		Flags:       protocol.FlagResp,
		NamespaceID: h.NamespaceID,
		RequestID:   h.RequestID,
	}
	_ = c.WriteFrames(resp, net.Buffers{body}, func() {
		e.bs.Return(body)
		c.GrantCredit(n)
	})
}

// startEcho spins up a listener + accept loop with the echo handler, returning
// the address and a teardown func.
func startEcho(t *testing.T, cfg Config) (addr string, teardown func()) {
	t.Helper()
	l, err := Listen(context.Background(), cfg)
	if err != nil {
		t.Fatal(err)
	}
	bs := HeapSource{}
	conns := make(chan *Conn, 16)
	acceptDone := make(chan struct{})
	go func() {
		defer close(acceptDone)
		for {
			c, aerr := l.Accept(bs, &echoHandler{bs: bs})
			if aerr != nil {
				return
			}
			conns <- c
		}
	}()
	return l.Addr().String(), func() {
		_ = l.Close()
		<-acceptDone
		close(conns)
		for c := range conns {
			_ = c.Close()
			<-c.Done()
		}
	}
}

// dial connects a raw test client.
func dial(t *testing.T, addr string) net.Conn {
	t.Helper()
	nc, err := net.DialTimeout("tcp", addr, 2*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	_ = nc.SetDeadline(time.Now().Add(10 * time.Second))
	return nc
}

// sendFrame writes header+payload (possibly as multiple slices) to nc.
func sendFrame(t *testing.T, nc net.Conn, h protocol.Header, payload ...[]byte) {
	t.Helper()
	var total int
	for _, p := range payload {
		total += len(p)
	}
	h.PayloadLen = uint32(total) //nolint:gosec // test sizes
	hdr := make([]byte, protocol.HeaderSize)
	h.MarshalTo(hdr)
	bufs := append(net.Buffers{hdr}, payload...)
	if _, err := bufs.WriteTo(nc); err != nil {
		t.Fatal(err)
	}
}

// readFrame reads one response frame, skipping unsolicited NOP/CREDIT frames.
func readFrame(t *testing.T, nc net.Conn) (protocol.Header, []byte) {
	t.Helper()
	hdr := make([]byte, protocol.HeaderSize)
	for {
		if _, err := io.ReadFull(nc, hdr); err != nil {
			t.Fatalf("read header: %v", err)
		}
		h, err := protocol.ParseHeader(hdr, protocol.DefaultMaxFrameLen)
		if err != nil {
			t.Fatalf("parse response header: %v", err)
		}
		body := make([]byte, h.PayloadLen)
		if _, err := io.ReadFull(nc, body); err != nil {
			t.Fatalf("read body: %v", err)
		}
		if h.Opcode == protocol.OpNop && h.Flags&protocol.FlagFatal == 0 {
			continue // grant-carrying keepalive; not the response we await
		}
		return h, body
	}
}

// TestLoopbackEchoMultiSlice: a frame whose payload is written as several
// slices arrives byte-identical through the full read→handler→writev path.
func TestLoopbackEchoMultiSlice(t *testing.T) {
	addr, teardown := startEcho(t, testConfig())
	defer teardown()
	nc := dial(t, addr)
	defer nc.Close()

	parts := [][]byte{
		bytes.Repeat([]byte{0x11}, 100),
		bytes.Repeat([]byte{0x22}, 64<<10),
		bytes.Repeat([]byte{0x33}, 7),
	}
	want := bytes.Join(parts, nil)

	req := protocol.Header{Opcode: protocol.OpBatchGet, NamespaceID: 3, RequestID: 42}
	sendFrame(t, nc, req, parts...)

	h, body := readFrame(t, nc)
	if h.RequestID != 42 || h.Flags&protocol.FlagResp == 0 {
		t.Fatalf("response header: %+v", h)
	}
	if !bytes.Equal(body, want) {
		t.Fatalf("echo not byte-identical: got %d bytes, want %d", len(body), len(want))
	}
}

// TestLoneFrameFlushesImmediately pins the no-delay coalescing rule: a single
// queued response must not wait for more traffic.
func TestLoneFrameFlushesImmediately(t *testing.T) {
	addr, teardown := startEcho(t, testConfig())
	defer teardown()
	nc := dial(t, addr)
	defer nc.Close()

	start := time.Now()
	sendFrame(t, nc, protocol.Header{Opcode: protocol.OpBatchExists, RequestID: 1}, []byte("k"))
	_, _ = readFrame(t, nc)
	if rtt := time.Since(start); rtt > 500*time.Millisecond {
		t.Fatalf("lone frame took %v — the writer waited to coalesce", rtt)
	}
}

// TestManyFramesPipelined: a pipelined burst echoes completely and in order
// (handlers run on the read goroutine, so ordering is preserved).
func TestManyFramesPipelined(t *testing.T) {
	addr, teardown := startEcho(t, testConfig())
	defer teardown()
	nc := dial(t, addr)
	defer nc.Close()

	const n = 200
	go func() {
		for i := 0; i < n; i++ {
			payload := bytes.Repeat([]byte{byte(i)}, 1024)
			h := protocol.Header{Opcode: protocol.OpBatchGet, RequestID: uint64(i)} //nolint:gosec // test loop index
			hb := make([]byte, protocol.HeaderSize)
			h.PayloadLen = uint32(len(payload)) //nolint:gosec // G115: test constant
			h.MarshalTo(hb)
			bufs := net.Buffers{hb, payload}
			if _, err := bufs.WriteTo(nc); err != nil {
				return
			}
		}
	}()
	for i := 0; i < n; i++ {
		h, body := readFrame(t, nc)
		if h.RequestID != uint64(i) { //nolint:gosec // test loop index
			t.Fatalf("frame %d: got request_id %d — ordering broke", i, h.RequestID)
		}
		if len(body) != 1024 || body[0] != byte(i) {
			t.Fatalf("frame %d: wrong body", i)
		}
	}
}

// TestOversizeSkipStaysInSync is the §4 recoverable path end-to-end: an
// over-cap frame draws ERR_TOO_LARGE echoing its request_id, and the NEXT
// frame on the same connection is served normally — the stream never desyncs.
func TestOversizeSkipStaysInSync(t *testing.T) {
	cfg := testConfig()
	cfg.PreNegCap = 4096 // small cap so an oversize frame is cheap to build
	addr, teardown := startEcho(t, cfg)
	defer teardown()
	nc := dial(t, addr)
	defer nc.Close()

	big := bytes.Repeat([]byte{0xEE}, 8192) // 2× the cap
	sendFrame(t, nc, protocol.Header{Opcode: protocol.OpBatchGet, RequestID: 7}, big)

	h, body := readFrame(t, nc)
	if h.RequestID != 7 {
		t.Fatalf("error response request_id %d, want 7", h.RequestID)
	}
	p, err := protocol.DecodePreamble(body)
	if err != nil || p.Status != protocol.StatusErrTooLarge {
		t.Fatalf("want ERR_TOO_LARGE preamble, got %+v (%v)", p, err)
	}

	sendFrame(t, nc, protocol.Header{Opcode: protocol.OpBatchGet, RequestID: 8}, []byte("after"))
	h, body = readFrame(t, nc)
	if h.RequestID != 8 || string(body) != "after" {
		t.Fatalf("stream desynced after oversize skip: %+v %q", h, body)
	}
}

// TestFatalBadCRCReportsAndCloses: a corrupted header draws the best-effort
// §9 fatal report (opcode 0, F_RESP|F_FATAL, FATAL_PROTOCOL preamble) and the
// connection closes with no resynchronization.
func TestFatalBadCRCReportsAndCloses(t *testing.T) {
	addr, teardown := startEcho(t, testConfig())
	defer teardown()
	nc := dial(t, addr)
	defer nc.Close()

	h := protocol.Header{Opcode: protocol.OpBatchGet, RequestID: 9}
	hdr := make([]byte, protocol.HeaderSize)
	h.MarshalTo(hdr)
	hdr[30] ^= 0xFF // corrupt a key byte: CRC now fails
	if _, err := nc.Write(hdr); err != nil {
		t.Fatal(err)
	}

	rh, body := readFrame(t, nc)
	if rh.Flags&protocol.FlagFatal == 0 || rh.Opcode != protocol.OpNop || rh.RequestID != 0 {
		t.Fatalf("fatal report header: %+v", rh)
	}
	if p, err := protocol.DecodePreamble(body); err != nil || p.Status != protocol.StatusFatalProtocol {
		t.Fatalf("fatal report status: %+v (%v)", p, err)
	}
	// The connection must now be closed by the server.
	_ = nc.SetReadDeadline(time.Now().Add(2 * time.Second))
	if _, err := nc.Read(make([]byte, 1)); err == nil {
		t.Fatal("connection still open after a fatal frame")
	}
}

// TestNopSwallowed: keepalives are consumed (payload and all) and never
// answered; the following frame is served normally.
func TestNopSwallowed(t *testing.T) {
	addr, teardown := startEcho(t, testConfig())
	defer teardown()
	nc := dial(t, addr)
	defer nc.Close()

	sendFrame(t, nc, protocol.Header{Opcode: protocol.OpNop}, []byte("nonconforming payload"))
	sendFrame(t, nc, protocol.Header{Opcode: protocol.OpBatchGet, RequestID: 11}, []byte("real"))

	h, body := readFrame(t, nc)
	if h.RequestID != 11 || string(body) != "real" {
		t.Fatalf("NOP was not swallowed cleanly: %+v %q", h, body)
	}
}

// TestCreditGrantsFlow: served bytes come back as header credit grants (or an
// unsolicited NOP/CREDIT), conserving the ledger end to end.
func TestCreditGrantsFlow(t *testing.T) {
	addr, teardown := startEcho(t, testConfig())
	defer teardown()
	nc := dial(t, addr)
	defer nc.Close()

	payload := bytes.Repeat([]byte{1}, 4096)
	var granted uint64
	deadline := time.Now().Add(5 * time.Second)
	sendFrame(t, nc, protocol.Header{Opcode: protocol.OpBatchGet, RequestID: 1}, payload)

	hdr := make([]byte, protocol.HeaderSize)
	for granted < uint64(len(payload)) && time.Now().Before(deadline) {
		if _, err := io.ReadFull(nc, hdr); err != nil {
			t.Fatalf("granted %d of %d, then: %v", granted, len(payload), err)
		}
		h, err := protocol.ParseHeader(hdr, protocol.DefaultMaxFrameLen)
		if err != nil {
			t.Fatal(err)
		}
		granted += uint64(h.Credit)
		if h.PayloadLen > 0 {
			if _, err := io.CopyN(io.Discard, nc, int64(h.PayloadLen)); err != nil {
				t.Fatal(err)
			}
		}
	}
	if granted < uint64(len(payload)) {
		t.Fatalf("granted %d bytes, want >= %d — the ledger leaks (a client would stall)", granted, len(payload))
	}
}

// TestWriteFramesAfterCloseReleases: WriteFrames on a closed conn fails fast
// and still fires the release (arena refcounts must not leak).
func TestWriteFramesAfterCloseReleases(t *testing.T) {
	client, server := net.Pipe()
	defer client.Close()
	c := startConn(server, HeapSource{}, &echoHandler{bs: HeapSource{}}, testConfig())
	_ = c.Close()
	<-c.Done()

	released := false
	err := c.WriteFrames(protocol.Header{}, nil, func() { released = true })
	if !errors.Is(err, ErrConnClosed) || !released {
		t.Fatalf("err=%v released=%v", err, released)
	}
}

// TestSlowReaderIsBounded pins the LMCache-class failure: a peer that never
// reads its responses stalls the connection at the write deadline — it never
// buffers unboundedly, and teardown releases everything queued.
func TestSlowReaderIsBounded(t *testing.T) {
	cfg := testConfig()
	cfg.WriteStallTimeout = 300 * time.Millisecond
	addr, teardown := startEcho(t, cfg)
	defer teardown()
	nc := dial(t, addr)
	defer nc.Close()

	// Blast frames without ever reading a response. The server's writer hits
	// the stall deadline once loopback+queue capacity is exhausted and closes.
	payload := bytes.Repeat([]byte{7}, 256<<10)
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		h := protocol.Header{Opcode: protocol.OpBatchGet, RequestID: 1}
		h.PayloadLen = uint32(len(payload)) //nolint:gosec // G115: test constant
		hb := make([]byte, protocol.HeaderSize)
		h.MarshalTo(hb)
		bufs := net.Buffers{hb, payload}
		_ = nc.SetWriteDeadline(time.Now().Add(time.Second))
		if _, err := bufs.WriteTo(nc); err != nil {
			return // server closed on us: exactly the bounded behavior we want
		}
	}
	t.Fatal("server buffered a never-reading peer for 10s without closing")
}
