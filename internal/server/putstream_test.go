package server

import (
	"context"
	"io"
	"net"
	"testing"
	"time"

	"github.com/zeebo/xxh3"

	"github.com/kvstash/kvblockd/internal/config"
	"github.com/kvstash/kvblockd/internal/protocol"
	"github.com/kvstash/kvblockd/internal/store/ramstub"
	"github.com/kvstash/kvblockd/internal/transport"
)

// Regression tests for the PUT_STREAM per-connection resource caps. These are
// internal (package server) on purpose: the wire behavior of a rejected BEGIN
// is IDENTICAL whether or not it mints a map entry (§5 treats an absent
// request_id exactly as tombstoned), so only direct inspection of the stream
// table can prove the map stays bounded.

// startRawSession boots a bare transport listener serving ONE pre-authed
// session (no HELLO, no reaper goroutine), so tests can flood PUT sub-ops on
// the wire and then inspect the session's stream table directly.
func startRawSession(t *testing.T, store Store) (net.Conn, *session) {
	t.Helper()
	sess := newSession(New(config.Default(), store, nil))
	sess.authed = true // fields set before the read loop starts — no race
	sess.ns = 7
	ln, err := transport.Listen(context.Background(), transport.DefaultConfig("127.0.0.1:0", 5000))
	if err != nil {
		t.Fatal(err)
	}
	type dialRes struct {
		nc  net.Conn
		err error
	}
	dialed := make(chan dialRes, 1)
	go func() {
		nc, derr := net.Dial("tcp", ln.Addr().String())
		dialed <- dialRes{nc, derr}
	}()
	conn, err := ln.Accept(sess, sess)
	if err != nil {
		_ = ln.Close()
		t.Fatal(err)
	}
	d := <-dialed
	if d.err != nil {
		t.Fatal(d.err)
	}
	t.Cleanup(func() {
		_ = d.nc.Close()
		<-conn.Done()
		_ = ln.Close()
	})
	return d.nc, sess
}

// appendFrame appends one raw PUT_STREAM frame (header + body) to dst.
func appendFrame(t *testing.T, dst []byte, sub uint8, key [32]byte, id uint64, body []byte) []byte {
	t.Helper()
	h := protocol.Header{
		Opcode: protocol.OpPutStream, Flags: protocol.WithSubOp(0, sub),
		Key: key, RequestID: id, PayloadLen: uint32(len(body)), //nolint:gosec // G115: test body
	}
	var hb [protocol.HeaderSize]byte
	h.MarshalTo(hb[:])
	dst = append(dst, hb[:]...)
	return append(dst, body...)
}

// readRespStatus reads the next RESPONSE frame's leading status byte,
// skipping control frames (credit-grant NOPs arrive interleaved under flood).
func readRespStatus(t *testing.T, nc net.Conn) protocol.Status {
	t.Helper()
	_ = nc.SetReadDeadline(time.Now().Add(10 * time.Second))
	hb := make([]byte, protocol.HeaderSize)
	for {
		if _, err := io.ReadFull(nc, hb); err != nil {
			t.Fatalf("read response header: %v", err)
		}
		h, err := protocol.ParseHeader(hb, protocol.DefaultMaxFrameLen)
		if err != nil {
			t.Fatal(err)
		}
		body := make([]byte, h.PayloadLen)
		if _, err := io.ReadFull(nc, body); err != nil {
			t.Fatalf("read response body: %v", err)
		}
		if h.Opcode == protocol.OpNop {
			continue
		}
		return protocol.Status(body[0])
	}
}

// TestBeginRejectFloodDoesNotGrowStreamMap pins the tombstone-DoS fix: a
// client hammering BEGIN on one sealed key with fresh request_ids used to
// mint an uncapped map entry per reject (tombstones bypassed maxLiveStreams
// and lingered a full reap grace period — tens of millions of entries per
// connection at wire rate). Rejected BEGINs must insert NOTHING; the wire
// answers stay exactly as §5 specifies either way.
func TestBeginRejectFloodDoesNotGrowStreamMap(t *testing.T) {
	st := ramstub.New()
	sealed := [32]byte{0xAA}
	if got := st.Put(7, sealed, []byte{1}, xxh3.Hash([]byte{1})); got != protocol.StatusOK {
		t.Fatalf("seed put: %s", got)
	}
	nc, sess := startRawSession(t, st)

	const n = maxStreamEntries + 64 // past every cap if rejects still inserted
	begin := protocol.AppendPutBegin(nil, protocol.PutBeginBody{TotalLen: 1})
	var out []byte
	for id := uint64(1); id <= n; id++ {
		out = appendFrame(t, out, protocol.PutBegin, sealed, id, begin)
	}
	if _, err := nc.Write(out); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < n; i++ {
		if got := readRespStatus(t, nc); got != protocol.StatusOKExists {
			t.Fatalf("BEGIN %d: got %s, want OK_EXISTS", i, got)
		}
	}

	sess.streamMu.Lock()
	entries, live := len(sess.streams), sess.liveStreams
	sess.streamMu.Unlock()
	if entries != 0 || live != 0 {
		t.Fatalf("HIGH regression: %d map entries (%d live) after %d rejected BEGINs, want 0", entries, live, n)
	}
}

// TestStreamMapBoundRefusesBusy pins the TOTAL map cap: tombstones minted the
// remaining legitimate way (live stream → overflow CHUNK) linger for the reap
// grace period, so without a bound a BEGIN+bad-CHUNK cycle still grows the
// map at wire rate. Once live + tombstoned entries reach maxStreamEntries,
// the next BEGIN must refuse with ERR_BUSY instead of inserting.
func TestStreamMapBoundRefusesBusy(t *testing.T) {
	nc, sess := startRawSession(t, ramstub.New())

	k := [32]byte{0xBB}
	begin := protocol.AppendPutBegin(nil, protocol.PutBeginBody{TotalLen: 1})
	var out []byte
	for id := uint64(1); id <= maxStreamEntries; id++ {
		out = appendFrame(t, out, protocol.PutBegin, k, id, begin)
		// 2 bytes against a declared total of 1: overflow → tombstone (silent).
		out = appendFrame(t, out, protocol.PutChunk, k, id, []byte{1, 2})
	}
	if _, err := nc.Write(out); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < maxStreamEntries; i++ {
		if got := readRespStatus(t, nc); got != protocol.StatusOK {
			t.Fatalf("BEGIN %d: got %s, want OK", i, got)
		}
	}

	// The map is now all tombstones at the cap: BEGIN must backpressure.
	out = appendFrame(t, nil, protocol.PutBegin, k, maxStreamEntries+1, begin)
	if _, err := nc.Write(out); err != nil {
		t.Fatal(err)
	}
	if got := readRespStatus(t, nc); got != protocol.StatusErrBusy {
		t.Fatalf("HIGH regression: BEGIN over the map bound got %s, want ERR_BUSY", got)
	}
	sess.streamMu.Lock()
	entries, live := len(sess.streams), sess.liveStreams
	sess.streamMu.Unlock()
	if entries != maxStreamEntries || live != 0 {
		t.Fatalf("map at %d entries (%d live), want exactly %d (0 live)", entries, live, maxStreamEntries)
	}
}

// TestReapTimeoutFloorZeroConfig pins the degenerate-config floor: a session
// built from a zero config (one that skipped config.Validate's 5s floor)
// must sweep with the protocol default, not timeout 0 — which tombstoned
// EVERY live stream on the reaper's first tick.
func TestReapTimeoutFloorZeroConfig(t *testing.T) {
	s := &session{srv: &Server{}, streams: map[uint64]*putStream{}}
	timeout := s.reapTimeout()
	if want := protocol.DefaultStreamTimeoutMS * time.Millisecond; timeout != want {
		t.Fatalf("reapTimeout with a zero config = %v, want %v", timeout, want)
	}
	now := time.Unix(1_000_000, 0)
	s.streams[1] = &putStream{buf: make([]byte, 8), lastActive: now}
	s.liveStreams = 1
	s.stagedBytes = 8
	s.sweepStreams(now.Add(time.Second), timeout)
	if s.streams[1].tombstoned {
		t.Fatal("MED regression: fresh live stream tombstoned under a zero-config sweep")
	}
}

// TestSweepMaintainsLiveCounter pins the counter the BEGIN cap reads: a sweep
// that tombstones idle streams must decrement liveStreams, or the cap would
// count ghosts and starve future BEGINs with ERR_BUSY forever.
func TestSweepMaintainsLiveCounter(t *testing.T) {
	s := &session{streams: map[uint64]*putStream{}}
	base := time.Unix(1_000_000, 0)
	s.streams[1] = &putStream{buf: make([]byte, 8), lastActive: base}
	s.streams[2] = &putStream{buf: make([]byte, 8), lastActive: base}
	s.liveStreams = 2
	s.stagedBytes = 16
	s.sweepStreams(base.Add(10*time.Second), 5*time.Second)
	if s.liveStreams != 0 || s.stagedBytes != 0 {
		t.Fatalf("after sweeping both idle streams: liveStreams=%d stagedBytes=%d, want 0/0",
			s.liveStreams, s.stagedBytes)
	}
	if !s.streams[1].tombstoned || !s.streams[2].tombstoned {
		t.Fatal("idle streams not tombstoned")
	}
}
