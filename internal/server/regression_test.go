package server_test

import (
	"bytes"
	"context"
	"encoding/binary"
	"io"
	"net"
	"testing"
	"time"

	"github.com/kvstash/kvblockd/internal/protocol"
	"github.com/kvstash/kvblockd/pkg/client"
)

// rawConn dials a plain TCP connection for the low-level protocol tests.
func rawConn(t *testing.T, addr string) net.Conn {
	t.Helper()
	nc, err := net.Dial("tcp", addr)
	if err != nil {
		t.Fatal(err)
	}
	return nc
}

// readBody reads one full response frame and returns its body.
func readBody(t *testing.T, nc net.Conn) []byte {
	t.Helper()
	_ = nc.SetReadDeadline(time.Now().Add(3 * time.Second))
	hb := make([]byte, protocol.HeaderSize)
	if _, err := io.ReadFull(nc, hb); err != nil {
		t.Fatalf("read response header: %v", err)
	}
	h, err := protocol.ParseHeader(hb, protocol.DefaultMaxFrameLen)
	if err != nil {
		t.Fatal(err)
	}
	body := make([]byte, h.PayloadLen)
	if _, err := io.ReadFull(nc, body); err != nil {
		t.Fatal(err)
	}
	return body
}

// assertInvisible fails the test if k is visible via BATCH_EXISTS — the
// invisible-until-COMMIT invariant after an aborted/failed PUT.
func assertInvisible(t *testing.T, nc net.Conn, k [32]byte) {
	t.Helper()
	writeRaw(t, nc, protocol.OpBatchExists, 0, [32]byte{}, 777, protocol.AppendKeyList(nil, 0, [][32]byte{k}))
	r, err := protocol.DecodeExistsResp(readBody(t, nc), false)
	if err != nil {
		t.Fatal(err)
	}
	if r.NConsecutive != 0 {
		t.Fatalf("key visible after failed/aborted PUT (n_consecutive=%d)", r.NConsecutive)
	}
}

// TestBatchGetFMoreSplit: a response larger than the negotiated max_frame_len
// is split into F_MORE frames server-side and reassembled by the client
// (the over-cap single-frame bug the review found).
func TestBatchGetFMoreSplit(t *testing.T) {
	addr, cleanup := startServer(t)
	defer cleanup()
	// Propose the §4 floor so a 20 MiB payload MUST split (floor is 16 MiB).
	c, err := client.Dial(context.Background(), addr, client.Options{
		Streams: 1, Namespace: "tenant-a", Token: testToken,
		MaxFrameLen: protocol.FloorMaxFrameLen,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	if got := c.Limits().MaxFrameLen; got != protocol.FloorMaxFrameLen {
		t.Fatalf("negotiated max_frame_len = %d, want the floor %d", got, protocol.FloorMaxFrameLen)
	}
	ctx := context.Background()

	const n = 20 // 20 × 1 MiB > 16 MiB → at least two frames
	keys := make([][32]byte, n)
	blobs := make([][]byte, n)
	for i := range keys {
		binary.LittleEndian.PutUint32(keys[i][:], uint32(i)+100) //nolint:gosec // G115: test index
		blobs[i] = bytes.Repeat([]byte{byte(i + 1)}, 1<<20)
		if err := c.Put(ctx, keys[i], blobs[i]); err != nil {
			t.Fatal(err)
		}
	}
	into := make([][]byte, n)
	statuses, err := c.BatchGet(ctx, keys, into)
	if err != nil {
		t.Fatal(err)
	}
	for i := range keys {
		if statuses[i] != protocol.StatusOK {
			t.Fatalf("key %d: %s", i, statuses[i])
		}
		if !bytes.Equal(into[i], blobs[i]) {
			t.Fatalf("key %d: wrong bytes back", i)
		}
	}
}

// TestPutAbort: ABORT on a live stream answers OK and discards staging; the
// key never becomes visible; COMMIT after ABORT and ABORT of an unknown id
// both answer ERR_STALE_STREAM (§5 exactly-one-terminal-response).
func TestPutAbort(t *testing.T) {
	addr, cleanup := startServer(t)
	defer cleanup()
	nc := rawConn(t, addr)
	defer nc.Close()
	helloRaw(t, nc)

	k := key(0x81)
	begin := protocol.AppendPutBegin(nil, protocol.PutBeginBody{TotalLen: 8})
	writeRaw(t, nc, protocol.OpPutStream, protocol.WithSubOp(0, protocol.PutBegin), k, 21, begin)
	if st := readStatus(t, nc); st != protocol.StatusOK {
		t.Fatalf("begin: %s", st)
	}
	writeRaw(t, nc, protocol.OpPutStream, protocol.WithSubOp(0, protocol.PutChunk), k, 21, []byte{1, 2, 3, 4})
	writeRaw(t, nc, protocol.OpPutStream, protocol.WithSubOp(0, protocol.PutAbort), k, 21, nil)
	if st := readStatus(t, nc); st != protocol.StatusOK {
		t.Fatalf("abort: %s", st)
	}
	// COMMIT after the terminal ABORT → stale.
	writeRaw(t, nc, protocol.OpPutStream, protocol.WithSubOp(0, protocol.PutCommit), k, 21, protocol.AppendPutCommit(nil, 0))
	if st := readStatus(t, nc); st != protocol.StatusErrStaleStream {
		t.Fatalf("commit-after-abort: got %s, want ERR_STALE_STREAM", st)
	}
	// ABORT of an id that never existed → stale.
	writeRaw(t, nc, protocol.OpPutStream, protocol.WithSubOp(0, protocol.PutAbort), k, 99, nil)
	if st := readStatus(t, nc); st != protocol.StatusErrStaleStream {
		t.Fatalf("abort-unknown: got %s, want ERR_STALE_STREAM", st)
	}
	assertInvisible(t, nc, k)
}

// TestPutOverflowTombstones: a CHUNK past the declared total_len tombstones the
// stream; the COMMIT fails and nothing becomes visible.
func TestPutOverflowTombstones(t *testing.T) {
	addr, cleanup := startServer(t)
	defer cleanup()
	nc := rawConn(t, addr)
	defer nc.Close()
	helloRaw(t, nc)

	k := key(0x82)
	begin := protocol.AppendPutBegin(nil, protocol.PutBeginBody{TotalLen: 4})
	writeRaw(t, nc, protocol.OpPutStream, protocol.WithSubOp(0, protocol.PutBegin), k, 31, begin)
	if st := readStatus(t, nc); st != protocol.StatusOK {
		t.Fatalf("begin: %s", st)
	}
	// 8 bytes against a declared 4 → overflow → tombstone (no response).
	writeRaw(t, nc, protocol.OpPutStream, protocol.WithSubOp(0, protocol.PutChunk), k, 31, []byte{1, 2, 3, 4, 5, 6, 7, 8})
	writeRaw(t, nc, protocol.OpPutStream, protocol.WithSubOp(0, protocol.PutCommit), k, 31, protocol.AppendPutCommit(nil, 0))
	if st := readStatus(t, nc); st != protocol.StatusErrStaleStream {
		t.Fatalf("commit-after-overflow: got %s, want ERR_STALE_STREAM", st)
	}
	assertInvisible(t, nc, k)
}

// TestPutLiveStreamCap: the 257th concurrent live stream on one connection is
// refused with ERR_BUSY (the BEGIN-amplification DoS fix), and the cap frees
// up once streams terminate.
func TestPutLiveStreamCap(t *testing.T) {
	addr, cleanup := startServer(t)
	defer cleanup()
	nc := rawConn(t, addr)
	defer nc.Close()
	helloRaw(t, nc)

	begin := protocol.AppendPutBegin(nil, protocol.PutBeginBody{TotalLen: 16})
	for i := 0; i < 256; i++ {
		var k [32]byte
		binary.LittleEndian.PutUint64(k[:], uint64(i)+0xC0FFEE)                                                   //nolint:gosec // G115: test index
		writeRaw(t, nc, protocol.OpPutStream, protocol.WithSubOp(0, protocol.PutBegin), k, uint64(i)+1000, begin) //nolint:gosec // G115: test id
		if st := readStatus(t, nc); st != protocol.StatusOK {
			t.Fatalf("begin %d: %s", i, st)
		}
	}
	over := key(0xEE)
	writeRaw(t, nc, protocol.OpPutStream, protocol.WithSubOp(0, protocol.PutBegin), over, 5000, begin)
	if st := readStatus(t, nc); st != protocol.StatusErrBusy {
		t.Fatalf("257th live stream: got %s, want ERR_BUSY", st)
	}
	// Terminate one stream; the slot must free.
	var k0 [32]byte
	binary.LittleEndian.PutUint64(k0[:], 0xC0FFEE)
	writeRaw(t, nc, protocol.OpPutStream, protocol.WithSubOp(0, protocol.PutAbort), k0, 1000, nil)
	if st := readStatus(t, nc); st != protocol.StatusOK {
		t.Fatalf("abort: %s", st)
	}
	writeRaw(t, nc, protocol.OpPutStream, protocol.WithSubOp(0, protocol.PutBegin), over, 5001, begin)
	if st := readStatus(t, nc); st != protocol.StatusOK {
		t.Fatalf("begin after slot freed: %s", st)
	}
}

// TestMalformedBatchKeepsConnHealthy: a syntactically bad key-list body gets a
// preamble-only ERR_MALFORMED and the SAME connection then serves a valid
// request — the §9 skippable-error contract end to end.
func TestMalformedBatchKeepsConnHealthy(t *testing.T) {
	addr, cleanup := startServer(t)
	defer cleanup()
	nc := rawConn(t, addr)
	defer nc.Close()
	helloRaw(t, nc)

	// n_keys=2 declared but only one key's bytes present → malformed.
	bad := make([]byte, 8+32)
	binary.LittleEndian.PutUint32(bad, 2)
	writeRaw(t, nc, protocol.OpBatchExists, 0, [32]byte{}, 41, bad)
	if st := readStatus(t, nc); st != protocol.StatusErrMalformed {
		t.Fatalf("malformed batch: got %s, want ERR_MALFORMED", st)
	}
	// Connection must still be in sync and serving.
	good := protocol.AppendKeyList(nil, 0, [][32]byte{key(0x01)})
	writeRaw(t, nc, protocol.OpBatchExists, 0, [32]byte{}, 42, good)
	if st := readStatus(t, nc); st != protocol.StatusOK {
		t.Fatalf("valid request after malformed: got %s, want OK", st)
	}
}
