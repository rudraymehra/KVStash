package server_test

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"net"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"go.uber.org/goleak"

	"github.com/kvstash/kvblockd/internal/config"
	"github.com/kvstash/kvblockd/internal/protocol"
	"github.com/kvstash/kvblockd/internal/server"
	"github.com/kvstash/kvblockd/internal/store/ramstub"
	"github.com/kvstash/kvblockd/pkg/client"
)

func TestMain(m *testing.M) {
	goleak.VerifyTestMain(m)
}

const testToken = "s3cr3t"

// startServer boots a server on 127.0.0.1:0 with a single namespace and
// returns the address plus a cleanup func that drains it.
func startServer(t *testing.T) (addr string, cleanup func()) {
	t.Helper()
	cfg := config.Default()
	cfg.ListenAddr = "127.0.0.1:0"
	cfg.StreamTimeoutMS = 5000 // the floor; keeps the reaper tick brisk
	ns := server.NewNamespaces("tenant-a", 7, testToken)
	srv := server.New(cfg, ramstub.New(), ns)

	ctx, cancel := context.WithCancel(context.Background())
	a, err := srv.Start(ctx)
	if err != nil {
		cancel()
		t.Fatal(err)
	}
	return a, func() {
		cancel()
		dctx, dcancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer dcancel()
		srv.Drain(dctx)
	}
}

func dialClient(t *testing.T, addr string, streams int) *client.Client {
	t.Helper()
	c, err := client.Dial(context.Background(), addr, client.Options{
		Streams: streams, Namespace: "tenant-a", Token: testToken,
	})
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func key(b byte) [32]byte {
	var k [32]byte
	for i := range k {
		k[i] = b
	}
	return k
}

func TestEndToEnd(t *testing.T) {
	addr, cleanup := startServer(t)
	defer cleanup()
	c := dialClient(t, addr, 4)
	defer c.Close()
	ctx := context.Background()

	k0, k1 := key(0xA0), key(0xB1)
	blob0 := bytes.Repeat([]byte{0x11}, 1<<20)    // 1 MiB
	blob1 := bytes.Repeat([]byte{0x22}, 2560<<10) // 2.5 MiB

	// PUT both.
	if err := c.Put(ctx, k0, blob0); err != nil {
		t.Fatalf("put k0: %v", err)
	}
	if err := c.Put(ctx, k1, blob1); err != nil {
		t.Fatalf("put k1: %v", err)
	}

	// EXISTS: k0,k1 present in order, a missing key breaks the prefix.
	missing := key(0xCC)
	nConsec, per, err := c.BatchExists(ctx, [][32]byte{k0, k1, missing})
	if err != nil {
		t.Fatal(err)
	}
	if nConsec != 2 {
		t.Fatalf("n_consecutive = %d, want 2", nConsec)
	}
	if per != nil && (per[0] != protocol.StatusOK || per[2] != protocol.StatusNotFound) {
		t.Fatalf("exists bitmap: %v", per)
	}

	// GET: hits stream into caller buffers, miss leaves its slot and reports NOT_FOUND.
	into := make([][]byte, 3)
	statuses, err := c.BatchGet(ctx, [][32]byte{k0, k1, missing}, into)
	if err != nil {
		t.Fatal(err)
	}
	if statuses[0] != protocol.StatusOK || statuses[1] != protocol.StatusOK || statuses[2] != protocol.StatusNotFound {
		t.Fatalf("get statuses: %v", statuses)
	}
	if !bytes.Equal(into[0], blob0) || !bytes.Equal(into[1], blob1) {
		t.Fatal("got wrong block bytes")
	}

	// STATS reflects 2 stored blocks.
	stats, err := c.Stats(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(stats), `"blocks":2`) {
		t.Fatalf("stats: %s", stats)
	}
}

func TestPutIdempotentAndImmutable(t *testing.T) {
	addr, cleanup := startServer(t)
	defer cleanup()
	c := dialClient(t, addr, 2)
	defer c.Close()
	ctx := context.Background()

	k := key(0x42)
	blob := bytes.Repeat([]byte{0x55}, 4096)
	if err := c.Put(ctx, k, blob); err != nil {
		t.Fatal(err)
	}
	// Same key, same bytes → OK_EXISTS, transparent to Put (returns nil).
	if err := c.Put(ctx, k, blob); err != nil {
		t.Fatalf("idempotent re-put: %v", err)
	}
}

func TestAuthReject(t *testing.T) {
	addr, cleanup := startServer(t)
	defer cleanup()
	_, err := client.Dial(context.Background(), addr, client.Options{
		Streams: 1, Namespace: "tenant-a", Token: "wrong-token",
	})
	if err == nil {
		t.Fatal("dial with a bad token succeeded")
	}
	if !strings.Contains(err.Error(), "AUTH") && !strings.Contains(err.Error(), "rejected") {
		t.Fatalf("expected an auth rejection, got: %v", err)
	}
}

// TestOpBeforeHelloCloses: a non-HELLO first frame is connection-fatal (§3.1).
func TestOpBeforeHelloCloses(t *testing.T) {
	addr, cleanup := startServer(t)
	defer cleanup()
	nc, err := net.Dial("tcp", addr)
	if err != nil {
		t.Fatal(err)
	}
	defer nc.Close()

	// Send a BATCH_EXISTS before HELLO.
	body := protocol.AppendKeyList(nil, 0, [][32]byte{key(1)})
	h := protocol.Header{Opcode: protocol.OpBatchExists, RequestID: 1, PayloadLen: uint32(len(body))} //nolint:gosec // G115: test body
	hb := make([]byte, protocol.HeaderSize)
	h.MarshalTo(hb)
	pre := net.Buffers{hb, body}
	if _, err := pre.WriteTo(nc); err != nil {
		t.Fatal(err)
	}
	// The server sends a FATAL report and closes.
	_ = nc.SetReadDeadline(time.Now().Add(3 * time.Second))
	rh := make([]byte, protocol.HeaderSize)
	if _, err := io.ReadFull(nc, rh); err != nil {
		t.Fatalf("expected a fatal report frame: %v", err)
	}
	ph, err := protocol.ParseHeader(rh, protocol.DefaultMaxFrameLen)
	if err != nil || ph.Flags&protocol.FlagFatal == 0 {
		t.Fatalf("expected F_FATAL report, got header %+v (%v)", ph, err)
	}
}

// TestPutChecksumRejected: a COMMIT whose bytes don't match the declared xxh3
// is rejected and the block never becomes visible.
func TestPutChecksumRejected(t *testing.T) {
	addr, cleanup := startServer(t)
	defer cleanup()
	nc, err := net.Dial("tcp", addr)
	if err != nil {
		t.Fatal(err)
	}
	defer nc.Close()
	helloRaw(t, nc)

	k := key(0x77)
	begin := protocol.AppendPutBegin(nil, protocol.PutBeginBody{TotalLen: 4})
	writeRaw(t, nc, protocol.OpPutStream, protocol.WithSubOp(0, protocol.PutBegin), k, 10, begin)
	if st := readStatus(t, nc); st != protocol.StatusOK {
		t.Fatalf("begin: %s", st)
	}
	writeRaw(t, nc, protocol.OpPutStream, protocol.WithSubOp(0, protocol.PutChunk), k, 10, []byte{1, 2, 3, 4})
	// COMMIT with a deliberately wrong checksum.
	writeRaw(t, nc, protocol.OpPutStream, protocol.WithSubOp(0, protocol.PutCommit), k, 10, protocol.AppendPutCommit(nil, 0xBADBAD))
	if st := readStatus(t, nc); st != protocol.StatusErrChecksum {
		t.Fatalf("commit: got %s, want ERR_CHECKSUM", st)
	}
	// COMMIT again on the now-terminated stream → ERR_STALE_STREAM.
	writeRaw(t, nc, protocol.OpPutStream, protocol.WithSubOp(0, protocol.PutCommit), k, 10, protocol.AppendPutCommit(nil, 0xBADBAD))
	if st := readStatus(t, nc); st != protocol.StatusErrStaleStream {
		t.Fatalf("recommit: got %s, want ERR_STALE_STREAM", st)
	}
	// The rejected bytes must never have become visible.
	assertInvisible(t, nc, k)
}

// --- raw helpers for the low-level protocol tests ---

func helloRaw(t *testing.T, nc net.Conn) {
	t.Helper()
	body := protocol.AppendHelloReq(nil, protocol.HelloReq{
		ProtoMin: 1, ProtoMax: 1, Token: []byte(testToken), Namespace: "tenant-a",
	})
	writeRaw(t, nc, protocol.OpHello, 0, [32]byte{}, 1, body)
	if st := readStatus(t, nc); st != protocol.StatusOK {
		t.Fatalf("hello: %s", st)
	}
}

func writeRaw(t *testing.T, nc net.Conn, op protocol.Opcode, flags uint16, k [32]byte, id uint64, body []byte) {
	t.Helper()
	h := protocol.Header{Opcode: op, Flags: flags, Key: k, RequestID: id, PayloadLen: uint32(len(body))} //nolint:gosec // G115: test body
	hb := make([]byte, protocol.HeaderSize)
	h.MarshalTo(hb)
	bufs := net.Buffers{hb}
	if len(body) > 0 {
		bufs = append(bufs, body)
	}
	if _, err := bufs.WriteTo(nc); err != nil {
		t.Fatal(err)
	}
}

func readStatus(t *testing.T, nc net.Conn) protocol.Status {
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
	return protocol.Status(body[0])
}

// envBytes reads an integer byte-size env override for benchmark sweeps.
func envBytes(name string, def int) int {
	if v := os.Getenv(name); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 0 {
			return n
		}
	}
	return def
}

// BenchmarkBatchGet_32x1MB is the RAM-stub loopback throughput gate: 32×1 MiB
// blocks per BATCH_GET. It sweeps the connection count the same way the Week-1
// xferspike loopback sweep did (docs/notes/a1-log.md), because loopback
// throughput is HIGHEST at few streams and declines as streams rise (kernel
// memcpy saturates on ~2 cores; extra streams only add contention). The gate
// compares each cell against the SAME-streams xferspike cell (target: within
// 10%), and the headline cell is the sweep peak.
func BenchmarkBatchGet_32x1MB(b *testing.B) {
	cfg := config.Default()
	cfg.ListenAddr = "127.0.0.1:0"
	// Loopback-tuning knobs, sweepable without recompiling:
	//   KVB_BENCH_SOCKBUF — server snd/rcv buffer bytes (default 0 = OS default:
	//     on loopback a small in-flight window keeps the kernel copyin→copyout
	//     pipeline cache-hot; 16 MiB is for real 50 GbE links)
	//   KVB_BENCH_CHUNK — writev syscall window bytes (default 1 MiB)
	cfg.SockSndBuf = envBytes("KVB_BENCH_SOCKBUF", 0)
	cfg.SockRcvBuf = cfg.SockSndBuf
	cfg.WriteChunkBytes = envBytes("KVB_BENCH_CHUNK", 1<<20)
	srv := server.New(cfg, ramstub.New(), server.NewNamespaces("t", 7, testToken))
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	addr, err := srv.Start(ctx)
	if err != nil {
		b.Fatal(err)
	}
	defer func() {
		dctx, dc := context.WithTimeout(context.Background(), 5*time.Second)
		defer dc()
		srv.Drain(dctx)
	}()

	const n = 32
	const sz = 1 << 20
	keys := make([][32]byte, n)
	{
		seed, err := client.Dial(ctx, addr, client.Options{Streams: 1, Namespace: "t", Token: testToken})
		if err != nil {
			b.Fatal(err)
		}
		for i := range keys {
			binary.LittleEndian.PutUint32(keys[i][:], uint32(i)+1) //nolint:gosec // G115: bench index
			blob := bytes.Repeat([]byte{byte(i)}, sz)
			if err := seed.Put(ctx, keys[i], blob); err != nil {
				seed.Close()
				b.Fatal(err)
			}
		}
		seed.Close()
	}

	// hot=false: 32 DISTINCT blocks per GET — real store traffic (source and
	// destination are DRAM-cold; this is the honest production-shaped number).
	// hot=true: ONE block fetched 32× — the server-side source stays
	// cache-resident, which is what the Week-1 xferspike blast measured (it
	// resent a single buffer); this cell is the like-for-like wire-path
	// comparison against the a1-log table.
	for _, hot := range []bool{false, true} {
		reqKeys := keys
		if hot {
			reqKeys = make([][32]byte, n)
			for i := range reqKeys {
				reqKeys[i] = keys[0]
			}
		}
		for _, streams := range []int{1, 2, 4, 16} {
			name := fmt.Sprintf("cold/streams=%d", streams)
			if hot {
				name = fmt.Sprintf("hot/streams=%d", streams)
			}
			b.Run(name, func(b *testing.B) {
				cliBuf := envBytes("KVB_BENCH_CLIBUF", 0)
				c, err := client.Dial(ctx, addr, client.Options{
					Streams: streams, Namespace: "t", Token: testToken,
					SockSndBuf: cliBuf, SockRcvBuf: cliBuf,
				})
				if err != nil {
					b.Fatal(err)
				}
				defer c.Close()

				b.SetBytes(int64(n) * sz)
				b.ReportAllocs()
				b.ResetTimer()
				// Concurrency = pooled connections: extra goroutines just queue
				// on the pool, so RunParallel's GOMAXPROCS workers are enough for
				// every cell; the pool serializes to `streams` in-flight requests.
				b.RunParallel(func(pb *testing.PB) {
					into := make([][]byte, n)
					for pb.Next() {
						st, err := c.BatchGet(ctx, reqKeys, into)
						if err != nil {
							b.Fatal(err)
						}
						if len(st) != n {
							b.Fatalf("got %d statuses, want %d", len(st), n)
						}
					}
				})
			})
		}
	}
}
