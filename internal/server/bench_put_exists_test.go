package server_test

import (
	"context"
	"encoding/binary"
	"fmt"
	"sync/atomic"
	"testing"
	"time"

	"github.com/kvstash/kvblockd/internal/config"
	"github.com/kvstash/kvblockd/internal/server"
	"github.com/kvstash/kvblockd/internal/store/ramstub"
	"github.com/kvstash/kvblockd/pkg/client"
)

// benchServer boots a server + client pair for the verb benchmarks and also
// returns the listen address for raw-conn benches.
func benchServer(b *testing.B, streams int) (string, *client.Client, func()) {
	b.Helper()
	cfg := config.Default()
	cfg.ListenAddr = "127.0.0.1:0"
	cfg.SockSndBuf = envBytes("KVB_BENCH_SOCKBUF", 0)
	cfg.SockRcvBuf = cfg.SockSndBuf
	cfg.WriteChunkBytes = envBytes("KVB_BENCH_CHUNK", 1<<20)
	srv := server.New(cfg, ramstub.New(), server.NewNamespaces("t", 7, testToken))
	ctx, cancel := context.WithCancel(context.Background())
	addr, err := srv.Start(ctx)
	if err != nil {
		cancel()
		b.Fatal(err)
	}
	c, err := client.Dial(ctx, addr, client.Options{Streams: streams, Namespace: "t", Token: testToken})
	if err != nil {
		cancel()
		b.Fatal(err)
	}
	return addr, c, func() {
		c.Close()
		cancel()
		dctx, dc := context.WithTimeout(context.Background(), 5*time.Second)
		defer dc()
		srv.Drain(dctx)
	}
}

// BenchmarkPut_1MB is the PUT-path throughput gate: stream one 1 MiB block
// (BEGIN → CHUNK → COMMIT) per op. Write-once means every op needs a fresh
// key; a DELETE rides in the same op to keep the store bounded — it is a
// payload-free metadata round trip, so the reported GB/s slightly UNDERSTATES
// pure PUT throughput (recorded property, not noise).
func BenchmarkPut_1MB(b *testing.B) {
	for _, streams := range []int{1, 4} {
		b.Run(fmt.Sprintf("streams=%d", streams), func(b *testing.B) {
			_, c, cleanup := benchServer(b, streams)
			defer cleanup()
			ctx := context.Background()
			const sz = 1 << 20
			blob := make([]byte, sz)
			for j := range blob {
				blob[j] = byte(j)
			}
			var ctr atomic.Uint64
			b.SetBytes(sz)
			b.ReportAllocs()
			b.ResetTimer()
			b.RunParallel(func(pb *testing.PB) {
				var k [32]byte
				for pb.Next() {
					id := ctr.Add(1) // unique key per op (write-once store)
					binary.LittleEndian.PutUint64(k[:], id)
					if err := c.Put(ctx, k, blob); err != nil {
						b.Fatal(err)
					}
					if _, err := c.Delete(ctx, [][32]byte{k}, false); err != nil {
						b.Fatal(err)
					}
				}
			})
		})
	}
}

// BenchmarkExists_32Keys is the metadata round-trip benchmark: the scheduler's
// load-vs-recompute probe (§3.2) is latency-bound, not bandwidth-bound —
// ns/op here IS the probe RTT.
func BenchmarkExists_32Keys(b *testing.B) {
	for _, streams := range []int{1, 4} {
		b.Run(fmt.Sprintf("streams=%d", streams), func(b *testing.B) {
			_, c, cleanup := benchServer(b, streams)
			defer cleanup()
			ctx := context.Background()
			keys := make([][32]byte, 32)
			blob := []byte("x")
			for i := range keys {
				binary.LittleEndian.PutUint32(keys[i][:], uint32(i)+1) //nolint:gosec // G115: bench index
				if err := c.Put(ctx, keys[i], blob); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportAllocs()
			b.ResetTimer()
			b.RunParallel(func(pb *testing.PB) {
				for pb.Next() {
					n, _, err := c.BatchExists(ctx, keys)
					if err != nil {
						b.Fatal(err)
					}
					if n != 32 {
						b.Fatalf("n_consecutive=%d", n)
					}
				}
			})
		})
	}
}
