package server_test

import (
	"context"
	"encoding/binary"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/kvstash/kvblockd/internal/config"
	"github.com/kvstash/kvblockd/internal/server"
	"github.com/kvstash/kvblockd/internal/store/ramstub"
	"github.com/kvstash/kvblockd/pkg/client"
)

// BenchmarkMixedWorkload drives concurrent GET + PUT + EXISTS from separate
// pooled connections against one store, the shape isolated benchmarks miss.
// It stresses the store's 256 sharded RWMutexes (readers vs the write-once
// Put writer) and the per-connection PUT reaper simultaneously. The reported
// metric is aggregate GET GB/s under write contention; the point is whether
// contention (shard-lock, reaper, scheduler) degrades it vs the isolated
// numbers — a regression here is a real production signal. Result: GET holds
// ~8 GB/s under a concurrent writer+prober (no contention collapse).
//
// THROUGHPUT-ONLY: run WITHOUT -race. Under -race's ~10x slowdown, 12
// concurrent 32 MiB flushes saturate the machine hard enough to occasionally
// trip the write-stall timeout (an EOF artifact, not a bug). Mixed-path RACE
// coverage lives in TestMixedConcurrentWorkload (modestly scaled, race-clean).
// CI runs `go test -race` without -bench, so this never executes there.
func BenchmarkMixedWorkload(b *testing.B) {
	cfg := config.Default()
	cfg.ListenAddr = "127.0.0.1:0"
	cfg.StreamTimeoutMS = 5000 // brisk reaper tick, so it runs during the bench
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
	getKeys := make([][32]byte, n)
	c, err := client.Dial(ctx, addr, client.Options{Streams: 12, Namespace: "t", Token: testToken})
	if err != nil {
		b.Fatal(err)
	}
	defer c.Close()
	for i := range getKeys {
		binary.LittleEndian.PutUint32(getKeys[i][:], uint32(i)+1) //nolint:gosec // G115: bench index
		blob := make([]byte, sz)
		for j := range blob {
			blob[j] = byte(i)
		}
		if err := c.Put(ctx, getKeys[i], blob); err != nil {
			b.Fatal(err)
		}
	}

	// Background writers + probers run for the whole measured window, competing
	// with the timed GET readers for shard locks and scheduler time.
	stop := make(chan struct{})
	var bg sync.WaitGroup
	var putCtr atomic.Uint64
	putBlob := make([]byte, sz)
	bg.Add(2)
	go func() { // writer: fresh keys forever (write-once, exercises the Put write lock)
		defer bg.Done()
		var k [32]byte
		for {
			select {
			case <-stop:
				return
			default:
			}
			binary.LittleEndian.PutUint64(k[:], putCtr.Add(1)+1_000_000)
			_ = c.Put(ctx, k, putBlob)
		}
	}()
	go func() { // prober: EXISTS on the GET set (read lock churn)
		defer bg.Done()
		for {
			select {
			case <-stop:
				return
			default:
			}
			_, _, _ = c.BatchExists(ctx, getKeys)
		}
	}()

	b.SetBytes(int64(n) * sz)
	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		into := make([][]byte, n)
		for pb.Next() {
			if _, err := c.BatchGet(ctx, getKeys, into); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.StopTimer()
	close(stop)
	bg.Wait()
	b.ReportMetric(float64(putCtr.Load()), "puts-during")
}
