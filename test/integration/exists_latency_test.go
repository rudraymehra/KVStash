//go:build integration

// Package integration holds the measured-gate tests: they boot the real
// daemon stack (DRAM tier over an mmap arena, real TCP loopback) and assert
// latency/throughput properties, so they are build-tagged out of the unit
// suite. Run: go test -tags integration -count=1 ./test/integration/
package integration

import (
	"bytes"
	"context"
	"os"
	"sort"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/kvstash/kvblockd/internal/config"
	"github.com/kvstash/kvblockd/internal/server"
	"github.com/kvstash/kvblockd/internal/store/dram"
	"github.com/kvstash/kvblockd/pkg/client"
)

const (
	gateToken = "gate-token"
	// The lane pattern: readers stream big blocks while schedulers probe
	// prefixes — EXISTS must stay on the index fast path, never queue behind
	// payload writes.
	nGetWorkers    = 8
	nExistsWorkers = 4
	nBlocks        = 64
	blockLen       = 1 << 20
	getBatch       = 4
	existsBatch    = 512 // the pre-registered gate shape (512-key probes)
	p99Budget      = time.Millisecond
)

// gateSeconds is the measurement window; override with KVB_GATE_SECONDS for
// quick local iterations (the recorded gate runs use the 60s default).
func gateSeconds() time.Duration {
	if v := os.Getenv("KVB_GATE_SECONDS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return time.Duration(n) * time.Second
		}
	}
	return 60 * time.Second
}

func gateKey(i int) [32]byte {
	var k [32]byte
	k[0], k[1], k[2] = byte(i), byte(i>>8), 0x6A
	return k
}

func TestExistsLatencyUnderGetLoad(t *testing.T) {
	arena, err := dram.NewArena(int64(nBlocks+16)<<20, false)
	if err != nil {
		t.Fatal(err)
	}
	st := dram.New(arena, dram.Params{LeaseDefaultMS: 5000, LeaseMaxMS: 60000})
	cfg := config.Default()
	cfg.ListenAddr = "127.0.0.1:0"
	srv := server.New(cfg, st, server.NewNamespaces("gate", 7, gateToken))
	ctx, cancel := context.WithCancel(context.Background())
	addr, err := srv.Start(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		cancel()
		dctx, dcancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer dcancel()
		srv.Drain(dctx)
		if err := st.Close(); err != nil {
			t.Error(err)
		}
	}()

	seed, err := client.Dial(context.Background(), addr, client.Options{
		Streams: 4, Namespace: "gate", Token: gateToken,
	})
	if err != nil {
		t.Fatal(err)
	}
	blob := bytes.Repeat([]byte{0x5C}, blockLen)
	for i := 0; i < nBlocks; i++ {
		if err := seed.Put(context.Background(), gateKey(i), blob); err != nil {
			t.Fatal(err)
		}
	}
	seed.Close()

	deadline := time.Now().Add(gateSeconds())
	var stopFlag atomic.Bool
	var wg sync.WaitGroup
	var getOps, workerErrs atomic.Int64

	// GET lanes: sustained big-payload reads.
	for w := 0; w < nGetWorkers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			// SkipVerify: the lane pattern needs full payloads ON THE WIRE;
			// client-side re-hashing at ~8 GB/s would starve the timing
			// goroutines of CPU and measure this process, not the server.
			c, err := client.Dial(context.Background(), addr, client.Options{
				Streams: 1, Namespace: "gate", Token: gateToken, SkipVerify: true,
			})
			if err != nil {
				workerErrs.Add(1)
				return
			}
			defer c.Close()
			keys := make([][32]byte, getBatch)
			into := make([][]byte, getBatch)
			for i := range into {
				into[i] = make([]byte, blockLen)
			}
			n := w // stagger start offsets across workers
			for !stopFlag.Load() {
				for i := range keys {
					keys[i] = gateKey((n + i) % nBlocks)
				}
				n += getBatch
				if _, err := c.BatchGet(context.Background(), keys, into); err != nil {
					workerErrs.Add(1)
					return
				}
				getOps.Add(1)
			}
		}(w)
	}

	// EXISTS lanes: the latency under test.
	samples := make([][]time.Duration, nExistsWorkers)
	for w := 0; w < nExistsWorkers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			c, err := client.Dial(context.Background(), addr, client.Options{
				Streams: 1, Namespace: "gate", Token: gateToken,
			})
			if err != nil {
				workerErrs.Add(1)
				return
			}
			defer c.Close()
			keys := make([][32]byte, existsBatch)
			for i := range keys {
				keys[i] = gateKey(i % nBlocks)
			}
			for !stopFlag.Load() {
				t0 := time.Now()
				if _, _, err := c.BatchExists(context.Background(), keys); err != nil {
					workerErrs.Add(1)
					return
				}
				samples[w] = append(samples[w], time.Since(t0))
			}
		}(w)
	}

	time.Sleep(time.Until(deadline))
	stopFlag.Store(true)
	wg.Wait()

	if n := workerErrs.Load(); n > 0 {
		t.Fatalf("%d worker errors during the run", n)
	}
	var all []time.Duration
	for _, s := range samples {
		all = append(all, s...)
	}
	if len(all) < 1000 {
		t.Fatalf("only %d EXISTS samples — the run is too short to call a p99", len(all))
	}
	sort.Slice(all, func(i, j int) bool { return all[i] < all[j] })
	p50 := all[len(all)/2]
	p99 := all[len(all)*99/100]
	p999 := all[len(all)*999/1000]
	t.Logf("EXISTS under %d GET lanes (%v window): %d samples, p50=%v p99=%v p99.9=%v; %d GET batches (%d MiB served)",
		nGetWorkers, gateSeconds(), len(all), p50, p99, p999,
		getOps.Load(), getOps.Load()*getBatch*blockLen>>20)
	if p99 >= p99Budget {
		t.Fatalf("EXISTS p99 = %v, gate demands < %v", p99, p99Budget)
	}
}
