// getbench is the out-of-process BATCH_GET load generator: it PUTs n blocks
// once, then drives GETs for -secs across -streams pooled connections and
// prints GB/s. Running the daemon and the load in separate processes removes
// the shared-scheduler artifact of the in-package benchmark and matches the
// production shape (docs/DESIGN.md "Week 2 wire-path results").
package main

import (
	"context"
	"encoding/binary"
	"flag"
	"fmt"
	"os"
	"sync"
	"sync/atomic"
	"time"

	"github.com/kvstash/kvblockd/pkg/client"
)

func main() {
	addr := flag.String("addr", "127.0.0.1:9440", "kvblockd address")
	ns := flag.String("ns", "bench", "namespace")
	token := flag.String("token", "bench-token", "bearer token")
	streams := flag.Int("streams", 4, "pooled connections (= concurrent GETs)")
	secs := flag.Int("secs", 5, "measure seconds")
	nBlocks := flag.Int("blocks", 32, "blocks per BATCH_GET")
	blockKiB := flag.Int("block-kib", 1024, "block size KiB")
	pool := flag.Int("pool", 0, "distinct block pool to draw batches from (0 = blocks; larger pool = colder DRAM, isolates working-set from batch size)")
	fillMBps := flag.Int("fill-mbps", 0, "throttle the pool fill to N MiB/s (0 = unthrottled). On a tiered daemon a loopback fill outruns NVMe demotion bandwidth and the arena evicts the pool instead of demoting it — cap the fill below the device write ceiling")
	sockbuf := flag.Int("sockbuf", 0, "client socket buffer bytes (0=OS default)")
	noverify := flag.Bool("noverify", false, "skip client-side xxh3 verification (isolates verification cost)")
	flag.Parse()

	ctx := context.Background()
	c, err := client.Dial(ctx, *addr, client.Options{
		Streams: *streams, Namespace: *ns, Token: *token,
		SockSndBuf: *sockbuf, SockRcvBuf: *sockbuf, SkipVerify: *noverify,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer c.Close()

	n, sz := *nBlocks, *blockKiB<<10
	poolN := *pool
	if poolN < n {
		poolN = n // pool must hold at least one batch
	}
	// Seed a pool of poolN distinct blocks; each GET draws a batch of n keys.
	// A pool larger than the CPU cache makes the source blocks DRAM-cold on
	// re-read (realistic large-store shape), isolating working-set effects
	// from the per-batch response size.
	pkeys := make([][32]byte, poolN)
	fillStart := time.Now()
	for i := range pkeys {
		binary.LittleEndian.PutUint32(pkeys[i][:], uint32(i)+1) //nolint:gosec // G115: bench index
		blob := make([]byte, sz)
		for j := range blob {
			blob[j] = byte(i)
		}
		if err := c.Put(ctx, pkeys[i], blob); err != nil {
			fmt.Fprintf(os.Stderr, "put %d: %v\n", i, err)
			c.Close()
			os.Exit(1) //nolint:gocritic // pool closed above; nothing else to release
		}
		if *fillMBps > 0 {
			// Pacing: bytes written so far may not outrun rate*elapsed (MiB/s).
			ahead := time.Duration(int64(i+1)*int64(sz)*1000/(int64(*fillMBps)<<20)) * time.Millisecond
			if lead := ahead - time.Since(fillStart); lead > 0 {
				time.Sleep(lead)
			}
		}
	}

	// Guard against the write-once-reuse artifact: if this daemon already held
	// these keys at a DIFFERENT block size (a prior getbench run at another
	// -block-kib), our Put silently returned OK_EXISTS and the store still has
	// the OLD size — but total is credited at the NEW size, inflating GB/s
	// (this is how a stale daemon once produced a physically-impossible reading).
	// Fetch key 0 and require the stored length to match; refuse to report a
	// bogus number.
	{
		probe := make([][]byte, 1)
		if _, err := c.BatchGet(ctx, pkeys[:1], probe); err != nil {
			fmt.Fprintln(os.Stderr, "probe get:", err)
			c.Close()
			os.Exit(1) //nolint:gocritic
		}
		if len(probe[0]) == 0 {
			fmt.Fprintln(os.Stderr, "probe: key 0 is GONE right after the fill — the store evicted or refused it."+
				" Disambiguate via kvb_nvme_admit_refusals_total: nonzero = the min-hits endurance gate deleted the"+
				" pure-PUT pool (set nvme_admit_min_hits: 0); zero = the fill outran demotion and eviction ate it"+
				" (throttle with -fill-mbps below the device write ceiling)")
			c.Close()
			os.Exit(1) //nolint:gocritic
		}
		if len(probe[0]) != sz {
			fmt.Fprintf(os.Stderr, "stale daemon: key 0 stored at %d bytes, want %d — restart the daemon before a different -block-kib\n", len(probe[0]), sz)
			c.Close()
			os.Exit(1) //nolint:gocritic
		}
	}

	var total atomic.Int64
	stop := make(chan struct{})
	var wg sync.WaitGroup
	for w := 0; w < *streams; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			into := make([][]byte, n)
			batch := make([][32]byte, n)
			off := (w * n) % poolN // each worker starts at a different pool offset
			for {
				select {
				case <-stop:
					return
				default:
				}
				for k := 0; k < n; k++ {
					batch[k] = pkeys[(off+k)%poolN]
				}
				off = (off + n) % poolN // advance so consecutive GETs hit fresh (colder) blocks
				if _, err := c.BatchGet(ctx, batch, into); err != nil {
					fmt.Fprintln(os.Stderr, "get:", err)
					return
				}
				total.Add(int64(n) * int64(sz))
			}
		}(w)
	}
	time.Sleep(time.Duration(*secs) * time.Second)
	close(stop)
	wg.Wait()
	gb := float64(total.Load()) / 1e9
	fmt.Printf("streams=%d batch=%dx%dKiB pool=%d (%dMiB): %.2f GB/s (%.1f GB in %ds)\n",
		*streams, n, *blockKiB, poolN, poolN**blockKiB>>10, gb/float64(*secs), gb, *secs)
}
