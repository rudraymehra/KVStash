// Command soak is the mixed-load endurance driver: 70% BATCH_GET / 25%
// PUT / 5% DELETE across three namespaces, sized so the working set
// exceeds the arena and the daemon evicts CONTINUOUSLY for the whole run.
//
// Correctness is the point, not throughput: every GET hit is verified TWICE
// — the client's xxh3 check against the descriptor, plus a byte-compare
// against the locally REGENERATED payload (payload(ns,i) is a pure function
// of identity), which catches cross-key corruption xxh3 alone could only
// catch via a collision. A soak that returns wrong bytes is a failed soak.
//
// Output: one JSONL line per report interval with op rates, log-bucketed
// latency percentiles, hit/miss/error counters, and the daemon's
// evictions_total (scraped via the wire STATS verb).
package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/zeebo/xxh3"

	"github.com/kvstash/kvblockd/internal/protocol"
	"github.com/kvstash/kvblockd/pkg/client"
)

var namespaces = []string{"soak-a", "soak-b", "soak-c"}

// payload regenerates block i of namespace ns deterministically.
func payload(nsIdx, i, size int) []byte {
	out := make([]byte, size)
	seed := uint64(nsIdx+1)*0x9E3779B97F4A7C15 + uint64(i)*0xD1B54A32D192ED03 //nolint:gosec // deterministic content
	var b [8]byte
	for off := 0; off < size; off += 8 {
		binary.LittleEndian.PutUint64(b[:], seed^(uint64(off)*0xBF58476D1CE4E5B9)) //nolint:gosec // as above
		copy(out[off:], b[:])
	}
	return out
}

// blockSize is deterministic per block: 256 KiB – 2 MiB, mean ~1.1 MiB
// (uniform over the range; well above the 64 KiB node-pool rule for 8 GiB
// arenas).
func blockSize(nsIdx, i int) int {
	h := xxh3.Hash([]byte{byte(nsIdx), byte(i), byte(i >> 8), byte(i >> 16)}) //nolint:gosec // G115: bounded key indices
	return 256<<10 + int(h%(1792<<10))
}

func key(nsIdx, i int) [32]byte {
	var k [32]byte
	binary.LittleEndian.PutUint32(k[0:4], uint32(i))     //nolint:gosec // bounded key index
	binary.LittleEndian.PutUint32(k[4:8], uint32(nsIdx)) //nolint:gosec // 0..2
	k[9] = byte(i * 7)                                   //nolint:gosec // G115: mixing byte, wrap is fine
	k[31] = 0x50
	return k
}

// hist is a lock-free log-bucketed latency histogram (factor ~1.25).
type hist struct{ buckets [64]atomic.Int64 }

func (h *hist) observe(d time.Duration) {
	us := float64(d.Microseconds())
	if us < 1 {
		us = 1
	}
	b := int(math.Log(us) / math.Log(1.25))
	if b < 0 {
		b = 0
	}
	if b >= len(h.buckets) {
		b = len(h.buckets) - 1
	}
	h.buckets[b].Add(1)
}

// snapshotAndReset returns p50/p99/p999 in µs (bucket ceilings —
// conservative) and the sample count, zeroing the histogram.
func (h *hist) snapshotAndReset() (p50, p99, p999 float64, n int64) {
	var counts [64]int64
	for i := range h.buckets {
		counts[i] = h.buckets[i].Swap(0)
		n += counts[i]
	}
	if n == 0 {
		return 0, 0, 0, 0
	}
	pct := func(q float64) float64 {
		target := int64(math.Ceil(q * float64(n)))
		var cum int64
		for i, c := range counts {
			cum += c
			if cum >= target {
				return math.Pow(1.25, float64(i+1)) // ceiling
			}
		}
		return math.Pow(1.25, float64(len(counts)))
	}
	return pct(0.50), pct(0.99), pct(0.999), n
}

type counters struct {
	gets, hits, misses, puts, dels, errors atomic.Int64
	backpressure                           atomic.Int64 // ERR_QUOTA_BYTES / ERR_BUSY — expected under overcommit, retry-class
	verifyFails                            atomic.Int64
}

// countErr classifies an op error. Quota/busy are the documented
// backpressure statuses of a deliberately overcommitted tier. A CHECKSUM
// failure from the client's own xxh3 verification is a data-integrity
// failure — it counts as a verify failure (the soak's one unforgivable
// class), never as a generic error. Everything else is a real error.
func (c *counters) countErr(err error) {
	var se *client.StatusError
	if errors.As(err, &se) && (se.Status == protocol.StatusErrQuotaBytes || se.Status == protocol.StatusErrBusy) {
		c.backpressure.Add(1)
		return
	}
	if err != nil && strings.Contains(err.Error(), "xxh3") {
		c.verifyFails.Add(1)
		fmt.Fprintln(os.Stderr, "soak: VERIFY FAIL (client checksum):", err)
		return
	}
	c.errors.Add(1)
}

func main() {
	addr := flag.String("addr", "127.0.0.1:9440", "kvblockd address")
	token := flag.String("token", "soak-token", "bearer token (same for all three namespaces)")
	duration := flag.Duration("duration", 24*time.Hour, "run length")
	workers := flag.Int("workers", 24, "concurrent workers (spread across namespaces)")
	keys := flag.Int("keys", 10800, "total distinct blocks across the 3 namespaces (10800 × ~1.1 MiB ≈ 12 GiB working set over an 8 GiB arena)")
	report := flag.Duration("report", 10*time.Second, "JSONL report interval")
	out := flag.String("out", "", "JSONL output path (empty = stdout)")
	flag.Parse()

	w := os.Stdout
	if *out != "" {
		f, err := os.Create(*out)
		if err != nil {
			fmt.Fprintln(os.Stderr, "soak:", err)
			os.Exit(1)
		}
		defer f.Close()
		w = f
	}

	ctx, cancel := context.WithTimeout(context.Background(), *duration)
	defer cancel()

	clients := make([]*client.Client, len(namespaces))
	for i, ns := range namespaces {
		c, err := client.Dial(ctx, *addr, client.Options{
			Streams: *workers / len(namespaces), Namespace: ns, Token: *token,
		})
		if err != nil {
			fmt.Fprintln(os.Stderr, "soak: dial", ns, ":", err)
			os.Exit(1) //nolint:gocritic // exitAfterDefer: startup failure — nothing buffered yet, the skipped defers are inert
		}
		defer c.Close()
		clients[i] = c
	}

	var (
		ctr     counters
		getHist hist
		putHist hist
		wg      sync.WaitGroup
	)
	perNS := *keys / len(namespaces)
	for wi := 0; wi < *workers; wi++ {
		wg.Add(1)
		go func(wi int) {
			defer wg.Done()
			nsIdx := wi % len(namespaces)
			c := clients[nsIdx]
			rng := rand.New(rand.NewPCG(uint64(wi), 0x50AC)) //nolint:gosec // load shape, not crypto
			into := make([][]byte, 1)
			for ctx.Err() == nil {
				i := rng.IntN(perNS)
				k := key(nsIdx, i)
				switch r := rng.IntN(100); {
				case r < 70: // GET + double verification
					t0 := time.Now()
					keysArg := [][32]byte{k}
					sts, err := c.BatchGet(ctx, keysArg, into)
					getHist.observe(time.Since(t0))
					ctr.gets.Add(1)
					if err != nil {
						if ctx.Err() == nil {
							ctr.countErr(err)
						}
						continue
					}
					if sts[0] == protocol.StatusOK {
						ctr.hits.Add(1)
						want := payload(nsIdx, i, blockSize(nsIdx, i))
						if !bytes.Equal(into[0], want) {
							ctr.verifyFails.Add(1)
							fmt.Fprintf(os.Stderr, "soak: VERIFY FAIL ns=%d i=%d len=%d\n", nsIdx, i, len(into[0]))
						}
					} else {
						ctr.misses.Add(1)
						// Miss = evicted: refill — this IS the churn engine.
						data := payload(nsIdx, i, blockSize(nsIdx, i))
						if err := c.Put(ctx, k, data); err != nil && ctx.Err() == nil {
							ctr.countErr(err)
						}
					}
				case r < 95: // PUT (idempotent re-put or refill)
					data := payload(nsIdx, i, blockSize(nsIdx, i))
					t0 := time.Now()
					err := c.Put(ctx, k, data)
					putHist.observe(time.Since(t0))
					ctr.puts.Add(1)
					if err != nil && ctx.Err() == nil {
						ctr.countErr(err)
					}
				default: // DELETE (force — leases are transient here)
					if _, err := c.Delete(ctx, [][32]byte{k}, true); err != nil && ctx.Err() == nil {
						ctr.countErr(err)
					}
					ctr.dels.Add(1)
				}
			}
		}(wi)
	}

	// Reporter loop.
	enc := json.NewEncoder(w)
	tick := time.NewTicker(*report)
	defer tick.Stop()
	start := time.Now()
	for {
		select {
		case <-ctx.Done():
			wg.Wait()
			line := finalLine(&ctr, start)
			_ = enc.Encode(line)
			if vf, er := ctr.verifyFails.Load(), ctr.errors.Load(); vf > 0 || er > 0 {
				fmt.Fprintln(os.Stderr, "soak: FAILED —", vf, "verification failures,", er, "hard errors")
				os.Exit(1) //nolint:gocritic // exitAfterDefer: deliberate — a failed soak exits nonzero; the defers are process-lifetime
			}
			return
		case <-tick.C:
			g50, g99, g999, gn := getHist.snapshotAndReset()
			p50, p99, _, pn := putHist.snapshotAndReset()
			var evictions any
			if raw, err := clients[0].Stats(context.Background()); err == nil {
				var doc struct {
					Evictions uint64 `json:"evictions_total"`
					Blocks    int64  `json:"blocks"`
				}
				if json.Unmarshal(raw, &doc) == nil {
					evictions = map[string]any{"evictions_total": doc.Evictions, "blocks": doc.Blocks}
				}
			}
			_ = enc.Encode(map[string]any{
				"ts": time.Now().UTC().Format(time.RFC3339), "elapsed_s": int(time.Since(start).Seconds()),
				"get_n": gn, "get_p50_us": g50, "get_p99_us": g99, "get_p999_us": g999,
				"put_n": pn, "put_p50_us": p50, "put_p99_us": p99,
				"hits": ctr.hits.Load(), "misses": ctr.misses.Load(),
				"errors": ctr.errors.Load(), "backpressure": ctr.backpressure.Load(), "verify_fails": ctr.verifyFails.Load(),
				"daemon": evictions,
			})
		}
	}
}

func finalLine(ctr *counters, start time.Time) map[string]any {
	return map[string]any{
		"final": true, "elapsed_s": int(time.Since(start).Seconds()),
		"gets": ctr.gets.Load(), "hits": ctr.hits.Load(), "misses": ctr.misses.Load(),
		"puts": ctr.puts.Load(), "deletes": ctr.dels.Load(),
		"errors": ctr.errors.Load(), "backpressure": ctr.backpressure.Load(), "verify_fails": ctr.verifyFails.Load(),
	}
}
