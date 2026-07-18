// Package replay adaptively replays a .kvops trace against a live store:
// per record, EXISTS(chain) → k consecutive hits, GET(chain[:k]),
// PUT(chain[k:]) — the store performs its OWN eviction, so hit rate is an
// OUTPUT of the run, never an input knob. That single property is what
// separates an honest hit-rate claim from a marketing one.
package replay

import (
	"context"
	"fmt"
	"io"
	"sync"
	"sync/atomic"
	"time"

	"github.com/kvstash/kvblockd/bench/kvbench/internal/gen"
	"github.com/kvstash/kvblockd/bench/kvbench/internal/kvops"
	"github.com/kvstash/kvblockd/bench/kvbench/internal/loadgen"
	"github.com/kvstash/kvblockd/bench/kvbench/internal/stats"
	"github.com/kvstash/kvblockd/bench/kvbench/internal/target"
)

// Mode selects the pacing model.
type Mode uint8

const (
	// ASAP ignores trace timestamps — the closed-loop throughput variant.
	ASAP Mode = iota
	// Timed schedules record i at ts_us[i]/Speedup with open-loop
	// (scheduled-time) latency accounting.
	Timed
)

// Config tunes one replay run.
type Config struct {
	Mode        Mode
	Speedup     float64 // Timed: trace time compression factor (≥1 speeds up)
	Workers     int
	PayloadSeed uint64
	MaxLag      time.Duration // Timed saturation trip wire (default 5s)
	// OpLog, when non-nil, receives one "idx exists k get g put p" line per
	// record — the Go↔Python parity artifact (use --workers 1 for a fully
	// deterministic sequence).
	OpLog io.Writer
}

// Result is the run's aggregate truth.
type Result struct {
	Records   int64
	KeysTotal int64
	Hits      int64
	Misses    int64
	HitRate   float64 // OUTPUT
	GetBytes  int64
	PutBytes  int64
	MeasuredS float64
	Saturated bool
	Errors    int64
}

// Run replays rd against t. recs must hold Workers recorders.
func Run(ctx context.Context, t target.Target, rd *kvops.Reader, clk loadgen.Clock, cfg Config, recs []*stats.Recorder) (Result, error) {
	if cfg.Workers <= 0 {
		cfg.Workers = 1
	}
	if cfg.Speedup <= 0 {
		cfg.Speedup = 1
	}
	if cfg.MaxLag <= 0 {
		cfg.MaxLag = 5 * time.Second
	}
	lim := target.LimitOf(t)
	blob := int(rd.Header().BlobBytes)

	// The reader is sequential; a small ordered channel feeds workers.
	type job struct {
		idx   int64
		sched time.Duration // Timed only
		keys  [][32]byte    // copied (the reader reuses its backing)
	}
	jobs := make(chan job, cfg.Workers*2)

	var res Result
	var hits, misses, getB, putB, errs atomic.Int64
	var saturated atomic.Bool
	var logMu sync.Mutex

	start := time.Now()
	var wg sync.WaitGroup
	worker := func(rec *stats.Recorder) {
		defer wg.Done()
		dst := make([][]byte, 0, 64)
		blobBuf := make([]byte, blob)
		for j := range jobs {
			if ctx.Err() != nil {
				return
			}
			// SCHEDULED-time accounting (Tene rule): in Timed mode the
			// reference instant is the record's SCHEDULED time, not actual
			// dispatch. A record's replayed latency = completion − scheduled,
			// so a stalled store shows its queue wait — the ladder caught the
			// old code starting the clock AFTER SleepUntil, hiding exactly
			// that (the coordinated-omission error loadgen.RunOpen prevents).
			// A replay record is one request (EXISTS→GET→PUT chain); its
			// user-visible latency is the whole-record completion, recorded
			// once under OpGet (what Chart 2 reads). OpLag isolates the
			// dispatch queue-wait for the saturation diagnosis.
			ref := clk.Now()
			if cfg.Mode == Timed {
				clk.SleepUntil(j.sched)
				ref = j.sched
				lag := clk.Now() - ref
				if lag > cfg.MaxLag {
					saturated.Store(true)
				}
				rec.Observe(stats.OpLag, lag)
			}

			k, err := target.TiledExists(ctx, t, lim, j.keys)
			if err != nil {
				rec.CountError(stats.OpExists)
				errs.Add(1)
				continue
			}
			if k > 0 {
				for len(dst) < k {
					dst = append(dst, make([]byte, blob))
				}
				sts, err := target.TiledGet(ctx, t, lim, j.keys[:k], dst[:k])
				if err != nil {
					rec.CountError(stats.OpGet)
					errs.Add(1)
					continue
				}
				for _, st := range sts {
					if st == target.OK {
						getB.Add(int64(blob))
					}
				}
			}
			putErr := false
			if k < len(j.keys) {
				missKeys := j.keys[k:]
				nPut := 0
				for _, key := range missKeys {
					gen.FillPayload(blobBuf, cfg.PayloadSeed, key)
					sts, err := target.TiledPut(ctx, t, lim, [][32]byte{key}, [][]byte{blobBuf})
					if err != nil {
						rec.CountError(stats.OpPut)
						errs.Add(1)
						putErr = true
						break
					}
					if sts[0].Wrote() { // genuine write only (Exists moved no bytes)
						nPut++
					}
				}
				putB.Add(int64(nPut) * int64(blob))
			}
			if putErr {
				continue // errored record: no latency sample, no hit/miss credit
			}
			// The CO-safe record latency (queue wait included in Timed mode).
			rec.Observe(stats.OpGet, clk.Now()-ref)
			hits.Add(int64(k))
			misses.Add(int64(len(j.keys) - k))
			if cfg.OpLog != nil {
				logMu.Lock()
				fmt.Fprintf(cfg.OpLog, "%d exists %d get %d put %d\n", j.idx, len(j.keys), k, len(j.keys)-k)
				logMu.Unlock()
			}
		}
	}
	for w := 0; w < cfg.Workers; w++ {
		wg.Add(1)
		go worker(recs[w])
	}

	var rec kvops.Record
	var idx int64
	var ts0 uint64
	feedErr := func() error {
		for {
			if err := rd.Next(&rec); err != nil {
				if err == io.EOF { //nolint:errorlint // Reader returns bare io.EOF on clean end
					return nil
				}
				return err
			}
			if idx == 0 {
				ts0 = rec.TSMicros
			}
			cp := make([][32]byte, len(rec.Keys))
			copy(cp, rec.Keys)
			var sched time.Duration
			if cfg.Mode == Timed {
				sched = time.Duration(float64(rec.TSMicros-ts0) / cfg.Speedup * float64(time.Microsecond))
			}
			select {
			case jobs <- job{idx: idx, sched: sched, keys: cp}:
			case <-ctx.Done():
				return ctx.Err()
			}
			idx++
		}
	}()
	close(jobs)
	wg.Wait()

	res.Records = idx
	res.Hits = hits.Load()
	res.Misses = misses.Load()
	res.KeysTotal = res.Hits + res.Misses
	if res.KeysTotal > 0 {
		res.HitRate = float64(res.Hits) / float64(res.KeysTotal)
	}
	res.GetBytes = getB.Load()
	res.PutBytes = putB.Load()
	res.MeasuredS = time.Since(start).Seconds()
	res.Saturated = saturated.Load()
	res.Errors = errs.Load()
	return res, feedErr
}
