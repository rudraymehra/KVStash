// Package loadgen owns the load scheduling: a closed-loop mode (workers
// back-to-back — the throughput ceiling) and an open-loop mode (Poisson
// arrivals on a GLOBAL precomputed timeline, latency measured from the
// SCHEDULED send time — the Tene rule, coordinated-omission-safe; all
// p99/p999 claims come from this mode).
//
// Design rule: the timeline is DATA, not goroutines. Arrival times are
// precomputed from the seed; workers claim events by atomic index; an
// event's op and keys derive from its INDEX, never from the worker that
// happens to run it — one seed ⇒ byte-identical op sequences regardless
// of scheduling jitter.
package loadgen

import (
	"context"
	"math"
	"math/rand/v2"
	"sync"
	"sync/atomic"
	"time"

	"github.com/kvstash/kvblockd/bench/kvbench/internal/stats"
)

// Clock is the time seam — the CO-safety unit test swaps in a fake.
type Clock interface {
	// Now is the monotonic offset since the clock's zero.
	Now() time.Duration
	// SleepUntil blocks until Now() >= t (never negative-sleeps).
	SleepUntil(t time.Duration)
}

// wallClock is the production clock.
type wallClock struct{ zero time.Time }

// NewWallClock starts a monotonic clock at zero-now.
func NewWallClock() Clock { return &wallClock{zero: time.Now()} }

func (c *wallClock) Now() time.Duration { return time.Since(c.zero) }

func (c *wallClock) SleepUntil(t time.Duration) {
	// Coarse sleep to just short of the deadline, then a yield loop —
	// time.Sleep alone overshoots by scheduler quanta, which would smear
	// the Poisson schedule at high rates.
	const spin = 500 * time.Microsecond
	for {
		now := c.Now()
		if now >= t {
			return
		}
		if d := t - now - spin; d > 0 {
			time.Sleep(d)
			continue
		}
		// Final approach: yield-spin.
		for c.Now() < t {
			runtimeGosched()
		}
		return
	}
}

// runtimeGosched is a var for testability of the spin loop.
var runtimeGosched = func() { yield() }

// ArrivalTimes precomputes n cumulative Poisson arrival offsets covering
// `total` at `rate` events/s from a PCG(seed) stream. Memory is small:
// the worst grid cell is a few million int64s.
func ArrivalTimes(seed uint64, rate float64, total time.Duration) []time.Duration {
	if rate <= 0 || total <= 0 {
		return nil
	}
	r := rand.New(rand.NewPCG(seed, 0x9E3779B97F4A7C15)) //nolint:gosec // G404: deterministic benchmark schedule, not crypto
	mean := float64(time.Second) / rate
	var ts []time.Duration
	var t float64
	for {
		// Exponential inter-arrival: -mean * ln(U), U ∈ (0,1].
		u := r.Float64()
		if u <= 0 {
			u = math.SmallestNonzeroFloat64
		}
		t += -mean * math.Log(u)
		d := time.Duration(t)
		if d >= total {
			return ts
		}
		ts = append(ts, d)
	}
}

// Issuer executes event i against the store: it derives the op and keys
// from the INDEX, performs the call, and reports the op class + payload
// bytes moved (for goodput) or an error.
type Issuer func(ctx context.Context, eventIdx int64) (op stats.Op, payloadBytes int, err error)

// OpenConfig tunes one open-loop cell.
type OpenConfig struct {
	Rate     float64       // events/s on the GLOBAL timeline
	Warmup   time.Duration // events scheduled before this are issued, not recorded
	Duration time.Duration // the measured window (total run = Warmup+Duration)
	Workers  int
	Seed     uint64
	// MaxLag flips the saturated flag when dispatch falls this far behind
	// schedule (default 5s). The run continues to the wall-clock end —
	// the 110% cell must still produce a JSONL line.
	MaxLag time.Duration
}

// OpenResult summarizes one open-loop run. Latency/lag distributions land
// in the per-worker recorders the caller passes in.
type OpenResult struct {
	Issued       int64
	Recorded     int64
	Errors       int64
	PayloadBytes int64 // measured window only
	Saturated    bool
	MeasuredS    float64
}

// RunOpen executes the cell. recs must hold cfg.Workers recorders (one per
// worker, merged by the caller).
func RunOpen(ctx context.Context, clk Clock, cfg OpenConfig, issue Issuer, recs []*stats.Recorder) (OpenResult, error) {
	if cfg.MaxLag <= 0 {
		cfg.MaxLag = 5 * time.Second
	}
	ts := ArrivalTimes(cfg.Seed, cfg.Rate, cfg.Warmup+cfg.Duration)
	endWall := cfg.Warmup + cfg.Duration
	var next atomic.Int64
	var res OpenResult
	var issued, recorded, errs, payload atomic.Int64
	var saturated atomic.Bool

	var wg sync.WaitGroup
	for w := 0; w < cfg.Workers; w++ {
		wg.Add(1)
		go func(rec *stats.Recorder) {
			defer wg.Done()
			for {
				i := next.Add(1) - 1
				if int(i) >= len(ts) || ctx.Err() != nil {
					return
				}
				sched := ts[i]
				clk.SleepUntil(sched)
				now := clk.Now()
				// HARD wall-clock cutoff (the ladder's confirmed throughput-
				// inflation fix): a run must NOT drain its whole backlog past
				// the measurement window and then divide by the nominal
				// duration — that reports achieved throughput ABOVE the
				// closed-loop ceiling, which is impossible. At saturation the
				// workers fall behind, hit endWall, and STOP dispatching, so
				// achieved < offered and the throughput curve bends honestly.
				if now >= endWall {
					saturated.Store(true)
					return
				}
				lag := now - sched // ≥0: dispatch lateness = queue growth
				if lag > cfg.MaxLag {
					saturated.Store(true)
				}
				op, nBytes, err := issue(ctx, i)
				done := clk.Now()
				issued.Add(1)
				if sched < cfg.Warmup {
					continue // issued for realism, excluded from the record
				}
				if err != nil {
					rec.CountError(op)
					errs.Add(1)
					continue
				}
				// THE rule: latency from the SCHEDULED time. A late start
				// (the previous op ran long) is part of THIS op's latency —
				// that is exactly the wait a real open-loop client suffers.
				rec.Observe(op, done-sched)
				rec.Observe(stats.OpLag, lag)
				recorded.Add(1)
				payload.Add(int64(nBytes))
			}
		}(recs[w])
	}
	wg.Wait()

	res.Issued = issued.Load()
	res.Recorded = recorded.Load()
	res.Errors = errs.Load()
	res.PayloadBytes = payload.Load()
	res.Saturated = saturated.Load()
	res.MeasuredS = cfg.Duration.Seconds()
	return res, ctx.Err()
}

// ClosedConfig tunes one closed-loop (throughput) cell.
type ClosedConfig struct {
	Workers  int
	Warmup   time.Duration
	Duration time.Duration
	Seed     uint64
}

// ClosedResult carries the ceiling the open-loop sweep divides against.
type ClosedResult struct {
	Ops          int64
	Errors       int64
	PayloadBytes int64
	OpsPerS      float64
	MeasuredS    float64
}

// RunClosed drives workers back-to-back (the getbench shape). Latencies
// are recorded too but the caller labels them closed-loop.
func RunClosed(ctx context.Context, clk Clock, cfg ClosedConfig, issue Issuer, recs []*stats.Recorder) (ClosedResult, error) {
	stopAt := cfg.Warmup + cfg.Duration
	var idx atomic.Int64
	var ops, errs, payload atomic.Int64

	var wg sync.WaitGroup
	for w := 0; w < cfg.Workers; w++ {
		wg.Add(1)
		go func(rec *stats.Recorder) {
			defer wg.Done()
			for ctx.Err() == nil {
				now := clk.Now()
				if now >= stopAt {
					return
				}
				i := idx.Add(1) - 1
				start := clk.Now()
				op, nBytes, err := issue(ctx, i)
				end := clk.Now()
				if start < cfg.Warmup {
					continue
				}
				if err != nil {
					rec.CountError(op)
					errs.Add(1)
					continue
				}
				rec.Observe(op, end-start)
				ops.Add(1)
				payload.Add(int64(nBytes))
			}
		}(recs[w])
	}
	wg.Wait()

	sec := cfg.Duration.Seconds()
	return ClosedResult{
		Ops:          ops.Load(),
		Errors:       errs.Load(),
		PayloadBytes: payload.Load(),
		OpsPerS:      float64(ops.Load()) / sec,
		MeasuredS:    sec,
	}, ctx.Err()
}

// RateFractions is the spec's open-loop sweep of the closed ceiling.
var RateFractions = []float64{0.10, 0.30, 0.50, 0.70, 0.90, 1.10}
