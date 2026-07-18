package loadgen

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/kvstash/kvblockd/bench/kvbench/internal/stats"
)

// fakeClock is a deterministic virtual clock: Now() returns the current
// virtual time; SleepUntil advances it. A configurable service time is
// charged per issue via advance().
type fakeClock struct {
	mu  sync.Mutex
	now time.Duration
}

func (c *fakeClock) Now() time.Duration {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *fakeClock) SleepUntil(t time.Duration) {
	c.mu.Lock()
	if t > c.now {
		c.now = t
	}
	c.mu.Unlock()
}

func (c *fakeClock) advance(d time.Duration) {
	c.mu.Lock()
	c.now += d
	c.mu.Unlock()
}

func TestArrivalTimesPoissonShape(t *testing.T) {
	ts := ArrivalTimes(1, 1000, 2*time.Second) // ~2000 events
	if len(ts) < 1500 || len(ts) > 2500 {
		t.Fatalf("%d arrivals for 2s @1000/s", len(ts))
	}
	for i := 1; i < len(ts); i++ {
		if ts[i] < ts[i-1] {
			t.Fatal("arrivals not monotonic")
		}
	}
	// Determinism.
	ts2 := ArrivalTimes(1, 1000, 2*time.Second)
	if len(ts2) != len(ts) || ts2[0] != ts[0] || ts2[len(ts2)-1] != ts[len(ts)-1] {
		t.Fatal("schedule not seed-deterministic")
	}
	// Different seed differs.
	ts3 := ArrivalTimes(2, 1000, 2*time.Second)
	if len(ts3) == len(ts) && ts3[0] == ts[0] {
		t.Fatal("seed does not change the schedule")
	}
}

// TestCoordinatedOmissionSafety is THE test: a store that stalls must show
// the stall in scheduled-time latency even though every individual call
// "measures fast" from its actual send time. A send-time measurement
// (computed in-test for contrast) must stay flat — proving the two
// accountings genuinely differ under overload.
func TestCoordinatedOmissionSafety(t *testing.T) {
	clk := &fakeClock{}
	const serviceTime = 10 * time.Millisecond // >> 1ms inter-arrival: rate 1000/s, 1 worker
	var sendTimeMax time.Duration
	var mu sync.Mutex

	rec := stats.NewRecorder()
	cfg := OpenConfig{
		Rate:     1000,
		Warmup:   0,
		Duration: 500 * time.Millisecond,
		Workers:  1, // single worker → strict head-of-line queueing
		Seed:     7,
		MaxLag:   50 * time.Millisecond,
	}
	issue := func(_ context.Context, _ int64) (stats.Op, int, error) {
		start := clk.Now()
		clk.advance(serviceTime)
		mu.Lock()
		if d := clk.Now() - start; d > sendTimeMax {
			sendTimeMax = d
		}
		mu.Unlock()
		return stats.OpGet, 128, nil
	}
	res, err := RunOpen(context.Background(), clk, cfg, issue, []*stats.Recorder{rec})
	if err != nil {
		t.Fatal(err)
	}
	s := stats.Merge([]*stats.Recorder{rec})
	g := s.Op(stats.OpGet)

	// Send-time view: every call took exactly 10ms.
	if sendTimeMax > 11*time.Millisecond {
		t.Fatalf("send-time max %v — the contrast measurement broke", sendTimeMax)
	}
	// Scheduled-time view: the queue grows ~9ms per event; p99 must be
	// FAR above the 10ms service time (CO-safe accounting sees the wait).
	if g.P99Us < 10*float64(serviceTime.Microseconds()) {
		t.Fatalf("scheduled-time p99 %vµs does not expose the queue — coordinated omission!", g.P99Us)
	}
	if !res.Saturated {
		t.Fatal("50ms MaxLag not tripped by a 10x-overloaded cell")
	}
	if res.Recorded == 0 || res.Errors != 0 {
		t.Fatalf("res: %+v", res)
	}
}

// TestOpenLoopThroughputCappedAtSaturation pins the ladder's confirmed
// throughput-inflation fix: an OVERLOADED cell must record FEWER events than
// its offered rate implies (the workers hit the wall-clock cutoff and stop),
// so achieved throughput stays below the offered rate — never above the
// ceiling. Before the fix, the run drained its whole backlog and reported
// recorded/nominal-duration ≈ offered rate, a straight y=x line.
func TestOpenLoopThroughputCappedAtSaturation(t *testing.T) {
	clk := &fakeClock{}
	const service = 10 * time.Millisecond // one worker can do ~100/s
	rec := stats.NewRecorder()
	cfg := OpenConfig{
		Rate:     1000, // 10x what one worker can serve → saturation
		Warmup:   0,
		Duration: 500 * time.Millisecond,
		Workers:  1,
		Seed:     11,
		MaxLag:   50 * time.Millisecond,
	}
	issue := func(_ context.Context, _ int64) (stats.Op, int, error) {
		clk.advance(service)
		return stats.OpGet, 100, nil
	}
	res, err := RunOpen(context.Background(), clk, cfg, issue, []*stats.Recorder{rec})
	if err != nil {
		t.Fatal(err)
	}
	// One worker over a 500ms window at 10ms/op serves ~50, NOT the ~500 the
	// 1000/s offered rate would imply. Recorded must be far below offered.
	offered := 1000.0 * cfg.Duration.Seconds() // 500
	achievedPerS := float64(res.Recorded) / cfg.Duration.Seconds()
	if achievedPerS > 200 { // generous; true ceiling is ~100/s
		t.Fatalf("achieved %.0f ops/s at saturation — throughput not capped (offered %.0f)", achievedPerS, offered/cfg.Duration.Seconds())
	}
	if float64(res.Recorded) > offered*0.5 {
		t.Fatalf("recorded %d of ~%.0f offered — backlog drained past the window", res.Recorded, offered)
	}
	if !res.Saturated {
		t.Fatal("saturation flag not set")
	}
}

func TestOpenLoopDeterministicOpSequence(t *testing.T) {
	// Two runs, same seed, different worker counts: the set of issued
	// event indices with their derived ops must be identical (ops derive
	// from the INDEX, never the worker).
	run := func(workers int) map[int64]byte {
		clk := &fakeClock{}
		got := map[int64]byte{}
		var mu sync.Mutex
		recs := make([]*stats.Recorder, workers)
		for i := range recs {
			recs[i] = stats.NewRecorder()
		}
		issue := func(_ context.Context, i int64) (stats.Op, int, error) {
			op := byte(i % 3) //nolint:gosec // G115: i%3 ∈ [0,2] — index-derived op choice (the sweep does the same via PCG(seed,i))
			mu.Lock()
			got[i] = op
			mu.Unlock()
			return stats.OpGet, 0, nil
		}
		_, err := RunOpen(context.Background(), clk, OpenConfig{
			Rate: 500, Duration: 400 * time.Millisecond, Workers: workers, Seed: 3,
		}, issue, recs)
		if err != nil {
			t.Fatal(err)
		}
		return got
	}
	a, b := run(1), run(4)
	if len(a) == 0 || len(a) != len(b) {
		t.Fatalf("issued %d vs %d events", len(a), len(b))
	}
	for i, op := range a {
		if b[i] != op {
			t.Fatalf("event %d: op %d vs %d — worker identity leaked into the stream", i, op, b[i])
		}
	}
}

func TestWarmupExcluded(t *testing.T) {
	clk := &fakeClock{}
	rec := stats.NewRecorder()
	issue := func(_ context.Context, _ int64) (stats.Op, int, error) {
		clk.advance(100 * time.Microsecond)
		return stats.OpGet, 64, nil
	}
	res, err := RunOpen(context.Background(), clk, OpenConfig{
		Rate: 1000, Warmup: 200 * time.Millisecond, Duration: 200 * time.Millisecond,
		Workers: 2, Seed: 5,
	}, issue, []*stats.Recorder{rec, stats.NewRecorder()})
	if err != nil {
		t.Fatal(err)
	}
	if res.Issued <= res.Recorded {
		t.Fatalf("warmup not excluded: issued=%d recorded=%d", res.Issued, res.Recorded)
	}
	if res.PayloadBytes != res.Recorded*64 {
		t.Fatalf("payload accounting includes warmup: %d vs %d", res.PayloadBytes, res.Recorded*64)
	}
}

func TestRunClosedCountsAndCeiling(t *testing.T) {
	clk := &fakeClock{}
	recs := []*stats.Recorder{stats.NewRecorder(), stats.NewRecorder()}
	issue := func(_ context.Context, _ int64) (stats.Op, int, error) {
		clk.advance(time.Millisecond)
		return stats.OpGet, 1000, nil
	}
	res, err := RunClosed(context.Background(), clk, ClosedConfig{
		Workers: 2, Warmup: 50 * time.Millisecond, Duration: 450 * time.Millisecond, Seed: 1,
	}, issue, recs)
	if err != nil {
		t.Fatal(err)
	}
	if res.Ops == 0 || res.OpsPerS == 0 {
		t.Fatalf("empty closed run: %+v", res)
	}
	if res.PayloadBytes != res.Ops*1000 {
		t.Fatalf("payload accounting off: %d vs %d", res.PayloadBytes, res.Ops*1000)
	}
}
