package replay

import (
	"bytes"
	"context"
	"strconv"
	"strings"
	"testing"

	"github.com/kvstash/kvblockd/bench/kvbench/internal/kvops"
	"github.com/kvstash/kvblockd/bench/kvbench/internal/loadgen"
	"github.com/kvstash/kvblockd/bench/kvbench/internal/stats"
	"github.com/kvstash/kvblockd/bench/kvbench/internal/target"
)

// buildTrace writes a small .kvops stream: chains over a repeating key
// population so replays produce predictable hit patterns.
func buildTrace(t *testing.T, blob uint32, chains [][]string) []byte {
	t.Helper()
	var buf bytes.Buffer
	w, err := kvops.NewWriter(&buf, blob, kvops.Meta{Trace: "unit", Converter: "test"})
	if err != nil {
		t.Fatal(err)
	}
	for i, chain := range chains {
		keys := make([][32]byte, len(chain))
		for j, id := range chain {
			keys[j] = kvops.TraceKey("unit", id)
		}
		if err := w.Write(uint64(i)*1000, keys); err != nil { //nolint:gosec // G115: small index
			t.Fatal(err)
		}
	}
	if err := w.Flush(); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func TestReplayAdaptiveHitRate(t *testing.T) {
	// Trace: A,B → A,B,C → A,B,C,D. First record all-miss (3 PUTs at 0
	// hits? no: chain A,B → 0 hits, 2 puts). Second: A,B hit, C put.
	// Third: A,B,C hit, D put. Total keys 2+3+4=9, hits 0+2+3=5.
	raw := buildTrace(t, 4096, [][]string{
		{"A", "B"},
		{"A", "B", "C"},
		{"A", "B", "C", "D"},
	})
	rd, err := kvops.NewReader(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	mem := target.NewMem(0, 0)
	var opLog bytes.Buffer
	res, err := Run(context.Background(), mem, rd, loadgen.NewWallClock(), Config{
		Workers: 1, PayloadSeed: 3, OpLog: &opLog,
	}, []*stats.Recorder{stats.NewRecorder()})
	if err != nil {
		t.Fatal(err)
	}
	if res.Records != 3 || res.KeysTotal != 9 {
		t.Fatalf("records=%d keys=%d", res.Records, res.KeysTotal)
	}
	if res.Hits != 5 || res.Misses != 4 {
		t.Fatalf("hits=%d misses=%d (want 5/4)", res.Hits, res.Misses)
	}
	if res.HitRate < 0.55 || res.HitRate > 0.56 {
		t.Fatalf("hit rate %.3f, want 5/9", res.HitRate)
	}
	if res.PutBytes != 4*4096 || res.GetBytes != 5*4096 {
		t.Fatalf("bytes: get=%d put=%d", res.GetBytes, res.PutBytes)
	}
	// The oplog is the Go↔Python parity artifact — deterministic at
	// workers=1.
	want := "0 exists 2 get 0 put 2\n1 exists 3 get 2 put 1\n2 exists 4 get 3 put 1\n"
	if opLog.String() != want {
		t.Fatalf("oplog:\n%s\nwant:\n%s", opLog.String(), want)
	}
}

func TestReplayCapacityMakesHitRateAnOutput(t *testing.T) {
	// The SAME trace at two store capacities yields two hit rates —
	// nobody set them. 20 sequential single-key chains, replayed twice
	// (second pass re-references the first pass's keys).
	var chains [][]string
	for pass := 0; pass < 2; pass++ {
		for i := 0; i < 20; i++ {
			chains = append(chains, []string{"k" + strconv.Itoa(i)})
		}
	}
	run := func(capBytes int64) float64 {
		raw := buildTrace(t, 4096, chains)
		rd, err := kvops.NewReader(bytes.NewReader(raw))
		if err != nil {
			t.Fatal(err)
		}
		res, err := Run(context.Background(), target.NewMem(capBytes, 0), rd,
			loadgen.NewWallClock(), Config{Workers: 1, PayloadSeed: 3},
			[]*stats.Recorder{stats.NewRecorder()})
		if err != nil {
			t.Fatal(err)
		}
		return res.HitRate
	}
	unbounded := run(0)   // all 20 resident on pass 2 → 50%
	tiny := run(5 * 4096) // FIFO holds 5 → pass 2 misses most
	if unbounded < 0.49 || unbounded > 0.51 {
		t.Fatalf("unbounded hit rate %.3f, want 0.50", unbounded)
	}
	if tiny >= unbounded {
		t.Fatalf("capacity did not move the hit rate: tiny=%.3f unbounded=%.3f", tiny, unbounded)
	}
}

func TestReplayOpLogStableAcrossRuns(t *testing.T) {
	chains := [][]string{{"x"}, {"x", "y"}, {"y", "z"}}
	logOf := func() string {
		raw := buildTrace(t, 4096, chains)
		rd, _ := kvops.NewReader(bytes.NewReader(raw))
		var lg strings.Builder
		_, err := Run(context.Background(), target.NewMem(0, 0), rd,
			loadgen.NewWallClock(), Config{Workers: 1, PayloadSeed: 1, OpLog: &lg},
			[]*stats.Recorder{stats.NewRecorder()})
		if err != nil {
			t.Fatal(err)
		}
		return lg.String()
	}
	first, second := logOf(), logOf()
	if first != second {
		t.Fatal("workers=1 oplog not deterministic")
	}
}
