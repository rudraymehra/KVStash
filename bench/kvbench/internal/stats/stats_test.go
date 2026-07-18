package stats

import (
	"bytes"
	"encoding/json"
	"math/rand/v2"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestRecorderPercentileAccuracy(t *testing.T) {
	// A known distribution: uniform 1..1000 µs. p50 must land within the
	// histogram's 3-sig-fig contract (±0.1%) of 500µs, p99 near 990µs.
	rs := make([]*Recorder, 4)
	rng := rand.New(rand.NewPCG(1, 1)) //nolint:gosec // G404: test distribution
	for i := range rs {
		rs[i] = NewRecorder()
	}
	const n = 400_000
	for i := 0; i < n; i++ {
		us := 1 + rng.Int64N(1000)
		rs[i%4].Observe(OpGet, time.Duration(us)*time.Microsecond)
	}
	s := Merge(rs)
	g := s.Op(OpGet)
	if g.N != n {
		t.Fatalf("merged N=%d", g.N)
	}
	within := func(got, want, tolPct float64) bool {
		return got >= want*(1-tolPct/100) && got <= want*(1+tolPct/100)
	}
	if !within(g.P50Us, 500, 1) {
		t.Fatalf("p50=%v, want ≈500µs", g.P50Us)
	}
	if !within(g.P99Us, 990, 1) {
		t.Fatalf("p99=%v, want ≈990µs", g.P99Us)
	}
	if g.MaxUs < 999 || g.MaxUs > 1001 {
		t.Fatalf("max=%v", g.MaxUs)
	}
}

func TestObserveClampsIntoRange(t *testing.T) {
	r := NewRecorder()
	r.Observe(OpPut, 1) // 1ns → clamped to 1µs floor
	r.Observe(OpPut, 2*time.Minute)
	s := Merge([]*Recorder{r})
	p := s.Op(OpPut)
	if p.N != 2 {
		t.Fatalf("N=%d", p.N)
	}
	if p.MaxUs > 61_000_000 {
		t.Fatalf("max %vµs above the 60s cap", p.MaxUs)
	}
}

func TestHGRMWellFormed(t *testing.T) {
	r := NewRecorder()
	for i := 1; i <= 1000; i++ {
		r.Observe(OpGet, time.Duration(i)*time.Microsecond)
	}
	var buf bytes.Buffer
	if err := Merge([]*Recorder{r}).WriteHGRM(OpGet, &buf); err != nil {
		t.Fatal(err)
	}
	out := buf.String()
	if !strings.Contains(out, "Percentile") || !strings.Contains(out, "TotalCount") {
		t.Fatal("hgrm header missing")
	}
	if !strings.Contains(out, "#[Mean =") {
		t.Fatal("hgrm footer missing")
	}
	lines := strings.Split(strings.TrimSpace(out), "\n")
	if len(lines) < 10 {
		t.Fatalf("hgrm suspiciously short: %d lines", len(lines))
	}
}

func TestJSONLWriterRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "out.jsonl")
	w, err := NewWriter(path)
	if err != nil {
		t.Fatal(err)
	}
	rec := NewCellRecord("sweep", "kvblockd", 42)
	rec.Cell = CellMeta{ID: "b462848_k32_s8_get_uniform", BlobBytes: 462848, BatchKeys: 32, Streams: 8, Mix: "get", Skew: "uniform", Mode: "open", RateOpsS: 100}
	rec.GoodputGBytesS = 6.37
	rec.CeilingGBytesS = 6.23
	rec.RatioVsCeiling = 6.37 / 6.23
	rec.Ops = map[string]OpSummary{"get": {N: 10, P99Us: 900}}
	if err := w.Write(rec); err != nil {
		t.Fatal(err)
	}
	if err := w.Write(rec); err != nil { // append-mode second line
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(path) //nolint:gosec // G304: t.TempDir path
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	if len(lines) != 2 {
		t.Fatalf("%d lines", len(lines))
	}
	var back CellRecord
	if err := json.Unmarshal([]byte(lines[0]), &back); err != nil {
		t.Fatal(err)
	}
	if back.SchemaVersion != 1 || back.Store != "kvblockd" || back.Cell.ID != rec.Cell.ID ||
		back.GOOS == "" || back.GOARCH == "" || back.Seed != 42 {
		t.Fatalf("round-trip lost fields: %+v", back)
	}
	if back.Ops["get"].P99Us != 900 {
		t.Fatal("ops map lost")
	}
}

func TestScrapeProcess(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(
			"# HELP process_cpu_seconds_total x\n" +
				"process_cpu_seconds_total 12.5\n" +
				"process_resident_memory_bytes 1.048576e+06\n" +
				"kvb_blocks{tier=\"dram\"} 3\n",
		))
	}))
	defer srv.Close()
	cpu, rss, src, err := DaemonProbe{MetricsURL: srv.URL}.Sample()
	if err != nil {
		t.Fatal(err)
	}
	if cpu != 12.5 || rss != 1048576 || src != "metrics" {
		t.Fatalf("cpu=%v rss=%d src=%s", cpu, rss, src)
	}

	// An endpoint WITHOUT the process collector must fail loudly, not
	// silently report zero cores.
	empty := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("kvb_blocks{tier=\"dram\"} 3\n"))
	}))
	defer empty.Close()
	if _, _, _, err := (DaemonProbe{MetricsURL: empty.URL}).Sample(); err == nil {
		t.Fatal("missing process metrics not reported")
	}
}

func TestSelfCPUAdvances(t *testing.T) {
	a := SelfCPU()
	x := 0.0
	for i := 0; i < 20_000_000; i++ {
		x += float64(i % 7)
	}
	_ = x
	if b := SelfCPU(); b < a {
		t.Fatalf("cpu went backwards: %v -> %v", a, b)
	}
}
