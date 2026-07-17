package metrics

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/kvstash/kvblockd/internal/protocol"
)

// fakeStats mimics the DRAM tier's Stats() document.
func fakeStats() []byte {
	return []byte(`{"schema":1,"store":"dram","blocks":3,"bytes":3145728,` +
		`"arena_bytes":67108864,"arena_free_bytes":63963136,` +
		`"largest_free_region_bytes":63963136,"hugepages":false,` +
		`"pinned_bytes":{"7":1048576}}`)
}

func TestEndpointAndReadiness(t *testing.T) {
	set := New(fakeStats)
	set.Op(protocol.OpBatchGet, 0.0004)
	set.GetResult(7, 5, 1, 5<<20)
	set.PutCommitted(7, 1<<20)

	ctx, cancel := context.WithCancel(context.Background())
	addr, wait, err := set.Serve(ctx, "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { cancel(); wait() }()

	// /healthz: 503 until SetReady, then 200.
	if code := httpCode(t, addr, "/healthz"); code != http.StatusServiceUnavailable {
		t.Fatalf("healthz before ready: %d", code)
	}
	set.SetReady()
	if code := httpCode(t, addr, "/healthz"); code != http.StatusOK {
		t.Fatalf("healthz after ready: %d", code)
	}

	// /metrics: every kvb_* family present, values threaded through.
	body := httpBody(t, addr, "/metrics")
	for _, want := range []string{
		`kvb_hits_total{ns="7",tier="dram"} 5`,
		`kvb_misses_total{ns="7"} 1`,
		`kvb_bytes_total{dir="out",ns="7"} 5.24288e+06`,
		`kvb_bytes_total{dir="in",ns="7"} 1.048576e+06`,
		`kvb_evictions_total 0`,
		`kvb_blocks{tier="dram"} 3`,
		`kvb_store_bytes{tier="dram"} 3.145728e+06`,
		`kvb_arena_bytes{state="free"} 6.3963136e+07`,
		`kvb_arena_bytes{state="total"} 6.7108864e+07`,
		`kvb_pinned_bytes{ns="7"} 1.048576e+06`,
		"go_gc_duration_seconds", // the Go collector (launch-day GC defense)
	} {
		if !strings.Contains(body, want) {
			t.Errorf("metrics output missing %q", want)
		}
	}

	// /debug/pprof reachable (the zero-alloc proof's capture point).
	if code := httpCode(t, addr, "/debug/pprof/"); code != http.StatusOK {
		t.Fatalf("pprof index: %d", code)
	}
}

// TestLabelCardinalityBounded: op labels come from a fixed table — hammering
// every opcode (including unknown ones) must not mint unbounded series.
func TestLabelCardinalityBounded(t *testing.T) {
	set := New(nil)
	for op := 0; op < 256; op++ {
		set.Op(protocol.Opcode(op), 0.001) //nolint:gosec // G115: bounded loop
	}
	fams, err := set.reg.Gather()
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range fams {
		if f.GetName() != "kvb_op_seconds" {
			continue
		}
		// 9 named verbs + "unknown".
		if n := len(f.GetMetric()); n != 10 {
			t.Fatalf("kvb_op_seconds series = %d, want 10 (bounded by the opName table)", n)
		}
		return
	}
	t.Fatal("kvb_op_seconds not gathered")
}

// TestStatsDecodeFailureIsSilent: a broken stats document yields no samples,
// never a scrape error.
func TestStatsDecodeFailureIsSilent(t *testing.T) {
	set := New(func() []byte { return []byte("not json") })
	if _, err := set.reg.Gather(); err != nil {
		t.Fatalf("gather with a broken stats doc: %v", err)
	}
}

func httpGet(t *testing.T, addr, path string) *http.Response {
	t.Helper()
	resp, err := http.Get(fmt.Sprintf("http://%s%s", addr, path)) //nolint:noctx // test-local
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

func httpCode(t *testing.T, addr, path string) int {
	t.Helper()
	resp := httpGet(t, addr, path)
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, resp.Body)
	return resp.StatusCode
}

func httpBody(t *testing.T, addr, path string) string {
	t.Helper()
	resp := httpGet(t, addr, path)
	defer resp.Body.Close()
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}
