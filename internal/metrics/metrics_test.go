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
		`"pinned_bytes":{"7":1048576},` +
		`"evictions_total":12,"live_allocs":3,"max_allocs":131072}`)
}

func TestEndpointAndReadiness(t *testing.T) {
	set := New(fakeStats)
	set.Op(protocol.OpBatchGet, 0.0004)
	set.GetResult(7, "dram", 5, 1, 5<<20)
	set.GetResult(7, "nvme", 2, 0, 2<<20)
	set.GetBusy(7, 3)
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
		`kvb_hits_total{ns="7",tier="nvme"} 2`,
		`kvb_get_busy_total{ns="7"} 3`,
		`kvb_misses_total{ns="7"} 1`,
		`kvb_bytes_total{dir="out",ns="7"} 7.340032e+06`, // 5 MiB dram + 2 MiB nvme
		`kvb_bytes_total{dir="in",ns="7"} 1.048576e+06`,
		`kvb_evictions_total{tier="dram"} 12`,
		`kvb_live_allocs{tier="dram"} 3`,
		`kvb_max_allocs{tier="dram"} 131072`,
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

	// A DRAM-only stats document must emit NO kvb_nvme_* scrape families
	// (the tier="nvme" hits counter above is request-driven, not scrape-
	// driven — it only exists because this test recorded tiered GETs).
	if strings.Contains(body, "kvb_nvme_") {
		t.Error("DRAM-only scrape leaked nvme families")
	}

	// /debug/pprof reachable (the zero-alloc proof's capture point).
	if code := httpCode(t, addr, "/debug/pprof/"); code != http.StatusOK {
		t.Fatalf("pprof index: %d", code)
	}
}

// TestNvmeSubdocScrape: a tiered store's "nvme" sub-document surfaces the
// full kvb_nvme_* family plus the tier="nvme" splits.
func TestNvmeSubdocScrape(t *testing.T) {
	tiered := func() []byte {
		return []byte(`{"schema":1,"store":"dram","blocks":3,"bytes":3145728,` +
			`"arena_bytes":67108864,"arena_free_bytes":63963136,` +
			`"largest_free_region_bytes":63963136,"hugepages":false,` +
			`"pinned_bytes":{},"evictions_total":0,"live_allocs":3,"max_allocs":131072,` +
			`"nvme":{"blocks":40,"bytes":41943040,"segments":5,"used_bytes":1342177280,` +
			`"max_bytes":10737418240,"demotions_total":40,"demote_drops_total":2,` +
			`"admit_refusals_total":6,` +
			`"dedup_skips_total":1,"promotions_total":3,"reclaims_total":1,` +
			`"reclaim_skips_total":0,"read_busy_total":7,"checksum_errors_total":0,` +
			`"recovered_blocks":40,"recovery_seconds":0.42}}`)
	}
	set := New(tiered)
	ctx, cancel := context.WithCancel(context.Background())
	addr, wait, err := set.Serve(ctx, "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { cancel(); wait() }()
	body := httpBody(t, addr, "/metrics")
	for _, want := range []string{
		`kvb_blocks{tier="nvme"} 40`,
		`kvb_store_bytes{tier="nvme"} 4.194304e+07`,
		`kvb_nvme_segments 5`,
		`kvb_nvme_used_bytes 1.34217728e+09`,
		`kvb_nvme_max_bytes 1.073741824e+10`,
		`kvb_nvme_demotions_total 40`,
		`kvb_nvme_demote_drops_total 2`,
		`kvb_nvme_admit_refusals_total 6`,
		`kvb_nvme_dedup_skips_total 1`,
		`kvb_nvme_promotions_total 3`,
		`kvb_nvme_reclaims_total 1`,
		`kvb_nvme_read_busy_total 7`,
		`kvb_nvme_recovered_blocks 40`,
		`kvb_nvme_recovery_seconds 0.42`,
	} {
		if !strings.Contains(body, want) {
			t.Errorf("tiered scrape missing %q", want)
		}
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
