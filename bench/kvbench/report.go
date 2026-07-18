package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"

	"github.com/kvstash/kvblockd/bench/kvbench/internal/stats"
)

// cmdReport aggregates JSONL results and runs the executable acceptance
// gates: --check-repeat enforces "open-loop p99 repeatable within 2% across
// two identical runs" (SPEC-4 §11), and the default mode prints a compact
// CSV the eye (and plot.py) can scan.
func cmdReport(_ context.Context, args []string) error {
	fs := flag.NewFlagSet("report", flag.ExitOnError)
	var (
		checkRepeat = fs.String("check-repeat", "", "second JSONL to compare against the first positional arg")
		tolerance   = fs.Float64("tolerance", 0.02, "max fractional p99 divergence for --check-repeat")
	)
	if err := fs.Parse(args); err != nil {
		return err
	}
	rest := fs.Args()
	if len(rest) == 0 {
		return fmt.Errorf("report needs at least one JSONL file")
	}

	if *checkRepeat != "" {
		return checkRepeatability(rest[0], *checkRepeat, *tolerance)
	}

	// Default: CSV summary across every record.
	recs, err := loadRecords(rest...)
	if err != nil {
		return err
	}
	fmt.Println("store,cell,mode,rate_frac,ops_per_s,goodput_gbytes_s,ratio_vs_ceiling,get_p99_us,hit_rate,client_cores,daemon_cores,saturated")
	for _, r := range recs {
		g := r.Ops["get"]
		fmt.Printf("%s,%s,%s,%.2f,%.0f,%.4f,%.3f,%.1f,%.4f,%.3f,%.3f,%v\n",
			r.Store, r.Cell.ID, r.Cell.Mode, r.Cell.RateFrac, r.OpsPerS,
			r.GoodputGBytesS, r.RatioVsCeiling, g.P99Us, r.HitRate,
			r.CPU.ClientCores, r.CPU.DaemonCores, r.SaturatedRun)
	}
	return nil
}

func loadRecords(paths ...string) ([]stats.CellRecord, error) {
	var out []stats.CellRecord
	for _, p := range paths {
		f, err := os.Open(p) //nolint:gosec // G304: operator-chosen results file
		if err != nil {
			return nil, err
		}
		sc := bufio.NewScanner(f)
		sc.Buffer(make([]byte, 1<<20), 8<<20)
		for sc.Scan() {
			if len(sc.Bytes()) == 0 {
				continue
			}
			var r stats.CellRecord
			if err := json.Unmarshal(sc.Bytes(), &r); err != nil {
				_ = f.Close()
				return nil, fmt.Errorf("%s: %w", p, err)
			}
			out = append(out, r)
		}
		if err := sc.Err(); err != nil {
			_ = f.Close()
			return nil, err
		}
		_ = f.Close()
	}
	return out, nil
}

// checkRepeatability compares open-loop, non-saturated cells by (id, rate
// fraction) and fails if any GET-p99 pair diverges beyond the tolerance —
// the acceptance gate made executable.
func checkRepeatability(aPath, bPath string, tol float64) error {
	a, err := loadRecords(aPath)
	if err != nil {
		return err
	}
	b, err := loadRecords(bPath)
	if err != nil {
		return err
	}
	key := func(r stats.CellRecord) string {
		return fmt.Sprintf("%s|%s|%.2f", r.Store, r.Cell.ID, r.Cell.RateFrac)
	}
	bmap := map[string]stats.CellRecord{}
	for _, r := range b {
		if r.Cell.Mode == "open" && !r.SaturatedRun {
			bmap[key(r)] = r
		}
	}
	compared, worst := 0, 0.0
	var worstKey string
	failures := 0
	for _, ra := range a {
		if ra.Cell.Mode != "open" || ra.SaturatedRun {
			continue
		}
		rb, ok := bmap[key(ra)]
		if !ok {
			continue
		}
		pa, pb := ra.Ops["get"].P99Us, rb.Ops["get"].P99Us
		if pa <= 0 || pb <= 0 {
			continue
		}
		div := math.Abs(pa-pb) / math.Max(pa, pb)
		compared++
		if div > worst {
			worst, worstKey = div, key(ra)
		}
		if div > tol {
			failures++
			fmt.Printf("REPEAT-FAIL %s: p99 %.1fµs vs %.1fµs = %.1f%% > %.1f%%\n",
				key(ra), pa, pb, div*100, tol*100)
		}
	}
	if compared == 0 {
		return fmt.Errorf("no comparable open-loop cells between the two files")
	}
	fmt.Printf("check-repeat: %d cells compared, worst divergence %.2f%% at %s (tolerance %.1f%%)\n",
		compared, worst*100, worstKey, tol*100)
	if failures > 0 {
		return fmt.Errorf("%d cells exceeded the %.1f%% repeatability tolerance", failures, tol*100)
	}
	return nil
}
