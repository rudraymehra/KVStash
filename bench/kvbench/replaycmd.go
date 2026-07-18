package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"github.com/kvstash/kvblockd/bench/kvbench/internal/kvops"
	"github.com/kvstash/kvblockd/bench/kvbench/internal/loadgen"
	"github.com/kvstash/kvblockd/bench/kvbench/internal/replay"
	"github.com/kvstash/kvblockd/bench/kvbench/internal/stats"
)

// cmdReplay drives a .kvops trace adaptively against the store. Capacity
// and policy are the DAEMON's config (recorded here as provenance) — hit
// rate is measured, never set.
func cmdReplay(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("replay", flag.ExitOnError)
	var tf targetFlags
	tf.register(fs)
	var (
		trace    = fs.String("kvops", "", ".kvops trace path")
		mode     = fs.String("mode", "timed", "timed | asap")
		speedup  = fs.Float64("speedup", 1, "timed: trace-time compression")
		seed     = fs.Uint64("seed", 1, "payload seed (misses regenerate deterministic blobs)")
		capacity = fs.Int64("capacity-bytes", 0, "daemon capacity (provenance only)")
		policy   = fs.String("policy", "", "daemon eviction policy (provenance only)")
		out      = fs.String("out", "", "JSONL path")
		oplog    = fs.String("oplog", "", "write op-sequence log here (parity artifact; use --streams 1)")
	)
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *trace == "" {
		return fmt.Errorf("replay needs --kvops")
	}
	tf2 := tf

	f, err := os.Open(*trace) //nolint:gosec // G304: operator-chosen trace
	if err != nil {
		return err
	}
	defer func() { _ = f.Close() }()
	rd, err := kvops.NewReader(f)
	if err != nil {
		return err
	}
	blob := int(rd.Header().BlobBytes)

	t, storeName, err := dialTarget(ctx, &tf2, blob)
	if err != nil {
		return err
	}
	defer func() { _ = t.Close() }()

	cfg := replay.Config{
		Speedup: *speedup, Workers: tf.streams, PayloadSeed: *seed,
	}
	switch *mode {
	case "timed":
		cfg.Mode = replay.Timed
	case "asap":
		cfg.Mode = replay.ASAP
	default:
		return fmt.Errorf("--mode must be timed or asap, got %q", *mode)
	}
	if *oplog != "" && tf.streams != 1 {
		return fmt.Errorf("--oplog requires --streams 1 (the op sequence is only deterministic single-worker; it's the Go↔Python parity artifact)")
	}
	var logF *os.File
	if *oplog != "" {
		logF, err = os.Create(*oplog) //nolint:gosec // G304: operator-chosen path
		if err != nil {
			return err
		}
		defer func() { _ = logF.Close() }()
		cfg.OpLog = logF
	}

	recs := make([]*stats.Recorder, tf.streams)
	for i := range recs {
		recs[i] = stats.NewRecorder()
	}
	res, err := replay.Run(ctx, t, rd, loadgen.NewWallClock(), cfg, recs)
	if err != nil {
		return err
	}
	sum := stats.Merge(recs)

	w, err := stats.NewWriter(*out)
	if err != nil {
		return err
	}
	defer func() { _ = w.Close() }()
	rec := stats.NewCellRecord("replay", storeName, *seed)
	rec.Cell = stats.CellMeta{
		BlobBytes: blob, Streams: tf.streams, Mode: *mode,
		CapacityBytes: *capacity, Policy: *policy,
		Trace: rd.Header().Meta.Trace, SpeedupF: *speedup,
	}
	rec.DurationS = res.MeasuredS
	rec.OpsPerS = float64(res.Records) / res.MeasuredS
	rec.GoodputGBytesS = float64(res.GetBytes) / 1e9 / res.MeasuredS
	rec.HitRate = res.HitRate
	rec.ErrorsTotal = res.Errors
	rec.SaturatedRun = res.Saturated
	rec.Ops = map[string]stats.OpSummary{
		"get": sum.Op(stats.OpGet), "put": sum.Op(stats.OpPut), "exists": sum.Op(stats.OpExists),
	}
	if err := w.Write(rec); err != nil {
		return err
	}
	fmt.Printf("replay: %d records, %d keys, hit_rate=%.4f (OUTPUT), get=%.2f GB/s, %.1fs%s\n",
		res.Records, res.KeysTotal, res.HitRate, rec.GoodputGBytesS, res.MeasuredS,
		map[bool]string{true: " [SATURATED]", false: ""}[res.Saturated])
	return nil
}
