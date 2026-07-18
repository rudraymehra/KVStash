package main

import (
	"context"
	"flag"
	"time"

	"github.com/kvstash/kvblockd/bench/kvbench/internal/gen"
	"github.com/kvstash/kvblockd/bench/kvbench/internal/stats"
	"github.com/kvstash/kvblockd/bench/kvbench/internal/target"
)

// cmdFill seeds [0, keys) of the deterministic pool — the warm-state
// parity primitive (every store in a comparison gets the same fill before
// its 10s warmup, per the methodology).
func cmdFill(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("fill", flag.ExitOnError)
	var tf targetFlags
	tf.register(fs)
	var (
		keys = fs.Int("keys", 4096, "pool size")
		blob = fs.Int("blob-bytes", gen.BlobSmall, "blob size (exact bytes)")
		seed = fs.Uint64("seed", 1, "workload seed")
		out  = fs.String("out", "", "JSONL path (empty = stdout)")
	)
	if err := fs.Parse(args); err != nil {
		return err
	}
	t, storeName, err := dialTarget(ctx, &tf, *blob)
	if err != nil {
		return err
	}
	defer func() { _ = t.Close() }()

	ks := gen.Keyspace{Seed: *seed ^ uint64(*blob)} //nolint:gosec // G115: blob sizes are small positive
	start := time.Now()
	if err := ensurePool(ctx, t, target.LimitOf(t), ks, *blob, *keys, *seed); err != nil {
		return err
	}
	took := time.Since(start).Seconds()

	w, err := stats.NewWriter(*out)
	if err != nil {
		return err
	}
	defer func() { _ = w.Close() }()
	rec := stats.NewCellRecord("fill", storeName, *seed)
	rec.Cell = stats.CellMeta{BlobBytes: *blob, Mode: "fill"}
	rec.DurationS = took
	rec.OpsPerS = float64(*keys) / took
	rec.GoodputGBytesS = float64(*keys) * float64(*blob) / 1e9 / took
	rec.Ops = map[string]stats.OpSummary{}
	return w.Write(rec)
}
