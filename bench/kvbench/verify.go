package main

import (
	"context"
	"flag"
	"fmt"
	"math/rand/v2"
	"time"

	"github.com/kvstash/kvblockd/bench/kvbench/internal/gen"
	"github.com/kvstash/kvblockd/bench/kvbench/internal/stats"
	"github.com/kvstash/kvblockd/bench/kvbench/internal/target"
)

// cmdVerify is the corruption oracle: GET stored pool keys, re-derive
// their payloads from (seed, key), and byte-compare. ANY mismatch is a
// failed benchmark (methodology rule 8) — nonzero exit.
func cmdVerify(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("verify", flag.ExitOnError)
	var tf targetFlags
	tf.register(fs)
	var (
		keys   = fs.Int("keys", 4096, "pool size to check")
		blob   = fs.Int("blob-bytes", gen.BlobSmall, "blob size (exact bytes)")
		seed   = fs.Uint64("seed", 1, "workload seed")
		sample = fs.Int("sample", 0, "check only N seeded-random keys (0 = all)")
		out    = fs.String("out", "", "JSONL path (empty = stdout)")
	)
	if err := fs.Parse(args); err != nil {
		return err
	}
	t, storeName, err := dialTarget(ctx, &tf, *blob)
	if err != nil {
		return err
	}
	defer func() { _ = t.Close() }()
	lim := target.LimitOf(t)
	ks := gen.Keyspace{Seed: *seed ^ uint64(*blob)} //nolint:gosec // G115: blob sizes are small positive

	idxs := make([]int, 0, *keys)
	if *sample > 0 && *sample < *keys {
		r := rand.New(rand.NewPCG(*seed, 0xC0FFEE)) //nolint:gosec // G404: seeded sample selection
		seen := map[int]bool{}
		for len(idxs) < *sample {
			i := r.IntN(*keys)
			if !seen[i] {
				seen[i] = true
				idxs = append(idxs, i)
			}
		}
	} else {
		for i := 0; i < *keys; i++ {
			idxs = append(idxs, i)
		}
	}

	const tile = 64
	var checked, missing, fails, retries int64
	start := time.Now()
	kbuf := make([][32]byte, 0, tile)
	dst := make([][]byte, tile)
	for i := range dst {
		dst[i] = make([]byte, *blob)
	}
	for off := 0; off < len(idxs); off += tile {
		end := min(off+tile, len(idxs))
		kbuf = kbuf[:0]
		for _, i := range idxs[off:end] {
			kbuf = append(kbuf, ks.Key(i))
		}
		sts, err := target.TiledGet(ctx, t, lim, kbuf, dst[:len(kbuf)])
		if err != nil {
			return err
		}
		for j, st := range sts {
			switch st {
			case target.OK:
				checked++
				if !gen.VerifyPayloadLen(dst[j], *blob, *seed, kbuf[j]) {
					fails++
					fmt.Printf("VERIFY-FAIL key=%x len=%d want=%d\n", kbuf[j][:8], len(dst[j]), *blob)
				}
			case target.Miss:
				missing++
			case target.Busy:
				retries++ // retryable backpressure — not corruption, re-probe later
			case target.Exists, target.Errored:
				fails++ // Exists can't happen on a GET; Errored is a real failure
			}
		}
	}
	took := time.Since(start).Seconds()

	w, err := stats.NewWriter(*out)
	if err != nil {
		return err
	}
	defer func() { _ = w.Close() }()
	rec := stats.NewCellRecord("verify", storeName, *seed)
	rec.Cell = stats.CellMeta{BlobBytes: *blob, Mode: "verify"}
	rec.DurationS = took
	rec.OpsPerS = float64(checked) / took
	rec.VerifyFails = fails
	rec.Ops = map[string]stats.OpSummary{}
	if err := w.Write(rec); err != nil {
		return err
	}
	fmt.Printf("verify: %d checked byte-identical, %d missing (evictions are legal), %d busy-retries, %d FAILURES\n",
		checked, missing, retries, fails)
	if fails > 0 {
		return fmt.Errorf("%d corrupt/unreadable blobs — a benchmark that returns wrong bytes is a failed benchmark", fails)
	}
	return nil
}
