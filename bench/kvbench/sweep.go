package main

import (
	"context"
	"flag"
	"fmt"
	"math/rand/v2"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"

	"github.com/kvstash/kvblockd/bench/kvbench/internal/gen"
	"github.com/kvstash/kvblockd/bench/kvbench/internal/loadgen"
	"github.com/kvstash/kvblockd/bench/kvbench/internal/stats"
	"github.com/kvstash/kvblockd/bench/kvbench/internal/target"
)

// cmdSweep runs grid cells: per cell, fill the pool (idempotent), run the
// closed-loop ceiling, then the CO-safe open-loop sweep at
// {10,30,50,70,90,110}% of that ceiling. One JSONL record per run.
func cmdSweep(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("sweep", flag.ExitOnError)
	var tf targetFlags
	tf.register(fs)
	var (
		headline = fs.Bool("headline", false, "only the chart cells ({small,large} blob × batch 32 × GET × uniform)")
		filter   = fs.String("filter", "", "substring match on cell IDs")
		pool     = fs.Int("pool", 4096, "distinct keys in the GET pool")
		seed     = fs.Uint64("seed", 1, "workload seed")
		warmup   = fs.Duration("warmup", 10*time.Second, "per-run warmup")
		duration = fs.Duration("duration", 60*time.Second, "per-run measured window")
		openRuns = fs.Bool("open", true, "run the open-loop rate sweep after the closed ceiling")
		out      = fs.String("out", "", "JSONL path (empty = stdout)")
		hgrmDir  = fs.String("hgrm-dir", "", "write per-run .hgrm files here")
		rig      = fs.String("rig", "", "rig label for provenance")
		ceiling  = fs.Float64("ceiling-gbytes", 0, "external same-rig ceiling (iperf3/fio) for ratio_vs_ceiling")
		dMetrics = fs.String("daemon-metrics", "", "daemon /metrics URL for cores accounting")
		dPID     = fs.Int("daemon-pid", 0, "daemon PID (loopback /proc fallback)")
		quick    = fs.Bool("quick", false, "2s/1s windows — smoke preset")
	)
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *quick {
		*warmup, *duration = time.Second, 2*time.Second
	}

	w, err := stats.NewWriter(*out)
	if err != nil {
		return err
	}
	defer func() { _ = w.Close() }()

	cells := gen.Grid(gen.GridConfig{Headline: *headline, Filter: *filter})
	if len(cells) == 0 {
		return fmt.Errorf("no cells match")
	}
	probe := stats.DaemonProbe{MetricsURL: *dMetrics, PID: *dPID}

	for _, cell := range cells {
		if err := runCell(ctx, &tf, cell, cellCfg{
			pool: *pool, seed: *seed, warmup: *warmup, duration: *duration,
			open: *openRuns, rig: *rig, ceiling: *ceiling, hgrmDir: *hgrmDir,
		}, probe, w); err != nil {
			return fmt.Errorf("cell %s: %w", cell.ID(), err)
		}
	}
	return nil
}

type cellCfg struct {
	pool     int
	seed     uint64
	warmup   time.Duration
	duration time.Duration
	open     bool
	rig      string
	ceiling  float64
	hgrmDir  string
}

// runCell executes one grid cell end to end.
func runCell(ctx context.Context, tf *targetFlags, cell gen.Cell, cfg cellCfg, probe stats.DaemonProbe, w *stats.Writer) error {
	// Streams come from the CELL (the grid axis), not the flag.
	tfCell := *tf
	tfCell.streams = cell.Streams
	t, storeName, err := dialTarget(ctx, &tfCell, cell.BlobBytes)
	if err != nil {
		return err
	}
	defer func() { _ = t.Close() }()
	lim := target.LimitOf(t)

	// The pool keyspace is scoped by (seed, blob) so different blob sizes
	// never alias — and the stale-size guard below still protects against
	// a daemon holding older same-key blobs of another size (write-once).
	ks := gen.Keyspace{Seed: cfg.seed ^ uint64(cell.BlobBytes)} //nolint:gosec // G115: blob sizes are small positive
	iss := newIssuer(t, lim, ks, cell, cfg.pool, cfg.seed)

	needsPool := cell.Mix != gen.MixPut
	if needsPool {
		if err := ensurePool(ctx, t, lim, ks, cell.BlobBytes, cfg.pool, cfg.seed); err != nil {
			return err
		}
	}

	// Closed-loop ceiling first.
	closedRec := newRecorders(cell.Streams)
	clk := loadgen.NewWallClock()
	cpu0, d0, _, _ := cpuSnap(probe)
	cres, err := loadgen.RunClosed(ctx, clk, loadgen.ClosedConfig{
		Workers: cell.Streams, Warmup: cfg.warmup, Duration: cfg.duration, Seed: cfg.seed,
	}, iss.issue, closedRec)
	if err != nil {
		return err
	}
	cpu1, d1, dsrc, dRSS := cpuSnap(probe)
	if err := emit(w, emitArgs{
		kind: "sweep", store: storeName, mode: "closed", cell: cell, cfg: cfg,
		opsPerS: cres.OpsPerS, sum: stats.Merge(closedRec), errs: cres.Errors,
		payload: cres.PayloadBytes, measuredS: cres.MeasuredS,
		selfCPU: cpu1 - cpu0, daemonCPU: d1 - d0, daemonSrc: dsrc, daemonRSS: dRSS,
	}); err != nil {
		return err
	}

	if !cfg.open || cres.OpsPerS <= 0 {
		return nil
	}
	// The CO-safe open-loop sweep against the just-measured ceiling.
	for _, frac := range loadgen.RateFractions {
		rate := cres.OpsPerS * frac
		recs := newRecorders(cell.Streams)
		clk = loadgen.NewWallClock()
		cpu0, d0, _, _ = cpuSnap(probe)
		ores, err := loadgen.RunOpen(ctx, clk, loadgen.OpenConfig{
			Rate: rate, Warmup: cfg.warmup, Duration: cfg.duration,
			Workers: cell.Streams, Seed: cfg.seed,
		}, iss.issue, recs)
		if err != nil {
			return err
		}
		cpu1, d1, dsrc, dRSS = cpuSnap(probe)
		sum := stats.Merge(recs)
		var hgrmPaths []string
		if cfg.hgrmDir != "" {
			name := fmt.Sprintf("%s_%s_f%02.0f.hgrm", storeName, cell.ID(), frac*100)
			p := filepath.Join(cfg.hgrmDir, name)
			if err := writeHgrm(p, sum, cell.Mix); err != nil {
				return err
			}
			hgrmPaths = []string{p}
		}
		if err := emit(w, emitArgs{
			kind: "sweep", store: storeName, mode: "open", cell: cell, cfg: cfg,
			rate: rate, frac: frac, opsPerS: float64(ores.Recorded) / ores.MeasuredS,
			sum: sum, errs: ores.Errors, payload: ores.PayloadBytes, measuredS: ores.MeasuredS,
			selfCPU: cpu1 - cpu0, daemonCPU: d1 - d0, daemonSrc: dsrc, daemonRSS: dRSS,
			saturated: ores.Saturated, closedCeilingOpsS: cres.OpsPerS, hgrmPaths: hgrmPaths,
		}); err != nil {
			return err
		}
	}
	return nil
}

// issuer derives event i's op and keys from PCG(seed, i) — worker identity
// never enters the stream, so one seed ⇒ one op sequence.
type issuer struct {
	t     target.Target
	lim   int
	ks    gen.Keyspace
	cell  gen.Cell
	pool  int
	seed  uint64
	zipf  *gen.Zipfian
	bufs  sync.Pool // per-worker dst + blob scratch
	blobN int
	// putCtr hands out globally-unique PUT key indices across the closed
	// run AND every open-loop fraction of a cell (one issuer per cell). The
	// ladder caught the old per-event formula repeating keys across runs →
	// write-once no-op acks measured as real writes. Never reset.
	putCtr atomic.Int64
}

type issueBufs struct {
	keys [][32]byte
	dst  [][]byte
	blob []byte
}

func newIssuer(t target.Target, lim int, ks gen.Keyspace, cell gen.Cell, pool int, seed uint64) *issuer {
	iss := &issuer{t: t, lim: lim, ks: ks, cell: cell, pool: pool, seed: seed, blobN: cell.BlobBytes}
	if cell.Skew == gen.SkewZipf {
		// The embedded PRNG is unused here (events call Draw with their own
		// per-index uniforms); NewZipfian just wants a source.
		iss.zipf = gen.NewZipfian(rand.New(rand.NewPCG(seed, 0x51bf)), uint64(pool), 0.99) //nolint:gosec // G404: deterministic benchmark skew, not crypto
	}
	iss.bufs.New = func() any {
		b := &issueBufs{
			keys: make([][32]byte, cell.BatchKeys),
			dst:  make([][]byte, cell.BatchKeys),
			blob: make([]byte, cell.BlobBytes),
		}
		for i := range b.dst {
			b.dst[i] = make([]byte, cell.BlobBytes)
		}
		return b
	}
	return iss
}

// issue executes event idx.
func (iss *issuer) issue(ctx context.Context, idx int64) (stats.Op, int, error) {
	r := rand.New(rand.NewPCG(iss.seed, uint64(idx))) //nolint:gosec // G404: index-deterministic workload stream
	b := iss.bufs.Get().(*issueBufs)
	defer iss.bufs.Put(b)

	op := stats.OpGet
	switch iss.cell.Mix {
	case gen.MixPut:
		op = stats.OpPut
	case gen.Mix9010:
		if r.IntN(10) == 0 {
			op = stats.OpPut
		}
	case gen.MixGet:
	}

	if op == stats.OpPut {
		// GLOBALLY-unique keys: putCtr never repeats across runs, so every
		// PUT is a genuine write on every store (no write-once no-op acks).
		base := int(iss.putCtr.Add(int64(iss.cell.BatchKeys))) - iss.cell.BatchKeys
		nWritten := 0
		for j := 0; j < iss.cell.BatchKeys; j++ {
			b.keys[j] = iss.ks.Key(iss.pool + base + j)
			gen.FillPayload(b.blob, iss.seed, b.keys[j])
			sts, err := target.TiledPut(ctx, iss.t, iss.lim, b.keys[j:j+1], [][]byte{b.blob})
			if err != nil {
				return op, 0, err
			}
			if sts[0].Wrote() { // count bytes only for genuine writes (Exists moved nothing)
				nWritten += iss.blobN
			}
		}
		return op, nWritten, nil
	}

	for j := 0; j < iss.cell.BatchKeys; j++ {
		var ki int
		if iss.zipf != nil {
			ki = int(iss.zipf.Draw(r.Float64())) //nolint:gosec // G115: < pool
		} else {
			ki = r.IntN(iss.pool)
		}
		b.keys[j] = iss.ks.Key(ki)
	}
	sts, err := target.TiledGet(ctx, iss.t, iss.lim, b.keys, b.dst)
	if err != nil {
		return op, 0, err
	}
	nBytes := 0
	for _, st := range sts {
		if st == target.OK {
			nBytes += iss.blobN
		}
	}
	return op, nBytes, nil
}

// ensurePool fills [0, pool) idempotently and runs getbench's stale-size
// guard: a daemon already holding key 0 at a DIFFERENT size means stale
// state from an earlier run (write-once would silently keep old bytes) —
// refuse loudly.
func ensurePool(ctx context.Context, t target.Target, lim int, ks gen.Keyspace, blob, pool int, seed uint64) error {
	probeKey := [][32]byte{ks.Key(0)}
	dst := [][]byte{make([]byte, blob)}
	sts, err := t.BatchGet(ctx, probeKey, dst)
	if err != nil {
		return err
	}
	if sts[0] == target.OK && len(dst[0]) != blob {
		return fmt.Errorf("stale pool: key 0 holds %d bytes, want %d — restart the store or change --seed", len(dst[0]), blob)
	}
	const tile = 64
	keys := make([][32]byte, 0, tile)
	blobs := make([][]byte, 0, tile)
	scratch := make([]byte, blob)
	for i := 0; i < pool; i += tile {
		keys, blobs = keys[:0], blobs[:0]
		for j := i; j < min(i+tile, pool); j++ {
			k := ks.Key(j)
			gen.FillPayload(scratch, seed, k)
			cp := make([]byte, blob)
			copy(cp, scratch)
			keys = append(keys, k)
			blobs = append(blobs, cp)
		}
		if _, err := target.TiledPut(ctx, t, lim, keys, blobs); err != nil {
			return fmt.Errorf("fill: %w", err)
		}
	}
	return nil
}

func newRecorders(n int) []*stats.Recorder {
	rs := make([]*stats.Recorder, n)
	for i := range rs {
		rs[i] = stats.NewRecorder()
	}
	return rs
}

func cpuSnap(p stats.DaemonProbe) (self, daemon float64, source string, rss int64) {
	self = stats.SelfCPU()
	daemon, rss, source, _ = p.Sample()
	return self, daemon, source, rss
}

// emitArgs bundles one run's measured outputs (kills the old 18-positional
// emit signature the ladder flagged).
type emitArgs struct {
	kind, store, mode  string
	cell               gen.Cell
	cfg                cellCfg
	rate, frac         float64
	opsPerS            float64
	sum                *stats.Summary
	errs, payload      int64
	measuredS          float64
	selfCPU, daemonCPU float64
	daemonSrc          string
	daemonRSS          int64
	saturated          bool
	closedCeilingOpsS  float64
	hgrmPaths          []string
}

func emit(w *stats.Writer, a emitArgs) error {
	rec := stats.NewCellRecord(a.kind, a.store, a.cfg.seed)
	rec.Rig = a.cfg.rig
	rec.Cell = stats.CellMeta{
		ID: a.cell.ID(), BlobBytes: a.cell.BlobBytes, BatchKeys: a.cell.BatchKeys,
		Streams: a.cell.Streams, Mix: string(a.cell.Mix), Skew: string(a.cell.Skew),
		Mode: a.mode, RateOpsS: a.rate, RateFrac: a.frac,
	}
	rec.WarmupS = a.cfg.warmup.Seconds()
	rec.DurationS = a.measuredS
	rec.Ops = map[string]stats.OpSummary{
		"get":    a.sum.Op(stats.OpGet),
		"put":    a.sum.Op(stats.OpPut),
		"exists": a.sum.Op(stats.OpExists),
	}
	lag := a.sum.Op(stats.OpLag)
	rec.Sched = stats.SchedStats{MaxLagUs: lag.MaxUs, P99LagUs: lag.P99Us, Saturated: a.saturated}
	rec.SaturatedRun = a.saturated
	rec.OpsPerS = a.opsPerS
	rec.ClosedCeilingOpsS = a.closedCeilingOpsS
	rec.HgrmPaths = a.hgrmPaths
	rec.GoodputGBytesS = float64(a.payload) / 1e9 / a.measuredS
	if a.cfg.ceiling > 0 {
		rec.CeilingGBytesS = a.cfg.ceiling
		rec.RatioVsCeiling = rec.GoodputGBytesS / a.cfg.ceiling
	}
	// CPU deltas span the WHOLE run (warmup + measured window), so divide by
	// the whole run's wall time, not just the measured window — the ladder
	// caught cores being overstated ~warmup/duration. Warmup runs the same
	// load, so total-cpu / total-time is the honest per-second rate.
	totalS := a.cfg.warmup.Seconds() + a.measuredS
	rec.CPU = stats.CPUSample{
		ClientCores: a.selfCPU / totalS, DaemonCores: a.daemonCPU / totalS,
		DaemonSource: a.daemonSrc, DaemonRSSBytes: a.daemonRSS,
	}
	rec.ErrorsTotal = a.errs
	return w.Write(rec)
}

// writeHgrm exports the cell's dominant-op distribution: PUT for a pure-put
// cell, GET otherwise (9010's latency story is its GETs).
func writeHgrm(path string, sum *stats.Summary, mix gen.Mix) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o750); err != nil {
		return err
	}
	f, err := os.Create(path) //nolint:gosec // G304: operator-chosen hgrm dir
	if err != nil {
		return err
	}
	defer func() { _ = f.Close() }()
	op := stats.OpGet
	if mix == gen.MixPut {
		op = stats.OpPut
	}
	return sum.WriteHGRM(op, f)
}
