// Command kvblockd is the single-binary KV-cache block store daemon: config →
// arena-backed DRAM tier (→ log-structured NVMe tier when nvme_paths is set,
// → async S3 cold tier when s3_bucket is set) → server.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"net"
	"os"
	"os/signal"
	"runtime"
	"syscall"
	"time"

	"github.com/kvstash/kvblockd/internal/config"
	"github.com/kvstash/kvblockd/internal/eviction"
	"github.com/kvstash/kvblockd/internal/metrics"
	"github.com/kvstash/kvblockd/internal/server"
	"github.com/kvstash/kvblockd/internal/store"
	"github.com/kvstash/kvblockd/internal/store/dram"
	"github.com/kvstash/kvblockd/internal/store/nvme"
	"github.com/kvstash/kvblockd/internal/store/s3spill"
	"github.com/kvstash/kvblockd/internal/tenant"
)

// version is stamped by the release build (-ldflags "-X main.version=…");
// "dev" means a non-release build.
var version = "dev"

// errDegradedShutdown marks a shutdown that abandoned resources to process
// exit (data-plane drain or s3compat handlers outliving their grace). It maps
// to its own exit code so systemd/monitoring can tell degraded from clean (0)
// and from startup failure (1).
var errDegradedShutdown = errors.New("degraded shutdown")

const exitDegradedShutdown = 3

func exitCode(err error) int {
	switch {
	case err == nil:
		return 0
	case errors.Is(err, errDegradedShutdown):
		return exitDegradedShutdown
	default:
		return 1
	}
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "kvblockd:", err)
		os.Exit(exitCode(err))
	}
}

// cleanupStack is run()'s shutdown ordering, declared ONCE at startup:
// each subsystem pushes its release as it starts and the single deferred
// run executes in reverse (acquire/release symmetry) on every path —
// early error return, degraded shutdown, clean exit. The alternative —
// re-deriving stop order by hand in every error branch — is how the
// spiller once outlived the store it serves by an accident of defer
// ordering.
type cleanupStack struct{ steps []func() }

func (c *cleanupStack) push(f func()) { c.steps = append(c.steps, f) }

func (c *cleanupStack) run() {
	for i := len(c.steps) - 1; i >= 0; i-- {
		c.steps[i]()
	}
	c.steps = nil
}

func run() (err error) {
	cfgPath := flag.String("config", "", "path to config YAML (empty = built-in defaults)")
	listen := flag.String("listen", "", "override listen_addr")
	namespaces := flag.String("namespaces", "", "override namespaces_path")
	showVersion := flag.Bool("version", false, "print version and exit")
	flag.Parse()
	if *showVersion {
		fmt.Printf("kvblockd %s (%s/%s)\n", version, runtime.GOOS, runtime.GOARCH)
		return nil
	}

	var ov config.Overrides
	if *listen != "" {
		ov.ListenAddr = listen
	}
	if *namespaces != "" {
		ov.NamespacesPath = namespaces
	}

	cfg, err := config.Load(*cfgPath, ov)
	if err != nil {
		return err
	}
	ns, err := server.LoadNamespaces(cfg.NamespacesPath)
	if err != nil {
		return err
	}
	// The registry is arena-ignorant, so the pin_quota ceiling check runs
	// here where both meet — an override above the arena would silently
	// unbound the pin cap (config.Validate bounds pinned_bytes_cap the same
	// way; the admin add path enforces it at runtime).
	if err := ns.Registry().ValidatePinQuotas(cfg.DramArenaBytes); err != nil {
		return fmt.Errorf("namespaces: %w", err)
	}
	// One accountant instance spans both tiers — dram charges/refunds its
	// side, the tiered orchestrator transfers/refunds the NVMe side.
	quotas := tenant.NewQuotas(ns.Registry())

	arena, err := dram.NewArena(cfg.DramArenaBytes, cfg.DramHugepages)
	if err != nil {
		return fmt.Errorf("dram arena: %w", err)
	}
	dstore := dram.New(arena, dram.Params{
		LeaseDefaultMS: cfg.LeaseDefaultMS,
		LeaseMaxMS:     cfg.LeaseMaxMS,
		PinnedBytesCap: cfg.PinnedBytesCap,
		PinCapFor:      ns.Registry().PinQuotaFor, // per-ns pin_quota overrides the global cap
		Quotas:         quotas,
	})

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	// The LIFO cleanup stack (see the type). closeStore is the ONE step with
	// per-path judgement: a failed drain leaves arena views (and pooled read
	// buffers) referenced by peers, so the degraded paths set skipStoreClose
	// and let process exit reclaim — unmapping then could send unrelated
	// process memory to a peer.
	var cleanup cleanupStack
	defer cleanup.run()
	skipStoreClose := false
	closeStore := dstore.Close
	cleanup.push(func() {
		if skipStoreClose {
			return
		}
		if cerr := closeStore(); cerr != nil && err == nil {
			err = fmt.Errorf("store close: %w", cerr)
		}
	})

	// Eviction: policy attach + the watermark goroutine. Its stop step runs
	// before every store close (LIFO) so no eviction free races the arena
	// unmap.
	var pol eviction.Policy
	if cfg.EvictionPolicy != "none" {
		ghost := cfg.EvictionGhostEntries
		if ghost == 0 {
			// Auto ceiling: one fingerprint per conceivable resident block
			// (arena / 64 KiB) — the policy itself stays arena-ignorant.
			// This is a CEILING, not a grant: each domain's ring tracks its
			// observed residency (mainHi, decayed per epoch), so tenant
			// churn no longer accumulates the sum of per-tenant peaks.
			ghost = int(cfg.DramArenaBytes >> 16)
		}
		var perr error
		pol, perr = eviction.New(cfg.EvictionPolicy, ghost)
		if perr != nil {
			return perr
		}
		dstore.AttachPolicy(pol)
		cleanup.push(dstore.StartEvictor(ctx, dram.EvictorConfig{
			WatermarkPct: cfg.EvictionWatermarkPct,
			BatchPct:     cfg.EvictionBatchPct,
		}))
		fmt.Fprintln(os.Stderr, "kvblockd: eviction policy", pol.Name(),
			"watermark", cfg.EvictionWatermarkPct, "batch", cfg.EvictionBatchPct)
	}

	// NVMe tier (nvme_paths set): open every volume — recovery (checkpoint +
	// footer scan + tail truncation) runs inside OpenVolume — then stack the
	// tiered orchestrator on top. The server sees ONE store either way.
	var srvStore server.Store = dstore
	statsFn := dstore.Stats
	if len(cfg.NvmePaths) > 0 {
		vols := make([]*nvme.Volume, 0, len(cfg.NvmePaths))
		reports := make([]*nvme.RecoveryReport, 0, len(cfg.NvmePaths))
		recovered := make([][]nvme.RecoveredEntry, 0, len(cfg.NvmePaths))
		perVol := cfg.NvmeMaxBytes / int64(len(cfg.NvmePaths))
		for _, dir := range cfg.NvmePaths {
			v, rep, ents, verr := nvme.OpenVolume(nvme.VolumeParams{
				Dir:            dir,
				SegmentBytes:   cfg.NvmeSegmentBytes,
				MaxBytes:       perVol,
				SyncEveryBytes: cfg.NvmeSyncEveryBytes,
				ReadWorkers:    cfg.NvmeReadWorkers,
				CkptEverySegs:  cfg.NvmeCkptEverySegments,
				MaxBlobLen:     cfg.MaxBlobLen,
			})
			if verr != nil {
				return fmt.Errorf("nvme volume %s: %w", dir, verr)
			}
			// Volume.Close is idempotent: once the tiered store owns the
			// volumes its Close (the closeStore step) closes them too; this
			// step covers the error paths before that ownership transfer and
			// honors skipStoreClose for the same reader-held-buffer reason.
			cleanup.push(func() {
				if !skipStoreClose {
					_ = v.Close()
				}
			})
			fmt.Fprintf(os.Stderr, "kvblockd: nvme volume %s recovered: %d segments scanned, %d blocks, %d bytes truncated, %s\n",
				dir, rep.SegmentsScanned, rep.BlocksRecovered, rep.BytesTruncated, rep.Duration)
			vols = append(vols, v)
			reports = append(reports, rep)
			recovered = append(recovered, ents)
		}
		var spillB store.SpillBackend
		var restoreB store.RestoreBackend
		if cfg.S3Bucket != "" {
			s3cfg := s3spill.Config{
				Bucket: cfg.S3Bucket, Region: cfg.S3Region, NodeID: cfg.S3NodeID,
				EndpointOverride: cfg.S3EndpointOverride, PathStyle: cfg.S3PathStyle,
			}
			api, aerr := s3spill.NewClient(ctx, s3cfg)
			if aerr != nil {
				return fmt.Errorf("s3 tier: %w", aerr)
			}
			sp := s3spill.NewSpiller(api, s3cfg, cfg.S3SpillQueue)
			// Pushed AFTER the volume steps and BEFORE the tier's stop: LIFO
			// then closes the spiller after the movers stop feeding it and
			// before the store (and its segment files) goes away — the
			// spiller once outlived the store by an accident of defer order.
			cleanup.push(sp.Close)
			rst := s3spill.NewRestorer(api, s3cfg)
			// Same slot: a belt-cut restore's DETACHED download can still be
			// running after the tier movers stop — this drain keeps it from
			// outliving process teardown (its adopt into an already-closed
			// volume is refused and discarded; see nvme.AdoptSegment).
			cleanup.push(rst.Close)
			spillB, restoreB = sp, rst
			fmt.Fprintln(os.Stderr, "kvblockd: s3 tier on", cfg.S3Bucket, "node", cfg.S3NodeID)
		}
		tiered := store.NewTiered(dstore, pol, vols, reports, recovered, store.Params{
			DemoteWatermarkPct: cfg.NvmeDemoteWatermarkPct,
			DemoteBatchPct:     cfg.NvmeDemoteBatchPct,
			AdmitMinHits:       cfg.NvmeAdmitMinHits,
			// 0 stays 0 = promotion disabled; the 60s default lives in the
			// config layer where the operator can see it.
			PromoteWindow:  time.Duration(cfg.NvmePromoteWindowMS) * time.Millisecond,
			LeaseDefaultMS: cfg.LeaseDefaultMS,
			LeaseMaxMS:     cfg.LeaseMaxMS,
			Quotas:         quotas,
			Spill:          spillB,
			Restore:        restoreB,
			S3ReadTimeout:  time.Duration(cfg.S3ReadTimeoutMS) * time.Millisecond,
		})
		// The tier movers stop strictly before the spiller/volumes/store
		// close (LIFO): the demoter releases its arena holds first.
		cleanup.push(tiered.Start(ctx))
		srvStore, statsFn, closeStore = tiered, tiered.Stats, tiered.Close
		fmt.Fprintln(os.Stderr, "kvblockd: nvme tier on", len(vols), "volume(s),",
			cfg.NvmeMaxBytes, "bytes budget, demote at", cfg.NvmeDemoteWatermarkPct, "%")
	}

	srv := server.New(cfg, srvStore, ns)

	set := metrics.New(statsFn)
	set.SetTenants(ns.Registry(), quotas)
	srv.SetRecorder(set)

	// Admin surface (loopback-enforced): namespace add / quota set / list.
	if cfg.AdminAddr != "" {
		admin := server.NewAdminServer(ns.Registry(), quotas, cfg.DramArenaBytes)
		aBound, aWait, aErr := admin.Serve(ctx, cfg.AdminAddr)
		if aErr != nil {
			return fmt.Errorf("admin endpoint: %w", aErr)
		}
		cleanup.push(func() { stop(); aWait() })
		fmt.Fprintln(os.Stderr, "kvblockd: admin on", aBound)
	}
	if cfg.MetricsAddr != "" {
		if host, _, herr := net.SplitHostPort(cfg.MetricsAddr); herr == nil {
			if ip := net.ParseIP(host); ip == nil || !ip.IsLoopback() {
				fmt.Fprintln(os.Stderr, "kvblockd: WARNING: metrics_addr", cfg.MetricsAddr,
					"is not loopback — /debug/pprof (heap, CPU, cmdline) is exposed unauthenticated on it")
			}
		}
		bound, wait, serr := set.Serve(ctx, cfg.MetricsAddr)
		if serr != nil {
			return fmt.Errorf("metrics endpoint: %w", serr)
		}
		// The step cancels the signal ctx BEFORE blocking on the ops
		// endpoint's shutdown, or an early error return (data port in use)
		// deadlocks in wait() with nothing ever cancelling ctx.
		cleanup.push(func() { stop(); wait() })
		fmt.Fprintln(os.Stderr, "kvblockd: metrics on", bound)
	}

	// S3-compat endpoint (s3compat_addr set): the NIXL obj / vLLM obj
	// zero-code path, on the SAME store and tenant table as the data plane.
	// Its handlers read the store, so its cleanup step must complete before
	// closeStore (LIFO guarantees it) — the drain-before-Close rule again.
	if cfg.S3CompatAddr != "" {
		if host, _, herr := net.SplitHostPort(cfg.S3CompatAddr); herr == nil {
			if ip := net.ParseIP(host); ip == nil || !ip.IsLoopback() {
				fmt.Fprintln(os.Stderr, "kvblockd: WARNING: s3compat_addr", cfg.S3CompatAddr,
					"is not loopback — tenant tokens (Authorization headers) and block bytes cross it in cleartext; keep it on a private network or terminate TLS in front")
			}
		}
		s3 := server.NewS3Compat(cfg, srvStore, ns)
		s3.SetRecorder(set) // obj-client traffic feeds the same meters as the wire plane
		bound, wait, serr := s3.Serve(ctx, cfg.S3CompatAddr)
		if serr != nil {
			return fmt.Errorf("s3compat endpoint: %w", serr)
		}
		// The s3compat server counts as one more reader population: an HTTP
		// handler outliving the shutdown grace (possibly mid-Get) gets the
		// same treatment as a failed Drain — skip the close, let process
		// exit reclaim, and exit degraded so monitoring sees it.
		cleanup.push(func() {
			stop()
			if !wait() {
				fmt.Fprintln(os.Stderr, "kvblockd: s3compat shutdown timed out — leaving the store open for process exit")
				skipStoreClose = true
				if err == nil {
					err = fmt.Errorf("%w: s3compat handlers outlived the shutdown grace", errDegradedShutdown)
				}
			}
		})
		fmt.Fprintln(os.Stderr, "kvblockd: s3compat on", bound)
	}

	if _, serr := srv.Start(ctx); serr != nil {
		return serr
	}
	set.SetReady() // arena prefaulted (NewArena) and listener accepting
	<-ctx.Done()
	// Deregister the signal handlers NOW: the ctx is already cancelled, so
	// the only thing continued registration buys is swallowing the
	// operator's second Ctrl-C during a hung shutdown — restore default
	// disposition so it kills the process.
	stop()
	fmt.Fprintln(os.Stderr, "kvblockd: draining...")
	drainCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if !srv.Drain(drainCtx) {
		// A writer may still hold arena views (a peer that stopped reading).
		// Unmapping now could send unrelated process memory to that peer —
		// skip the unmap; the process exit reclaims everything anyway (open
		// segment fds included; kill -9 is the recovery path's whole job).
		// The cleanup stack still stops the movers/evictor/listeners.
		fmt.Fprintln(os.Stderr, "kvblockd: drain timed out — leaving the arena mapped for process exit")
		skipStoreClose = true
		return fmt.Errorf("%w: data-plane drain timed out", errDegradedShutdown)
	}
	// Clean path: the deferred cleanup stack runs everything in reverse —
	// listeners (s3compat wait included), tier movers (the demoter releases
	// its arena holds), evictor, then close: volumes first (writer drain +
	// final sync), dram arena last.
	return nil
}
