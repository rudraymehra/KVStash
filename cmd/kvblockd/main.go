// Command kvblockd is the single-binary KV-cache block store daemon: config →
// arena-backed DRAM tier (→ log-structured NVMe tier when nvme_paths is set)
// → server. The S3 tier stacks underneath later.
package main

import (
	"context"
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

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "kvblockd:", err)
		os.Exit(1)
	}
}

func run() error {
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
		Quotas:         quotas,
	})

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	// Eviction: policy attach + the watermark goroutine. stopEvict must run
	// before EVERY store close (and on the drain-timeout path too) so no
	// eviction free races the arena unmap; it is safe to call repeatedly.
	stopEvict := func() {}
	var pol eviction.Policy
	if cfg.EvictionPolicy != "none" {
		ghost := cfg.EvictionGhostEntries
		if ghost == 0 {
			// Auto ceiling: one fingerprint per conceivable resident block
			// (arena / 64 KiB) — the policy itself stays arena-ignorant.
			ghost = int(cfg.DramArenaBytes >> 16)
		}
		var perr error
		pol, perr = eviction.New(cfg.EvictionPolicy, ghost)
		if perr != nil {
			_ = dstore.Close()
			return perr
		}
		dstore.AttachPolicy(pol)
		stopOnce := dstore.StartEvictor(ctx, dram.EvictorConfig{
			WatermarkPct: cfg.EvictionWatermarkPct,
			BatchPct:     cfg.EvictionBatchPct,
		})
		var stopped bool
		stopEvict = func() {
			if !stopped {
				stopped = true
				stopOnce()
			}
		}
		defer stopEvict()
		fmt.Fprintln(os.Stderr, "kvblockd: eviction policy", pol.Name(),
			"watermark", cfg.EvictionWatermarkPct, "batch", cfg.EvictionBatchPct)
	}

	// NVMe tier (nvme_paths set): open every volume — recovery (checkpoint +
	// footer scan + tail truncation) runs inside OpenVolume — then stack the
	// tiered orchestrator on top. The server sees ONE store either way.
	var srvStore server.Store = dstore
	statsFn := dstore.Stats
	closeStore := dstore.Close
	stopTier := func() {}
	if len(cfg.NvmePaths) > 0 {
		vols := make([]*nvme.Volume, 0, len(cfg.NvmePaths))
		reports := make([]*nvme.RecoveryReport, 0, len(cfg.NvmePaths))
		recovered := make([][]nvme.RecoveredEntry, 0, len(cfg.NvmePaths))
		perVol := cfg.NvmeMaxBytes / int64(len(cfg.NvmePaths))
		closeVols := func() {
			for _, v := range vols {
				_ = v.Close()
			}
		}
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
				closeVols()
				stopEvict()
				_ = dstore.Close()
				return fmt.Errorf("nvme volume %s: %w", dir, verr)
			}
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
				closeVols()
				stopEvict()
				_ = dstore.Close()
				return fmt.Errorf("s3 tier: %w", aerr)
			}
			sp := s3spill.NewSpiller(api, s3cfg, cfg.S3SpillQueue)
			defer sp.Close()
			spillB, restoreB = sp, s3spill.NewRestorer(api, s3cfg)
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
		stopT := tiered.Start(ctx)
		var tierStopped bool
		stopTier = func() {
			if !tierStopped {
				tierStopped = true
				stopT()
			}
		}
		defer stopTier()
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
		admin := server.NewAdminServer(ns.Registry(), quotas)
		aBound, aWait, aErr := admin.Serve(ctx, cfg.AdminAddr)
		if aErr != nil {
			stopTier()
			stopEvict()
			_ = closeStore()
			return fmt.Errorf("admin endpoint: %w", aErr)
		}
		defer func() { stop(); aWait() }()
		fmt.Fprintln(os.Stderr, "kvblockd: admin on", aBound)
	}
	if cfg.MetricsAddr != "" {
		if host, _, herr := net.SplitHostPort(cfg.MetricsAddr); herr == nil {
			if ip := net.ParseIP(host); ip == nil || !ip.IsLoopback() {
				fmt.Fprintln(os.Stderr, "kvblockd: WARNING: metrics_addr", cfg.MetricsAddr,
					"is not loopback — /debug/pprof (heap, CPU, cmdline) is exposed unauthenticated on it")
			}
		}
		bound, wait, err := set.Serve(ctx, cfg.MetricsAddr)
		if err != nil {
			stopTier()
			stopEvict()
			_ = closeStore()
			return fmt.Errorf("metrics endpoint: %w", err)
		}
		// Defers run LIFO: cancel the signal ctx BEFORE blocking on the ops
		// endpoint's shutdown, or an early error return (data port in use)
		// deadlocks in wait() with nothing ever cancelling ctx.
		defer func() { stop(); wait() }()
		fmt.Fprintln(os.Stderr, "kvblockd: metrics on", bound)
	}

	if _, err := srv.Start(ctx); err != nil {
		stopTier()
		stopEvict()
		_ = closeStore()
		return err
	}
	set.SetReady() // arena prefaulted (NewArena) and listener accepting
	<-ctx.Done()
	fmt.Fprintln(os.Stderr, "kvblockd: draining...")
	drainCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if !srv.Drain(drainCtx) {
		// A writer may still hold arena views (a peer that stopped reading).
		// Unmapping now could send unrelated process memory to that peer —
		// skip the unmap; the process exit reclaims everything anyway (open
		// segment fds included; kill -9 is the recovery path's whole job).
		fmt.Fprintln(os.Stderr, "kvblockd: drain timed out — leaving the arena mapped for process exit")
		stopTier()
		stopEvict()
		return nil
	}
	// Strictly AFTER a SUCCESSFUL Drain: stop the tier movers (the demoter
	// releases its arena holds), then the evictor, then close — volumes
	// first (writer drain + final sync), dram arena last.
	stopTier()
	stopEvict()
	return closeStore()
}
