// Command kvblockd is the single-binary KV-cache block store daemon: config →
// arena-backed DRAM tier → server. The NVMe and S3 tiers stack underneath the
// DRAM tier later.
package main

import (
	"context"
	"flag"
	"fmt"
	"net"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/kvstash/kvblockd/internal/config"
	"github.com/kvstash/kvblockd/internal/eviction"
	"github.com/kvstash/kvblockd/internal/metrics"
	"github.com/kvstash/kvblockd/internal/server"
	"github.com/kvstash/kvblockd/internal/store/dram"
)

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
	flag.Parse()

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

	arena, err := dram.NewArena(cfg.DramArenaBytes, cfg.DramHugepages)
	if err != nil {
		return fmt.Errorf("dram arena: %w", err)
	}
	store := dram.New(arena, dram.Params{
		LeaseDefaultMS: cfg.LeaseDefaultMS,
		LeaseMaxMS:     cfg.LeaseMaxMS,
		PinnedBytesCap: cfg.PinnedBytesCap,
	})

	srv := server.New(cfg, store, ns)

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	// Eviction: policy attach + the watermark goroutine. stopEvict must run
	// before EVERY store.Close (and on the drain-timeout path too) so no
	// eviction free races the arena unmap; it is safe to call repeatedly.
	stopEvict := func() {}
	if cfg.EvictionPolicy != "none" {
		ghost := cfg.EvictionGhostEntries
		if ghost == 0 {
			// Auto ceiling: one fingerprint per conceivable resident block
			// (arena / 64 KiB) — the policy itself stays arena-ignorant.
			ghost = int(cfg.DramArenaBytes >> 16)
		}
		pol, perr := eviction.New(cfg.EvictionPolicy, ghost)
		if perr != nil {
			_ = store.Close()
			return perr
		}
		store.AttachPolicy(pol)
		stopOnce := store.StartEvictor(ctx, dram.EvictorConfig{
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

	set := metrics.New(store.Stats)
	srv.SetRecorder(set)
	if cfg.MetricsAddr != "" {
		if host, _, herr := net.SplitHostPort(cfg.MetricsAddr); herr == nil {
			if ip := net.ParseIP(host); ip == nil || !ip.IsLoopback() {
				fmt.Fprintln(os.Stderr, "kvblockd: WARNING: metrics_addr", cfg.MetricsAddr,
					"is not loopback — /debug/pprof (heap, CPU, cmdline) is exposed unauthenticated on it")
			}
		}
		bound, wait, err := set.Serve(ctx, cfg.MetricsAddr)
		if err != nil {
			stopEvict()
			_ = store.Close()
			return fmt.Errorf("metrics endpoint: %w", err)
		}
		// Defers run LIFO: cancel the signal ctx BEFORE blocking on the ops
		// endpoint's shutdown, or an early error return (data port in use)
		// deadlocks in wait() with nothing ever cancelling ctx.
		defer func() { stop(); wait() }()
		fmt.Fprintln(os.Stderr, "kvblockd: metrics on", bound)
	}

	if _, err := srv.Start(ctx); err != nil {
		stopEvict()
		_ = store.Close()
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
		// skip the unmap; the process exit reclaims everything anyway.
		fmt.Fprintln(os.Stderr, "kvblockd: drain timed out — leaving the arena mapped for process exit")
		stopEvict()
		return nil
	}
	// Strictly AFTER a SUCCESSFUL Drain (and a stopped evictor): every
	// conn's writer has flushed and no eviction pass is mid-free, so the
	// arena unmaps with no live views.
	stopEvict()
	return store.Close()
}
