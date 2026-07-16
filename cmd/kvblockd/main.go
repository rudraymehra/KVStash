// Command kvblockd is the single-binary KV-cache block store daemon. This
// week it wires config → ramstub store → server for end-to-end bring-up; the
// real DRAM/NVMe/S3 tiers replace ramstub later.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/kvstash/kvblockd/internal/config"
	"github.com/kvstash/kvblockd/internal/server"
	"github.com/kvstash/kvblockd/internal/store/ramstub"
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

	srv := server.New(cfg, ramstub.New(), ns)

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	if _, err := srv.Start(ctx); err != nil {
		return err
	}
	<-ctx.Done()
	fmt.Fprintln(os.Stderr, "kvblockd: draining...")
	drainCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	srv.Drain(drainCtx)
	return nil
}
