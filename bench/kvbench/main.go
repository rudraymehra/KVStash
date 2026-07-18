// Command kvbench is the storage-level benchmark harness (SPEC-4): grid
// sweeps and trace replays against kvblockd, Redis/Valkey, and an NVMe-fs
// floor, with coordinated-omission-safe open-loop latency, deterministic
// incompressible payloads, corruption-checked GETs, and one JSONL record
// per cell — the file every chart is rendered from.
//
// Subcommands: sweep | replay | fill | verify | convert | report.
// (getbench, the original single-cell GET driver, remains for the older
// BENCHMARKS.md repro lines; sweep supersedes it.)
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"
)

func main() {
	os.Exit(run())
}

// run keeps os.Exit out of the defer scope (the signal stop must fire).
func run() int {
	if len(os.Args) < 2 {
		usage()
		return 2
	}
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	var err error
	switch os.Args[1] {
	case "sweep":
		err = cmdSweep(ctx, os.Args[2:])
	case "fill":
		err = cmdFill(ctx, os.Args[2:])
	case "verify":
		err = cmdVerify(ctx, os.Args[2:])
	case "convert":
		err = cmdConvert(ctx, os.Args[2:])
	case "replay":
		err = cmdReplay(ctx, os.Args[2:])
	case "report":
		err = cmdReport(ctx, os.Args[2:])
	case "-h", "--help", "help":
		usage()
		return 0
	default:
		fmt.Fprintf(os.Stderr, "kvbench: unknown subcommand %q\n", os.Args[1])
		usage()
		return 2
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "kvbench:", err)
		return 1
	}
	return 0
}

func usage() {
	fmt.Fprint(os.Stderr, `kvbench <subcommand> [flags]

  sweep    run grid cells (closed-loop ceiling + CO-safe open-loop rate sweep)
  fill     seed a deterministic key pool (warm-state parity across stores)
  verify   re-derive and byte-compare stored blobs (corruption oracle)
  convert  turn Bailian/Mooncake traces into .kvops op streams
  replay   adaptively replay a .kvops trace (EXISTS→GET→PUT; hit rate is an OUTPUT)
  report   aggregate JSONL; --check-repeat enforces the 2%% repeatability gate

Every subcommand prints one JSON object per measurement (JSONL). Decimal
GB/s; payload-only goodput; quote ratios against the same-rig ceiling.
`)
}

// targetFlags is the shared --target/... surface.
type targetFlags struct {
	kind    string
	addr    string
	ns      string
	token   string
	streams int
	sockbuf int
	dir     string
	fsync   bool
	verify  bool
}

func (tf *targetFlags) register(fs *flag.FlagSet) {
	fs.StringVar(&tf.kind, "target", "kvblockd", "kvblockd | redis | valkey | nvmefs | mem")
	fs.StringVar(&tf.addr, "addr", "127.0.0.1:9440", "store address (kvblockd/redis/valkey)")
	fs.StringVar(&tf.ns, "ns", "bench", "kvblockd namespace")
	fs.StringVar(&tf.token, "token", "bench-token", "kvblockd bearer token")
	fs.IntVar(&tf.streams, "streams", 8, "connections / pool size / fs workers")
	fs.IntVar(&tf.sockbuf, "sockbuf", 16<<20, "kvblockd socket buffers (0 = OS default)")
	fs.StringVar(&tf.dir, "dir", "", "nvmefs block directory")
	fs.BoolVar(&tf.fsync, "fsync", false, "nvmefs: fdatasync each PUT (disclosed durable-floor variant)")
	fs.BoolVar(&tf.verify, "client-verify", true, "kvblockd client-side xxh3 verification (methodology rule 8; off isolates its cost)")
}
