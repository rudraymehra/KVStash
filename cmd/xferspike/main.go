// Command xferspike is the A1 transport measurement rig. It blasts
// fixed-size frames over N parallel TCP connections and reports aggregate
// throughput (GB/s), frame rate, and CPU cores consumed as one JSON line — the
// evidence behind the A1 kill-gate (>=12 GB/s loopback, >=85% of the iperf3
// ceiling on the cloud pair). The --mode=soak variant additionally backs the A2
// kill-gate: it serves blobs from an off-heap mmap arena and reports GC-pause
// percentiles, proving a large cache doesn't cause GC stalls. It is a throwaway
// rig kept forever for reproducibility; it is NOT the product.
package main

import (
	"flag"
	"fmt"
	"os"
	"time"
)

type config struct {
	mode        string
	addr        string
	streams     int
	frameBytes  int
	duration    time.Duration
	sndBuf      int
	rcvBuf      int
	noDelay     bool
	maxFrame    int
	arenaBytes  int
	soakStreams int
}

func main() {
	var cfg config
	flag.StringVar(&cfg.mode, "mode", "", "server | client")
	flag.StringVar(&cfg.addr, "addr", "127.0.0.1:9999", "host:port to serve/dial")
	flag.IntVar(&cfg.streams, "streams", 1, "parallel connections (client mode)")
	flag.IntVar(&cfg.frameBytes, "frame-bytes", 1<<20, "payload bytes per frame (client mode)")
	flag.DurationVar(&cfg.duration, "duration", 5*time.Second, "blast duration (client mode)")
	flag.IntVar(&cfg.sndBuf, "sndbuf", 0, "SO_SNDBUF bytes, 0 = OS default")
	flag.IntVar(&cfg.rcvBuf, "rcvbuf", 0, "SO_RCVBUF bytes, 0 = OS default")
	flag.BoolVar(&cfg.noDelay, "nodelay", true, "set TCP_NODELAY")
	flag.IntVar(&cfg.maxFrame, "max-frame", 64<<20, "server: reject frames larger than this (anti-OOM)")
	flag.IntVar(&cfg.arenaBytes, "arena-bytes", 4<<30, "soak: off-heap arena size in bytes")
	flag.IntVar(&cfg.soakStreams, "soak-streams", 8, "soak: concurrent client goroutines")
	flag.Parse()

	var err error
	switch cfg.mode {
	case "server":
		err = runServer(cfg)
	case "client":
		err = runClient(cfg)
	case "soak":
		err = runSoak(cfg)
	default:
		fmt.Fprintln(os.Stderr, "usage: xferspike --mode=server|client|soak [flags]")
		flag.PrintDefaults()
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "xferspike:", err)
		os.Exit(1)
	}
}
