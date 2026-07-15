// Command nvmeprobe is the A3 kill-gate rig: can Go drive an NVMe device at
// >=6 GB/s? The active engine is a pinned-thread pread/pwrite pool with
// O_DIRECT (the pre-registered fallback the gate accepts); the io_uring engine
// is deferred — see io_linux.go — pending a Go-1.26-compatible binding. Reports
// honest decimal GB/s + IOPS + CPU as one JSON line. Gate rule: >=6 GB/s per
// device passes; "undecided" is not an outcome. Throwaway rig; not the product.
package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"time"

	"golang.org/x/sys/unix"
)

type probeConfig struct {
	backend  string
	path     string
	op       string
	bs       int
	qd       int
	duration time.Duration
	fileSize int64
	direct   bool
}

// probeResult is the one JSON line. Decimal GB/s (bytes/1e9/s) to match fio and
// the transport rig's units.
type probeResult struct {
	Backend    string  `json:"backend"`
	Op         string  `json:"op"`
	BlockBytes int     `json:"block_bytes"`
	QueueDepth int     `json:"queue_depth"`
	Direct     bool    `json:"direct"`
	DurationS  float64 `json:"duration_s"`
	BytesTotal int64   `json:"bytes_total"`
	GBytesPerS float64 `json:"gbytes_per_s"`
	IOPS       float64 `json:"iops"`
	Ops        int64   `json:"ops"`
	IOErrors   int64   `json:"io_errors"`
	CPUCores   float64 `json:"cpu_cores"`
	Note       string  `json:"note,omitempty"`
}

func (r probeResult) print() {
	if err := json.NewEncoder(os.Stdout).Encode(r); err != nil {
		fmt.Fprintln(os.Stderr, "nvmeprobe: encode result:", err)
	}
}

func cpuTime() time.Duration {
	var ru unix.Rusage
	if err := unix.Getrusage(unix.RUSAGE_SELF, &ru); err != nil {
		return 0
	}
	return time.Duration(ru.Utime.Nano()) + time.Duration(ru.Stime.Nano())
}

func main() {
	var cfg probeConfig
	flag.StringVar(&cfg.backend, "backend", "threadpool", "threadpool | uring (uring is linux-only)")
	flag.StringVar(&cfg.path, "path", "", "file or block device to probe (REQUIRED; a file is created/filled if needed)")
	flag.StringVar(&cfg.op, "op", "read", "read | write")
	flag.IntVar(&cfg.bs, "bs", 1<<20, "I/O block size in bytes (128k=131072, 1m=1048576)")
	flag.IntVar(&cfg.qd, "qd", 32, "queue depth (in-flight I/Os / worker threads)")
	flag.DurationVar(&cfg.duration, "duration", 10*time.Second, "measurement window")
	flag.Int64Var(&cfg.fileSize, "file-size", 4<<30, "test file size (ignored for block devices)")
	flag.BoolVar(&cfg.direct, "direct", true, "bypass the page cache (O_DIRECT on linux, F_NOCACHE on darwin)")
	flag.Parse()

	if err := run(cfg); err != nil {
		fmt.Fprintln(os.Stderr, "nvmeprobe:", err)
		os.Exit(1)
	}
}

func run(cfg probeConfig) error {
	if cfg.path == "" {
		return errors.New("--path is required")
	}
	if cfg.op != "read" && cfg.op != "write" {
		return fmt.Errorf("op %q must be read or write", cfg.op)
	}
	if cfg.bs < 4096 || cfg.bs%4096 != 0 {
		return fmt.Errorf("bs %d must be a multiple of 4096 (O_DIRECT alignment)", cfg.bs)
	}
	if cfg.qd < 1 || cfg.qd > 1024 {
		return errors.New("qd must be in [1,1024]")
	}

	fd, size, err := openTarget(cfg)
	if err != nil {
		return err
	}
	defer unix.Close(fd)
	if size < int64(cfg.bs) {
		return fmt.Errorf("target size %d smaller than one block %d", size, cfg.bs)
	}

	cpuStart := cpuTime()
	wallStart := time.Now()

	var bytesTotal, ops, ioErrs int64
	switch cfg.backend {
	case "threadpool":
		bytesTotal, ops, ioErrs = runThreadpool(cfg, fd, size)
	case "uring":
		bytesTotal, ops, ioErrs, err = runUring(cfg, fd, size)
	default:
		return fmt.Errorf("backend %q must be threadpool or uring", cfg.backend)
	}
	if err != nil {
		return err
	}

	wall := time.Since(wallStart)
	cpuUsed := cpuTime() - cpuStart

	res := probeResult{
		Backend:    cfg.backend,
		Op:         cfg.op,
		BlockBytes: cfg.bs,
		QueueDepth: cfg.qd,
		Direct:     cfg.direct,
		DurationS:  wall.Seconds(),
		BytesTotal: bytesTotal,
		Ops:        ops,
		IOErrors:   ioErrs,
	}
	if s := wall.Seconds(); s > 0 {
		res.GBytesPerS = float64(bytesTotal) / 1e9 / s
		res.IOPS = float64(ops) / s
		res.CPUCores = cpuUsed.Seconds() / s
	}
	if ioErrs > 0 {
		res.Note = "io_errors > 0: treat the GB/s as suspect"
	}
	res.print()
	return nil
}

// alignedBuf returns a bs-sized slice whose base address is 4096-aligned, as
// O_DIRECT requires.
func alignedBuf(bs int) []byte {
	raw := make([]byte, bs+4096)
	off := 0
	if a := align4096(raw); a != 0 {
		off = int(a)
	}
	return raw[off : off+bs]
}
