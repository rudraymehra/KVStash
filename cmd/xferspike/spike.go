package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"os/signal"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"golang.org/x/sys/unix"
)

// result is the single JSON line xferspike prints on exit. Units are explicit
// in the field names: GBytesPerS is decimal gigaBYTES/s (bytes/1e9/s), so the
// A1 "12 GB/s" target = 96 Gbit/s when compared against iperf3's Gbit/s.
// CPUCoresSender counts ONLY this process (the client) — the server's CPU is
// not included, so this is not a whole-system efficiency figure.
type result struct {
	Mode           string  `json:"mode"`
	Streams        int     `json:"streams"`
	FrameBytes     int     `json:"frame_bytes"`
	DurationS      float64 `json:"duration_s"`
	BytesTotal     int64   `json:"bytes_total"`
	Frames         int64   `json:"frames"`
	GBytesPerS     float64 `json:"gbytes_per_s"`
	GbitPerS       float64 `json:"gbit_per_s"`
	FramesPerS     float64 `json:"frames_per_s"`
	CPUCoresSender float64 `json:"cpu_cores_sender"`
	Note           string  `json:"note,omitempty"`
}

func (r result) print() {
	if err := json.NewEncoder(os.Stdout).Encode(r); err != nil {
		fmt.Fprintln(os.Stderr, "xferspike: encode result:", err)
	}
}

// tune applies the socket options that matter for throughput. SetNoDelay,
// SetReadBuffer and SetWriteBuffer are portable across linux and darwin, so no
// raw setsockopt is needed here.
func tune(conn net.Conn, cfg config) {
	tc, ok := conn.(*net.TCPConn)
	if !ok {
		return
	}
	if cfg.noDelay {
		_ = tc.SetNoDelay(true)
	}
	if cfg.sndBuf > 0 {
		_ = tc.SetWriteBuffer(cfg.sndBuf)
	}
	if cfg.rcvBuf > 0 {
		_ = tc.SetReadBuffer(cfg.rcvBuf)
	}
}

// cpuTime returns total (user+system) CPU time consumed by this process so far.
// The delta across a run, divided by wall time, gives cores consumed.
func cpuTime() time.Duration {
	var ru unix.Rusage
	if err := unix.Getrusage(unix.RUSAGE_SELF, &ru); err != nil {
		return 0
	}
	return time.Duration(ru.Utime.Nano()) + time.Duration(ru.Stime.Nano())
}

// runServer accepts connections and drains frames, counting received bytes.
// It runs until SIGINT/SIGTERM, then prints a result line.
func runServer(cfg config) error {
	ln, err := net.Listen("tcp", cfg.addr)
	if err != nil {
		return fmt.Errorf("listen %s: %w", cfg.addr, err)
	}
	fmt.Fprintf(os.Stderr, "xferspike server listening on %s (Ctrl-C to stop)\n", ln.Addr())

	var bytesTotal, frames atomic.Int64
	var wg sync.WaitGroup

	go func() {
		for {
			conn, err := ln.Accept()
			if err != nil {
				return // listener closed
			}
			tune(conn, cfg)
			wg.Add(1)
			go func() {
				defer wg.Done()
				drain(conn, cfg.maxFrame, &bytesTotal, &frames)
			}()
		}
	}()

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	<-sig

	_ = ln.Close()

	// Give in-flight drain goroutines a bounded moment to settle so the totals
	// aren't a mid-flight snapshot; don't block forever on a stuck peer.
	done := make(chan struct{})
	go func() { wg.Wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(500 * time.Millisecond):
	}

	result{
		Mode:       "server",
		BytesTotal: bytesTotal.Load(),
		Frames:     frames.Load(),
		Note:       "server received totals (best-effort snapshot)",
	}.print()
	return nil
}

// drain reads frames from conn until it closes or errors, counting bytes. A
// frame whose declared length exceeds maxFrame is rejected to bound memory
// against a hostile or desynchronized peer.
func drain(conn net.Conn, maxFrame int, bytesTotal, frames *atomic.Int64) {
	defer conn.Close()
	hdr := make([]byte, headerSize)
	var buf []byte
	for {
		if _, err := io.ReadFull(conn, hdr); err != nil {
			return
		}
		h, err := decodeHeader(hdr)
		if err != nil {
			fmt.Fprintf(os.Stderr, "xferspike server: %v; dropping conn\n", err)
			return
		}
		if int(h.length) > maxFrame {
			fmt.Fprintf(os.Stderr, "xferspike server: frame length %d exceeds max-frame %d; dropping conn\n", h.length, maxFrame)
			return
		}
		if int(h.length) > cap(buf) {
			buf = make([]byte, h.length)
		}
		body := buf[:h.length]
		if _, err := io.ReadFull(conn, body); err != nil {
			return
		}
		bytesTotal.Add(int64(headerSize) + int64(h.length))
		frames.Add(1)
	}
}

// runClient opens cfg.streams connections, blasts frames on each for
// cfg.duration, and prints aggregate throughput.
func runClient(cfg config) error {
	if cfg.streams < 1 {
		return errors.New("streams must be >= 1")
	}
	if cfg.frameBytes < 1 {
		return errors.New("frame-bytes must be >= 1")
	}
	if cfg.frameBytes > cfg.maxFrame {
		return fmt.Errorf("frame-bytes %d exceeds max-frame %d; a server with the default max would drop the connection", cfg.frameBytes, cfg.maxFrame)
	}

	ctx, cancel := context.WithTimeout(context.Background(), cfg.duration)
	defer cancel()

	var bytesTotal, frames atomic.Int64
	var wg sync.WaitGroup
	errCh := make(chan error, cfg.streams)

	cpuStart := cpuTime()
	wallStart := time.Now()

	for i := 0; i < cfg.streams; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := blast(ctx, cfg, &bytesTotal, &frames); err != nil {
				errCh <- err
			}
		}()
	}
	wg.Wait()

	wall := time.Since(wallStart)
	cpuUsed := cpuTime() - cpuStart

	select {
	case err := <-errCh:
		return err
	default:
	}

	total := bytesTotal.Load()
	secs := wall.Seconds()
	res := result{
		Mode:       "client",
		Streams:    cfg.streams,
		FrameBytes: cfg.frameBytes,
		DurationS:  secs,
		BytesTotal: total,
		Frames:     frames.Load(),
	}
	if secs > 0 {
		res.GBytesPerS = float64(total) / 1e9 / secs
		res.GbitPerS = res.GBytesPerS * 8
		res.FramesPerS = float64(res.Frames) / secs
		res.CPUCoresSender = cpuUsed.Seconds() / secs
	}
	res.print()
	return nil
}

// blast dials the server and sends frames until ctx is done. The header and
// payload go to the kernel together via net.Buffers (writev), so the payload is
// never copied into a combined buffer.
func blast(ctx context.Context, cfg config, bytesTotal, frames *atomic.Int64) error {
	var d net.Dialer
	conn, err := d.DialContext(ctx, "tcp", cfg.addr)
	if err != nil {
		return fmt.Errorf("dial %s: %w", cfg.addr, err)
	}
	defer conn.Close()
	tune(conn, cfg)

	// A per-write deadline is the safety net: without it, a stalled or
	// non-reading peer would block WriteTo forever, hanging the run past
	// --duration and emitting no result (an invisible A1 failure). We set the
	// deadline to the run's end so a blocked write unblocks exactly when the
	// blast should stop.
	deadline, hasDeadline := ctx.Deadline()

	hdr := make([]byte, headerSize)
	payload := make([]byte, cfg.frameBytes)
	var seq uint64

	for ctx.Err() == nil {
		putHeader(hdr, frameHeader{magic: magic, seq: seq, length: uint32(cfg.frameBytes)})
		if hasDeadline {
			if err := conn.SetWriteDeadline(deadline); err != nil {
				return fmt.Errorf("set write deadline: %w", err)
			}
		}
		bufs := net.Buffers{hdr, payload}
		n, err := bufs.WriteTo(conn)
		bytesTotal.Add(n) // count whatever actually made it onto the wire
		if err != nil {
			// Deadline exceeded == the blast reached --duration: a clean stop.
			// Any OTHER error is a real transport failure and is returned
			// unconditionally (never swallowed as "must be shutdown").
			if errors.Is(err, os.ErrDeadlineExceeded) {
				return nil
			}
			return fmt.Errorf("write: %w", err)
		}
		frames.Add(1)
		seq++
	}
	return nil
}
