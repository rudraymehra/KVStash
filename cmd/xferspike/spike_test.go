package main

import (
	"context"
	"net"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// TestClientServerLoopback exercises the real concurrency the rig ships — the
// accept loop, per-conn drain goroutines, both WaitGroups, and the writev blast
// path — under `go test -race`, which the unit tests alone never touch.
func TestClientServerLoopback(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()

	var srvBytes, srvFrames atomic.Int64
	var srvWG sync.WaitGroup
	accepted := make(chan struct{})
	go func() {
		close(accepted)
		for {
			conn, aerr := ln.Accept()
			if aerr != nil {
				return
			}
			srvWG.Add(1)
			go func() {
				defer srvWG.Done()
				drain(conn, 64<<20, &srvBytes, &srvFrames)
			}()
		}
	}()
	<-accepted

	cfg := config{
		addr:       ln.Addr().String(),
		streams:    4,
		frameBytes: 64 << 10, // 64 KiB — small & fast for a race test
		duration:   150 * time.Millisecond,
		noDelay:    true,
		maxFrame:   64 << 20,
	}

	ctx, cancel := context.WithTimeout(context.Background(), cfg.duration)
	defer cancel()
	var cliBytes, cliFrames atomic.Int64
	var cliWG sync.WaitGroup
	for i := 0; i < cfg.streams; i++ {
		cliWG.Add(1)
		go func() {
			defer cliWG.Done()
			_ = blast(ctx, cfg, &cliBytes, &cliFrames)
		}()
	}
	cliWG.Wait()
	_ = ln.Close()
	srvWG.Wait()

	if cliFrames.Load() == 0 {
		t.Fatal("client sent no frames")
	}
	// The server may not have drained the final in-flight frame when the client
	// closed, so allow the server count to be <= client (never more).
	if srvBytes.Load() > cliBytes.Load() {
		t.Fatalf("server received %d > client sent %d", srvBytes.Load(), cliBytes.Load())
	}
	if cliBytes.Load()-srvBytes.Load() > int64(cfg.frameBytes+headerSize) {
		t.Fatalf("server/client byte gap %d exceeds one frame", cliBytes.Load()-srvBytes.Load())
	}
}
