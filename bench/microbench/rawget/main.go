// rawget: fair loopback ceiling for a GET-shaped round trip.
// Client sends a 1 KiB request; server writev's 32 distinct 1 MiB buffers
// (optionally in 1 MiB windows); client reads into 32 distinct 1 MiB buffers.
// No protocol, no hashing — the harness ceiling for kvblockd's BATCH_GET.
package main

import (
	"flag"
	"fmt"
	"net"
	"sync"
	"sync/atomic"
	"time"
)

func main() {
	streams := flag.Int("streams", 4, "concurrent connections")
	secs := flag.Int("secs", 4, "measure seconds")
	chunk := flag.Int("chunk", 1<<20, "writev window bytes (0=whole response)")
	hot := flag.Bool("hot", false, "server sends one hot buffer 32x")
	flag.Parse()

	const n = 32
	const sz = 1 << 20

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		panic(err)
	}
	go func() {
		for {
			c, err := ln.Accept()
			if err != nil {
				return
			}
			go func(c net.Conn) {
				defer c.Close()
				_ = c.(*net.TCPConn).SetNoDelay(true)
				// 32 distinct source blocks (or 1 hot one).
				blocks := make([][]byte, n)
				for i := range blocks {
					if *hot && i > 0 {
						blocks[i] = blocks[0]
						continue
					}
					b := make([]byte, sz)
					for j := range b {
						b[j] = byte(i)
					}
					blocks[i] = b
				}
				req := make([]byte, 1024)
				for {
					// read the 1 KiB "request"
					for off := 0; off < len(req); {
						m, err := c.Read(req[off:])
						if err != nil {
							return
						}
						off += m
					}
					// respond: 32 MiB as iovecs, chunked windows
					iovs := make(net.Buffers, n)
					copy(iovs, blocks)
					for start := 0; start < len(iovs); {
						end, bytes := start, 0
						for end < len(iovs) && (*chunk <= 0 || bytes < *chunk) {
							bytes += len(iovs[end])
							end++
						}
						w := iovs[start:end]
						if _, err := (&w).WriteTo(c); err != nil {
							return
						}
						start = end
					}
				}
			}(c)
		}
	}()

	var total atomic.Int64
	stop := make(chan struct{})
	var wg sync.WaitGroup
	for s := 0; s < *streams; s++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			c, err := net.Dial("tcp", ln.Addr().String())
			if err != nil {
				panic(err)
			}
			defer c.Close()
			_ = c.(*net.TCPConn).SetNoDelay(true)
			req := make([]byte, 1024)
			into := make([][]byte, n)
			for i := range into {
				into[i] = make([]byte, sz)
			}
			for {
				select {
				case <-stop:
					return
				default:
				}
				if _, err := c.Write(req); err != nil {
					return
				}
				for i := 0; i < n; i++ {
					for off := 0; off < sz; {
						m, err := c.Read(into[i][off:])
						if err != nil {
							return
						}
						off += m
					}
				}
				total.Add(n * sz)
			}
		}()
	}
	time.Sleep(time.Duration(*secs) * time.Second)
	close(stop)
	elapsed := float64(*secs)
	gb := float64(total.Load()) / 1e9
	fmt.Printf("streams=%d hot=%v chunk=%d: %.2f GB/s (%.1f GB in %.0fs)\n",
		*streams, *hot, *chunk, gb/elapsed, gb, elapsed)
	_ = ln.Close()
	wg.Wait()
}
