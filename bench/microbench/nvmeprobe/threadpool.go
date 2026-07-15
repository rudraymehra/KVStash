package main

import (
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

// runThreadpool is the portable engine: qd OS-thread-pinned workers doing
// synchronous pread/pwrite on 4096-aligned buffers. This is the known-viable
// fallback the A3 gate accepts if io_uring misses; PegaFlow-class numbers
// (~6+ GB/s/device) are achievable this way on modern NVMe.
func runThreadpool(cfg probeConfig, fd int, size int64) (bytesTotal, ops, ioErrs int64) {
	var b, n, e atomic.Int64
	stop := time.Now().Add(cfg.duration)

	var wg sync.WaitGroup
	for w := 0; w < cfg.qd; w++ {
		wg.Add(1)
		go func(seed uint64) {
			defer wg.Done()
			// Pin: synchronous disk I/O parks OS threads; pinning keeps the
			// scheduler from migrating and gives steadier tail behavior.
			runtime.LockOSThread()
			defer runtime.UnlockOSThread()

			buf := alignedBuf(cfg.bs)
			offs := newOffsets(seed, size, cfg.bs)
			for time.Now().Before(stop) {
				off := offs.next()
				var nn int
				var werr error
				if cfg.op == "read" {
					nn, werr = preadFull(fd, buf, off)
				} else {
					nn, werr = pwriteFull(fd, buf, off)
				}
				if werr != nil {
					e.Add(1)
					continue
				}
				b.Add(int64(nn))
				n.Add(1)
			}
		}(uint64(w) + 1)
	}
	wg.Wait()
	return b.Load(), n.Load(), e.Load()
}
