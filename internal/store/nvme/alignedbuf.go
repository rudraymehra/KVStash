package nvme

import (
	"fmt"
	"sync/atomic"

	"golang.org/x/sys/unix"
)

// bufPool hands out page-aligned buffers for direct I/O. Buffers are
// anonymous mmap allocations (page alignment ≥ the 4 KiB O_DIRECT
// requirement) — plain []byte, no unsafe, GC-invisible like the arena.
// Three size classes cover the read shapes: small metadata reads, typical
// blocks, and the max record span; each class retains a bounded free list
// and overflow is munmapped on Put, so the pool is bounded-by-demand even
// while the transport holds released-late buffers.
//
// The retain bound must follow the HOLDER's lifetime, not the producer's
// parallelism: buffers stay out until the transport's writev completes, so
// once outstanding responses exceed the bound EVERY read pays an mmap on
// Get plus a munmap on Put — cross-core TLB-shootdown IPIs plus first-touch
// page faults, a throughput cliff at exactly peak load. The counters make
// that cliff visible on the scrape (allocs = pool misses, overflowMunmaps =
// returns the free list could not hold).
type bufPool struct {
	classes []uint32      // ascending buffer sizes, page-multiples
	free    []chan []byte // one bounded free list per class

	allocs         atomic.Uint64 // fresh mmaps (free list empty on Get)
	overflowMunmap atomic.Uint64 // Puts munmapped because the free list was full
}

// newBufPool sizes the largest class to maxSpan (rounded to recordAlign)
// and retains up to `retain` free buffers per class.
func newBufPool(maxSpan uint32, retain int) *bufPool {
	if retain < 1 {
		retain = 1
	}
	classes := []uint32{128 << 10, 1 << 20}
	maxClass := uint32(roundUpAlign(uint64(maxSpan))) //nolint:gosec // G115: maxSpan ≤ recordSpan(MaxBlobLen) < 4 GiB by config validation
	for len(classes) > 0 && classes[len(classes)-1] >= maxClass {
		classes = classes[:len(classes)-1]
	}
	classes = append(classes, maxClass)
	p := &bufPool{classes: classes, free: make([]chan []byte, len(classes))}
	for i := range p.free {
		p.free[i] = make(chan []byte, retain)
	}
	return p
}

// Get returns an aligned buffer with len ≥ n (the full class size — callers
// subslice for I/O but must Put back the original).
func (p *bufPool) Get(n uint32) ([]byte, error) {
	for i, c := range p.classes {
		if n > c {
			continue
		}
		select {
		case b := <-p.free[i]:
			return b, nil
		default:
			p.allocs.Add(1)
			return mmapBuf(int(c))
		}
	}
	return nil, fmt.Errorf("nvme: buffer request %d exceeds max class %d", n, p.classes[len(p.classes)-1])
}

// Put returns a buffer to its class free list, munmapping on overflow or
// unknown size (never blocks, never leaks).
func (p *bufPool) Put(b []byte) {
	if b == nil {
		return
	}
	for i, c := range p.classes {
		if uint32(cap(b)) != c { //nolint:gosec // G115: class sizes are < 4 GiB
			continue
		}
		select {
		case p.free[i] <- b[:cap(b)]:
		default:
			p.overflowMunmap.Add(1)
			_ = unix.Munmap(b[:cap(b)])
		}
		return
	}
	_ = unix.Munmap(b[:cap(b)])
}

// statsInto merges the pool's counters into the volume stats document.
func (p *bufPool) statsInto(m map[string]int64) {
	m["pool_alloc_total"] += int64(p.allocs.Load())                   //nolint:gosec // G115: counters
	m["pool_overflow_munmap_total"] += int64(p.overflowMunmap.Load()) //nolint:gosec // G115: counters
}

// Close munmaps every buffer currently retained in the free lists. The
// volume calls it only after its readers have drained (close ordering); a
// straggler Put after Close still lands safely in the (now-empty) list or
// munmaps on overflow. There is no outstanding-buffer counter — leak
// detection here is the goleak TestMain plus the bounded lists.
func (p *bufPool) Close() {
	for _, ch := range p.free {
	drain:
		for {
			select {
			case b := <-ch:
				_ = unix.Munmap(b)
			default:
				break drain
			}
		}
	}
}

func mmapBuf(n int) ([]byte, error) {
	b, err := unix.Mmap(-1, 0, n, unix.PROT_READ|unix.PROT_WRITE, unix.MAP_ANON|unix.MAP_PRIVATE)
	if err != nil {
		return nil, fmt.Errorf("nvme: mmap %d-byte buffer: %w", n, err)
	}
	return b, nil
}

// alignedTemp is a one-shot page-aligned buffer for metadata I/O on direct
// fds (seal tables, trailers, scan headers). Plain `make` slices happen to
// be page-aligned for 4096-multiples under today's allocator, but the
// language guarantees nothing — every metadata path used to gamble on it.
// free() must be called exactly once.
func alignedTemp(n int) (buf []byte, free func(), err error) {
	b, err := mmapBuf(int(roundUpAlign(uint64(n)))) //nolint:gosec // G115: metadata sizes ≪ 4 GiB
	if err != nil {
		return nil, nil, err
	}
	return b[:n], func() { _ = unix.Munmap(b) }, nil
}
