package dram

import (
	"fmt"
	"log/slog"
	"sync/atomic"
	"time"
	"unsafe"

	"golang.org/x/sys/unix"
)

// Arena is a GC-invisible anonymous mmap region: blob bytes obtained straight
// from the kernel, bypassing the Go allocator, so they never appear in
// HeapAlloc and the GC never scans them (the A2-verified mechanism).
//
// LOAD-BEARING CONTRACT:
//
//   - Arena bytes cross API boundaries ONLY as (offset, len) + refcount.
//     Views are materialized on demand by Bytes and must never be stored in
//     heap structures or round-tripped through uintptr (go-learning-track
//     gotcha #8: uintptr conversions are only legal within one expression).
//   - The caller must drain all outstanding Bytes views (tier refcount == 0)
//     BEFORE Close; Close does not wait. The tier's refcount gate (Day 5)
//     makes munmap-under-reader structurally impossible — until then the
//     atomically-niled base turns a post-Close stale use into a loud panic
//     (a use RACING Close is excluded only by the drain contract itself).
type Arena struct {
	region []byte // the full mapping; retained only for Munmap
	// base is the pointer Bytes derives views from, and the ATOMIC source of
	// truth for liveness: Close swaps it to nil before munmap, and Bytes
	// loads it exactly once — a racing stale Bytes observes either the old
	// mapping (still mapped at that point) or nil (loud panic). A plain field
	// here was a confirmed -race TOCTOU: Bytes could pass a closed check,
	// then read a niled base and fabricate a wild slice at raw address off.
	base   atomic.Pointer[byte]
	size   uint64
	huge   bool // what the kernel actually granted, not what was requested
	closed atomic.Bool
}

// NewArena maps an anonymous region of at least `bytes` bytes. huge requests
// MAP_HUGETLB on Linux (2 MiB pages) and degrades gracefully — with a warning
// log — to normal pages when hugepages are unavailable; on other platforms
// huge is a documented no-op. The region is pre-faulted (MADV_POPULATE_WRITE
// on Linux ≥5.14, else a touch-per-page loop) so first-request latency never
// pays the fault storm; readiness is logged explicitly.
func NewArena(bytes int64, huge bool) (*Arena, error) {
	if bytes <= 0 {
		return nil, fmt.Errorf("dram: arena size %d: must be positive", bytes)
	}
	start := time.Now()
	region, gotHuge, err := mmapRegion(bytes, huge)
	if err != nil {
		return nil, fmt.Errorf("dram: mmap %d bytes: %w", bytes, err)
	}
	if huge && !gotHuge {
		slog.Warn("dram: hugepages requested but unavailable — fell back to normal pages",
			"bytes", bytes)
	}

	// Pre-fault: commit every page now. Madvise(POPULATE_WRITE) does it in
	// one syscall on Linux ≥5.14; where the ADVICE is unsupported we fall
	// back to touching one byte per page — but a genuine population failure
	// (ENOMEM: the pages don't exist to commit) is a hard error, not a
	// fallback (the fallback loop would just fault the process instead).
	populated, perr := advisePopulate(region)
	if perr != nil {
		_ = unix.Munmap(region)
		return nil, fmt.Errorf("dram: prefault %d bytes: %w", len(region), perr)
	}
	if !populated {
		stride := unix.Getpagesize()
		if gotHuge {
			stride = 2 << 20
		}
		for i := 0; i < len(region); i += stride {
			region[i] = 0
		}
	}

	a := &Arena{
		region: region,
		size:   uint64(len(region)),
		huge:   gotHuge,
	}
	a.base.Store(&region[0])
	slog.Info("arena ready",
		"bytes", a.size, "hugepages", a.huge, "prefault", time.Since(start))
	return a, nil
}

// Bytes materializes a [off, off+n) view of arena memory. It is the package's
// single unsafe seam: bounds are checked (overflow-safely) BEFORE any pointer
// arithmetic, and the base pointer is loaded ATOMICALLY, exactly once — after
// Close has swapped it to nil, a contract-violating stale call panics loudly
// instead of fabricating a wild slice. (A call racing Close may still observe
// the pre-swap pointer; the drain-before-Close contract is what excludes that
// window, and the Day-5 tier refcount gate enforces it mechanically.)
// Panics on out-of-range views or a closed arena: both are tier bugs, and a
// loud panic beats a silent wild pointer.
func (a *Arena) Bytes(off uint64, n uint32) []byte {
	base := a.base.Load() // single atomic load: the liveness check AND the view source
	if base == nil {
		panic("dram: Bytes on closed arena")
	}
	if off > a.size || uint64(n) > a.size-off { // overflow-safe form
		panic(fmt.Sprintf("dram: Bytes(%d, %d) out of arena [0, %d)", off, n, a.size))
	}
	return unsafe.Slice((*byte)(unsafe.Add(unsafe.Pointer(base), uintptr(off))), int(n))
}

// Size returns the mapped length in bytes (may exceed the requested size by
// hugepage rounding).
func (a *Arena) Size() uint64 { return a.size }

// Huge reports whether the mapping actually uses hugepages.
func (a *Arena) Huge() bool { return a.huge }

// Close unmaps the region. Idempotent for its outcome; note that a LOSING
// concurrent Close returns nil possibly before the winner's munmap completes
// (callers needing "unmapped by return" must serialize Close themselves —
// the daemon has exactly one shutdown path, so this is documentation, not a
// footgun). The caller must have drained all outstanding Bytes views first
// (see the type contract) — Close does not wait for readers.
func (a *Arena) Close() error {
	if !a.closed.CompareAndSwap(false, true) {
		return nil
	}
	a.base.Store(nil) // stale Bytes now panics (atomic; see Bytes)
	region := a.region
	a.region = nil
	return unix.Munmap(region)
}
