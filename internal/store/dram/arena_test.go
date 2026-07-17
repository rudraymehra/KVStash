package dram

import (
	"runtime"
	"strings"
	"testing"
)

// TestArenaGCInvisibility is the A2 proof in test form: pushing 1 GiB of
// 2 MiB blobs through arena+allocator must leave the Go heap essentially
// untouched (HeapAlloc delta < 10 MB) because the blob bytes live in the mmap
// region the GC never sees.
func TestArenaGCInvisibility(t *testing.T) {
	total := int64(1 << 30)
	if testing.Short() {
		total = 256 << 20
	}
	a, err := NewArena(total, false)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()

	alloc := NewAllocator(uint32(total)) //nolint:gosec // G115: total <= 1 GiB, fits uint32; unit == byte is legal below 4 GiB

	runtime.GC()
	var m0 runtime.MemStats
	runtime.ReadMemStats(&m0)

	const blob = 2 << 20
	pattern := byte(0x5C)
	var handles []Allocation
	for {
		al, ok := alloc.Alloc(blob)
		if !ok {
			break
		}
		seg := a.Bytes(uint64(al.Offset), blob)
		for i := range seg {
			seg[i] = pattern
		}
		handles = append(handles, al)
	}
	if len(handles) < int(total/blob)-1 {
		t.Fatalf("only %d blobs allocated of ~%d expected", len(handles), total/blob)
	}

	runtime.GC()
	var m1 runtime.MemStats
	runtime.ReadMemStats(&m1)

	delta := int64(m1.HeapAlloc) - int64(m0.HeapAlloc) //nolint:gosec // G115: HeapAlloc ≪ 2^62 in this test
	if delta > 10<<20 {
		t.Fatalf("HeapAlloc grew %d bytes (>10MB) — blob bytes are leaking onto the Go heap", delta)
	}
	// Spot-check the bytes actually landed.
	last := handles[len(handles)-1]
	if got := a.Bytes(uint64(last.Offset), blob)[blob-1]; got != pattern {
		t.Fatalf("arena byte = %#x, want %#x", got, pattern)
	}
	t.Logf("GC-invisibility: %d blobs (%d MiB) written, HeapAlloc delta %d bytes",
		len(handles), int64(len(handles))*blob>>20, delta)
}

// TestArenaHugeFallback: requesting hugepages must be functional on EVERY
// outcome — real hugepages on a provisioned box, silent-but-logged fallback on
// hugepage-less CI and non-Linux. Huge() must tell the truth either way.
func TestArenaHugeFallback(t *testing.T) {
	a, err := NewArena(64<<20, true)
	if err != nil {
		t.Fatalf("NewArena(huge=true) must never fail outright: %v", err)
	}
	defer a.Close()
	if runtime.GOOS != "linux" && a.Huge() {
		t.Fatal("Huge() true off Linux — the flag must be a no-op")
	}
	seg := a.Bytes(0, 4096)
	seg[0], seg[4095] = 0xAB, 0xCD
	if seg[0] != 0xAB || seg[4095] != 0xCD {
		t.Fatal("arena not writable after hugepage request")
	}
	t.Logf("hugepages requested: got=%v (functional either way)", a.Huge())
}

// TestArenaBytesBounds: every out-of-range view panics BEFORE pointer math,
// with the offsets in the message; a closed arena panics too.
func TestArenaBytesBounds(t *testing.T) {
	a, err := NewArena(1<<20, false)
	if err != nil {
		t.Fatal(err)
	}

	mustPanic := func(name string, contains string, f func()) {
		t.Run(name, func(t *testing.T) {
			defer func() {
				r := recover()
				if r == nil {
					t.Fatal("no panic")
				}
				if s, ok := r.(string); ok && contains != "" && !strings.Contains(s, contains) {
					t.Fatalf("panic %q does not mention %q", s, contains)
				}
			}()
			f()
		})
	}

	mustPanic("off-at-size", "out of arena", func() { a.Bytes(a.Size(), 1) })
	mustPanic("len-overruns", "out of arena", func() { a.Bytes(a.Size()-1, 2) })
	mustPanic("off-overflow", "out of arena", func() { a.Bytes(^uint64(0), 1) })

	// In-bounds edge is fine.
	if got := len(a.Bytes(a.Size()-1, 1)); got != 1 {
		t.Fatalf("edge view len %d", got)
	}
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
	if err := a.Close(); err != nil { // idempotent
		t.Fatalf("second Close: %v", err)
	}
	mustPanic("closed", "closed arena", func() { a.Bytes(0, 1) })
}

// TestArenaUnitsRoundTrip exercises the REAL tier-boundary conversion: an
// allocator in AllocUnit granules over an arena, every allocation's unit
// offset converted to bytes for Arena.Bytes, written and read back — the
// arithmetic the Day-5 tier will live on (replaces a tautological
// arithmetic-identity test a ladder reviewer rejected).
func TestArenaUnitsRoundTrip(t *testing.T) {
	const arenaBytes = int64(64 << 20)
	a, err := NewArena(arenaBytes, false)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()

	units := uint32(a.Size() >> AllocUnitShift) //nolint:gosec // G115: 64 MiB >> 12 = 2^14
	alloc := NewAllocator(units)

	type rec struct {
		al  Allocation
		n   uint32 // bytes
		tag byte
	}
	var recs []rec
	sizes := []uint32{400 << 10, 1 << 20, 2560 << 10}
	for i := 0; ; i++ {
		n := sizes[i%len(sizes)]
		nUnits := (n + AllocUnit - 1) >> AllocUnitShift
		al, ok := alloc.Alloc(nUnits)
		if !ok {
			break
		}
		byteOff := uint64(al.Offset) << AllocUnitShift
		seg := a.Bytes(byteOff, n)
		tag := byte(i + 1)
		for j := range seg {
			seg[j] = tag
		}
		recs = append(recs, rec{al: al, n: n, tag: tag})
	}
	if len(recs) < 20 {
		t.Fatalf("only %d unit-mode allocations", len(recs))
	}
	// Read back through the SAME conversion; any unit-math slip corrupts.
	for _, r := range recs {
		byteOff := uint64(r.al.Offset) << AllocUnitShift
		seg := a.Bytes(byteOff, r.n)
		if seg[0] != r.tag || seg[r.n-1] != r.tag {
			t.Fatalf("unit round-trip corrupted: offsetUnits=%d tag=%d got %d/%d",
				r.al.Offset, r.tag, seg[0], seg[r.n-1])
		}
	}
}

// TestArenaBytesCloseRace pins the TOCTOU fix (a confirmed HIGH): concurrent
// Bytes vs Close must be -race-clean on the base pointer, and every Bytes
// outcome must be either a valid view or the loud closed-arena panic — never
// a silent wild slice fabricated from a torn base (the W3 TOCTOU). The base
// atomic is the detector: if it were a plain field, -race flags the Bytes
// read against the Close write; nil-on-close turns a post-Close call into a
// panic, not a wild slice.
//
// The racing goroutine CONSTRUCTS views but never dereferences one that a
// concurrent Close may have munmap'd — a write/read through an unmapped page
// is a `fatal error: fault`, uncatchable by recover(), so touching the view
// mid-munmap would make this test crash-flaky (it did, intermittently, on
// CI). Production forbids the shape entirely via drain-before-Close;
// munmap-under-a-live-view being structurally impossible is a future arena
// refcount change (see arena.go doc), not something this test can safely
// exercise.
func TestArenaBytesCloseRace(t *testing.T) {
	for i := 0; i < 50; i++ {
		a, err := NewArena(1<<20, false)
		if err != nil {
			t.Fatal(err)
		}
		done := make(chan struct{})
		go func() {
			defer close(done)
			defer func() { _ = recover() }() // post-Close call panics — the correct loud outcome
			for j := 0; j < 100; j++ {
				// Construct the view (the atomic base load -race checks); do
				// NOT deref it — a concurrent munmap would fault fatally.
				_ = a.Bytes(0, 16)
			}
		}()
		_ = a.Close()
		<-done
		// After Close returns, Bytes is deterministically the loud panic.
		func() {
			defer func() {
				if recover() == nil {
					t.Fatal("Bytes on a closed arena did not panic")
				}
			}()
			_ = a.Bytes(0, 16)
		}()
	}
}
