package dram

import (
	"math"
	"math/rand"
	"sort"
	"testing"

	"pgregory.net/rapid"
)

// ---------------------------------------------------------------------------
// 1. SmallFloat falsifier — proven against a brute-force reference BEFORE the
// allocator structure is trusted (the port's hardest bug class).

// refBinFloor returns floatToUint via independent arithmetic (no shared code
// with the implementation): the canonical value of bin f.
func refBinFloor(f uint32) uint64 {
	exp := f >> mantissaBits
	man := uint64(f & mantissaMask)
	if exp == 0 {
		return man
	}
	return (man | mantissaValue) << (exp - 1)
}

// refRoundUp/refRoundDown: linear scan over all 256 bins — the slow, obviously
// correct reference the fast implementation must match.
func refRoundDown(size uint32) uint32 {
	best := uint32(0)
	for b := uint32(0); b < numLeafBins; b++ {
		if refBinFloor(b) <= uint64(size) {
			best = b
		} else {
			break
		}
	}
	return best
}

func refRoundUp(size uint32) uint32 {
	for b := uint32(0); b < numLeafBins; b++ {
		if refBinFloor(b) >= uint64(size) {
			return b
		}
	}
	return numLeafBins // above the largest representable bin
}

func TestSmallFloatExhaustive(t *testing.T) {
	check := func(size uint32) {
		t.Helper()
		up, down := uintToFloatRoundUp(size), uintToFloatRoundDown(size)
		wantDown := refRoundDown(size)
		if down != wantDown {
			t.Fatalf("roundDown(%d) = %d, want %d", size, down, wantDown)
		}
		wantUp := refRoundUp(size)
		if uint64(up) != uint64(wantUp) {
			t.Fatalf("roundUp(%d) = %d, want %d", size, up, wantUp)
		}
		// The load-bearing pair property, directly:
		// floatToUint(roundUp(s)) >= s (when representable) and
		// floatToUint(roundDown(s)) <= s.
		if refBinFloor(down) > uint64(size) {
			t.Fatalf("roundDown(%d)=%d has floor %d > size", size, down, refBinFloor(down))
		}
		if up < numLeafBins && refBinFloor(up) < uint64(size) {
			t.Fatalf("roundUp(%d)=%d has floor %d < size", size, up, refBinFloor(up))
		}
	}
	// Dense low range: covers all denormals and the first exponents.
	for s := uint32(0); s <= 1<<17; s++ {
		check(s)
	}
	// Every bin edge ±1 across the full range.
	for b := uint32(0); b < numLeafBins; b++ {
		floor := refBinFloor(b)
		if floor > math.MaxUint32 {
			break
		}
		f := uint32(floor)
		check(f)
		if f > 0 {
			check(f - 1)
		}
		if f < math.MaxUint32 {
			check(f + 1)
		}
	}
	// Powers of two ±1 and the extremes.
	for i := 0; i < 32; i++ {
		p := uint32(1) << i
		check(p)
		check(p - 1)
		if p < math.MaxUint32 {
			check(p + 1)
		}
	}
	check(math.MaxUint32)
}

// ---------------------------------------------------------------------------
// 2/3. Table tests over the working band with fill-pattern overlap detection.

// harness pairs an allocator with a heap backing where each allocation writes
// a distinct pattern; verify detects any cross-write between live ranges.
type harness struct {
	t       *testing.T
	a       *Allocator
	backing []byte
	live    map[uint32]hrec // Meta -> record
	nextTag byte
}

type hrec struct {
	al   Allocation
	size uint32
	tag  byte
}

func newHarness(t *testing.T, capacity uint32) *harness {
	return &harness{
		t: t, a: NewAllocator(capacity),
		backing: make([]byte, capacity),
		live:    make(map[uint32]hrec),
	}
}

func (h *harness) alloc(n uint32) (Allocation, bool) {
	al, ok := h.a.Alloc(n)
	if !ok {
		return al, false
	}
	h.nextTag++
	tag := h.nextTag
	seg := h.backing[al.Offset : uint64(al.Offset)+uint64(n)]
	for i := range seg {
		seg[i] = 0xA5 ^ tag
	}
	h.live[al.Meta] = hrec{al: al, size: n, tag: tag}
	h.checkInvariants()
	return al, true
}

func (h *harness) free(meta uint32) {
	rec, ok := h.live[meta]
	if !ok {
		h.t.Fatalf("free of unknown meta %d", meta)
	}
	h.verifyPattern(rec)
	h.a.Free(rec.al)
	delete(h.live, meta)
	h.checkInvariants()
}

func (h *harness) verifyPattern(rec hrec) {
	h.t.Helper()
	seg := h.backing[rec.al.Offset : uint64(rec.al.Offset)+uint64(rec.size)]
	for i, b := range seg {
		if b != 0xA5^rec.tag {
			h.t.Fatalf("overlap: allocation at offset %d size %d corrupted at +%d (got %#x want %#x)",
				rec.al.Offset, rec.size, i, b, 0xA5^rec.tag)
		}
	}
}

// checkInvariants: live intervals disjoint & in-bounds; free-space accounting
// exact (the allocator is exact-fit, so TotalFreeSpace == cap − Σ live).
func (h *harness) checkInvariants() {
	h.t.Helper()
	var sum uint64
	type iv struct{ lo, hi uint64 }
	ivs := make([]iv, 0, len(h.live))
	for _, r := range h.live {
		lo := uint64(r.al.Offset)
		hi := lo + uint64(r.size)
		if hi > uint64(len(h.backing)) {
			h.t.Fatalf("allocation [%d,%d) exceeds capacity %d", lo, hi, len(h.backing))
		}
		ivs = append(ivs, iv{lo, hi})
		sum += uint64(r.size)
	}
	for i := range ivs {
		for j := i + 1; j < len(ivs); j++ {
			a, b := ivs[i], ivs[j]
			if a.lo < b.hi && b.lo < a.hi {
				h.t.Fatalf("overlapping live intervals [%d,%d) and [%d,%d)", a.lo, a.hi, b.lo, b.hi)
			}
		}
	}
	got := h.a.StorageReport().TotalFreeSpace
	want := uint64(len(h.backing)) - sum
	if uint64(got) != want {
		h.t.Fatalf("TotalFreeSpace = %d, want %d (cap %d − live %d)", got, want, len(h.backing), sum)
	}
}

// binExactCapacity rounds a capacity DOWN to a bin floor so free-all →
// Alloc(capacity) is assertable (initial insert uses roundDown; original
// behavior).
func binExactCapacity(c uint32) uint32 {
	return floatToUint(uintToFloatRoundDown(c))
}

// bandSizes: the 0.4–2.5 MB vLLM block band pinned at bin edges ±1.
func bandSizes() []uint32 {
	var sizes []uint32
	for _, base := range []uint32{400 << 10, 1 << 20, 2560 << 10} {
		bin := uintToFloatRoundDown(base)
		for _, b := range []uint32{bin - 1, bin, bin + 1} {
			f := floatToUint(b)
			sizes = append(sizes, f-1, f, f+1)
		}
	}
	return sizes
}

func TestAllocFreeRefillBand(t *testing.T) {
	const rawCap = 64 << 20
	capacity := binExactCapacity(rawCap)

	for _, order := range []string{"fifo", "lifo", "random"} {
		t.Run(order, func(t *testing.T) {
			h := newHarness(t, capacity)
			rng := rand.New(rand.NewSource(42)) //nolint:gosec // G404: deterministic test shuffle, not crypto

			fill := func() []uint32 {
				var metas []uint32
				sizes := bandSizes()
				i := 0
				for {
					al, ok := h.alloc(sizes[i%len(sizes)])
					if !ok {
						break
					}
					metas = append(metas, al.Meta)
					i++
				}
				return metas
			}
			round1 := fill()
			if len(round1) == 0 {
				t.Fatal("no allocations succeeded")
			}
			switch order {
			case "lifo":
				for i := len(round1) - 1; i >= 0; i-- {
					h.free(round1[i])
				}
			case "random":
				rng.Shuffle(len(round1), func(i, j int) { round1[i], round1[j] = round1[j], round1[i] })
				fallthrough
			default:
				for _, m := range round1 {
					h.free(m)
				}
			}
			// After freeing everything, coalescing must have restored the
			// single full-capacity node.
			if got := h.a.StorageReport().TotalFreeSpace; got != capacity {
				t.Fatalf("after free-all TotalFreeSpace = %d, want %d", got, capacity)
			}
			if _, ok := h.a.Alloc(capacity); !ok {
				t.Fatal("Alloc(capacity) failed after free-all — capacity permanently fragmented")
			}
			// Reset for the refill-identity check.
			h2 := newHarness(t, capacity)
			round2 := 0
			sizes := bandSizes()
			for {
				if _, ok := h2.alloc(sizes[round2%len(sizes)]); !ok {
					break
				}
				round2++
			}
			if round2 != len(round1) {
				t.Fatalf("refill count %d != first fill %d (capacity lost)", round2, len(round1))
			}
		})
	}
}

// ---------------------------------------------------------------------------
// 4. rapid property state machine.

func TestAllocatorRapid(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		capacity := binExactCapacity(rapid.Uint32Range(1<<16, 1<<26).Draw(rt, "cap"))
		a := NewAllocator(capacity)
		backing := make([]byte, capacity)
		type rec struct {
			al   Allocation
			size uint32
			tag  byte
		}
		live := map[uint32]rec{}
		tag := byte(0)

		sizeGen := rapid.OneOf(
			rapid.SampledFrom(bandSizes()),
			rapid.Uint32Range(1, 16),
			rapid.Uint32Range(1<<12, 1<<21),
		)

		verify := func(r rec) {
			seg := backing[r.al.Offset : uint64(r.al.Offset)+uint64(r.size)]
			for i, b := range seg {
				if b != 0xA5^r.tag {
					rt.Fatalf("overlap: [%d,+%d) corrupted at +%d", r.al.Offset, r.size, i)
				}
			}
		}
		checkInv := func() {
			var sum uint64
			type iv struct{ lo, hi uint64 }
			ivs := make([]iv, 0, len(live))
			for _, r := range live {
				lo, hi := uint64(r.al.Offset), uint64(r.al.Offset)+uint64(r.size)
				if hi > uint64(capacity) {
					rt.Fatalf("allocation out of bounds")
				}
				ivs = append(ivs, iv{lo, hi})
				sum += uint64(r.size)
			}
			// Pairwise disjointness via sort+scan: catches overlaps even when
			// the byte tag wraps at 256 live allocations.
			sort.Slice(ivs, func(i, j int) bool { return ivs[i].lo < ivs[j].lo })
			for i := 1; i < len(ivs); i++ {
				if ivs[i].lo < ivs[i-1].hi {
					rt.Fatalf("overlapping live intervals [%d,%d) and [%d,%d)",
						ivs[i-1].lo, ivs[i-1].hi, ivs[i].lo, ivs[i].hi)
				}
			}
			if got := a.StorageReport().TotalFreeSpace; uint64(got) != uint64(capacity)-sum {
				rt.Fatalf("TotalFreeSpace %d != cap %d − live %d", got, capacity, sum)
			}
		}

		rt.Repeat(map[string]func(*rapid.T){
			"alloc": func(rt *rapid.T) {
				n := sizeGen.Draw(rt, "n")
				al, ok := a.Alloc(n)
				if !ok {
					return // exhaustion is legal at any time
				}
				if _, dup := live[al.Meta]; dup {
					rt.Fatalf("meta %d handed out twice", al.Meta)
				}
				tag++
				seg := backing[al.Offset : uint64(al.Offset)+uint64(n)]
				for i := range seg {
					seg[i] = 0xA5 ^ tag
				}
				live[al.Meta] = rec{al: al, size: n, tag: tag}
			},
			"free": func(rt *rapid.T) {
				if len(live) == 0 {
					return
				}
				metas := make([]uint32, 0, len(live))
				for m := range live {
					metas = append(metas, m)
				}
				m := rapid.SampledFrom(metas).Draw(rt, "meta")
				verify(live[m])
				a.Free(live[m].al)
				delete(live, m)
			},
			"": func(*rapid.T) { checkInv() },
		})

		// Epilogue: free everything → full capacity must be re-allocatable in
		// one piece (the strongest no-permanent-loss statement).
		for m, r := range live {
			verify(r)
			a.Free(r.al)
			delete(live, m)
		}
		if got := a.StorageReport().TotalFreeSpace; got != capacity {
			rt.Fatalf("after free-all TotalFreeSpace = %d, want %d", got, capacity)
		}
		if _, ok := a.Alloc(capacity); !ok {
			rt.Fatalf("Alloc(full capacity %d) failed after free-all", capacity)
		}
	})
}

// ---------------------------------------------------------------------------
// 5. Exhaustion paths — clean false, never panic (release build).

func TestAllocExhaustionSpace(t *testing.T) {
	capacity := binExactCapacity(1 << 20)
	a := NewAllocator(capacity)
	n := 0
	for {
		if _, ok := a.Alloc(64 << 10); !ok {
			break
		}
		n++
	}
	if n == 0 {
		t.Fatal("nothing allocated before exhaustion")
	}
	if _, ok := a.Alloc(1); ok {
		// Some slack can remain from bin rounding; drain it fully.
		for {
			if _, ok := a.Alloc(1); !ok {
				break
			}
		}
	}
	if _, ok := a.Alloc(1); ok {
		t.Fatal("Alloc succeeded on a full allocator")
	}
}

func TestAllocExhaustionNodePool(t *testing.T) {
	a := newAllocator(1<<20, 4) // tiny node pool
	got := 0
	for {
		if _, ok := a.Alloc(4 << 10); !ok {
			break
		}
		got++
	}
	if got == 0 {
		t.Fatal("no allocations before node-pool exhaustion")
	}
	if _, ok := a.Alloc(4 << 10); ok {
		t.Fatal("Alloc succeeded with an exhausted node pool")
	}
}

func TestAllocZeroAndOversize(t *testing.T) {
	a := NewAllocator(binExactCapacity(1 << 20))
	if _, ok := a.Alloc(0); ok {
		t.Fatal("Alloc(0) succeeded")
	}
	if _, ok := a.Alloc(math.MaxUint32); ok {
		t.Fatal("Alloc(MaxUint32) succeeded on a 1 MiB allocator")
	}
}

// TestDoubleFree behavior differs by build: no-op in release, panic under
// -tags kvbdebug. Each branch is asserted by its tagged sibling file
// (doublefree_release_test.go / doublefree_debug_test.go).

// ---------------------------------------------------------------------------
// 6. Benchmarks — steady-state Alloc+Free pair; targets <200 ns, 0 allocs/op.

func benchAllocFree(b *testing.B, size uint32) {
	a := NewAllocator(binExactCapacity(1 << 30))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		al, ok := a.Alloc(size)
		if !ok {
			b.Fatal("alloc failed")
		}
		a.Free(al)
	}
}

func BenchmarkAlloc(b *testing.B) {
	b.Run("0.4MB", func(b *testing.B) { benchAllocFree(b, 400<<10) })
	b.Run("1MB", func(b *testing.B) { benchAllocFree(b, 1<<20) })
	b.Run("2.5MB", func(b *testing.B) { benchAllocFree(b, 2560<<10) })
}

// BenchmarkAllocParallel models the Day-5 tier: the allocator behind a mutex,
// contended from all procs (run with -cpu=1,16).
func BenchmarkAllocParallel(b *testing.B) {
	mu := make(chan struct{}, 1)
	mu <- struct{}{}
	a := NewAllocator(binExactCapacity(1 << 30))
	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			<-mu
			al, ok := a.Alloc(1 << 20)
			if ok {
				a.Free(al)
			}
			mu <- struct{}{}
		}
	})
}

// ---------------------------------------------------------------------------
// Ladder findings — regression pins.

// TestStaleFreeRejected / TestFailedAllocSentinel live in tagged siblings:
// release asserts the silent no-op, kvbdebug asserts the loud panic
// (stalefree_release_test.go / stalefree_debug_test.go).

// TestStorageReportPoolExhausted pins the C++ pool-exhaustion guard: when no
// node remains, the report must not advertise allocatable space.
func TestStorageReportPoolExhausted(t *testing.T) {
	a := newAllocator(1<<20, 2) // 4 node slots
	for {
		if _, ok := a.Alloc(4 << 10); !ok {
			break
		}
	}
	if a.freeOffset == 0 {
		r := a.StorageReport()
		if r.TotalFreeSpace != 0 || r.LargestFreeRegion != 0 {
			t.Fatalf("pool-exhausted report advertises space: %+v", r)
		}
	}
}

// TestAllocatorNonBinExactCapacity: production capacities are arbitrary
// (arena Size()>>unitShift), not bin-exact. Free-all must restore the FULL
// capacity, and the bin-exact floor of it must be allocatable in one piece.
func TestAllocatorNonBinExactCapacity(t *testing.T) {
	const capacity = uint32(64<<20) + 12345 // deliberately not bin-exact
	h := newHarness(t, capacity)
	var metas []uint32
	sizes := bandSizes()
	for i := 0; ; i++ {
		al, ok := h.alloc(sizes[i%len(sizes)])
		if !ok {
			break
		}
		metas = append(metas, al.Meta)
	}
	for _, m := range metas {
		h.free(m)
	}
	if got := h.a.StorageReport().TotalFreeSpace; got != capacity {
		t.Fatalf("free-all TotalFreeSpace = %d, want the full %d", got, capacity)
	}
	if _, ok := h.a.Alloc(binExactCapacity(capacity)); !ok {
		t.Fatal("bin-exact floor of a non-exact capacity not allocatable after free-all")
	}
}

// BenchmarkAllocChurn measures the realistic fragmented steady state the
// Alloc+Free pair benchmark cannot: prefill to ~70%, then free-one-random /
// alloc-one forever (a ladder finding: the pair benchmark only ever touches
// the pristine single-free-node fast path).
func BenchmarkAllocChurn(b *testing.B) {
	a := NewAllocator(binExactCapacity(1 << 30))
	sizes := []uint32{400 << 10, 1 << 20, 2560 << 10}
	var live []Allocation
	var total uint64
	for i := 0; total < (1<<30)*7/10; i++ {
		al, ok := a.Alloc(sizes[i%len(sizes)])
		if !ok {
			break
		}
		live = append(live, al)
		total += uint64(sizes[i%len(sizes)])
	}
	rng := rand.New(rand.NewSource(7)) //nolint:gosec // G404: deterministic bench churn
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		j := rng.Intn(len(live))
		a.Free(live[j])
		al, ok := a.Alloc(sizes[i%len(sizes)])
		if !ok {
			// Fragmented miss: put the freed size class back to keep churning.
			al, ok = a.Alloc(400 << 10)
			if !ok {
				b.Fatal("churn wedged")
			}
		}
		live[j] = al
	}
}
