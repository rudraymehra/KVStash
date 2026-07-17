//go:build !kvbdebug

package dram

import "testing"

// Release builds: a STALE Free (slot recycled by a later Alloc) and a
// failed-Alloc sentinel Free are silent no-ops — the generation check rejects
// them without touching live memory (the ABA a ladder review confirmed as a
// HIGH before the generation counter existed).
// TestStaleFreeRejected pins the generation check: Alloc(A) → Free(A) →
// Alloc(B) recycles A's slot, so a STALE Free(A) must be rejected instead of
// silently freeing the live B (the ABA the C++ original does not defend
// against; a confirmed HIGH here before the generation counter).
func TestStaleFreeRejected(t *testing.T) {
	a := NewAllocator(binExactCapacity(1 << 20))
	alA, ok := a.Alloc(4 << 10)
	if !ok {
		t.Fatal("alloc A failed")
	}
	a.Free(alA)
	alB, ok := a.Alloc(4 << 10)
	if !ok {
		t.Fatal("alloc B failed")
	}
	if alB.Meta&metaSlotMask != alA.Meta&metaSlotMask {
		t.Skip("allocator did not recycle the slot — scenario not reproduced")
	}
	freeBefore := a.StorageReport().TotalFreeSpace
	a.Free(alA) // STALE handle: same slot, older generation → must be a no-op
	if got := a.StorageReport().TotalFreeSpace; got != freeBefore {
		t.Fatalf("stale Free released live memory: TotalFreeSpace %d -> %d", freeBefore, got)
	}
	// B must still be live and freeable exactly once.
	a.Free(alB)
	if got := a.StorageReport().TotalFreeSpace; got != binExactCapacity(1<<20) {
		t.Fatalf("after real free, TotalFreeSpace = %d", got)
	}
}

// TestFailedAllocSentinel pins that a failed Alloc's Allocation cannot free
// anything: its Meta is the unusedIndex sentinel, rejected by Free (a
// confirmed MED: the zero value used to alias the first live handle).
func TestFailedAllocSentinel(t *testing.T) {
	a := NewAllocator(binExactCapacity(1 << 20))
	live, ok := a.Alloc(100) // Meta slot 0 — what Allocation{} used to alias
	if !ok {
		t.Fatal("alloc failed")
	}
	_ = live
	bad, ok := a.Alloc(1 << 25) // too big → must fail
	if ok {
		t.Fatal("oversize alloc succeeded")
	}
	if bad.Meta != unusedIndex {
		t.Fatalf("failed Alloc Meta = %#x, want unusedIndex", bad.Meta)
	}
	freeBefore := a.StorageReport().TotalFreeSpace
	a.Free(bad) // must be a no-op
	if got := a.StorageReport().TotalFreeSpace; got != freeBefore {
		t.Fatalf("Free(failed-alloc sentinel) changed accounting: %d -> %d", freeBefore, got)
	}
}
