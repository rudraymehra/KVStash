//go:build kvbdebug

package dram

import "testing"

// Debug builds: the same stale/sentinel Frees must fail LOUDLY.
func TestStaleFreePanicsUnderKvbdebug(t *testing.T) {
	a := NewAllocator(binExactCapacity(1 << 20))
	alA, ok := a.Alloc(4 << 10)
	if !ok {
		t.Fatal("alloc A failed")
	}
	a.Free(alA)
	if _, ok := a.Alloc(4 << 10); !ok { // recycles A's slot
		t.Fatal("alloc B failed")
	}
	defer func() {
		if recover() == nil {
			t.Fatal("stale Free did not panic under kvbdebug")
		}
	}()
	a.Free(alA)
}

func TestFailedAllocSentinelPanicsUnderKvbdebug(t *testing.T) {
	a := NewAllocator(binExactCapacity(1 << 20))
	bad, ok := a.Alloc(1 << 25)
	if ok {
		t.Fatal("oversize alloc succeeded")
	}
	defer func() {
		if recover() == nil {
			t.Fatal("sentinel Free did not panic under kvbdebug")
		}
	}()
	a.Free(bad)
}
