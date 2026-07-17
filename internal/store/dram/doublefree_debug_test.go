//go:build kvbdebug

package dram

import "testing"

// Debug builds: a double Free must fail LOUDLY (assertf panic), so a
// use-after-free in the tier can't rot silently.
func TestDoubleFreePanicsUnderKvbdebug(t *testing.T) {
	a := NewAllocator(binExactCapacity(1 << 20))
	al, ok := a.Alloc(1 << 12)
	if !ok {
		t.Fatal("alloc failed")
	}
	a.Free(al)
	defer func() {
		if recover() == nil {
			t.Fatal("double Free did not panic under kvbdebug")
		}
	}()
	a.Free(al)
}
