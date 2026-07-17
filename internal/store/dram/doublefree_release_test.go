//go:build !kvbdebug

package dram

import "testing"

// Release builds: an invalid/double Free is a silent no-op — never a panic,
// never an accounting change (the "no panic paths" contract).
func TestDoubleFreeNoOpInRelease(t *testing.T) {
	a := NewAllocator(binExactCapacity(1 << 20))
	al, ok := a.Alloc(1 << 12)
	if !ok {
		t.Fatal("alloc failed")
	}
	a.Free(al)
	free1 := a.StorageReport().TotalFreeSpace
	a.Free(al) // no-op
	if got := a.StorageReport().TotalFreeSpace; got != free1 {
		t.Fatalf("double free changed accounting: %d -> %d", free1, got)
	}
}
