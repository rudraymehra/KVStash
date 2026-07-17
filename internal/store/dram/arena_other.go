//go:build !linux

package dram

import "golang.org/x/sys/unix"

// mmapRegion on non-Linux platforms: plain anonymous private mapping.
// Hugepages are a Linux concept here — the flag is a documented no-op
// (gotHuge always false), matching the spec's darwin stub.
func mmapRegion(bytes int64, _ bool) (region []byte, gotHuge bool, err error) {
	region, err = unix.Mmap(-1, 0, int(bytes),
		unix.PROT_READ|unix.PROT_WRITE, unix.MAP_ANON|unix.MAP_PRIVATE)
	return region, false, err
}

// advisePopulate: no MADV_POPULATE_WRITE off Linux — the caller's touch loop
// does the pre-fault.
func advisePopulate([]byte) (bool, error) { return false, nil }
