//go:build linux

package main

import "golang.org/x/sys/unix"

// rssMul converts unix.Rusage.Maxrss to bytes. On Linux ru_maxrss is in
// kilobytes; getting this wrong makes RSS read 1024x too small.
const rssMul = 1024

const hugePageBytes = 2 << 20 // 2 MiB, the common Linux huge-page size

// mmapArena maps an anonymous private RW region of at least `bytes`. On Linux it
// first tries huge pages (MAP_HUGETLB), which need vm.nr_hugepages > 0; on any
// failure it falls back to ordinary 4 KiB pages. Returns the region, a munmap
// closure, whether huge pages were used, and any error.
func mmapArena(bytes int) ([]byte, func() error, bool, error) {
	prot := unix.PROT_READ | unix.PROT_WRITE

	// Try huge pages: length must be a multiple of the huge-page size.
	huge := roundUp(bytes, hugePageBytes)
	region, err := unix.Mmap(-1, 0, huge, prot, unix.MAP_ANON|unix.MAP_PRIVATE|unix.MAP_HUGETLB)
	if err == nil {
		return region, func() error { return unix.Munmap(region) }, true, nil
	}

	// Fall back to normal pages (expected when no huge pages are reserved).
	region, err = unix.Mmap(-1, 0, bytes, prot, unix.MAP_ANON|unix.MAP_PRIVATE)
	if err != nil {
		return nil, nil, false, err
	}
	return region, func() error { return unix.Munmap(region) }, false, nil
}

func roundUp(n, mult int) int {
	if mult <= 0 {
		return n
	}
	return ((n + mult - 1) / mult) * mult
}
