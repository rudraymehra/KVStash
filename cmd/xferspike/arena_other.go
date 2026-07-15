//go:build !linux

package main

import "golang.org/x/sys/unix"

// rssMul converts unix.Rusage.Maxrss to bytes. On darwin/BSD ru_maxrss is
// already in bytes, so the multiplier is 1.
const rssMul = 1

// mmapArena maps an anonymous private RW region of `bytes`. Non-Linux platforms
// (macOS dev box) have no MAP_HUGETLB, so only ordinary pages are used.
func mmapArena(bytes int) ([]byte, func() error, bool, error) {
	region, err := unix.Mmap(-1, 0, bytes,
		unix.PROT_READ|unix.PROT_WRITE,
		unix.MAP_ANON|unix.MAP_PRIVATE)
	if err != nil {
		return nil, nil, false, err
	}
	return region, func() error { return unix.Munmap(region) }, false, nil
}
