//go:build linux

package main

import (
	"errors"

	"golang.org/x/sys/unix"
)

// openIO opens the target with O_DIRECT (bypassing the page cache) on Linux.
func openIO(path string, forWrite bool, direct bool) (int, error) {
	flags := unix.O_RDONLY
	if forWrite {
		flags = unix.O_RDWR
	}
	if direct {
		flags |= unix.O_DIRECT
	}
	return unix.Open(path, flags, 0)
}

func preadFull(fd int, buf []byte, off int64) (int, error) {
	return unix.Pread(fd, buf, off)
}

func pwriteFull(fd int, buf []byte, off int64) (int, error) {
	return unix.Pwrite(fd, buf, off)
}

// runUring — DEFERRED per the pre-registered A3 decision rule.
//
// Attempted with github.com/pawelgaczynski/giouring (the SPEC-1 pick): its
// linkname directive into syscall.munmap is rejected by Go 1.26's rules
// ("invalid reference to syscall.munmap" at link time), and the library has
// been untouched since 2023. Rather than vendor-patching a dead dependency for
// a throwaway rig, we take the plan's written fallback: the threadpool engine
// is the A3 candidate (>=6 GB/s/device = PASS), and an io_uring engine (raw
// syscalls or a maintained binding) is the v1.1 spike alongside the Week-13
// io_uring deep dive. The product's NVMe tier keeps an IOBackend seam so both
// engines remain pluggable.
func runUring(probeConfig, int, int64) (int64, int64, int64, error) {
	return 0, 0, 0, errors.New("uring backend deferred: giouring is incompatible with Go 1.26 linkname rules; use --backend=threadpool (see comment)")
}
