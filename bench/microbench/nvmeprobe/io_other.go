//go:build !linux

package main

import (
	"errors"

	"golang.org/x/sys/unix"
)

// openIO opens the target on non-Linux (the macOS dev box). There is no
// O_DIRECT on darwin; F_NOCACHE after open is the closest equivalent (skips
// the unified buffer cache for subsequent I/O).
func openIO(path string, forWrite bool, direct bool) (int, error) {
	flags := unix.O_RDONLY
	if forWrite {
		flags = unix.O_RDWR
	}
	fd, err := unix.Open(path, flags, 0)
	if err != nil {
		return -1, err
	}
	if direct {
		// Best-effort: darwin's F_NOCACHE. If it fails, keep going — this
		// platform is only the dev smoke test, never the A3 gate.
		_, _ = unix.FcntlInt(uintptr(fd), unix.F_NOCACHE, 1)
	}
	return fd, nil
}

func preadFull(fd int, buf []byte, off int64) (int, error) {
	return unix.Pread(fd, buf, off)
}

func pwriteFull(fd int, buf []byte, off int64) (int, error) {
	return unix.Pwrite(fd, buf, off)
}

// runUring: io_uring is Linux-only. The A3 gate runs on Linux; on the dev box
// only the threadpool backend is available.
func runUring(probeConfig, int, int64) (int64, int64, int64, error) {
	return 0, 0, 0, errors.New("uring backend is linux-only; use --backend=threadpool on this platform")
}
