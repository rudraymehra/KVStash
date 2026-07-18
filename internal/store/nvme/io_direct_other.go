//go:build !linux

package nvme

import (
	"errors"
	"fmt"

	"golang.org/x/sys/unix"
)

// directBackend on non-Linux (the macOS dev box) is a CORRECTNESS backend:
// F_NOCACHE approximates O_DIRECT (no alignment enforcement, partial
// caching) and darwin fsync does not flush the drive cache (F_FULLFSYNC is
// deliberately not used — this platform is never quoted for performance or
// durability; see doc.go). Same alignment discipline is kept so the exact
// Linux I/O shapes are exercised.
type directBackend struct{}

type directFile struct {
	fd int
}

func (directBackend) Open(path string, forWrite bool) (File, error) {
	flags := unix.O_RDONLY
	if forWrite {
		flags = unix.O_RDWR | unix.O_CREAT
	}
	fd, err := unix.Open(path, flags, 0o600) // tenant KV-cache payloads: never world-readable (ladder finding)
	if err != nil {
		return nil, fmt.Errorf("nvme: open %s: %w", path, err)
	}
	// Best-effort page-cache bypass; failure is fine on the dev box.
	_, _ = unix.FcntlInt(uintptr(fd), unix.F_NOCACHE, 1)
	return &directFile{fd: fd}, nil
}

func (f *directFile) ReadAt(p []byte, off int64) error {
	for len(p) > 0 {
		n, err := unix.Pread(f.fd, p, off)
		if err != nil {
			if errors.Is(err, unix.EINTR) {
				continue
			}
			return fmt.Errorf("nvme: pread at %d: %w", off, err)
		}
		if n == 0 {
			return fmt.Errorf("nvme: pread at %d: unexpected EOF", off)
		}
		p = p[n:]
		off += int64(n)
	}
	return nil
}

func (f *directFile) WriteAt(p []byte, off int64) error {
	for len(p) > 0 {
		n, err := unix.Pwrite(f.fd, p, off)
		if err != nil {
			if errors.Is(err, unix.EINTR) {
				continue
			}
			return fmt.Errorf("nvme: pwrite at %d: %w", off, err)
		}
		p = p[n:]
		off += int64(n)
	}
	return nil
}

func (f *directFile) Datasync() error {
	if err := unix.Fsync(f.fd); err != nil {
		return fmt.Errorf("nvme: fsync: %w", err)
	}
	return nil
}

func (f *directFile) Preallocate(size int64) error {
	// No fallocate on darwin; F_PREALLOCATE is best-effort and fiddly.
	// Truncate sizes the file (sparse) — correctness backend only.
	if err := unix.Ftruncate(f.fd, size); err != nil {
		return fmt.Errorf("nvme: ftruncate to %d: %w", size, err)
	}
	return nil
}

func (f *directFile) Close() error {
	if err := unix.Close(f.fd); err != nil {
		return fmt.Errorf("nvme: close: %w", err)
	}
	return nil
}
