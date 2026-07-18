//go:build linux

package nvme

import (
	"errors"
	"fmt"
	"log/slog"

	"golang.org/x/sys/unix"
)

// directBackend is the Linux engine: O_DIRECT pread/pwrite on a raw fd —
// the measured A3 default (98%+ of the device fio ceiling at <0.25 cores).
// O_DIRECT demands 4 KiB-aligned buffers/offsets/lengths; the aligned pool
// and record padding provide that. On filesystems that reject O_DIRECT
// (tmpfs, some overlayfs) we fall back to buffered I/O with a warning —
// correctness is identical, only the page-cache bypass is lost.
type directBackend struct{}

type directFile struct {
	fd int
}

func (directBackend) Open(path string, forWrite bool) (File, error) {
	flags := unix.O_RDONLY
	if forWrite {
		flags = unix.O_RDWR | unix.O_CREAT
	}
	fd, err := unix.Open(path, flags|unix.O_DIRECT, 0o600) // tenant KV-cache payloads: never world-readable (ladder finding)
	if errors.Is(err, unix.EINVAL) {
		// Filesystem refuses O_DIRECT. Buffered fallback, logged once per open.
		fd, err = unix.Open(path, flags, 0o600) // tenant KV-cache payloads: never world-readable (ladder finding)
		if err == nil {
			slog.Warn("nvme: filesystem rejected O_DIRECT — buffered fallback (correctness unchanged)", "path", path)
		}
	}
	if err != nil {
		return nil, fmt.Errorf("nvme: open %s: %w", path, err)
	}
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
	if err := unix.Fdatasync(f.fd); err != nil {
		return fmt.Errorf("nvme: fdatasync: %w", err)
	}
	return nil
}

func (f *directFile) Preallocate(size int64) error {
	err := unix.Fallocate(f.fd, 0, 0, size)
	if err == nil {
		return nil
	}
	if errors.Is(err, unix.EOPNOTSUPP) || errors.Is(err, unix.ENOTSUP) || errors.Is(err, unix.EINVAL) {
		// tmpfs and friends: size the file with truncate instead (sparse —
		// fine for tests; real volumes sit on XFS/ext4 where fallocate works).
		if terr := unix.Ftruncate(f.fd, size); terr != nil {
			return fmt.Errorf("nvme: ftruncate fallback to %d: %w", size, terr)
		}
		return nil
	}
	return fmt.Errorf("nvme: fallocate to %d: %w", size, err)
}

func (f *directFile) Close() error {
	if err := unix.Close(f.fd); err != nil {
		return fmt.Errorf("nvme: close: %w", err)
	}
	return nil
}
