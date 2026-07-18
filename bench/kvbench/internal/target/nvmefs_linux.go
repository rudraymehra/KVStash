//go:build linux

package target

import (
	"errors"
	"sync/atomic"

	"golang.org/x/sys/unix"
)

// directFellBack records whether any open on the volume dropped O_DIRECT
// (tmpfs/overlay rejecting it) — the floor bar discloses this so a
// page-cached measurement is never labeled the NVMe device floor.
var directFellBack atomic.Bool

// Direct I/O block file access — the same O_DIRECT + aligned-buffer
// discipline as the daemon's NVMe tier (bench-local copy; internal/store/
// nvme is unexported). Blob sizes are 4 KiB multiples by the grid
// invariant, and pooled buffers come from mmap (page-aligned).

func writeBlock(path string, blob []byte, fsync bool) error {
	fd, err := unix.Open(path, unix.O_WRONLY|unix.O_CREAT|unix.O_DIRECT, 0o600)
	if errors.Is(err, unix.EINVAL) {
		directFellBack.Store(true)
		fd, err = unix.Open(path, unix.O_WRONLY|unix.O_CREAT, 0o600) // tmpfs fallback (disclosed in JSONL)
	}
	if err != nil {
		return err
	}
	buf, free, err := alignedCopy(blob)
	if err != nil {
		_ = unix.Close(fd)
		return err
	}
	defer free()
	off := 0
	for off < len(buf) {
		n, werr := unix.Pwrite(fd, buf[off:], int64(off))
		if werr != nil {
			_ = unix.Close(fd)
			return werr
		}
		off += n
	}
	if fsync {
		if err := unix.Fdatasync(fd); err != nil {
			_ = unix.Close(fd)
			return err
		}
	}
	return unix.Close(fd)
}

func readBlock(path string, dst *[]byte, blob int) (int, error) {
	fd, err := unix.Open(path, unix.O_RDONLY|unix.O_DIRECT, 0)
	if errors.Is(err, unix.EINVAL) {
		directFellBack.Store(true)
		fd, err = unix.Open(path, unix.O_RDONLY, 0)
	}
	if err != nil {
		return 0, wrapNotExist(err)
	}
	defer func() { _ = unix.Close(fd) }()
	buf, free, err := alignedTemp(blob)
	if err != nil {
		return 0, err
	}
	defer free()
	off := 0
	for off < blob {
		n, rerr := unix.Pread(fd, buf[off:blob], int64(off))
		if rerr != nil {
			return 0, rerr
		}
		if n == 0 {
			break
		}
		off += n
	}
	if len(*dst) < off {
		*dst = make([]byte, off)
	}
	copy((*dst)[:off], buf[:off])
	return off, nil
}

func alignedCopy(b []byte) ([]byte, func(), error) {
	buf, free, err := alignedTemp(len(b))
	if err != nil {
		return nil, nil, err
	}
	copy(buf, b)
	return buf, free, nil
}

func alignedTemp(n int) ([]byte, func(), error) {
	sz := (n + 4095) &^ 4095
	if sz == 0 {
		sz = 4096
	}
	b, err := unix.Mmap(-1, 0, sz, unix.PROT_READ|unix.PROT_WRITE, unix.MAP_ANON|unix.MAP_PRIVATE)
	if err != nil {
		return nil, nil, err
	}
	return b[:n], func() { _ = unix.Munmap(b) }, nil
}

func wrapNotExist(err error) error {
	if errors.Is(err, unix.ENOENT) {
		return &notExistError{err}
	}
	return err
}

type notExistError struct{ err error }

func (e *notExistError) Error() string { return e.err.Error() }
func (e *notExistError) Unwrap() error { return e.err }

// Is makes errors.Is(err, fs.ErrNotExist) work on the wrapped unix error
// (os.IsNotExist does NOT — it never unwraps custom types; callers must
// use errors.Is).
func (e *notExistError) Is(target error) bool { return errors.Is(e.err, target) }
