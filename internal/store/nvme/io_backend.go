package nvme

import (
	"fmt"
	"os"
)

// IOBackend opens segment/checkpoint files for the tier. It exists so the
// I/O engine stays swappable (the A3 seam: threadpool today, io_uring behind
// kvb_uring later) and so tests can inject fault/spy backends — the
// EXISTS-never-touches-NVMe guard test panics inside a spy File.
type IOBackend interface {
	// Open opens (creating if forWrite) the file at path. Write handles are
	// opened for direct I/O where the platform supports it; callers must
	// then keep buffers, offsets, and lengths 4 KiB-aligned.
	Open(path string, forWrite bool) (File, error)
}

// File is the narrow surface the tier needs. ReadAt/WriteAt are full-length
// or error (short I/O is retried internally); both must be called with
// 4 KiB-aligned p/off on direct-I/O handles.
type File interface {
	ReadAt(p []byte, off int64) error
	WriteAt(p []byte, off int64) error
	// Datasync flushes file data to the device (fdatasync on Linux; plain
	// fsync on darwin — which does NOT flush the drive cache; see doc.go).
	Datasync() error
	// Preallocate makes [0, size) durable-sized (fallocate on Linux with a
	// truncate fallback; truncate elsewhere).
	Preallocate(size int64) error
	Close() error
}

// DefaultBackend returns the platform's measured-default engine: O_DIRECT
// pread/pwrite on Linux, F_NOCACHE on darwin (correctness only).
func DefaultBackend() IOBackend { return directBackend{} }

// SyncDir fsyncs a directory so a create/rename inside it is durable.
func SyncDir(dir string) error {
	d, err := os.Open(dir) //nolint:gosec // G304: dir is the operator-configured nvme_paths volume directory
	if err != nil {
		return fmt.Errorf("nvme: open dir for sync: %w", err)
	}
	syncErr := d.Sync()
	closeErr := d.Close()
	if syncErr != nil {
		return fmt.Errorf("nvme: fsync dir %s: %w", dir, syncErr)
	}
	return closeErr
}
