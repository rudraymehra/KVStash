package target

import (
	"context"
	"encoding/hex"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sync"
)

// FS is the filesystem floor bar: one file per block on the volume the
// operator points at (XFS on the rig), direct I/O where the platform has
// it, a bounded worker pool. It exists to answer "why not just files?" with
// a measured number, not an argument.
//
// Disclosure baked in: files are padded to 4 KiB multiples for O_DIRECT
// (the run's fixed blob size is a multiple already — grid invariant), and
// the fsync policy is a flag recorded in the JSONL cell.
type FS struct {
	dir     string
	blob    int
	fsync   bool
	workers int
	sem     chan struct{}
	mu      sync.Mutex // serializes directory creation only
	made    map[string]bool
}

// FSOptions configures the floor driver.
type FSOptions struct {
	Dir       string
	BlobBytes int
	Workers   int
	Fdatasync bool // fsync each PUT (the "durable floor" variant) — disclosed
}

// OpenFS validates the directory and builds the pool.
func OpenFS(o FSOptions) (*FS, error) {
	if o.Workers <= 0 {
		o.Workers = 16
	}
	if o.BlobBytes <= 0 || o.BlobBytes%4096 != 0 {
		return nil, fmt.Errorf("target: nvmefs blob %d must be a positive 4 KiB multiple", o.BlobBytes)
	}
	if err := os.MkdirAll(o.Dir, 0o750); err != nil {
		return nil, err
	}
	return &FS{
		dir: o.Dir, blob: o.BlobBytes, fsync: o.Fdatasync, workers: o.Workers,
		sem:  make(chan struct{}, o.Workers),
		made: make(map[string]bool),
	}, nil
}

func (t *FS) path(key [32]byte) (dir, file string) {
	sub := filepath.Join(t.dir, hex.EncodeToString(key[:1]))
	return sub, filepath.Join(sub, hex.EncodeToString(key[:])+".blk")
}

func (t *FS) ensureDir(dir string) error {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.made[dir] {
		return nil
	}
	if err := os.MkdirAll(dir, 0o750); err != nil {
		return err
	}
	t.made[dir] = true
	return nil
}

// BatchPut writes each blob through the pool (write-once: existing files
// are left alone).
func (t *FS) BatchPut(ctx context.Context, keys [][32]byte, blobs [][]byte) ([]Status, error) {
	return t.each(ctx, len(keys), func(i int) (Status, error) {
		dir, file := t.path(keys[i])
		if _, err := os.Stat(file); err == nil {
			return Exists, nil // write-once idempotent hit — moved no bytes
		}
		if err := t.ensureDir(dir); err != nil {
			return Errored, err
		}
		if err := writeBlock(file, blobs[i], t.fsync); err != nil {
			return Errored, err
		}
		return OK, nil
	})
}

// BatchGet reads each block through the pool.
func (t *FS) BatchGet(ctx context.Context, keys [][32]byte, dst [][]byte) ([]Status, error) {
	return t.each(ctx, len(keys), func(i int) (Status, error) {
		_, file := t.path(keys[i])
		n, err := readBlock(file, &dst[i], t.blob)
		if err != nil {
			// errors.Is, NOT os.IsNotExist: the Linux O_DIRECT reader wraps
			// ENOENT in its own type, and os.IsNotExist only unwraps the
			// stdlib Path/Link/Syscall error wrappers.
			if errors.Is(err, fs.ErrNotExist) {
				return Miss, nil
			}
			return Errored, err
		}
		dst[i] = dst[i][:n]
		return OK, nil
	})
}

// BatchExists stats the chain until the first absent file.
func (t *FS) BatchExists(_ context.Context, chain [][32]byte) (int, error) {
	n := 0
	for _, k := range chain {
		_, file := t.path(k)
		if _, err := os.Stat(file); err != nil {
			break
		}
		n++
	}
	return n, nil
}

// Close is a no-op (files persist — that IS the floor's story).
func (t *FS) Close() error { return nil }

// each fans fn across the worker pool preserving index order in the
// returned statuses; the first hard error wins. It ALWAYS waits for the
// launched goroutines before returning (even on ctx cancel) — the ladder
// caught a return-before-wg.Wait that let goroutines write out/errs after
// the caller reclaimed the slices (a data race).
func (t *FS) each(ctx context.Context, n int, fn func(i int) (Status, error)) ([]Status, error) {
	out := make([]Status, n)
	errs := make([]error, n)
	var wg sync.WaitGroup
	canceled := false
	for i := 0; i < n; i++ {
		select {
		case t.sem <- struct{}{}:
		case <-ctx.Done():
			canceled = true
		}
		if canceled {
			break
		}
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			defer func() { <-t.sem }()
			out[i], errs[i] = fn(i)
		}(i)
	}
	wg.Wait()
	if canceled {
		return out, ctx.Err()
	}
	for _, err := range errs {
		if err != nil {
			return out, err
		}
	}
	return out, nil
}

// DirectFallback reports whether ANY open on this volume fell back from
// O_DIRECT to buffered I/O (a tmpfs/overlay mount) — the floor bar must
// disclose it in the JSONL, since buffered I/O benchmarks the page cache,
// not the device.
func (t *FS) DirectFallback() bool { return directFellBack.Load() }
