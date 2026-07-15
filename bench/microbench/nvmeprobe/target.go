package main

import (
	"fmt"
	"os"
	"unsafe"

	"golang.org/x/sys/unix"
)

// align4096 returns how many bytes to skip from the slice base so the address
// is 4096-aligned.
func align4096(b []byte) uintptr {
	addr := uintptr(unsafe.Pointer(&b[0]))
	rem := addr & 4095
	if rem == 0 {
		return 0
	}
	return 4096 - rem
}

// openTarget opens the file/device for probing and returns (fd, usable size).
// For a regular file it ensures the file exists at cfg.fileSize and is FULLY
// WRITTEN (not sparse): reading holes returns zeros straight from the kernel
// without touching the disk, which would report a dishonestly high GB/s.
func openTarget(cfg probeConfig) (int, int64, error) {
	// Create/fill regular files with a plain buffered fd first.
	st, statErr := os.Stat(cfg.path)
	isBlock := statErr == nil && st.Mode()&os.ModeDevice != 0
	if !isBlock {
		if err := ensureFilled(cfg.path, cfg.fileSize); err != nil {
			return -1, 0, err
		}
	}

	fd, err := openIO(cfg.path, cfg.op == "write", cfg.direct)
	if err != nil {
		return -1, 0, fmt.Errorf("open %s: %w", cfg.path, err)
	}
	size, err := unix.Seek(fd, 0, 2 /* SEEK_END */)
	if err != nil {
		_ = unix.Close(fd)
		return -1, 0, fmt.Errorf("size %s: %w", cfg.path, err)
	}
	return fd, size, nil
}

// ensureFilled makes sure path exists, is >= size bytes, and has real data
// (written, fsynced) so O_DIRECT reads hit the device rather than a hole.
func ensureFilled(path string, size int64) error {
	st, err := os.Stat(path)
	if err == nil && st.Size() >= size {
		return nil // assume previously filled by us
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY, 0o600) //nolint:gosec // G304: path is the CLI's own --path flag
	if err != nil {
		return err
	}
	defer f.Close()
	fmt.Fprintf(os.Stderr, "nvmeprobe: filling %s to %d MiB (one-time)...\n", path, size>>20)
	buf := make([]byte, 4<<20)
	for i := range buf {
		buf[i] = byte((i * 31) % 256) //nolint:gosec // %256 always fits a byte
	}
	var written int64
	for written < size {
		n, werr := f.Write(buf)
		if werr != nil {
			return werr
		}
		written += int64(n)
	}
	return f.Sync()
}

// offsets generates bs-aligned offsets pseudo-randomly across [0, size-bs]
// (LCG, deterministic per seed) — the serving path is random reads at large
// block size; writes use the same distribution for comparability.
type offsets struct {
	rng    uint64
	blocks uint64
	bs     int64
}

func newOffsets(seed uint64, size int64, bs int) offsets {
	blocks := max(size/int64(bs), 1) // size>=bs is validated in run(), so >=1
	return offsets{rng: seed | 1, blocks: uint64(blocks), bs: int64(bs)}
}

func (o *offsets) next() int64 {
	o.rng = o.rng*6364136223846793005 + 1442695040888963407
	block := o.rng >> 17 % o.blocks // < blocks <= size/bs, fits int64
	return int64(block) * o.bs      //nolint:gosec // bounded by blocks (validated size)
}
