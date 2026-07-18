//go:build !linux

package target

import (
	"os"
	"sync/atomic"
)

// directFellBack is always true off Linux (no O_DIRECT) — the floor bar's
// numbers are Linux-only anyway (never quoted from a Mac).
var directFellBack atomic.Bool

func init() { directFellBack.Store(true) }

// Buffered fallbacks for the dev box — the floor bar's NUMBERS are
// Linux-only (never quoted from a Mac, per the methodology rules); this
// path exists so the harness and its tests run everywhere.

func writeBlock(path string, blob []byte, fsync bool) error {
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o600) //nolint:gosec // G304: bench-local block file
	if err != nil {
		return err
	}
	if _, err := f.Write(blob); err != nil {
		_ = f.Close()
		return err
	}
	if fsync {
		if err := f.Sync(); err != nil {
			_ = f.Close()
			return err
		}
	}
	return f.Close()
}

func readBlock(path string, dst *[]byte, blob int) (int, error) {
	b, err := os.ReadFile(path) //nolint:gosec // G304: bench-local block file
	if err != nil {
		return 0, err
	}
	if len(*dst) < len(b) {
		*dst = make([]byte, len(b))
	}
	copy((*dst)[:len(b)], b)
	_ = blob
	return len(b), nil
}
