package target

import (
	"context"
	"sync"
)

// Mem is the test double: a map-backed store with FIFO capacity eviction
// (for replay hit-rate tests), a configurable batching cap (tiling tests),
// and byte-flip fault injection (the verify-catches-corruption test).
type Mem struct {
	mu       sync.Mutex
	m        map[[32]byte][]byte
	order    [][32]byte // FIFO insertion order
	capBytes int64
	used     int64
	maxBatch int
	flip     map[[32]byte]int // key → offset to corrupt on the NEXT get
}

// NewMem builds the double. capBytes ≤ 0 = unbounded; maxBatch ≤ 0 = uncapped.
func NewMem(capBytes int64, maxBatch int) *Mem {
	return &Mem{
		m:        make(map[[32]byte][]byte),
		capBytes: capBytes,
		maxBatch: maxBatch,
		flip:     make(map[[32]byte]int),
	}
}

// Limits implements Limiter.
func (t *Mem) Limits() Limits { return Limits{MaxBatchKeys: t.maxBatch} }

// FlipByteOnce corrupts the stored blob at off on its next read — the
// injected-flip oracle for `kvbench verify`.
func (t *Mem) FlipByteOnce(key [32]byte, off int) {
	t.mu.Lock()
	t.flip[key] = off
	t.mu.Unlock()
}

func (t *Mem) checkBatch(n int) bool { return t.maxBatch <= 0 || n <= t.maxBatch }

// BatchPut stores copies (write-once: an existing key is an OK no-op —
// kvblockd's OK_EXISTS shape).
func (t *Mem) BatchPut(_ context.Context, keys [][32]byte, blobs [][]byte) ([]Status, error) {
	if !t.checkBatch(len(keys)) {
		return nil, errBatchTooBig(len(keys), t.maxBatch)
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	out := make([]Status, len(keys))
	for i, k := range keys {
		if _, ok := t.m[k]; ok {
			out[i] = Exists // write-once no-op — moved no bytes
			continue
		}
		cp := make([]byte, len(blobs[i]))
		copy(cp, blobs[i])
		t.m[k] = cp
		t.order = append(t.order, k)
		t.used += int64(len(cp))
		for t.capBytes > 0 && t.used > t.capBytes && len(t.order) > 0 {
			victim := t.order[0]
			t.order = t.order[1:]
			if b, ok := t.m[victim]; ok {
				t.used -= int64(len(b))
				delete(t.m, victim)
			}
		}
		out[i] = OK
	}
	return out, nil
}

// BatchGet copies into dst (resizing when needed, mirroring pkg/client).
func (t *Mem) BatchGet(_ context.Context, keys [][32]byte, dst [][]byte) ([]Status, error) {
	if !t.checkBatch(len(keys)) {
		return nil, errBatchTooBig(len(keys), t.maxBatch)
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	out := make([]Status, len(keys))
	for i, k := range keys {
		b, ok := t.m[k]
		if !ok {
			out[i] = Miss
			continue
		}
		if len(dst[i]) != len(b) { //nolint:gosec // G602: len(dst)==len(keys) is the Target contract (pkg/client shape)
			dst[i] = make([]byte, len(b)) //nolint:gosec // G602: as above
		}
		copy(dst[i], b) //nolint:gosec // G602: as above
		if off, hurt := t.flip[k]; hurt && off < len(dst[i]) {
			dst[i][off] ^= 0x01 //nolint:gosec // G602: off < len(dst[i]) checked on the line above
			delete(t.flip, k)
		}
		out[i] = OK
	}
	return out, nil
}

// BatchExists counts the consecutive-from-0 present prefix.
func (t *Mem) BatchExists(_ context.Context, chain [][32]byte) (int, error) {
	if !t.checkBatch(len(chain)) {
		return 0, errBatchTooBig(len(chain), t.maxBatch)
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	n := 0
	for _, k := range chain {
		if _, ok := t.m[k]; !ok {
			break
		}
		n++
	}
	return n, nil
}

// Close is a no-op.
func (t *Mem) Close() error { return nil }

type batchErr struct{ n, lim int }

func errBatchTooBig(n, lim int) error { return batchErr{n, lim} }
func (e batchErr) Error() string {
	return "target: batch over the cap (caller must tile)"
}
