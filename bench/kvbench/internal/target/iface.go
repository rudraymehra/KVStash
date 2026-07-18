// Package target abstracts the stores under test behind SPEC-4 §2.1's
// interface, so the sweep and the trace replayer drive kvblockd, Redis/
// Valkey, and the NVMe-fs floor with the SAME op stream.
package target

import "context"

// Status is kvbench's own per-key outcome (drivers map their native codes).
type Status uint8

const (
	OK     Status = iota // a GET hit, or a PUT that genuinely wrote bytes
	Exists               // a PUT whose key already existed — NO bytes moved (write-once no-op)
	Miss                 // a GET miss
	Busy                 // retryable backpressure (kvblockd's per-key ERR_BUSY)
	Errored
)

// Wrote reports whether a PUT status moved payload bytes (goodput counts
// only genuine writes — an idempotent OK_EXISTS ack moved nothing, and
// crediting it payload bytes was the ladder's confirmed PUT-inflation bug).
func (s Status) Wrote() bool { return s == OK }

// Target is the store under test. dst buffers in BatchGet are caller-owned
// and reused across calls.
type Target interface {
	BatchPut(ctx context.Context, keys [][32]byte, blobs [][]byte) ([]Status, error)
	BatchGet(ctx context.Context, keys [][32]byte, dst [][]byte) ([]Status, error)
	BatchExists(ctx context.Context, chain [][32]byte) (consecutiveHits int, err error)
	Close() error
}

// Limits exposes a driver's batching cap (0 = unbounded). kvblockd's comes
// from HELLO negotiation; callers tile with the helpers in tile.go.
type Limits struct {
	MaxBatchKeys int
}

// Limiter is the optional cap-reporting extension.
type Limiter interface {
	Limits() Limits
}

// LimitOf returns the target's batching cap, 0 when unbounded.
func LimitOf(t Target) int {
	if l, ok := t.(Limiter); ok {
		return l.Limits().MaxBatchKeys
	}
	return 0
}
