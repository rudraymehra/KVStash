package target

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/kvstash/kvblockd/internal/protocol"
	"github.com/kvstash/kvblockd/pkg/client"
)

// KVBlockd drives the real daemon through pkg/client — the same pooled
// synchronous connections a production adapter uses.
//
// Honesty notes baked into the driver:
//   - There is NO batch-PUT wire verb (a "batch PUT" is N pipelined
//     streams): BatchPut issues per-key Puts sequentially on this call;
//     PUT-side concurrency comes from the harness running streams-many
//     workers, matching the client's own pool model. Disclosed in the
//     README and the JSONL notes.
//   - Client-side xxh3 verification stays ON by default (every GET is
//     corruption-checked — methodology rule 8); --noverify exists to
//     isolate the verification cost, labeled in the cell.
type KVBlockd struct {
	c *client.Client
}

// KVBOptions configures the driver.
type KVBOptions struct {
	Addr       string
	Namespace  string
	Token      string
	Streams    int
	SockBuf    int
	SkipVerify bool
}

// DialKVBlockd connects the pool.
func DialKVBlockd(ctx context.Context, o KVBOptions) (*KVBlockd, error) {
	c, err := client.Dial(ctx, o.Addr, client.Options{
		Streams:     o.Streams,
		Namespace:   o.Namespace,
		Token:       o.Token,
		SockSndBuf:  o.SockBuf,
		SockRcvBuf:  o.SockBuf,
		SkipVerify:  o.SkipVerify,
		DialTimeout: 5 * time.Second,
	})
	if err != nil {
		return nil, fmt.Errorf("target: dial kvblockd: %w", err)
	}
	return &KVBlockd{c: c}, nil
}

// Limits reports the negotiated batching cap.
func (t *KVBlockd) Limits() Limits {
	return Limits{MaxBatchKeys: int(t.c.Limits().MaxBatchKeys)}
}

// BatchPut stores each key (write-once: OK_EXISTS is success).
func (t *KVBlockd) BatchPut(ctx context.Context, keys [][32]byte, blobs [][]byte) ([]Status, error) {
	out := make([]Status, len(keys))
	for i, k := range keys {
		if err := t.c.Put(ctx, k, blobs[i]); err != nil {
			st, ok := statusOf(err)
			if !ok {
				return out, err
			}
			out[i] = st
			continue
		}
		out[i] = OK
	}
	return out, nil
}

// BatchGet streams payloads into dst.
func (t *KVBlockd) BatchGet(ctx context.Context, keys [][32]byte, dst [][]byte) ([]Status, error) {
	sts, err := t.c.BatchGet(ctx, keys, dst)
	if err != nil {
		return nil, err
	}
	out := make([]Status, len(sts))
	for i, s := range sts {
		out[i] = mapStatus(s)
	}
	return out, nil
}

// BatchExists returns the consecutive-prefix hit count.
func (t *KVBlockd) BatchExists(ctx context.Context, chain [][32]byte) (int, error) {
	n, _, err := t.c.BatchExists(ctx, chain)
	return n, err
}

// Close shuts the pool down.
func (t *KVBlockd) Close() error {
	t.c.Close()
	return nil
}

func mapStatus(s protocol.Status) Status {
	switch s {
	case protocol.StatusOK, protocol.StatusOKExists:
		return OK
	case protocol.StatusNotFound, protocol.StatusEvicted:
		return Miss
	case protocol.StatusErrBusy:
		return Busy
	default:
		return Errored
	}
}

func statusOf(err error) (Status, bool) {
	var se *client.StatusError
	if errors.As(err, &se) {
		return mapStatus(se.Status), true
	}
	return Errored, false
}
