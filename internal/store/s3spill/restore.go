package s3spill

import (
	"context"
	"fmt"
	"io"
	"sync"
	"sync/atomic"

	"github.com/aws/aws-sdk-go-v2/service/s3"
)

// Restorer serves cold reads: ONE ranged GetObject for exactly (segID,
// offset, len), and a singleflight whole-segment restore so two concurrent
// cold misses on one segment trigger exactly one download (a 256 MiB GET
// must never run twice for the same bytes).
type Restorer struct {
	api S3API
	cfg Config

	mu       sync.Mutex
	inflight map[uint64]*restoreCall

	rangedGets atomic.Uint64 // cold per-block reads served
	restores   atomic.Uint64 // whole-segment downloads completed
}

type restoreCall struct {
	done chan struct{}
	err  error
}

// NewRestorer builds the read side over the same S3API seam.
func NewRestorer(api S3API, cfg Config) *Restorer {
	return &Restorer{api: api, cfg: cfg.withDefaults(), inflight: make(map[uint64]*restoreCall)}
}

// ReadRange serves one cold block: bytes [off, off+n) of the segment
// object, streamed into dst (len(dst) == n). The caller's ctx carries the
// wire deadline — a slow S3 maps to a per-key error, never a frame stall.
func (r *Restorer) ReadRange(ctx context.Context, segID uint64, off, n int64, dst []byte) error {
	if int64(len(dst)) != n {
		return fmt.Errorf("s3spill: dst %d != range %d", len(dst), n)
	}
	ctx, cancel := context.WithTimeout(ctx, r.cfg.OpTimeout)
	defer cancel()
	key := segKey(r.cfg.NodeID, segID)
	rng := fmt.Sprintf("bytes=%d-%d", off, off+n-1)
	out, err := r.api.GetObject(ctx, &s3.GetObjectInput{
		Bucket: &r.cfg.Bucket, Key: &key, Range: &rng,
	})
	if err != nil {
		return err
	}
	defer drainClose(out.Body)
	if _, err := io.ReadFull(out.Body, dst); err != nil {
		return fmt.Errorf("s3spill: ranged read seg %d: %w", segID, err)
	}
	r.rangedGets.Add(1)
	return nil
}

// RestoreSegment downloads the WHOLE segment through sink (the caller
// writes it back into a local NVMe volume). Singleflight per segment:
// concurrent callers coalesce onto one download and share its verdict.
// sink is only invoked on the winning call.
func (r *Restorer) RestoreSegment(ctx context.Context, segID uint64, sink func(io.Reader) error) error {
	r.mu.Lock()
	if c, ok := r.inflight[segID]; ok {
		r.mu.Unlock()
		select {
		case <-c.done:
			return c.err
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	c := &restoreCall{done: make(chan struct{})}
	r.inflight[segID] = c
	r.mu.Unlock()

	c.err = r.download(ctx, segID, sink)
	close(c.done)

	r.mu.Lock()
	delete(r.inflight, segID)
	r.mu.Unlock()
	return c.err
}

func (r *Restorer) download(ctx context.Context, segID uint64, sink func(io.Reader) error) error {
	ctx, cancel := context.WithTimeout(ctx, 4*r.cfg.OpTimeout) // whole segments are big
	defer cancel()
	key := segKey(r.cfg.NodeID, segID)
	out, err := r.api.GetObject(ctx, &s3.GetObjectInput{Bucket: &r.cfg.Bucket, Key: &key})
	if err != nil {
		return err
	}
	defer drainClose(out.Body)
	if err := sink(out.Body); err != nil {
		return err
	}
	r.restores.Add(1)
	return nil
}

// Stats exposes the read-side counters.
func (r *Restorer) Stats() (rangedGets, restores uint64) {
	return r.rangedGets.Load(), r.restores.Load()
}
