package s3spill

import (
	"context"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/aws/aws-sdk-go-v2/service/s3/types"
	"github.com/aws/smithy-go"
)

// objectNotFound wraps an object-level miss so the orchestrator can separate
// "this object is gone" (per-key miss, endpoint healthy) from "the endpoint
// is sick" (breaker food) without a shared sentinel: the store side matches
// errors.As against the structural ObjectNotFound() method — the same
// SDK-free posture as the S3API seam itself.
type objectNotFound struct{ err error }

func (e *objectNotFound) Error() string        { return e.err.Error() }
func (e *objectNotFound) Unwrap() error        { return e.err }
func (e *objectNotFound) ObjectNotFound() bool { return true }

// classifyNotFound wraps the SDK's object-miss shapes: the typed NoSuchKey,
// plus a compat target's bare NoSuchKey/NotFound code (MinIO-class endpoints
// don't always deserialize to the typed error). Everything else — transport,
// deadline, 5xx — passes through untouched.
func classifyNotFound(err error) error {
	var nsk *types.NoSuchKey
	if errors.As(err, &nsk) {
		return &objectNotFound{err: err}
	}
	var ae smithy.APIError
	if errors.As(err, &ae) {
		switch ae.ErrorCode() {
		case "NoSuchKey", "NotFound":
			return &objectNotFound{err: err}
		}
	}
	return err
}

// Restorer serves cold reads: ONE ranged GetObject for exactly (segID,
// offset, len), and a singleflight whole-segment restore so two concurrent
// cold misses on one segment trigger exactly one download (a 256 MiB GET
// must never run twice for the same bytes).
type Restorer struct {
	api S3API
	cfg Config

	mu       sync.Mutex
	inflight map[uint64]*restoreCall
	closed   bool           // set by Close under mu — no new downloads spawn after
	wg       sync.WaitGroup // tracks detached downloads; Close drains it

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
		return classifyNotFound(err)
	}
	defer drainClose(out.Body)
	// A Range-ignoring endpoint (some proxies and compat targets) answers
	// 200 + the WHOLE object; ReadFull would then silently hand back the
	// segment's first n bytes as [off, off+n). Verify the response window
	// before reading — one loud error beats a checksum-error storm plus a
	// whole-segment drain per read.
	if err := verifyRange(out, off, n); err != nil {
		return fmt.Errorf("s3spill: ranged read seg %d: %w", segID, err)
	}
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
//
// ctx cancels the WAIT, not the work: the shared download runs detached (an
// impatient winner must not poison coalesced followers), so this can return
// ctx.Err() with the download AND sink still running — a retry coalesces
// onto that same running download, and Close is the barrier that drains
// whatever is still detached at shutdown.
func (r *Restorer) RestoreSegment(ctx context.Context, segID uint64, sink func(io.Reader) error) error {
	r.mu.Lock()
	if r.closed {
		r.mu.Unlock()
		return fmt.Errorf("s3spill: restore seg %d: restorer closed", segID)
	}
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
	// The Add happens under mu, mutually exclusive with Close's closed flip:
	// any download that got past the closed check is counted before Close
	// starts waiting, so Add can never race the Wait.
	r.wg.Add(1)
	r.mu.Unlock()

	// The download is SHARED state: it runs detached from the winner's ctx
	// (an impatient winner must not poison every coalesced follower), and
	// its close(done)+map-delete are deferred with panic recovery — a
	// panicking caller-supplied sink must surface as this call's error,
	// never wedge the segment's restores until process restart.
	go func() {
		defer func() {
			if p := recover(); p != nil {
				c.err = fmt.Errorf("s3spill: restore seg %d: sink panic: %v", segID, p)
			}
			close(c.done)
			r.mu.Lock()
			delete(r.inflight, segID)
			r.mu.Unlock()
			r.wg.Done()
		}()
		c.err = r.download(context.WithoutCancel(ctx), segID, sink)
	}()
	select {
	case <-c.done:
		return c.err
	case <-ctx.Done():
		return ctx.Err()
	}
}

// Close refuses new restores and BLOCKS until every detached download —
// including ones whose RestoreSegment callers gave up long ago — has
// finished. It is the shutdown barrier that keeps a detached sink (writing
// into an NVMe volume) from outliving the store teardown: callers stop
// issuing restores first (the tiered store's loops are already down when
// the cleanup stack reaches this), then Close drains the stragglers. The
// download's own 4×OpTimeout deadline bounds the wait.
func (r *Restorer) Close() {
	r.mu.Lock()
	r.closed = true
	r.mu.Unlock()
	r.wg.Wait()
}

func (r *Restorer) download(ctx context.Context, segID uint64, sink func(io.Reader) error) error {
	ctx, cancel := context.WithTimeout(ctx, 4*r.cfg.OpTimeout) // whole segments are big; sole deadline once detached
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

// verifyRange asserts a GetObject response actually honors the requested
// [off, off+n) window: a well-behaved 206 carries "bytes off-(off+n-1)/…"
// in Content-Range; absent that header, Content-Length must equal n.
func verifyRange(out *s3.GetObjectOutput, off, n int64) error {
	if cr := out.ContentRange; cr != nil {
		want := fmt.Sprintf("bytes %d-%d/", off, off+n-1)
		if !strings.HasPrefix(*cr, want) {
			return fmt.Errorf("endpoint returned range %q, want %q", *cr, want+"*")
		}
		return nil
	}
	if out.ContentLength == nil || *out.ContentLength != n {
		return fmt.Errorf("endpoint ignored Range (no Content-Range, Content-Length %v != %d)", out.ContentLength, n)
	}
	return nil
}

// Stats exposes the read-side counters.
func (r *Restorer) Stats() (rangedGets, restores uint64) {
	return r.rangedGets.Load(), r.restores.Load()
}
