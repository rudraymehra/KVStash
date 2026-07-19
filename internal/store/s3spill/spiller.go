package s3spill

import (
	"context"
	"fmt"
	"io"
	"sync"
	"sync/atomic"

	"github.com/aws/aws-sdk-go-v2/service/s3"
)

// Spiller writes sealed segments to S3 on an async, BOUNDED write-back
// queue: fire-and-forget from the seal pipeline, drop-on-overflow with a
// counter — the cold tier must NEVER block a foreground PUT (the PegaFlow
// posture). A dropped spill is not loss: the local NVMe segment stays
// authoritative until Drop() — spill is a COPY, never a move.
type Spiller struct {
	api    S3API
	cfg    Config
	queue  chan spillReq
	wg     sync.WaitGroup
	cancel context.CancelFunc

	spilled   atomic.Uint64 // segments landed on S3
	dropped   atomic.Uint64 // enqueue overflows (counter — never silent)
	putErrors atomic.Uint64 // failed uploads (the segment stays local-only)
}

type spillReq struct {
	segID uint64
	size  int64
	open  func() (io.ReadSeekCloser, error) // caller-owned segment reader (files ARE seekable; SigV4 payload hashing needs Seek)
	onUp  func(segID uint64, ok bool)       // completion hook (tier flip on ok)
}

// NewSpiller starts the write-back worker. queueDepth bounds in-flight
// spill requests (segments, not bytes); 0 = 8.
func NewSpiller(api S3API, cfg Config, queueDepth int) *Spiller {
	if queueDepth <= 0 {
		queueDepth = 8
	}
	ctx, cancel := context.WithCancel(context.Background())
	sp := &Spiller{
		api:    api,
		cfg:    cfg.withDefaults(),
		queue:  make(chan spillReq, queueDepth),
		cancel: cancel,
	}
	sp.wg.Add(1)
	go sp.loop(ctx)
	return sp
}

// DemoteSegment enqueues one sealed segment for upload. Returns false when
// the queue is full — counted, never blocking; the seal pipeline retries a
// dropped segment on its next pass (the segment is still local).
func (sp *Spiller) DemoteSegment(segID uint64, size int64, open func() (io.ReadSeekCloser, error), onUp func(uint64, bool)) bool {
	select {
	case sp.queue <- spillReq{segID: segID, size: size, open: open, onUp: onUp}:
		return true
	default:
		sp.dropped.Add(1)
		return false
	}
}

// Drop deletes a segment's S3 object (reclaim retired it everywhere).
// Best-effort with the op deadline; an orphaned object costs cents and the
// bucket lifecycle rule is the backstop — never fail reclaim over it.
func (sp *Spiller) Drop(ctx context.Context, segID uint64) error {
	ctx, cancel := context.WithTimeout(ctx, sp.cfg.OpTimeout)
	defer cancel()
	key := segKey(sp.cfg.NodeID, segID)
	_, err := sp.api.DeleteObject(ctx, &s3.DeleteObjectInput{Bucket: &sp.cfg.Bucket, Key: &key})
	return err
}

// Stats exposes the counters (Stats JSON → kvb_s3_* scrape).
func (sp *Spiller) Stats() (spilled, dropped, putErrors uint64) {
	return sp.spilled.Load(), sp.dropped.Load(), sp.putErrors.Load()
}

// Close stops the worker after the queue drains (bounded by per-op
// deadlines; in-flight uploads finish or time out).
func (sp *Spiller) Close() {
	sp.cancel()
	sp.wg.Wait()
}

func (sp *Spiller) loop(ctx context.Context) {
	defer sp.wg.Done()
	for {
		select {
		case req := <-sp.queue:
			ok := sp.upload(ctx, req)
			if req.onUp != nil {
				req.onUp(req.segID, ok)
			}
		case <-ctx.Done():
			// Drain-with-callbacks: every queued request gets its answer
			// (false) — abandoned callbacks once leaked arena refs in the
			// NVMe writer; same contract here.
			for {
				select {
				case req := <-sp.queue:
					if req.onUp != nil {
						req.onUp(req.segID, false)
					}
				default:
					return
				}
			}
		}
	}
}

// upload PUTs one whole segment. Plain PutObject, deliberately not the
// transfermanager: a sealed segment is ONE object by design (the request-
// cost coalescing IS the feature), the SDK streams the body, and plain
// PutObject keeps the static-build surface minimal.
func (sp *Spiller) upload(ctx context.Context, req spillReq) bool {
	rc, err := req.open()
	if err != nil {
		sp.putErrors.Add(1)
		return false
	}
	defer func() { _ = rc.Close() }()
	ctx, cancel := context.WithTimeout(ctx, sp.cfg.OpTimeout)
	defer cancel()
	key := segKey(sp.cfg.NodeID, req.segID)
	_, err = sp.api.PutObject(ctx, &s3.PutObjectInput{
		Bucket:        &sp.cfg.Bucket,
		Key:           &key,
		Body:          rc,
		ContentLength: &req.size,
	})
	if err != nil {
		sp.putErrors.Add(1)
		return false
	}
	sp.spilled.Add(1)
	return true
}

// Verify HeadObjects a segment (integration checks, never the hot path).
func (sp *Spiller) Verify(ctx context.Context, segID uint64, wantSize int64) error {
	ctx, cancel := context.WithTimeout(ctx, sp.cfg.OpTimeout)
	defer cancel()
	key := segKey(sp.cfg.NodeID, segID)
	out, err := sp.api.HeadObject(ctx, &s3.HeadObjectInput{Bucket: &sp.cfg.Bucket, Key: &key})
	if err != nil {
		return err
	}
	if out.ContentLength == nil || *out.ContentLength != wantSize {
		return fmt.Errorf("s3spill: seg %d size mismatch: have %v want %d", segID, out.ContentLength, wantSize)
	}
	return nil
}
