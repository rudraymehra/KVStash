package s3spill

import (
	"bytes"
	"context"
	"io"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	awsc "github.com/aws/aws-sdk-go-v2/aws"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/johannesboyne/gofakes3"
	"github.com/johannesboyne/gofakes3/backend/s3mem"
)

// fakeS3 boots an in-process gofakes3 and returns a real SDK client aimed
// at it — the same code path production takes, endpoint-overridden.
func fakeS3(t *testing.T, bucket string) S3API { //nolint:unparam // bucket names the fixture, not a constant contract
	t.Helper()
	backend := s3mem.New()
	faker := gofakes3.New(backend)
	srv := httptest.NewServer(faker.Server())
	t.Cleanup(srv.Close)
	if err := backend.CreateBucket(bucket); err != nil {
		t.Fatal(err)
	}
	cfg, err := awsconfig.LoadDefaultConfig(
		context.Background(),
		awsconfig.WithRegion("us-east-1"),
		awsconfig.WithCredentialsProvider(credentials.NewStaticCredentialsProvider("test", "test", "")),
	)
	if err != nil {
		t.Fatal(err)
	}
	return s3.NewFromConfig(cfg, func(o *s3.Options) {
		o.BaseEndpoint = awsc.String(srv.URL)
		o.UsePathStyle = true
		o.RequestChecksumCalculation = awsc.RequestChecksumCalculationWhenRequired
	})
}

// nopSeekCloser adapts a bytes.Reader to the seekable-body seam.
type nopSeekCloser struct{ *bytes.Reader }

func (nopSeekCloser) Close() error { return nil }

func segBody(n int, seed byte) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = seed ^ byte(i)
	}
	return b
}

func TestSpillUploadNamingAndRoundTrip(t *testing.T) {
	api := fakeS3(t, "kvb-test")
	cfg := Config{Bucket: "kvb-test", NodeID: "node-1", OpTimeout: 10 * time.Second}
	sp := NewSpiller(api, cfg, 4)
	defer sp.Close()

	body := segBody(1<<20, 0x5A)
	done := make(chan bool, 1)
	ok := sp.DemoteSegment(
		7, int64(len(body)),
		func() (io.ReadSeekCloser, error) { return nopSeekCloser{bytes.NewReader(body)}, nil },
		func(_ uint64, up bool) { done <- up },
	)
	if !ok {
		t.Fatal("enqueue refused on an empty queue")
	}
	if up := <-done; !up {
		t.Fatal("upload failed")
	}
	if err := sp.Verify(context.Background(), 7, int64(len(body))); err != nil {
		t.Fatalf("verify (naming/size): %v", err)
	}

	// Ranged read gets EXACTLY the requested window back.
	r := NewRestorer(api, cfg)
	dst := make([]byte, 4096)
	if err := r.ReadRange(context.Background(), 7, 512<<10, 4096, dst); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(dst, body[512<<10:512<<10+4096]) {
		t.Fatal("ranged read returned wrong bytes")
	}

	// Drop removes the object; Verify must fail after.
	if err := sp.Drop(context.Background(), 7); err != nil {
		t.Fatal(err)
	}
	if err := sp.Verify(context.Background(), 7, int64(len(body))); err == nil {
		t.Fatal("verify succeeded after Drop")
	}
}

// TestSpillPutFailureCountsAndAnswersFalse: a failed upload (here: the
// bucket does not exist) must bump put_errors and answer the completion
// hook false — the segment stays local-only and the caller retries next
// pass; a silent drop would strand the segment off both ledgers.
func TestSpillPutFailureCountsAndAnswersFalse(t *testing.T) {
	api := fakeS3(t, "kvb-test") // fixture bucket exists; we aim elsewhere
	sp := NewSpiller(api, Config{Bucket: "no-such-bucket", NodeID: "n", OpTimeout: 5 * time.Second}, 2)
	defer sp.Close()

	done := make(chan bool, 1)
	ok := sp.DemoteSegment(1, 1024,
		func() (io.ReadSeekCloser, error) { return nopSeekCloser{bytes.NewReader(segBody(1024, 7))}, nil },
		func(_ uint64, up bool) { done <- up })
	if !ok {
		t.Fatal("enqueue refused on an empty queue")
	}
	if up := <-done; up {
		t.Fatal("upload into a missing bucket reported success")
	}
	if _, _, putErrs := sp.Stats(); putErrs != 1 {
		t.Fatalf("put_errors = %d, want 1", putErrs)
	}
}

// slowAPI wraps an S3API making PutObject block until released — the
// never-blocks-foreground proof.
type slowAPI struct {
	S3API
	gate chan struct{}
}

func (s *slowAPI) PutObject(ctx context.Context, in *s3.PutObjectInput, opts ...func(*s3.Options)) (*s3.PutObjectOutput, error) {
	select {
	case <-s.gate:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	return s.S3API.PutObject(ctx, in, opts...)
}

func TestSpillNeverBlocksForeground(t *testing.T) {
	api := &slowAPI{S3API: fakeS3(t, "kvb-test"), gate: make(chan struct{})}
	sp := NewSpiller(api, Config{Bucket: "kvb-test", NodeID: "n", OpTimeout: 5 * time.Second}, 2)
	defer sp.Close()
	defer close(api.gate)

	open := func() (io.ReadSeekCloser, error) { return nopSeekCloser{bytes.NewReader(segBody(1024, 1))}, nil }
	// Fill the queue (worker holds req 1 on the gate; 2 more queue).
	accepted := 0
	for i := 0; i < 8; i++ {
		start := time.Now()
		ok := sp.DemoteSegment(uint64(i), 1024, open, nil)
		if el := time.Since(start); el > 100*time.Millisecond {
			t.Fatalf("DemoteSegment blocked %v with a stalled S3", el)
		}
		if ok {
			accepted++
		}
	}
	if accepted >= 8 {
		t.Fatal("bounded queue accepted everything — no backpressure signal")
	}
	_, dropped, _ := sp.Stats()
	if dropped == 0 {
		t.Fatal("overflow drops uncounted — the silent-cap sin")
	}
}

func TestRestoreSingleflight(t *testing.T) {
	bucket := "kvb-test"
	api := fakeS3(t, bucket)
	cfg := Config{Bucket: bucket, NodeID: "n", OpTimeout: 10 * time.Second}
	sp := NewSpiller(api, cfg, 2)
	defer sp.Close()
	body := segBody(256<<10, 0x33)
	done := make(chan bool, 1)
	sp.DemoteSegment(3, int64(len(body)),
		func() (io.ReadSeekCloser, error) { return nopSeekCloser{bytes.NewReader(body)}, nil },
		func(_ uint64, up bool) { done <- up })
	if !<-done {
		t.Fatal("upload failed")
	}

	r := NewRestorer(api, cfg)
	var sinkRuns atomic.Int32
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			err := r.RestoreSegment(context.Background(), 3, func(rd io.Reader) error {
				sinkRuns.Add(1)
				got, err := io.ReadAll(rd)
				if err != nil {
					return err
				}
				if !bytes.Equal(got, body) {
					t.Error("restored bytes differ")
				}
				return nil
			})
			if err != nil {
				t.Error(err)
			}
		}()
	}
	wg.Wait()
	// Concurrency makes 1..k downloads possible across waves, but 8 blind
	// concurrent calls must coalesce far below 8 — assert the dedup works
	// at all AND that at least the winning sink saw correct bytes.
	if got := sinkRuns.Load(); got > 4 {
		t.Fatalf("singleflight let %d/8 concurrent restores download", got)
	}
}

func TestCloseDrainsWithCallbacks(t *testing.T) {
	api := &slowAPI{S3API: fakeS3(t, "kvb-test"), gate: make(chan struct{})}
	sp := NewSpiller(api, Config{Bucket: "kvb-test", NodeID: "n", OpTimeout: 200 * time.Millisecond}, 4)
	var answered atomic.Int32
	open := func() (io.ReadSeekCloser, error) { return nopSeekCloser{bytes.NewReader(segBody(64, 2))}, nil }
	for i := 0; i < 4; i++ {
		sp.DemoteSegment(uint64(10+i), 64, open, func(_ uint64, _ bool) { answered.Add(1) })
	}
	close(api.gate) // release the worker; short op timeout bounds the rest
	sp.Close()
	if got := answered.Load(); got != 4 {
		t.Fatalf("Close abandoned callbacks: %d/4 answered", got)
	}
}

// stubAPI answers every call instantly and successfully — for stress tests
// where the fixture's HTTP round-trip would hide interleavings.
type stubAPI struct{}

func (stubAPI) PutObject(context.Context, *s3.PutObjectInput, ...func(*s3.Options)) (*s3.PutObjectOutput, error) {
	return &s3.PutObjectOutput{}, nil
}

func (stubAPI) GetObject(context.Context, *s3.GetObjectInput, ...func(*s3.Options)) (*s3.GetObjectOutput, error) {
	return &s3.GetObjectOutput{}, nil
}

func (stubAPI) HeadObject(context.Context, *s3.HeadObjectInput, ...func(*s3.Options)) (*s3.HeadObjectOutput, error) {
	return &s3.HeadObjectOutput{}, nil
}

func (stubAPI) DeleteObject(context.Context, *s3.DeleteObjectInput, ...func(*s3.Options)) (*s3.DeleteObjectOutput, error) {
	return &s3.DeleteObjectOutput{}, nil
}

// TestDemoteSegmentAfterCloseRefuses: post-Close DemoteSegment must refuse
// (mirrors Volume.Append) — accepting into a dead queue means onUp never
// fires and a later Flush spins against a counter nobody drains.
func TestDemoteSegmentAfterCloseRefuses(t *testing.T) {
	sp := NewSpiller(stubAPI{}, Config{Bucket: "b", NodeID: "n", OpTimeout: time.Second}, 2)
	sp.Close()
	ok := sp.DemoteSegment(1, 64,
		func() (io.ReadSeekCloser, error) { return nopSeekCloser{bytes.NewReader(segBody(64, 3))}, nil },
		nil)
	if ok {
		t.Fatal("DemoteSegment accepted a segment after Close")
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if !sp.Flush(ctx) {
		t.Fatal("Flush hung after a post-Close DemoteSegment")
	}
}

// TestInflightNeverNegative: the barrier counter must LEAD the queue — a
// worker decrement landing between the producer's send and its increment
// makes inflight transiently negative, which is exactly the window where
// Flush observes a false zero with a request still queued.
func TestInflightNeverNegative(t *testing.T) {
	sp := NewSpiller(stubAPI{}, Config{Bucket: "b", NodeID: "n", OpTimeout: time.Second}, 8)
	defer sp.Close()
	open := func() (io.ReadSeekCloser, error) { return nopSeekCloser{bytes.NewReader(segBody(8, 9))}, nil }

	stop := make(chan struct{})
	var sawNegative atomic.Bool
	go func() {
		for {
			select {
			case <-stop:
				return
			default:
			}
			if sp.inflight.Load() < 0 {
				sawNegative.Store(true)
				return
			}
		}
	}()
	deadline := time.Now().Add(300 * time.Millisecond)
	for i := uint64(0); time.Now().Before(deadline) && !sawNegative.Load(); i++ {
		sp.DemoteSegment(i, 8, open, nil)
	}
	close(stop)
	if sawNegative.Load() {
		t.Fatal("inflight observed negative: the counter trails the queue and Flush can see a false zero")
	}
}

// TestRestoreSinkPanicDoesNotWedge: a panicking caller-supplied sink must
// surface as the winner's error and leave the segment restorable — not
// strand an inflight entry every future caller joins and times out on.
func TestRestoreSinkPanicDoesNotWedge(t *testing.T) {
	api := fakeS3(t, "kvb-test")
	cfg := Config{Bucket: "kvb-test", NodeID: "n", OpTimeout: 10 * time.Second}
	sp := NewSpiller(api, cfg, 2)
	defer sp.Close()
	body := segBody(4096, 0x21)
	done := make(chan bool, 1)
	sp.DemoteSegment(5, int64(len(body)),
		func() (io.ReadSeekCloser, error) { return nopSeekCloser{bytes.NewReader(body)}, nil },
		func(_ uint64, up bool) { done <- up })
	if !<-done {
		t.Fatal("upload failed")
	}

	r := NewRestorer(api, cfg)
	err := r.RestoreSegment(context.Background(), 5, func(io.Reader) error { panic("sink boom") })
	if err == nil {
		t.Fatal("panicking sink reported success")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	err = r.RestoreSegment(ctx, 5, func(rd io.Reader) error {
		got, rerr := io.ReadAll(rd)
		if rerr != nil {
			return rerr
		}
		if !bytes.Equal(got, body) {
			t.Error("restored bytes differ after a prior sink panic")
		}
		return nil
	})
	if err != nil {
		t.Fatalf("segment permanently unrestorable after a sink panic: %v", err)
	}
}

// gatedGetAPI blocks whole-object GETs (Range==nil) on a gate so the test
// controls when the shared download proceeds.
type gatedGetAPI struct {
	S3API
	started chan struct{}
	gate    chan struct{}
}

func (g *gatedGetAPI) GetObject(ctx context.Context, in *s3.GetObjectInput, opts ...func(*s3.Options)) (*s3.GetObjectOutput, error) {
	if in.Range == nil {
		g.started <- struct{}{}
		<-g.gate
	}
	return g.S3API.GetObject(ctx, in, opts...)
}

// TestRestoreWinnerCancelDoesNotPoisonFollowers: the singleflight winner's
// ctx must not govern the SHARED download — an impatient winner cancelling
// must not fail every coalesced follower.
func TestRestoreWinnerCancelDoesNotPoisonFollowers(t *testing.T) {
	base := fakeS3(t, "kvb-test")
	cfg := Config{Bucket: "kvb-test", NodeID: "n", OpTimeout: 10 * time.Second}
	sp := NewSpiller(base, cfg, 2)
	defer sp.Close()
	body := segBody(4096, 0x44)
	done := make(chan bool, 1)
	sp.DemoteSegment(6, int64(len(body)),
		func() (io.ReadSeekCloser, error) { return nopSeekCloser{bytes.NewReader(body)}, nil },
		func(_ uint64, up bool) { done <- up })
	if !<-done {
		t.Fatal("upload failed")
	}

	api := &gatedGetAPI{S3API: base, started: make(chan struct{}, 1), gate: make(chan struct{})}
	r := NewRestorer(api, cfg)

	winnerCtx, winnerCancel := context.WithCancel(context.Background())
	winnerErr := make(chan error, 1)
	go func() {
		winnerErr <- r.RestoreSegment(winnerCtx, 6, func(rd io.Reader) error {
			_, err := io.Copy(io.Discard, rd)
			return err
		})
	}()
	<-api.started // winner registered and its download is gated

	followerErr := make(chan error, 1)
	go func() {
		followerErr <- r.RestoreSegment(context.Background(), 6, func(io.Reader) error {
			t.Error("follower sink invoked — singleflight broke")
			return nil
		})
	}()

	winnerCancel() // impatient winner walks away mid-download
	close(api.gate)

	if err := <-followerErr; err != nil {
		t.Fatalf("winner's cancellation poisoned the coalesced follower: %v", err)
	}
	<-winnerErr
}

// rangeIgnoringAPI strips Range from GETs — the misbehaving proxy /
// compat-target shape that answers 200 + whole object.
type rangeIgnoringAPI struct{ S3API }

func (a rangeIgnoringAPI) GetObject(ctx context.Context, in *s3.GetObjectInput, opts ...func(*s3.Options)) (*s3.GetObjectOutput, error) {
	full := *in
	full.Range = nil
	return a.S3API.GetObject(ctx, &full, opts...)
}

// TestReadRangeRejectsRangeIgnoringEndpoint: a 200-whole-object answer to a
// ranged GET must be one loud error, never the segment's FIRST n bytes
// silently handed back as [off, off+n).
func TestReadRangeRejectsRangeIgnoringEndpoint(t *testing.T) {
	base := fakeS3(t, "kvb-test")
	cfg := Config{Bucket: "kvb-test", NodeID: "n", OpTimeout: 10 * time.Second}
	sp := NewSpiller(base, cfg, 2)
	defer sp.Close()
	body := segBody(64<<10, 0x66)
	done := make(chan bool, 1)
	sp.DemoteSegment(9, int64(len(body)),
		func() (io.ReadSeekCloser, error) { return nopSeekCloser{bytes.NewReader(body)}, nil },
		func(_ uint64, up bool) { done <- up })
	if !<-done {
		t.Fatal("upload failed")
	}

	r := NewRestorer(rangeIgnoringAPI{base}, cfg)
	dst := make([]byte, 512)
	err := r.ReadRange(context.Background(), 9, 1024, 512, dst)
	if err == nil {
		t.Fatal("Range-ignoring endpoint went undetected: offset-0 bytes would be served as the requested range")
	}
}

// TestNewClientValidatesConfig: key-shape and region mistakes must fail at
// startup, not as a baffling fleet-wide symptom at first spill.
func TestNewClientValidatesConfig(t *testing.T) {
	ctx := context.Background()
	if _, err := NewClient(ctx, Config{Bucket: "b", NodeID: "", Region: "us-east-1"}); err == nil {
		t.Fatal("empty NodeID accepted: two blank-id nodes sharing a bucket collide on segment keys")
	}
	if _, err := NewClient(ctx, Config{Bucket: "b", NodeID: "a/b", Region: "us-east-1"}); err == nil {
		t.Fatal("slash in NodeID accepted: key layout no longer listable by node prefix")
	}
	if _, err := NewClient(ctx, Config{Bucket: "b", NodeID: "n", Region: ""}); err == nil {
		t.Fatal("empty Region without endpoint override accepted: fails deep inside the SDK at first call")
	}
	if _, err := NewClient(ctx, Config{Bucket: "b", NodeID: "n", Region: "", EndpointOverride: "http://127.0.0.1:1"}); err != nil {
		t.Fatalf("endpoint-override config without Region rejected: %v", err)
	}
}

// TestRestorerCloseDrainsDetachedDownloads: RestoreSegment's ctx cancels
// the WAIT, not the shared download — the sink can still be running after
// the caller (and its latch) walked away. Close is the shutdown barrier:
// it must refuse new restores and BLOCK until the detached sink finished,
// or a mid-teardown sink writes into a store being closed underneath it.
func TestRestorerCloseDrainsDetachedDownloads(t *testing.T) {
	base := fakeS3(t, "kvb-test")
	cfg := Config{Bucket: "kvb-test", NodeID: "n", OpTimeout: 10 * time.Second}
	sp := NewSpiller(base, cfg, 2)
	defer sp.Close()
	body := segBody(4096, 0x55)
	uploaded := make(chan bool, 1)
	sp.DemoteSegment(12, int64(len(body)),
		func() (io.ReadSeekCloser, error) { return nopSeekCloser{bytes.NewReader(body)}, nil },
		func(_ uint64, up bool) { uploaded <- up })
	if !<-uploaded {
		t.Fatal("upload failed")
	}

	api := &gatedGetAPI{S3API: base, started: make(chan struct{}, 1), gate: make(chan struct{})}
	r := NewRestorer(api, cfg)

	ctx, cancel := context.WithCancel(context.Background())
	var sinkDone atomic.Bool
	waitErr := make(chan error, 1)
	go func() {
		waitErr <- r.RestoreSegment(ctx, 12, func(rd io.Reader) error {
			_, err := io.Copy(io.Discard, rd)
			sinkDone.Store(true)
			return err
		})
	}()
	<-api.started // the detached download is registered and gated
	cancel()      // the caller gives up — the belt-cut shape
	if err := <-waitErr; err == nil {
		t.Fatal("cancelled RestoreSegment returned nil — fixture broke")
	}

	closed := make(chan struct{})
	go func() {
		r.Close()
		close(closed)
	}()
	select {
	case <-closed:
		t.Fatal("Close returned with the detached download still gated — nothing drains stragglers")
	case <-time.After(100 * time.Millisecond):
	}
	close(api.gate) // the download proceeds; Close must now come home
	select {
	case <-closed:
	case <-time.After(5 * time.Second):
		t.Fatal("Close never returned after the download finished")
	}
	if !sinkDone.Load() {
		t.Fatal("Close returned before the detached sink completed")
	}
	if err := r.RestoreSegment(context.Background(), 12, func(io.Reader) error { return nil }); err == nil {
		t.Fatal("RestoreSegment accepted work after Close")
	}
}
