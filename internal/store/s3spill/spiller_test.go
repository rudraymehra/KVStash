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
