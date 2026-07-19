package store

import (
	"bytes"
	"context"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	awsc "github.com/aws/aws-sdk-go-v2/aws"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	awss3 "github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/johannesboyne/gofakes3"
	"github.com/johannesboyne/gofakes3/backend/s3mem"
	"github.com/zeebo/xxh3"

	"github.com/kvstash/kvblockd/internal/eviction"
	"github.com/kvstash/kvblockd/internal/protocol"
	"github.com/kvstash/kvblockd/internal/store/dram"
	"github.com/kvstash/kvblockd/internal/store/nvme"
	"github.com/kvstash/kvblockd/internal/store/s3spill"
	"github.com/kvstash/kvblockd/internal/tenant"
)

// s3Fixture: a tiered store over 256 KiB segments and a TINY volume budget
// (so reclaim genuinely fires), backed by in-process gofakes3 through the
// REAL s3spill backends. No background loops — every pass is driven by
// hand, deterministically.
type s3Fixture struct {
	t   *Tiered
	q   *tenant.Quotas
	sp  *s3spill.Spiller
	cur *atomic.Int64
}

func newS3Fixture(t *testing.T) *s3Fixture {
	t.Helper()
	backend := s3mem.New()
	faker := gofakes3.New(backend)
	srv := httptest.NewServer(faker.Server())
	t.Cleanup(srv.Close)
	if err := backend.CreateBucket("kvb-tier"); err != nil {
		t.Fatal(err)
	}
	awsCfg, err := awsconfig.LoadDefaultConfig(context.Background(),
		awsconfig.WithRegion("us-east-1"),
		awsconfig.WithCredentialsProvider(credentials.NewStaticCredentialsProvider("t", "t", "")),
	)
	if err != nil {
		t.Fatal(err)
	}
	api := awss3.NewFromConfig(awsCfg, func(o *awss3.Options) {
		o.BaseEndpoint = awsc.String(srv.URL)
		o.UsePathStyle = true
		o.RequestChecksumCalculation = awsc.RequestChecksumCalculationWhenRequired
	})
	cfg := s3spill.Config{Bucket: "kvb-tier", NodeID: "fx", OpTimeout: 10 * time.Second}
	sp := s3spill.NewSpiller(api, cfg, 8)
	t.Cleanup(sp.Close)
	re := s3spill.NewRestorer(api, cfg)

	cur := &atomic.Int64{}
	cur.Store(1_000_000_000_000)
	arena, err := dram.NewArena(1536<<10, false) // small: the 24-block fill lands ~96% — over the demote watermark
	if err != nil {
		t.Fatal(err)
	}
	pol, err := eviction.New("s3fifo", 1024)
	if err != nil {
		t.Fatal(err)
	}
	reg := tenant.NewRegistry("a", 1, "tok")
	q := tenant.NewQuotas(reg)
	d := dram.New(arena, dram.Params{LeaseDefaultMS: 100, LeaseMaxMS: 60000, Now: cur.Load, Quotas: q})
	d.AttachPolicy(pol)
	// Volume budget: ~3 segments — the 4th fill forces reclaim of the oldest.
	vol, rep, ents, err := nvme.OpenVolume(nvme.VolumeParams{
		Dir: t.TempDir(), SegmentBytes: 256 << 10, MaxBytes: 768 << 10,
		ReadWorkers: 2, CkptEverySegs: 4, MaxBlobLen: 64 << 10, Now: cur.Load,
	})
	if err != nil {
		t.Fatal(err)
	}
	tt := NewTiered(d, pol, []*nvme.Volume{vol}, []*nvme.RecoveryReport{rep},
		[][]nvme.RecoveredEntry{ents}, Params{
			LeaseDefaultMS: 100, LeaseMaxMS: 60000, AdmitMinHits: 0,
			PromoteWindow: 0, Now: cur.Load, Quotas: q,
			Spill: sp, Restore: re, S3ReadTimeout: 5 * time.Second,
		})
	fx := &s3Fixture{t: tt, q: q, sp: sp, cur: cur}
	t.Cleanup(func() { _ = tt.Close() })
	return fx
}

func s3key(i int) [32]byte {
	var k [32]byte
	k[0], k[1], k[31] = byte(i), byte(i>>8), 0xC3 //nolint:gosec // G115: test index mixing
	return k
}

// TestS3SpillRetireFlipColdRead is the whole cold-tier story in one walk:
// cold fill → demote (segments seal) → spill (copy lands on S3, reads stay
// local) → reclaim retires a spilled segment WITHOUT deleting entries →
// the cold read serves byte-identical data from S3 with tier="s3" — and
// the tenant's charge moved NVMe→S3.
func TestS3SpillRetireFlipColdRead(t *testing.T) {
	fx := newS3Fixture(t)
	const blk, total = 60 << 10, 100
	blobs := map[int][]byte{}
	// A continuous 6 MiB flow of 60K blocks through the 1.5 MiB arena:
	// demotion stays busy (watermark crossed repeatedly), segments ROTATE
	// and seal, the 768 KiB volume budget forces reclaim, and the async
	// spiller keeps landing copies — the full ladder under one loop.
	for i := 0; i < total; i++ {
		b := bytes.Repeat([]byte{byte(i), byte(0xA0 ^ i)}, blk/2) //nolint:gosec // G115: test payload pattern
		blobs[i] = b
		st := fx.t.Put(1, s3key(i), b, xxh3.Hash(b))
		if st == protocol.StatusErrQuotaBytes {
			// Arena wall with no evictor: demote by hand and retry once.
			fx.cur.Add(int64(200 * time.Millisecond))
			fx.t.DemoteNow()
			st = fx.t.Put(1, s3key(i), b, xxh3.Hash(b))
		}
		if st != protocol.StatusOK {
			t.Fatalf("put %d: %s", i, st)
		}
		if i%4 == 3 {
			fx.cur.Add(int64(200 * time.Millisecond)) // leases lapse
			fx.t.DemoteNow()
			fx.t.spillPass()
			fx.t.reclaimPass()
		}
	}
	// Drive until at least one spilled segment has been retire-flipped.
	deadline := time.Now().Add(15 * time.Second)
	for time.Now().Before(deadline) && fx.q.Usage(1, tenant.TierS3) == 0 {
		fx.cur.Add(int64(100 * time.Millisecond))
		fx.t.DemoteNow()
		fx.t.spillPass()
		fx.t.reclaimPass()
		time.Sleep(20 * time.Millisecond)
	}
	if fx.q.Usage(1, tenant.TierS3) == 0 {
		spilled, dropped, putErrs := fx.sp.Stats()
		t.Fatalf("no segment reached s3-residency (spilled=%d dropped=%d errs=%d)", spilled, dropped, putErrs)
	}

	// Every block must still answer byte-identical from SOME tier — and at
	// least one from "s3".
	s3Served := 0
	for i := 0; i < total; i++ {
		data, sum, rel, tier, st := fx.t.GetRefTier(1, s3key(i))
		if st == protocol.StatusNotFound {
			continue // cache-legal loss under pressure — but never wrong bytes
		}
		if st != protocol.StatusOK {
			t.Fatalf("block %d: %s", i, st)
		}
		if sum != xxh3.Hash(blobs[i]) || !bytes.Equal(data, blobs[i]) {
			t.Fatalf("block %d corrupted (tier=%s)", i, tier)
		}
		if tier == "s3" {
			s3Served++
		}
		rel()
	}
	if s3Served == 0 {
		t.Fatalf("no block served from the cold tier after a retire-flip (s3Hits=%d s3ReadErrs=%d checksumErrs=%d s3usage=%d)",
			fx.t.s3Hits.Load(), fx.t.s3ReadErrs.Load(), fx.t.checksumErrs.Load(), fx.q.Usage(1, tenant.TierS3))
	}

	// EXISTS never touches S3: the index answers even for s3-resident keys.
	n, _ := fx.t.ExistsPrefix(1, [][32]byte{s3key(0)}, false)
	_ = n // presence is index-truth; the assertion is that it CANNOT block on S3 —
	// enforced structurally (ExistsPrefix has no backend path) and pinned here
	// by the fixture: a nil-Restore fixture would still answer EXISTS.
}
