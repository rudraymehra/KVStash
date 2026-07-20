package store

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
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
// hand, deterministically. backend and srv are exposed so the verify-path
// tests can corrupt an object in place or kill the transport.
type s3Fixture struct {
	t       *Tiered
	q       *tenant.Quotas
	sp      *s3spill.Spiller
	cur     *atomic.Int64
	backend *s3mem.Backend
	srv     *httptest.Server
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
	awsCfg, err := awsconfig.LoadDefaultConfig(
		context.Background(),
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
	fx := &s3Fixture{t: tt, q: q, sp: sp, cur: cur, backend: backend, srv: srv}
	t.Cleanup(func() { _ = tt.Close() })
	return fx
}

// driveToS3Residency fills the fixture and hand-drives the ladder until at
// least one segment is retire-flipped, then returns a key currently SERVED
// from the cold tier (proving the read path before anyone breaks it).
func driveToS3Residency(t *testing.T, fx *s3Fixture) (key [32]byte, blob []byte) {
	t.Helper()
	const blk, total = 60 << 10, 100
	blobs := map[int][]byte{}
	for i := 0; i < total; i++ {
		b := bytes.Repeat([]byte{byte(i), byte(0xA0 ^ i)}, blk/2) //nolint:gosec // G115: test payload pattern
		blobs[i] = b
		st := fx.t.Put(1, s3key(i), b, xxh3.Hash(b))
		if st == protocol.StatusErrQuotaBytes {
			fx.cur.Add(int64(200 * time.Millisecond))
			fx.t.DemoteNow()
			st = fx.t.Put(1, s3key(i), b, xxh3.Hash(b))
		}
		if st != protocol.StatusOK {
			t.Fatalf("put %d: %s", i, st)
		}
		if i%4 == 3 {
			fx.cur.Add(int64(200 * time.Millisecond))
			fx.t.DemoteNow()
		}
	}
	deadline := time.Now().Add(15 * time.Second)
	for time.Now().Before(deadline) && fx.q.Usage(1, tenant.TierS3) == 0 {
		fx.cur.Add(int64(100 * time.Millisecond))
		fx.t.DemoteNow()
		time.Sleep(20 * time.Millisecond)
	}
	if fx.q.Usage(1, tenant.TierS3) == 0 {
		t.Fatal("no segment reached s3-residency")
	}
	for i := 0; i < total; i++ {
		data, _, rel, tier, st := fx.t.GetRefTier(1, s3key(i))
		if st != protocol.StatusOK {
			continue
		}
		served := tier == "s3" && bytes.Equal(data, blobs[i])
		rel()
		if served {
			return s3key(i), blobs[i]
		}
	}
	t.Fatal("no block served from the cold tier")
	return [32]byte{}, nil
}

// tripwireRestore panics on ANY backend call — the EXISTS guard's teeth.
type tripwireRestore struct{}

func (tripwireRestore) ReadRange(context.Context, uint64, int64, int64, []byte) error {
	panic("EXISTS (or another index-only path) touched the S3 backend")
}
func (tripwireRestore) Stats() (uint64, uint64) { return 0, 0 }

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
	firstS3 := -1
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
			if firstS3 < 0 {
				firstS3 = i
			}
		}
		rel()
	}
	if s3Served == 0 {
		t.Fatalf("no block served from the cold tier after a retire-flip (s3Hits=%d s3ReadErrs=%d s3ChecksumErrs=%d s3usage=%d)",
			fx.t.s3Hits.Load(), fx.t.s3ReadErrs.Load(), fx.t.s3ChecksumErrs.Load(), fx.q.Usage(1, tenant.TierS3))
	}

	// EXISTS never touches S3: swap in a Restore backend that PANICS on any
	// read — the index must answer for s3-resident keys without a byte of
	// backend traffic, and the answer itself must be a hit (not a silent
	// miss that never needed the backend). Restored after, before the
	// delete stanza below cold-reads again.
	realRestore := fx.t.p.Restore
	fx.t.p.Restore = tripwireRestore{}
	if n, _ := fx.t.ExistsPrefix(1, [][32]byte{s3key(firstS3)}, false); n != 1 {
		t.Fatalf("EXISTS on an s3-resident key = %d, want 1 (index-truth)", n)
	}
	allKeys := make([][32]byte, total)
	for i := range allKeys {
		allKeys[i] = s3key(i)
	}
	if _, perKey := fx.t.ExistsPrefix(1, allKeys, true); len(perKey) != total {
		t.Fatalf("bitmap EXISTS answered %d/%d", len(perKey), total)
	}
	fx.t.p.Restore = realRestore

	// Stats: the "s3" sub-document is live and its residency matches the
	// quota ledger — the tier split the scrape collector consumes.
	var doc struct {
		S3 *struct {
			Blocks  int64  `json:"blocks"`
			Bytes   int64  `json:"bytes"`
			Spilled uint64 `json:"spilled_segments_total"`
			Hits    uint64 `json:"hits_total"`
		} `json:"s3"`
	}
	if err := json.Unmarshal(fx.t.Stats(), &doc); err != nil {
		t.Fatal(err)
	}
	if doc.S3 == nil || doc.S3.Blocks == 0 || doc.S3.Spilled == 0 || doc.S3.Hits == 0 {
		t.Fatalf("s3 stats sub-doc missing or zero: %+v", doc.S3)
	}
	if doc.S3.Bytes != fx.q.Usage(1, tenant.TierS3) {
		t.Fatalf("s3 stats bytes %d != quota ledger %d", doc.S3.Bytes, fx.q.Usage(1, tenant.TierS3))
	}

	// DELETE of an s3-resident block refunds the S3 side of the ledger —
	// refunding NVMe here would leak the S3 charge forever (and the stats
	// residency with it). force=true: the cold read just auto-leased it.
	preS3 := fx.q.Usage(1, tenant.TierS3)
	preNvme := fx.q.Usage(1, tenant.TierNVMe)
	if st := fx.t.Delete(1, s3key(firstS3), true); st != protocol.StatusOK {
		t.Fatalf("delete s3-resident block %d: %s", firstS3, st)
	}
	if got := fx.q.Usage(1, tenant.TierS3); got != preS3-blk {
		t.Fatalf("s3 usage after delete: %d, want %d", got, preS3-blk)
	}
	if got := fx.q.Usage(1, tenant.TierNVMe); got != preNvme {
		t.Fatalf("nvme usage changed by an s3-resident delete: %d → %d", preNvme, got)
	}
	if err := json.Unmarshal(fx.t.Stats(), &doc); err != nil {
		t.Fatal(err)
	}
	if doc.S3.Bytes != preS3-blk {
		t.Fatalf("s3 stats bytes after delete: %d, want %d", doc.S3.Bytes, preS3-blk)
	}
}

// TestS3ColdReadRefusesCorruptObject overwrites the segment OBJECT in place
// (same key, same size, garbage bytes) and proves the verify-before-serve
// contract: the cold read answers NOT_FOUND, never the wrong bytes, and the
// failure lands on the S3-side checksum counter — not the device's.
func TestS3ColdReadRefusesCorruptObject(t *testing.T) {
	fx := newS3Fixture(t)
	key, blob := driveToS3Residency(t, fx)

	ref := fx.t.idx.get(dram.Key{NS: 1, Hash: key})
	if ref == nil {
		t.Fatal("served key has no index entry")
	}
	objKey := fmt.Sprintf("kvblockd/fx/segments/seg-%08d.seg", ref.Loc.SegmentID)
	obj, err := fx.backend.GetObject("kvb-tier", objKey, nil)
	if err != nil {
		t.Fatalf("segment object missing: %v", err)
	}
	_ = obj.Contents.Close()
	rot := bytes.Repeat([]byte{0xDE}, int(obj.Size))
	if _, err := fx.backend.PutObject("kvb-tier", objKey, map[string]string{}, bytes.NewReader(rot), obj.Size, nil); err != nil {
		t.Fatal(err)
	}

	preS3 := fx.t.s3ChecksumErrs.Load()
	preNvme := fx.t.checksumErrs.Load()
	data, _, rel, tier, st := fx.t.GetRefTier(1, key)
	if st == protocol.StatusOK {
		same := bytes.Equal(data, blob)
		rel()
		t.Fatalf("corrupt object served (tier=%s, bytes-match=%v) — verification is off", tier, same)
	}
	if st != protocol.StatusNotFound {
		t.Fatalf("corrupt cold read answered %s, want NOT_FOUND", st)
	}
	if got := fx.t.s3ChecksumErrs.Load(); got != preS3+1 {
		t.Fatalf("s3 checksum errors %d → %d, want +1", preS3, got)
	}
	if got := fx.t.checksumErrs.Load(); got != preNvme {
		t.Fatalf("object rot misattributed to the DEVICE counter (%d → %d)", preNvme, got)
	}
}

// TestS3ColdReadTransportErrorIsMiss kills the fake S3 endpoint and proves a
// dead object store degrades to per-key misses (counted on read_errors) —
// never a hang, never an unverified byte.
func TestS3ColdReadTransportErrorIsMiss(t *testing.T) {
	fx := newS3Fixture(t)
	key, _ := driveToS3Residency(t, fx)

	fx.srv.Close() // the endpoint drops mid-flight; ReadRange now fails fast
	pre := fx.t.s3ReadErrs.Load()
	_, _, rel, _, st := fx.t.GetRefTier(1, key)
	if st == protocol.StatusOK {
		rel()
		t.Fatal("cold read served with the transport down")
	}
	if st != protocol.StatusNotFound {
		t.Fatalf("dead-transport cold read answered %s, want NOT_FOUND", st)
	}
	if got := fx.t.s3ReadErrs.Load(); got != pre+1 {
		t.Fatalf("s3 read errors %d → %d, want +1", pre, got)
	}
}
