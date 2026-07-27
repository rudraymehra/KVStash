package store

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http/httptest"
	"sync"
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
	// Default: a tiny volume budget so reclaim genuinely fires (the retire-
	// flip → cold-read walk needs it). Tests that want spilled-but-LIVE
	// segments (no reclaim) pass a large budget via newS3FixtureSized.
	return newS3FixtureSized(t, 768<<10)
}

func newS3FixtureSized(t *testing.T, volMaxBytes int64) *s3Fixture {
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
	sp := s3spill.NewSpiller(api, cfg, 32)
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
		Dir: t.TempDir(), SegmentBytes: 256 << 10, MaxBytes: volMaxBytes,
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
// pump runs ONE deterministic tier cycle: lapse leases, demote to NVMe,
// enqueue spills, WAIT for the async spiller to ack them (Flush — so the
// S3 flags are settled), THEN reclaim (which now FLIPS spilled segments
// instead of racing the ack and deleting them). Replacing the old
// wall-clock/sleep drive with a Flush barrier is what makes these tests
// deterministic on a slow, contended CI runner (the async spiller missing
// a 15s deadline was the CI-only flake).
func (fx *s3Fixture) pump() {
	fx.cur.Add(int64(200 * time.Millisecond))
	fx.t.demotePass(true)
	fx.t.spillPass()
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	fx.sp.Flush(ctx)
	cancel()
	fx.t.reclaimPass()
}

func driveToS3Residency(t *testing.T, fx *s3Fixture) (key [32]byte, blob []byte) {
	t.Helper()
	const blk, total = 60 << 10, 100
	blobs := map[int][]byte{}
	for i := 0; i < total && fx.q.Usage(1, tenant.TierS3) == 0; i++ {
		b := bytes.Repeat([]byte{byte(i), byte(0xA0 ^ i)}, blk/2) //nolint:gosec // G115: test payload pattern
		blobs[i] = b
		st := fx.t.Put(1, s3key(i), b, xxh3.Hash(b))
		if st == protocol.StatusErrQuotaBytes {
			fx.pump()
			st = fx.t.Put(1, s3key(i), b, xxh3.Hash(b))
		}
		if st != protocol.StatusOK {
			t.Fatalf("put %d: %s", i, st)
		}
		if i%4 == 3 {
			fx.pump()
		}
	}
	// A few extra cycles to flip the last spilled segments (bounded, no wall
	// clock — Flush inside pump is the barrier, so this converges even on a
	// 1-CPU runner).
	for k := 0; k < 40 && fx.q.Usage(1, tenant.TierS3) == 0; k++ {
		fx.pump()
	}
	if fx.q.Usage(1, tenant.TierS3) == 0 {
		spilled, dropped, putErrs := fx.sp.Stats()
		t.Fatalf("no segment reached s3-residency (spilled=%d dropped=%d errs=%d)", spilled, dropped, putErrs)
	}
	for i := 0; i < total; i++ {
		data, _, rel, tier, st := fx.t.GetRefTier(context.Background(), 1, s3key(i))
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

// TestS3FlipStrandAbortsRetire is the deterministic white-box pin for the
// strand-abort branch (it originally shipped with zero executing coverage —
// only the probabilistic ack-vs-reclaim hunt touched it, and that never
// fires on a quiet box). It drives a segment to
// sealed+spilled+LIVE (demote + spill, NO reclaim), soft-pins one of its
// entries, then calls s3FlipRetired directly — sidestepping the advisory
// pre-gate exactly as a lease/pin landing mid-retire would. The pinned entry
// must STRAND the retire (a protected block may never become s3-only), stay
// un-flipped, and keep serving locally after RetireAbort.
func TestS3FlipStrandAbortsRetire(t *testing.T) {
	// A LARGE volume budget (8 MiB) so reclaim NEVER fires — segments seal
	// and spill but stay live, which is the state this test needs (the
	// default fixture's tiny budget would reclaim-flip them instead).
	fx := newS3FixtureSized(t, 8<<20)
	const blk = 60 << 10
	// Continuous 60K-block flow through the 1.5 MiB arena: it must actually
	// fill past the 90% demote watermark (the arena is not tiny), so this
	// takes ~40 blocks with a hand-driven demote whenever the arena walls.
	for i := 0; i < 40; i++ {
		b := bytes.Repeat([]byte{byte(i), byte(0x5A ^ i)}, blk/2) //nolint:gosec // G115: test payload
		st := fx.t.Put(1, s3key(i), b, xxh3.Hash(b))
		if st == protocol.StatusErrQuotaBytes {
			fx.cur.Add(int64(200 * time.Millisecond))
			fx.t.demotePass(true)
			st = fx.t.Put(1, s3key(i), b, xxh3.Hash(b))
		}
		if st != protocol.StatusOK {
			t.Fatalf("put %d: %s", i, st)
		}
		if i%4 == 3 {
			fx.cur.Add(int64(200 * time.Millisecond))
			fx.t.demotePass(true)
		}
	}
	// The spiller is ASYNC: spillPass enqueues, then Flush WAITS for the acks
	// so S3=true is settled — deterministic, no wall-clock race (the CI-only
	// flake). This fixture never reclaims (large volume), so spilled segments
	// stay live and their entries accumulate the S3 flag.
	var segID uint32
	var keys [][32]byte
	for k := 0; k < 40 && len(keys) < 2; k++ {
		fx.t.spillPass()
		ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
		fx.sp.Flush(ctx)
		cancel()
		bySeg := map[uint32][][32]byte{}
		fx.t.idx.rangeAll(func(k dram.Key, ref *nvmeRef) bool {
			if ref.S3.Load() && !ref.S3Only.Load() {
				bySeg[ref.Loc.SegmentID] = append(bySeg[ref.Loc.SegmentID], k.Hash)
			}
			return true
		})
		for id, ks := range bySeg {
			if len(ks) >= 2 {
				segID, keys = id, ks
				break
			}
		}
		if len(keys) < 2 { // seal more if the fill didn't produce ≥2 yet
			fx.cur.Add(int64(200 * time.Millisecond))
			fx.t.demotePass(true)
		}
	}
	if len(keys) < 2 {
		t.Fatalf("could not stage a spilled+live segment with ≥2 entries (got %d)", len(keys))
	}
	vol := fx.t.volumeFor(keys[0])
	if !vol.IsSpilled(segID) {
		t.Fatalf("segment %d not spilled", segID)
	}

	// Soft-pin one entry: it must strand the retire (protection is the
	// authoritative gate INSIDE the flip closure, not just the pre-gate).
	pin := dram.Key{NS: 1, Hash: keys[0]}
	fx.t.idx.withShardLock(pin, func(ref *nvmeRef) {
		if ref == nil {
			t.Fatal("pinned key vanished before the flip")
		}
		ref.PinFlags = nvPinSoftBit
	})

	if !vol.RetireBegin(segID) {
		t.Fatalf("RetireBegin(%d) refused", segID)
	}
	preS3 := fx.q.Usage(1, tenant.TierS3)
	stranded := fx.t.s3FlipRetired(vol, segID)
	if stranded == 0 {
		t.Fatal("a soft-pinned entry did not strand the retire — a protected block could become s3-only")
	}
	// The pinned entry must NOT have flipped.
	fx.t.idx.withShardLock(pin, func(ref *nvmeRef) {
		if ref == nil || ref.S3Only.Load() {
			t.Fatal("pinned entry was flipped to s3-only despite stranding")
		}
	})
	vol.RetireAbort(segID)

	// After the abort the segment is live again: the pinned block still
	// serves LOCALLY (never from S3), byte-intact.
	_, _, rel, tier, st := fx.t.GetRefTier(context.Background(), 1, keys[0])
	if st != protocol.StatusOK {
		t.Fatalf("pinned block unreadable after strand-abort: %s", st)
	}
	if tier != "nvme" {
		t.Fatalf("pinned block served from %q, want local nvme after abort", tier)
	}
	rel()
	// The stranded flip must not have leaked its tenant charge onto S3 for
	// the pinned entry (any earlier-walked entry that legitimately flipped
	// before the strand may have moved, so this only bounds the direction).
	if got := fx.q.Usage(1, tenant.TierS3); got < preS3 {
		t.Fatalf("S3 usage went backwards across a stranded flip: %d < %d", got, preS3)
	}
}

// tripwireRestore panics on ANY backend call — the EXISTS guard's teeth.
type tripwireRestore struct{}

func (tripwireRestore) ReadRange(context.Context, uint64, int64, int64, []byte) error {
	panic("EXISTS (or another index-only path) touched the S3 backend")
}

func (tripwireRestore) RestoreSegment(context.Context, uint64, func(io.Reader) error) error {
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
	// and seal, the 768 KiB volume budget forces reclaim, and pump() spills
	// + FLUSHES before reclaiming — so reclaim flips (not deletes) the
	// spilled segments deterministically, even on a slow CI runner.
	for i := 0; i < total && fx.q.Usage(1, tenant.TierS3) == 0; i++ {
		b := bytes.Repeat([]byte{byte(i), byte(0xA0 ^ i)}, blk/2) //nolint:gosec // G115: test payload pattern
		blobs[i] = b
		st := fx.t.Put(1, s3key(i), b, xxh3.Hash(b))
		if st == protocol.StatusErrQuotaBytes {
			fx.pump() // arena wall: demote→spill→flush→reclaim, then retry once
			st = fx.t.Put(1, s3key(i), b, xxh3.Hash(b))
		}
		if st != protocol.StatusOK {
			t.Fatalf("put %d: %s", i, st)
		}
		if i%4 == 3 {
			fx.pump()
		}
	}
	// A few extra cycles to flip the last spilled segments (bounded, Flush is
	// the barrier — converges without a wall-clock deadline).
	for k := 0; k < 40 && fx.q.Usage(1, tenant.TierS3) == 0; k++ {
		fx.pump()
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
		data, sum, rel, tier, st := fx.t.GetRefTier(context.Background(), 1, s3key(i))
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
	data, _, rel, tier, st := fx.t.GetRefTier(context.Background(), 1, key)
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

// TestS3BreakerContainsOutage pins the circuit breaker: after
// s3BreakerTripAfter consecutive ReadRange failures the breaker opens —
// further cold reads answer NOT_FOUND WITHOUT touching the backend (no new
// read errors, skips counted), and after the cooldown exactly one probe per
// interval re-tests the endpoint. Before the breaker, every cold key on
// every connection paid the full deadline for the whole outage.
func TestS3BreakerContainsOutage(t *testing.T) {
	fx := newS3Fixture(t)
	key, _ := driveToS3Residency(t, fx)
	fx.srv.Close() // outage: every ReadRange now fails fast

	for i := 0; i < s3BreakerTripAfter; i++ {
		if _, _, _, _, st := fx.t.GetRefTier(context.Background(), 1, key); st != protocol.StatusNotFound {
			t.Fatalf("outage read %d: %s, want NOT_FOUND", i, st)
		}
	}
	if got := fx.t.s3BreakerOpens.Load(); got != 1 {
		t.Fatalf("breaker opens = %d after the trip streak, want 1", got)
	}

	// Open breaker: contained — no backend traffic, no new read errors.
	preErrs, preSkips := fx.t.s3ReadErrs.Load(), fx.t.s3BreakerSkips.Load()
	for i := 0; i < 5; i++ {
		if _, _, _, _, st := fx.t.GetRefTier(context.Background(), 1, key); st != protocol.StatusNotFound {
			t.Fatalf("breaker-open read %d: %s, want NOT_FOUND", i, st)
		}
	}
	if got := fx.t.s3ReadErrs.Load(); got != preErrs {
		t.Fatalf("breaker open but reads still hit the backend: errors %d → %d", preErrs, got)
	}
	if got := fx.t.s3BreakerSkips.Load(); got != preSkips+5 {
		t.Fatalf("breaker skips %d → %d, want +5", preSkips, got)
	}

	// Cooldown lapse: exactly ONE probe goes through per interval (it fails
	// — the endpoint is still dead — and re-opens the breaker).
	fx.cur.Add(int64(s3BreakerCooldown) + 1)
	preErrs = fx.t.s3ReadErrs.Load()
	for i := 0; i < 3; i++ {
		_, _, _, _, _ = fx.t.GetRefTier(context.Background(), 1, key)
	}
	if got := fx.t.s3ReadErrs.Load(); got != preErrs+1 {
		t.Fatalf("post-cooldown probes hit the backend %d times, want exactly 1", got-preErrs)
	}
}

// TestS3DeadObjectDoesNotOpenBreaker pins the breaker's own invariant —
// ENDPOINT health, never OBJECT health: a full trip-streak of reads against
// an externally-deleted object (a misconfigured bucket lifecycle rule is the
// realistic shape) answers per-key NOT_FOUND on its own counter, never opens
// the breaker, and never rides the endpoint-error counter — partial object
// loss must not amplify into a self-sustaining cold-tier outage.
func TestS3DeadObjectDoesNotOpenBreaker(t *testing.T) {
	fx := newS3Fixture(t)
	key, _ := driveToS3Residency(t, fx)
	ref := fx.t.idx.get(dram.Key{NS: 1, Hash: key})
	if ref == nil {
		t.Fatal("served key has no index entry")
	}
	objKey := fmt.Sprintf("kvblockd/fx/segments/seg-%08d.seg", ref.Loc.SegmentID)
	if _, err := fx.backend.DeleteObject("kvb-tier", objKey); err != nil {
		t.Fatalf("delete segment object: %v", err)
	}

	preErrs := fx.t.s3ReadErrs.Load()
	const reads = 2 * s3BreakerTripAfter
	for i := 0; i < reads; i++ {
		if _, _, _, _, st := fx.t.GetRefTier(context.Background(), 1, key); st != protocol.StatusNotFound {
			t.Fatalf("dead-object read %d: %s, want NOT_FOUND", i, st)
		}
	}
	if got := fx.t.s3BreakerOpens.Load(); got != 0 {
		t.Fatalf("a dead OBJECT opened the breaker (%d opens) — every healthy cold key just blackholed for the cooldown", got)
	}
	if got := fx.t.s3ReadNotFound.Load(); got != reads {
		t.Fatalf("dead-object reads on the wrong counter: read_not_found_total=%d, want %d", got, reads)
	}
	if got := fx.t.s3ReadErrs.Load(); got != preErrs {
		t.Fatalf("dead object misattributed to endpoint sickness: read errors %d → %d", preErrs, got)
	}
}

// cancellingRestore simulates a client that disconnects MID-read: ReadRange
// cancels the caller's context, then fails with its error — the exact shape
// a server drain or a dropped connection produces under the ctx plumb.
type cancellingRestore struct{ cancel context.CancelFunc }

func (c *cancellingRestore) ReadRange(ctx context.Context, _ uint64, _, _ int64, _ []byte) error {
	c.cancel()
	<-ctx.Done()
	return ctx.Err()
}

func (c *cancellingRestore) RestoreSegment(context.Context, uint64, func(io.Reader) error) error {
	return fmt.Errorf("cancelling backend")
}
func (c *cancellingRestore) Stats() (uint64, uint64) { return 0, 0 }

// TestS3CancelledColdReadIsNotAnEndpointError pins the taxonomy leg the ctx
// plumb introduced: a caller-cancelled cold read (client disconnect, server
// drain) is neither an endpoint failure nor breaker food — read_errors_total
// stays flat and the breaker stays closed through a full trip-streak of
// cancellations, so a routine Drain can never page as endpoint sickness.
func TestS3CancelledColdReadIsNotAnEndpointError(t *testing.T) {
	fx := newS3Fixture(t)
	key, _ := driveToS3Residency(t, fx)

	preErrs := fx.t.s3ReadErrs.Load()
	for i := 0; i < s3BreakerTripAfter; i++ {
		ctx, cancel := context.WithCancel(context.Background())
		fx.t.p.Restore = &cancellingRestore{cancel: cancel}
		_, _, _, _, st := fx.t.GetRefTier(ctx, 1, key)
		cancel()
		if st != protocol.StatusNotFound {
			t.Fatalf("cancelled cold read %d: %s, want NOT_FOUND", i, st)
		}
	}
	if got := fx.t.s3ReadErrs.Load(); got != preErrs {
		t.Fatalf("client disconnects counted as S3 read errors: %d → %d", preErrs, got)
	}
	if got := fx.t.s3BreakerOpens.Load(); got != 0 {
		t.Fatalf("breaker opened on caller cancellations: %d opens", got)
	}
}

// TestPinHardBreakerOpenAnswersBusy pins the PIN_HARD status mapping while
// the breaker is open: the block EXISTS and only its tier's endpoint is sick,
// so the promotion must answer retryable ERR_BUSY — NOT_FOUND here would
// break the "a pin refusing NOT_FOUND would be a lie" contract for a whole
// cooldown window.
func TestPinHardBreakerOpenAnswersBusy(t *testing.T) {
	fx := newS3Fixture(t)
	key, _ := driveToS3Residency(t, fx)
	fx.srv.Close() // outage: every ReadRange now fails fast
	for i := 0; i < s3BreakerTripAfter; i++ {
		_, _, _, _, _ = fx.t.GetRefTier(context.Background(), 1, key)
	}
	if got := fx.t.s3BreakerOpens.Load(); got != 1 {
		t.Fatalf("breaker opens = %d after the trip streak, want 1", got)
	}
	if st := fx.t.PinOp(1, key, protocol.PinHard); st != protocol.StatusErrBusy {
		t.Fatalf("PIN_HARD with the breaker open: %s, want ERR_BUSY (the block exists; the endpoint is sick)", st)
	}
}

// parkedRestore parks every ReadRange (announcing on entered) until gate
// closes, then fails — the seam for holding cold reads in flight.
type parkedRestore struct {
	entered chan struct{}
	gate    chan struct{}
}

func (p *parkedRestore) ReadRange(context.Context, uint64, int64, int64, []byte) error {
	p.entered <- struct{}{}
	<-p.gate
	return fmt.Errorf("parked backend: read failed")
}

func (p *parkedRestore) RestoreSegment(context.Context, uint64, func(io.Reader) error) error {
	return fmt.Errorf("parked backend")
}
func (p *parkedRestore) Stats() (uint64, uint64) { return 0, 0 }

// TestS3ColdReadConcurrencyBound pins the semaphore: with every slot parked
// in a slow backend read, the next cold GET answers ERR_BUSY (retryable)
// instead of opening yet another backend read — N connections can no longer
// fan an outage into N stalled goroutines.
func TestS3ColdReadConcurrencyBound(t *testing.T) {
	fx := newS3Fixture(t)
	key, _ := driveToS3Residency(t, fx)
	park := &parkedRestore{
		entered: make(chan struct{}, 2*s3ColdReadConcurrency),
		gate:    make(chan struct{}),
	}
	fx.t.p.Restore = park

	var wg sync.WaitGroup
	for i := 0; i < s3ColdReadConcurrency; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, _, _, _, _ = fx.t.GetRefTier(context.Background(), 1, key)
		}()
	}
	for i := 0; i < s3ColdReadConcurrency; i++ {
		select {
		case <-park.entered:
		case <-time.After(10 * time.Second):
			t.Fatalf("only %d of %d cold reads reached the backend", i, s3ColdReadConcurrency)
		}
	}

	// Every slot held: the overflow read must bounce as ERR_BUSY without a
	// 17th backend call.
	preBusy := fx.t.s3ReadBusy.Load()
	_, _, _, _, st := fx.t.GetRefTier(context.Background(), 1, key)
	if st != protocol.StatusErrBusy {
		t.Fatalf("cold read past the bound: %s, want ERR_BUSY", st)
	}
	if got := fx.t.s3ReadBusy.Load(); got != preBusy+1 {
		t.Fatalf("read_busy %d → %d, want +1", preBusy, got)
	}
	select {
	case <-park.entered:
		t.Fatal("the bounded-out read still reached the backend")
	default:
	}
	close(park.gate)
	wg.Wait()
}

// hangDropSpill delegates everything to the real spiller except Drop, which
// hangs until its ctx dies — the S3 endpoint that accepts connections and
// never answers DeleteObject.
type hangDropSpill struct {
	inner   SpillBackend
	entered chan struct{}
}

func (h *hangDropSpill) DemoteSegment(segID uint64, size int64, open func() (io.ReadSeekCloser, error), onUp func(uint64, bool)) bool {
	return h.inner.DemoteSegment(segID, size, open, onUp)
}

func (h *hangDropSpill) Drop(ctx context.Context, _ uint64) error {
	select {
	case h.entered <- struct{}{}:
	default:
	}
	<-ctx.Done()
	return ctx.Err()
}

func (h *hangDropSpill) Stats() (uint64, uint64, uint64) { return h.inner.Stats() }

// TestHungObjectDropDoesNotStallDemoter pins the GC decoupling: with the
// drop worker wedged in a DeleteObject that never answers, the demote/
// reclaim ticker must keep its cadence — fresh pressure still demotes.
// Before the split, gcPass ran inline on the demoter goroutine and one hung
// endpoint froze the whole memory ladder for the drop deadline (demotions
// became evictions exactly under DRAM pressure).
func TestHungObjectDropDoesNotStallDemoter(t *testing.T) {
	fx := newS3Fixture(t)
	_, _ = driveToS3Residency(t, fx)

	// Free every s3-resident block: the last removal of a segment nominates
	// its object for GC (force: cold reads just auto-leased them).
	var coldKeys [][32]byte
	fx.t.idx.rangeAll(func(k dram.Key, ref *nvmeRef) bool {
		if ref.S3Only.Load() {
			coldKeys = append(coldKeys, k.Hash)
		}
		return true
	})
	if len(coldKeys) == 0 {
		t.Fatal("no s3-resident keys to free")
	}
	for _, k := range coldKeys {
		if st := fx.t.Delete(1, k, true); st != protocol.StatusOK {
			t.Fatalf("delete cold key: %s", st)
		}
	}
	if len(fx.t.dropq) == 0 {
		t.Fatal("no GC candidate nominated")
	}

	hang := &hangDropSpill{inner: fx.t.p.Spill, entered: make(chan struct{}, 1)}
	fx.t.p.Spill = hang
	fx.t.p.S3ReadTimeout = 30 * time.Second // the drop deadline a stalled ladder would ride out
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	stop := fx.t.Start(ctx)
	defer stop()

	select {
	case <-hang.entered: // the drop worker is now wedged
	case <-time.After(10 * time.Second):
		t.Fatal("GC never attempted the drop")
	}

	// Fresh demote pressure with the drop wedged: the ticker must move it.
	preDem := fx.t.demotions.Load()
	for i := 0; i < 30; i++ {
		b := bytes.Repeat([]byte{byte(i), 0x77}, 30<<10) //nolint:gosec // G115: test payload
		_ = fx.t.Put(1, s3key(7000+i), b, xxh3.Hash(b))  // quota refusals under pressure are fine
	}
	deadline := time.Now().Add(10 * time.Second)
	for fx.t.demotions.Load() == preDem {
		if time.Now().After(deadline) {
			t.Fatal("demoter made no progress while the object drop hung — the ladder stalled behind S3")
		}
		fx.cur.Add(int64(200 * time.Millisecond))
		time.Sleep(10 * time.Millisecond)
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
	_, _, rel, _, st := fx.t.GetRefTier(context.Background(), 1, key)
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
