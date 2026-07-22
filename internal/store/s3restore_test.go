package store

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/zeebo/xxh3"

	"github.com/kvstash/kvblockd/internal/eviction"
	"github.com/kvstash/kvblockd/internal/protocol"
	"github.com/kvstash/kvblockd/internal/store/dram"
	"github.com/kvstash/kvblockd/internal/store/nvme"
	"github.com/kvstash/kvblockd/internal/tenant"
)

// White-box tests for the lazy whole-segment restore and the cold-tier
// object GC. Everything is hand-driven on a fake clock against an inline
// (synchronous) recording backend — the same deterministic pump discipline
// as s3tier_test.go, no wall-clock races (CI runs this under GOMAXPROCS=1
// -race).

// recS3 is the recording cold tier: uploads land inline (onUp fires before
// DemoteSegment returns, so spillPass leaves the S3 flags settled), objects
// live in a map, every Drop is logged, and two hooks stage the failure and
// race tests — failRestore fails the download outright; onRestore runs
// mid-download, after the object bytes are fetched and before the sink
// adopts them (the delete-vs-restore window).
type recS3 struct {
	mu          sync.Mutex
	objects     map[uint64][]byte
	drops       []uint64
	failRestore bool
	onRestore   func()
}

func (b *recS3) DemoteSegment(segID uint64, size int64, open func() (io.ReadSeekCloser, error), onUp func(uint64, bool)) bool {
	r, err := open()
	if err != nil {
		onUp(segID, false)
		return true
	}
	data, rerr := io.ReadAll(r)
	_ = r.Close()
	if rerr != nil || int64(len(data)) != size {
		onUp(segID, false)
		return true
	}
	b.mu.Lock()
	b.objects[segID] = data
	b.mu.Unlock()
	onUp(segID, true)
	return true
}

func (b *recS3) Drop(_ context.Context, segID uint64) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	delete(b.objects, segID)
	b.drops = append(b.drops, segID)
	return nil
}

func (b *recS3) ReadRange(_ context.Context, segID uint64, off, n int64, dst []byte) error {
	b.mu.Lock()
	obj, ok := b.objects[segID]
	b.mu.Unlock()
	if !ok || off < 0 || n < 0 || off+n > int64(len(obj)) {
		return fmt.Errorf("recS3: no bytes for seg %d [%d,+%d)", segID, off, n)
	}
	copy(dst, obj[off:off+n])
	return nil
}

func (b *recS3) RestoreSegment(_ context.Context, segID uint64, sink func(io.Reader) error) error {
	b.mu.Lock()
	fail, hook := b.failRestore, b.onRestore
	obj, ok := b.objects[segID]
	b.mu.Unlock()
	if fail {
		return fmt.Errorf("recS3: restore failed (injected)")
	}
	if !ok {
		return fmt.Errorf("recS3: no object %d", segID)
	}
	if hook != nil {
		hook()
	}
	return sink(bytes.NewReader(obj))
}

func (b *recS3) dropCount(segID uint64) int {
	b.mu.Lock()
	defer b.mu.Unlock()
	n := 0
	for _, id := range b.drops {
		if id == segID {
			n++
		}
	}
	return n
}

func (b *recS3) hasObject(segID uint64) bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	_, ok := b.objects[segID]
	return ok
}

// recSpill / recRestore split the two backend interfaces over one recorder
// (their Stats signatures differ) — the fakeSpill/fakeRestore shape.
type (
	recSpill   struct{ *recS3 }
	recRestore struct{ *recS3 }
)

func (recSpill) Stats() (spilled, dropped, putErrors uint64) { return 0, 0, 0 }
func (recRestore) Stats() (rangedGets, restores uint64)      { return 0, 0 }

// restoreFixture mirrors s3Fixture's geometry (256 KiB segments — three
// 60 KiB blocks each — under a 768 KiB budget so reclaim genuinely flips)
// but over the inline recorder, so no Flush barrier is needed: every pump
// leaves the tier flags settled. PromoteWindow is ARMED (unlike s3Fixture):
// these tests exist to drive the 2nd-hit machinery.
type restoreFixture struct {
	t   *Tiered
	q   *tenant.Quotas
	b   *recS3
	vol *nvme.Volume
	cur *atomic.Int64
}

func newRestoreFixture(t *testing.T) *restoreFixture {
	t.Helper()
	return newRestoreFixtureSized(t, 768<<10)
}

// newRestoreFixtureSized varies the volume budget: the default (768 KiB, ~3
// segments) makes reclaim genuinely fire; a LARGE budget stages spilled-but-
// LIVE segments (no reclaim) for the stranded-retire / GC-deadness tests.
func newRestoreFixtureSized(t *testing.T, volMaxBytes int64) *restoreFixture {
	t.Helper()
	cur := &atomic.Int64{}
	cur.Store(1_000_000_000_000)
	arena, err := dram.NewArena(1536<<10, false)
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
	vol, rep, ents, err := nvme.OpenVolume(nvme.VolumeParams{
		Dir: t.TempDir(), SegmentBytes: 256 << 10, MaxBytes: volMaxBytes,
		ReadWorkers: 2, CkptEverySegs: 4, MaxBlobLen: 64 << 10, Now: cur.Load,
	})
	if err != nil {
		t.Fatal(err)
	}
	b := &recS3{objects: map[uint64][]byte{}}
	tt := NewTiered(d, pol, []*nvme.Volume{vol}, []*nvme.RecoveryReport{rep},
		[][]nvme.RecoveredEntry{ents}, Params{
			LeaseDefaultMS: 100, LeaseMaxMS: 60000, AdmitMinHits: 0,
			PromoteWindow: time.Minute, Now: cur.Load, Quotas: q,
			Spill: recSpill{b}, Restore: recRestore{b}, S3ReadTimeout: 5 * time.Second,
		})
	fx := &restoreFixture{t: tt, q: q, b: b, vol: vol, cur: cur}
	t.Cleanup(func() { _ = tt.Close() })
	return fx
}

// pump is one deterministic tier cycle. The recorder acks inline, so the
// pass order alone guarantees reclaim sees settled S3 flags — no Flush.
func (fx *restoreFixture) pump() {
	fx.cur.Add(int64(200 * time.Millisecond))
	fx.t.demotePass(true)
	fx.t.spillPass()
	fx.t.reclaimPass()
	fx.t.gcPass()
}

// driveCold fills until at least one segment is retire-flipped and returns
// the lowest s3-resident segment holding ≥2 entries (each carries 3), with
// its keys and their payloads.
func (fx *restoreFixture) driveCold(t *testing.T) (segID uint32, keys [][32]byte, blobs map[[32]byte][]byte) {
	t.Helper()
	const blk, total = 60 << 10, 100
	blobs = map[[32]byte][]byte{}
	for i := 0; i < total && fx.q.Usage(1, tenant.TierS3) == 0; i++ {
		b := bytes.Repeat([]byte{byte(i), byte(0xB7 ^ i)}, blk/2) //nolint:gosec // G115: test payload pattern
		k := s3key(i)
		blobs[k] = b
		st := fx.t.Put(1, k, b, xxh3.Hash(b))
		if st == protocol.StatusErrQuotaBytes {
			fx.pump()
			st = fx.t.Put(1, k, b, xxh3.Hash(b))
		}
		if st != protocol.StatusOK {
			t.Fatalf("put %d: %s", i, st)
		}
		if i%4 == 3 {
			fx.pump()
		}
	}
	for k := 0; k < 40 && fx.q.Usage(1, tenant.TierS3) == 0; k++ {
		fx.pump()
	}
	if fx.q.Usage(1, tenant.TierS3) == 0 {
		t.Fatal("no segment reached s3-residency")
	}
	bySeg := map[uint32][][32]byte{}
	fx.t.idx.rangeAll(func(k dram.Key, ref *nvmeRef) bool {
		if ref.S3Only.Load() {
			bySeg[ref.Loc.SegmentID] = append(bySeg[ref.Loc.SegmentID], k.Hash)
		}
		return true
	})
	found := false
	for id, ks := range bySeg {
		if len(ks) >= 2 && (!found || id < segID) {
			segID, keys, found = id, ks, true
		}
	}
	if !found {
		t.Fatal("no s3-resident segment with ≥2 entries")
	}
	return segID, keys, blobs
}

// reclaimToHeadroom pumps until the volume can take one adopted segment —
// the restore path refuses (and kicks the reclaimer) without headroom, so
// tests stage it deterministically instead of racing the pressure loop.
func (fx *restoreFixture) reclaimToHeadroom(t *testing.T) {
	t.Helper()
	for i := 0; fx.vol.UsedBytes()+fx.vol.SegmentBytes() > fx.vol.MaxBytes(); i++ {
		if i >= 40 {
			t.Fatalf("volume never freed restore headroom (used=%d)", fx.vol.UsedBytes())
		}
		fx.pump()
	}
}

// coldGet asserts one verified cold-tier read.
func (fx *restoreFixture) coldGet(t *testing.T, key [32]byte, want []byte) {
	t.Helper()
	data, _, rel, tier, st := fx.t.GetRefTier(1, key)
	if st != protocol.StatusOK {
		t.Fatalf("cold get: %s", st)
	}
	defer rel()
	if tier != "s3" {
		t.Fatalf("cold get served from %q, want s3", tier)
	}
	if !bytes.Equal(data, want) {
		t.Fatal("cold get bytes differ")
	}
}

// TestSecondColdHitRestoresWholeSegment is the happy path end to end: two
// cold hits inside the promote window (≥ the min gap apart) restore the
// WHOLE segment — every entry serves locally again, byte-identical; the
// tenant charge moves S3→NVMe exactly; and the object is RETAINED, the
// adopted segment spilled=true — the state a later reclaim needs to FLIP
// instead of delete.
func TestSecondColdHitRestoresWholeSegment(t *testing.T) {
	fx := newRestoreFixture(t)
	segID, keys, blobs := fx.driveCold(t)

	fx.coldGet(t, keys[0], blobs[keys[0]]) // 1st hit arms the window
	fx.reclaimToHeadroom(t)

	var segBytes int64
	for _, k := range keys {
		segBytes += int64(len(blobs[k]))
	}
	preS3 := fx.q.Usage(1, tenant.TierS3)
	preNvme := fx.q.Usage(1, tenant.TierNVMe)

	fx.cur.Add(int64(20 * time.Millisecond)) // ≥ promoteMinGap, ≪ PromoteWindow
	fx.coldGet(t, keys[0], blobs[keys[0]])   // 2nd hit — still cold, enqueues the restore
	fx.t.restorePass()

	if got := fx.t.s3SegRestores.Load(); got != 1 {
		t.Fatalf("segment restores = %d, want 1", got)
	}
	for _, k := range keys {
		data, _, rel, tier, st := fx.t.GetRefTier(1, k)
		if st != protocol.StatusOK {
			t.Fatalf("post-restore get: %s", st)
		}
		if tier != "nvme" {
			rel()
			t.Fatalf("post-restore get served from %q, want nvme (local again)", tier)
		}
		if !bytes.Equal(data, blobs[k]) {
			rel()
			t.Fatal("post-restore bytes differ")
		}
		rel()
	}
	if got := fx.q.Usage(1, tenant.TierS3); got != preS3-segBytes {
		t.Fatalf("s3 usage after restore: %d, want %d (charge must move home)", got, preS3-segBytes)
	}
	if got := fx.q.Usage(1, tenant.TierNVMe); got != preNvme+segBytes {
		t.Fatalf("nvme usage after restore: %d, want %d", got, preNvme+segBytes)
	}
	// The object is RETAINED and the adopted segment is spilled=true: the
	// object backs the segment's next retire-flip. Dropping it here was the
	// reproduced data-loss blocker (adopt-as-unspilled + inline drop meant a
	// reclaim before the next spill-ack DELETED the entries with the object
	// already gone).
	if n := fx.b.dropCount(uint64(segID)); n != 0 {
		t.Fatalf("object drops for seg %d = %d, want 0 (retained after restore)", segID, n)
	}
	if !fx.b.hasObject(uint64(segID)) {
		t.Fatal("segment object vanished across a restore — it must be retained")
	}
	if !fx.vol.IsSpilled(segID) {
		t.Fatal("adopted segment not spilled=true — its next reclaim would DELETE instead of flip")
	}
	if got := fx.t.s3GCs.Load(); got != 0 {
		t.Fatalf("object gcs = %d, want 0 (nothing is dead)", got)
	}
}

// TestObjectGCDropsOnLastS3EntryDelete: deleting s3-resident entries leaves
// the object alone until the LAST one goes; the delete itself only enqueues
// (the foreground op never waits on S3), the GC pass drops exactly once,
// and a re-run is a no-op.
func TestObjectGCDropsOnLastS3EntryDelete(t *testing.T) {
	fx := newRestoreFixture(t)
	segID, keys, _ := fx.driveCold(t)

	for _, k := range keys[:len(keys)-1] {
		if st := fx.t.Delete(1, k, false); st != protocol.StatusOK {
			t.Fatalf("delete: %s", st)
		}
	}
	fx.t.gcPass()
	if n := fx.b.dropCount(uint64(segID)); n != 0 {
		t.Fatal("object dropped with an s3-resident entry still live")
	}

	last := keys[len(keys)-1]
	if st := fx.t.Delete(1, last, false); st != protocol.StatusOK {
		t.Fatalf("last delete: %s", st)
	}
	// The removal only NOMINATED the object — nothing dropped inline.
	if n := fx.b.dropCount(uint64(segID)); n != 0 {
		t.Fatal("foreground delete performed the S3 drop itself")
	}
	fx.t.gcPass()
	if n := fx.b.dropCount(uint64(segID)); n != 1 {
		t.Fatalf("object drops after last-entry delete = %d, want 1", n)
	}
	if fx.b.hasObject(uint64(segID)) {
		t.Fatal("dead segment object survived gc")
	}
	if got := fx.t.s3GCs.Load(); got != 1 {
		t.Fatalf("object gcs = %d, want 1", got)
	}
	fx.t.gcPass() // idempotent: the queue is drained, nothing re-drops
	if n := fx.b.dropCount(uint64(segID)); n != 1 {
		t.Fatalf("gc re-pass re-dropped: %d", n)
	}
}

// TestRestoreFailureKeepsColdService: a failed download changes NOTHING —
// entries stay s3-only, keep serving byte-identical through readS3, no
// charge moves, no object drops — and the next 2nd hit retries clean.
func TestRestoreFailureKeepsColdService(t *testing.T) {
	fx := newRestoreFixture(t)
	segID, keys, blobs := fx.driveCold(t)
	fx.reclaimToHeadroom(t)

	fx.b.mu.Lock()
	fx.b.failRestore = true
	fx.b.mu.Unlock()

	k0 := keys[0]
	preS3 := fx.q.Usage(1, tenant.TierS3)
	fx.coldGet(t, k0, blobs[k0])
	fx.cur.Add(int64(20 * time.Millisecond))
	fx.coldGet(t, k0, blobs[k0])
	fx.t.restorePass()

	if got := fx.t.s3RestoreErrs.Load(); got != 1 {
		t.Fatalf("restore errors = %d, want 1", got)
	}
	if got := fx.t.s3SegRestores.Load(); got != 0 {
		t.Fatalf("segment restores = %d after a failed download", got)
	}
	ref := fx.t.idx.get(dram.Key{NS: 1, Hash: k0})
	if ref == nil || !ref.S3Only.Load() {
		t.Fatal("entry left s3-residency across a FAILED restore")
	}
	if got := fx.q.Usage(1, tenant.TierS3); got != preS3 {
		t.Fatalf("s3 usage moved across a failed restore: %d → %d", preS3, got)
	}
	if n := fx.b.dropCount(uint64(segID)); n != 0 {
		t.Fatal("failed restore dropped the object it still depends on")
	}
	fx.coldGet(t, k0, blobs[k0]) // loss-free fallback: still served via readS3

	// The path heals: with the backend back, the next qualifying hit
	// restores for real.
	fx.b.mu.Lock()
	fx.b.failRestore = false
	fx.b.mu.Unlock()
	fx.cur.Add(int64(20 * time.Millisecond))
	fx.coldGet(t, k0, blobs[k0])
	fx.t.restorePass()
	if got := fx.t.s3SegRestores.Load(); got != 1 {
		t.Fatalf("segment restores after retry = %d, want 1", got)
	}
	if _, _, rel, tier, st := fx.t.GetRefTier(1, k0); st != protocol.StatusOK || tier != "nvme" {
		t.Fatalf("post-retry get: %s tier=%q, want OK nvme", st, tier)
	} else {
		rel()
	}
}

// TestDeleteRacingRestoreDoesNotResurrect stages the dangerous interleave
// deterministically: a DELETE lands after the download fetched the bytes
// but before the segment is adopted. The deleted key must NOT come back —
// the flip walk mutates only refs that still exist, never inserts — while
// the surviving entries flip home and the ledger stays exact.
func TestDeleteRacingRestoreDoesNotResurrect(t *testing.T) {
	fx := newRestoreFixture(t)
	segID, keys, blobs := fx.driveCold(t)
	fx.reclaimToHeadroom(t)

	victim, trigger := keys[0], keys[1]
	victimLen := int64(len(blobs[victim]))
	fx.b.mu.Lock()
	fx.b.onRestore = func() {
		if st := fx.t.Delete(1, victim, false); st != protocol.StatusOK {
			t.Errorf("mid-restore delete: %s", st)
		}
	}
	fx.b.mu.Unlock()

	var segBytes int64
	for _, k := range keys {
		segBytes += int64(len(blobs[k]))
	}
	preS3 := fx.q.Usage(1, tenant.TierS3)
	preNvme := fx.q.Usage(1, tenant.TierNVMe)

	fx.coldGet(t, trigger, blobs[trigger])
	fx.cur.Add(int64(20 * time.Millisecond))
	fx.coldGet(t, trigger, blobs[trigger])
	fx.t.restorePass()

	// No resurrection: the victim's bytes sit in the adopted file, but the
	// index never re-learns the key.
	if n, _ := fx.t.ExistsPrefix(1, [][32]byte{victim}, false); n != 0 {
		t.Fatal("deleted key EXISTS after a racing restore — resurrected")
	}
	if _, _, _, _, st := fx.t.GetRefTier(1, victim); st != protocol.StatusNotFound {
		t.Fatalf("deleted key GET = %s after a racing restore, want NOT_FOUND", st)
	}
	// Survivors flipped home and serve locally.
	data, _, rel, tier, st := fx.t.GetRefTier(1, trigger)
	if st != protocol.StatusOK || tier != "nvme" || !bytes.Equal(data, blobs[trigger]) {
		t.Fatalf("survivor after racing restore: %s tier=%q", st, tier)
	}
	rel()
	// Ledger exactness: the victim's charge was REFUNDED from S3 (by the
	// delete), the survivors' charges TRANSFERRED S3→NVMe — no tear, no
	// double-count.
	if got := fx.q.Usage(1, tenant.TierS3); got != preS3-segBytes {
		t.Fatalf("s3 usage: %d, want %d", got, preS3-segBytes)
	}
	if got := fx.q.Usage(1, tenant.TierNVMe); got != preNvme+segBytes-victimLen {
		t.Fatalf("nvme usage: %d, want %d", got, preNvme+segBytes-victimLen)
	}
	// The object is NOT dead: the adopted segment is live and spilled=true,
	// so the object backs its next retire-flip — only the segment's own
	// departure (with no s3-resident entries left) can free it.
	if n := fx.b.dropCount(uint64(segID)); n != 0 {
		t.Fatalf("object drops = %d, want 0 (live spilled segment claims it)", n)
	}
	if !fx.b.hasObject(uint64(segID)) {
		t.Fatal("object vanished while a live spilled segment claims it")
	}
}

// TestRestoredSegmentSurvivesReclaim is the reproduced data-loss BLOCKER
// (F1) as a permanent regression net. Old behavior: the restore adopted the
// segment spilled=false AND dropped its object inline, so a reclaim landing
// before the next spill-ack (with the real async spiller that window is
// seconds wide; here reclaim simply runs without an intervening spillPass,
// the same interleave) took the DELETE branch with the object already gone —
// the restored blocks vanished from EVERY tier. New behavior: the adopt
// publishes spilled=true and the object is retained, so the same reclaim
// takes the FLIP branch — the blocks move back to s3-residency, readable and
// xxh3-verified. This test FAILS against drop-on-restore and passes now.
func TestRestoredSegmentSurvivesReclaim(t *testing.T) {
	fx := newRestoreFixture(t)
	_, keys, blobs := fx.driveCold(t)

	fx.coldGet(t, keys[0], blobs[keys[0]]) // 1st hit arms the window
	fx.reclaimToHeadroom(t)
	fx.cur.Add(int64(20 * time.Millisecond))
	fx.coldGet(t, keys[0], blobs[keys[0]]) // 2nd hit — enqueues the restore
	fx.t.restorePass()
	if got := fx.t.s3SegRestores.Load(); got != 1 {
		t.Fatalf("segment restores = %d, want 1", got)
	}

	// Push the volume into the >90% reclaim band with fresh fill — WITHOUT a
	// spillPass in between, so the adopted segment stands on the state the
	// restore alone published (the async-spiller gap). Usually the volume is
	// already at 100% right after the adopt and this loop no-ops.
	for i := 1000; fx.vol.UsedBytes()*100 <= fx.vol.MaxBytes()*90; i++ {
		if i >= 1200 {
			t.Fatalf("volume never crossed the reclaim band (used=%d)", fx.vol.UsedBytes())
		}
		b := bytes.Repeat([]byte{byte(i), byte(0x3C ^ i)}, (60<<10)/2) //nolint:gosec // G115: test payload pattern
		st := fx.t.Put(1, s3key(i), b, xxh3.Hash(b))
		if st == protocol.StatusErrQuotaBytes {
			fx.cur.Add(int64(200 * time.Millisecond))
			fx.t.demotePass(true)
			st = fx.t.Put(1, s3key(i), b, xxh3.Hash(b))
		}
		if st != protocol.StatusOK {
			t.Fatalf("fill put %d: %s", i, st)
		}
		fx.cur.Add(int64(200 * time.Millisecond))
		fx.t.demotePass(true)
	}
	fx.cur.Add(int64(300 * time.Millisecond)) // the cold hits' auto-leases lapse
	fx.t.reclaimPass()                        // retires the adopted segment (the oldest sealed)
	fx.t.gcPass()

	// THE assertion: every restored block SURVIVES, byte-identical, on some
	// tier. Against drop-on-restore all of them read NOT_FOUND here.
	for _, k := range keys {
		data, sum, rel, tier, st := fx.t.GetRefTier(1, k)
		if st != protocol.StatusOK {
			t.Fatalf("restored block LOST after reclaim: %s (the drop-on-restore blocker)", st)
		}
		if sum != xxh3.Hash(blobs[k]) || !bytes.Equal(data, blobs[k]) {
			rel()
			t.Fatalf("restored block corrupt after reclaim (tier=%q)", tier)
		}
		rel()
	}
}

// TestStrandedRetireGCKeepsObject pins dropObjectHeld's IsSpilled deadness
// arm (the mutation-confirmed coverage gap): a retire strands mid-flip on a
// lease — RetireAbort, segment back live, the flips already applied STAY —
// then the flipped entries are deleted, draining the liveness count to zero
// and nominating the object. gcPass must REFUSE the drop: the live spilled
// segment's object backs its next retire-flip, and dropping it would point
// that flip at nothing. The second retire then proves the arm's rationale —
// the survivor flips out against the retained object and cold-serves.
func TestStrandedRetireGCKeepsObject(t *testing.T) {
	fx := newRestoreFixtureSized(t, 8<<20) // large budget: segments spill but never reclaim
	const blk = 60 << 10
	blobs := map[[32]byte][]byte{}
	for i := 0; i < 40; i++ {
		b := bytes.Repeat([]byte{byte(i), byte(0x6E ^ i)}, blk/2) //nolint:gosec // G115: test payload pattern
		k := s3key(i)
		blobs[k] = b
		st := fx.t.Put(1, k, b, xxh3.Hash(b))
		if st == protocol.StatusErrQuotaBytes {
			fx.cur.Add(int64(200 * time.Millisecond))
			fx.t.demotePass(true)
			st = fx.t.Put(1, k, b, xxh3.Hash(b))
		}
		if st != protocol.StatusOK {
			t.Fatalf("put %d: %s", i, st)
		}
		if i%4 == 3 {
			fx.cur.Add(int64(200 * time.Millisecond))
			fx.t.demotePass(true)
		}
	}
	fx.t.spillPass() // inline recorder: every sealed segment acks before return

	// Pick a spilled LIVE segment with ≥2 still-homed entries, in FOOTER
	// order — the flip walk visits that order, so leasing the LAST homed
	// entry guarantees at least one earlier entry flips before the strand.
	var segID uint32
	found := false
	bySeg := map[uint32]int{}
	fx.t.idx.rangeAll(func(k dram.Key, ref *nvmeRef) bool {
		if ref.S3.Load() && !ref.S3Only.Load() {
			bySeg[ref.Loc.SegmentID]++
		}
		return true
	})
	for id, n := range bySeg {
		if n >= 2 && fx.vol.IsSpilled(id) && (!found || id < segID) {
			segID, found = id, true
		}
	}
	if !found {
		t.Fatal("no spilled live segment with ≥2 entries")
	}
	var keys [][32]byte
	fx.vol.SegmentEntryKeys(segID, func(ns uint32, key [32]byte, _, _ uint32) {
		if ref := fx.t.idx.get(dram.Key{NS: ns, Hash: key}); ref != nil && ref.Loc.SegmentID == segID {
			keys = append(keys, key)
		}
	})
	if len(keys) < 2 {
		t.Fatalf("segment %d has %d homed entries, want ≥2", segID, len(keys))
	}
	strander := keys[len(keys)-1]
	if st := fx.t.TouchLease(1, strander, protocol.LeaseGrant, 60_000); st != protocol.StatusOK {
		t.Fatalf("lease grant: %s", st)
	}

	if !fx.vol.RetireBegin(segID) {
		t.Fatalf("RetireBegin(%d) refused", segID)
	}
	if stranded := fx.t.s3FlipRetired(fx.vol, segID); stranded == 0 {
		t.Fatal("the leased entry did not strand the retire")
	}
	fx.vol.RetireAbort(segID)

	// Flips already applied stay — delete them; the LAST delete drains the
	// liveness count to zero and nominates the object.
	var flipped [][32]byte
	for _, k := range keys[:len(keys)-1] {
		if ref := fx.t.idx.get(dram.Key{NS: 1, Hash: k}); ref != nil && ref.S3Only.Load() {
			flipped = append(flipped, k)
		}
	}
	if len(flipped) == 0 {
		t.Fatal("no entry flipped before the strand — the walk order changed?")
	}
	for _, k := range flipped {
		if st := fx.t.Delete(1, k, false); st != protocol.StatusOK {
			t.Fatalf("delete flipped entry: %s", st)
		}
	}
	fx.t.gcPass()
	if !fx.b.hasObject(uint64(segID)) {
		t.Fatal("gc dropped a live spilled segment's object — its next retire-flip now points at nothing")
	}
	if n := fx.b.dropCount(uint64(segID)); n != 0 {
		t.Fatalf("object drops = %d, want 0 (IsSpilled deadness arm)", n)
	}
	if got := fx.t.s3GCs.Load(); got != 0 {
		t.Fatalf("object gcs = %d, want 0", got)
	}

	// The arm's rationale, proven: release the lease, retire for real — the
	// survivor flips out against the RETAINED object and serves cold.
	if st := fx.t.TouchLease(1, strander, protocol.LeaseRelease, 0); st != protocol.StatusOK {
		t.Fatalf("lease release: %s", st)
	}
	if !fx.vol.RetireBegin(segID) {
		t.Fatalf("second RetireBegin(%d) refused", segID)
	}
	if stranded := fx.t.s3FlipRetired(fx.vol, segID); stranded != 0 {
		t.Fatalf("second retire stranded %d entries", stranded)
	}
	if err := fx.vol.RetireFinish(segID); err != nil {
		t.Fatal(err)
	}
	fx.coldGet(t, strander, blobs[strander])
}

// TestConcurrentRestoreVsReclaim races a whole-segment restore against the
// reclaimer over the same volume under -race. The spillInflight latch must
// linearize them: whichever owns the segment, the other skips — no entry may
// be lost or ghosted, every byte stays servable and verified, and the
// object-GC liveness counts settle EXACTLY equal to the index's s3-resident
// entries (no double-count, no negative; the kvbdebug build asserts the
// negative half).
func TestConcurrentRestoreVsReclaim(t *testing.T) {
	fx := newRestoreFixture(t)
	segID, keys, blobs := fx.driveCold(t)

	fx.coldGet(t, keys[0], blobs[keys[0]])
	fx.reclaimToHeadroom(t)
	fx.cur.Add(int64(20 * time.Millisecond))
	fx.coldGet(t, keys[0], blobs[keys[0]]) // 2nd cold hit — queues the restore
	var req restoreReq
	select {
	case req = <-fx.t.restoreq:
	default:
		t.Fatal("no restore queued")
	}

	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		fx.t.restoreOne(context.Background(), req)
	}()
	go func() {
		defer wg.Done()
		for i := 0; i < 400; i++ {
			if i%20 == 19 {
				// Lapse the cold hits' auto-leases mid-race so the reclaimer
				// genuinely contends for the adopted segment.
				fx.cur.Add(int64(50 * time.Millisecond))
			}
			fx.t.reclaimSegment(fx.vol, 0)
			runtime.Gosched()
		}
	}()
	wg.Wait()
	fx.t.gcPass()

	// No lost or ghost entries: every key of the restored segment serves
	// byte-identical from SOME tier (nvme if the restore's flip stood, s3 if
	// a reclaim re-flipped it out — both are loss-free).
	for _, k := range keys {
		data, sum, rel, tier, st := fx.t.GetRefTier(1, k)
		if st != protocol.StatusOK {
			t.Fatalf("entry lost in the restore-vs-reclaim race: %s", st)
		}
		if sum != xxh3.Hash(blobs[k]) || !bytes.Equal(data, blobs[k]) {
			rel()
			t.Fatalf("entry corrupt after the race (tier=%q)", tier)
		}
		rel()
	}
	// Refs settle exact: the liveness map must equal the index's s3-resident
	// census, segment by segment.
	expect := map[uint32]int{}
	var expectS3Bytes int64
	fx.t.idx.rangeAll(func(_ dram.Key, ref *nvmeRef) bool {
		if ref.S3Only.Load() {
			expect[ref.Loc.SegmentID]++
			expectS3Bytes += int64(ref.Len)
		}
		return true
	})
	fx.t.s3SegMu.Lock()
	got := make(map[uint32]int, len(fx.t.s3SegRefs))
	for id, n := range fx.t.s3SegRefs {
		got[id] = n
	}
	fx.t.s3SegMu.Unlock()
	if len(got) != len(expect) {
		t.Fatalf("liveness map %v != index census %v", got, expect)
	}
	for id, n := range expect {
		if got[id] != n {
			t.Fatalf("segment %d liveness %d != %d s3-resident entries", id, got[id], n)
		}
	}
	if b := fx.t.s3Bytes.Load(); b != expectS3Bytes {
		t.Fatalf("s3 bytes counter %d != index census %d", b, expectS3Bytes)
	}
	if u := fx.q.Usage(1, tenant.TierS3); u != expectS3Bytes {
		t.Fatalf("s3 quota ledger %d != index census %d", u, expectS3Bytes)
	}
	// Whatever the interleave, the object must have survived: either the
	// segment is live+spilled (restore stood) or its entries are s3-resident
	// again (reclaim re-flipped) — both need it.
	if !fx.b.hasObject(uint64(segID)) {
		t.Fatal("segment object dropped during the restore-vs-reclaim race")
	}
}

// TestReclaimLatchBlocksRestoreOvertake pins the reclaim-side spillInflight
// latch itself — the piece whose REMOVAL survived every earlier test (the
// M4 mutation: TestConcurrentRestoreVsReclaim above cannot force the
// interleave on an idle box). The tear it must prevent, staged
// deterministically via the mid-walk seam: the restore flips its first
// entry home → an unlatched reclaim pre-gates clean, RetireBegin, its
// flip-out walk re-flips THAT entry and skips the rest (still s3-only,
// "already flipped"), RetireFinish → the restore walk then flips the
// remainder "home" into a segment that no longer exists → deleting the
// re-flipped entry drains liveness to zero → gcPass drops the object → the
// remainder is unreadable on every tier while still indexed. With the
// latch, the mid-walk reclaim answers busy and nothing tears. Red-proofed:
// this test FAILS with the latch check in reclaimSegment removed and
// passes with it present.
func TestReclaimLatchBlocksRestoreOvertake(t *testing.T) {
	fx := newRestoreFixture(t)
	_, keys, blobs := fx.driveCold(t)

	fx.coldGet(t, keys[0], blobs[keys[0]]) // 1st hit arms the window
	fx.reclaimToHeadroom(t)
	fx.cur.Add(int64(20 * time.Millisecond))
	fx.coldGet(t, keys[0], blobs[keys[0]]) // 2nd hit — queues the restore

	hookFired := false
	fx.t.restoreFlipHookForTest = func() {
		if hookFired {
			return // one overtake attempt, at the first flipped entry
		}
		hookFired = true
		// Lapse the cold hits' auto-leases so the pre-gate is genuinely
		// clean, then attack the adopted segment (the oldest sealed)
		// exactly as the demoter would. With the latch this answers busy;
		// without it the retire completes out from under the walk.
		fx.cur.Add(int64(300 * time.Millisecond))
		fx.t.reclaimSegment(fx.vol, 0)
	}
	fx.t.restorePass()
	if !hookFired {
		t.Fatal("mid-walk seam never fired — the restore path changed shape?")
	}

	// Drive the loss sequence to its end: remove whatever the interleave
	// left s3-only (with the latch: nothing), then run the GC pass that
	// drops the object once its liveness drained.
	deleted := map[[32]byte]bool{}
	for _, k := range keys {
		if ref := fx.t.idx.get(dram.Key{NS: 1, Hash: k}); ref != nil && ref.S3Only.Load() {
			if st := fx.t.Delete(1, k, false); st != protocol.StatusOK {
				t.Fatalf("delete s3-only leftover: %s", st)
			}
			deleted[k] = true
		}
	}
	fx.t.gcPass()

	// THE invariant: every non-deleted block of the restored segment still
	// GETs byte-identical from some tier. With the latch removed the
	// remainder reads NOT_FOUND here — indexed entries servable nowhere.
	for _, k := range keys {
		if deleted[k] {
			continue // a legal removal, not a loss
		}
		data, sum, rel, tier, st := fx.t.GetRefTier(1, k)
		if st != protocol.StatusOK {
			t.Fatalf("block lost to the restore-overtake tear: %s", st)
		}
		if sum != xxh3.Hash(blobs[k]) || !bytes.Equal(data, blobs[k]) {
			rel()
			t.Fatalf("block corrupt after the overtake (tier=%q)", tier)
		}
		rel()
	}
}
