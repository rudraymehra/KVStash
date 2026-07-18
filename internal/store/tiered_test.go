package store

import (
	"bytes"
	"context"
	"fmt"
	"sync/atomic"
	"testing"
	"time"

	"github.com/zeebo/xxh3"
	"go.uber.org/goleak"

	"github.com/kvstash/kvblockd/internal/eviction"
	"github.com/kvstash/kvblockd/internal/protocol"
	"github.com/kvstash/kvblockd/internal/store/dram"
	"github.com/kvstash/kvblockd/internal/store/nvme"
	"github.com/kvstash/kvblockd/internal/store/storetest"
)

func TestMain(m *testing.M) {
	goleak.VerifyTestMain(m)
}

const msNanos = int64(1_000_000)

// fillN × 60 KiB ≈ 7.5 MiB = 91.5% of the 8 MiB arena — past the 90%
// demote watermark with room to spare before the Put quota wall.
const fillN = 125

type fixture struct {
	t    *Tiered
	cur  *atomic.Int64 // fake clock (nanos)
	stop func()
}

// newFixture builds a small two-tier store: 8 MiB arena, one volume with
// 256 KiB segments, s3fifo policy shared between the dram evict machinery
// and the demoter, fake clock.
func newFixture(t *testing.T, volMaxBytes int64, backend nvme.IOBackend) *fixture {
	t.Helper()
	cur := &atomic.Int64{}
	cur.Store(1)
	arena, err := dram.NewArena(8<<20, false)
	if err != nil {
		t.Fatal(err)
	}
	d := dram.New(arena, dram.Params{LeaseDefaultMS: 100, LeaseMaxMS: 60000, Now: cur.Load})
	pol, err := eviction.New("s3fifo", 1024)
	if err != nil {
		t.Fatal(err)
	}
	d.AttachPolicy(pol)
	vol, rep, ents, err := nvme.OpenVolume(nvme.VolumeParams{
		Dir: t.TempDir(), SegmentBytes: 256 << 10, MaxBytes: volMaxBytes,
		ReadWorkers: 2, CkptEverySegs: 2, MaxBlobLen: 64 << 10,
		Backend: backend, Now: cur.Load,
	})
	if err != nil {
		t.Fatal(err)
	}
	tt := NewTiered(d, pol, []*nvme.Volume{vol}, []*nvme.RecoveryReport{rep},
		[][]nvme.RecoveredEntry{ents}, Params{
			LeaseDefaultMS: 100, LeaseMaxMS: 60000,
			PromoteWindow: time.Minute, Now: cur.Load,
		})
	fx := &fixture{t: tt, cur: cur, stop: func() {}}
	t.Cleanup(func() {
		fx.stop()
		_ = tt.Close()
	})
	return fx
}

func tk(i int) [32]byte {
	var k [32]byte
	copy(k[:], fmt.Sprintf("tiered-%06d", i))
	return k
}

func tblob(i, n int) []byte {
	p := make([]byte, n)
	for j := range p {
		p[j] = byte(i ^ j) //nolint:gosec // G115: deliberate wrap — test pattern
	}
	return p
}

// fill puts the standard fillN × 60 KiB blocks and GETs each once
// (hits=1 → demotable), then lapses the auto-leases.
func (fx *fixture) fill(t *testing.T) []uint64 {
	t.Helper()
	return fx.fillRange(t, 0, fillN, 60<<10)
}

// fillRange is fill starting at key index start (fresh keys for follow-up
// pressure waves).
func (fx *fixture) fillRange(t *testing.T, start, n, sz int) []uint64 {
	t.Helper()
	sums := make([]uint64, n)
	for j := 0; j < n; j++ {
		i := start + j
		b := tblob(i, sz)
		sums[j] = xxh3.Hash(b)
		if st := fx.t.Put(1, tk(i), b, sums[j]); st != protocol.StatusOK {
			t.Fatalf("put %d: %s", i, st)
		}
		data, _, rel, tier, st := fx.t.GetRefTier(1, tk(i))
		if st != protocol.StatusOK || tier != "dram" {
			t.Fatalf("get %d: %s tier=%s", i, st, tier)
		}
		_ = data
		rel()
	}
	fx.cur.Add(500 * msNanos) // all auto-leases lapse
	return sums
}

func TestTieredConformance(t *testing.T) {
	storetest.RunConformance(t, func(t *testing.T) storetest.Store {
		return newFixture(t, 64<<20, nil).t
	})
}

func TestDemoteThenGetRoundTrip(t *testing.T) {
	fx := newFixture(t, 64<<20, nil)
	sums := fx.fill(t)

	moved := fx.t.DemoteNow()
	if moved == 0 {
		t.Fatal("demotion pass moved nothing at 90% occupancy")
	}
	if fx.t.demotions.Load() == 0 {
		t.Fatal("no demotion completed")
	}

	// Every block still answers byte-identical — from whichever tier.
	nvmeHits := 0
	for i := 0; i < fillN; i++ {
		data, sum, rel, tier, st := fx.t.GetRefTier(1, tk(i))
		if st != protocol.StatusOK {
			t.Fatalf("block %d lost after demotion: %s", i, st)
		}
		if sum != sums[i] || !bytes.Equal(data, tblob(i, 60<<10)) {
			t.Fatalf("block %d corrupted across the tier move", i)
		}
		rel()
		if tier == "nvme" {
			nvmeHits++
		}
	}
	if nvmeHits == 0 {
		t.Fatal("no block served from the NVMe tier")
	}
	// EXISTS still sees all of them (merged view).
	keys := make([][32]byte, fillN)
	for i := range keys {
		keys[i] = tk(i)
	}
	if n, _ := fx.t.ExistsPrefix(1, keys, false); n != fillN {
		t.Fatalf("ExistsPrefix consecutive = %d, want fillN", n)
	}
}

func TestWriteOnceAcrossTiers(t *testing.T) {
	fx := newFixture(t, 64<<20, nil)
	fx.fill(t)
	if fx.t.DemoteNow() == 0 {
		t.Fatal("no demotion")
	}
	// Find an NVMe-only resident.
	victim := -1
	for i := 0; i < fillN; i++ {
		_, _, rel, tier, st := fx.t.GetRefTier(1, tk(i))
		if st == protocol.StatusOK {
			rel()
			if tier == "nvme" {
				victim = i
				break
			}
		}
	}
	if victim < 0 {
		t.Fatal("no nvme-resident block found")
	}
	b := tblob(victim, 60<<10)
	if st := fx.t.Put(1, tk(victim), b, xxh3.Hash(b)); st != protocol.StatusOKExists {
		t.Fatalf("same-content re-put: %s, want OK_EXISTS", st)
	}
	other := tblob(victim+1, 60<<10)
	if st := fx.t.Put(1, tk(victim), other, xxh3.Hash(other)); st != protocol.StatusErrImmutableConflict {
		t.Fatalf("different-content re-put: %s, want ERR_IMMUTABLE_CONFLICT", st)
	}
}

func TestPromoteOnSecondHit(t *testing.T) {
	fx := newFixture(t, 64<<20, nil)
	fx.fill(t)
	if fx.t.DemoteNow() == 0 {
		t.Fatal("no demotion")
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	fx.stop = fx.t.Start(ctx)

	victim := -1
	for i := 0; i < fillN; i++ {
		_, _, rel, tier, st := fx.t.GetRefTier(1, tk(i))
		if st == protocol.StatusOK {
			rel()
			if tier == "nvme" {
				victim = i
				break
			}
		}
	}
	if victim < 0 {
		t.Fatal("no nvme-resident block")
	}
	// First hit above set LastAccess; a second hit within the window (but
	// past the 10ms frame-boundary guard) enqueues the promotion. Poll
	// until the promoter lands it in DRAM, advancing the fake clock so
	// consecutive hits are distinguishable touches.
	deadline := time.Now().Add(5 * time.Second)
	for {
		fx.cur.Add(50 * msNanos)
		_, _, rel, tier, st := fx.t.GetRefTier(1, tk(victim))
		if st != protocol.StatusOK {
			t.Fatalf("victim lost during promotion: %s", st)
		}
		rel()
		if tier == "dram" {
			break // promoted
		}
		if time.Now().After(deadline) {
			t.Fatal("2nd-hit promotion never landed")
		}
		time.Sleep(5 * time.Millisecond)
	}
	if fx.t.promotions.Load() == 0 {
		t.Fatal("promotions counter untouched")
	}
}

func TestHardPinPromotes(t *testing.T) {
	fx := newFixture(t, 64<<20, nil)
	fx.fill(t)
	if fx.t.DemoteNow() == 0 {
		t.Fatal("no demotion")
	}
	victim := -1
	for i := 0; i < fillN; i++ {
		_, _, rel, tier, st := fx.t.GetRefTier(1, tk(i))
		if st == protocol.StatusOK {
			rel()
			if tier == "nvme" {
				victim = i
				break
			}
		}
	}
	if victim < 0 {
		t.Fatal("no nvme-resident block")
	}
	if st := fx.t.PinOp(1, tk(victim), protocol.PinHard); st != protocol.StatusOK {
		t.Fatalf("hard pin on nvme resident: %s", st)
	}
	// The pin promoted it: DRAM serves it now, and DELETE refuses hard.
	_, _, rel, tier, st := fx.t.GetRefTier(1, tk(victim))
	if st != protocol.StatusOK || tier != "dram" {
		t.Fatalf("hard-pinned block: %s tier=%s, want dram", st, tier)
	}
	rel()
	if st := fx.t.Delete(1, tk(victim), true); st != protocol.StatusErrPinned {
		t.Fatalf("delete of hard-pinned: %s, want ERR_PINNED", st)
	}
}

func TestLeaseProtectsNvmeResidentFromReclaim(t *testing.T) {
	// Small volume budget: 1 MiB = 4 segments; filling drives reclaim.
	fx := newFixture(t, 1<<20, nil)
	sums := fx.fill(t)
	if fx.t.DemoteNow() == 0 {
		t.Fatal("no demotion")
	}
	// Lease every nvme-resident block, then force reclaim pressure with
	// another demotion wave: the reclaimer must SKIP leased segments.
	leased := make(map[int]bool)
	for i := 0; i < fillN; i++ {
		if fx.t.d.Contains(1, tk(i)) {
			continue
		}
		if fx.t.idx.contains(dram.Key{NS: 1, Hash: tk(i)}) {
			if st := fx.t.TouchLease(1, tk(i), protocol.LeaseGrant, 60000); st != protocol.StatusOK {
				t.Fatalf("lease on nvme block %d: %s", i, st)
			}
			leased[i] = true
		}
	}
	if len(leased) == 0 {
		t.Fatal("nothing nvme-resident to lease")
	}
	fx.t.reclaimPass()
	if fx.t.reclaims.Load() > 0 && fx.t.reclaimSkips.Load() == 0 {
		t.Log("reclaim proceeded — verifying no leased block was dropped")
	}
	for i := range leased {
		data, _, rel, _, st := fx.t.GetRefTier(1, tk(i))
		if st != protocol.StatusOK {
			t.Fatalf("leased nvme block %d dropped by reclaim: %s", i, st)
		}
		if !bytes.Equal(data, tblob(i, 60<<10)) || xxh3.Hash(data) != sums[i] {
			t.Fatalf("leased block %d corrupted", i)
		}
		rel()
	}
}

func TestReclaimFreesUnprotectedSegments(t *testing.T) {
	fx := newFixture(t, 1<<20, nil) // 1 MiB budget → reclaim must fire
	fx.fill(t)
	if fx.t.DemoteNow() == 0 {
		t.Fatal("no demotion")
	}
	fx.cur.Add(500 * msNanos) // lapse every auto-lease
	// Drive more demotion waves (fresh keys, sized to the headroom demotion
	// frees — no evictor runs in this fixture) to push the volume over budget.
	for w := 0; w < 6; w++ {
		fx.fillRange(t, 1000+w*100, 15, 60<<10)
		fx.t.DemoteNow()
		fx.cur.Add(500 * msNanos)
	}
	fx.t.reclaimPass()
	if fx.t.reclaims.Load() == 0 && fx.t.vols[0].UsedBytes()*100 > fx.t.vols[0].MaxBytes()*90 {
		t.Fatalf("volume over budget (%d/%d) but nothing reclaimed (skips=%d)",
			fx.t.vols[0].UsedBytes(), fx.t.vols[0].MaxBytes(), fx.t.reclaimSkips.Load())
	}
	// Reclaimed blocks are honest misses; everything still resident verifies.
	if bad := fx.t.Scrub(); bad != 0 {
		t.Fatalf("scrub after reclaim: bad=%d", bad)
	}
}

func TestNvmeLifecycleVerbs(t *testing.T) {
	fx := newFixture(t, 64<<20, nil)
	fx.fill(t)
	if fx.t.DemoteNow() == 0 {
		t.Fatal("no demotion")
	}
	victim := -1
	for i := 0; i < fillN; i++ {
		if !fx.t.d.Contains(1, tk(i)) && fx.t.idx.contains(dram.Key{NS: 1, Hash: tk(i)}) {
			victim = i
			break
		}
	}
	if victim < 0 {
		t.Fatal("no nvme-only block")
	}
	k := tk(victim)
	// TOUCH with TTL, LEASE, RELEASE all answer OK on the nvme tier.
	for _, sub := range []uint8{protocol.TouchRecency, protocol.LeaseGrant, protocol.LeaseRelease} {
		if st := fx.t.TouchLease(1, k, sub, 1000); st != protocol.StatusOK {
			t.Fatalf("sub %d on nvme resident: %s", sub, st)
		}
	}
	// Soft pin protects from delete; unpin releases it.
	if st := fx.t.PinOp(1, k, protocol.PinSoft); st != protocol.StatusOK {
		t.Fatal(st)
	}
	if st := fx.t.Delete(1, k, false); st != protocol.StatusErrPinned {
		t.Fatalf("delete soft-pinned nvme block: %s, want ERR_PINNED", st)
	}
	if st := fx.t.PinOp(1, k, protocol.Unpin); st != protocol.StatusOK {
		t.Fatal(st)
	}
	if st := fx.t.Delete(1, k, false); st != protocol.StatusOK {
		t.Fatalf("delete unpinned nvme block: %s", st)
	}
	if fx.t.Contains(1, k) {
		t.Fatal("deleted nvme block still visible")
	}
}
