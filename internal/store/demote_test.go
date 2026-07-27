package store

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/zeebo/xxh3"

	"github.com/kvstash/kvblockd/internal/protocol"
	"github.com/kvstash/kvblockd/internal/store/dram"
)

// TestReclaimSkipsProtectedOldestSegment pins the FIFO bend for protected
// heads: a soft pin (no expiry until UNPIN) on ONE block in the OLDEST
// sealed segment must not head-of-line-block the volume — the pass skips it
// under the reclaimBusy budget, retires a newer segment, counts the skip on
// reclaim_blocked_protected_total, and the pinned block keeps serving.
// Before the fix the pre-gate returned reclaimStop and the whole volume
// wedged over budget behind the pin (reclaims stayed 0).
func TestReclaimSkipsProtectedOldestSegment(t *testing.T) {
	fx := newFixture(t, 1<<20, nil) // 1 MiB budget = 4 segments — reclaim must fire
	fx.fill(t)
	if fx.t.demotePass(true) == 0 {
		t.Fatal("no demotion at 90% occupancy")
	}
	fx.cur.Add(500 * msNanos)
	vol := fx.t.vols[0]

	// Soft-pin one block still homed in the oldest sealed segment.
	oldest, entries, ok := vol.OldestSealed(0)
	if !ok || len(entries) == 0 {
		t.Fatal("no sealed segment after the demote wave")
	}
	var pinned [32]byte
	found := false
	for _, e := range entries {
		if ref := fx.t.idx.get(dram.Key{NS: e.NS, Hash: e.Key}); ref != nil && ref.Loc.SegmentID == oldest {
			pinned, found = e.Key, true
			break
		}
	}
	if !found {
		t.Fatalf("segment %d has no index-homed entry to pin", oldest)
	}
	if st := fx.t.PinOp(1, pinned, protocol.PinSoft); st != protocol.StatusOK {
		t.Fatalf("soft pin: %s", st)
	}

	// Drive the volume over its reclaim watermark with fresh demote waves
	// (demotePass only — reclaim runs once, explicitly, below).
	for w := 0; w < 10 && vol.UsedBytes()*100 <= vol.MaxBytes()*90; w++ {
		fx.fillRange(t, 1000+w*100, 15, 60<<10)
		fx.t.demotePass(true)
		fx.cur.Add(500 * msNanos)
	}
	if vol.UsedBytes()*100 <= vol.MaxBytes()*90 {
		t.Fatalf("could not drive the volume over budget (%d/%d)", vol.UsedBytes(), vol.MaxBytes())
	}

	fx.t.reclaimPass()

	if got := fx.t.reclaimBlockedProtected.Load(); got == 0 {
		t.Fatal("protected oldest segment did not count on reclaim_blocked_protected_total")
	}
	if got := fx.t.reclaims.Load(); got == 0 {
		t.Fatal("one soft pin in the oldest segment wedged ALL reclaim (pass ended instead of skipping)")
	}
	if !vol.HasSegment(oldest) {
		t.Fatal("the protected segment itself was retired")
	}
	data, _, rel, tier, st := fx.t.GetRefTier(context.Background(), 1, pinned)
	if st != protocol.StatusOK || tier != "nvme" {
		t.Fatalf("pinned block after the skip-reclaim pass: %s tier=%q", st, tier)
	}
	if len(data) != 60<<10 {
		t.Fatalf("pinned block truncated: %d bytes", len(data))
	}
	rel()
}

// TestDemoteNowSingleflightsWithTicker provokes the scratch race demoteMu
// closes: a fast ticker loop and concurrent DemoteNow callers on a LIVE
// store share scUsages/scCands — without the singleflight the race detector
// flags the slice reuse (the exported trigger used to be safe only with
// loops stopped, an ordering contract the type system never enforced).
func TestDemoteNowSingleflightsWithTicker(t *testing.T) {
	fx := newFixture(t, 64<<20, nil)
	fx.fill(t)
	fx.t.p.Interval = 2 * time.Millisecond // before Start — the ticker cadence
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	fx.stop = fx.t.Start(ctx)

	var wg sync.WaitGroup
	for g := 0; g < 4; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			for i := 0; i < 25; i++ {
				b := tblob(g*1000+i, 60<<10)
				// Refill pressure so demotePass keeps consulting the policy;
				// quota refusals under pressure are expected and fine.
				_ = fx.t.Put(1, tk(5000+g*1000+i), b, xxh3.Hash(b))
				fx.t.DemoteNow()
			}
		}(g)
	}
	done := make(chan struct{})
	go func() { wg.Wait(); close(done) }()
	for {
		select {
		case <-done:
			fx.stop()
			return
		case <-time.After(5 * time.Millisecond):
			fx.cur.Add(500 * msNanos) // lapse auto-leases so victims stay eligible
		}
	}
}
