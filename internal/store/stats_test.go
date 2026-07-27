package store

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/kvstash/kvblockd/internal/protocol"
	"github.com/kvstash/kvblockd/internal/store/dram"
)

// statsWalk recomputes what the funnel-maintained NVMe counters claim, the
// slow way. The counters replaced an O(total blocks) walk per scrape; this
// audit is the proof they never drift from the seams they were installed
// at — the same discipline the tenant quota refunds get. (The DRAM side's
// walk audit lives in the dram package, which owns that index.)
func statsWalk(tt *Tiered) (nvmeBlocks int, nvmeBytes int64) {
	tt.idx.rangeAll(func(_ dram.Key, ref *nvmeRef) bool {
		nvmeBlocks++
		nvmeBytes += int64(ref.Len)
		return true
	})
	return nvmeBlocks, nvmeBytes
}

func auditStats(t *testing.T, fx *fixture, phase string) {
	t.Helper()
	nvB, nvBy := statsWalk(fx.t)
	gotNvB, gotNvBy := fx.t.idx.stats()
	if gotNvB != nvB || gotNvBy != nvBy {
		t.Fatalf("%s: nvme counters (%d blocks, %d bytes) drifted from the walk (%d, %d)",
			phase, gotNvB, gotNvBy, nvB, nvBy)
	}
}

// TestStatsCountersMatchWalkAcrossLifecycles runs blocks through every
// mutation funnel — put, demote (dram delete + nvme publish), delete on
// both tiers, promotion — auditing the O(1) counters against a full walk
// after each phase.
func TestStatsCountersMatchWalkAcrossLifecycles(t *testing.T) {
	fx := newFixture(t, 64<<20, nil)
	fx.fill(t)
	auditStats(t, fx, "after fill")

	if fx.t.DemoteNow() == 0 {
		t.Fatal("no demotion")
	}
	auditStats(t, fx, "after demotion")

	// Delete a spread of keys — some DRAM-resident, some NVMe-resident.
	deleted := 0
	for i := 0; i < fillN; i += 3 {
		if st := fx.t.Delete(1, tk(i), false); st == protocol.StatusOK {
			deleted++
		}
	}
	if deleted == 0 {
		t.Fatal("no delete landed")
	}
	auditStats(t, fx, "after deletes")

	// Promotions (second GET hits) republish into DRAM.
	for i := 1; i < fillN; i += 5 {
		if _, _, rel, _, st := fx.t.GetRefTier(context.Background(), 1, tk(i)); st == protocol.StatusOK {
			rel()
		}
		if _, _, rel, _, st := fx.t.GetRefTier(context.Background(), 1, tk(i)); st == protocol.StatusOK {
			rel()
		}
	}
	auditStats(t, fx, "after promotions")

	// The published document reports the SAME numbers the counters hold.
	var doc struct {
		Blocks int64 `json:"blocks"`
		Bytes  int64 `json:"bytes"`
		Nvme   struct {
			Blocks int   `json:"blocks"`
			Bytes  int64 `json:"bytes"`
		} `json:"nvme"`
	}
	if err := json.Unmarshal(fx.t.Stats(), &doc); err != nil {
		t.Fatalf("stats doc: %v", err)
	}
	nvB, nvBy := statsWalk(fx.t)
	snap := fx.t.d.StatsSnapshot()
	if doc.Blocks != snap.Blocks || doc.Bytes != snap.Bytes {
		t.Fatalf("doc dram (%d, %d) != snapshot (%d, %d)", doc.Blocks, doc.Bytes, snap.Blocks, snap.Bytes)
	}
	if doc.Nvme.Blocks != nvB || doc.Nvme.Bytes != nvBy {
		t.Fatalf("doc nvme (%d, %d) != walk (%d, %d)", doc.Nvme.Blocks, doc.Nvme.Bytes, nvB, nvBy)
	}
}
