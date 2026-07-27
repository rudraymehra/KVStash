package tenant

import (
	"errors"
	"sync"
	"sync/atomic"
	"testing"
)

func quotasWith(t *testing.T, dram int64) *Quotas {
	t.Helper()
	r := NewRegistry("a", 1, "tok")
	if !r.SetQuota("a", TierDRAM, dram) {
		t.Fatal("SetQuota")
	}
	return NewQuotas(r)
}

func TestChargeRefundExactness(t *testing.T) {
	q := quotasWith(t, 100)
	if err := q.Charge(1, TierDRAM, 60); err != nil {
		t.Fatal(err)
	}
	if err := q.Charge(1, TierDRAM, 60); !errors.Is(err, ErrQuota) {
		t.Fatalf("over-quota admitted: %v", err)
	}
	q.Refund(1, TierDRAM, 60)
	if got := q.Usage(1, TierDRAM); got != 0 {
		t.Fatalf("usage after exact refund: %d", got)
	}
	if err := q.Charge(1, TierDRAM, 100); err != nil {
		t.Fatalf("full-quota charge after refund: %v", err)
	}
}

func TestChargeUnlimitedWhenZero(t *testing.T) {
	q := quotasWith(t, 0)
	if err := q.Charge(1, TierDRAM, 1<<40); err != nil {
		t.Fatalf("unlimited tier refused: %v", err)
	}
}

// I3, the storm form: 32 goroutines racing a tight quota land AT MOST the
// quota — the CAS loop never lets check-and-add interleave (each racer's
// admission is atomic; the +1-block slack in production comes from BEGIN
// reserving before COMMIT, not from the counter).
func TestChargeStormNeverExceedsQuota(t *testing.T) {
	const quota, blockSz, workers, tries = 1000, 10, 32, 100
	q := quotasWith(t, quota)
	var admitted atomic.Int64
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < tries; i++ {
				if q.Charge(1, TierDRAM, blockSz) == nil {
					admitted.Add(blockSz)
				}
			}
		}()
	}
	wg.Wait()
	if got := q.Usage(1, TierDRAM); got != quota {
		t.Fatalf("usage %d after storm, want exactly the %d quota (admissions must stop AT the line)", got, quota)
	}
	if admitted.Load() != quota {
		t.Fatalf("admitted %d bytes, want %d", admitted.Load(), quota)
	}
}

func TestTransferMovesBetweenTiersAndNeverFails(t *testing.T) {
	r := NewRegistry("a", 1, "tok")
	r.SetQuota("a", TierDRAM, 100)
	r.SetQuota("a", TierNVMe, 10) // destination quota TIGHTER than the move
	q := NewQuotas(r)
	if err := q.Charge(1, TierDRAM, 80); err != nil {
		t.Fatal(err)
	}
	q.Transfer(1, TierDRAM, TierNVMe, 80) // must not fail: moves correct via eviction, not refusal
	if got := q.Usage(1, TierDRAM); got != 0 {
		t.Fatalf("dram after transfer: %d", got)
	}
	if got := q.Usage(1, TierNVMe); got != 80 {
		t.Fatalf("nvme after transfer: %d (transfer may overshoot quota by design)", got)
	}
	if got := q.OverRatio(1, TierNVMe); got != 8000 {
		t.Fatalf("over-ratio thousandths: %d, want 8000", got)
	}
}

// TestSetQuotaTakesEffectWithoutReload: a limit update must be ONE phase —
// the old SetQuota-then-Reload protocol silently left the ENFORCED limit
// stale forever when a call site forgot the second half (customer-visible
// ERR_QUOTA_BYTES despite a raised quota; a cut that never enforced).
func TestSetQuotaTakesEffectWithoutReload(t *testing.T) {
	r := NewRegistry("a", 1, "tok")
	if !r.SetQuota("a", TierDRAM, 100) {
		t.Fatal("SetQuota")
	}
	q := NewQuotas(r)
	if err := q.Charge(1, TierDRAM, 100); err != nil {
		t.Fatal(err)
	}
	if err := q.Charge(1, TierDRAM, 50); err == nil {
		t.Fatal("over-quota charge admitted")
	}

	// Raise — NO Reload call anywhere.
	if !r.SetQuota("a", TierDRAM, 200) {
		t.Fatal("SetQuota raise")
	}
	if err := q.Charge(1, TierDRAM, 50); err != nil {
		t.Fatalf("raised quota not enforced without Reload: %v", err)
	}

	// Cut — usage (150) is now over the 10-byte limit; nothing new admits.
	if !r.SetQuota("a", TierDRAM, 10) {
		t.Fatal("SetQuota cut")
	}
	if err := q.Charge(1, TierDRAM, 5); err == nil {
		t.Fatal("quota cut not enforced without Reload")
	}
}

// TestSetQuotaInsideFirstTouchWindowIsNotLost: domain() snapshots the
// registry BEFORE publishing the nsUsage entry into q.by. A SetQuota landing
// in that window notifies reloadNS against a map with no entry to refresh —
// without the post-publish re-read, the cut below would be silently lost and
// the entry would enforce the stale pre-cut limit until the next SetQuota.
func TestSetQuotaInsideFirstTouchWindowIsNotLost(t *testing.T) {
	r := NewRegistry("a", 1, "tok")
	if !r.SetQuota("a", TierDRAM, 1000) {
		t.Fatal("SetQuota")
	}
	q := NewQuotas(r)

	firstTouchHookForTest = func() {
		firstTouchHookForTest = nil // the hook's own SetQuota charges nothing — fire once
		if !r.SetQuota("a", TierDRAM, 10) {
			t.Error("SetQuota inside the window")
		}
	}
	defer func() { firstTouchHookForTest = nil }()

	// First touch: mints the domain with the hook's SetQuota landing between
	// the registry snapshot and the q.by publish.
	if err := q.Charge(1, TierDRAM, 5); err != nil {
		t.Fatal(err)
	}
	if got := q.Limit(1, TierDRAM); got != 10 {
		t.Fatalf("enforced limit %d after an in-window SetQuota, want 10 — the cut was lost", got)
	}
	if err := q.Charge(1, TierDRAM, 100); err == nil {
		t.Fatal("charge over the in-window quota cut admitted — stale limit enforced")
	}
}

// TestReadAccessorsDoNotInsert: the read-only accessors must never mint a
// usage domain — a stats path iterating stale ids (or any unvalidated id)
// would otherwise grow q.by without bound, one write lock per miss.
func TestReadAccessorsDoNotInsert(t *testing.T) {
	r := NewRegistry("a", 1, "tok")
	if !r.SetQuota("a", TierDRAM, 100) {
		t.Fatal("SetQuota")
	}
	q := NewQuotas(r)

	_ = q.Usage(1, TierDRAM)
	_ = q.OverRatio(1, TierDRAM)
	_ = q.WouldExceed(1, TierDRAM, 10)
	_ = q.Usage(999, TierS3) // unseen id — the stale-scan shape
	_ = q.Limit(999, TierNVMe)
	if got := q.Limit(1, TierDRAM); got != 100 {
		t.Fatalf("Limit(uncharged ns) = %d, want 100 from the registry", got)
	}
	if !q.WouldExceed(1, TierDRAM, 150) {
		t.Fatal("WouldExceed ignored the configured limit before first Charge")
	}
	if q.WouldExceed(1, TierDRAM, 50) {
		t.Fatal("WouldExceed false positive under the limit")
	}

	q.mu.RLock()
	n := len(q.by)
	q.mu.RUnlock()
	if n != 0 {
		t.Fatalf("read-only accessors inserted %d map entries", n)
	}
}

// TestLookupReturnsDetachedCopy pins the API contract: mutating a Lookup
// result must never write through into registry state (the aliased-pointer
// race QuotaSnapshot exists to prevent).
func TestLookupReturnsDetachedCopy(t *testing.T) {
	r := NewRegistry("a", 1, "tok")
	if !r.SetQuota("a", TierDRAM, 100) {
		t.Fatal("SetQuota")
	}
	ns, ok := r.Lookup(1)
	if !ok {
		t.Fatal("lookup")
	}
	ns.Quota[TierDRAM] = 999_999
	if snap, _ := r.QuotaSnapshot(1); snap[TierDRAM] != 100 {
		t.Fatalf("Lookup handed out registry-owned state: quota now %d", snap[TierDRAM])
	}
}
