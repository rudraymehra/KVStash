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
