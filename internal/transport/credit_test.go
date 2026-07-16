package transport

import (
	"math/rand"
	"testing"
	"time"
)

func TestCreditEscalation(t *testing.T) {
	var w CreditWindow
	w.SetWindow(1000, 500)

	if v := w.Consume(1000); v != ViolationNone {
		t.Fatalf("within window: %v", v)
	}
	// §8 rule 3: one frame of overshoot from a small-positive window is legal.
	if v := w.Consume(500); v != ViolationNone {
		t.Fatalf("sanctioned one-frame overshoot: %v", v)
	}
	// Crossing window+maxFrame is the violation: first Busy, then Fatal.
	if v := w.Consume(1); v != ViolationBusy {
		t.Fatalf("first breach: %v, want Busy", v)
	}
	if v := w.Consume(1); v != ViolationFatal {
		t.Fatalf("second breach: %v, want Fatal", v)
	}
}

func TestCreditGrantAndTake(t *testing.T) {
	var w CreditWindow
	w.SetWindow(1<<20, 1<<16)

	w.Consume(4096)
	w.Grant(4096)
	if !w.PendingGrant() {
		t.Fatal("granted bytes not pending")
	}
	if g := w.TakeGrant(); g != 4096 {
		t.Fatalf("TakeGrant = %d", g)
	}
	if w.PendingGrant() {
		t.Fatal("pending after harvest")
	}
	consumed, granted := w.Totals()
	if consumed != 4096 || granted != 4096 {
		t.Fatalf("totals %d/%d", consumed, granted)
	}
}

// TestCreditConservationProperty hammers the ledger with a random mix of
// served and skipped frames: every consumed byte must be granted back, and
// the ledger must end balanced — the invariant whose violation stalls a real
// client ~payload_len bytes at a time.
func TestCreditConservationProperty(t *testing.T) {
	rng := rand.New(rand.NewSource(time.Now().UnixNano())) //nolint:gosec // property test, crypto strength irrelevant
	var w CreditWindow
	w.SetWindow(1<<30, 1<<20)

	var want uint64
	for i := 0; i < 10_000; i++ {
		n := uint32(rng.Intn(1 << 20)) //nolint:gosec // bounded by Intn
		w.Consume(n)
		w.Grant(n) // served or skipped — either way the bytes come back
		want += uint64(n)
	}
	consumed, granted := w.Totals()
	if consumed != want || granted != want {
		t.Fatalf("ledger imbalance: consumed=%d granted=%d want=%d", consumed, granted, want)
	}

	// Harvest everything; the sum of grants on the wire must equal the total.
	var wire uint64
	for w.PendingGrant() {
		wire += uint64(w.TakeGrant())
	}
	if wire != want {
		t.Fatalf("wire grants %d, want %d", wire, want)
	}
}

func TestStallTimeout(t *testing.T) {
	if got := StallTimeout(30000); got != 60*time.Second {
		t.Fatalf("StallTimeout(30000) = %v, want 60s (2×stream_timeout)", got)
	}
}
