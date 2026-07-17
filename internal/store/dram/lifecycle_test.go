package dram

import (
	"sync"
	"testing"
	"time"

	"github.com/kvstash/kvblockd/internal/protocol"
)

// pinClock pins the lifecycle clock for a test and restores it on cleanup.
func pinClock(t *testing.T, start int64) *int64 {
	t.Helper()
	cur := start
	old := lifeNow
	lifeNow = func() int64 { return cur }
	t.Cleanup(func() { lifeNow = old })
	return &cur
}

func msToNanos(ms int64) int64 { return ms * int64(time.Millisecond) }

// TestBlockRefAcquireRelease pins the refcount CAS semantics: Acquire fails
// at ≤0 (no resurrection), Release reports exactly one zero-crossing, and the
// count survives concurrent hammering under -race.
func TestBlockRefAcquireRelease(t *testing.T) {
	ref := &BlockRef{}
	if ref.Acquire() {
		t.Fatal("Acquire succeeded on an unpublished (refcount=0) block")
	}
	ref.Refcount.Store(1) // publish (the index ref)
	if !ref.Acquire() {
		t.Fatal("Acquire failed on a live block")
	}
	if ref.Release() {
		t.Fatal("first Release reported freed with the index ref still held")
	}
	if !ref.Release() { // drops the index ref → zero
		t.Fatal("last Release did not report freed")
	}
	if ref.Acquire() {
		t.Fatal("Acquire resurrected a torn-down block")
	}

	// Concurrency: N acquirers against a publish/teardown cycle stay ≥0 and
	// produce exactly one freed=true per zero-crossing.
	ref2 := &BlockRef{}
	ref2.Refcount.Store(1)
	var wg sync.WaitGroup
	freed := make(chan struct{}, 64)
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 1000; j++ {
				if ref2.Acquire() {
					if ref2.Release() {
						freed <- struct{}{}
					}
				}
			}
		}()
	}
	wg.Wait()
	// The index ref (the initial 1) was held throughout, so no paired
	// Acquire/Release above can have crossed zero — the drop below MUST be
	// the one and only zero-crossing.
	if !ref2.Release() {
		t.Fatal("index-ref drop did not report the free")
	}
	close(freed)
	if n := len(freed); n != 0 {
		t.Fatalf("paired Acquire/Release crossed zero %d times while the index ref was live", n)
	}
}

// TestLeaseTTLExpiry pins Leased/Expired against the fake clock, including
// the lease clamp and the lazy-expiry rule.
func TestLeaseTTLExpiry(t *testing.T) {
	clk := pinClock(t, msToNanos(1_000_000))
	l := newLifecycle(5000, 60000, 0)
	ref := &BlockRef{}

	// Auto-lease: 5s default.
	l.GrantReadLease(ref)
	if !ref.Leased(*clk) {
		t.Fatal("no lease after GrantReadLease")
	}
	*clk += msToNanos(5001)
	if ref.Leased(*clk) {
		t.Fatal("lease survived past lease_default")
	}

	// Explicit LEASE clamps to lease_max (request 10min → 60s).
	l.Lease(ref, 600_000)
	*clk += msToNanos(60_001)
	if ref.Leased(*clk) {
		t.Fatal("lease survived past lease_max clamp")
	}

	// RELEASE drops early.
	l.Lease(ref, 30_000)
	l.ReleaseLease(ref)
	if ref.Leased(*clk) {
		t.Fatal("lease survived RELEASE")
	}

	// TTL: Touch extends; ttl=0 leaves it; expiry = eligible, not gone.
	l.Touch(ref, 1000)
	if ref.Expired(*clk) {
		t.Fatal("expired immediately after Touch(ttl)")
	}
	*clk += msToNanos(1001)
	if !ref.Expired(*clk) {
		t.Fatal("not expired after ttl lapsed")
	}
	l.Touch(ref, 0) // recency only — must NOT resurrect the TTL
	if !ref.Expired(*clk) {
		t.Fatal("Touch(0) reset the TTL")
	}
}

// TestDeleteGatingTruthTable pins the §3.7 matrix exactly.
func TestDeleteGatingTruthTable(t *testing.T) {
	pinClock(t, msToNanos(1_000_000))
	l := newLifecycle(5000, 60000, 0)

	cases := []struct {
		name       string
		lease      bool
		soft, hard bool
		force      bool
		want       protocol.Status
	}{
		{"plain", false, false, false, false, protocol.StatusOK},
		{"plain-force", false, false, false, true, protocol.StatusOK},
		{"leased", true, false, false, false, protocol.StatusErrLeased},
		{"leased-force", true, false, false, true, protocol.StatusOK},
		{"soft", false, true, false, false, protocol.StatusErrPinned},
		{"soft-force", false, true, false, true, protocol.StatusOK},
		{"hard", false, false, true, false, protocol.StatusErrPinned},
		{"hard-force", false, false, true, true, protocol.StatusErrPinned},
		// canDelete's documented order is hard → force → lease → soft, so a
		// leased+soft block reports the LEASE first. This row pins that
		// order: a reorder of canDelete is a wire-visible change.
		{"leased+soft", true, true, false, false, protocol.StatusErrLeased},
		{"leased+hard-force", true, false, true, true, protocol.StatusErrPinned},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ref := &BlockRef{Len: 100}
			if tc.lease {
				l.Lease(ref, 30_000)
			}
			if tc.soft {
				if st := l.Pin(ref, false); st != protocol.StatusOK {
					t.Fatal(st)
				}
			}
			if tc.hard {
				if st := l.Pin(ref, true); st != protocol.StatusOK {
					t.Fatal(st)
				}
			}
			if got := l.canDelete(ref, tc.force); got != tc.want {
				t.Fatalf("got %s, want %s", got, tc.want)
			}
		})
	}
}

// TestPinQuota pins §3.6's quota semantics: ONLY hard pins debit the
// per-namespace cap (soft pins are quota-free), every transition into hard
// passes the gate (incl. soft→hard upgrades), and leaving hard refunds.
func TestPinQuota(t *testing.T) {
	pinClock(t, msToNanos(1_000_000))
	l := newLifecycle(5000, 60000, 1000) // 1000-byte cap

	a := &BlockRef{NamespaceID: 7, Len: 600}
	b := &BlockRef{NamespaceID: 7, Len: 600}
	c := &BlockRef{NamespaceID: 8, Len: 600} // different namespace

	if st := l.Pin(a, true); st != protocol.StatusOK {
		t.Fatalf("first hard pin: %s", st)
	}
	if st := l.Pin(b, true); st != protocol.StatusErrPinQuota {
		t.Fatalf("over-cap hard pin: got %s, want ERR_PIN_QUOTA", st)
	}
	if b.Pinned() {
		t.Fatal("rejected pin still set the flag")
	}
	// Soft pins are QUOTA-FREE (§3.6: emergency-droppable, never rejected).
	if st := l.Pin(b, false); st != protocol.StatusOK {
		t.Fatalf("soft pin at a full cap must succeed: %s", st)
	}
	if got := l.pinnedBytes()[7]; got != 600 {
		t.Fatalf("soft pin charged the quota: ns7 = %d, want 600", got)
	}
	// A soft→hard UPGRADE passes the gate (the cap stays unbypassable).
	if st := l.Pin(b, true); st != protocol.StatusErrPinQuota {
		t.Fatalf("over-cap soft→hard upgrade: got %s, want ERR_PIN_QUOTA", st)
	}
	if !b.Pinned() || b.HardPinned() {
		t.Fatal("failed upgrade must leave the soft pin intact")
	}
	if st := l.Pin(c, true); st != protocol.StatusOK {
		t.Fatalf("other-namespace hard pin blocked: %s", st)
	}
	// Hard→soft downgrade refunds; the freed budget admits the upgrade.
	l.Pin(a, false)
	if got := l.pinnedBytes()[7]; got != 0 {
		t.Fatalf("downgrade did not refund: ns7 = %d, want 0", got)
	}
	if st := l.Pin(b, true); st != protocol.StatusOK {
		t.Fatalf("upgrade after refund: %s", st)
	}
	// Unpin of a hard pin refunds exactly once.
	l.Unpin(b)
	l.Unpin(a) // soft — no refund, must not go negative (kvbdebug asserts)
	if got := l.pinnedBytes()[7]; got != 0 {
		t.Fatalf("final ns7 pinned bytes = %d, want 0", got)
	}
}

// TestCanEvictLadder pins the evictor pre-filter in its REAL context: a
// block resident in the index (Refcount==1, exactly the index's own ref).
// The idle case is the load-bearing row — an idle, unleased, unpinned
// resident block MUST be evictable, else the Week-4 evictor finds nothing.
func TestCanEvictLadder(t *testing.T) {
	clk := pinClock(t, msToNanos(1_000_000))
	l := newLifecycle(5000, 60000, 0)

	ref := &BlockRef{Len: 10}
	ref.Refcount.Store(1) // resident: the index's own reference

	if !ref.CanEvict(*clk) {
		t.Fatal("idle resident block (refcount=1, no lease, no pin) not evictable")
	}
	// An in-flight reader (refcount 2) blocks the pre-filter.
	if !ref.Acquire() {
		t.Fatal("acquire")
	}
	if ref.CanEvict(*clk) {
		t.Fatal("CanEvict true with a live reader ref")
	}
	ref.Release()
	l.GrantReadLease(ref)
	if ref.CanEvict(*clk) {
		t.Fatal("CanEvict true under a live lease")
	}
	*clk += msToNanos(5001) // lease expires
	if !ref.CanEvict(*clk) {
		t.Fatal("CanEvict false after lease expiry")
	}
	l.Pin(ref, false)
	if ref.CanEvict(*clk) {
		t.Fatal("CanEvict true under a soft pin (normal pressure)")
	}
	l.Unpin(ref)
	l.Pin(ref, true)
	if ref.CanEvict(*clk) {
		t.Fatal("CanEvict true under a hard pin")
	}
}

// TestIndexBasics: put/get/delete/lost-race + ExistsPrefix parity + expired-
// still-present, with aggregate-only assertions (never shard placement).
func TestIndexBasics(t *testing.T) {
	clk := pinClock(t, msToNanos(1_000_000))
	idx := NewIndex()
	k := func(b byte) Key { var h [32]byte; h[0] = b; return Key{NS: 1, Hash: h} }

	r1 := &BlockRef{Len: 1}
	if _, inserted := idx.Put(k(1), r1); !inserted {
		t.Fatal("fresh insert reported existing")
	}
	if existing, inserted := idx.Put(k(1), &BlockRef{}); inserted || existing != r1 {
		t.Fatal("lost-race insert did not return the existing ref")
	}
	if got, ok := idx.Get(k(1)); !ok || got != r1 {
		t.Fatal("Get after Put")
	}

	// Expired-TTL still present (lazy expiry).
	r1.TTLUntil.Store(*clk - 1)
	if _, ok := idx.Get(k(1)); !ok {
		t.Fatal("expired block vanished from the index (must be lazy)")
	}

	// ExistsPrefix parity with the wire contract.
	idx.Put(k(2), &BlockRef{})
	idx.Put(k(4), &BlockRef{})
	keys := [][32]byte{k(1).Hash, k(2).Hash, k(3).Hash, k(4).Hash}
	n, per := idx.ExistsPrefix(1, keys, true)
	if n != 2 {
		t.Fatalf("nConsecutive = %d, want 2", n)
	}
	want := []bool{true, true, false, true}
	for i := range want {
		if per[i] != want[i] {
			t.Fatalf("perKey[%d] = %v, want %v", i, per[i], want[i])
		}
	}
	if n2, per2 := idx.ExistsPrefix(1, keys, false); n2 != 2 || per2 != nil {
		t.Fatalf("no-bitmap: n=%d per=%v", n2, per2)
	}
	// Namespace isolation.
	if _, ok := idx.Get(Key{NS: 2, Hash: k(1).Hash}); ok {
		t.Fatal("key visible across namespaces")
	}

	if ref, ok := idx.Delete(k(1)); !ok || ref != r1 {
		t.Fatal("Delete did not return the ref")
	}
	if _, ok := idx.Get(k(1)); ok {
		t.Fatal("Get after Delete")
	}
}
