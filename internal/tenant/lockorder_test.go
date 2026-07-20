package tenant

import (
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// Regression 1 (data race): Quotas.Reload writes u.limit while
// Charge/WouldExceed/Limit read u.limit[tier] with no q.mu held at the read
// — the limits must be atomics. Mirrors: admin POST /v1/quota racing
// data-plane PUT admission.
func TestLimitReloadRace(t *testing.T) {
	reg := NewRegistry("a", 1, "tok")
	reg.SetQuota("a", TierDRAM, 1<<20)
	q := NewQuotas(reg)
	_ = q.Charge(1, TierDRAM, 1) // materialize the domain (hot path thereafter)

	var wg sync.WaitGroup
	stop := make(chan struct{})
	wg.Add(2)
	go func() { // data plane
		defer wg.Done()
		for {
			select {
			case <-stop:
				return
			default:
			}
			_ = q.Charge(1, TierDRAM, 64)
			q.Refund(1, TierDRAM, 64)
			_ = q.WouldExceed(1, TierDRAM, 128)
		}
	}()
	go func() { // admin plane
		defer wg.Done()
		for i := 0; i < 20000; i++ {
			reg.SetQuota("a", TierDRAM, int64(1<<20+i))
			q.Reload()
		}
	}()
	time.Sleep(300 * time.Millisecond)
	close(stop)
	wg.Wait()
}

// Regression 2 (data race): Quotas.domain once copied ns.Quota (the [3]int64
// array) AFTER Registry.Lookup released the registry RLock, while
// Registry.SetQuota writes ns.Quota[tier] under the registry write lock —
// the copy must happen inside the lock (Registry.QuotaSnapshot).
func TestNamespaceQuotaCopyRace(t *testing.T) {
	reg := NewRegistry("a", 1, "tok")
	var wg sync.WaitGroup
	stop := make(chan struct{})
	wg.Add(2)
	go func() {
		defer wg.Done()
		for {
			select {
			case <-stop:
				return
			default:
			}
			q := NewQuotas(reg)
			_ = q.Charge(1, TierDRAM, 1) // domain(): first-touch limit snapshot
		}
	}()
	go func() {
		defer wg.Done()
		for i := 0; i < 20000; i++ {
			reg.SetQuota("a", TierDRAM, int64(i))
		}
	}()
	time.Sleep(300 * time.Millisecond)
	close(stop)
	wg.Wait()
}

// Regression 3 (deadlock): Reload once held q.mu across reg.Lookup — the
// reg.mu <-> q.mu inversion whose full three-party cycle (scrape holding
// reg.mu.R into q.Usage; Reload holding q.mu.W into the registry; a second
// admin mutation pending reg.mu.W) wedged every plane at once. The q-side
// leaf discipline is pinned HERE (Reload/domain never hold q.mu into the
// registry); the caller-side half — never call q.* under reg.Each — is
// pinned where those callers live (the metrics collector's deadlock test).
// Shape: hammer Reload against a first-touch storm (domain's q.mu.W path)
// and a registry writer; stall of every loop = the inversion is back.
func TestReloadNeverHoldsAccountantLockIntoRegistry(t *testing.T) {
	reg := NewRegistry("a", 1, "tok")
	for i := uint32(2); i <= 20; i++ {
		_ = reg.Add(&Namespace{ID: i, Name: string(rune('a' + i)), TokenHash: [32]byte{byte(i)}})
	}
	reg.SetQuota("a", TierDRAM, 1<<20)
	q := NewQuotas(reg)

	var progress atomic.Int64
	stop := make(chan struct{})
	var wg sync.WaitGroup
	worker := func(fn func()) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-stop:
					return
				default:
				}
				fn()
				progress.Add(1)
			}
		}()
	}
	// A: data-plane first touches (domain's slow path takes q.mu.W after its
	// registry snapshot) — a fresh accountant each lap keeps first-touch hot.
	worker(func() {
		fresh := NewQuotas(reg)
		for i := uint32(1); i <= 20; i++ {
			_ = fresh.Charge(i, TierDRAM, 1)
		}
	})
	// B: admin SetQuota handler shape (SetQuota then Reload).
	worker(func() {
		reg.SetQuota("a", TierDRAM, 1<<20)
		q.Reload()
	})
	// C: a second concurrent admin mutation keeps a registry writer pending.
	worker(func() {
		reg.SetQuota("a", TierDRAM, 2<<20)
	})

	budget := 5 * time.Second
	if testing.Short() {
		budget = 2 * time.Second
	}
	deadline := time.After(budget)
	last := int64(-1)
	stalls := 0
	for {
		select {
		case <-deadline:
			close(stop)
			wg.Wait()
			return // every loop kept making progress
		default:
		}
		time.Sleep(200 * time.Millisecond)
		cur := progress.Load()
		if cur == last {
			stalls++
			if stalls >= 10 { // 2s with zero progress across all three loops
				buf := make([]byte, 1<<20)
				n := runtime.Stack(buf, true)
				t.Fatalf("DEADLOCK: no progress for 2s (count=%d)\n%s", cur, buf[:n])
			}
		} else {
			stalls = 0
		}
		last = cur
	}
}
