package metrics

import (
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/kvstash/kvblockd/internal/tenant"
)

// Regression (deadlock, reproduced in 2.2s pre-fix): tenantView.Collect once
// called q.Usage/q.Limit INSIDE reg.Each — holding reg.mu.R into the
// accountant while Reload held the accountant's lock into the registry, with
// a second admin mutation pending reg.mu.W. Cycle: scrape holds reg.R waits
// q; Reload holds q waits reg.R behind the pending writer; the writer waits
// reg.W behind the scrape. All three wedge, and every later HELLO and PUT
// wedges behind them. The fix snapshots the registry first and reads the
// accountant only after Each returns (reg.mu and q.mu are lock-graph
// LEAVES). This drives the REAL scrape path (registry Gather → Collect)
// against the real admin shapes and fails on any stall.
func TestTenantScrapeVsQuotaAdminNoDeadlock(t *testing.T) {
	reg := tenant.NewRegistry("a", 1, "tok")
	for i := uint32(2); i <= 20; i++ {
		_ = reg.Add(&tenant.Namespace{ID: i, Name: string(rune('a' + i)), TokenHash: [32]byte{byte(i)}})
	}
	reg.SetQuota("a", tenant.TierDRAM, 1<<20)
	q := tenant.NewQuotas(reg)
	set := New(nil)
	set.SetTenants(reg, q)

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
	// A: the scrape — Gather runs every registered collector, tenantView
	// included, exactly as promhttp does.
	worker(func() {
		if _, err := set.reg.Gather(); err != nil {
			t.Error(err)
		}
	})
	// B: the admin POST /v1/quota shape (SetQuota, then accountant Reload).
	worker(func() {
		reg.SetQuota("a", tenant.TierDRAM, 1<<20)
		q.Reload()
	})
	// C: a second concurrent admin mutation keeps a registry writer pending
	// (writers block NEW readers — the cycle's third edge).
	worker(func() {
		reg.SetQuota("a", tenant.TierDRAM, 2<<20)
	})

	budget := 10 * time.Second
	if testing.Short() {
		budget = 3 * time.Second
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
