package tenant

import (
	"fmt"
	"sync"
	"sync/atomic"
)

// ErrQuota is returned by Charge when the admission would exceed the
// tenant's tier quota — the server maps it to ERR_QUOTA_BYTES (0x30) at
// PUT_STREAM BEGIN.
var ErrQuota = fmt.Errorf("tenant: tier quota exceeded")

// Quotas is the per-(namespace, tier) byte accountant. Admission is a CAS
// loop — no mutex on the hot path — with the documented invariant:
//
//	usage(ns, tier) ≤ quota(ns, tier) + one max-block in-flight slack (I3)
//
// The slack exists because Charge happens at BEGIN (reserve) while racing
// BEGINs each individually observed headroom; a lost race costs at most one
// block per racer, and Refund on EVERY exit path (abort, failed commit,
// delete, evict, reclaim) is what keeps the counter honest. Tier MOVES
// (demote, spill) use Transfer, which never fails: refusing a demotion for
// destination-quota would wedge the memory ladder — instead the evictor's
// over-quota-first pass corrects the destination tier on its next cycle.
type Quotas struct {
	mu  sync.RWMutex
	reg *Registry
	by  map[uint32]*nsUsage
}

type nsUsage struct {
	used  [tierCount]atomic.Int64
	limit [tierCount]int64 // 0 = unlimited; snapshot at first touch, refreshed by SetQuota via Reload
}

// NewQuotas builds the accountant over a registry. Namespaces added to the
// registry later are picked up lazily on first Charge.
func NewQuotas(reg *Registry) *Quotas {
	return &Quotas{reg: reg, by: make(map[uint32]*nsUsage)}
}

func (q *Quotas) domain(ns uint32) *nsUsage {
	q.mu.RLock()
	u, ok := q.by[ns]
	q.mu.RUnlock()
	if ok {
		return u
	}
	q.mu.Lock()
	defer q.mu.Unlock()
	if u, ok = q.by[ns]; ok {
		return u
	}
	u = &nsUsage{}
	if q.reg != nil {
		if n, found := q.reg.Lookup(ns); found {
			u.limit = n.Quota
		}
	}
	q.by[ns] = u
	return u
}

// Reload re-snapshots limits from the registry (admin SetQuota path).
func (q *Quotas) Reload() {
	q.mu.Lock()
	defer q.mu.Unlock()
	for ns, u := range q.by {
		if q.reg == nil {
			continue
		}
		if n, found := q.reg.Lookup(ns); found {
			u.limit = n.Quota
		}
	}
}

// Charge admits n bytes into (ns, tier) or returns ErrQuota. The CAS loop:
// load; headroom check against the loaded value; CompareAndSwap; retry on
// contention. No mutex, no lost updates, and the check-and-add is atomic —
// a plain Add-then-check would briefly overshoot and need a compensating
// Sub that races readers.
func (q *Quotas) Charge(ns uint32, tier Tier, n int64) error {
	if n <= 0 || tier < 0 || tier >= tierCount {
		return nil
	}
	u := q.domain(ns)
	limit := u.limit[tier]
	if limit <= 0 { // unlimited
		u.used[tier].Add(n)
		return nil
	}
	for {
		cur := u.used[tier].Load()
		if cur+n > limit {
			return ErrQuota
		}
		if u.used[tier].CompareAndSwap(cur, cur+n) {
			return nil
		}
	}
}

// Refund returns n bytes to (ns, tier) — abort, failed commit, delete,
// evict, reclaim. Clamps at zero in release builds rather than going
// negative (a double-refund is an accounting bug, not a corruption risk;
// the kvbdebug sibling asserts).
func (q *Quotas) Refund(ns uint32, tier Tier, n int64) {
	if n <= 0 || tier < 0 || tier >= tierCount {
		return
	}
	u := q.domain(ns)
	if now := u.used[tier].Add(-n); now < 0 {
		quotaUnderflow(ns, tier, now)
		// Heal: never leave a negative balance to silently widen the quota.
		u.used[tier].Add(-now)
	}
}

// Transfer moves n bytes from one tier to another (demote DRAM→NVMe, spill
// NVMe→S3, promote back). It NEVER fails — see the type comment.
func (q *Quotas) Transfer(ns uint32, from, to Tier, n int64) {
	if n <= 0 {
		return
	}
	u := q.domain(ns)
	if to >= 0 && to < tierCount {
		u.used[to].Add(n)
	}
	q.Refund(ns, from, n)
}

// WouldExceed is the ADVISORY headroom probe (PUT_STREAM BEGIN): true when
// admitting n bytes now would break the quota. It reserves nothing — the
// binding admission is Charge at publish; the gap between the two is the
// documented one-block slack (I3).
func (q *Quotas) WouldExceed(ns uint32, tier Tier, n int64) bool {
	if n <= 0 || tier < 0 || tier >= tierCount {
		return false
	}
	u := q.domain(ns)
	limit := u.limit[tier]
	return limit > 0 && u.used[tier].Load()+n > limit
}

// Usage reports the current bytes charged to (ns, tier).
func (q *Quotas) Usage(ns uint32, tier Tier) int64 {
	if tier < 0 || tier >= tierCount {
		return 0
	}
	return q.domain(ns).used[tier].Load()
}

// Limit reports the configured quota (0 = unlimited).
func (q *Quotas) Limit(ns uint32, tier Tier) int64 {
	if tier < 0 || tier >= tierCount {
		return 0
	}
	return q.domain(ns).limit[tier]
}

// OverRatio reports usage/quota as thousandths (0 when unlimited) — the
// evictor's over-quota-first ordering key, integer to stay allocation-free.
func (q *Quotas) OverRatio(ns uint32, tier Tier) int64 {
	u := q.domain(ns)
	limit := u.limit[tier]
	if limit <= 0 {
		return 0
	}
	return u.used[tier].Load() * 1000 / limit
}
