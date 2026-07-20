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
// destination-quota would wedge the memory ladder. Only the DRAM quota is
// actively enforced (Charge at publish; the evictor's over-quota-first pass
// runs on DRAM alone) — the NVMe and S3 limits are REPORTING/ADVISORY at
// v0.2: Transfer and Seed land bytes there unchecked and nothing corrects an
// overage (ledgered in docs/IMPROVEMENTS.md with its trigger).
//
// LOCK ORDER: q.mu and the registry's reg.mu are LEAVES — never held across
// each other (or across any store lock). domain/Reload snapshot the registry
// through QuotaSnapshot BEFORE touching q.mu; reg.Each callers must not call
// back into this accountant while the registry lock is held.
type Quotas struct {
	mu  sync.RWMutex
	reg *Registry
	by  map[uint32]*nsUsage
}

type nsUsage struct {
	used [tierCount]atomic.Int64
	// limit: 0 = unlimited; snapshot at first touch, refreshed by SetQuota
	// via Reload. Atomic — Reload's stores race Charge/WouldExceed's bare
	// reads (no q.mu on the hot path).
	limit [tierCount]atomic.Int64
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
	// Registry snapshot BEFORE q.mu (the leaf rule): holding q.mu into a
	// registry read once cycled against reg.Each→Usage and wedged both
	// planes. QuotaSnapshot also copies under reg.mu, so the limits never
	// race SetQuota's locked write.
	var limits [tierCount]int64
	if q.reg != nil {
		if l, found := q.reg.QuotaSnapshot(ns); found {
			limits = l
		}
	}
	q.mu.Lock()
	defer q.mu.Unlock()
	if u, ok = q.by[ns]; ok {
		return u
	}
	u = &nsUsage{}
	for t := range limits {
		u.limit[t].Store(limits[t])
	}
	q.by[ns] = u
	return u
}

// Reload re-snapshots limits from the registry (admin SetQuota path). The
// map snapshot releases q.mu before any registry read (the leaf rule);
// nsUsage pointers are stable once published, so the limit stores need no
// lock of their own.
func (q *Quotas) Reload() {
	if q.reg == nil {
		return
	}
	q.mu.RLock()
	snap := make(map[uint32]*nsUsage, len(q.by))
	for ns, u := range q.by {
		snap[ns] = u
	}
	q.mu.RUnlock()
	for ns, u := range snap {
		if l, found := q.reg.QuotaSnapshot(ns); found {
			for t := range l {
				u.limit[t].Store(l[t])
			}
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
	limit := u.limit[tier].Load() // one snapshot for the whole loop — a mid-loop Reload must not tear the check
	if limit <= 0 {               // unlimited
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
	if bal := u.used[tier].Add(-n); bal < 0 {
		quotaUnderflow(ns, tier, bal)
		// Heal: never leave a negative balance to silently widen the quota.
		// CAS, not Add — two racing underflow heals stacking their adds once
		// minted a positive phantom balance; a failed CAS means another op
		// moved the counter and its own heal (or a later one) resolves it.
		u.used[tier].CompareAndSwap(bal, 0)
	}
}

// Seed charges n bytes UNCHECKED (recovery re-seeding a restarted process's
// ledger — refusing recovery over quota would lose data). A tenant seeded
// over its NVMe limit stays over: that tier's quota is reporting-only (see
// the type comment).
func (q *Quotas) Seed(ns uint32, tier Tier, n int64) {
	if n <= 0 || tier < 0 || tier >= tierCount {
		return
	}
	q.domain(ns).used[tier].Add(n)
}

// Transfer moves n bytes from one tier to another (demote DRAM→NVMe, the
// retire-flip NVMe→S3; promotion is NOT a transfer — the block goes
// dual-resident and each tier keeps its own charge). It NEVER fails — see
// the type comment.
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
	limit := u.limit[tier].Load()
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
	return q.domain(ns).limit[tier].Load()
}

// OverRatio reports usage/quota as thousandths (0 when unlimited) — the
// evictor's over-quota-first ordering key, integer to stay allocation-free.
func (q *Quotas) OverRatio(ns uint32, tier Tier) int64 {
	u := q.domain(ns)
	limit := u.limit[tier].Load()
	if limit <= 0 {
		return 0
	}
	return u.used[tier].Load() * 1000 / limit
}
