package dram

import (
	"sync"
	"sync/atomic"
	"time"

	"github.com/kvstash/kvblockd/internal/protocol"
)

// The Mooncake lease/pin/TTL ladder (PROTOCOL.md §6), verbatim semantics:
//
//	protection under pressure, weakest → strongest:
//	  unpinned+expired → unpinned → soft-pinned (quota emergency only)
//	  → leased (never while the lease is valid) → hard-pinned (never)
//
// Leases and TTLs are atomics on the BlockRef (readers race freely); pin
// flags mutate only under the index shard lock (Index.WithShardLock).

// lifecycle owns the lease/pin/TTL rules plus the per-namespace pinned-bytes
// accounting (the Week-3 slice of tenancy: the cap check and ERR_PIN_QUOTA
// exist now; the full quota machinery is a later week).
type lifecycle struct {
	leaseDefaultMS uint32
	leaseMaxMS     uint32
	pinnedCap      int64        // per-namespace pinned-bytes cap (0 = unlimited)
	now            func() int64 // instance clock (unix nanos) — Params.Now seam

	mu     sync.Mutex // guards pinned (cold path: pin/unpin/delete only)
	pinned map[uint32]int64

	// ttlBlocks counts blocks whose TTLUntil is set — the evictor's hint to
	// skip the full-index expired sweep entirely in TTL-free workloads
	// (a hint, not a gate: an unnecessary sweep is only wasted work).
	ttlBlocks atomic.Int64
}

func newLifecycle(leaseDefaultMS, leaseMaxMS uint32, pinnedCap int64, now func() int64) *lifecycle {
	return &lifecycle{
		leaseDefaultMS: leaseDefaultMS,
		leaseMaxMS:     leaseMaxMS,
		pinnedCap:      pinnedCap,
		now:            now,
		pinned:         make(map[uint32]int64),
	}
}

// clampLeaseMS resolves a requested lease duration: 0 means "the default",
// anything above lease_max is clamped down (§3.5 LEASE).
func (l *lifecycle) clampLeaseMS(ttlMS uint32) uint32 {
	if ttlMS == 0 {
		ttlMS = l.leaseDefaultMS
	}
	if ttlMS > l.leaseMaxMS {
		ttlMS = l.leaseMaxMS
	}
	return ttlMS
}

// GrantReadLease is the §3.3 auto-lease every OK GET carries: the block
// cannot be evicted or reclaimed mid-transfer or before the client finishes
// copying. Returns the now it drew so the caller can reuse the same instant
// (the policy Touch hook) without a second clock read.
//
// The grant is MONOTONIC (extendLease): leases are grant/extend semantics
// (§3.5; RELEASE is the only shortening op), so a 5s auto-grant must never
// TRUNCATE a live 60s explicit lease — the eviction protection a client
// paid for would silently evaporate on its own read.
func (l *lifecycle) GrantReadLease(ref *BlockRef) int64 {
	now := l.now()
	extendLease(ref, now+int64(l.leaseDefaultMS)*int64(time.Millisecond))
	ref.LastAccess.Store(now)
	return now
}

// extendLease raises LeaseUntil to at least until (never lowers it).
func extendLease(ref *BlockRef, until int64) {
	for {
		cur := ref.LeaseUntil.Load()
		if cur >= until {
			return
		}
		if ref.LeaseUntil.CompareAndSwap(cur, until) {
			return
		}
	}
}

// Touch (§3.5 sub-op 0) bumps recency and, with ttlMS>0, extends the TTL.
// Metadata-only: never triggers restores, never grants leases.
func (l *lifecycle) Touch(ref *BlockRef, ttlMS uint32) {
	now := l.now()
	ref.LastAccess.Store(now)
	if ttlMS > 0 {
		if ref.TTLUntil.Swap(now+int64(ttlMS)*int64(time.Millisecond)) == 0 {
			l.ttlBlocks.Add(1) // first TTL on this block
		}
	}
}

// noteTTLGone decrements the TTL hint when a TTL-bearing block leaves the
// index (delete/evict).
func (l *lifecycle) noteTTLGone(ref *BlockRef) {
	if ref.TTLUntil.Load() != 0 {
		l.ttlBlocks.Add(-1)
	}
}

// Lease (§3.5 sub-op 1) grants/extends the eviction-protection lease —
// the same mechanism GET auto-grants, with an explicit clamped duration.
// Monotonic like the auto-grant: "grants/extends", never shortens (RELEASE
// is the shortening op).
func (l *lifecycle) Lease(ref *BlockRef, ttlMS uint32) {
	d := l.clampLeaseMS(ttlMS)
	now := l.now()
	extendLease(ref, now+int64(d)*int64(time.Millisecond))
	ref.LastAccess.Store(now)
}

// ReleaseLease (§3.5 sub-op 2) drops the lease early.
func (l *lifecycle) ReleaseLease(ref *BlockRef) { ref.LeaseUntil.Store(0) }

// Pin (§3.6) sets the requested pin level. Per the spec, ONLY hard pins debit
// the namespace's pinned-bytes budget (soft pins are quota-free — under
// quota emergency they are DROPPED, never rejected): the charge happens on
// every transition INTO hard (including a soft→hard upgrade — this is the
// check that keeps the cap unbypassable), exhaustion answers ERR_PIN_QUOTA
// and the flags are left unchanged, and a hard→soft downgrade refunds.
// Caller must hold the index shard lock (PinFlags convention).
func (l *lifecycle) Pin(ref *BlockRef, hard bool) protocol.Status {
	wasHard := ref.PinFlags&pinHardBit != 0
	switch {
	case hard && !wasHard: // into hard: the quota gate
		l.mu.Lock()
		if l.pinnedCap > 0 && l.pinned[ref.NamespaceID]+int64(ref.Len) > l.pinnedCap {
			l.mu.Unlock()
			return protocol.StatusErrPinQuota
		}
		l.pinned[ref.NamespaceID] += int64(ref.Len)
		l.mu.Unlock()
		ref.PinFlags = pinHardBit
	case hard: // hard→hard: idempotent, already charged
		ref.PinFlags = pinHardBit
	case wasHard: // hard→soft downgrade: refund the charge
		l.refund(ref)
		ref.PinFlags = pinSoftBit
	default: // soft (fresh or soft→soft): quota-free
		ref.PinFlags = pinSoftBit
	}
	return protocol.StatusOK
}

// refund returns a hard pin's bytes to the namespace budget. Caller must
// hold the index shard lock and have verified the hard bit is set.
func (l *lifecycle) refund(ref *BlockRef) {
	l.mu.Lock()
	l.pinned[ref.NamespaceID] -= int64(ref.Len)
	if l.pinned[ref.NamespaceID] < 0 { // defensive; assert in debug
		assertf(false, "dram: pinned-bytes accounting below zero (ns=%d)", ref.NamespaceID)
		l.pinned[ref.NamespaceID] = 0
	}
	l.mu.Unlock()
}

// Unpin (§3.6 sub-op 2) clears BOTH pin levels, refunding the budget iff the
// block was hard-pinned (soft pins never charged). Caller must hold the
// index shard lock.
func (l *lifecycle) Unpin(ref *BlockRef) {
	if ref.PinFlags&pinHardBit != 0 {
		l.refund(ref)
	}
	ref.PinFlags = 0
}

// unpinOnDelete refunds the budget when a pinned block leaves the index
// (delete path). Caller holds the shard lock.
func (l *lifecycle) unpinOnDelete(ref *BlockRef) { l.Unpin(ref) }

// canDelete is the §3.7 gating truth table. Evaluation order is load-bearing:
//
//	hard pin      → ERR_PINNED, always (F_FORCE never overrides hard)
//	force         → OK (overrides lease and soft pin)
//	live lease    → ERR_LEASED
//	soft pin      → ERR_PINNED
//	otherwise     → OK
//
// Caller must hold the index shard lock (reads PinFlags).
func (l *lifecycle) canDelete(ref *BlockRef, force bool) protocol.Status {
	if ref.HardPinned() {
		return protocol.StatusErrPinned
	}
	if force {
		return protocol.StatusOK
	}
	if ref.Leased(l.now()) {
		return protocol.StatusErrLeased
	}
	if ref.Pinned() { // soft (hard handled above)
		return protocol.StatusErrPinned
	}
	return protocol.StatusOK
}

// pinnedBytes reports every namespace's current pinned-byte debit (stats).
func (l *lifecycle) pinnedBytes() map[uint32]int64 {
	l.mu.Lock()
	defer l.mu.Unlock()
	out := make(map[uint32]int64, len(l.pinned))
	for ns, b := range l.pinned {
		if b > 0 {
			out[ns] = b
		}
	}
	return out
}
