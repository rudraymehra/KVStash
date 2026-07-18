package dram

import "sync/atomic"

// Tier identifies which storage tier a block currently lives in. Week 3 is
// DRAM-only; NVMe/S3 arrive with their tiers.
type Tier uint8

const (
	TierDRAM Tier = 0
	// TierNVMe / TierS3 reserved.
)

// Pin flag bits (PinFlags). NOT atomic: every mutation happens under the
// owning index shard's write lock (Pin/Unpin/Delete all take it), and the
// evictor reads them only inside its DeleteIf gate — also under the shard
// write lock — so the non-atomic representation stays correct by design.
const (
	pinSoftBit = 1 << 0
	pinHardBit = 1 << 1
)

// BlockRef is one committed block's metadata: identity is immutable after the
// index insert publishes it; lifecycle state (lease/TTL/recency/refcount) is
// atomic so readers, lifecycle verbs, and the future evictor race safely.
//
// REFCOUNT MODEL (load-bearing):
//   - The index insert publishes with Refcount=1 — the index's own reference.
//   - Every in-flight GET Acquires (+1) and Releases after the transport has
//     written the bytes (the WriteFrames release callback).
//   - Delete removes the block from the index and drops the index reference;
//     WHICHEVER Release reaches zero frees the arena extent. A reader holding
//     a ref therefore keeps the extent alive across a concurrent Delete — the
//     allocator can never recycle bytes under a live view.
type BlockRef struct {
	// Immutable after publish (written once, before the index insert).
	Offset      uint32 // arena offset in AllocUnit GRANULES (byte off = Offset << AllocUnitShift) — units keep uint32 spanning 16 TiB
	Len         uint32 // exact blob length in BYTES
	NamespaceID uint32
	XXH3        uint64 // xxh3_64 of the blob (C-35 naming)
	allocMeta   uint32 // Allocation.Meta — needed to Free the extent
	ArenaID     uint8  // 0 this week (multi-arena is later)
	Tier        Tier
	PinFlags    uint8 // guarded by the index shard lock (see const doc)

	// Mutable lifecycle state — atomics, unix-nanos (0 = unset).
	LeaseUntil atomic.Int64
	TTLUntil   atomic.Int64
	LastAccess atomic.Int64
	Refcount   atomic.Int32

	// hits counts OK GETs over the block's lifetime (relaxed; approximate
	// under contention is fine). The demotion admission filter reads it:
	// never-read blocks are evicted rather than written to NVMe (SSD
	// endurance — the KVBM freq-filter idea).
	hits atomic.Uint32
}

// Acquire takes a reader reference. It FAILS (returns false) when the count
// is already ≤0 — the block is unpublished or tearing down after its last
// release — so a Get racing a Delete treats the block as a miss instead of
// resurrecting freed memory.
func (r *BlockRef) Acquire() bool {
	for {
		v := r.Refcount.Load()
		if v <= 0 {
			return false
		}
		if r.Refcount.CompareAndSwap(v, v+1) {
			return true
		}
	}
}

// Release drops one reference and reports whether this was the LAST one
// (count hit zero) — the caller must then free the arena extent. Going below
// zero is a caller bug: loud under -tags kvbdebug, clamped no-op in release.
func (r *BlockRef) Release() (freed bool) {
	n := r.Refcount.Add(-1)
	if n < 0 {
		assertf(false, "dram: BlockRef.Release below zero (refcount=%d)", n)
		r.Refcount.Store(0)
		return false
	}
	return n == 0
}

// Leased reports whether an eviction-protection lease is live at now (nanos).
func (r *BlockRef) Leased(now int64) bool { return r.LeaseUntil.Load() > now }

// Expired reports whether the block's TTL has lapsed at now. An expired block
// is EVICTION-ELIGIBLE, not invisible: EXISTS/GET still serve it until the
// evictor reclaims it (lazy expiry — recorded in DESIGN.md).
func (r *BlockRef) Expired(now int64) bool {
	t := r.TTLUntil.Load()
	return t != 0 && t <= now
}

// Pinned reports any pin; HardPinned only the hard level. Callers MUST hold
// the index shard lock (see PinFlags doc) — an unlocked read is a data race
// under the Go memory model, not merely a stale value.
func (r *BlockRef) Pinned() bool     { return r.PinFlags != 0 }
func (r *BlockRef) HardPinned() bool { return r.PinFlags&pinHardBit != 0 }

// CanEvict is the evictor's normal-pressure PRE-FILTER for a block still
// RESIDENT in the index, evaluated under the shard write lock immediately
// before a DeleteIf-style removal: Refcount==1 means exactly the index's own
// reference remains (no in-flight reader), and no lease or pin protects it.
// It is a pre-filter, not the free gate — the safe-to-free mechanism stays
// the whichever-Release-hits-zero rule, so a reader that Acquires between
// this check and the removal is still safe (its release frees the extent).
// evictOne's DeleteIf gate calls this; the quota-emergency widening (soft
// pins become eligible, §6) is the gate's own extra branch, deliberately
// absent from this normal-pressure form.
func (r *BlockRef) CanEvict(now int64) bool {
	return r.Refcount.Load() == 1 && !r.Leased(now) && !r.Pinned()
}
