package dram

import (
	"github.com/kvstash/kvblockd/internal/protocol"
)

// The demotion seam: the three methods the tiered orchestrator uses to move
// a cold block's bytes to the NVMe tier without dram ever knowing NVMe
// exists. The refcount protocol is the GET protocol — the demoter Acquires,
// the writer copies out of the arena view, and CompleteDemotion swaps the
// block's home under the SAME shard-locked DeleteIf gate an eviction uses,
// with the tier publish running INSIDE the gate: a concurrent GET is either
// before the swap (serves DRAM) or after (serves the other tier's index) —
// never a gap, never both frees.

// RefForDemotion is GetRef for the demoter: Acquires a reader reference and
// returns the arena view WITHOUT the §3.3 auto-lease, without a policy
// Touch, and without bumping recency/hit state — a demotion read is not a
// client access and must not make its own victim look hot. Zero-length
// blocks refuse (no extent — nothing to demote; the count sweep owns them).
// hits is the block's lifetime GET count (the SSD-endurance admission input).
func (s *Store) RefForDemotion(ns uint32, key [32]byte) (data []byte, xxh3 uint64, hits uint32, release func(), ok bool) {
	ref, found := s.index.Get(Key{NS: ns, Hash: key})
	if !found || ref.Len == 0 || !ref.Acquire() {
		return nil, 0, 0, nil, false
	}
	data = s.arena.Bytes(uint64(ref.Offset)<<AllocUnitShift, ref.Len)
	rel := func() {
		if ref.Release() {
			s.free(ref)
		}
	}
	return data, ref.XXH3, ref.hits.Load(), rel, true
}

// CompleteDemotion removes the block from this tier after its bytes are
// durable elsewhere, publishing the new home INSIDE the shard-locked gate
// (the zero-width swap). The gate re-verifies:
//
//   - identity: the resident block's xxh3 still matches what was written
//     (a delete+re-put during the demotion queue delay swaps content —
//     refuse rather than orphan the new bytes);
//   - eligibility: CanEvict — refcount==1 (index only), unleased, unpinned.
//     A GET during the queue delay auto-leased the block: it proved itself
//     hot, so it STAYS in DRAM and the caller re-admits it to the policy.
//
// publish (nil = plain eviction — the admission-filter path) runs under the
// shard write lock; it must be cheap and must not call back into this store.
// The caller must have dequeued the key from the eviction policy (Victims'
// contract) — no policy.Remove fires here. Returns false when the gate
// refused or the block vanished.
func (s *Store) CompleteDemotion(ns uint32, key [32]byte, xxh3 uint64, publish func(ref *BlockRef)) bool {
	k := Key{NS: ns, Hash: key}
	ref, _ := s.index.DeleteIf(k, func(ref *BlockRef) protocol.Status {
		if ref.XXH3 != xxh3 || !ref.CanEvict(s.now()) {
			return protocol.StatusErrBusy // aborts the removal — block stays resident
		}
		if publish != nil {
			publish(ref)
		}
		return protocol.StatusOK
	})
	if ref == nil {
		return false
	}
	s.life.noteTTLGone(ref)
	if publish == nil {
		s.evictions.Add(1) // the admission-filter path IS an eviction
	}
	if ref.Release() { // index ref drop OUTSIDE the shard lock — the one free story
		s.free(ref)
	}
	return true
}

// Occupancy exposes the byte-watermark inputs (the demoter's 90% trigger
// reads the same allocator truth the evictor's 95% trigger does).
func (s *Store) Occupancy() (usedBytes, totalBytes int64) {
	used, total := s.occupancy()
	return int64(used) << AllocUnitShift, int64(total) << AllocUnitShift
}
