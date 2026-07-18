package dram

import (
	"encoding/json"
	"sync"
	"sync/atomic"
	"time"

	"github.com/kvstash/kvblockd/internal/eviction"
	"github.com/kvstash/kvblockd/internal/protocol"
)

// Params configures the DRAM tier around an Arena the caller owns.
type Params struct {
	LeaseDefaultMS uint32
	LeaseMaxMS     uint32
	PinnedBytesCap int64 // per-namespace; 0 = unlimited
	// Now is the tier's clock (unix nanos); nil = real time. The fake-clock
	// seam: modeltest drives lease/TTL expiry deterministically through it,
	// and the evictor draws every eligibility decision from the same source.
	Now func() int64
}

// Store is the DRAM tier: arena bytes + O(1) allocator + 256-shard index +
// the lease/pin/TTL ladder, implementing the server's 6-method Store surface
// plus the GetRef (zero-copy release) and lifecycle extensions.
//
// LOCK ORDER (global, never inverted): index shard lock FIRST, then allocMu —
// or drop the shard lock before taking allocMu. GET never takes allocMu.
type Store struct {
	arena *Arena
	index *Index
	life  *lifecycle
	now   func() int64 // same clock instance the lifecycle uses

	allocMu sync.Mutex // serializes the non-goroutine-safe Allocator
	alloc   *Allocator

	// Eviction (nil-guarded — a store without a policy behaves exactly as
	// before). policy is set once before traffic (AttachPolicy) and read
	// without synchronization after; its methods are leaf-locked and are
	// NEVER called while a shard lock or allocMu is held (the lock story).
	policy     eviction.Policy
	evict      *evictor      // nil until StartEvictor (modeltest never starts it)
	evictMu    sync.Mutex    // singleflight: one eviction pass at a time
	evictions  atomic.Uint64 // total blocks evicted (Stats → kvb_evictions_total)
	liveAllocs atomic.Int64  // live extents; the node-pool watermark numerator

	// Pass scratch, guarded by evictMu (an eviction pass under sustained
	// pressure runs at BEGIN cadence — per-pass allocations would be GC
	// pressure exactly when the tier is busiest).
	evVictims []Key
	evUsages  []eviction.DomainUsage
	evCands   []eviction.Candidate
}

// New builds the tier over an arena the caller has already mapped. The
// allocator is sized in AllocUnit granules over the WHOLE arena, with a node
// pool budgeted for the smallest band block (so capacity, not the pool, is
// always the binding constraint).
func New(arena *Arena, p Params) *Store {
	units := uint32(arena.Size() >> AllocUnitShift) //nolint:gosec // G115: arena sizes are validated well below 16 TiB
	// Smallest realistic block ≈ 0.4 MB = 100 units; budget maxAllocs for
	// 64 KiB blocks (16 units) so tiny-block tests never starve the pool.
	// HONESTY CEILING: NewAllocatorMax clamps the node pool to 2^17 live
	// allocations (Allocation.Meta's 18 slot bits), so an arena above ~8 GiB
	// of 64 KiB blocks (or ~50 GiB of 0.4 MB blocks) hits the pool before
	// capacity and Put answers ERR_QUOTA_BYTES with free bytes remaining.
	// Fine for the Week-3 1 GiB default; widening Meta is scheduled with the
	// evictor work (recorded in DESIGN.md).
	maxAllocs := units / 16
	if maxAllocs < 1024 {
		maxAllocs = 1024
	}
	now := p.Now
	if now == nil {
		now = func() int64 { return time.Now().UnixNano() }
	}
	return &Store{
		arena: arena,
		index: NewIndex(),
		life:  newLifecycle(p.LeaseDefaultMS, p.LeaseMaxMS, p.PinnedBytesCap, now),
		now:   now,
		alloc: NewAllocatorMax(units, maxAllocs),
	}
}

// bytesToUnits rounds a byte length up to allocation units.
func bytesToUnits(n int) uint32 {
	return uint32((n + AllocUnit - 1) >> AllocUnitShift) //nolint:gosec // G115: n ≤ max_blob_len ≪ 16 TiB
}

// AttachPolicy installs the eviction policy. Call before serving traffic;
// the hooks read it without synchronization.
func (s *Store) AttachPolicy(p eviction.Policy) { s.policy = p }

// free returns a block's extent to the allocator. The ONLY caller is the
// Release()-hit-zero path, so an extent is never recycled under a live ref.
// Zero-length blocks own no extent (Put never allocated one) — only their
// liveAllocs charge returns (they carry a NOMINAL accounting slot so an
// empty-block flood still trips the count trigger; the measured exposure
// was ~181 B of index heap per block with no bound at all).
func (s *Store) free(ref *BlockRef) {
	s.liveAllocs.Add(-1)
	if ref.Len == 0 {
		return
	}
	s.allocMu.Lock()
	s.alloc.Free(Allocation{Offset: ref.Offset, Meta: ref.allocMeta})
	s.allocMu.Unlock()
}

// CanStore is the server's advisory BEGIN-time capacity probe (§3.4 answers
// ERR_QUOTA_BYTES at BEGIN; §5 "quota check" annotation). With a running
// evictor the question is "can this tier MAKE room", so a failing probe
// runs one synchronous eviction pass and re-checks — otherwise a full-but-
// reclaimable arena would reject every BEGIN before the commit path's
// retry could ever help. Advisory: a yes can still lose the race to a
// competing commit — that rare loser maps to ERR_BUSY at COMMIT (retry;
// the fresh BEGIN then reports honestly).
func (s *Store) CanStore(n uint32) bool {
	if n == 0 {
		return true // empty blocks own no extent
	}
	units := bytesToUnits(int(n))
	fits := func() bool {
		s.allocMu.Lock()
		rep := s.alloc.StorageReport()
		s.allocMu.Unlock()
		return rep.LargestFreeRegion >= units
	}
	if fits() {
		return true
	}
	if s.evict != nil {
		s.EvictNow()
		if fits() {
			return true
		}
	}
	s.kickEvictor() // protections lapse; the background pass keeps trying
	return false
}

// ---------------------------------------------------------------------------
// The 6-method server.Store surface.

// ExistsPrefix serves the §3.2 probe from the index only — never blocks on
// anything but a shard RLock.
func (s *Store) ExistsPrefix(ns uint32, keys [][32]byte, withBitmap bool) (uint32, []protocol.Status) {
	n, hits := s.index.ExistsPrefix(ns, keys, withBitmap)
	if hits == nil {
		return n, nil
	}
	perKey := make([]protocol.Status, len(hits))
	for i, ok := range hits {
		if ok {
			perKey[i] = protocol.StatusOK
		} else {
			perKey[i] = protocol.StatusNotFound
		}
	}
	return n, perKey
}

// Get is the plain interface method (ramstub-parity: no release hook). It
// COPIES the block to the heap before releasing its reference: returning the
// raw arena view after release would hand out bytes a concurrent remote
// DELETE (not just the caller's own logic) could free and let the next Put
// recycle — a silent torn read -race cannot see. The wire path never pays
// this copy (the server uses GetRef); plain Get serves the conformance
// suite and non-wire tooling, where safety beats zero-copy.
func (s *Store) Get(ns uint32, key [32]byte) ([]byte, uint64, bool) {
	data, sum, release, ok := s.GetRef(ns, key)
	if !ok {
		return nil, 0, false
	}
	out := make([]byte, len(data))
	copy(out, data)
	release()
	return out, sum, true
}

// GetRef is the zero-copy read the server's BATCH_GET uses: it Acquires a
// reader reference and returns the arena view plus a release func the caller
// MUST fire exactly once after the bytes have left (the transport fires it
// from WriteFrames' post-writev release hook). The reference keeps the extent
// from being freed/recycled by a concurrent DELETE.
func (s *Store) GetRef(ns uint32, key [32]byte) (data []byte, xxh3 uint64, release func(), ok bool) {
	ref, found := s.index.Get(Key{NS: ns, Hash: key})
	if !found || !ref.Acquire() {
		// Absent, or lost the race with a Delete's final teardown: a miss.
		return nil, 0, nil, false
	}
	now := s.life.GrantReadLease(ref) // §3.3 auto-lease on every OK GET
	ref.hits.Add(1)                   // demotion admission input (relaxed count)
	if p := s.policy; p != nil {
		p.Touch(eviction.Key{NS: ns, Hash: key}, now) // no lock held here; Touch is a leaf + zero-alloc
	}
	if ref.Len > 0 { // a zero-length block owns no extent (§3.4: legal; GET desc is OK/len=0)
		data = s.arena.Bytes(uint64(ref.Offset)<<AllocUnitShift, ref.Len)
	}
	rel := func() {
		if ref.Release() {
			s.free(ref)
		}
	}
	return data, ref.XXH3, rel, true
}

// Put commits a fully-staged block (write-once, §13). The commit path:
// idempotent-hit check → allocate under allocMu → ONE copy buf→arena →
// publish via index insert (the atomic visibility flip). A lost insert race
// frees the extent and defers to the winner. Ownership of data transfers in
// (the caller never touches it again); its bytes are copied to the arena and
// the heap buffer dies.
func (s *Store) Put(ns uint32, key [32]byte, data []byte, xxh3 uint64) protocol.Status {
	k := Key{NS: ns, Hash: key}
	if existing, ok := s.index.Get(k); ok {
		return putExistsStatus(existing, xxh3)
	}

	// A zero-length block (§3.4: total_len=0 is legal) owns no extent: no
	// Alloc, and free()/GetRef gate on Len==0 symmetrically.
	var al Allocation
	if len(data) > 0 {
		units := bytesToUnits(len(data))
		var allocOK bool
		s.allocMu.Lock()
		al, allocOK = s.alloc.Alloc(units)
		s.allocMu.Unlock()
		if !allocOK && s.evict != nil {
			// A running evictor turns the wall into reclamation: one
			// synchronous pass, one retry. Gated on the EVICTOR (not the
			// policy) so a policy-attached-but-evictor-off store — the
			// deterministic modeltest shape — never self-evicts.
			s.EvictNow()
			s.allocMu.Lock()
			al, allocOK = s.alloc.Alloc(units)
			s.allocMu.Unlock()
		}
		if !allocOK {
			// Arena full (or node pool exhausted) and eviction couldn't help
			// (everything protected, or no evictor): graceful backpressure.
			// Nudge the background evictor anyway — protections lapse.
			s.kickEvictor()
			return protocol.StatusErrQuotaBytes
		}
		s.liveAllocs.Add(1)
		copy(s.arena.Bytes(uint64(al.Offset)<<AllocUnitShift, uint32(len(data))), data) //nolint:gosec // G115: len ≤ max_blob_len
	}

	ref := &BlockRef{
		Offset:      al.Offset,         // UNITS (see BlockRef doc)
		Len:         uint32(len(data)), //nolint:gosec // G115: len ≤ max_blob_len
		NamespaceID: ns,
		XXH3:        xxh3,
		allocMeta:   al.Meta,
		Tier:        TierDRAM,
	}
	ref.Refcount.Store(1) // the index's own reference
	if existing, inserted := s.index.Put(k, ref); !inserted {
		// Lost the publish race: free our extent (none for a zero-length
		// block), defer to the winner (who also owns the policy Admit).
		if len(data) > 0 {
			s.allocMu.Lock()
			s.alloc.Free(al)
			s.allocMu.Unlock()
			s.liveAllocs.Add(-1)
		}
		return putExistsStatus(existing, xxh3)
	}
	if len(data) == 0 {
		// The nominal accounting slot: an extent-less block still counts
		// toward the node-pool trigger, so an empty-block flood cannot grow
		// the index without eviction backpressure (the confirmed DoS).
		s.liveAllocs.Add(1)
	}
	if p := s.policy; p != nil && len(data) > 0 {
		// After the winning publish, no locks held. Zero-length blocks are
		// never admitted (no extent to reclaim — the count sweep reclaims
		// them). A Delete interleaving before this Admit leaves a policy
		// orphan — self-healing: it surfaces as a candidate, fails the gate
		// as gone, and drops.
		p.Admit(eviction.Key{NS: ns, Hash: key}, int64(len(data)), s.now())
	}
	return protocol.StatusOK
}

// putExistsStatus maps a write-once collision: same digest → idempotent hit,
// different digest → corruption alarm (content-derived keys can't disagree).
func putExistsStatus(existing *BlockRef, xxh3 uint64) protocol.Status {
	if existing.XXH3 == xxh3 {
		return protocol.StatusOKExists
	}
	return protocol.StatusErrImmutableConflict
}

// Contains reports whether a committed block exists (PUT BEGIN short-circuit).
func (s *Store) Contains(ns uint32, key [32]byte) bool {
	_, ok := s.index.Get(Key{NS: ns, Hash: key})
	return ok
}

// Delete removes a block (§3.7): the lifecycle gate first (leased/pinned,
// force semantics), then the index removal + the index-reference drop. The
// extent is freed by whichever Release hits zero — immediately when no reader
// holds it, else by the last in-flight GET's release.
func (s *Store) Delete(ns uint32, key [32]byte, force bool) protocol.Status {
	k := Key{NS: ns, Hash: key}
	ref, st := s.index.DeleteIf(k, func(ref *BlockRef) protocol.Status {
		if g := s.life.canDelete(ref, force); g != protocol.StatusOK {
			return g
		}
		if p := s.policy; p != nil {
			// INSIDE the gate: atomic with the index removal, so a
			// re-publishing Put's Admit can never be erased by this Remove
			// (policy locks are leaves — shard→policy nests acyclically).
			p.Remove(eviction.Key{NS: ns, Hash: key})
		}
		s.life.unpinOnDelete(ref) // refund pinned bytes under the shard lock
		return protocol.StatusOK
	})
	if ref == nil {
		return st
	}
	s.life.noteTTLGone(ref)
	if ref.Release() { // drop the index ref OUTSIDE the shard lock; zero → free
		s.free(ref)
	}
	return protocol.StatusOK
}

// ---------------------------------------------------------------------------
// Lifecycle extension (TOUCH_LEASE / PIN wire verbs).

// TouchLease serves §3.5 sub-ops: 0=TOUCH (recency+TTL), 1=LEASE, 2=RELEASE.
func (s *Store) TouchLease(ns uint32, key [32]byte, sub uint8, ttlMS uint32) protocol.Status {
	ref, ok := s.index.Get(Key{NS: ns, Hash: key})
	if !ok {
		return protocol.StatusNotFound
	}
	switch sub {
	case protocol.TouchRecency:
		s.life.Touch(ref, ttlMS)
		if p := s.policy; p != nil {
			p.Touch(eviction.Key{NS: ns, Hash: key}, s.now())
		}
	case protocol.LeaseGrant:
		s.life.Lease(ref, ttlMS)
	case protocol.LeaseRelease:
		s.life.ReleaseLease(ref)
	default:
		return protocol.StatusErrUnsupported
	}
	return protocol.StatusOK
}

// PinOp serves §3.6 sub-ops: 0=PIN_SOFT, 1=PIN_HARD, 2=UNPIN. PinFlags mutate
// under the shard lock (the BlockRef convention).
func (s *Store) PinOp(ns uint32, key [32]byte, sub uint8) protocol.Status {
	k := Key{NS: ns, Hash: key}
	st := protocol.StatusNotFound
	s.index.WithShardLock(k, func(ref *BlockRef) {
		if ref == nil {
			return
		}
		switch sub {
		case protocol.PinSoft:
			st = s.life.Pin(ref, false)
		case protocol.PinHard:
			st = s.life.Pin(ref, true)
		case protocol.Unpin:
			s.life.Unpin(ref)
			st = protocol.StatusOK
		default:
			st = protocol.StatusErrUnsupported
		}
	})
	return st
}

// ---------------------------------------------------------------------------

// Stats returns the tier's JSON document — a superset of ramstub's shape so
// existing wire assertions (`"blocks":N`) keep working.
func (s *Store) Stats() []byte {
	var blocks int
	var bytes uint64
	s.index.Range(func(_ Key, ref *BlockRef) bool {
		blocks++
		bytes += uint64(ref.Len)
		return true
	})
	s.allocMu.Lock()
	rep := s.alloc.StorageReport()
	s.allocMu.Unlock()

	doc := map[string]any{
		"schema":                    1,
		"store":                     "dram",
		"blocks":                    blocks,
		"bytes":                     bytes,
		"shard_count":               indexShards,
		"arena_bytes":               s.arena.Size(),
		"arena_free_bytes":          uint64(rep.TotalFreeSpace) << AllocUnitShift,
		"largest_free_region_bytes": uint64(rep.LargestFreeRegion) << AllocUnitShift,
		"hugepages":                 s.arena.Huge(),
		"pinned_bytes":              s.life.pinnedBytes(),
		"evictions_total":           s.evictions.Load(),
		"live_allocs":               s.liveAllocs.Load(),
		"max_allocs":                s.alloc.MaxAllocs(),
	}
	b, err := json.Marshal(doc)
	if err != nil {
		return []byte(`{"schema":1,"store":"dram","error":"stats encode failed"}`)
	}
	return b
}

// Close asserts (kvbdebug) that no reader references remain, then closes the
// arena. Call ONLY after the server has drained (srv.Drain waits every conn's
// writer, which fires every GET release) — the arena drain-before-Close rule.
func (s *Store) Close() error {
	if debugAssertsEnabled() {
		s.index.Range(func(k Key, ref *BlockRef) bool {
			assertf(ref.Refcount.Load() <= 1, // the index ref itself is fine
				"dram: Close with live reader refs (ns=%d refcount=%d)", ref.NamespaceID, ref.Refcount.Load())
			return true
		})
	}
	return s.arena.Close()
}
