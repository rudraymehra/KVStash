package dram

import (
	"context"
	"time"

	"github.com/kvstash/kvblockd/internal/eviction"
	"github.com/kvstash/kvblockd/internal/protocol"
)

// The eviction execution engine. The POLICY (internal/eviction) decides who
// should go; THIS file is the sole authority on whether a block MAY go —
// every candidate is re-gated under its index shard write lock through the
// same DeleteIf discipline as a client DELETE, so a stale candidate
// (deleted, newly leased, newly pinned, read-held) resolves to a no-op or a
// re-admission, never a torn read. Eviction is metadata-only: index remove
// + refcount drop + allocator free — payload bytes are never touched.
//
// Ladder precedence enforced here (PROTOCOL.md §6):
//
//	unpinned+expired → unpinned → soft-pinned (quota emergency ONLY)
//	→ leased (never while valid) → hard-pinned (never) → read-held (never)

// EvictorConfig tunes the watermark trigger.
type EvictorConfig struct {
	WatermarkPct int           // trigger at used ≥ this % of arena (default 95)
	BatchPct     int           // free down to (Watermark−Batch)% (default 5)
	Interval     time.Duration // occupancy poll cadence (default 100ms)
}

func (c EvictorConfig) withDefaults() EvictorConfig {
	if c.WatermarkPct == 0 {
		c.WatermarkPct = 95
	}
	if c.BatchPct == 0 {
		c.BatchPct = 5
	}
	if c.Interval == 0 {
		c.Interval = 100 * time.Millisecond
	}
	return c
}

type evictor struct {
	cfg  EvictorConfig
	kick chan struct{} // Put's alloc-failure nudge (1-buffered)
	done chan struct{}
}

// StartEvictor runs the watermark goroutine. The returned stop func cancels
// and WAITS for the goroutine — call it after a successful Drain and before
// Store.Close, so no eviction free can race the arena unmap.
func (s *Store) StartEvictor(ctx context.Context, cfg EvictorConfig) (stop func()) {
	ev := &evictor{
		cfg:  cfg.withDefaults(),
		kick: make(chan struct{}, 1),
		done: make(chan struct{}),
	}
	s.evict = ev
	ctx, cancel := context.WithCancel(ctx)
	go func() {
		defer close(ev.done)
		t := time.NewTicker(ev.cfg.Interval)
		defer t.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-t.C:
			case <-ev.kick:
			}
			s.evictOnce(ev.cfg)
		}
	}()
	return func() {
		cancel()
		<-ev.done
	}
}

// EvictNow runs one synchronous eviction pass with the evictor's config (or
// defaults when no evictor is started — the modeltest handle) and reports
// the bytes freed.
func (s *Store) EvictNow() int64 {
	cfg := EvictorConfig{}.withDefaults()
	if s.evict != nil {
		cfg = s.evict.cfg
	}
	return s.evictOnce(cfg)
}

// occupancy reads the trigger inputs in one allocMu hold.
func (s *Store) occupancy() (usedUnits, totalUnits uint32) {
	s.allocMu.Lock()
	rep := s.alloc.StorageReport()
	s.allocMu.Unlock()
	total := uint32(s.arena.Size() >> AllocUnitShift) //nolint:gosec // G115: validated arena sizes
	return total - rep.TotalFreeSpace, total
}

// evictOnce is one full pass under the singleflight lock: expired-first
// sweep, proportional policy pass, then the quota-emergency sweep if the
// tier is still over the watermark. Returns bytes freed.
func (s *Store) evictOnce(cfg EvictorConfig) int64 {
	s.evictMu.Lock()
	defer s.evictMu.Unlock()

	used, total := s.occupancy()
	byteTrigger := uint64(used)*100 >= uint64(total)*uint64(cfg.WatermarkPct) //nolint:gosec // G115: pct validated in [50,99]
	// The node-pool trigger: small-block workloads exhaust the allocator's
	// live-allocation budget with free bytes remaining — a bytes-only
	// watermark would never fire (the 2^17 Meta clamp honesty ceiling).
	maxAllocs := int64(s.alloc.MaxAllocs())
	allocTrigger := s.liveAllocs.Load()*100 >= maxAllocs*95
	if !byteTrigger && !allocTrigger {
		return 0
	}

	// Free down to (watermark − batch)% of capacity.
	floorUnits := uint32(uint64(total) * uint64(cfg.WatermarkPct-cfg.BatchPct) / 100) //nolint:gosec // G115: pct math within u32
	var needUnits uint32
	if used > floorUnits {
		needUnits = used - floorUnits
	}
	// Count-based need: drop the live-slot count to 90% of the pool. This
	// is the whole defense for EXTENT-LESS blocks (zero-length: legal,
	// index-resident, invisible to the byte watermark and to the policy —
	// the confirmed empty-block flood DoS): they carry a nominal slot in
	// liveAllocs, so the count goal shrinks their population via the
	// sweeps even when needUnits is zero.
	var needCount int64
	if allocTrigger {
		if excess := s.liveAllocs.Load() - maxAllocs*90/100; excess > 0 {
			needCount = excess
		}
	}
	if needUnits == 0 && needCount == 0 {
		return 0
	}
	needBytes := int64(needUnits) << AllocUnitShift
	now := s.now()
	var freed int64

	// Pass 1 — expired first (the lazy-TTL enforcement point): the ladder's
	// weakest rung yields before anything the policy values. Skipped
	// entirely when no resident block carries a TTL (the counter is a hint;
	// a spurious sweep is only wasted work, never a correctness issue).
	if s.life.ttlBlocks.Load() > 0 {
		freed += s.sweepExpired(now, needBytes)
	}

	// Pass 2 — the policy pass, split across tenants proportionally to
	// their resident bytes (strict pressure isolation arrives with the
	// Week-6 per-namespace quotas; proportionality bounds each tenant's
	// loss to its own footprint share until then).
	if freed < needBytes {
		freed += s.policyPass(now, needBytes-freed)
	}

	// Pass 3 — quota emergency: still over EITHER trigger after everything
	// unprotected went ⇒ soft pins yield (§6: "quota emergency only").
	// Leases, hard pins, and read-held blocks never do. The node-pool
	// trigger arms this too: pool exhaustion surfaces to clients as
	// ERR_QUOTA_BYTES, which IS the quota emergency §6 names. Need is
	// recomputed from fresh occupancy: passes 1–2 may have freed fewer
	// bytes than they claimed.
	used2, total2 := s.occupancy()
	stillBytes := uint64(used2)*100 >= uint64(total2)*uint64(cfg.WatermarkPct) //nolint:gosec // G115: as above
	stillAllocs := s.liveAllocs.Load()*100 >= maxAllocs*95
	if stillBytes || stillAllocs {
		var emBytes int64
		if used2 > floorUnits {
			emBytes = int64(used2-floorUnits) << AllocUnitShift
		}
		var emCount int64
		if stillAllocs {
			if excess := s.liveAllocs.Load() - maxAllocs*90/100; excess > 0 {
				emCount = excess
			}
		}
		freed += s.sweepEmergency(now, emBytes, emCount)
	}
	return freed
}

// evictOne outcomes: the caller's re-admission decision hinges on WHY
// nothing was freed — a gate refusal re-admits, a vanished key drops.
const (
	evictedOK = iota
	evictRefused
	evictGone
)

// evictOne removes one block through the DeleteIf discipline. emergency
// admits soft-pinned victims (§6 quota-emergency rung). policy.Remove fires
// INSIDE the gate, under the shard write lock — atomic with the index
// removal, so a concurrent re-Put's Admit of the same key can never be
// erased by our removal (policy locks are leaves; shard→policy nesting is
// acyclic by construction).
func (s *Store) evictOne(k Key, emergency bool) (freed int64, outcome int) {
	outcome = evictGone
	ref, _ := s.index.DeleteIf(k, func(ref *BlockRef) protocol.Status {
		now := s.now()
		ok := ref.CanEvict(now) // the normal-pressure pre-filter: refcount==1, unleased, unpinned
		if !ok && emergency {
			// Emergency widens exactly one rung: soft pins yield; leases,
			// hard pins, and read-held blocks still never do.
			ok = ref.Refcount.Load() == 1 && !ref.Leased(now) && !ref.HardPinned()
		}
		if !ok {
			outcome = evictRefused
			return protocol.StatusErrBusy // aborts the removal
		}
		if p := s.policy; p != nil {
			p.Remove(eviction.Key{NS: k.NS, Hash: k.Hash})
		}
		s.life.unpinOnDelete(ref) // refunds an emergency-yielded soft pin's flags
		return protocol.StatusOK
	})
	if ref == nil {
		return 0, outcome
	}
	s.life.noteTTLGone(ref)
	s.evictions.Add(1)
	freed = int64(ref.Len)
	if ref.Release() { // index ref drop OUTSIDE the shard lock — the one free story
		s.free(ref)
	}
	return freed, evictedOK
}

// sweepExpired walks the index collecting expired, plausibly-unprotected
// blocks (racy atomic pre-check — the gate re-verifies) and evicts up to
// need bytes. The policy is told about every success (it did not nominate
// these).
func (s *Store) sweepExpired(now int64, need int64) int64 {
	if need <= 0 {
		return 0
	}
	victims := s.evVictims[:0]
	var claimed int64
	s.index.Range(func(k Key, ref *BlockRef) bool {
		if ref.Expired(now) && ref.Refcount.Load() == 1 && !ref.Leased(now) {
			victims = append(victims, k)
			claimed += int64(ref.Len)
		}
		return claimed < need
	})
	s.evVictims = victims[:0]
	var freed int64
	for _, k := range victims {
		// policy.Remove happens inside evictOne's gate; no follow-up here.
		if n, _ := s.evictOne(k, false); n > 0 {
			freed += n
		}
		if freed >= need {
			break
		}
	}
	return freed
}

// policyPass asks the policy for victims, tenant by tenant, proportional to
// resident bytes; if protections leave the need uncovered, a second round
// asks every tenant for the remainder (largest first) so reclaimable bytes
// are never stranded behind a protected dominant tenant's rounding. A
// gate-REFUSED candidate is handed back via Admit (the re-admission
// contract); a GONE candidate (concurrently deleted) is dropped — re-
// admitting it would mint a permanent phantom.
func (s *Store) policyPass(now int64, need int64) int64 {
	p := s.policy
	if p == nil || need <= 0 {
		return 0
	}
	usages := p.Usage(s.evUsages[:0])
	s.evUsages = usages[:0]
	var totalBytes int64
	for _, u := range usages {
		totalBytes += u.Bytes
	}
	if totalBytes == 0 {
		return 0
	}
	var freed int64
	harvest := func(ns uint32, want int64) {
		cands := p.Victims(ns, want, now, s.evCands[:0])
		s.evCands = cands[:0]
		for _, c := range cands {
			n, outcome := s.evictOne(Key{NS: c.Key.NS, Hash: c.Key.Hash}, false)
			switch outcome {
			case evictedOK:
				freed += n
			case evictRefused:
				p.Admit(c.Key, c.Size, now) // proved protected: upgrade via the ghost route
			case evictGone:
				// concurrently deleted — the key left the store; drop it
			}
		}
	}
	// Round 1: proportional-to-resident-bytes (float math — the int64
	// product need×bytes overflows at multi-GiB arenas).
	for _, u := range usages {
		share := int64(float64(need) * float64(u.Bytes) / float64(totalBytes))
		if share <= 0 {
			continue
		}
		harvest(u.NS, share)
	}
	// Round 2: cover any remainder from whoever has evictable bytes.
	for _, u := range usages {
		if freed >= need {
			break
		}
		harvest(u.NS, need-freed)
	}
	return freed
}

// sweepEmergency is the §6 quota-emergency rung: a full-index sweep where
// soft pins yield — but STRICTLY after every unpinned candidate (§6 lists
// unpinned before soft-pinned; index Range order must not reorder the
// ladder). Goals are byte-based AND count-based: the count goal is what
// reclaims extent-less (zero-length) blocks, which no byte target can
// reach. PinFlags reads here are safe: Range holds the shard RLock, which
// excludes the write-locked mutators.
func (s *Store) sweepEmergency(now int64, needBytes, needCount int64) int64 {
	if needBytes <= 0 && needCount <= 0 {
		return 0
	}
	unpinned := s.evVictims[:0]
	var softPinned []Key
	var claimedBytes, claimedCount int64
	s.index.Range(func(k Key, ref *BlockRef) bool {
		if ref.Refcount.Load() == 1 && !ref.Leased(now) && !ref.HardPinned() {
			if ref.Pinned() {
				softPinned = append(softPinned, k)
			} else {
				unpinned = append(unpinned, k)
			}
			claimedBytes += int64(ref.Len)
			claimedCount++
		}
		return claimedBytes < needBytes || claimedCount < needCount
	})
	s.evVictims = unpinned[:0]
	var freed, count int64
	done := func() bool { return freed >= needBytes && count >= needCount }
	for _, k := range unpinned {
		if n, outcome := s.evictOne(k, true); outcome == evictedOK {
			freed += n
			count++
		}
		if done() {
			return freed
		}
	}
	for _, k := range softPinned {
		if n, outcome := s.evictOne(k, true); outcome == evictedOK {
			freed += n
			count++
		}
		if done() {
			break
		}
	}
	return freed
}

// kickEvictor nudges a running evictor (Put's alloc-failure path). Non-blocking.
func (s *Store) kickEvictor() {
	if ev := s.evict; ev != nil {
		select {
		case ev.kick <- struct{}{}:
		default:
		}
	}
}
