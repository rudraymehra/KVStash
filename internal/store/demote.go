package store

import (
	"sync"

	"github.com/kvstash/kvblockd/internal/eviction"
	"github.com/kvstash/kvblockd/internal/protocol"
	"github.com/kvstash/kvblockd/internal/store/dram"
	"github.com/kvstash/kvblockd/internal/store/nvme"
	"github.com/kvstash/kvblockd/internal/tenant"
)

// Demotion / promotion / reclaim — the tier movement machinery.
//
// Demotion (D-steps from the design review):
//
//	D1  at ≥90% DRAM occupancy, ask the policy for victims (proportional per
//	    namespace — the same shape as the evictor's policyPass, run BELOW
//	    the evictor's 95% so demotion normally beats eviction to the punch);
//	D2  RefForDemotion holds a reader ref across the whole write — the arena
//	    extent cannot be recycled under the writer's copy;
//	    · dual-resident with the same xxh3 → skip the append (write-amp
//	      guard), just complete the move;
//	    · fewer lifetime GETs than admit_min_hits → plain eviction — a
//	      never-read block is not worth SSD endurance;
//	D3  bounded Volume.Append (fire-and-forget; full queue → drop + policy
//	    re-admit — the PegaFlow posture);
//	D4  OnWritten → release the ref → CompleteDemotion publishes the NVMe
//	    index entry INSIDE the dram shard-lock gate (zero-width swap). A
//	    refused gate (the block got leased/read mid-queue) means it proved
//	    itself hot: it stays in DRAM, the segment bytes become garbage that
//	    reclaim sweeps with the segment.

// demotePass runs one watermark-driven pass; wait blocks until every
// accepted append completes (the modeltest handle). Returns victims
// processed.
func (t *Tiered) demotePass(wait bool) int {
	if len(t.vols) == 0 || t.pol == nil {
		return 0
	}
	used, total := t.d.Occupancy()
	if total == 0 || used*100 < total*int64(t.p.DemoteWatermarkPct) {
		return 0
	}
	floor := total * int64(t.p.DemoteWatermarkPct-t.p.DemoteBatchPct) / 100
	need := used - floor
	if need <= 0 {
		return 0
	}
	now := t.now()
	usages := t.pol.Usage(t.scUsages[:0])
	t.scUsages = usages[:0]
	var totalBytes int64
	for _, u := range usages {
		totalBytes += u.Bytes
	}
	if totalBytes == 0 {
		return 0
	}

	var wg *sync.WaitGroup
	if wait {
		wg = &sync.WaitGroup{}
	}
	processed := 0
	var moved int64
	harvest := func(ns uint32, want int64) {
		cands := t.pol.Victims(ns, want, now, t.scCands[:0])
		t.scCands = cands[:0]
		for _, c := range cands {
			moved += t.demoteOne(dram.Key{NS: c.Key.NS, Hash: c.Key.Hash}, c.Size, now, wg)
			processed++
		}
	}
	// Round 1: proportional to resident bytes (float — int64 products
	// overflow at multi-GiB arenas, same rationale as the evictor).
	for _, u := range usages {
		share := int64(float64(need) * float64(u.Bytes) / float64(totalBytes))
		if share <= 0 {
			continue
		}
		harvest(u.NS, share)
	}
	// Round 2: cover the remainder from whoever has demotable bytes.
	for _, u := range usages {
		if moved >= need {
			break
		}
		harvest(u.NS, need-moved)
	}
	if wg != nil {
		wg.Wait()
	}
	return processed
}

// demoteOne moves one policy-nominated victim. Returns the bytes it expects
// to free from DRAM (0 when the victim was dropped/refused). The Victims
// call already dequeued the key: a refused/failed victim is handed back via
// Admit (the re-admission contract); a vanished one is dropped — re-
// admitting it would mint a phantom.
func (t *Tiered) demoteOne(k dram.Key, size int64, now int64, wg *sync.WaitGroup) int64 {
	data, sum, hits, rel, ok := t.d.RefForDemotion(k.NS, k.Hash)
	if !ok {
		return 0 // gone (or zero-length) — drop
	}

	// Dual residency: the bytes are ALREADY on NVMe (demoted before,
	// promoted back, cold again). Never rewrite them — complete the move.
	// A reclaim retiring that copy between this check and the completion
	// loses both copies: cache-legal loss (the block simply misses), and
	// the swap can never serve WRONG bytes — the nvme entry's removal makes
	// the key an honest miss.
	if e := t.idx.get(k); e != nil && e.XXH3 == sum {
		rel()
		if t.d.CompleteDemotion(k.NS, k.Hash, sum, func(*dram.BlockRef) {}) {
			t.dedupSkips.Add(1)
			// Dual-residency collapse: the DRAM copy is gone, the NVMe
			// charge already exists from the first demotion — refund DRAM.
			if t.p.Quotas != nil {
				t.p.Quotas.Refund(k.NS, tenant.TierDRAM, size)
			}
			return size
		}
		t.pol.Admit(eviction.Key{NS: k.NS, Hash: k.Hash}, size, now)
		return 0
	}

	// SSD-endurance admission: a block nobody ever read gets evicted, not
	// written to flash.
	if hits < t.p.AdmitMinHits {
		rel()
		if t.d.CompleteDemotion(k.NS, k.Hash, sum, nil) {
			// The block is GONE (deleted, not moved) — SSD endurance says a
			// never-read block isn't flash-worthy. Count it: an operator
			// watching a pure-PUT fill melt away deserves a metric, and a
			// silent variant of this branch once ate a 20 GiB rig fill.
			t.admitRefusals.Add(1)
			if t.p.Quotas != nil {
				t.p.Quotas.Refund(k.NS, tenant.TierDRAM, size)
			}
			return size
		}
		t.pol.Admit(eviction.Key{NS: k.NS, Hash: k.Hash}, size, now)
		return 0
	}

	if wg != nil {
		wg.Add(1)
	}
	accepted := t.volumeFor(k.Hash).Append(nvme.AppendReq{
		NS: k.NS, Key: k.Hash, XXH3: sum, Data: data,
		OnWritten: func(loc nvme.Loc, wok bool) {
			if wg != nil {
				defer wg.Done()
			}
			rel() // the arena view is copied (or abandoned) — drop the hold
			if !wok {
				t.demoteDrops.Add(1)
				t.pol.Admit(eviction.Key{NS: k.NS, Hash: k.Hash}, size, now)
				return
			}
			published := t.d.CompleteDemotion(k.NS, k.Hash, sum, func(ref *dram.BlockRef) {
				nr := &nvmeRef{Loc: loc, Len: loc.Len, XXH3: sum}
				nr.TTLUntil.Store(ref.TTLUntil.Load())
				t.idx.put(k, nr) // nvme shard lock UNDER the dram shard lock — the documented order
			})
			if !published {
				// The block proved itself hot mid-queue (leased/held) or was
				// replaced: it stays in DRAM; the segment bytes are garbage
				// reclaim will sweep. Hand the key back to the policy.
				t.demoteDrops.Add(1)
				t.pol.Admit(eviction.Key{NS: k.NS, Hash: k.Hash}, size, now)
				return
			}
			t.demotions.Add(1)
			// The tier MOVE: NVMe gains the bytes, DRAM refunds. Transfer
			// never fails — refusing a demotion for destination quota would
			// wedge the memory ladder; the evictor's over-quota-first pass
			// corrects the destination on its next cycle.
			if t.p.Quotas != nil {
				t.p.Quotas.Transfer(k.NS, tenant.TierDRAM, tenant.TierNVMe, size)
			}
		},
	})
	if !accepted {
		if wg != nil {
			wg.Done()
		}
		rel()
		t.demoteDrops.Add(1)
		t.pol.Admit(eviction.Key{NS: k.NS, Hash: k.Hash}, size, now)
		return 0
	}
	return size
}

// promoteOne is the background 2nd-hit promotion: re-read the block off the
// device and Put it back into DRAM (write-once makes the race benign —
// OK_EXISTS is success). The NVMe entry is KEPT: dual residency means a
// future re-demotion is a metadata move, not an SSD rewrite.
func (t *Tiered) promoteOne(req promoteReq) {
	if cur := t.idx.get(req.k); cur != req.ref {
		return // replaced or removed since the hit — stale request
	}
	_ = t.promoteSync(req.k, req.ref)
}

// promoteSync is the synchronous promotion core (also the PIN_HARD path).
func (t *Tiered) promoteSync(k dram.Key, e *nvmeRef) protocol.Status {
	var heap []byte
	blob, rel, st := t.volumeFor(k.Hash).Read(e.Loc, k.NS, k.Hash, e.XXH3)
	switch st {
	case nvme.ReadOK:
		heap = make([]byte, len(blob))
		copy(heap, blob)
		rel()
	case nvme.ReadBusy:
		return protocol.StatusErrBusy
	case nvme.ReadGone:
		// Retired locally but s3-resident: promotion (and PIN_HARD's "must
		// survive" = DRAM residency) must work from the cold tier too — a
		// GET serves this block, so a pin refusing NOT_FOUND would be a lie.
		// readS3 verifies before a byte escapes and hands us a heap buffer.
		data, crel, ok := t.readS3(e, k.NS, k.Hash)
		if !ok {
			return protocol.StatusNotFound
		}
		heap = data
		crel()
	case nvme.ReadCorrupt:
		return protocol.StatusNotFound
	}
	switch pst := t.d.Put(k.NS, k.Hash, heap, e.XXH3); pst {
	case protocol.StatusOK:
		t.promotions.Add(1)
		return protocol.StatusOK
	case protocol.StatusOKExists:
		return protocol.StatusOK
	default:
		// DRAM couldn't take it (quota, pool) — the block stays NVMe-only.
		return protocol.StatusErrBusy
	}
}

// reclaimPass frees whole segments FIFO while a volume sits above 90% of
// its budget (bounded per pass — reclaim shares the demoter goroutine).
// It also re-arms volumes left write-dead by a failed rotation.
func (t *Tiered) reclaimPass() {
	for _, vol := range t.vols {
		for i := 0; i < 4; i++ {
			if vol.MaxBytes() <= 0 || vol.UsedBytes()*100 <= vol.MaxBytes()*90 {
				break
			}
			if !t.reclaimSegment(vol) {
				break
			}
		}
		vol.TryRecoverWrites()
	}
}

// reclaimSegment retires the oldest sealed segment through the R-protocol:
// pre-gate (any live-protected entry → skip the whole segment — the
// documented choice), dying flag, shard-locked gate+remove per entry
// (abort restores service if protection landed mid-retire), unlink-with-
// open-fd, drain, close.
func (t *Tiered) reclaimSegment(vol *nvme.Volume) bool {
	id, entries, ok := vol.OldestSealed()
	if !ok {
		return false
	}
	now := t.now()
	for i := range entries {
		k := dram.Key{NS: entries[i].NS, Hash: entries[i].Key}
		// PinFlags is guarded by the shard lock (its own documented rule —
		// the ladder caught this pre-gate reading it bare, a data race with
		// PinOp's locked writes). The pre-gate stays advisory; the per-entry
		// deleteIf gate below re-checks authoritatively under the same lock.
		protected := false
		t.idx.withShardLock(k, func(e *nvmeRef) {
			if e == nil || e.Loc.SegmentID != id {
				return // moved or gone — these bytes are already garbage
			}
			protected = e.leased(now) || e.pinned()
		})
		if protected {
			t.reclaimSkips.Add(1)
			return false // segment skipped this round (retry when the lease lapses)
		}
	}
	if !vol.RetireBegin(id) {
		return false
	}
	if t.p.Spill != nil && vol.IsSpilled(id) {
		// The FLIP: this segment's bytes live on S3 byte-identically, so
		// retiring the local file loses nothing — entries stay in the index
		// (Loc now addresses the object), the tenant charge moves
		// NVMe→S3, and cold reads route through the restorer. Protection
		// still holds: leased/pinned entries abort the retire exactly as
		// the delete path does (a hard-pinned block never becomes s3-only).
		for i := range entries {
			k := dram.Key{NS: entries[i].NS, Hash: entries[i].Key}
			protected := false
			t.idx.withShardLock(k, func(e *nvmeRef) {
				if e == nil || e.Loc.SegmentID != id {
					return
				}
				protected = e.leased(t.now()) || e.pinned()
			})
			if protected {
				vol.RetireAbort(id)
				t.reclaimSkips.Add(1)
				return false
			}
		}
		t.s3FlipRetired(vol, id)
	} else {
		for i := range entries {
			k := dram.Key{NS: entries[i].NS, Hash: entries[i].Key}
			removed, st := t.idx.deleteIf(k, func(ref *nvmeRef) protocol.Status {
				if ref.Loc.SegmentID != id {
					return protocol.StatusErrBusy // newer home elsewhere — keep the entry
				}
				if ref.leased(t.now()) || ref.pinned() {
					return protocol.StatusErrLeased // protection landed mid-retire
				}
				return protocol.StatusOK
			})
			t.refundNvmeQuota(k.NS, removed)
			if st == protocol.StatusErrLeased {
				vol.RetireAbort(id) // reads resume; already-removed entries were unprotected (legal evictions)
				t.reclaimSkips.Add(1)
				return false
			}
		}
	}
	if err := vol.RetireFinish(id); err != nil {
		return false
	}
	t.reclaims.Add(1)
	return true
}
