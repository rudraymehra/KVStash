package store

import (
	"context"
	"io"
	"time"

	"github.com/zeebo/xxh3"

	"github.com/kvstash/kvblockd/internal/store/dram"
	"github.com/kvstash/kvblockd/internal/store/nvme"
	"github.com/kvstash/kvblockd/internal/tenant"
)

// The S3 cold-tier orchestration. Structural interfaces (implemented by
// internal/store/s3spill) keep this package free of the SDK; nil = no cold
// tier, byte-for-byte the two-tier daemon.
//
// The lifecycle: sealed segment → async whole-object spill (a COPY — reads
// stay local) → reclaim of a SPILLED segment FLIPS its entries to
// s3-resident instead of deleting them (quota moves NVMe→S3) → a cold GET
// is one ranged GetObject, verified against the entry's xxh3 before a byte
// escapes → a 2nd cold hit inside the promote window restores the WHOLE
// segment back to the volume and flips its entries home (reads go local
// again; the object is RETAINED — it backs the restored segment's next
// retire-flip, and leaves only through the liveness-driven object GC once
// the segment is gone and its last s3-resident entry left). EXISTS never
// touches S3 (index-only, as always).

// SpillBackend uploads sealed segments and drops retired objects.
type SpillBackend interface {
	DemoteSegment(segID uint64, size int64, open func() (io.ReadSeekCloser, error), onUp func(uint64, bool)) bool
	Drop(ctx context.Context, segID uint64) error
	Stats() (spilled, dropped, putErrors uint64)
}

// RestoreBackend serves cold ranged reads and whole-segment lazy restores.
// RestoreSegment streams the byte-identical segment object through sink;
// implementations dedup concurrent calls per segment and bound their own
// I/O (the orchestrator adds a generous belt deadline on top).
type RestoreBackend interface {
	ReadRange(ctx context.Context, segID uint64, off, n int64, dst []byte) error
	RestoreSegment(ctx context.Context, segID uint64, sink func(io.Reader) error) error
	Stats() (rangedGets, restores uint64)
}

// spillPass runs on the demote/reclaim ticker: enqueue every sealed,
// not-yet-spilled segment. Fire-and-forget — a full queue is counted by
// the backend and simply retried next tick (the segment is still local).
func (t *Tiered) spillPass() {
	if t.p.Spill == nil {
		return
	}
	for _, vol := range t.vols {
		vol := vol
		t.scSegs = vol.SealedUnspilled(t.scSegs[:0])
		for _, si := range t.scSegs {
			si := si
			if _, loaded := t.spillInflight.LoadOrStore(si.ID, struct{}{}); loaded {
				continue // already queued
			}
			ok := t.p.Spill.DemoteSegment(uint64(si.ID), si.Size,
				func() (io.ReadSeekCloser, error) {
					f, _, err := vol.OpenSegmentReadOnly(si.ID)
					return f, err
				},
				func(segID uint64, up bool) {
					defer t.spillInflight.Delete(si.ID)
					if !up {
						return // still local-only; next tick retries
					}
					// Flag the index refs BEFORE the spilled marker: the
					// block's bytes now ALSO live on S3 (reads stay local
					// until reclaim retires the segment). Reclaim reads
					// IsSpilled as "every ref of this segment carries the
					// S3 flag" — marking first opened a window where the
					// retire-flip saw bare refs and stranded them as ghosts
					// (indexed, unreadable forever). MarkSpilled's volume
					// lock publishes the flag stores to IsSpilled readers.
					vol.SegmentEntryKeys(si.ID, func(ns uint32, key [32]byte, _, _ uint32) {
						if ref := t.idx.get(dram.Key{NS: ns, Hash: key}); ref != nil &&
							ref.Loc.SegmentID == si.ID {
							ref.S3.Store(true)
						}
					})
					vol.MarkSpilled(si.ID)
				})
			if !ok {
				t.spillInflight.Delete(si.ID)
			}
		}
	}
}

// readS3 serves one cold block from the segment object, verified before a
// byte escapes. Heap buffer (no arena hold), wire-deadline context.
func (t *Tiered) readS3(ref *nvmeRef) (data []byte, release func(), ok bool) {
	if t.p.Restore == nil || !ref.S3.Load() {
		return nil, nil, false
	}
	ctx, cancel := context.WithTimeout(context.Background(), t.s3ReadDeadline())
	defer cancel()
	buf := make([]byte, ref.Len)
	// Loc.Offset addresses the RECORD (56-byte header first); the payload
	// starts at RecordDataOffset. The object is the byte-identical segment
	// file, so file offsets transfer 1:1.
	if err := t.p.Restore.ReadRange(ctx, uint64(ref.Loc.SegmentID), nvme.RecordDataOffset(ref.Loc.Offset), int64(ref.Len), buf); err != nil {
		t.s3ReadErrs.Add(1)
		return nil, nil, false
	}
	if xxh3.Hash(buf) != ref.XXH3 {
		// The cold tier's OWN counter: an operator chasing corruption must
		// know whether to suspect the device or the object store — this once
		// rode the NVMe counter and misattributed S3 rot to the local disk.
		t.s3ChecksumErrs.Add(1)
		return nil, nil, false // never serve unverified bytes
	}
	t.s3Hits.Add(1)
	return buf, func() {}, true
}

func (t *Tiered) s3ReadDeadline() time.Duration {
	if t.p.S3ReadTimeout > 0 {
		return t.p.S3ReadTimeout
	}
	return 2 * time.Second
}

// s3FlipRetired converts a retired-but-spilled segment's entries to
// s3-resident: keep the index entry (Loc unchanged — it addresses the
// OBJECT now), move the tenant charge NVMe→S3. Each flip runs under the
// entry's shard lock so it linearizes against deleteNvme: a racing DELETE
// either removes the entry first (no transfer — the refund stayed NVMe)
// or observes S3Only and refunds the S3 side. The closure is also the
// AUTHORITATIVE protection gate, the same discipline as the delete path:
// a lease or pin landing after the advisory pre-gate strands the retire,
// never the block (a hard-pinned block must not become s3-only). Charges
// move ref.Len, not the footer entry's length — the index ref is the
// accounting truth, and a same-segment older generation's footer length
// would tear the charge/refund pairing.
//
// Returns the number of entries that BLOCK the retire: still homed in this
// segment but unflippable — protected, or their S3 flag never landed (a
// spill-ack ordering violation; belt over the flag-before-MarkSpilled
// contract). The caller must RetireAbort when it is nonzero: finishing
// would leave ghosts (indexed, unreadable forever). Entries that are gone,
// moved, or already flipped never block, and flips already applied stay —
// the flip is one-way and stays correct with the segment back live (reads
// keep serving locally; a delete refunds the S3 side the charge moved to).
func (t *Tiered) s3FlipRetired(vol *nvme.Volume, id uint32) (stranded int) {
	vol.SegmentEntryKeys(id, func(ns uint32, key [32]byte, _, _ uint32) {
		t.idx.withShardLock(dram.Key{NS: ns, Hash: key}, func(ref *nvmeRef) {
			if ref == nil || ref.Loc.SegmentID != id || ref.S3Only.Load() {
				return // gone, moved, or already flipped — never blocks
			}
			if stranded > 0 {
				return // retire already doomed — move no more charges early
			}
			if !ref.S3.Load() || ref.leased(t.now()) || ref.pinned() {
				stranded++
				return
			}
			if t.p.Quotas != nil {
				t.p.Quotas.Transfer(ns, tenant.TierNVMe, tenant.TierS3, int64(ref.Len))
			}
			ref.S3Only.Store(true)
			t.s3Blocks.Add(1)
			t.s3Bytes.Add(int64(ref.Len))
			// The object-GC liveness count moves with the flag, under the
			// same shard hold — a racing removal decrements only a count
			// that already landed (the demotion-publish charge discipline).
			t.s3SegAdd(id, 1)
		})
	})
	return stranded
}

// ---------------------------------------------------------------------------
// Lazy whole-segment restore (the promotion pass's cold half).
//
// A 2nd cold hit inside the promote window does NOT promote one block to
// DRAM (a cold segment's blocks come back together or not at all — one
// download amortizes over every entry): it downloads the WHOLE segment
// object back into the volume and flips the entries nvme-resident. The
// object is NOT dropped — the adopted segment republishes spilled=true, so
// the object backs its next retire-flip exactly like a fresh spill-ack's.
// A failed restore changes nothing — the entries stay s3-only and keep
// serving through readS3, loss-free.

// s3RestoreBelt bounds one whole-segment restore end to end. The backend's
// own OpTimeout is the real governor (the s3spill restorer allows 4× its
// per-op deadline for whole objects); this belt only exists so a hung
// third-party backend cannot wedge the restore goroutine — and with it the
// per-segment latch — forever. A cut download fails loss-free.
const s3RestoreBelt = 60 * time.Second

// restoreOne runs one queued restore on the restorer goroutine. The
// spillInflight latch is the cold tier's per-segment mutex: while held here,
// spillPass cannot re-upload the segment, gcPass cannot drop its object, and
// reclaimSegment cannot retire it mid-flip — every holder TRY-acquires and
// skips on busy, so the latch linearizes the segment's owners without ever
// blocking (no lock-order edge exists; it is not a mutex anyone waits on).
func (t *Tiered) restoreOne(ctx context.Context, req restoreReq) {
	if t.p.Restore == nil {
		return
	}
	if cur := t.idx.get(req.k); cur != req.ref || !req.ref.S3Only.Load() {
		return // replaced, removed, or flipped home since the hit — stale
	}
	segID := req.ref.Loc.SegmentID
	vol := t.volumeFor(req.k.Hash)
	if _, loaded := t.spillInflight.LoadOrStore(segID, struct{}{}); loaded {
		t.s3RestoreSkips.Add(1) // spill/GC owns the segment — a later hit retries
		return
	}
	defer t.spillInflight.Delete(segID)
	// Fresh adopt vs already local: an aborted retire can leave the segment
	// LIVE with s3-only stragglers — those need only the flip, never the
	// download.
	if !vol.HasSegment(segID) {
		if vol.MaxBytes() > 0 && vol.UsedBytes()+vol.SegmentBytes() > vol.MaxBytes() {
			// No headroom: adopting now would put the volume over budget and
			// reclaim would retire the restored bytes right back out (the
			// adopted segment is the OLDEST — it would thrash first). Kick
			// the reclaimer and let a later hit retry into freed room.
			// SegmentBytes is CURRENT-config geometry — a pre-retune object
			// may differ; the gate is advisory headroom only, AdoptSegment
			// validates the real size.
			t.s3RestoreSkips.Add(1)
			select {
			case t.kick <- struct{}{}:
			default:
			}
			return
		}
		rctx, cancel := context.WithTimeout(ctx, s3RestoreBelt)
		err := t.p.Restore.RestoreSegment(rctx, uint64(segID), func(r io.Reader) error {
			return vol.AdoptSegment(segID, r)
		})
		cancel()
		if err != nil {
			t.s3RestoreErrs.Add(1)
			return // entries stay s3-only and keep serving via readS3
		}
	}
	t.s3FlipRestored(vol, segID)
	t.s3SegRestores.Add(1)
	// The object is deliberately RETAINED. The adopted segment published
	// spilled=true (byte-identical copy — sealed segments are write-once),
	// so its next reclaim takes the FLIP branch against this object instead
	// of deleting the entries. Dropping it here was the reproduced data-loss
	// blocker: adopt-as-unspilled plus an inline drop made a reclaim landing
	// before the next spill-ack take the DELETE branch with the object
	// already gone — blocks lost on every tier. The object leaves ONLY via
	// the refund-driven liveness GC (dropq → gcPass) once the segment is
	// gone and its last s3-resident entry left.
}

// s3FlipRestored is s3FlipRetired's inverse: walk the (now local) segment's
// footer keys and flip every s3-only entry home — quota back S3→NVMe, the
// residency counters down, the GC liveness count down. Same discipline:
// each flip runs under the entry's shard lock (linearizes against deleteIf —
// a racing DELETE either removed the entry first, refunding the S3 side, or
// observes the flip and refunds NVMe), mutates only refs that still exist
// (a deleted key is NEVER resurrected — the walk inserts nothing), and
// moves ref.Len, the accounting truth, never the footer length.
//
// The per-ref S3 flag is KEPT on every flip, both callers: the segment
// object is retained (a fresh adopt republishes spilled=true; the
// already-local case never lost it), so the flag stays true in fact — and
// clearing it would strand the segment's NEXT retire forever (s3FlipRetired
// reads a bare flag as an ack-ordering violation and aborts, so a cleared
// flag would wedge reclaim on the segment for good). No protection gate:
// flipping home only ever INCREASES residency.
func (t *Tiered) s3FlipRestored(vol *nvme.Volume, id uint32) (flipped int) {
	vol.SegmentEntryKeys(id, func(ns uint32, key [32]byte, _, _ uint32) {
		before := flipped
		t.idx.withShardLock(dram.Key{NS: ns, Hash: key}, func(ref *nvmeRef) {
			// Gone, moved, or already home never flips (a same-segment older
			// generation visits one ref twice — the flag makes it flip once).
			if ref == nil || ref.Loc.SegmentID != id || !ref.S3Only.Load() {
				return
			}
			if t.p.Quotas != nil {
				t.p.Quotas.Transfer(ns, tenant.TierS3, tenant.TierNVMe, int64(ref.Len))
			}
			ref.S3Only.Store(false)
			t.s3Blocks.Add(-1)
			t.s3Bytes.Add(-int64(ref.Len))
			// Reaching zero here never nominates a drop: the segment is live
			// and spilled — its retained object backs the next retire-flip
			// (dropObjectHeld's IsSpilled arm would refuse anyway).
			t.s3SegAdd(id, -1)
			flipped++
		})
		if flipped > before && t.restoreFlipHookForTest != nil {
			// Test-only seam (nil in production): hold the mid-walk overtake
			// window open — some entries home, the rest still s3-only — so
			// the latch test can land a reclaim attempt inside it.
			t.restoreFlipHookForTest()
		}
	})
	return flipped
}

// ---------------------------------------------------------------------------
// Cold-tier object GC.
//
// A segment object is dead when its last s3-resident entry leaves the index
// (DELETE, corrupt-entry self-heal, reclaim removal, restore flip-back) and
// no live spilled segment still claims it. Dropping is best-effort and
// deadline-bounded, and NEVER fails or stalls the foreground op that freed
// the last entry: removals only enqueue the candidate; the demoter tick does
// the deleting, a bounded few per pass. A lost or failed drop just leaves an
// orphan the bucket lifecycle rule reaps.

// s3SegAdd moves a segment's s3-resident liveness count by d (+1 per
// retire-flip, −1 per removal or flip-back) and reports whether the count
// just reached zero. t.s3SegMu is a LEAF (the tenant-accountant posture):
// taken inside shard-locked closures, never held across any other lock.
// Every −1 pairs with a +1 that landed under the same shard hold (S3Only is
// the receipt), so a negative count is an accounting tear — the debug build
// trips on it; release self-heals to zero (delete + GC re-check).
func (t *Tiered) s3SegAdd(id uint32, d int) (zero bool) {
	t.s3SegMu.Lock()
	n := t.s3SegRefs[id] + d
	assertf(n >= 0, "s3SegAdd: segment %d liveness went negative (%d%+d)", id, n-d, d)
	if n <= 0 {
		delete(t.s3SegRefs, id)
		zero = true
	} else {
		t.s3SegRefs[id] = n
	}
	t.s3SegMu.Unlock()
	return zero && d < 0
}

func (t *Tiered) s3SegLive(id uint32) bool {
	t.s3SegMu.Lock()
	n := t.s3SegRefs[id]
	t.s3SegMu.Unlock()
	return n > 0
}

// enqueueObjectGC queues a dead-object candidate for the next gcPass. Drop
// on full: the object merely orphans (counted) and the lifecycle rule reaps
// it — GC must never block the removal path that produced the candidate.
// One nomination per candidate (s3GCQueued): a segment can cross zero
// liveness more than once (a retire-flip re-adds, removals re-drain), and
// candidates now sit in dropq across passes (gcDropBudget carries a
// backlog) — without the guard one id could queue twice and Drop twice.
// The set entry clears when gcPass dequeues, so a re-dead object
// re-nominates cleanly.
func (t *Tiered) enqueueObjectGC(segID uint32) {
	if t.p.Spill == nil {
		return
	}
	if _, queued := t.s3GCQueued.LoadOrStore(segID, struct{}{}); queued {
		return // already nominated — the queued candidate re-checks deadness
	}
	select {
	case t.dropq <- segID:
	default:
		t.s3GCQueued.Delete(segID)
		t.s3GCSkips.Add(1)
	}
}

// gcDropBudget bounds the object-Drop I/O one gcPass runs inline on the
// demoter goroutine: each drop is a deadline-bounded DeleteObject (up to
// ~2 s on a hung backend), and an unbounded drain of a full dropq could
// stall the memory ladder for minutes. The remainder stays queued for the
// next tick (100 ms cadence). Chosen over a separate drop worker for
// simplicity: the budget caps the worst-case tick stall at a few deadlines,
// which the ladder tolerates, and adds no goroutine, no shutdown edge.
const gcDropBudget = 4

// gcPass processes up to gcDropBudget queued candidates (demoter goroutine;
// also the tests' deterministic handle).
func (t *Tiered) gcPass() {
	for i := 0; i < gcDropBudget; i++ {
		select {
		case id := <-t.dropq:
			// Clear the nomination BEFORE the drop: a zero-crossing that
			// lands mid-drop re-nominates (an extra deadness re-check)
			// instead of being swallowed into a never-reaped orphan.
			t.s3GCQueued.Delete(id)
			t.dropObject(id)
		default:
			return
		}
	}
}

func (t *Tiered) dropObject(id uint32) {
	if _, loaded := t.spillInflight.LoadOrStore(id, struct{}{}); loaded {
		// A spill or restore owns the segment right now — deleting under a
		// concurrent PUT has no ordering guarantee. Skip; orphan-safe.
		t.s3GCSkips.Add(1)
		return
	}
	defer t.spillInflight.Delete(id)
	t.dropObjectHeld(id)
}

// dropObjectHeld re-checks deadness under the caller-held latch and drops.
// The re-check matters: between the enqueue and this pass the segment may
// have re-spilled and re-flipped — its object is live again.
func (t *Tiered) dropObjectHeld(id uint32) {
	if t.p.Spill == nil || t.s3SegLive(id) {
		return
	}
	for _, vol := range t.vols {
		if vol.IsSpilled(id) {
			return // a live spilled segment's object backs its next retire-flip
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), t.s3ReadDeadline())
	defer cancel()
	if err := t.p.Spill.Drop(ctx, uint64(id)); err != nil {
		t.s3GCErrs.Add(1) // orphan — the bucket lifecycle rule is the backstop
		return
	}
	t.s3GCs.Add(1)
}
