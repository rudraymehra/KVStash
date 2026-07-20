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
// escapes. EXISTS never touches S3 (index-only, as always).

// SpillBackend uploads sealed segments and drops retired objects.
type SpillBackend interface {
	DemoteSegment(segID uint64, size int64, open func() (io.ReadSeekCloser, error), onUp func(uint64, bool)) bool
	Drop(ctx context.Context, segID uint64) error
	Stats() (spilled, dropped, putErrors uint64)
}

// RestoreBackend serves cold ranged reads (and, later, whole-segment lazy
// restores — wired in the promotion pass).
type RestoreBackend interface {
	ReadRange(ctx context.Context, segID uint64, off, n int64, dst []byte) error
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
		})
	})
	return stranded
}
