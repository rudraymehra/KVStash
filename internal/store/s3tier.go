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
					vol.MarkSpilled(si.ID)
					// Flip index refs: the block's bytes now ALSO live on
					// S3. Reads stay local until reclaim removes the
					// segment; the flag is what reclaim's flip consults.
					vol.SegmentEntryKeys(si.ID, func(ns uint32, key [32]byte, _, _ uint32) {
						if ref := t.idx.get(dram.Key{NS: ns, Hash: key}); ref != nil &&
							ref.Loc.SegmentID == si.ID {
							ref.S3.Store(true)
						}
					})
				})
			if !ok {
				t.spillInflight.Delete(si.ID)
			}
		}
	}
}

// readS3 serves one cold block from the segment object, verified before a
// byte escapes. Heap buffer (no arena hold), wire-deadline context.
func (t *Tiered) readS3(ref *nvmeRef, ns uint32, key [32]byte) (data []byte, release func(), ok bool) {
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
		t.checksumErrs.Add(1)
		return nil, nil, false // never serve unverified bytes
	}
	_ = ns
	_ = key
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
// OBJECT now), move the tenant charge NVMe→S3. Returns entries flipped.
func (t *Tiered) s3FlipRetired(vol *nvme.Volume, id uint32) int {
	flipped := 0
	vol.SegmentEntryKeys(id, func(ns uint32, key [32]byte, _, length uint32) {
		ref := t.idx.get(dram.Key{NS: ns, Hash: key})
		if ref == nil || ref.Loc.SegmentID != id || !ref.S3.Load() {
			return
		}
		if t.p.Quotas != nil {
			t.p.Quotas.Transfer(ns, tenant.TierNVMe, tenant.TierS3, int64(length))
		}
		flipped++
	})
	return flipped
}
