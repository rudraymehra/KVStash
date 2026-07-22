package nvme

import (
	"fmt"
	"io"
	"os"
)

// The spill surface: what the S3 cold tier needs from a volume — sealed
// segments enumerated, opened read-only as plain files, marked spilled, and
// (for the lazy-restore path) adopted back whole from a downloaded object.
// Spilled state is IN-MEMORY only: after a restart every sealed segment
// re-spills, and the whole-object PUT is idempotent (same key, same bytes),
// so a crash costs one duplicate upload, never correctness.

// SegmentInfo describes one sealed segment for the spill driver.
type SegmentInfo struct {
	ID      uint32
	Size    int64 // full file size (records + footer) — the S3 object length
	Entries int
}

// SealedUnspilled lists sealed, not-yet-spilled, not-dying segments.
func (v *Volume) SealedUnspilled(dst []SegmentInfo) []SegmentInfo {
	v.mu.RLock()
	defer v.mu.RUnlock()
	for _, s := range v.segs {
		if !s.sealed || s.spilled || s.dying.Load() {
			continue
		}
		dst = append(dst, SegmentInfo{ID: s.id, Size: s.size, Entries: len(s.entries)})
	}
	return dst
}

// MarkSpilled records that a segment's bytes landed on S3 (spill-ack).
func (v *Volume) MarkSpilled(id uint32) {
	v.mu.Lock()
	defer v.mu.Unlock()
	if s, ok := v.segs[id]; ok {
		s.spilled = true
	}
}

// IsSpilled reports whether the segment's S3 copy exists (reclaim's
// flip-vs-delete decision).
func (v *Volume) IsSpilled(id uint32) bool {
	v.mu.RLock()
	defer v.mu.RUnlock()
	s, ok := v.segs[id]
	return ok && s.spilled
}

// OpenSegmentReadOnly opens the segment FILE for the uploader. A plain
// second fd: POSIX keeps the bytes alive even if reclaim unlinks the path
// mid-upload (the upload then finishes from the orphaned inode — harmless;
// the spill-ack simply finds the segment gone and no entry flips).
func (v *Volume) OpenSegmentReadOnly(id uint32) (*os.File, int64, error) {
	v.mu.RLock()
	s, ok := v.segs[id]
	v.mu.RUnlock()
	if !ok {
		return nil, 0, fmt.Errorf("nvme: segment %d gone", id)
	}
	f, err := os.Open(s.path) //nolint:gosec // G304: the volume's own segment path
	if err != nil {
		return nil, 0, err
	}
	return f, s.size, nil
}

// SegmentEntryKeys iterates a sealed segment's footer entries (the spill-ack
// flip walks these to mark index refs s3-backed).
func (v *Volume) SegmentEntryKeys(id uint32, fn func(ns uint32, key [32]byte, off, length uint32)) {
	v.mu.RLock()
	s, ok := v.segs[id]
	if !ok {
		v.mu.RUnlock()
		return
	}
	ents := s.entries
	v.mu.RUnlock()
	for _, e := range ents {
		fn(e.NS, e.Key, e.Off, e.Len)
	}
}

// RecordDataOffset converts a footer/Loc record offset (the record START)
// into the payload's byte offset inside the segment file — the ranged-read
// offset a byte-identical S3 object serves.
func RecordDataOffset(recordOff uint32) int64 { return int64(recordOff) + recordHdrSize }

// HasSegment reports whether the segment is currently live in this volume —
// the restore orchestrator's adopt-vs-already-local routing.
func (v *Volume) HasSegment(id uint32) bool {
	v.mu.RLock()
	defer v.mu.RUnlock()
	_, ok := v.segs[id]
	return ok
}

// AdoptSegment writes a whole-object restore back into the volume as a live
// sealed segment. r must stream the byte-identical segment file the spiller
// uploaded: the bytes land in a temp file first, the seal (trailer magic /
// CRC / segID + per-entry bounds) is validated BEFORE publication, and the
// rename + dir sync make the adoption durable — a crash mid-adopt leaves
// only a .tmp that recovery prunes. Record payloads are NOT re-verified
// here: every read still xxh3-verifies before a byte escapes, so object rot
// downgrades to per-key self-heal misses, never served corruption.
//
// The adopted segment publishes spilled=true: the object it was downloaded
// from still exists and is byte-identical (sealed segments are write-once),
// so the S3 copy is exactly as valid as any spill-ack's. That keeps the
// segment off SealedUnspilled (no wasteful re-upload) and — the part that
// matters — makes a later reclaim take its FLIP branch: entries move back
// to s3-residency against the retained object instead of being deleted.
// The previous spilled=false publish paired with an inline object drop, and
// a reclaim landing before the next spill-ack then took the DELETE branch
// with the object already gone — the restored blocks were lost on every
// tier.
//
// The caller serializes adopts per segment id (the tiered store's
// spillInflight latch) and only adopts ids this volume once minted, so an
// adopted id can never collide with the active segment's.
func (v *Volume) AdoptSegment(id uint32, r io.Reader) error {
	if v.closed.Load() {
		return fmt.Errorf("nvme: adopt segment %d: volume closed", id)
	}
	v.mu.RLock()
	_, exists := v.segs[id]
	next := v.nextID
	v.mu.RUnlock()
	if exists {
		return fmt.Errorf("nvme: adopt segment %d: already live", id)
	}
	if id >= next {
		// Ids at or above nextID belong to future active segments; an object
		// claiming one was never this volume's to restore.
		return fmt.Errorf("nvme: adopt segment %d: beyond next id %d", id, next)
	}
	path := segPath(v.p.Dir, id)
	tmp := path + ".tmp"
	f, err := v.backend.Open(tmp, true)
	if err != nil {
		return fmt.Errorf("nvme: adopt segment %d: %w", id, err)
	}
	discard := func() {
		_ = f.Close()
		_ = os.Remove(tmp)
	}
	size, err := copyAligned(f, r)
	if err != nil {
		discard()
		return fmt.Errorf("nvme: adopt segment %d: %w", id, err)
	}
	if size < 2*recordAlign || size%recordAlign != 0 || size >= int64(^uint32(0)) {
		discard()
		return fmt.Errorf("nvme: adopt segment %d: object size %d is no segment", id, size)
	}
	if err := f.Datasync(); err != nil {
		discard()
		return fmt.Errorf("nvme: adopt segment %d: %w", id, err)
	}
	tr, entries, sealed := readSeal(f, size, id, v.p.MaxBlobLen)
	if !sealed {
		discard()
		return fmt.Errorf("nvme: adopt segment %d: seal invalid", id)
	}
	if err := os.Rename(tmp, path); err != nil {
		discard()
		return fmt.Errorf("nvme: adopt segment %d: %w", id, err)
	}
	if err := SyncDir(v.p.Dir); err != nil {
		// Not durable: a crash could unwind the rename after the caller
		// flipped entries home. Refuse to publish — the caller counts a
		// failed restore and leaves the entries s3-only (loss-free: the
		// object is retained and reads keep serving through it), and a
		// later cold hit retries the whole restore.
		_ = f.Close()
		_ = os.Remove(path)
		return fmt.Errorf("nvme: adopt segment %d: dir sync: %w", id, err)
	}
	v.mu.Lock()
	if v.closed.Load() || v.segs[id] != nil {
		v.mu.Unlock()
		_ = f.Close()
		_ = os.Remove(path)
		return fmt.Errorf("nvme: adopt segment %d: lost the publish race", id)
	}
	v.segs[id] = &segment{
		id: id, f: f, path: path, size: size, sealed: true, spilled: true,
		dataEnd: tr.DataEnd, entries: entries,
	}
	v.used.Add(size)
	v.mu.Unlock()
	return nil
}

// copyAligned streams r into f through a page-aligned buffer (direct-I/O
// handles need aligned p/off; segment files are 4 KiB-multiples, so every
// full chunk and any legal final chunk stay aligned). The copy ABORTS the
// moment the stream would cross the format's maximum segment size (sizes
// and offsets are uint32): a corrupt or hostile oversized object must fail
// here, not fill the filesystem first and only then flunk the post-copy
// size check. Returns bytes written.
func copyAligned(f File, r io.Reader) (int64, error) {
	const maxSegBytes = int64(^uint32(0)) // no legal segment reaches 4 GiB
	buf, free, err := alignedTemp(1 << 20)
	if err != nil {
		return 0, err
	}
	defer free()
	var off int64
	for {
		n, rerr := io.ReadFull(r, buf)
		if n > 0 {
			if n%recordAlign != 0 {
				return 0, fmt.Errorf("stream tail %d bytes off record alignment", n)
			}
			if off+int64(n) >= maxSegBytes {
				return 0, fmt.Errorf("stream exceeds the %d-byte segment size limit", maxSegBytes)
			}
			if werr := f.WriteAt(buf[:n], off); werr != nil {
				return 0, werr
			}
			off += int64(n)
		}
		if rerr == io.EOF || rerr == io.ErrUnexpectedEOF {
			return off, nil
		}
		if rerr != nil {
			return 0, rerr
		}
	}
}
