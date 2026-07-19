package nvme

import (
	"fmt"
	"os"
)

// The spill surface: what the S3 cold tier needs from a volume — sealed
// segments enumerated, opened read-only as plain files, and marked spilled.
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
