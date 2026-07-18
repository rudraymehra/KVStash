package nvme

import (
	"fmt"
	"os"
	"time"
)

// Whole-segment FIFO reclaim mechanics. The reclaim POLICY loop (when to
// reclaim, which entries to gate on lease/pin) lives in the tiered store —
// this file provides the safe retire primitives.
//
// Retire protocol (the tiered store's reclaimSegment drives it):
//
//	RetireBegin  — flips dying: new reads refuse, in-flight reads finish.
//	RetireAbort  — un-flips after a failed re-gate.
//	RetireFinish — unlinks the file (open fd keeps data readable for
//	               stragglers — POSIX, linux+darwin), waits for the read
//	               count to drain, closes the fd, forgets the segment.

// OldestSealed returns the lowest-ID sealed segment and a copy of its entry
// table, or ok=false when nothing is reclaimable.
func (v *Volume) OldestSealed() (id uint32, entries []footerEntry, ok bool) {
	v.mu.RLock()
	defer v.mu.RUnlock()
	found := false
	for sid, s := range v.segs {
		if !s.sealed || s.dying.Load() {
			continue
		}
		if !found || sid < id {
			id, found = sid, true
		}
	}
	if !found {
		return 0, nil, false
	}
	src := v.segs[id].entries
	entries = make([]footerEntry, len(src))
	copy(entries, src)
	return id, entries, true
}

// RetireBegin marks the segment dying. False = unknown, active, or already
// dying.
func (v *Volume) RetireBegin(id uint32) bool {
	v.mu.RLock()
	s := v.segs[id]
	v.mu.RUnlock()
	if s == nil || !s.sealed {
		return false
	}
	return s.dying.CompareAndSwap(false, true)
}

// RetireAbort reverses RetireBegin after a failed gate re-check.
func (v *Volume) RetireAbort(id uint32) {
	v.mu.RLock()
	s := v.segs[id]
	v.mu.RUnlock()
	if s != nil {
		s.dying.Store(false)
	}
}

// RetireFinish unlinks and forgets the segment. The caller has already
// removed every index entry pointing here. Blocks briefly while straggler
// reads (acquired before dying flipped) drain.
func (v *Volume) RetireFinish(id uint32) error {
	v.mu.Lock()
	s := v.segs[id]
	if s == nil || !s.dying.Load() {
		v.mu.Unlock()
		return fmt.Errorf("nvme: retire finish on segment %d not begun", id)
	}
	delete(v.segs, id)
	v.mu.Unlock()

	if err := os.Remove(s.path); err != nil && !os.IsNotExist(err) {
		v.log.Warn("nvme: unlink reclaimed segment", "path", s.path, "err", err)
	}
	// In-flight readers hold the open fd; data stays readable post-unlink.
	for s.reads.Load() > 0 {
		time.Sleep(time.Millisecond)
	}
	if err := s.f.Close(); err != nil {
		v.log.Warn("nvme: close reclaimed segment", "path", s.path, "err", err)
	}
	v.used.Add(-s.size) // the segment's OWN geometry, not the current config's
	v.reclaims.Add(1)
	v.clearReadOnly() // freed space — writes may resume
	if err := SyncDir(v.p.Dir); err != nil {
		v.log.Warn("nvme: dir sync after reclaim", "err", err)
	}
	return nil
}
