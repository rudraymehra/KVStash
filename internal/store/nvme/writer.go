package nvme

import (
	"errors"
	"time"

	"golang.org/x/sys/unix"
)

// pendingRec tracks one staged-but-unwritten record through a flush.
type pendingRec struct {
	entry     footerEntry
	onWritten func(loc Loc, ok bool)
}

// writerLoop is the volume's single writer goroutine: it drains AppendReqs
// greedily into an aligned staging buffer (group commit), flushes with one
// pwrite, fdatasyncs on the SyncEveryBytes ledger and at every seal, and
// rotates segments leaving room for the seal footer. All segment mutation
// happens here; readers only ever see immutable bytes at offsets below a
// published Loc.
func (v *Volume) writerLoop() {
	defer close(v.writerDone)

	stagingCap := int(recordSpan(v.p.MaxBlobLen)) //nolint:gosec // G115: span < 4 GiB (validated in OpenVolume)
	if stagingCap < 4<<20 {
		stagingCap = 4 << 20
	}
	stagingBuf, err := mmapBuf(stagingCap)
	if err != nil {
		// Cannot stage — the volume is write-dead. Reads still work.
		v.log.Error("nvme: writer staging alloc failed — volume read-only", "err", err)
		v.setReadOnly()
		for {
			select {
			case req := <-v.reqs:
				req.OnWritten(Loc{}, false)
			case <-v.writerStop:
				return
			}
		}
	}
	defer func() { _ = unix.Munmap(stagingBuf) }()

	staging := stagingBuf[:0]
	var pending []pendingRec
	var unsynced int64

	flush := func() bool {
		if len(pending) == 0 {
			return true
		}
		ok := v.flushBatch(staging, pending, &unsynced)
		staging = stagingBuf[:0]
		pending = pending[:0]
		return ok
	}

	for {
		var req AppendReq
		select {
		case req = <-v.reqs:
		case <-v.writerStop:
			if !v.crashed.Load() {
				_ = flush()
				v.finalSync(unsynced)
			}
			return
		}

	stage:
		if len(req.Data) > int(v.p.MaxBlobLen) { // bound FIRST — the conversions below rely on it
			req.OnWritten(Loc{}, false)
			continue
		}
		span := int(recordSpan(uint32(len(req.Data)))) //nolint:gosec // G115: len ≤ MaxBlobLen (checked above); span < 4 GiB

		// Rotation: the record plus the eventual footer must fit the segment.
		if !v.fitsActive(span, len(staging), len(pending)+1) {
			if !flush() || !v.rotate() {
				req.OnWritten(Loc{}, false)
				continue
			}
		}
		// Staging room: flush the current batch first if this record spills.
		if len(staging)+span > cap(staging) {
			if !flush() {
				req.OnWritten(Loc{}, false)
				continue
			}
			goto stage // re-check rotation against the freshly advanced dataEnd
		}

		off := v.activeDataEnd() + uint32(len(staging))                           //nolint:gosec // G115: staging < 4 GiB
		h := recordHeader{NS: req.NS, Len: uint32(len(req.Data)), XXH3: req.XXH3} //nolint:gosec // G115: bounded above
		h.Key = req.Key
		staging = appendRecord(staging, h, req.Data)
		pending = append(pending, pendingRec{
			entry:     footerEntry{NS: req.NS, Key: req.Key, Off: off, Len: h.Len, XXH3: req.XXH3},
			onWritten: req.OnWritten,
		})

		// Greedy drain: batch everything already queued that still fits.
		for len(v.reqs) > 0 && len(pending) < 64 {
			select {
			case nxt := <-v.reqs:
				if len(nxt.Data) > int(v.p.MaxBlobLen) {
					nxt.OnWritten(Loc{}, false)
					continue
				}
				nspan := int(recordSpan(uint32(len(nxt.Data)))) //nolint:gosec // G115: len ≤ MaxBlobLen (checked above)
				if len(staging)+nspan > cap(staging) || !v.fitsActive(nspan, len(staging), len(pending)+1) {
					// Doesn't fit this batch — write what we have, then re-stage it.
					if !flush() {
						nxt.OnWritten(Loc{}, false)
						continue
					}
					req = nxt
					goto stage
				}
				noff := v.activeDataEnd() + uint32(len(staging))                           //nolint:gosec // G115: staging < 4 GiB
				nh := recordHeader{NS: nxt.NS, Len: uint32(len(nxt.Data)), XXH3: nxt.XXH3} //nolint:gosec // G115: bounded above
				nh.Key = nxt.Key
				staging = appendRecord(staging, nh, nxt.Data)
				pending = append(pending, pendingRec{
					entry:     footerEntry{NS: nxt.NS, Key: nxt.Key, Off: noff, Len: nh.Len, XXH3: nxt.XXH3},
					onWritten: nxt.OnWritten,
				})
			default:
			}
		}
		_ = flush()
	}
}

// flushBatch writes the staging buffer at the active segment's dataEnd,
// advances it, applies the group-commit sync ledger, and fires OnWritten.
func (v *Volume) flushBatch(staging []byte, pending []pendingRec, unsynced *int64) bool {
	seg := v.activeSeg()
	if seg == nil {
		for _, p := range pending {
			p.onWritten(Loc{}, false)
		}
		return false
	}
	err := seg.f.WriteAt(staging, int64(seg.dataEnd))
	if err != nil {
		v.appendFailed(err)
		for _, p := range pending {
			p.onWritten(Loc{}, false)
		}
		return false
	}
	// Publish entries + advance dataEnd under the lock so a concurrent seal
	// observer (Stats/recovery-in-tests) sees a consistent segment.
	v.mu.Lock()
	seg.dataEnd += uint32(len(staging)) //nolint:gosec // G115: dataEnd bounded by segment_bytes < 4 GiB
	for _, p := range pending {
		seg.entries = append(seg.entries, p.entry)
	}
	v.mu.Unlock()

	*unsynced += int64(len(staging))
	if *unsynced >= v.p.SyncEveryBytes {
		if err := seg.f.Datasync(); err != nil {
			v.log.Error("nvme: group-commit fdatasync failed — volume read-only", "err", err)
			v.setReadOnly()
		}
		*unsynced = 0
	}
	v.appended.Add(uint64(len(pending)))
	for _, p := range pending {
		p.onWritten(Loc{SegmentID: seg.id, Offset: p.entry.Off, Len: p.entry.Len}, true)
	}
	return true
}

func (v *Volume) finalSync(unsynced int64) {
	if unsynced == 0 {
		return
	}
	if seg := v.activeSeg(); seg != nil {
		if err := seg.f.Datasync(); err != nil {
			v.log.Warn("nvme: close-time fdatasync", "err", err)
		}
	}
}

func (v *Volume) activeSeg() *segment {
	v.mu.RLock()
	defer v.mu.RUnlock()
	return v.active
}

func (v *Volume) activeDataEnd() uint32 {
	v.mu.RLock()
	defer v.mu.RUnlock()
	if v.active == nil {
		return 0
	}
	return v.active.dataEnd
}

// fitsActive reports whether span more bytes — ON TOP of the staged-but-
// unflushed bytes — plus the seal footer for the current entries + extra
// records still fit the active segment. Forgetting `staged` was a real
// bug the scrub caught: a greedy batch checked each record against a stale
// dataEnd, and the seal trailer later overwrote the last record's tail.
func (v *Volume) fitsActive(span, staged, extra int) bool {
	v.mu.RLock()
	defer v.mu.RUnlock()
	if v.active == nil {
		return false
	}
	entries := len(v.active.entries) + extra
	reserve := roundUpAlign(uint64(entries)*footerEntrySz) + recordAlign                            //nolint:gosec // G115: entries is a small non-negative count — table + trailer chunk
	return uint64(v.active.dataEnd)+uint64(staged)+uint64(span)+reserve <= uint64(v.p.SegmentBytes) //nolint:gosec // G115: staged/span small non-negative; SegmentBytes validated positive
}

// rotate seals the active segment and opens a fresh one. False = the volume
// is write-dead (ENOSPC or seal failure); it flips read-only.
func (v *Volume) rotate() bool {
	if err := v.sealActive(); err != nil {
		v.appendFailed(err)
		return false
	}
	if err := v.openActive(); err != nil {
		v.appendFailed(err)
		return false
	}
	v.maybeCheckpoint()
	return true
}

// sealActive writes the footer (entry table + trailer), fdatasyncs, and
// marks the segment immutable. Runs on the writer goroutine. A nil active
// segment is a no-op — after a failed rotation the retry path must reach
// openActive, not dead-loop on "nothing to seal" (the ladder's write-dead
// MED).
func (v *Volume) sealActive() error {
	seg := v.activeSeg()
	if seg == nil {
		return nil
	}
	// Snapshot under RLock; entries slice is writer-owned so this is safe.
	v.mu.RLock()
	entries := seg.entries
	dataEnd := seg.dataEnd
	v.mu.RUnlock()

	if err := writeSeal(seg.f, seg.size, seg.id, dataEnd, entries); err != nil {
		return err
	}
	v.mu.Lock()
	seg.sealed = true
	v.active = nil
	v.sealsSinceCkpt++
	v.mu.Unlock()
	v.seals.Add(1)
	return nil
}

// writeSeal is the shared seal writer (also used by recovery to seal a
// recovered tail prefix in place): entry-table chunk at dataEnd, trailer in
// the file's final 4 KiB, one fdatasync. All device writes go through
// explicitly page-aligned buffers — heap slices only HAPPEN to be aligned
// under today's allocator (ladder finding).
func writeSeal(f File, segBytes int64, segID, dataEnd uint32, entries []footerEntry) error {
	table := encodeEntries(entries)
	tr := trailer{
		EntryCount: uint32(len(entries)), //nolint:gosec // G115: entries bounded by segment capacity
		DataEnd:    dataEnd,
		FooterOff:  dataEnd,
		SegID:      segID,
	}
	pre := encodeTrailer(tr)
	tr.CRC = sealCRC(table, pre[:60])
	full := encodeTrailer(tr)

	if len(table) > 0 {
		atable, freeTable, err := alignedTemp(len(table))
		if err != nil {
			return err
		}
		copy(atable, table)
		err = f.WriteAt(atable, int64(dataEnd))
		freeTable()
		if err != nil {
			return err
		}
	}
	tail, freeTail, err := alignedTemp(recordAlign)
	if err != nil {
		return err
	}
	copy(tail[recordAlign-trailerSize:], full[:])
	err = f.WriteAt(tail, segBytes-recordAlign)
	freeTail()
	if err != nil {
		return err
	}
	return f.Datasync()
}

// appendFailed handles a write-path error: ENOSPC (and any persistent write
// failure) flips the volume read-only until reclaim frees space; logged once
// per flip.
func (v *Volume) appendFailed(err error) {
	if errors.Is(err, unix.ENOSPC) {
		v.enospc.Add(1)
	}
	v.log.Error("nvme: append path failed — volume read-only until reclaim", "err", err)
	v.setReadOnly()
}

func (v *Volume) setReadOnly() {
	v.mu.Lock()
	v.readOnly = true
	v.mu.Unlock()
}

// clearReadOnly is called by reclaim after freeing a segment.
func (v *Volume) clearReadOnly() {
	v.mu.Lock()
	v.readOnly = false
	v.mu.Unlock()
}

// maybeCheckpoint writes a checkpoint of every sealed segment's entries
// after CkptEverySegs seals. Runs on the writer goroutine; failure is
// logged and retried at the next seal (checkpoints are an optimization —
// recovery falls back to footer scans).
func (v *Volume) maybeCheckpoint() {
	v.mu.RLock()
	due := v.p.CkptEverySegs > 0 && v.sealsSinceCkpt >= v.p.CkptEverySegs
	v.mu.RUnlock()
	if !due {
		return
	}
	start := time.Now()
	if err := v.writeCheckpoint(); err != nil {
		v.log.Warn("nvme: checkpoint write failed (will retry next seal)", "err", err)
		return
	}
	v.mu.Lock()
	v.sealsSinceCkpt = 0
	v.mu.Unlock()
	v.ckpts.Add(1)
	v.log.Info("nvme: checkpoint written", "dir", v.p.Dir, "took", time.Since(start))
}
