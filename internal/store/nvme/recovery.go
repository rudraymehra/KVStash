package nvme

import (
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/zeebo/xxh3"
)

// RecoveryReport is what OpenVolume found — logged at startup and quoted by
// the kill-9 demo.
type RecoveryReport struct {
	SegmentsScanned int
	BlocksRecovered int
	BytesTruncated  int64 // lower bound: the first torn region observed per tail
	Duration        time.Duration
}

// RecoveredEntry is one readable block recovery vouches for (its xxh3 will
// still be re-verified on every read).
type RecoveredEntry struct {
	NS   uint32
	Key  [32]byte
	Loc  Loc
	XXH3 uint64
}

// recoverDir rebuilds the segment map and the recoverable entry set:
//
//  1. newest CRC-valid checkpoint → entries for segments ≤ maxSealedSegID
//     (validated, then trusted without footer reads — the warm-restart win).
//     The entries are ALSO attached to the in-memory segment (the review
//     ladder's confirmed blocker: a trusted segment with a nil entry table
//     poisoned the NEXT checkpoint into dropping it, and gave reclaim
//     nothing to gate on);
//  2. sealed segments newer than the checkpoint → footer scan; on the same
//     (ns, key) the LATER (segID, offset) wins — offset order matters too,
//     because delete + re-put can land two generations of one key in the
//     SAME segment (the ladder's confirmed serves-superseded-bytes HIGH);
//  3. every unsealed segment (crash-mid-rotation can leave more than one)
//     → forward record scan, verify xxh3 per record, truncate at the first
//     torn/short/bad record, then SEAL the recovered prefix in place — no
//     append ever resumes into a scanned tail;
//  4. empty unsealed segments and genuinely partial creates are deleted.
//     A sealed segment whose size differs from the CURRENT config (an
//     operator retuned nvme_segment_bytes) is NOT deleted — geometry is
//     read from the file itself (the ladder's config-edit-wipes-the-tier
//     HIGH); only non-4KiB-multiple / too-small files are partial creates.
//
// Lose data, never serve corruption.
func (v *Volume) recoverDir() (*RecoveryReport, []RecoveredEntry, error) {
	start := time.Now()
	report := &RecoveryReport{}

	pruneTmpFiles(v.p.Dir, v.log.Warn)

	segIDs, err := listSegIDs(v.p.Dir, v.log.Warn)
	if err != nil {
		return nil, nil, err
	}
	sort.Slice(segIDs, func(i, j int) bool { return segIDs[i] < segIDs[j] })

	ckEntries, ckMaxSealed, ckSeq, ckOK := loadNewestCkpt(v.p.Dir, v.log.Warn)
	if ckOK {
		v.ckptSeq = ckSeq
	}
	// Group checkpoint entries by segment so trusted segments carry their
	// own entry tables (checkpoint regeneration + reclaim gating need them).
	ckBySeg := make(map[uint32][]footerEntry)
	for _, ce := range ckEntries {
		ckBySeg[ce.SegID] = append(ckBySeg[ce.SegID],
			footerEntry{NS: ce.NS, Key: ce.Key, Off: ce.Off, Len: ce.Len, XXH3: ce.XXH3})
	}

	// keyed merge — later (segID, offset) wins on the same (ns, key).
	merged := make(map[[36]byte]RecoveredEntry)
	mergeKey := func(ns uint32, key [32]byte) [36]byte {
		var k [36]byte
		binary.LittleEndian.PutUint32(k[0:4], ns)
		copy(k[4:], key[:])
		return k
	}
	insert := func(e RecoveredEntry) {
		k := mergeKey(e.NS, e.Key)
		if prev, ok := merged[k]; ok &&
			(prev.Loc.SegmentID > e.Loc.SegmentID ||
				(prev.Loc.SegmentID == e.Loc.SegmentID && prev.Loc.Offset >= e.Loc.Offset)) {
			return
		}
		merged[k] = e
	}
	indexSegment := func(id uint32, entries []footerEntry) {
		for _, e := range entries {
			insert(RecoveredEntry{
				NS: e.NS, Key: e.Key, XXH3: e.XXH3,
				Loc: Loc{SegmentID: id, Offset: e.Off, Len: e.Len},
			})
			report.BlocksRecovered++
		}
	}

	minSize := MinSegmentBytes(v.p.MaxBlobLen)
	for _, id := range segIDs {
		path := segPath(v.p.Dir, id)
		f, err := v.backend.Open(path, true)
		if err != nil {
			v.log.Warn("nvme: recovery cannot open segment — skipped", "path", path, "err", err)
			continue
		}
		st, err := os.Stat(path)
		var size int64 = -1
		if err == nil {
			size = st.Size()
		}
		if err != nil || size%recordAlign != 0 || size < minSize {
			// Crash during create (partial fallocate — the create fsync had
			// not completed): nothing durable can be in here. Delete. A
			// MERELY different-but-valid size (config retune) never lands
			// here — geometry is honored from the file below.
			v.log.Warn("nvme: recovery dropping partial-create segment", "path", path, "size", size)
			_ = f.Close()
			_ = os.Remove(path)
			continue
		}
		report.SegmentsScanned++

		if ckOK && id <= ckMaxSealed {
			// Trusted-sealed via checkpoint: entries come from the ckpt (the
			// footer stays unread; reads still xxh3-verify every block).
			// Bounds are validated against the file's own size; dataEnd is
			// derived from the entries so downstream checks stay honest.
			entries := ckBySeg[id]
			valid := make([]footerEntry, 0, len(entries))
			var dataEnd uint64
			limit := uint64(size) - recordAlign //nolint:gosec // G115: size ≥ minSize > recordAlign
			for _, e := range entries {
				end := uint64(e.Off) + recordSpan(e.Len)
				if e.Len <= v.p.MaxBlobLen && e.Off%recordAlign == 0 && end <= limit {
					valid = append(valid, e)
					if end > dataEnd {
						dataEnd = end
					}
				}
			}
			if len(valid) < len(entries) {
				v.log.Warn("nvme: recovery dropped out-of-bounds checkpoint entries",
					"segment", id, "dropped", len(entries)-len(valid))
			}
			seg := &segment{
				id: id, f: f, path: path, size: size, sealed: true,
				dataEnd: uint32(dataEnd), entries: valid,
			} //nolint:gosec // G115: dataEnd ≤ size < 4 GiB
			v.segs[id] = seg
			v.used.Add(size)
			indexSegment(id, valid)
			continue
		}

		tr, entries, sealed := readSeal(f, size, id, v.p.MaxBlobLen)
		if sealed {
			seg := &segment{
				id: id, f: f, path: path, size: size, sealed: true,
				dataEnd: tr.DataEnd, entries: entries,
			}
			v.segs[id] = seg
			v.used.Add(size)
			indexSegment(id, entries)
			continue
		}

		// Unsealed tail: forward scan + truncate + seal-in-place.
		recovered, dataEnd, truncated := v.scanTail(f, size, id)
		report.BytesTruncated += truncated
		if len(recovered) == 0 {
			_ = f.Close()
			_ = os.Remove(path)
			v.log.Info("nvme: recovery dropped empty tail segment", "path", path)
			continue
		}
		if err := writeSeal(f, size, id, dataEnd, recovered); err != nil {
			// Can't make the prefix immutable — refuse to serve from it.
			v.log.Warn("nvme: recovery could not seal tail — segment dropped", "path", path, "err", err)
			_ = f.Close()
			_ = os.Remove(path)
			continue
		}
		seg := &segment{
			id: id, f: f, path: path, size: size, sealed: true,
			dataEnd: dataEnd, entries: recovered,
		}
		v.segs[id] = seg
		v.used.Add(size)
		indexSegment(id, recovered)
	}

	// nextID must clear BOTH the present files and everything a surviving
	// checkpoint could ever vouch for — reusing an ID ≤ maxSealedSegID
	// would let a stale checkpoint "trust" a brand-new unsealed segment
	// (the ladder's ID-reuse MED).
	if n := len(segIDs); n > 0 {
		v.nextID = segIDs[n-1] + 1
	}
	if ckOK && ckMaxSealed+1 > v.nextID {
		v.nextID = ckMaxSealed + 1
	}

	out := make([]RecoveredEntry, 0, len(merged))
	for _, e := range merged {
		out = append(out, e)
	}
	report.Duration = time.Since(start)
	v.log.Info("nvme: volume recovered",
		"dir", v.p.Dir,
		"segments_scanned", report.SegmentsScanned,
		"blocks_recovered", len(out),
		"bytes_truncated", report.BytesTruncated,
		"took", report.Duration)
	return report, out, nil
}

// readSeal validates a segment's seal (trailer magic/version/segID, CRC over
// entry table + trailer prefix) and returns its entries. segBytes is the
// FILE's actual size — geometry is per segment, never the current config.
// sealed=false means "treat as unsealed tail".
func readSeal(f File, segBytes int64, segID, maxBlob uint32) (trailer, []footerEntry, bool) {
	tail, freeTail, err := alignedTemp(recordAlign)
	if err != nil {
		return trailer{}, nil, false
	}
	defer freeTail()
	if err := f.ReadAt(tail, segBytes-recordAlign); err != nil {
		return trailer{}, nil, false
	}
	tr, err := decodeTrailer(tail[recordAlign-trailerSize:], segID)
	if err != nil {
		return trailer{}, nil, false
	}
	tableLen := roundUpAlign(uint64(tr.EntryCount) * footerEntrySz)
	if uint64(tr.FooterOff) != uint64(tr.DataEnd) ||
		uint64(tr.FooterOff)+tableLen+recordAlign > uint64(segBytes) { //nolint:gosec // G115: segBytes validated < 4 GiB
		return trailer{}, nil, false
	}
	var table []byte
	if tableLen > 0 {
		var freeTable func()
		table, freeTable, err = alignedTemp(int(tableLen)) //nolint:gosec // G115: tableLen ≤ segBytes < 4 GiB
		if err != nil {
			return trailer{}, nil, false
		}
		defer freeTable()
		if err := f.ReadAt(table, int64(tr.FooterOff)); err != nil {
			return trailer{}, nil, false
		}
	}
	pre := tail[recordAlign-trailerSize : recordAlign-4]
	if sealCRC(table, pre) != tr.CRC {
		return trailer{}, nil, false
	}
	entries, err := decodeEntries(table, tr.EntryCount)
	if err != nil {
		return trailer{}, nil, false
	}
	// Per-entry sanity even on a CRC-valid footer — belt and braces.
	valid := entries[:0]
	for _, e := range entries {
		if e.Len <= maxBlob && e.Off%recordAlign == 0 &&
			uint64(e.Off)+recordSpan(e.Len) <= uint64(tr.DataEnd) {
			valid = append(valid, e)
		}
	}
	return tr, valid, true
}

// scanTail forward-scans an unsealed segment record by record, verifying
// each payload's xxh3, and stops at the first torn/short/bad record. The
// returned dataEnd is the aligned end of the last GOOD record. segBytes is
// the file's actual size.
func (v *Volume) scanTail(f File, segBytes int64, segID uint32) (recovered []footerEntry, dataEnd uint32, truncated int64) {
	hdrBuf, freeHdr, err := alignedTemp(recordAlign)
	if err != nil {
		v.log.Warn("nvme: tail scan header buffer", "err", err)
		return nil, 0, 0
	}
	defer freeHdr()
	limit := uint64(segBytes) - recordAlign //nolint:gosec // G115: validated ≥ minSize
	var off uint64
	for off+recordAlign <= limit {
		if err := f.ReadAt(hdrBuf, int64(off)); err != nil { //nolint:gosec // G115: off < segBytes
			truncated += recordAlign
			break
		}
		h, err := parseRecordHeader(hdrBuf, v.p.MaxBlobLen)
		if err != nil {
			// Zero magic = clean end of written data (fallocated zeros) —
			// nothing torn. Anything else is a torn header.
			if !allZero(hdrBuf[:recordHdrSize]) {
				truncated += recordAlign
			}
			break
		}
		span := recordSpan(h.Len)
		if off+span > limit {
			truncated += int64(span) //nolint:gosec // G115: span < 4 GiB
			break
		}
		buf, err := v.pool.Get(uint32(span)) //nolint:gosec // G115: span ≤ recordSpan(MaxBlobLen)
		if err != nil {
			v.log.Warn("nvme: tail scan buffer", "err", err)
			break
		}
		chunk := buf[:span]
		if err := f.ReadAt(chunk, int64(off)); err != nil { //nolint:gosec // G115: bounded by limit
			v.pool.Put(buf)
			truncated += int64(span) //nolint:gosec // G115: span < 4 GiB
			break
		}
		payload := chunk[recordHdrSize : recordHdrSize+int(h.Len)]
		sum := xxh3.Hash(payload)
		v.pool.Put(buf)
		if sum != h.XXH3 {
			truncated += int64(span) //nolint:gosec // G115: span < 4 GiB
			break
		}
		recovered = append(recovered, footerEntry{
			NS: h.NS, Key: h.Key, Off: uint32(off), Len: h.Len, XXH3: h.XXH3, //nolint:gosec // G115: off < segBytes < 4 GiB
		})
		off += span
	}
	if len(recovered) > 0 {
		v.log.Info("nvme: tail scan recovered records", "segment", segID, "records", len(recovered))
	}
	return recovered, uint32(off), truncated //nolint:gosec // G115: off < segBytes < 4 GiB
}

func allZero(b []byte) bool {
	for _, x := range b {
		if x != 0 {
			return false
		}
	}
	return true
}

// listSegIDs accepts ONLY canonical segment filenames (seg-%08d.kvbs).
// Sscanf alone also matches seg-0.kvbs / seg-007.kvbs, which would alias a
// canonical ID, double-count used bytes, and leak the first fd (the
// ladder's adversarial-filename MED) — the volume dir's contents are
// corruptible in the threat model.
func listSegIDs(dir string, warn func(msg string, args ...any)) ([]uint32, error) {
	des, err := os.ReadDir(dir)
	if err != nil {
		return nil, fmt.Errorf("nvme: list volume dir: %w", err)
	}
	var ids []uint32
	for _, de := range des {
		name := de.Name()
		if !strings.HasPrefix(name, "seg-") || !strings.HasSuffix(name, ".kvbs") {
			continue
		}
		var id uint32
		if _, err := fmt.Sscanf(name, "seg-%d.kvbs", &id); err != nil ||
			filepath.Base(segPath(dir, id)) != name {
			warn("nvme: ignoring non-canonical segment filename", "name", name)
			continue
		}
		ids = append(ids, id)
	}
	return ids, nil
}

// pruneTmpFiles removes temp files a crash left behind — checkpoint temps
// (crash-during-checkpoint) and segment-adopt temps (crash-mid-restore).
// Neither is matched by its lister, so they'd accumulate forever; both are
// non-durable by construction (the rename is the publish).
func pruneTmpFiles(dir string, warn func(msg string, args ...any)) {
	des, err := os.ReadDir(dir)
	if err != nil {
		return
	}
	for _, de := range des {
		name := de.Name()
		if strings.HasSuffix(name, ".kvbi.tmp") || strings.HasSuffix(name, ".kvbs.tmp") {
			if err := os.Remove(filepath.Join(dir, name)); err != nil {
				warn("nvme: prune orphaned tmp file", "name", name, "err", err)
			}
		}
	}
}
