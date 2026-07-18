package nvme

import (
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Checkpoint format ckpt-<seq>.kvbi (buffered I/O — a different file class
// from segments; written tmp → fsync → rename → fsync(dir)):
//
//	magic "KVBC" u32 | version u32 | createdNanos i64 | maxSealedSegID u32 | entryCount u32
//	entries: {ns u32, key [32]byte, segID u32, off u32, len u32, xxh3_64 u64} × N   (56 B)
//	crc32c u32 over everything above
//
// A checkpoint holds ONLY entries of sealed segments — recovery trusts it
// for segments ≤ maxSealedSegID (skipping their footer reads: the
// seconds-per-GB warm-restart win) and footer-scans anything newer.

const (
	ckptMagic     = uint32('K') | uint32('V')<<8 | uint32('B')<<16 | uint32('C')<<24
	ckptVersion   = 1
	ckptHdrSize   = 24
	ckptEntrySize = 56
)

type ckptEntry struct {
	NS    uint32
	Key   [32]byte
	SegID uint32
	Off   uint32
	Len   uint32
	XXH3  uint64
}

func ckptPath(dir string, seq uint64) string {
	return filepath.Join(dir, fmt.Sprintf("ckpt-%08d.kvbi", seq))
}

// writeCheckpoint snapshots every sealed segment's entries and persists
// them atomically. Runs on the writer goroutine; the snapshot is taken
// under RLock but the I/O happens unlocked.
//
// maxSealedSegID is a TRUST boundary: recovery skips footer reads for every
// present segment ≤ it, believing this file holds ALL their entries. So a
// dying segment (mid-retire — excluded from the snapshot, but the retire
// may still ABORT and leave it live) must CAP maxSealed below itself, or a
// crash after an aborted retire silently loses the whole segment (ladder
// blocker B1b, reproduced). If that cap would cover nothing, skip this
// checkpoint round entirely — footer scans stay authoritative.
func (v *Volume) writeCheckpoint() error {
	v.mu.RLock()
	var maxSealed uint32
	var haveSealed, dyingBelow bool
	var dyingMin uint32
	sealed := make([]*segment, 0, len(v.segs))
	total := 0
	for _, s := range v.segs {
		if !s.sealed {
			continue
		}
		if s.dying.Load() {
			if !dyingBelow || s.id < dyingMin {
				dyingMin, dyingBelow = s.id, true
			}
			continue
		}
		sealed = append(sealed, s)
		total += len(s.entries)
		if !haveSealed || s.id > maxSealed {
			maxSealed, haveSealed = s.id, true
		}
	}
	if dyingBelow && dyingMin <= maxSealed {
		if dyingMin == 0 {
			v.mu.RUnlock()
			return nil // nothing coverable this round; retry at the next seal
		}
		maxSealed = dyingMin - 1
	}
	if !haveSealed {
		v.mu.RUnlock()
		return nil
	}
	entries := make([]ckptEntry, 0, total)
	for _, s := range sealed {
		if s.id > maxSealed {
			continue // beyond the trust boundary — recovery footer-scans it anyway
		}
		for _, e := range s.entries {
			entries = append(entries, ckptEntry{NS: e.NS, Key: e.Key, SegID: s.id, Off: e.Off, Len: e.Len, XXH3: e.XXH3})
		}
	}
	seq := v.ckptSeq + 1
	v.mu.RUnlock()

	if err := writeCkptFile(v.p.Dir, seq, v.now(), maxSealed, entries); err != nil {
		return err
	}
	v.mu.Lock()
	v.ckptSeq = seq
	v.mu.Unlock()
	pruneCkpts(v.p.Dir, seq, v.log.Warn)
	return nil
}

func writeCkptFile(dir string, seq uint64, nowNanos int64, maxSealed uint32, entries []ckptEntry) error {
	buf := make([]byte, ckptHdrSize, ckptHdrSize+len(entries)*ckptEntrySize+4)
	binary.LittleEndian.PutUint32(buf[0:4], ckptMagic)
	binary.LittleEndian.PutUint32(buf[4:8], ckptVersion)
	binary.LittleEndian.PutUint64(buf[8:16], uint64(nowNanos)) //nolint:gosec // G115: unix nanos round-trip
	binary.LittleEndian.PutUint32(buf[16:20], maxSealed)
	binary.LittleEndian.PutUint32(buf[20:24], uint32(len(entries))) //nolint:gosec // G115: entry count bounded by volume capacity
	for _, e := range entries {
		var rec [ckptEntrySize]byte
		binary.LittleEndian.PutUint32(rec[0:4], e.NS)
		copy(rec[4:36], e.Key[:])
		binary.LittleEndian.PutUint32(rec[36:40], e.SegID)
		binary.LittleEndian.PutUint32(rec[40:44], e.Off)
		binary.LittleEndian.PutUint32(rec[44:48], e.Len)
		binary.LittleEndian.PutUint64(rec[48:56], e.XXH3)
		buf = append(buf, rec[:]...)
	}
	crc := crc32.Checksum(buf, castagnoli)
	buf = binary.LittleEndian.AppendUint32(buf, crc)

	tmp := ckptPath(dir, seq) + ".tmp"
	f, err := os.OpenFile(tmp, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o600) //nolint:gosec // G304: path is the volume's own dir
	if err != nil {
		return fmt.Errorf("nvme: ckpt tmp create: %w", err)
	}
	if _, err := f.Write(buf); err != nil {
		_ = f.Close()
		_ = os.Remove(tmp)
		return fmt.Errorf("nvme: ckpt write: %w", err)
	}
	if err := f.Sync(); err != nil {
		_ = f.Close()
		_ = os.Remove(tmp)
		return fmt.Errorf("nvme: ckpt fsync: %w", err)
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("nvme: ckpt close: %w", err)
	}
	if err := os.Rename(tmp, ckptPath(dir, seq)); err != nil {
		return fmt.Errorf("nvme: ckpt rename: %w", err)
	}
	return SyncDir(dir)
}

// loadNewestCkpt returns the newest CRC-valid checkpoint (falling back
// older on damage), or ok=false when none is usable.
func loadNewestCkpt(dir string, warn func(msg string, args ...any)) (entries []ckptEntry, maxSealed uint32, seq uint64, ok bool) {
	seqs := listCkptSeqs(dir)
	sort.Slice(seqs, func(i, j int) bool { return seqs[i] > seqs[j] }) // newest first
	for _, s := range seqs {
		e, ms, err := readCkptFile(ckptPath(dir, s))
		if err != nil {
			warn("nvme: checkpoint unusable — falling back", "seq", s, "err", err)
			continue
		}
		return e, ms, s, true
	}
	return nil, 0, 0, false
}

// ckptMaxBytes bounds the checkpoint read — the ONE parse of disk bytes
// that would otherwise allocate before any validation (a hostile multi-GB
// .kvbi would OOM the daemon at startup; ladder finding). 512 MiB covers
// ~9.6M entries — far beyond any real volume's block count.
const ckptMaxBytes = 512 << 20

func readCkptFile(path string) ([]ckptEntry, uint32, error) {
	if st, err := os.Stat(path); err != nil {
		return nil, 0, err
	} else if st.Size() > ckptMaxBytes {
		return nil, 0, fmt.Errorf("checkpoint %d bytes exceeds the %d sanity cap", st.Size(), int64(ckptMaxBytes))
	}
	buf, err := os.ReadFile(path) //nolint:gosec // G304: path is the volume's own dir listing (size-capped above)
	if err != nil {
		return nil, 0, err
	}
	if len(buf) < ckptHdrSize+4 {
		return nil, 0, fmt.Errorf("short file: %d bytes", len(buf))
	}
	body, crcB := buf[:len(buf)-4], buf[len(buf)-4:]
	if crc32.Checksum(body, castagnoli) != binary.LittleEndian.Uint32(crcB) {
		return nil, 0, fmt.Errorf("crc mismatch")
	}
	if m := binary.LittleEndian.Uint32(body[0:4]); m != ckptMagic {
		return nil, 0, fmt.Errorf("magic %#x", m)
	}
	if ver := binary.LittleEndian.Uint32(body[4:8]); ver != ckptVersion {
		return nil, 0, fmt.Errorf("version %d", ver)
	}
	maxSealed := binary.LittleEndian.Uint32(body[16:20])
	count := binary.LittleEndian.Uint32(body[20:24])
	if uint64(len(body)) != ckptHdrSize+uint64(count)*ckptEntrySize {
		return nil, 0, fmt.Errorf("size %d does not match count %d", len(body), count)
	}
	entries := make([]ckptEntry, count)
	for i := range entries {
		rec := body[ckptHdrSize+i*ckptEntrySize:]
		e := &entries[i]
		e.NS = binary.LittleEndian.Uint32(rec[0:4])
		copy(e.Key[:], rec[4:36])
		e.SegID = binary.LittleEndian.Uint32(rec[36:40])
		e.Off = binary.LittleEndian.Uint32(rec[40:44])
		e.Len = binary.LittleEndian.Uint32(rec[44:48])
		e.XXH3 = binary.LittleEndian.Uint64(rec[48:56])
	}
	return entries, maxSealed, nil
}

func listCkptSeqs(dir string) []uint64 {
	des, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}
	var seqs []uint64
	for _, de := range des {
		name := de.Name()
		if !strings.HasPrefix(name, "ckpt-") || !strings.HasSuffix(name, ".kvbi") {
			continue
		}
		var s uint64
		if _, err := fmt.Sscanf(name, "ckpt-%d.kvbi", &s); err == nil {
			seqs = append(seqs, s)
		}
	}
	return seqs
}

func pruneCkpts(dir string, keep uint64, warn func(msg string, args ...any)) {
	for _, s := range listCkptSeqs(dir) {
		if s >= keep {
			continue
		}
		if err := os.Remove(ckptPath(dir, s)); err != nil {
			warn("nvme: prune old checkpoint", "seq", s, "err", err)
		}
	}
}
