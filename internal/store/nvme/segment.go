package nvme

import (
	"encoding/binary"
	"fmt"
	"hash/crc32"
)

// On-disk formats (all little-endian, portable, versioned):
//
// Record (4 KiB-aligned within the segment file):
//
//	header (56 B): magic "KVBR" u32 | ns u32 | key [32]byte | len u32 | reserved u32 | xxh3_64 u64
//	payload: len bytes
//	pad: zeros to the next 4 KiB boundary
//
// Sealed-segment footer, written once at seal time and never touched again:
//
//	entry table @footerOff (4 KiB-aligned, right after the last record):
//	    {ns u32, key [32]byte, off u32, len u32, xxh3_64 u64} × entryCount   (52 B each)
//	trailer: the LAST 64 bytes of the file —
//	    magic "KVBF" u32 | version u32 | entryCount u32 | dataEnd u32 |
//	    footerOff u32 | segID u32 | reserved [36]byte | crc32c u32
//	crc32c (Castagnoli, mirroring the wire header's CRC discipline) covers
//	the entry table bytes followed by trailer[0:60].
//
// The dead zone between the entry table and the trailer is never read.
// Sealed-detection = read the file's last 4 KiB and validate the trailer.

const (
	recordAlign   = 4096
	recordHdrSize = 56
	footerEntrySz = 52
	trailerSize   = 64

	recordMagic  = uint32('K') | uint32('V')<<8 | uint32('B')<<16 | uint32('R')<<24
	trailerMagic = uint32('K') | uint32('V')<<8 | uint32('B')<<16 | uint32('F')<<24

	segmentVersion = 1
)

var castagnoli = crc32.MakeTable(crc32.Castagnoli)

// Loc addresses one record: Offset is the byte offset of the record HEADER
// within segment SegmentID; Len is the exact payload byte length.
type Loc struct {
	SegmentID uint32
	Offset    uint32
	Len       uint32
}

// recordSpan is the full aligned footprint of a record with an n-byte
// payload. uint64 arithmetic so a hostile length can't wrap.
func recordSpan(payloadLen uint32) uint64 {
	return roundUpAlign(recordHdrSize + uint64(payloadLen))
}

// MinSegmentBytes is the smallest legal segment size for a given payload
// cap: one max-size record plus the seal footer (entry-table chunk +
// trailer chunk). Config validation and OpenVolume both use THIS — the
// review ladder caught the two computing slightly different minima.
func MinSegmentBytes(maxBlobLen uint32) int64 {
	return int64(recordSpan(maxBlobLen)) + 2*recordAlign + trailerSize //nolint:gosec // G115: span < 4 GiB
}

func roundUpAlign(n uint64) uint64 {
	return (n + recordAlign - 1) &^ (recordAlign - 1)
}

// recordHeader is the parsed 56-byte record header.
type recordHeader struct {
	NS   uint32
	Key  [32]byte
	Len  uint32
	XXH3 uint64
}

// appendRecord frames one record (header + payload + zero pad) onto dst —
// the writer's staging-buffer builder. dst must have the span available;
// the returned slice ends exactly on a 4 KiB boundary.
func appendRecord(dst []byte, h recordHeader, payload []byte) []byte {
	start := len(dst)
	var hdr [recordHdrSize]byte
	binary.LittleEndian.PutUint32(hdr[0:4], recordMagic)
	binary.LittleEndian.PutUint32(hdr[4:8], h.NS)
	copy(hdr[8:40], h.Key[:])
	binary.LittleEndian.PutUint32(hdr[40:44], h.Len)
	// hdr[44:48] reserved, zero.
	binary.LittleEndian.PutUint64(hdr[48:56], h.XXH3)
	dst = append(dst, hdr[:]...)
	dst = append(dst, payload...)
	span := int(recordSpan(h.Len)) //nolint:gosec // G115: span ≤ recordSpan(MaxBlobLen) < 4 GiB by config validation
	for pad := start + span - len(dst); pad > 0; pad-- {
		dst = append(dst, 0)
	}
	return dst
}

// parseRecordHeader decodes and validates a record header chunk. maxLen
// bounds the payload length (the volume's MaxBlobLen) so a torn/garbage
// length can never drive a huge read.
func parseRecordHeader(b []byte, maxLen uint32) (recordHeader, error) {
	if len(b) < recordHdrSize {
		return recordHeader{}, fmt.Errorf("nvme: record header short: %d bytes", len(b))
	}
	if m := binary.LittleEndian.Uint32(b[0:4]); m != recordMagic {
		return recordHeader{}, fmt.Errorf("nvme: record magic %#x", m)
	}
	var h recordHeader
	h.NS = binary.LittleEndian.Uint32(b[4:8])
	copy(h.Key[:], b[8:40])
	h.Len = binary.LittleEndian.Uint32(b[40:44])
	h.XXH3 = binary.LittleEndian.Uint64(b[48:56])
	if h.Len > maxLen {
		return recordHeader{}, fmt.Errorf("nvme: record len %d exceeds max blob %d", h.Len, maxLen)
	}
	return h, nil
}

// footerEntry is one sealed-segment index entry.
type footerEntry struct {
	NS   uint32
	Key  [32]byte
	Off  uint32
	Len  uint32
	XXH3 uint64
}

// encodeEntries serializes the entry table, padded to a 4 KiB multiple
// (the aligned chunk written at footerOff).
func encodeEntries(entries []footerEntry) []byte {
	raw := roundUpAlign(uint64(len(entries)) * footerEntrySz)
	out := make([]byte, raw)
	for i, e := range entries {
		b := out[i*footerEntrySz:]
		binary.LittleEndian.PutUint32(b[0:4], e.NS)
		copy(b[4:36], e.Key[:])
		binary.LittleEndian.PutUint32(b[36:40], e.Off)
		binary.LittleEndian.PutUint32(b[40:44], e.Len)
		binary.LittleEndian.PutUint64(b[44:52], e.XXH3)
	}
	return out
}

// decodeEntries parses count entries from an entry-table chunk.
func decodeEntries(b []byte, count uint32) ([]footerEntry, error) {
	need := uint64(count) * footerEntrySz
	if uint64(len(b)) < need {
		return nil, fmt.Errorf("nvme: entry table short: %d bytes for %d entries", len(b), count)
	}
	entries := make([]footerEntry, count)
	for i := range entries {
		e := &entries[i]
		c := b[i*footerEntrySz:]
		e.NS = binary.LittleEndian.Uint32(c[0:4])
		copy(e.Key[:], c[4:36])
		e.Off = binary.LittleEndian.Uint32(c[36:40])
		e.Len = binary.LittleEndian.Uint32(c[40:44])
		e.XXH3 = binary.LittleEndian.Uint64(c[44:52])
	}
	return entries, nil
}

// trailer is the parsed 64-byte seal trailer.
type trailer struct {
	EntryCount uint32
	DataEnd    uint32
	FooterOff  uint32
	SegID      uint32
	CRC        uint32
}

// encodeTrailer writes the 64-byte trailer; crc must already cover
// entryTable || trailer[0:60] — computed by sealCRC below.
func encodeTrailer(t trailer) [trailerSize]byte {
	var b [trailerSize]byte
	binary.LittleEndian.PutUint32(b[0:4], trailerMagic)
	binary.LittleEndian.PutUint32(b[4:8], segmentVersion)
	binary.LittleEndian.PutUint32(b[8:12], t.EntryCount)
	binary.LittleEndian.PutUint32(b[12:16], t.DataEnd)
	binary.LittleEndian.PutUint32(b[16:20], t.FooterOff)
	binary.LittleEndian.PutUint32(b[20:24], t.SegID)
	// b[24:60] reserved, zero.
	binary.LittleEndian.PutUint32(b[60:64], t.CRC)
	return b
}

// decodeTrailer validates magic/version and returns the fields. It does NOT
// check the CRC (the entry table is needed for that — see sealCRC); callers
// treat any error as "not a sealed segment".
func decodeTrailer(b []byte, wantSegID uint32) (trailer, error) {
	if len(b) < trailerSize {
		return trailer{}, fmt.Errorf("nvme: trailer short: %d bytes", len(b))
	}
	if m := binary.LittleEndian.Uint32(b[0:4]); m != trailerMagic {
		return trailer{}, fmt.Errorf("nvme: trailer magic %#x", m)
	}
	if v := binary.LittleEndian.Uint32(b[4:8]); v != segmentVersion {
		return trailer{}, fmt.Errorf("nvme: trailer version %d", v)
	}
	t := trailer{
		EntryCount: binary.LittleEndian.Uint32(b[8:12]),
		DataEnd:    binary.LittleEndian.Uint32(b[12:16]),
		FooterOff:  binary.LittleEndian.Uint32(b[16:20]),
		SegID:      binary.LittleEndian.Uint32(b[20:24]),
		CRC:        binary.LittleEndian.Uint32(b[60:64]),
	}
	if t.SegID != wantSegID {
		return trailer{}, fmt.Errorf("nvme: trailer segID %d, want %d", t.SegID, wantSegID)
	}
	return t, nil
}

// sealCRC computes the seal checksum: CRC32C over the (padded) entry table
// bytes followed by the first 60 trailer bytes.
func sealCRC(entryTable []byte, trailerPrefix60 []byte) uint32 {
	c := crc32.Update(0, castagnoli, entryTable)
	return crc32.Update(c, castagnoli, trailerPrefix60)
}
