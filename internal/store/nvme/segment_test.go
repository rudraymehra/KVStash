package nvme

import (
	"bytes"
	"testing"

	"github.com/zeebo/xxh3"
)

func TestRecordSpan(t *testing.T) {
	cases := []struct {
		payload uint32
		want    uint64
	}{
		{0, 4096},                     // header alone still occupies one aligned unit
		{1, 4096},                     // header+1 fits the first unit
		{4040, 4096},                  // 56+4040 = 4096 exactly — no pad
		{4041, 8192},                  // one byte over — next unit
		{4096, 8192},                  //
		{2560 << 10, 2621440 + 4096},  // 2.5 MiB block: 56B header pushes one more unit
		{(32 << 20), 33554432 + 4096}, // MaxBlobLen default
		{^uint32(0), 4294971392},      // hostile max: must not wrap (uint64 math)
	}
	for _, c := range cases {
		if got := recordSpan(c.payload); got != c.want {
			t.Fatalf("recordSpan(%d) = %d, want %d", c.payload, got, c.want)
		}
		if got := recordSpan(c.payload); got%recordAlign != 0 {
			t.Fatalf("recordSpan(%d) = %d not aligned", c.payload, got)
		}
	}
}

func TestRecordRoundTrip(t *testing.T) {
	payloads := [][]byte{
		nil, // empty block is legal (GET desc status=OK len=0)
		[]byte("x"),
		bytes.Repeat([]byte{0xA5}, 4040), // exact-fit boundary
		bytes.Repeat([]byte{0x5A}, 400<<10),
	}
	for i, p := range payloads {
		h := recordHeader{NS: uint32(i + 1), Len: uint32(len(p)), XXH3: xxh3.Hash(p)} //nolint:gosec // G115: test payloads are small
		h.Key[0], h.Key[31] = byte(i), 0xEE
		buf := appendRecord(nil, h, p)
		if uint64(len(buf)) != recordSpan(h.Len) {
			t.Fatalf("case %d: framed %d bytes, want span %d", i, len(buf), recordSpan(h.Len))
		}
		got, err := parseRecordHeader(buf, 32<<20)
		if err != nil {
			t.Fatalf("case %d: parse: %v", i, err)
		}
		if got != h {
			t.Fatalf("case %d: header round-trip: got %+v want %+v", i, got, h)
		}
		if !bytes.Equal(buf[recordHdrSize:recordHdrSize+len(p)], p) {
			t.Fatalf("case %d: payload corrupted in framing", i)
		}
		for _, b := range buf[recordHdrSize+len(p):] {
			if b != 0 {
				t.Fatalf("case %d: pad not zeroed", i)
			}
		}
	}
}

func TestParseRecordHeaderRejects(t *testing.T) {
	good := appendRecord(nil, recordHeader{NS: 1, Len: 3, XXH3: 7}, []byte("abc"))

	// Short chunk.
	if _, err := parseRecordHeader(good[:recordHdrSize-1], 32<<20); err == nil {
		t.Fatal("short header accepted")
	}
	// Bad magic.
	bad := append([]byte(nil), good...)
	bad[0] ^= 0xFF
	if _, err := parseRecordHeader(bad, 32<<20); err == nil {
		t.Fatal("bad magic accepted")
	}
	// Hostile length: over maxLen must be rejected BEFORE any read sizing.
	bad = append([]byte(nil), good...)
	bad[40], bad[41], bad[42], bad[43] = 0xFF, 0xFF, 0xFF, 0xFF
	if _, err := parseRecordHeader(bad, 32<<20); err == nil {
		t.Fatal("hostile length accepted")
	}
	// A volume with a smaller negotiated MaxBlobLen rejects lengths that a
	// default-sized volume would accept.
	over := appendRecord(nil, recordHeader{NS: 1, Len: 2 << 20, XXH3: 1}, make([]byte, 2<<20))
	if _, err := parseRecordHeader(over, 1<<20); err == nil {
		t.Fatal("len above the volume's max accepted")
	}
	if _, err := parseRecordHeader(over, 32<<20); err != nil {
		t.Fatalf("len within the volume's max rejected: %v", err)
	}
}

func TestFooterRoundTrip(t *testing.T) {
	entries := make([]footerEntry, 100)
	for i := range entries {
		e := &entries[i]
		e.NS = uint32(i%3 + 1) //nolint:gosec // G115: tiny test values
		e.Key[0] = byte(i)
		e.Off = uint32(i) * 8192 //nolint:gosec // G115: tiny test values
		e.Len = uint32(i * 100)  //nolint:gosec // G115: tiny test values
		e.XXH3 = uint64(i) * 0x9E3779B97F4A7C15
	}
	table := encodeEntries(entries)
	if len(table)%recordAlign != 0 {
		t.Fatalf("entry table not aligned: %d", len(table))
	}

	tr := trailer{EntryCount: 100, DataEnd: 819200, FooterOff: 819200, SegID: 7}
	pre := encodeTrailer(tr)
	tr.CRC = sealCRC(table, pre[:60])
	full := encodeTrailer(tr)

	got, err := decodeTrailer(full[:], 7)
	if err != nil {
		t.Fatalf("decodeTrailer: %v", err)
	}
	if got != tr {
		t.Fatalf("trailer round-trip: got %+v want %+v", got, tr)
	}
	if want := sealCRC(table, full[:60]); got.CRC != want {
		t.Fatalf("CRC mismatch after round-trip: %#x vs %#x", got.CRC, want)
	}

	back, err := decodeEntries(table, got.EntryCount)
	if err != nil {
		t.Fatalf("decodeEntries: %v", err)
	}
	for i := range entries {
		if back[i] != entries[i] {
			t.Fatalf("entry %d round-trip: got %+v want %+v", i, back[i], entries[i])
		}
	}
}

func TestTrailerRejects(t *testing.T) {
	tr := trailer{EntryCount: 1, DataEnd: 4096, FooterOff: 4096, SegID: 3}
	pre := encodeTrailer(tr)
	tr.CRC = sealCRC(nil, pre[:60])
	full := encodeTrailer(tr)

	// Magic flip.
	bad := full
	bad[0] ^= 0x01
	if _, err := decodeTrailer(bad[:], 3); err == nil {
		t.Fatal("bad magic accepted")
	}
	// Version bump.
	bad = full
	bad[4] = 99
	if _, err := decodeTrailer(bad[:], 3); err == nil {
		t.Fatal("future version accepted")
	}
	// Wrong segment identity (a copied/renamed file must not seal-validate).
	if _, err := decodeTrailer(full[:], 4); err == nil {
		t.Fatal("segID mismatch accepted")
	}
	// Short buffer.
	if _, err := decodeTrailer(full[:trailerSize-1], 3); err == nil {
		t.Fatal("short trailer accepted")
	}
	// CRC flip is the caller's job (needs the table) — assert the helper sees it.
	flipped := full
	flipped[60] ^= 0xFF
	got, err := decodeTrailer(flipped[:], 3)
	if err != nil {
		t.Fatalf("decodeTrailer on crc-flip should parse fields: %v", err)
	}
	if got.CRC == tr.CRC {
		t.Fatal("crc flip not visible")
	}
	if want := sealCRC(nil, flipped[:60]); got.CRC == want {
		t.Fatal("flipped CRC still verifies — checksum is vacuous")
	}
}

func TestDecodeEntriesShort(t *testing.T) {
	table := encodeEntries(make([]footerEntry, 2))
	if _, err := decodeEntries(table[:footerEntrySz], 2); err == nil {
		t.Fatal("short entry table accepted")
	}
}
