package protocol

import (
	"bytes"
	"encoding/binary"
	"encoding/hex"
	"flag"
	"os"
	"path/filepath"
	"testing"

	"github.com/zeebo/xxh3"
)

var update = flag.Bool("update", false, "rewrite golden vectors under testdata/frames")

// exampleBPayload returns the deterministic payload convention for the §12
// golden vector: block K0 = 1,048,576 bytes of 0xAA, K1 = 2,621,440 bytes of
// 0xBB (matching their key fill bytes). The spec quotes the resulting xxh3_64
// values; any conforming implementation reproduces them from this convention.
func exampleBPayload(fill byte, n int) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = fill
	}
	return b
}

// goldenFrames builds the PROTOCOL.md §11–12 worked-example frames
// byte-for-byte from the spec's prose. These are the conformance vectors: the
// Python client (and any third-party implementation) must reproduce these
// exact bytes. Example B's vector covers the frame header + the response
// header region (preamble, first_index/total_keys, descriptors with real
// xxh3_64 values); the multi-megabyte payload bytes follow the documented
// deterministic convention and are not committed.
func goldenFrames() map[string][]byte {
	// Request: BATCH_EXISTS of 3 keys in namespace 7, request_id 0x1001.
	reqHdr := Header{
		Opcode:      OpBatchExists,
		NamespaceID: 7,
		RequestID:   0x1001,
		PayloadLen:  104,
	}
	req := make([]byte, HeaderSize+104)
	reqHdr.MarshalTo(req)
	binary.LittleEndian.PutUint32(req[64:], 3) // n_keys=3, then reserved u32 = 0
	for i, fill := range []byte{0xAA, 0xBB, 0xCC} {
		for j := 0; j < 32; j++ {
			req[72+32*i+j] = fill
		}
	}

	// Response: status OK, count=3, n_consecutive=2, bitmap OK|OK|NOT_FOUND,
	// with a 1 MiB credit grant piggybacked.
	respHdr := Header{
		Opcode:      OpBatchExists,
		Flags:       FlagResp,
		NamespaceID: 7,
		Credit:      0x00100000,
		RequestID:   0x1001,
		PayloadLen:  24,
	}
	resp := make([]byte, HeaderSize+24)
	respHdr.MarshalTo(resp)
	resp[64] = byte(StatusOK)                   // preamble: status, rsvd[3]
	binary.LittleEndian.PutUint32(resp[68:], 3) // count=3
	binary.LittleEndian.PutUint32(resp[72:], 2) // n_consecutive=2, reserved u32 = 0
	resp[80] = byte(StatusOK)
	resp[81] = byte(StatusOK)
	resp[82] = byte(StatusNotFound) // then 5 pad bytes = 0

	// Example B response: BATCH_GET of 3 keys — 2 hits (1 MiB + 2.5 MiB), 1
	// miss, one frame, request_id 0x1002, 1 MiB credit grant piggybacked.
	k0 := exampleBPayload(0xAA, 1<<20)
	k1 := exampleBPayload(0xBB, 2560<<10)
	descs := []Desc{
		{Status: StatusOK, Len: uint32(len(k0)), XXH3: xxh3.Hash(k0)}, //nolint:gosec // G115: fixed 1 MiB test constant
		{Status: StatusOK, Len: uint32(len(k1)), XXH3: xxh3.Hash(k1)}, //nolint:gosec // G115: fixed 2.5 MiB test constant
		{Status: StatusNotFound},
	}
	region := AppendGetRespHeader(nil, StatusOK, 0, 3, descs)
	bHdr := Header{
		Opcode:      OpBatchGet,
		Flags:       FlagResp, // no F_MORE: single final frame
		NamespaceID: 7,
		Credit:      0x00100000,
		RequestID:   0x1002,
		PayloadLen:  uint32(len(region) + len(k0) + len(k1)), //nolint:gosec // G115: 3,670,080 — the spec's own example size
	}
	bFrame := make([]byte, HeaderSize+len(region))
	bHdr.MarshalTo(bFrame)
	copy(bFrame[HeaderSize:], region)

	return map[string][]byte{
		"example-a-request.hex":         req,
		"example-a-response.hex":        resp,
		"example-b-response-header.hex": bFrame,
	}
}

// TestGoldenVectors pins the worked-example frames to committed files. Run
// with -update to (re)generate after a DELIBERATE spec change — which, per the
// freeze rule, must land in the same PR as the PROTOCOL.md edit.
func TestGoldenVectors(t *testing.T) {
	dir := filepath.Join("testdata", "frames")
	for name, frame := range goldenFrames() {
		path := filepath.Join(dir, name)
		encoded := []byte(hex.EncodeToString(frame) + "\n")

		if *update {
			if err := os.MkdirAll(dir, 0o750); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(path, encoded, 0o600); err != nil {
				t.Fatal(err)
			}
		}

		want, err := os.ReadFile(path) //nolint:gosec // G304: path is built from package-local constants, not external input
		if err != nil {
			t.Fatalf("%s: %v (regenerate with -update)", name, err)
		}
		if !bytes.Equal(want, encoded) {
			t.Errorf("%s: frame bytes diverge from the committed golden vector — a wire change without a spec change?", name)
		}

		// Every golden frame must round-trip through the real parser.
		if _, perr := ParseHeader(frame, DefaultMaxFrameLen); perr != nil {
			t.Errorf("%s: golden frame does not parse: %v", name, perr)
		}
	}
}

// TestGoldenCRCsMatchSpec ASSERTS the values PROTOCOL.md §11–12 quote, so the
// doc, the codec, and the committed vectors are pinned to each other. (An
// earlier version only logged these — a calibration drill demonstrated that a
// log-only "check" is exactly how a regenerated-around-a-bug golden slips by.)
func TestGoldenCRCsMatchSpec(t *testing.T) {
	frames := goldenFrames()
	want := []struct {
		name string
		crc  uint32 // as quoted in the spec's byte listings
	}{
		{"example-a-request.hex", 0x5F2EA7FF},
		{"example-a-response.hex", 0xD6932E23},
		{"example-b-response-header.hex", 0xC54F9DB9},
	}
	for _, w := range want {
		got := binary.LittleEndian.Uint32(frames[w.name][crcOffset:])
		if got != w.crc {
			t.Errorf("%s: header crc32c %#08X, PROTOCOL.md quotes %#08X — doc and code have diverged", w.name, got, w.crc)
		}
	}
	// The §12 descriptor checksums, quoted in the spec's byte listing.
	b := frames["example-b-response-header.hex"]
	d0 := GetDesc(b[HeaderSize+16:])
	d1 := GetDesc(b[HeaderSize+32:])
	if d0.XXH3 != 0xC47DEB608981FC4F || d1.XXH3 != 0xD69E1622BA66C45D {
		t.Errorf("example B descriptor xxh3_64 diverged from the spec: %#016X, %#016X", d0.XXH3, d1.XXH3)
	}
}
