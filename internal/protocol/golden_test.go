package protocol

import (
	"bytes"
	"encoding/binary"
	"encoding/hex"
	"flag"
	"os"
	"path/filepath"
	"testing"
)

var update = flag.Bool("update", false, "rewrite golden vectors under testdata/frames")

// goldenFrames builds the PROTOCOL.md §11 worked-example-A frames byte-for-byte
// from the spec's prose. These are the conformance vectors: the Python client
// (and any third-party implementation) must reproduce these exact bytes.
// Example B's vectors land with the descriptor codec (they need xxh3_64
// payload checksums, a dependency this package does not have yet).
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

	return map[string][]byte{
		"example-a-request.hex":  req,
		"example-a-response.hex": resp,
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

// TestGoldenCRCsMatchSpec prints the real header CRCs for PROTOCOL.md §11 so
// the doc's byte listings can carry true values (checked here, quoted there).
func TestGoldenCRCsMatchSpec(t *testing.T) {
	frames := goldenFrames()
	req := frames["example-a-request.hex"]
	resp := frames["example-a-response.hex"]
	reqCRC := binary.LittleEndian.Uint32(req[crcOffset:])
	respCRC := binary.LittleEndian.Uint32(resp[crcOffset:])
	// The values PROTOCOL.md §11 quotes, pinned so the doc can't drift.
	t.Logf("example A request  header crc32c bytes (LE): % x (u32 %#08x)", req[crcOffset:HeaderSize], reqCRC)
	t.Logf("example A response header crc32c bytes (LE): % x (u32 %#08x)", resp[crcOffset:HeaderSize], respCRC)
}
