package protocol

import (
	"errors"
	"hash/crc32"
	"testing"
)

// wireMagic is the spec's magic bytes, written out longhand so the tests pin
// the wire contract independently of the package's own constant.
var wireMagic = [4]byte{'K', 'V', 'B', '1'}

// sampleHeader returns a fully populated header and its marshalled bytes.
func sampleHeader() (Header, []byte) {
	h := Header{
		Opcode:      OpBatchGet,
		Flags:       FlagResp | FlagMore,
		NamespaceID: 7,
		Credit:      1 << 20,
		RequestID:   0x1002,
		PayloadLen:  3_670_080,
	}
	for i := range h.Key {
		h.Key[i] = byte(i)
	}
	buf := make([]byte, HeaderSize)
	h.MarshalTo(buf)
	return h, buf
}

// TestCRC32CVectors pins the Castagnoli table to the RFC 3720 §B.4 known-answer
// vectors. If any of these fail, the table constant is wrong and nothing else
// in the package can be trusted.
func TestCRC32CVectors(t *testing.T) {
	iscsiRead := []byte{
		0x01, 0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00,
		0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x18,
		0x28, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	}
	ascending := make([]byte, 32)
	descending := make([]byte, 32)
	allFF := make([]byte, 32)
	for i := 0; i < 32; i++ {
		ascending[i] = byte(i)
		descending[i] = byte(31 - i)
		allFF[i] = 0xFF
	}
	tests := []struct {
		name string
		in   []byte
		want uint32
	}{
		{"32 zeros", make([]byte, 32), 0x8A9136AA},
		{"32 ones", allFF, 0x62A8AB43},
		{"ascending", ascending, 0x46DD794E},
		{"descending", descending, 0x113FDB5C},
		{"iSCSI read command", iscsiRead, 0xD9963A56},
	}
	for _, tc := range tests {
		if got := crc32.Checksum(tc.in, castagnoli); got != tc.want {
			t.Errorf("%s: crc32c = %#08x, want %#08x", tc.name, got, tc.want)
		}
	}
}

// TestHeaderCRCCoversFirst60Bytes proves HeaderCRC ignores the CRC field
// itself and nothing else.
func TestHeaderCRCCoversFirst60Bytes(t *testing.T) {
	_, buf := sampleHeader()
	want := crc32.Checksum(buf[:60], castagnoli)
	if got := HeaderCRC(buf); got != want {
		t.Fatalf("HeaderCRC = %#08x, want checksum of bytes 0..59 = %#08x", got, want)
	}
	// Mutating the stored CRC bytes must not change HeaderCRC's result.
	buf[60] ^= 0xFF
	if got := HeaderCRC(buf); got != want {
		t.Fatalf("HeaderCRC changed (%#08x) when only the CRC field was mutated", got)
	}
}

// TestHeaderCRCBoundary pins the exact length contract: 60 bytes is enough
// (the CRC covers exactly bytes 0..59), 59 is not. Kills boundary mutants of
// the length sentinel that a panic-only test cannot distinguish.
func TestHeaderCRCBoundary(t *testing.T) {
	exact := make([]byte, 60, 61) // cap 61: the append below must not reallocate away the test's point
	for i := range exact {
		exact[i] = byte(i)
	}
	want := crc32.Checksum(exact, castagnoli)
	if got := HeaderCRC(exact); got != want {
		t.Fatalf("HeaderCRC on an exactly-60-byte buffer = %#08x, want %#08x", got, want)
	}
	// One byte more must not change the result (byte 60 is outside the CRC).
	if got := HeaderCRC(append(exact, 0xFF)); got != want {
		t.Fatalf("HeaderCRC read past byte 59")
	}
}

func TestHeaderCRCPanicsOnShortBuffer(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("HeaderCRC on a short-LENGTH buffer did not panic")
		}
	}()
	// A 59-length re-slice of a 64-capacity array: without the explicit index
	// check, b[:60] would silently extend to capacity and checksum a byte past
	// the slice's length. The panic must key on len, not cap.
	backing := make([]byte, HeaderSize)
	HeaderCRC(backing[:59])
}

func TestRoundTrip(t *testing.T) {
	want, buf := sampleHeader()

	got, err := ParseHeader(buf, DefaultMaxFrameLen)
	if err != nil {
		t.Fatalf("ParseHeader: %v", err)
	}
	// ParseHeader fills Magic/Version/CRC from the wire; complete `want` with
	// the values MarshalTo computed so a plain struct compare is exact.
	want.Magic = wireMagic
	want.Version = Version1
	want.CRC = HeaderCRC(buf)
	if got != want {
		t.Fatalf("round-trip mismatch:\n got %+v\nwant %+v", got, want)
	}

	// Byte identity: re-marshalling the parsed header reproduces the frame.
	buf2 := make([]byte, HeaderSize)
	got.MarshalTo(buf2)
	if string(buf) != string(buf2) {
		t.Fatalf("re-marshal not byte-identical:\n got %x\nwant %x", buf2, buf)
	}
}

// TestZeroHeaderMarshalsValid: MarshalTo writes magic and version from package
// constants, so even Header{} produces a structurally valid frame.
func TestZeroHeaderMarshalsValid(t *testing.T) {
	var h Header
	buf := make([]byte, HeaderSize)
	h.MarshalTo(buf)
	if _, err := ParseHeader(buf, 0); err != nil {
		t.Fatalf("zero-value header did not parse: %v", err)
	}
}

// TestCorruptEveryOffset flips one byte at every offset of a valid header and
// demands rejection each time: offsets 0..59 are CRC-protected, 60..63 are the
// CRC itself. A single surviving flip would mean an unauthenticated field.
func TestCorruptEveryOffset(t *testing.T) {
	_, buf := sampleHeader()
	for off := 0; off < HeaderSize; off++ {
		corrupt := make([]byte, HeaderSize)
		copy(corrupt, buf)
		corrupt[off] ^= 0xFF
		if _, err := ParseHeader(corrupt, DefaultMaxFrameLen); err == nil {
			t.Errorf("offset %d: corrupted header accepted", off)
		}
	}
}

// TestValidationOrder pins the spec's magic → version → CRC → cap sequence.
func TestValidationOrder(t *testing.T) {
	_, valid := sampleHeader()

	// Bad magic wins over the (also broken) CRC.
	badMagic := append([]byte(nil), valid...)
	badMagic[0] = 'X'
	if _, err := ParseHeader(badMagic, DefaultMaxFrameLen); !errors.Is(err, ErrBadMagic) {
		t.Errorf("bad magic: got %v, want ErrBadMagic", err)
	}

	// Bad version wins over the (also broken) CRC.
	badVersion := append([]byte(nil), valid...)
	badVersion[versionOffset] = 0x02
	if _, err := ParseHeader(badVersion, DefaultMaxFrameLen); !errors.Is(err, ErrBadVersion) {
		t.Errorf("bad version: got %v, want ErrBadVersion", err)
	}

	// CRC wins over the cap: corrupt payload_len is reported as ErrBadCRC,
	// never as a length problem — an unauthenticated length is meaningless.
	badLen := append([]byte(nil), valid...)
	badLen[payloadOffset+3] = 0xFF
	if _, err := ParseHeader(badLen, 16); !errors.Is(err, ErrBadCRC) {
		t.Errorf("corrupt payload_len: got %v, want ErrBadCRC", err)
	}

	// A CRC-valid frame over the caller's cap is the recoverable error.
	if _, err := ParseHeader(valid, 16); !errors.Is(err, ErrPayloadTooLarge) {
		t.Errorf("over-cap: got %v, want ErrPayloadTooLarge", err)
	}

	if _, err := ParseHeader(valid[:HeaderSize-1], DefaultMaxFrameLen); !errors.Is(err, ErrShortHeader) {
		t.Errorf("short buffer: got %v, want ErrShortHeader", err)
	}
}

// TestOverCapReturnsPopulatedHeader pins the recovery contract: on
// ErrPayloadTooLarge the header IS fully decoded (CRC-authenticated before the
// cap check), so the transport can skip PayloadLen bytes and answer
// ERR_TOO_LARGE echoing RequestID without re-parsing raw bytes.
func TestOverCapReturnsPopulatedHeader(t *testing.T) {
	want, buf := sampleHeader()
	h, err := ParseHeader(buf, 16)
	if !errors.Is(err, ErrPayloadTooLarge) {
		t.Fatalf("got %v, want ErrPayloadTooLarge", err)
	}
	if h.PayloadLen != want.PayloadLen || h.RequestID != want.RequestID ||
		h.Opcode != want.Opcode || h.NamespaceID != want.NamespaceID || h.Key != want.Key {
		t.Fatalf("over-cap header not fully populated: %+v", h)
	}
}

// TestFatalClassification: exactly the §1 violations are connection-fatal;
// an over-cap payload is not (the frame is skippable by construction).
func TestFatalClassification(t *testing.T) {
	fatal := []error{ErrShortHeader, ErrBadMagic, ErrBadVersion, ErrBadCRC}
	for _, err := range fatal {
		if !errors.Is(err, ErrFatalFrame) {
			t.Errorf("%v must classify as ErrFatalFrame", err)
		}
	}
	if errors.Is(ErrPayloadTooLarge, ErrFatalFrame) {
		t.Error("ErrPayloadTooLarge must NOT be fatal: the frame is skippable via payload_len")
	}
}

var (
	sinkHeader Header
	sinkErr    error
)

// TestParseHeaderZeroAlloc asserts the codec's hot-path contract on both the
// accept path and every reject path: zero allocations.
func TestParseHeaderZeroAlloc(t *testing.T) {
	_, valid := sampleHeader()
	badCRC := append([]byte(nil), valid...)
	badCRC[keyOffset] ^= 0xFF

	cases := []struct {
		name string
		buf  []byte
		cap  uint32
	}{
		{"accept", valid, DefaultMaxFrameLen},
		{"reject bad crc", badCRC, DefaultMaxFrameLen},
		{"reject over cap", valid, 16},
		{"reject short", valid[:10], DefaultMaxFrameLen},
	}
	for _, tc := range cases {
		allocs := testing.AllocsPerRun(1000, func() {
			sinkHeader, sinkErr = ParseHeader(tc.buf, tc.cap)
		})
		if allocs != 0 {
			t.Errorf("%s: %v allocs/op, want 0", tc.name, allocs)
		}
	}

	buf := make([]byte, HeaderSize)
	h, _ := ParseHeader(valid, DefaultMaxFrameLen)
	allocs := testing.AllocsPerRun(1000, func() {
		h.MarshalTo(buf)
	})
	if allocs != 0 {
		t.Errorf("MarshalTo: %v allocs/op, want 0", allocs)
	}
}

func TestMarshalToPanicsOnShortBuffer(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("MarshalTo on a short buffer did not panic")
		}
	}()
	var h Header
	h.MarshalTo(make([]byte, HeaderSize-1))
}

// TestStatusCodesMatchSpec pins every status constant to its PROTOCOL.md §9
// number and name — a renumbering would silently break every peer.
func TestStatusCodesMatchSpec(t *testing.T) {
	want := []struct {
		s    Status
		code uint8
		name string
	}{
		{StatusOK, 0x00, "OK"},
		{StatusOKExists, 0x01, "OK_EXISTS"},
		{StatusNotFound, 0x10, "NOT_FOUND"},
		{StatusEvicted, 0x11, "EVICTED"},
		{StatusErrAuthRequired, 0x20, "ERR_AUTH_REQUIRED"},
		{StatusErrAuthFailed, 0x21, "ERR_AUTH_FAILED"},
		{StatusErrNamespaceUnknown, 0x22, "ERR_NAMESPACE_UNKNOWN"},
		{StatusErrForbidden, 0x23, "ERR_FORBIDDEN"},
		{StatusErrQuotaBytes, 0x30, "ERR_QUOTA_BYTES"},
		{StatusErrPinQuota, 0x31, "ERR_PIN_QUOTA"},
		{StatusErrTooLarge, 0x32, "ERR_TOO_LARGE"},
		{StatusErrBatchTooLarge, 0x33, "ERR_BATCH_TOO_LARGE"},
		{StatusErrBusy, 0x34, "ERR_BUSY"},
		{StatusErrChecksum, 0x40, "ERR_CHECKSUM"},
		{StatusErrShortStream, 0x41, "ERR_SHORT_STREAM"},
		{StatusErrStaleStream, 0x42, "ERR_STALE_STREAM"},
		{StatusErrImmutableConflict, 0x43, "ERR_IMMUTABLE_CONFLICT"},
		{StatusErrLeased, 0x44, "ERR_LEASED"},
		{StatusErrPinned, 0x45, "ERR_PINNED"},
		{StatusErrUnsupported, 0x50, "ERR_UNSUPPORTED"},
		{StatusErrMalformed, 0x51, "ERR_MALFORMED"},
		{StatusErrInternal, 0x60, "ERR_INTERNAL"},
		{StatusFatalProtocol, 0xF0, "FATAL_PROTOCOL"},
	}
	if len(want) != 23 {
		t.Fatalf("spec defines 23 status codes, table has %d", len(want))
	}
	for _, tc := range want {
		if uint8(tc.s) != tc.code {
			t.Errorf("%s = %#02x, spec says %#02x", tc.name, uint8(tc.s), tc.code)
		}
		if tc.s.String() != tc.name {
			t.Errorf("Status(%#02x).String() = %q, want %q", tc.code, tc.s.String(), tc.name)
		}
	}
	if Status(0xEE).String() != "UNKNOWN_STATUS" {
		t.Error("unknown status must stringify as UNKNOWN_STATUS")
	}
}

// TestOpcodesMatchSpec pins the eight families + NOP to PROTOCOL.md §3.
func TestOpcodesMatchSpec(t *testing.T) {
	want := map[Opcode]uint8{
		OpNop: 0x00, OpHello: 0x01, OpBatchExists: 0x02, OpBatchGet: 0x03,
		OpPutStream: 0x04, OpTouchLease: 0x05, OpPin: 0x06, OpDelete: 0x07,
		OpStats: 0x08,
	}
	for op, code := range want {
		if uint8(op) != code {
			t.Errorf("opcode %#02x, spec says %#02x", uint8(op), code)
		}
	}
}

func TestSubOpRoundTrip(t *testing.T) {
	for s := uint8(0); s <= 15; s++ {
		flags := WithSubOp(FlagResp|FlagFatal, s)
		if got := SubOp(flags); got != s {
			t.Errorf("SubOp(WithSubOp(%d)) = %d", s, got)
		}
		// Sub-op must not disturb the low flag bits.
		if flags&FlagResp == 0 || flags&FlagFatal == 0 {
			t.Errorf("WithSubOp(%d) clobbered flag bits: %#04x", s, flags)
		}
	}
	// Overwriting an existing sub-op replaces it rather than ORing into it.
	if got := SubOp(WithSubOp(WithSubOp(0, PutAbort), PutBegin)); got != PutBegin {
		t.Errorf("sub-op overwrite: got %d, want %d", got, PutBegin)
	}
	// Values above 15 truncate to their low 4 bits (documented: callers pass
	// named constants only). Pinned so the masking never changes silently.
	if got := SubOp(WithSubOp(0, 16)); got != 0 {
		t.Errorf("WithSubOp(16) = sub-op %d, want documented truncation to 0", got)
	}
	// Absolute wire pin: the sub-op occupies flag bits 4-7 (PROTOCOL.md §2),
	// so PUT COMMIT (sub-op 2) alone encodes as 0x0020. Round-trip tests are
	// self-consistent under a shifted field; this is what actually pins it.
	if got := WithSubOp(0, PutCommit); got != 0x0020 {
		t.Errorf("WithSubOp(0, PutCommit) = %#04x, spec says 0x0020 (bits 4-7)", got)
	}
}

// TestWireLayoutGolden pins each field to its byte offset with a hand-checked
// frame — the defense against a refactor silently reordering the struct while
// round-trip tests keep passing.
func TestWireLayoutGolden(t *testing.T) {
	h := Header{
		Opcode:      OpBatchExists,
		Flags:       0x0001,
		NamespaceID: 7,
		Credit:      0x00100000,
		RequestID:   0x1001,
		PayloadLen:  104,
	}
	// A NONZERO key with distinct bytes: an all-zero key lets a key-offset
	// off-by-one shift marshal and parse consistently and pass every
	// round-trip test while breaking interop. Absolute offsets below kill it.
	for i := range h.Key {
		h.Key[i] = 0xA0 + byte(i)
	}
	buf := make([]byte, HeaderSize)
	h.MarshalTo(buf)

	checks := []struct {
		name string
		off  int
		want []byte
	}{
		{"magic", 0, []byte{0x4B, 0x56, 0x42, 0x31}},
		{"version", 4, []byte{0x01}},
		{"opcode", 5, []byte{0x02}},
		{"flags LE", 6, []byte{0x01, 0x00}},
		{"namespace_id LE", 8, []byte{0x07, 0x00, 0x00, 0x00}},
		{"credit LE", 12, []byte{0x00, 0x00, 0x10, 0x00}},
		{"request_id LE", 16, []byte{0x01, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}},
		{"payload_len LE", 56, []byte{0x68, 0x00, 0x00, 0x00}},
	}
	for _, c := range checks {
		got := buf[c.off : c.off+len(c.want)]
		if string(got) != string(c.want) {
			t.Errorf("%s at offset %d: got % x, want % x", c.name, c.off, got, c.want)
		}
	}
	// Key bytes at ABSOLUTE spec offsets 24..55 (not keyOffset, deliberately —
	// this is what pins the layout against a mutated offset constant).
	for i := 0; i < 32; i++ {
		if buf[24+i] != 0xA0+byte(i) {
			t.Errorf("key byte at absolute offset %d: got %#02x, want %#02x", 24+i, buf[24+i], 0xA0+byte(i))
		}
	}
}

// TestLimitsFlagsFeatureBitsMatchSpec pins every remaining wire-visible
// constant to PROTOCOL.md §2/§3.4-3.6/§4/§10 — the mutation run showed these
// were the only unpinned constants left (a typo'd limit would silently change
// negotiation behavior against conforming peers).
func TestLimitsFlagsFeatureBitsMatchSpec(t *testing.T) {
	pins := []struct {
		name string
		got  uint64
		want uint64
	}{
		{"FlagResp", uint64(FlagResp), 0x0001},
		{"FlagMore", uint64(FlagMore), 0x0002},
		{"FlagFatal", uint64(FlagFatal), 0x0004},
		{"FlagForce", uint64(FlagForce), 0x0008},
		{"PutBegin", uint64(PutBegin), 0},
		{"PutChunk", uint64(PutChunk), 1},
		{"PutCommit", uint64(PutCommit), 2},
		{"PutAbort", uint64(PutAbort), 3},
		{"TouchRecency", uint64(TouchRecency), 0},
		{"LeaseGrant", uint64(LeaseGrant), 1},
		{"LeaseRelease", uint64(LeaseRelease), 2},
		{"PinSoft", uint64(PinSoft), 0},
		{"PinHard", uint64(PinHard), 1},
		{"Unpin", uint64(Unpin), 2},
		{"FeatOOO", FeatOOO, 1 << 0},
		{"FeatExistsBitmap", FeatExistsBitmap, 1 << 1},
		{"FeatPayloadCRC32C", FeatPayloadCRC32C, 1 << 2},
		{"FeatCreditSymmetric", FeatCreditSymmetric, 1 << 3},
		{"FeatTLSUpgrade", FeatTLSUpgrade, 1 << 4},
		{"DefaultMaxBatchKeys", DefaultMaxBatchKeys, 512},
		{"FloorMaxBatchKeys", FloorMaxBatchKeys, 128},
		{"DefaultMaxFrameLen", DefaultMaxFrameLen, 268_435_456},
		{"FloorMaxFrameLen", FloorMaxFrameLen, 16_777_216},
		{"DefaultMaxBlobLen", DefaultMaxBlobLen, 33_554_432},
		{"FloorMaxBlobLen", FloorMaxBlobLen, 4_194_304},
		{"DefaultInitialCredit", DefaultInitialCredit, 134_217_728},
		{"FloorInitialCredit", FloorInitialCredit, 16_777_216},
		{"DefaultLeaseMS", DefaultLeaseMS, 5_000},
		{"MaxLeaseMS", MaxLeaseMS, 60_000},
		{"DefaultStreamTimeoutMS", DefaultStreamTimeoutMS, 30_000},
		{"HeaderSize", HeaderSize, 64},
		{"Version1", uint64(Version1), 0x01},
	}
	for _, p := range pins {
		if p.got != p.want {
			t.Errorf("%s = %d, spec says %d", p.name, p.got, p.want)
		}
	}
}
