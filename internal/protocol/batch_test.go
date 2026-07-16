package protocol

import (
	"bytes"
	"errors"
	"testing"
)

func testKeys(n int) [][32]byte {
	keys := make([][32]byte, n)
	for i := range keys {
		for j := range keys[i] {
			keys[i][j] = byte((i*31 + j) % 256) //nolint:gosec // %256 always fits a byte
		}
	}
	return keys
}

func TestPreambleRoundTrip(t *testing.T) {
	b := AppendPreamble(nil, StatusErrBusy, 7)
	if len(b) != PreambleSize {
		t.Fatalf("preamble length %d, want %d", len(b), PreambleSize)
	}
	p, err := DecodePreamble(b)
	if err != nil || p.Status != StatusErrBusy || p.Count != 7 {
		t.Fatalf("round-trip: %+v, %v", p, err)
	}
	if _, err := DecodePreamble(b[:7]); !errors.Is(err, ErrBadBody) {
		t.Fatalf("short preamble: got %v", err)
	}
}

func TestAppendErrorRespIsPreambleOnly(t *testing.T) {
	b := AppendErrorResp(nil, StatusErrMalformed)
	if len(b) != PreambleSize {
		t.Fatalf("error response must be exactly the preamble (§3), got %d bytes", len(b))
	}
	p, _ := DecodePreamble(b)
	if p.Status != StatusErrMalformed || p.Count != 0 {
		t.Fatalf("error response preamble: %+v", p)
	}
}

func TestKeyListRoundTrip(t *testing.T) {
	for _, n := range []int{0, 1, 3, 128, 512} {
		keys := testKeys(n)
		body := AppendKeyList(nil, 0xDEAD, keys)
		aux, got, err := DecodeKeyList(body, 512, nil)
		if err != nil {
			t.Fatalf("n=%d: %v", n, err)
		}
		if aux != 0xDEAD || len(got) != n {
			t.Fatalf("n=%d: aux=%#x len=%d", n, aux, len(got))
		}
		for i := range got {
			if got[i] != keys[i] {
				t.Fatalf("n=%d: key %d mismatch", n, i)
			}
		}
	}
}

// TestKeyListCapBeforeAlloc pins the load-bearing order: n_keys over the cap
// is rejected before anything is appended, with zero allocations.
func TestKeyListCapBeforeAlloc(t *testing.T) {
	body := AppendKeyList(nil, 0, testKeys(9))
	dst := make([][32]byte, 0, 16)

	_, got, err := DecodeKeyList(body, 8, dst)
	if !errors.Is(err, ErrKeyCount) {
		t.Fatalf("over-cap: got %v, want ErrKeyCount", err)
	}
	if len(got) != 0 {
		t.Fatalf("over-cap decode appended %d keys; must append nothing", len(got))
	}
	allocs := testing.AllocsPerRun(1000, func() {
		_, _, sinkErr = DecodeKeyList(body, 8, dst)
	})
	if allocs != 0 {
		t.Fatalf("over-cap reject path: %v allocs, want 0", allocs)
	}
	if ErrorStatus(err) != StatusErrBatchTooLarge {
		t.Fatalf("ErrorStatus(ErrKeyCount) = %v, want ERR_BATCH_TOO_LARGE", ErrorStatus(err))
	}
}

func TestKeyListExactLength(t *testing.T) {
	body := AppendKeyList(nil, 0, testKeys(3))
	for _, mutate := range []func([]byte) []byte{
		func(b []byte) []byte { return b[:len(b)-1] },
		func(b []byte) []byte { return append(b, 0) },
		func(b []byte) []byte { return b[:7] },
	} {
		if _, _, err := DecodeKeyList(mutate(append([]byte(nil), body...)), 512, nil); !errors.Is(err, ErrBodyLength) {
			t.Fatalf("length-mutated body accepted")
		}
	}
}

// TestKeyListCopiesNotAliases pins the ownership contract: keys survive the
// body buffer being recycled/poisoned.
func TestKeyListCopiesNotAliases(t *testing.T) {
	body := AppendKeyList(nil, 0, testKeys(2))
	_, keys, err := DecodeKeyList(body, 512, nil)
	if err != nil {
		t.Fatal(err)
	}
	want := keys[0]
	for i := range body {
		body[i] = 0xDE // simulate the transport poisoning the returned buffer
	}
	if keys[0] != want {
		t.Fatal("decoded key aliased the body buffer: poisoned bytes leaked into the key")
	}
}

func TestExistsRespRoundTrip(t *testing.T) {
	perKey := []Status{StatusOK, StatusOK, StatusNotFound}

	withBitmap := AppendExistsResp(nil, 3, 2, perKey)
	if len(withBitmap)%8 != 0 {
		t.Fatalf("bitmap response not 8-aligned: %d", len(withBitmap))
	}
	r, err := DecodeExistsResp(withBitmap, true)
	if err != nil || r.Count != 3 || r.NConsecutive != 2 || len(r.PerKey) != 3 {
		t.Fatalf("bitmap round-trip: %+v, %v", r, err)
	}
	if Status(r.PerKey[2]) != StatusNotFound {
		t.Fatalf("per-key status: %v", r.PerKey)
	}

	noBitmap := AppendExistsResp(nil, 3, 2, nil)
	if len(noBitmap) != PreambleSize+8 {
		t.Fatalf("no-bitmap length %d", len(noBitmap))
	}
	r, err = DecodeExistsResp(noBitmap, false)
	if err != nil || r.PerKey != nil {
		t.Fatalf("no-bitmap round-trip: %+v, %v", r, err)
	}

	// Mismatched expectation = length error, both directions.
	if _, err := DecodeExistsResp(withBitmap, false); !errors.Is(err, ErrBodyLength) {
		t.Fatal("bitmap body accepted without bitmap expectation")
	}
	if _, err := DecodeExistsResp(noBitmap, true); !errors.Is(err, ErrBodyLength) {
		t.Fatal("no-bitmap body accepted with bitmap expectation")
	}

	// Non-OK is preamble-only (§3).
	errResp := AppendErrorResp(nil, StatusErrBatchTooLarge)
	r, err = DecodeExistsResp(errResp, true)
	if err != nil || r.Status != StatusErrBatchTooLarge {
		t.Fatalf("non-OK: %+v, %v", r, err)
	}
	if _, err := DecodeExistsResp(append(errResp, 0, 0, 0, 0, 0, 0, 0, 0), true); !errors.Is(err, ErrBodyLength) {
		t.Fatal("non-OK with trailing bytes accepted")
	}
}

func TestKeyStatusRespPadEdges(t *testing.T) {
	for _, n := range []int{0, 1, 7, 8, 9, 16, 17} {
		perKey := make([]Status, n)
		for i := range perKey {
			perKey[i] = StatusErrLeased
		}
		body := AppendKeyStatusResp(nil, perKey)
		if len(body) != PreambleSize+padTo8(n) {
			t.Fatalf("n=%d: length %d, want %d", n, len(body), PreambleSize+padTo8(n))
		}
		p, got, err := DecodeKeyStatusResp(body)
		if err != nil || int(p.Count) != n || len(got) != n {
			t.Fatalf("n=%d: %+v len=%d err=%v", n, p, len(got), err)
		}
		if _, _, err := DecodeKeyStatusResp(body[:len(body)-1]); !errors.Is(err, ErrBodyLength) {
			t.Fatalf("n=%d: truncated body accepted", n)
		}
	}
}

// TestKeyStatusNonOKStrict pins the §3 rule for the per-key-status verbs: a
// non-OK response is EXACTLY the preamble — a nonconforming frame carrying a
// count and per-key bytes on an error status is malformed, not honored.
func TestKeyStatusNonOKStrict(t *testing.T) {
	bad := AppendPreamble(nil, StatusErrQuotaBytes, 13)
	bad = append(bad, make([]byte, 16)...)
	if _, _, err := DecodeKeyStatusResp(bad); !errors.Is(err, ErrBodyLength) {
		t.Fatal("non-OK response with per-key bytes accepted")
	}
	// And the well-formed preamble-only error decodes with a nil status slice.
	p, st, err := DecodeKeyStatusResp(AppendErrorResp(nil, StatusErrQuotaBytes))
	if err != nil || st != nil || p.Status != StatusErrQuotaBytes {
		t.Fatalf("preamble-only non-OK: %+v, %v, %v", p, st, err)
	}
}

func TestDescRoundTrip(t *testing.T) {
	d := Desc{Status: StatusOK, Len: 1 << 20, XXH3: 0x0123456789ABCDEF}
	var b [DescSize]byte
	PutDesc(b[:], d)
	if got := GetDesc(b[:]); got != d {
		t.Fatalf("desc round-trip: %+v", got)
	}
	if b[1] != 0 || b[2] != 0 || b[3] != 0 {
		t.Fatal("desc reserved bytes not zeroed")
	}
	defer func() {
		if recover() == nil {
			t.Fatal("PutDesc on a short buffer did not panic")
		}
	}()
	PutDesc(b[:DescSize-1], d)
}

func TestGetRespHeaderRoundTrip(t *testing.T) {
	descs := []Desc{
		{Status: StatusOK, Len: 1 << 20, XXH3: 1},
		{Status: StatusOK, Len: 2560 << 10, XXH3: 2},
		{Status: StatusNotFound},
	}
	// first_index=2, total_keys=5: a continuation frame, so TotalKeys differs
	// from the descriptor count (a mutant aliasing the two fields must die).
	body := AppendGetRespHeader(nil, StatusOK, 2, 5, descs)
	if len(body) != GetRespHeaderSize(3) {
		t.Fatalf("header region length %d, want %d", len(body), GetRespHeaderSize(3))
	}

	// Exact-length decode (final frame shape: region only, no trailing bytes).
	if g, err := DecodeGetRespHeader(body); err != nil || g.FirstIndex != 2 || g.TotalKeys != 5 {
		t.Fatalf("exact-length decode: %+v, %v", g, err)
	}

	// Trailing payload bytes after the region are legal (§3.3).
	withPayload := append(append([]byte(nil), body...), 0xAA, 0xBB)
	g, err := DecodeGetRespHeader(withPayload)
	if err != nil || g.Count != 3 || g.FirstIndex != 2 || g.TotalKeys != 5 {
		t.Fatalf("decode: %+v, %v", g, err)
	}
	for i, want := range descs {
		if g.Desc(i) != want {
			t.Fatalf("desc %d: %+v", i, g.Desc(i))
		}
	}
	if g.PayloadOffset() != GetRespHeaderSize(3) {
		t.Fatalf("payload offset %d", g.PayloadOffset())
	}

	if _, err := DecodeGetRespHeader(body[:len(body)-1]); !errors.Is(err, ErrBodyLength) {
		t.Fatal("truncated descriptor region accepted")
	}
	// Non-OK is preamble-only.
	if g, err := DecodeGetRespHeader(AppendErrorResp(nil, StatusErrBusy)); err != nil || g.Status != StatusErrBusy {
		t.Fatalf("non-OK: %+v, %v", g, err)
	}
}

func TestHelloReqRoundTrip(t *testing.T) {
	cases := []HelloReq{
		{
			ProtoMin: 1, ProtoMax: 1, Features: ServerFeatures,
			MaxBatchKeys: 256, MaxFrameLen: 32 << 20,
			Token: []byte("secret-token"), Namespace: "tenant-a", ClientName: "vllm-node-7",
		},
		{ProtoMin: 1, ProtoMax: 2},                               // all strings empty
		{Token: bytes.Repeat([]byte{0x55}, 300), Namespace: "n"}, // uneven pad
	}
	for i, want := range cases {
		body := AppendHelloReq(nil, want)
		if len(body)%8 != 0 {
			t.Fatalf("case %d: HELLO request body not 8-aligned: %d", i, len(body))
		}
		got, err := DecodeHelloReq(body)
		if err != nil {
			t.Fatalf("case %d: %v", i, err)
		}
		if got.ProtoMin != want.ProtoMin || got.ProtoMax != want.ProtoMax ||
			got.Features != want.Features || got.MaxBatchKeys != want.MaxBatchKeys ||
			got.MaxFrameLen != want.MaxFrameLen || !bytes.Equal(got.Token, want.Token) ||
			got.Namespace != want.Namespace || got.ClientName != want.ClientName {
			t.Fatalf("case %d round-trip:\n got %+v\nwant %+v", i, got, want)
		}
		if _, err := DecodeHelloReq(body[:len(body)-1]); !errors.Is(err, ErrBodyLength) {
			t.Fatalf("case %d: truncated body accepted", i)
		}
		if _, err := DecodeHelloReq(append(append([]byte(nil), body...), make([]byte, 8)...)); !errors.Is(err, ErrBodyLength) {
			t.Fatalf("case %d: trailing bytes accepted", i)
		}
	}
	// Copy contract: token must survive body poisoning (auth data vs recycled buffer).
	body := AppendHelloReq(nil, cases[0])
	got, _ := DecodeHelloReq(body)
	for i := range body {
		body[i] = 0xDE
	}
	if !bytes.Equal(got.Token, []byte("secret-token")) || got.Namespace != "tenant-a" {
		t.Fatal("HELLO decode aliased the body buffer")
	}

	// Append into a NON-EMPTY dst: the §0 pad must be computed from the body's
	// own length, not the scratch buffer's absolute length (a start-offset sign
	// mutant is invisible when dst is always empty).
	prefix := []byte{1, 2, 3} // deliberately not 8-aligned
	full := AppendHelloReq(prefix, cases[0])
	if got2, err := DecodeHelloReq(full[len(prefix):]); err != nil ||
		got2.Namespace != "tenant-a" || !bytes.Equal(got2.Token, []byte("secret-token")) {
		t.Fatalf("prefix-append round-trip: %+v, %v", got2, err)
	}
}

func TestHelloRespRoundTrip(t *testing.T) {
	want := HelloResp{
		Proto:    1,
		Features: FeatOOO,
		Limits: Limits{
			MaxBatchKeys: 512, MaxFrameLen: 256 << 20,
			MaxBlobLen: 32 << 20, InitialCredit: 128 << 20,
		},
		NamespaceID:     7,
		LeaseDefaultMS:  5000,
		LeaseMaxMS:      60000,
		StreamTimeoutMS: 30000,
		ServerName:      "kvblockd/0.0.2",
	}
	body := AppendHelloResp(nil, want)
	if len(body)%8 != 0 {
		t.Fatalf("HELLO response body not 8-aligned: %d", len(body))
	}
	p, got, err := DecodeHelloResp(body)
	if err != nil || p.Status != StatusOK {
		t.Fatalf("decode: %+v, %v", p, err)
	}
	if got != want {
		t.Fatalf("round-trip:\n got %+v\nwant %+v", got, want)
	}

	// Non-OK HELLO is preamble-only.
	p, _, err = DecodeHelloResp(AppendErrorResp(nil, StatusErrAuthFailed))
	if err != nil || p.Status != StatusErrAuthFailed {
		t.Fatalf("non-OK: %+v, %v", p, err)
	}
	if _, _, err := DecodeHelloResp(body[:len(body)-1]); !errors.Is(err, ErrBodyLength) {
		t.Fatal("truncated response accepted")
	}

	// Empty server name: the body is EXACTLY the fixed prefix (56 bytes, already
	// 8-aligned) — the minimum legal OK response, and the boundary a length
	// off-by-one would break.
	minimal := AppendHelloResp(nil, HelloResp{Proto: 1})
	if len(minimal) != helloRespFixed {
		t.Fatalf("empty-name response length %d, want %d", len(minimal), helloRespFixed)
	}
	if p, got, err := DecodeHelloResp(minimal); err != nil || p.Status != StatusOK || got.ServerName != "" {
		t.Fatalf("empty-name decode: %+v, %v", got, err)
	}

	// Append into a non-empty, non-aligned dst (pad computed from body length).
	prefix := []byte{9, 9, 9}
	full := AppendHelloResp(prefix, want)
	if _, got, err := DecodeHelloResp(full[len(prefix):]); err != nil || got != want {
		t.Fatalf("prefix-append round-trip: %+v, %v", got, err)
	}
}

func TestPutBodiesRoundTrip(t *testing.T) {
	want := PutBeginBody{TotalLen: 2560 << 10, TTLms: 60000, XXH3Hint: 0xFEED, Flags: 0}
	body := AppendPutBegin(nil, want)
	if len(body) != putBeginSize {
		t.Fatalf("BEGIN body length %d", len(body))
	}
	got, err := DecodePutBegin(body)
	if err != nil || got != want {
		t.Fatalf("BEGIN round-trip: %+v, %v", got, err)
	}
	// Nonzero flags are returned, never rejected (§3.4: ignored on receive).
	body[16] = 0xFF
	if got, err := DecodePutBegin(body); err != nil || got.Flags != 0xFF {
		t.Fatalf("nonzero BEGIN flags: %+v, %v", got, err)
	}
	if _, err := DecodePutBegin(body[:23]); !errors.Is(err, ErrBodyLength) {
		t.Fatal("short BEGIN accepted")
	}

	cb := AppendPutCommit(nil, 0xABCDEF)
	if x, err := DecodePutCommit(cb); err != nil || x != 0xABCDEF {
		t.Fatalf("COMMIT round-trip: %#x, %v", x, err)
	}
	if _, err := DecodePutCommit(cb[:7]); !errors.Is(err, ErrBodyLength) {
		t.Fatal("short COMMIT accepted")
	}

	sb := AppendStatsReq(nil, 0)
	if s, err := DecodeStatsReq(sb); err != nil || s != 0 {
		t.Fatalf("STATS round-trip: %d, %v", s, err)
	}
}

func TestNegotiateLimits(t *testing.T) {
	server := DefaultLimits()
	cases := []struct {
		batch, frame                   uint32
		wantBatch, wantFrame, wantBlob uint32
	}{
		{0, 0, DefaultMaxBatchKeys, DefaultMaxFrameLen, DefaultMaxBlobLen},          // no opinion
		{128, 16 << 20, 128, 16 << 20, 16 << 20},                                    // lower frame → blob clamped under it
		{1024, 1 << 30, DefaultMaxBatchKeys, DefaultMaxFrameLen, DefaultMaxBlobLen}, // client higher loses; blob unchanged
	}
	for i, c := range cases {
		got := NegotiateLimits(server, c.batch, c.frame)
		if got.MaxBatchKeys != c.wantBatch || got.MaxFrameLen != c.wantFrame {
			t.Fatalf("case %d: %+v", i, got)
		}
		if got.MaxBlobLen != c.wantBlob {
			t.Fatalf("case %d: MaxBlobLen = %d, want %d (blob must stay <= frame)", i, got.MaxBlobLen, c.wantBlob)
		}
		if got.MaxBlobLen > got.MaxFrameLen {
			t.Fatalf("case %d: blob %d exceeds frame %d — unservable blocks", i, got.MaxBlobLen, got.MaxFrameLen)
		}
		if got.InitialCredit != server.InitialCredit {
			t.Fatalf("case %d: credit (server-dictated) changed: %+v", i, got)
		}
	}
}

func TestIntersectFeatures(t *testing.T) {
	// Unknown bits (5–63) are stripped even if both sides claim them.
	unknown := uint64(1) << 42
	if got := IntersectFeatures(FeatOOO|unknown, FeatOOO|unknown); got != FeatOOO {
		t.Fatalf("unknown bit survived negotiation: %#x", got)
	}
	if got := IntersectFeatures(FeatOOO|FeatExistsBitmap, FeatOOO); got != FeatOOO {
		t.Fatalf("intersection: %#x", got)
	}
}

func TestErrorStatusMapping(t *testing.T) {
	if ErrorStatus(ErrBodyLength) != StatusErrMalformed {
		t.Fatal("ErrBodyLength must map to ERR_MALFORMED")
	}
	if ErrorStatus(ErrKeyCount) != StatusErrBatchTooLarge {
		t.Fatal("ErrKeyCount must map to ERR_BATCH_TOO_LARGE")
	}
	if ErrorStatus(errors.New("bug")) != StatusErrInternal {
		t.Fatal("unknown errors must map to ERR_INTERNAL")
	}
	for _, err := range []error{ErrBodyLength, ErrKeyCount} {
		if !errors.Is(err, ErrBadBody) {
			t.Fatalf("%v must classify as ErrBadBody", err)
		}
		if errors.Is(err, ErrFatalFrame) {
			t.Fatalf("%v must never be connection-fatal", err)
		}
	}
}

// TestEncodersZeroAllocWithCapacity pins the pooled-scratch contract: given a
// pre-sized dst, the hot-path encoders allocate nothing.
func TestEncodersZeroAllocWithCapacity(t *testing.T) {
	keys := testKeys(32)
	perKey := make([]Status, 32)
	descs := []Desc{{Status: StatusOK, Len: 1 << 20, XXH3: 1}, {Status: StatusNotFound}}
	scratch := make([]byte, 0, 4096)

	cases := []struct {
		name string
		fn   func()
	}{
		{"AppendKeyList", func() { sinkBytes = AppendKeyList(scratch[:0], 0, keys) }},
		{"AppendExistsResp", func() { sinkBytes = AppendExistsResp(scratch[:0], 32, 7, perKey) }},
		{"AppendKeyStatusResp", func() { sinkBytes = AppendKeyStatusResp(scratch[:0], perKey) }},
		{"AppendGetRespHeader", func() { sinkBytes = AppendGetRespHeader(scratch[:0], StatusOK, 0, 2, descs) }},
		{"AppendErrorResp", func() { sinkBytes = AppendErrorResp(scratch[:0], StatusErrBusy) }},
	}
	for _, c := range cases {
		if allocs := testing.AllocsPerRun(1000, c.fn); allocs != 0 {
			t.Errorf("%s: %v allocs/op with pre-sized dst, want 0", c.name, allocs)
		}
	}
}

var sinkBytes []byte

// TestCanonicalEncoding pins the exact wire bytes the encoders emit —
// reserved fields and §0 pad bytes MUST be 0x00 on send even though receivers
// ignore them; fixpoint tests alone cannot see a nonzero reserved byte.
func TestCanonicalEncoding(t *testing.T) {
	got := AppendExistsResp(nil, 3, 2, []Status{StatusOK, StatusOK, StatusNotFound})
	want := []byte{
		0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, // preamble: OK, count=3
		0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // n_consecutive=2, reserved=0
		0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, // bitmap OK,OK,NOT_FOUND + 5 zero pad
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("EXISTS response bytes:\n got %x\nwant %x", got, want)
	}

	hr := AppendHelloReq(nil, HelloReq{ProtoMin: 1, ProtoMax: 1, Namespace: "ns"})
	wantHR := []byte{
		0x01, 0x01, 0x00, 0x00, // proto_min, proto_max, reserved u16
		0, 0, 0, 0, 0, 0, 0, 0, // feature_bits
		0, 0, 0, 0, 0, 0, 0, 0, // max_batch_keys, max_frame_len
		0, 0, 0, 0, 0, 0, 0, 0, // reserved u64
		0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, // token_len=0, ns_len=2, name_len=0, reserved
		'n', 's', 0x00, 0x00, // ns + 2 zero pad (36+2 → padTo8 = 40)
	}
	if !bytes.Equal(hr, wantHR) {
		t.Fatalf("HELLO request bytes:\n got %x\nwant %x", hr, wantHR)
	}

	// PutDesc into a dirty buffer: every one of the 16 bytes must be written
	// (a reserved byte left unwritten would leak pooled-buffer residue).
	dirty := bytes.Repeat([]byte{0xFF}, DescSize)
	PutDesc(dirty, Desc{Status: StatusOK, Len: 1, XXH3: 2})
	wantDesc := []byte{0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0}
	if !bytes.Equal(dirty, wantDesc) {
		t.Fatalf("PutDesc left residue: %x", dirty)
	}
}

// TestOKExistsIsFullBodied pins that OK_EXISTS (a success, §9) takes the
// full-body decode path, never the preamble-only error path.
func TestOKExistsIsFullBodied(t *testing.T) {
	eb := AppendExistsResp(nil, 1, 1, []Status{StatusOKExists})
	eb[0] = byte(StatusOKExists)
	if r, err := DecodeExistsResp(eb, true); err != nil || r.NConsecutive != 1 || len(r.PerKey) != 1 {
		t.Fatalf("OK_EXISTS exists-resp: %+v, %v", r, err)
	}

	kb := AppendKeyStatusResp(nil, []Status{StatusOKExists})
	kb[0] = byte(StatusOKExists)
	if p, st, err := DecodeKeyStatusResp(kb); err != nil || len(st) != 1 || p.Status != StatusOKExists {
		t.Fatalf("OK_EXISTS key-status: %+v, %v", p, err)
	}

	gb := AppendGetRespHeader(nil, StatusOKExists, 0, 1, []Desc{{Len: 5}})
	if g, err := DecodeGetRespHeader(gb); err != nil || g.Count != 1 || g.Desc(0).Len != 5 {
		t.Fatalf("OK_EXISTS get-resp: %+v, %v", g, err)
	}
}

// TestShortBodyGuards pins every decoder's leading length guard with bodies
// short enough that a removed guard would read (or panic) past the input, and
// asserts error-path returns are zero-valued.
func TestShortBodyGuards(t *testing.T) {
	// 7-byte key-list body whose first 4 bytes decode to a huge n: the length
	// guard must fire BEFORE the n_keys cap comparison ever happens.
	shortHugeN := []byte{0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0}
	aux, keys, err := DecodeKeyList(shortHugeN, 8, nil)
	if !errors.Is(err, ErrBodyLength) {
		t.Fatalf("short key-list: got %v, want ErrBodyLength (never ErrKeyCount)", err)
	}
	if aux != 0 || keys != nil {
		t.Fatalf("error path must return zero values: aux=%d keys=%v", aux, keys)
	}

	if s, err := DecodeStatsReq(make([]byte, 7)); !errors.Is(err, ErrBodyLength) || s != 0 {
		t.Fatalf("short stats req: %d, %v", s, err)
	}
	if _, err := DecodeHelloReq(make([]byte, 10)); !errors.Is(err, ErrBodyLength) {
		t.Fatalf("short hello req accepted")
	}
	// An OK-status HELLO response shorter than its fixed prefix.
	if _, _, err := DecodeHelloResp(AppendPreamble(nil, StatusOK, 0)); !errors.Is(err, ErrBodyLength) {
		t.Fatal("short OK hello resp accepted")
	}
	// Non-OK responses with trailing bytes, for the two decoders not already
	// covered by their round-trip tests.
	if _, err := DecodeGetRespHeader(append(AppendErrorResp(nil, StatusErrBusy), 1, 2, 3, 4, 5, 6, 7, 8)); !errors.Is(err, ErrBodyLength) {
		t.Fatal("non-OK get-resp with trailing bytes accepted")
	}
	if _, _, err := DecodeHelloResp(append(AppendErrorResp(nil, StatusErrAuthFailed), 1, 2, 3, 4, 5, 6, 7, 8)); !errors.Is(err, ErrBodyLength) {
		t.Fatal("non-OK hello resp with trailing bytes accepted")
	}
}
