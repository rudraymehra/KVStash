package protocol

import (
	"encoding/binary"
	"errors"
)

// Body codecs for PROTOCOL.md §3. Conventions, extending header.go's style:
//
//   - Encoders are append-style — AppendX(dst, ...) []byte — writing into
//     caller scratch and returning the extended slice; zero-alloc when cap(dst)
//     suffices (the pooled-scratch one-writev pattern).
//   - Decoders are strict: the expected body length is computed from the
//     counts (including the §0 pad-to-8 rule) and anything else is malformed.
//     Pad CONTENTS are ignored per §0; trailing bytes beyond the padded length
//     are rejected — forward evolution arrives behind feature bits, never as
//     silent trailing bytes.
//   - Body errors are never connection-fatal (§9: the frame was already read
//     in full; the server answers a status and framing stays in sync). Classify
//     with errors.Is(err, ErrBadBody); map to the wire status via ErrorStatus.

// ErrBadBody classifies request/response-body decode failures. Never fatal.
var ErrBadBody = errors.New("protocol: malformed body")

// bodyError is a pre-constructed sentinel (reject paths allocate nothing)
// carrying the §9 status a server should answer with.
type bodyError struct {
	msg    string
	status Status
}

func (e *bodyError) Error() string { return e.msg }

// Is makes errors.Is(err, ErrBadBody) the one classification test callers need.
func (e *bodyError) Is(target error) bool { return target == ErrBadBody }

var (
	// ErrBodyLength: the body's length is inconsistent with its counts and
	// the §0 padding rule. Answered as ERR_MALFORMED.
	ErrBodyLength = &bodyError{msg: "protocol: body length inconsistent with counts", status: StatusErrMalformed}

	// ErrKeyCount: n_keys exceeds the negotiated max_batch_keys (§4).
	// Detected BEFORE any allocation. Answered as ERR_BATCH_TOO_LARGE.
	ErrKeyCount = &bodyError{msg: "protocol: n_keys exceeds negotiated max_batch_keys", status: StatusErrBatchTooLarge}
)

// ErrorStatus maps a body-decode error to the §9 status the server answers
// with. Unknown errors map to ERR_INTERNAL (a server bug, not wire input).
func ErrorStatus(err error) Status {
	var be *bodyError
	if errors.As(err, &be) {
		return be.status
	}
	return StatusErrInternal
}

// padTo8 rounds n up to the next multiple of 8 (§0: pad bytes are 0x00,
// included in payload_len, ignored on receive; already-aligned → no pad).
func padTo8(n int) int { return (n + 7) &^ 7 }

// appendPad8 zero-pads dst so the BODY REGION (unpadded bytes, not dst's total
// length — dst may carry a prefix) reaches the next 8-byte multiple. Pads with
// zeros, the §0 pad byte value.
func appendPad8(dst []byte, unpadded int) []byte {
	for i := unpadded; i < padTo8(unpadded); i++ {
		dst = append(dst, 0)
	}
	return dst
}

// ---------------------------------------------------------------------------
// Uniform response preamble (§3): status u8 | reserved u8[3] | count u32.

// PreambleSize is the length of the uniform response preamble.
const PreambleSize = 8

// Preamble is the decoded uniform response preamble.
type Preamble struct {
	Status Status
	Count  uint32
}

// AppendPreamble appends the 8-byte preamble.
func AppendPreamble(dst []byte, status Status, count uint32) []byte {
	dst = append(dst, byte(status), 0, 0, 0)
	return binary.LittleEndian.AppendUint32(dst, count)
}

// DecodePreamble decodes the leading 8 bytes of a response payload.
func DecodePreamble(b []byte) (Preamble, error) {
	if len(b) < PreambleSize {
		return Preamble{}, ErrBodyLength
	}
	return Preamble{
		Status: Status(b[0]),
		Count:  binary.LittleEndian.Uint32(b[4:]),
	}, nil
}

// AppendErrorResp encodes the §3 rule "on any non-OK/OK_EXISTS batch-level
// status, the response payload is exactly the 8-byte preamble with count=0".
// It is also the payload of the §9 protocol-fatal report frame.
func AppendErrorResp(dst []byte, status Status) []byte {
	return AppendPreamble(dst, status, 0)
}

// ---------------------------------------------------------------------------
// Key-list request body, shared by BATCH_EXISTS / BATCH_GET / TOUCH_LEASE /
// PIN / DELETE (§3.2–3.7): n_keys u32 | aux u32 | key[32] × n.
// aux is reserved (zero) for every verb except TOUCH_LEASE, where it is ttl_ms.

// DecodeKeyList decodes the shared request body, appending the keys to dst
// (pass dst[:0] of a per-connection reusable slice for amortized zero-alloc).
//
// Keys are always COPIED, never aliased: the transport lends body buffers
// that are returned and recycled (0xDE-poisoned in debug builds) after the
// handler finishes, while keys outlive the frame in index probes and lease
// tables. The cost ceiling is 512×32 B = 16 KiB per batch — bought safety.
//
// n_keys is validated against maxKeys BEFORE anything is appended or
// allocated; the body length must then be exactly 8 + 32×n (inherently
// 8-aligned, so the §0 rule adds no pad).
func DecodeKeyList(body []byte, maxKeys uint32, dst [][32]byte) (aux uint32, keys [][32]byte, err error) {
	if len(body) < 8 {
		return 0, dst, ErrBodyLength
	}
	n := binary.LittleEndian.Uint32(body[0:])
	if n > maxKeys {
		return 0, dst, ErrKeyCount
	}
	if len(body) != 8+32*int(n) {
		return 0, dst, ErrBodyLength
	}
	aux = binary.LittleEndian.Uint32(body[4:])
	for i := 0; i < int(n); i++ {
		var k [32]byte
		copy(k[:], body[8+32*i:])
		dst = append(dst, k)
	}
	return aux, dst, nil
}

// AppendKeyList appends the shared request body (client side, tests, goldens).
func AppendKeyList(dst []byte, aux uint32, keys [][32]byte) []byte {
	dst = binary.LittleEndian.AppendUint32(dst, uint32(len(keys))) //nolint:gosec // G115: len(keys) is capped by max_batch_keys (512) far below MaxUint32
	dst = binary.LittleEndian.AppendUint32(dst, aux)
	for i := range keys {
		dst = append(dst, keys[i][:]...)
	}
	return dst
}

// ---------------------------------------------------------------------------
// BATCH_EXISTS response (§3.2).

// AppendExistsResp appends the EXISTS response payload. perKey == nil means
// FEAT_EXISTS_BITMAP was not negotiated: no status region, no pad. Otherwise
// the per-key bytes are zero-padded to the next 8-byte multiple; the caller
// MUST pass nKeys == len(perKey) in that case (the preamble count and the
// bitmap length are the same n_keys — a mismatch would let the decoder read a
// pad byte as a status).
func AppendExistsResp(dst []byte, nKeys, nConsecutive uint32, perKey []Status) []byte {
	dst = AppendPreamble(dst, StatusOK, nKeys)
	dst = binary.LittleEndian.AppendUint32(dst, nConsecutive)
	dst = binary.LittleEndian.AppendUint32(dst, 0) // reserved
	if perKey != nil {
		for _, s := range perKey {
			dst = append(dst, byte(s))
		}
		dst = appendPad8(dst, len(perKey))
	}
	return dst
}

// ExistsResp is the decoded EXISTS response.
type ExistsResp struct {
	Preamble
	NConsecutive uint32
	// PerKey aliases the input body (the client owns its receive buffer until
	// it finishes processing). nil when the bitmap is absent. Elements are §9
	// per-key statuses: Status(PerKey[i]).
	PerKey []byte
}

// DecodeExistsResp decodes an EXISTS response payload. Bitmap presence is a
// negotiation fact, not inferable from the bytes (padding makes short bitmaps
// ambiguous), so the caller states it. A non-OK batch status is preamble-only
// per §3 and returns with only Preamble populated.
func DecodeExistsResp(body []byte, expectBitmap bool) (ExistsResp, error) {
	p, err := DecodePreamble(body)
	if err != nil {
		return ExistsResp{}, err
	}
	if p.Status != StatusOK && p.Status != StatusOKExists {
		// §3: a non-OK response is EXACTLY the preamble with count=0.
		if len(body) != PreambleSize || p.Count != 0 {
			return ExistsResp{}, ErrBodyLength
		}
		return ExistsResp{Preamble: p}, nil
	}
	want := PreambleSize + 8
	if expectBitmap {
		want += padTo8(int(p.Count))
	}
	if len(body) != want {
		return ExistsResp{}, ErrBodyLength
	}
	r := ExistsResp{
		Preamble:     p,
		NConsecutive: binary.LittleEndian.Uint32(body[PreambleSize:]),
	}
	if expectBitmap {
		r.PerKey = body[PreambleSize+8 : PreambleSize+8+int(p.Count)]
	}
	return r, nil
}

// ---------------------------------------------------------------------------
// Per-key status responses for TOUCH_LEASE / PIN / DELETE (§3.5–3.7):
// preamble + status bytes zero-padded to the next 8-byte multiple.

// AppendKeyStatusResp appends the response payload (count = len(perKey)).
func AppendKeyStatusResp(dst []byte, perKey []Status) []byte {
	dst = AppendPreamble(dst, StatusOK, uint32(len(perKey))) //nolint:gosec // G115: len(perKey) is capped by max_batch_keys (512)
	for _, s := range perKey {
		dst = append(dst, byte(s))
	}
	return appendPad8(dst, len(perKey))
}

// DecodeKeyStatusResp decodes the response payload. The returned status bytes
// alias body (client-owned buffer); nil on a non-OK preamble-only response.
func DecodeKeyStatusResp(body []byte) (Preamble, []byte, error) {
	p, err := DecodePreamble(body)
	if err != nil {
		return Preamble{}, nil, err
	}
	if p.Status != StatusOK && p.Status != StatusOKExists {
		if len(body) != PreambleSize || p.Count != 0 {
			return Preamble{}, nil, ErrBodyLength
		}
		return p, nil, nil
	}
	if len(body) != PreambleSize+padTo8(int(p.Count)) {
		return Preamble{}, nil, ErrBodyLength
	}
	return p, body[PreambleSize : PreambleSize+int(p.Count)], nil
}

// ---------------------------------------------------------------------------
// BATCH_GET response header region (§3.3). The codec never touches payload
// bytes: the server emits ONE writev with iov[0] = this header region from a
// pooled scratch buffer and iov[1..] = block bytes straight from arena memory.
// F_MORE frame splitting is server logic, not codec logic.

// DescSize is the length of one §3 descriptor:
// status u8 | reserved u8[3] | len u32 | xxh3_64 u64.
const DescSize = 16

// Desc is one decoded descriptor.
type Desc struct {
	Status Status
	Len    uint32
	XXH3   uint64
}

// PutDesc writes d into the first DescSize bytes of b. It panics if b is
// shorter — callers own a correctly sized region (MarshalTo's contract style).
func PutDesc(b []byte, d Desc) {
	_ = b[DescSize-1] //nolint:gosec // G602: panic on short regions is the documented contract (TestDescRoundTrip)
	b[0] = byte(d.Status)
	b[1], b[2], b[3] = 0, 0, 0
	binary.LittleEndian.PutUint32(b[4:], d.Len)
	binary.LittleEndian.PutUint64(b[8:], d.XXH3)
}

// GetDesc reads a descriptor from the first DescSize bytes of b; panics if
// short — used only under a length-validated region.
func GetDesc(b []byte) Desc {
	_ = b[DescSize-1] //nolint:gosec // G602: panic on short regions is the documented contract; used under validated lengths
	return Desc{
		Status: Status(b[0]),
		Len:    binary.LittleEndian.Uint32(b[4:]),
		XXH3:   binary.LittleEndian.Uint64(b[8:]),
	}
}

// GetRespHeaderSize returns the size of the GET response header region for
// nDesc descriptors — what the server sizes its pooled iov[0] scratch to.
func GetRespHeaderSize(nDesc int) int {
	return PreambleSize + 8 + DescSize*nDesc
}

// AppendGetRespHeader appends preamble(status, count=len(descs)) +
// first_index + total_keys + the descriptor table.
func AppendGetRespHeader(dst []byte, status Status, firstIndex, totalKeys uint32, descs []Desc) []byte {
	dst = AppendPreamble(dst, status, uint32(len(descs))) //nolint:gosec // G115: len(descs) is capped by max_batch_keys (512)
	dst = binary.LittleEndian.AppendUint32(dst, firstIndex)
	dst = binary.LittleEndian.AppendUint32(dst, totalKeys)
	for _, d := range descs {
		var db [DescSize]byte
		PutDesc(db[:], d)
		dst = append(dst, db[:]...)
	}
	return dst
}

// GetRespHeader is the decoded GET response header region.
type GetRespHeader struct {
	Preamble
	FirstIndex uint32
	TotalKeys  uint32
	descs      []byte // validated region aliasing body; len = DescSize*Count
}

// DecodeGetRespHeader decodes the header region at the START of a GET
// response payload. body may extend beyond the region (the concatenated
// payload bytes follow); PayloadOffset says where they start. A non-OK batch
// status is preamble-only per §3.
func DecodeGetRespHeader(body []byte) (GetRespHeader, error) {
	p, err := DecodePreamble(body)
	if err != nil {
		return GetRespHeader{}, err
	}
	if p.Status != StatusOK && p.Status != StatusOKExists {
		if len(body) != PreambleSize || p.Count != 0 {
			return GetRespHeader{}, ErrBodyLength
		}
		return GetRespHeader{Preamble: p}, nil
	}
	want := GetRespHeaderSize(int(p.Count))
	if len(body) < want {
		return GetRespHeader{}, ErrBodyLength
	}
	return GetRespHeader{
		Preamble:   p,
		FirstIndex: binary.LittleEndian.Uint32(body[PreambleSize:]),
		TotalKeys:  binary.LittleEndian.Uint32(body[PreambleSize+4:]),
		descs:      body[PreambleSize+8 : want],
	}, nil
}

// Desc returns descriptor i. It panics outside [0, Count) — callers iterate
// under Count, which DecodeGetRespHeader validated against the body length.
func (g GetRespHeader) Desc(i int) Desc {
	return GetDesc(g.descs[i*DescSize:])
}

// PayloadOffset returns the offset within the response payload where the
// concatenated block bytes begin.
func (g GetRespHeader) PayloadOffset() int { return GetRespHeaderSize(int(g.Count)) }

// ---------------------------------------------------------------------------
// HELLO (§3.1) — the cold path (once per connection), so these are the only
// allocating decoders in the package (strings are copied out of the body).

// Limits are the §4 negotiated per-connection caps.
type Limits struct {
	MaxBatchKeys  uint32
	MaxFrameLen   uint32
	MaxBlobLen    uint32
	InitialCredit uint32
}

// DefaultLimits returns the §4 defaults.
func DefaultLimits() Limits {
	return Limits{
		MaxBatchKeys:  DefaultMaxBatchKeys,
		MaxFrameLen:   DefaultMaxFrameLen,
		MaxBlobLen:    DefaultMaxBlobLen,
		InitialCredit: DefaultInitialCredit,
	}
}

// ServerFeatures is the v1 reference-implementation feature set (§10).
const ServerFeatures uint64 = FeatOOO | FeatExistsBitmap

// knownFeatures masks the §10 bits assigned in v1; reserved bits 5–63 are
// stripped during negotiation so nothing unnegotiable can be "agreed".
const knownFeatures uint64 = FeatOOO | FeatExistsBitmap | FeatPayloadCRC32C |
	FeatCreditSymmetric | FeatTLSUpgrade

// IntersectFeatures returns the negotiated feature set: the intersection of
// both sides, restricted to bits this protocol version defines.
func IntersectFeatures(client, server uint64) uint64 {
	return client & server & knownFeatures
}

// NegotiateLimits computes min(server, client) per §3.1. The HELLO request
// carries only the client's max_batch_keys and max_frame_len proposals
// (blob/credit are server-dictated); a client value of 0 means "no opinion".
// Callers guarantee the server side already satisfies the §4 floors
// (config.Validate's job). MaxBlobLen is clamped under the negotiated frame
// MINUS the single-descriptor GET response header: otherwise a blob equal to
// the frame cap could be stored (chunked PUT, each chunk ≤ frame) whose every
// GET response frame — payload plus the 32-byte header region — would exceed
// max_frame_len and be rejected as over-cap by a conformant client parser.
func NegotiateLimits(server Limits, clientBatchKeys, clientFrameLen uint32) Limits {
	l := server
	if clientBatchKeys != 0 && clientBatchKeys < l.MaxBatchKeys {
		l.MaxBatchKeys = clientBatchKeys
	}
	if clientFrameLen != 0 && clientFrameLen < l.MaxFrameLen {
		l.MaxFrameLen = clientFrameLen
	}
	if headroom := uint32(GetRespHeaderSize(1)); l.MaxBlobLen > l.MaxFrameLen-headroom { //nolint:gosec // G115: constant 32
		l.MaxBlobLen = l.MaxFrameLen - headroom
	}
	return l
}

// helloReqFixed is the fixed prefix of the HELLO request body:
// proto_min u8 | proto_max u8 | reserved u16 | feature_bits u64 |
// max_batch_keys u32 | max_frame_len u32 | reserved u64 |
// token_len u16 | ns_len u16 | client_name_len u16 | reserved u16.
const helloReqFixed = 36

// MaxHelloBody bounds a HELLO request body: the u16 length prefixes cap each
// string at 64 KiB, so fixed + 3×65535 padded ≈ 197 KiB. The transport uses
// this (rounded up) as its pre-negotiation ParseHeader cap.
const MaxHelloBody = helloReqFixed + 3*65535 + 7

// HelloReq is the §3.1 request body.
type HelloReq struct {
	ProtoMin, ProtoMax uint8
	Features           uint64
	MaxBatchKeys       uint32
	MaxFrameLen        uint32
	Token              []byte // []byte, not string: harder to accidentally log
	Namespace          string
	ClientName         string
}

// AppendHelloReq appends the request body, padded per §0.
func AppendHelloReq(dst []byte, r HelloReq) []byte {
	start := len(dst)
	dst = append(dst, r.ProtoMin, r.ProtoMax, 0, 0)
	dst = binary.LittleEndian.AppendUint64(dst, r.Features)
	dst = binary.LittleEndian.AppendUint32(dst, r.MaxBatchKeys)
	dst = binary.LittleEndian.AppendUint32(dst, r.MaxFrameLen)
	dst = binary.LittleEndian.AppendUint64(dst, 0)                         // reserved
	dst = binary.LittleEndian.AppendUint16(dst, uint16(len(r.Token)))      //nolint:gosec // G115: encoder trusts its own caller; the decoder is the trust boundary
	dst = binary.LittleEndian.AppendUint16(dst, uint16(len(r.Namespace)))  //nolint:gosec // G115: as above
	dst = binary.LittleEndian.AppendUint16(dst, uint16(len(r.ClientName))) //nolint:gosec // G115: as above
	dst = binary.LittleEndian.AppendUint16(dst, 0)
	dst = append(dst, r.Token...)
	dst = append(dst, r.Namespace...)
	dst = append(dst, r.ClientName...)
	return appendPad8(dst, len(dst)-start)
}

// DecodeHelloReq decodes and COPIES the request body (auth data must not
// alias a recycled transport buffer).
func DecodeHelloReq(body []byte) (HelloReq, error) {
	if len(body) < helloReqFixed {
		return HelloReq{}, ErrBodyLength
	}
	tokenLen := int(binary.LittleEndian.Uint16(body[28:]))
	nsLen := int(binary.LittleEndian.Uint16(body[30:]))
	nameLen := int(binary.LittleEndian.Uint16(body[32:]))
	if len(body) != padTo8(helloReqFixed+tokenLen+nsLen+nameLen) {
		return HelloReq{}, ErrBodyLength
	}
	r := HelloReq{
		ProtoMin:     body[0],
		ProtoMax:     body[1],
		Features:     binary.LittleEndian.Uint64(body[4:]),
		MaxBatchKeys: binary.LittleEndian.Uint32(body[12:]),
		MaxFrameLen:  binary.LittleEndian.Uint32(body[16:]),
	}
	off := helloReqFixed
	r.Token = append([]byte(nil), body[off:off+tokenLen]...)
	off += tokenLen
	r.Namespace = string(body[off : off+nsLen])
	off += nsLen
	r.ClientName = string(body[off : off+nameLen])
	return r, nil
}

// helloRespFixed is the fixed prefix of the HELLO response body (§3.1):
// preamble | proto u8, reserved u8[3] | feature_bits u64 |
// max_batch_keys u32 | max_frame_len u32 | max_blob_len u32 |
// namespace_id u32 | initial_credit u32 | lease_default_ms u32 |
// lease_max_ms u32 | stream_timeout_ms u32 | server_name_len u16, reserved u16.
const helloRespFixed = PreambleSize + 4 + 8 + 32 + 4

// HelloResp is the §3.1 response body (the OK shape; non-OK responses are
// preamble-only and carry F_FATAL on the frame).
type HelloResp struct {
	Proto           uint8
	Features        uint64
	Limits          Limits
	NamespaceID     uint32
	LeaseDefaultMS  uint32
	LeaseMaxMS      uint32
	StreamTimeoutMS uint32
	ServerName      string
}

// AppendHelloResp appends the OK response body, padded per §0.
func AppendHelloResp(dst []byte, r HelloResp) []byte {
	start := len(dst)
	dst = AppendPreamble(dst, StatusOK, 0)
	dst = append(dst, r.Proto, 0, 0, 0)
	dst = binary.LittleEndian.AppendUint64(dst, r.Features)
	dst = binary.LittleEndian.AppendUint32(dst, r.Limits.MaxBatchKeys)
	dst = binary.LittleEndian.AppendUint32(dst, r.Limits.MaxFrameLen)
	dst = binary.LittleEndian.AppendUint32(dst, r.Limits.MaxBlobLen)
	dst = binary.LittleEndian.AppendUint32(dst, r.NamespaceID)
	dst = binary.LittleEndian.AppendUint32(dst, r.Limits.InitialCredit)
	dst = binary.LittleEndian.AppendUint32(dst, r.LeaseDefaultMS)
	dst = binary.LittleEndian.AppendUint32(dst, r.LeaseMaxMS)
	dst = binary.LittleEndian.AppendUint32(dst, r.StreamTimeoutMS)
	dst = binary.LittleEndian.AppendUint16(dst, uint16(len(r.ServerName))) //nolint:gosec // G115: encoder trusts its own caller
	dst = binary.LittleEndian.AppendUint16(dst, 0)
	dst = append(dst, r.ServerName...)
	return appendPad8(dst, len(dst)-start)
}

// DecodeHelloResp decodes the response body (client side; copies the name).
// On a non-OK status it returns the preamble (for the caller to inspect
// Status) with a zero HelloResp and nil error; ErrBadBody is returned only for
// a structurally malformed body (wrong length, or a non-OK count != 0).
func DecodeHelloResp(body []byte) (Preamble, HelloResp, error) {
	p, err := DecodePreamble(body)
	if err != nil {
		return Preamble{}, HelloResp{}, err
	}
	if p.Status != StatusOK {
		if len(body) != PreambleSize || p.Count != 0 {
			return Preamble{}, HelloResp{}, ErrBodyLength
		}
		return p, HelloResp{}, nil
	}
	if len(body) < helloRespFixed {
		return Preamble{}, HelloResp{}, ErrBodyLength
	}
	nameLen := int(binary.LittleEndian.Uint16(body[helloRespFixed-4:]))
	if len(body) != padTo8(helloRespFixed+nameLen) {
		return Preamble{}, HelloResp{}, ErrBodyLength
	}
	r := HelloResp{
		Proto:    body[PreambleSize],
		Features: binary.LittleEndian.Uint64(body[PreambleSize+4:]),
		Limits: Limits{
			MaxBatchKeys:  binary.LittleEndian.Uint32(body[PreambleSize+12:]),
			MaxFrameLen:   binary.LittleEndian.Uint32(body[PreambleSize+16:]),
			MaxBlobLen:    binary.LittleEndian.Uint32(body[PreambleSize+20:]),
			InitialCredit: binary.LittleEndian.Uint32(body[PreambleSize+28:]),
		},
		NamespaceID:     binary.LittleEndian.Uint32(body[PreambleSize+24:]),
		LeaseDefaultMS:  binary.LittleEndian.Uint32(body[PreambleSize+32:]),
		LeaseMaxMS:      binary.LittleEndian.Uint32(body[PreambleSize+36:]),
		StreamTimeoutMS: binary.LittleEndian.Uint32(body[PreambleSize+40:]),
		ServerName:      string(body[helloRespFixed : helloRespFixed+nameLen]),
	}
	return p, r, nil
}

// ---------------------------------------------------------------------------
// PUT_STREAM bodies (§3.4). CHUNK is raw bytes and ABORT is empty — no codec.

// putBeginSize: total_len u32 | ttl_ms u32 | xxh3_64_hint u64 | flags u32 |
// reserved u32.
const putBeginSize = 24

// PutBeginBody is the §3.4 BEGIN payload.
type PutBeginBody struct {
	TotalLen uint32
	TTLms    uint32
	XXH3Hint uint64
	Flags    uint32 // no bits defined in v1: 0 on send, IGNORED on receive
}

// AppendPutBegin appends the BEGIN body.
func AppendPutBegin(dst []byte, b PutBeginBody) []byte {
	dst = binary.LittleEndian.AppendUint32(dst, b.TotalLen)
	dst = binary.LittleEndian.AppendUint32(dst, b.TTLms)
	dst = binary.LittleEndian.AppendUint64(dst, b.XXH3Hint)
	dst = binary.LittleEndian.AppendUint32(dst, b.Flags)
	return binary.LittleEndian.AppendUint32(dst, 0)
}

// DecodePutBegin decodes the BEGIN body. Nonzero Flags are returned as-is and
// IGNORED per §3.4 ("MUST be 0 on send, ignored on receive") — never rejected.
func DecodePutBegin(body []byte) (PutBeginBody, error) {
	if len(body) != putBeginSize {
		return PutBeginBody{}, ErrBodyLength
	}
	return PutBeginBody{
		TotalLen: binary.LittleEndian.Uint32(body[0:]),
		TTLms:    binary.LittleEndian.Uint32(body[4:]),
		XXH3Hint: binary.LittleEndian.Uint64(body[8:]),
		Flags:    binary.LittleEndian.Uint32(body[16:]),
	}, nil
}

// AppendPutCommit appends the COMMIT body: the authoritative xxh3_64.
func AppendPutCommit(dst []byte, xxh3 uint64) []byte {
	return binary.LittleEndian.AppendUint64(dst, xxh3)
}

// DecodePutCommit decodes the COMMIT body.
func DecodePutCommit(body []byte) (uint64, error) {
	if len(body) != 8 {
		return 0, ErrBodyLength
	}
	return binary.LittleEndian.Uint64(body), nil
}

// ---------------------------------------------------------------------------
// STATS request (§3.8): sections u32 bitmask (0 = all; no bits assigned in
// v1, servers MAY ignore) | reserved u32. The JSON response is server-built.

// DecodeStatsReq decodes the STATS request body.
func DecodeStatsReq(body []byte) (sections uint32, err error) {
	if len(body) != 8 {
		return 0, ErrBodyLength
	}
	return binary.LittleEndian.Uint32(body), nil
}

// AppendStatsReq appends the STATS request body.
func AppendStatsReq(dst []byte, sections uint32) []byte {
	dst = binary.LittleEndian.AppendUint32(dst, sections)
	return binary.LittleEndian.AppendUint32(dst, 0)
}
