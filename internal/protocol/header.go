package protocol

import (
	"encoding/binary"
	"errors"
)

// HeaderSize is the fixed length of every frame header (PROTOCOL.md §1).
const HeaderSize = 64

// Version1 is the only header-layout generation in existence. It is bumped
// only if the 64-byte layout ever changes (intended: never) — capability
// evolution happens in feature bits, not here.
const Version1 uint8 = 0x01

// magicLE is the frame magic as a little-endian u32: the wire bytes
// 4B 56 42 31 read literally as ASCII "KVB1" in a hexdump. It is a const —
// not a var — so no code in the process can reassign the wire contract
// (a mutable magic would let one stray write silently redefine the protocol
// for every subsequent frame while local round-trips kept passing).
const magicLE uint32 = 0x3142564B

// Field offsets within the 64-byte header. crcOffset doubles as the length of
// the CRC-protected prefix (bytes 0–59).
const (
	magicOffset     = 0
	versionOffset   = 4
	opcodeOffset    = 5
	flagsOffset     = 6
	namespaceOffset = 8
	creditOffset    = 12
	requestIDOffset = 16
	keyOffset       = 24
	payloadOffset   = 56
	crcOffset       = 60
)

// Header is the decoded 64-byte frame header. It is a plain 64-byte value
// (zero padding) — parsing and marshalling allocate nothing themselves.
//
// One caveat for callers: buffers handed to ParseHeader/MarshalTo/HeaderCRC
// escape to the heap (the stdlib's hardware-CRC assembly routine is not
// marked noescape), so a stack array like `var b [HeaderSize]byte` passed in
// will be heap-allocated by the compiler. Use pooled or heap buffers on the
// hot path — which the transport's buffer-lending design does anyway.
type Header struct {
	Magic       [4]byte
	Version     uint8
	Opcode      Opcode
	Flags       uint16
	NamespaceID uint32
	Credit      uint32
	RequestID   uint64
	Key         [32]byte
	PayloadLen  uint32
	CRC         uint32 // populated by ParseHeader (the wire value); ignored and not updated by MarshalTo
}

// ErrFatalFrame classifies header failures after which the connection MUST be
// closed with no resynchronization attempt (PROTOCOL.md §1): bad magic,
// unsupported version, CRC mismatch, or a buffer that cannot hold a header.
// Test with errors.Is(err, ErrFatalFrame). Recoverable errors (payload over
// cap) do not match; the frame is skipped via payload_len and the connection
// stays healthy.
var ErrFatalFrame = errors.New("protocol: fatal frame error")

// frameError is a pre-constructed sentinel so the reject path allocates
// nothing (asserted by TestParseHeaderZeroAlloc). The exported Err* vars below
// deliberately have this unexported type: callers only ever compare them with
// errors.Is, never construct or type-assert them.
type frameError struct {
	msg   string
	fatal bool
}

func (e *frameError) Error() string { return e.msg }

// Is reports whether target is ErrFatalFrame and this error is fatal, making
// errors.Is(err, ErrFatalFrame) the one classification test callers need.
func (e *frameError) Is(target error) bool { return target == ErrFatalFrame && e.fatal }

var (
	// ErrShortHeader: the buffer holds fewer than HeaderSize bytes. The
	// transport always reads exactly 64 bytes (io.ReadFull) before parsing, so
	// hitting this means a caller bug or a truncated stream — fatal either way.
	ErrShortHeader = &frameError{msg: "protocol: buffer shorter than the 64-byte header", fatal: true}

	// ErrBadMagic: the frame does not start with "KVB1" — a port scan, a TLS
	// handshake, or a desynchronized stream. Fatal.
	ErrBadMagic = &frameError{msg: `protocol: bad magic (want "KVB1")`, fatal: true}

	// ErrBadVersion: unknown header-layout generation. Fatal.
	ErrBadVersion = &frameError{msg: "protocol: unsupported header version", fatal: true}

	// ErrBadCRC: header_crc32c does not match bytes 0–59. The one error that
	// would otherwise desync framing (payload_len is untrustworthy). Fatal.
	ErrBadCRC = &frameError{msg: "protocol: header CRC32C mismatch", fatal: true}

	// ErrPayloadTooLarge: payload_len exceeds the caller's cap. Recoverable —
	// and unlike the fatal errors, the returned Header IS fully populated
	// (the CRC authenticated it before the cap was checked), so the transport
	// can skip exactly PayloadLen bytes and answer ERR_TOO_LARGE echoing
	// RequestID, without re-parsing raw bytes (PROTOCOL.md §4).
	ErrPayloadTooLarge = &frameError{msg: "protocol: payload_len exceeds cap", fatal: false}
)

// ParseHeader decodes and validates a frame header from b, in the order the
// spec fixes: magic, then version, then CRC, then the payload_len cap. Nothing
// after a failed check is trusted; in particular payload_len is only
// meaningful once the CRC has authenticated it. maxPayloadLen is the caller's
// negotiated (or pre-negotiation) frame cap — validating it HERE, before any
// caller allocates payload buffers, is the codec's contract.
//
// Returns: a zero Header with a fatal error (errors.Is ErrFatalFrame) when the
// bytes cannot be trusted at all; a FULLY POPULATED Header with
// ErrPayloadTooLarge when the header is authentic but the payload exceeds the
// cap (the caller skips PayloadLen bytes and stays in sync); a populated
// Header and nil error otherwise. Parsing allocates nothing on any path
// (given a heap/pooled buffer — see the Header doc note on stack buffers).
func ParseHeader(b []byte, maxPayloadLen uint32) (Header, error) {
	var h Header
	if len(b) < HeaderSize {
		return h, ErrShortHeader
	}
	if binary.LittleEndian.Uint32(b[magicOffset:]) != magicLE {
		return h, ErrBadMagic
	}
	if b[versionOffset] != Version1 {
		return h, ErrBadVersion
	}
	wireCRC := binary.LittleEndian.Uint32(b[crcOffset:])
	if wireCRC != HeaderCRC(b) {
		return h, ErrBadCRC
	}

	binary.LittleEndian.PutUint32(h.Magic[:], magicLE)
	h.Version = b[versionOffset]
	h.Opcode = Opcode(b[opcodeOffset])
	h.Flags = binary.LittleEndian.Uint16(b[flagsOffset:])
	h.NamespaceID = binary.LittleEndian.Uint32(b[namespaceOffset:])
	h.Credit = binary.LittleEndian.Uint32(b[creditOffset:])
	h.RequestID = binary.LittleEndian.Uint64(b[requestIDOffset:])
	copy(h.Key[:], b[keyOffset:keyOffset+32])
	h.PayloadLen = binary.LittleEndian.Uint32(b[payloadOffset:])
	h.CRC = wireCRC

	if h.PayloadLen > maxPayloadLen {
		return h, ErrPayloadTooLarge
	}
	return h, nil
}

// MarshalTo writes h into the first HeaderSize bytes of b, computing and
// storing the CRC32C over bytes 0–59 (the h.CRC field is ignored on input and
// not updated; the wire always carries the freshly computed value). The magic
// and version are written from the package constants, not from h, so a
// zero-valued Header still marshals a structurally valid frame. It panics if
// b is shorter than HeaderSize — callers own a correctly sized buffer,
// matching the transport's pooled-buffer contract (heap/pooled buffers also
// keep this zero-alloc; see the Header doc note on stack buffers).
func (h *Header) MarshalTo(b []byte) {
	_ = b[HeaderSize-1] //nolint:gosec // G602: panic on short buffers is the documented contract (TestMarshalToPanicsOnShortBuffer)

	binary.LittleEndian.PutUint32(b[magicOffset:], magicLE)
	b[versionOffset] = Version1
	b[opcodeOffset] = uint8(h.Opcode)
	binary.LittleEndian.PutUint16(b[flagsOffset:], h.Flags)
	binary.LittleEndian.PutUint32(b[namespaceOffset:], h.NamespaceID)
	binary.LittleEndian.PutUint32(b[creditOffset:], h.Credit)
	binary.LittleEndian.PutUint64(b[requestIDOffset:], h.RequestID)
	copy(b[keyOffset:], h.Key[:])
	binary.LittleEndian.PutUint32(b[payloadOffset:], h.PayloadLen)
	binary.LittleEndian.PutUint32(b[crcOffset:], HeaderCRC(b))
}
