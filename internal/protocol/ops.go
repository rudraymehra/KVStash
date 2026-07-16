package protocol

// Opcode is the verb family carried in header byte 5 (PROTOCOL.md §3).
// Responses reuse the request opcode with FlagResp set.
type Opcode uint8

const (
	// OpNop is keepalive + unsolicited credit grants (header credit field).
	// request_id must be 0. Never responded to.
	OpNop Opcode = 0x00

	OpHello       Opcode = 0x01 // auth + negotiation; MUST be first on a connection
	OpBatchExists Opcode = 0x02 // scheduler-blocking probe; DRAM index only
	OpBatchGet    Opcode = 0x03 // descriptor table + concatenated payloads
	OpPutStream   Opcode = 0x04 // the only per-key verb; sub-ops BEGIN/CHUNK/COMMIT/ABORT
	OpTouchLease  Opcode = 0x05 // sub-ops TOUCH/LEASE/RELEASE
	OpPin         Opcode = 0x06 // sub-ops PIN_SOFT/PIN_HARD/UNPIN
	OpDelete      Opcode = 0x07 // logical delete; lease/pin protected
	OpStats       Opcode = 0x08 // JSON stats document; cold path
)

// Flag bits, header bytes 6–7 (PROTOCOL.md §2). Bits 4–7 carry the sub-op;
// bits 8–15 are reserved (zero on send, ignored on receive).
const (
	FlagResp  uint16 = 0x0001 // frame is a response
	FlagMore  uint16 = 0x0002 // more response frames follow for this request_id
	FlagFatal uint16 = 0x0004 // sender closes the connection after this frame
	FlagForce uint16 = 0x0008 // DELETE only: override lease/soft-pin protection

	subOpShift = 4
	subOpMask  = 0xF
)

// SubOp extracts the 4-bit sub-operation selector from a flags field.
func SubOp(flags uint16) uint8 {
	return uint8(flags >> subOpShift & subOpMask)
}

// WithSubOp returns flags with the 4-bit sub-op field set to s.
// Values above 15 do not fit in the field; callers pass named constants only.
func WithSubOp(flags uint16, s uint8) uint16 {
	return flags&^(subOpMask<<subOpShift) | uint16(s&subOpMask)<<subOpShift
}

// PUT_STREAM sub-ops (PROTOCOL.md §3.4, §5).
const (
	PutBegin  uint8 = 0
	PutChunk  uint8 = 1
	PutCommit uint8 = 2
	PutAbort  uint8 = 3
)

// TOUCH_LEASE sub-ops (PROTOCOL.md §3.5).
const (
	TouchRecency uint8 = 0 // TOUCH: recency bump + TTL extend; metadata-only
	LeaseGrant   uint8 = 1 // LEASE: grant/extend eviction protection
	LeaseRelease uint8 = 2 // RELEASE: drop the lease early
)

// PIN sub-ops (PROTOCOL.md §3.6).
const (
	PinSoft uint8 = 0
	PinHard uint8 = 1
	Unpin   uint8 = 2
)

// Status is the per-batch (response preamble) and per-key (descriptor /
// status-byte array) outcome code (PROTOCOL.md §9).
type Status uint8

const (
	StatusOK       Status = 0x00
	StatusOKExists Status = 0x01 // write-once idempotent hit; a success

	StatusNotFound Status = 0x10
	StatusEvicted  Status = 0x11 // observability nicety; treat as NOT_FOUND

	StatusErrAuthRequired     Status = 0x20 // non-HELLO before HELLO (F_FATAL)
	StatusErrAuthFailed       Status = 0x21 // bad token (F_FATAL)
	StatusErrNamespaceUnknown Status = 0x22 // (F_FATAL at HELLO)
	StatusErrForbidden        Status = 0x23 // token lacks op permission

	StatusErrQuotaBytes    Status = 0x30
	StatusErrPinQuota      Status = 0x31
	StatusErrTooLarge      Status = 0x32 // blob/frame over negotiated cap
	StatusErrBatchTooLarge Status = 0x33 // n_keys over negotiated cap
	StatusErrBusy          Status = 0x34 // backpressure; retry with backoff

	StatusErrChecksum          Status = 0x40 // COMMIT xxh3_64 mismatch
	StatusErrShortStream       Status = 0x41 // COMMIT with bytes != total_len
	StatusErrStaleStream       Status = 0x42 // COMMIT/ABORT on timed-out/tombstoned stream
	StatusErrImmutableConflict Status = 0x43 // same key, different xxh3_64 — corruption alarm
	StatusErrLeased            Status = 0x44 // DELETE blocked by live lease (F_FORCE overrides)
	StatusErrPinned            Status = 0x45 // DELETE blocked by pin (F_FORCE overrides soft only)

	StatusErrUnsupported Status = 0x50 // unknown opcode/sub-op; frame skipped
	StatusErrMalformed   Status = 0x51 // body unparseable; frame skipped
	StatusErrInternal    Status = 0x60 // server fault; MAY be retried

	StatusFatalProtocol Status = 0xF0 // header CRC/magic/version violation; always with F_FATAL
)

// String returns the PROTOCOL.md §9 name for s, for logs and test failures.
func (s Status) String() string {
	switch s {
	case StatusOK:
		return "OK"
	case StatusOKExists:
		return "OK_EXISTS"
	case StatusNotFound:
		return "NOT_FOUND"
	case StatusEvicted:
		return "EVICTED"
	case StatusErrAuthRequired:
		return "ERR_AUTH_REQUIRED"
	case StatusErrAuthFailed:
		return "ERR_AUTH_FAILED"
	case StatusErrNamespaceUnknown:
		return "ERR_NAMESPACE_UNKNOWN"
	case StatusErrForbidden:
		return "ERR_FORBIDDEN"
	case StatusErrQuotaBytes:
		return "ERR_QUOTA_BYTES"
	case StatusErrPinQuota:
		return "ERR_PIN_QUOTA"
	case StatusErrTooLarge:
		return "ERR_TOO_LARGE"
	case StatusErrBatchTooLarge:
		return "ERR_BATCH_TOO_LARGE"
	case StatusErrBusy:
		return "ERR_BUSY"
	case StatusErrChecksum:
		return "ERR_CHECKSUM"
	case StatusErrShortStream:
		return "ERR_SHORT_STREAM"
	case StatusErrStaleStream:
		return "ERR_STALE_STREAM"
	case StatusErrImmutableConflict:
		return "ERR_IMMUTABLE_CONFLICT"
	case StatusErrLeased:
		return "ERR_LEASED"
	case StatusErrPinned:
		return "ERR_PINNED"
	case StatusErrUnsupported:
		return "ERR_UNSUPPORTED"
	case StatusErrMalformed:
		return "ERR_MALFORMED"
	case StatusErrInternal:
		return "ERR_INTERNAL"
	case StatusFatalProtocol:
		return "FATAL_PROTOCOL"
	default:
		return "UNKNOWN_STATUS"
	}
}

// Feature bits, negotiated as the intersection at HELLO (PROTOCOL.md §10).
const (
	FeatOOO             uint64 = 1 << 0 // out-of-order responses
	FeatExistsBitmap    uint64 = 1 << 1 // per-key status bytes in BATCH_EXISTS
	FeatPayloadCRC32C   uint64 = 1 << 2 // reserved; not implemented in v1
	FeatCreditSymmetric uint64 = 1 << 3 // reserved; not implemented in v1
	FeatTLSUpgrade      uint64 = 1 << 4 // reserved
)

// Negotiated-limit defaults and the floors a server MUST accept
// (PROTOCOL.md §4).
const (
	DefaultMaxBatchKeys = 512
	FloorMaxBatchKeys   = 128

	DefaultMaxFrameLen = 256 << 20 // 256 MiB
	FloorMaxFrameLen   = 16 << 20  // 16 MiB — the coalescing knee observed in prior art

	DefaultMaxBlobLen = 32 << 20 // 32 MiB
	FloorMaxBlobLen   = 4 << 20  // 4 MiB

	DefaultInitialCredit = 128 << 20 // 128 MiB
	FloorInitialCredit   = 16 << 20  // 16 MiB

	DefaultLeaseMS         = 5_000  // read lease auto-granted by BATCH_GET
	MaxLeaseMS             = 60_000 // LEASE grants clamp here
	DefaultStreamTimeoutMS = 30_000 // PUT_STREAM inactivity reaper; floor 5s
)
