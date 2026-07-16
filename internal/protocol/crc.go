package protocol

import "hash/crc32"

// castagnoli is the CRC32C table (iSCSI polynomial 0x1EDC6F41). Go's hash/crc32
// dispatches to hardware instructions (SSE4.2 CRC32 / ARMv8 CRC32C) for this
// table, so the per-frame cost is on the order of ten nanoseconds for 60 bytes.
var castagnoli = crc32.MakeTable(crc32.Castagnoli)

// HeaderCRC computes the CRC32C over the first 60 bytes of a frame header —
// everything up to but excluding the header_crc32c field itself
// (PROTOCOL.md §1). It panics if len(b) < 60: callers own a full header
// buffer by contract, and a short slice here is a bug upstream, not a wire
// condition. The explicit index check below enforces the contract on LENGTH —
// without it, a re-slice like buf[:10] of a 64-byte array would silently
// checksum bytes past its length (slice expressions extend to capacity).
func HeaderCRC(b []byte) uint32 {
	_ = b[crcOffset-1]                               //nolint:gosec // G602: enforces the length contract — panic on <60-byte input is documented (TestHeaderCRCPanicsOnShortBuffer)
	return crc32.Checksum(b[:crcOffset], castagnoli) //nolint:gosec // G602: panic on <60-byte input is the documented contract (TestHeaderCRCPanicsOnShortBuffer); the index check above enforces it on len, not cap
}
