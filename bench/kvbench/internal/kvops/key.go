package kvops

import (
	"encoding/binary"

	"lukechampine.com/blake3"
)

// KeyDerivation names the Go↔Python key contract recorded in every header.
const KeyDerivation = "kvbench-trace-v1"

// traceKeyDomain deliberately differs from the product wire-key domain
// ("kvblockd-cek-v1\x00" in pkg/client.WireKey): a bench trace key can
// never collide with a production content key by construction. The recipe
// otherwise mirrors WireKey exactly — BLAKE3-256 over the domain followed
// by u32-LE length-prefixed UTF-8 fields — so the Python replayer reuses
// its existing hashing code with one domain swap.
const traceKeyDomain = "kvbench-trace-v1\x00"

// TraceKey derives the key for a trace entity. Field conventions:
//
//	bailian:  TraceKey(traceName, decimalHashID)
//	mooncake: TraceKey(traceName, decimalHashID, decimalSubIdx)  subIdx ∈ [0,32)
func TraceKey(fields ...string) [32]byte {
	h := blake3.New(32, nil)
	_, _ = h.Write([]byte(traceKeyDomain))
	var lenBuf [4]byte
	for _, f := range fields {
		binary.LittleEndian.PutUint32(lenBuf[:], uint32(len(f))) //nolint:gosec // G115: field lengths are tiny
		_, _ = h.Write(lenBuf[:])
		_, _ = h.Write([]byte(f))
	}
	var out [32]byte
	copy(out[:], h.Sum(nil))
	return out
}
