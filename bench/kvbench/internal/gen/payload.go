// Package gen produces kvbench's deterministic workload inputs: payloads,
// key spaces, skew distributions, and the sweep grid. Everything derives
// from (seed, index) so two runs with one seed are byte-identical, and a
// verify pass can regenerate any blob from its key alone.
package gen

import (
	"encoding/binary"

	"github.com/zeebo/xxh3"
)

// Payload spec "kvbench-payload-v1" — the cross-language contract (the
// future Python replayer must reproduce it bit-for-bit; golden vectors live
// in testdata/payload_golden.json):
//
//	P       = "kvbench-payload-v1" (18 ASCII bytes) || u64le(seed) || key[32]
//	chunk_j = XXH3_128(P || u64le(j)) as the CANONICAL BIG-ENDIAN 16 bytes
//	blob    = (chunk_0 || chunk_1 || …)[:blob_bytes]
//
// Each 16-byte chunk is an independent hash — incompressible by
// construction (compression can't flatter any store), fast (one XXH3_128
// per 16 bytes), and regenerable from (seed, key) alone so `kvbench verify`
// byte-compares any GET against a fresh derivation.
const payloadDomain = "kvbench-payload-v1"

// FillPayload writes the deterministic blob for (seed, key) into dst
// (len(dst) = the cell's blob_bytes). Zero heap allocations.
func FillPayload(dst []byte, seed uint64, key [32]byte) {
	var scratch [len(payloadDomain) + 8 + 32 + 8]byte
	copy(scratch[:], payloadDomain)
	binary.LittleEndian.PutUint64(scratch[len(payloadDomain):], seed)
	copy(scratch[len(payloadDomain)+8:], key[:])
	ctr := scratch[len(payloadDomain)+8+32:]

	var j uint64
	for off := 0; off < len(dst); off += 16 {
		binary.LittleEndian.PutUint64(ctr, j)
		sum := xxh3.Hash128(scratch[:]).Bytes() // canonical big-endian 16 bytes
		copy(dst[off:], sum[:])                 // copy truncates at the tail
		j++
	}
}

// VerifyPayloadLen is VerifyPayload with an EXPECTED length: a short read
// (a torn block returning fewer bytes than the cell's blob size) is
// corruption, even if every returned byte matches. The ladder caught the
// bare VerifyPayload passing a 4 KiB prefix of a 462 KiB blob.
func VerifyPayloadLen(got []byte, want int, seed uint64, key [32]byte) bool {
	if len(got) != want {
		return false
	}
	return VerifyPayload(got, seed, key)
}

// VerifyPayload reports whether got matches the derived blob for
// (seed, key) over its OWN length — the byte-content oracle. Callers that
// know the expected length should use VerifyPayloadLen so a short read is
// caught. Allocation-light: compares in 16-byte strides against fresh
// chunks.
func VerifyPayload(got []byte, seed uint64, key [32]byte) bool {
	var scratch [len(payloadDomain) + 8 + 32 + 8]byte
	copy(scratch[:], payloadDomain)
	binary.LittleEndian.PutUint64(scratch[len(payloadDomain):], seed)
	copy(scratch[len(payloadDomain)+8:], key[:])
	ctr := scratch[len(payloadDomain)+8+32:]

	var j uint64
	for off := 0; off < len(got); off += 16 {
		binary.LittleEndian.PutUint64(ctr, j)
		sum := xxh3.Hash128(scratch[:]).Bytes()
		end := off + 16
		if end > len(got) {
			end = len(got)
		}
		for i := off; i < end; i++ {
			if got[i] != sum[i-off] {
				return false
			}
		}
		j++
	}
	return true
}
