package client

import (
	"encoding/binary"

	"lukechampine.com/blake3"
)

// wireKeyDomain domain-separates the CacheEngineKey→wire-key hash so a wire
// key can never collide with any other BLAKE3 use in the system.
var wireKeyDomain = []byte("kvblockd-cek-v1\x00")

// WireKey maps an ordered set of string fields to the 32-byte opaque wire key
// the server stores blind (T3 — the server never derives or interprets it).
// The encoding is the exact mirror of Python kvblockd.hashing.wire_key: a
// domain prefix followed by each field as u32-LE length + UTF-8 bytes, then
// BLAKE3-256. Length-prefixing (not separator-joining) is load-bearing —
// model names legally contain '/' and '@', so a joined form would be
// ambiguous and let distinct keys collide. hash_chain.json is the shared
// Go↔Python oracle (see hashchain_test.go).
//
// The adapter passes the CacheEngineKey in field order:
// (fmt, model_name, str(world_size), str(worker_id), str(chunk_hash)).
func WireKey(fields ...string) [32]byte {
	h := blake3.New(32, nil)
	// hash.Hash.Write never returns an error (documented contract); the
	// blank assignments satisfy errcheck without noise.
	_, _ = h.Write(wireKeyDomain)
	var lenbuf [4]byte
	for _, f := range fields {
		binary.LittleEndian.PutUint32(lenbuf[:], uint32(len(f))) //nolint:gosec // field lengths are small
		_, _ = h.Write(lenbuf[:])
		_, _ = h.Write([]byte(f))
	}
	var out [32]byte
	copy(out[:], h.Sum(nil))
	return out
}
