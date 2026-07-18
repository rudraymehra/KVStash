package gen

import "encoding/binary"

// Keyspace derives sweep keys: deterministic, seed-scoped, and structurally
// disjoint from trace keys (BLAKE3 "kvbench-trace-v1" domain), soak keys,
// and product wire keys — the trailing "kvbench1" tag plus the seed word
// make collisions with any other key family impossible by construction.
type Keyspace struct {
	Seed uint64
}

// Key returns the i'th sweep key.
func (ks Keyspace) Key(i int) [32]byte {
	var k [32]byte
	binary.LittleEndian.PutUint64(k[0:8], uint64(i)) //nolint:gosec // G115: pool indices are small non-negative
	binary.LittleEndian.PutUint64(k[8:16], ks.Seed)
	copy(k[24:32], "kvbench1")
	return k
}
