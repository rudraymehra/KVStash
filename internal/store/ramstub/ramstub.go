// Package ramstub is a temporary in-heap block store for wire-path
// bring-up. It implements just enough of the store surface for the server to
// dispatch all eight verbs end to end; the real DRAM/NVMe/S3 tiers replace it
// later. Blob bytes live on the Go heap here (a disclosed rig property,
// recorded in docs/DESIGN.md) — the off-heap arena is the DRAM tier's job.
package ramstub

import (
	"encoding/json"
	"sync"

	"github.com/kvstash/kvblockd/internal/protocol"
)

// shardCount buckets keys across independently-locked maps so concurrent
// connections rarely contend. 256 keeps each shard small and is a power of two
// (cheap masking off the key's first byte).
const shardCount = 256

// blockKey is the (namespace, key) block identity — dedup never crosses a
// namespace (PROTOCOL.md tenancy rule).
type blockKey struct {
	ns  uint32
	key [32]byte
}

type entry struct {
	data []byte
	xxh3 uint64
}

type shard struct {
	mu sync.RWMutex
	m  map[blockKey]entry
}

// Store is the temporary sharded in-heap store.
type Store struct {
	shards [shardCount]shard
}

// New returns an empty store.
func New() *Store {
	s := &Store{}
	for i := range s.shards {
		s.shards[i].m = make(map[blockKey]entry)
	}
	return s
}

func (s *Store) shard(k blockKey) *shard { return &s.shards[k.key[0]] }

// ExistsPrefix reports the number of consecutive hits from position 0 (the
// scheduler's load-vs-recompute number) and, when withBitmap, a per-key status
// slice. Served entirely from memory; never blocks on I/O (§3.2).
func (s *Store) ExistsPrefix(ns uint32, keys [][32]byte, withBitmap bool) (nConsecutive uint32, perKey []protocol.Status) {
	if withBitmap {
		perKey = make([]protocol.Status, len(keys))
	}
	consecutiveDone := false
	for i, k := range keys {
		bk := blockKey{ns: ns, key: k}
		sh := s.shard(bk)
		sh.mu.RLock()
		_, ok := sh.m[bk]
		sh.mu.RUnlock()
		if ok && !consecutiveDone {
			nConsecutive++
		} else {
			consecutiveDone = true
		}
		if withBitmap {
			if ok {
				perKey[i] = protocol.StatusOK
			} else {
				perKey[i] = protocol.StatusNotFound
			}
		}
	}
	return nConsecutive, perKey
}

// Get returns a committed block's bytes and checksum. The returned slice
// aliases store memory: it is immutable (write-once) and stays alive while a
// caller holds it, so the transport may writev from it directly without a copy.
func (s *Store) Get(ns uint32, key [32]byte) (data []byte, xxh3 uint64, ok bool) {
	bk := blockKey{ns: ns, key: key}
	sh := s.shard(bk)
	sh.mu.RLock()
	e, ok := sh.m[bk]
	sh.mu.RUnlock()
	return e.data, e.xxh3, ok
}

// Put commits a fully-staged block. Write-once (§13): a second Put of an
// existing key returns OK_EXISTS if the checksum matches, or
// ERR_IMMUTABLE_CONFLICT if it differs (content-derived keys mean a mismatch
// is corruption — never overwrite). data must already be validated (length +
// xxh3) by the caller; the store copies it so the caller's staging buffer can
// be reused.
func (s *Store) Put(ns uint32, key [32]byte, data []byte, xxh3 uint64) protocol.Status {
	bk := blockKey{ns: ns, key: key}
	sh := s.shard(bk)
	sh.mu.Lock()
	defer sh.mu.Unlock()
	if existing, ok := sh.m[bk]; ok {
		if existing.xxh3 == xxh3 {
			return protocol.StatusOKExists
		}
		return protocol.StatusErrImmutableConflict
	}
	cp := make([]byte, len(data))
	copy(cp, data)
	sh.m[bk] = entry{data: cp, xxh3: xxh3}
	return protocol.StatusOK
}

// Contains reports whether a committed block exists (used by PUT BEGIN for the
// write-once idempotent-hit short-circuit).
func (s *Store) Contains(ns uint32, key [32]byte) bool {
	bk := blockKey{ns: ns, key: key}
	sh := s.shard(bk)
	sh.mu.RLock()
	_, ok := sh.m[bk]
	sh.mu.RUnlock()
	return ok
}

// Delete removes a block. force is accepted for the wire contract; ramstub has
// no leases/pins, so it always succeeds (real lease/pin gating comes with the tiers).
func (s *Store) Delete(ns uint32, key [32]byte, _ bool) protocol.Status {
	bk := blockKey{ns: ns, key: key}
	sh := s.shard(bk)
	sh.mu.Lock()
	_, ok := sh.m[bk]
	delete(sh.m, bk)
	sh.mu.Unlock()
	if !ok {
		return protocol.StatusNotFound
	}
	return protocol.StatusOK
}

// Stats returns a JSON stats document (§3.8). ramstub reports block/byte
// counts; the real per-tier/quota metrics arrive with the tiers.
func (s *Store) Stats() []byte {
	var blocks int
	var bytes uint64
	for i := range s.shards {
		s.shards[i].mu.RLock()
		blocks += len(s.shards[i].m)
		for _, e := range s.shards[i].m {
			bytes += uint64(len(e.data))
		}
		s.shards[i].mu.RUnlock()
	}
	doc := map[string]any{
		"schema":      1,
		"store":       "ramstub",
		"blocks":      blocks,
		"bytes":       bytes,
		"shard_count": shardCount,
		"note":        "temporary in-heap store; DRAM/NVMe/S3 tiers not yet wired",
	}
	b, err := json.Marshal(doc)
	if err != nil {
		return []byte(`{"schema":1,"store":"ramstub","error":"stats encode failed"}`)
	}
	return b
}
