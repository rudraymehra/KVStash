package dram

import (
	"hash/maphash"
	"sync"

	"github.com/kvstash/kvblockd/internal/protocol"
)

// Key is the block identity: (namespace, content hash). Pointer-free and
// comparable — dedup never crosses a namespace (PROTOCOL.md tenancy rule),
// and [32]byte keys keep the GC out of the map (no interior pointers).
type Key struct {
	NS   uint32
	Hash [32]byte
}

const indexShards = 256

type indexShard struct {
	mu sync.RWMutex
	m  map[Key]*BlockRef
}

// Index is the 256-shard DRAM block index. Shard selection hashes the content
// key through hash/maphash with a PER-PROCESS RANDOM SEED, so a client who
// controls key bytes cannot aim every key at one shard's lock (hash-flood
// defense — the InfiniStore #200 lesson). Tests must therefore never assert
// WHICH shard a key lands in — only aggregate behavior.
type Index struct {
	seed   maphash.Seed
	shards [indexShards]indexShard
}

// NewIndex returns an empty index with a fresh random shard seed.
func NewIndex() *Index { return newIndexWithSeed(maphash.MakeSeed()) }

// newIndexWithSeed pins the seed — for tests that need a reproducible shard
// distribution. The public constructor always randomizes.
func newIndexWithSeed(seed maphash.Seed) *Index {
	idx := &Index{seed: seed}
	for i := range idx.shards {
		idx.shards[i].m = make(map[Key]*BlockRef)
	}
	return idx
}

// shardOf picks the shard by seeded maphash over (ns, key).
func (idx *Index) shardOf(k Key) *indexShard {
	var h maphash.Hash
	h.SetSeed(idx.seed)
	var ns [4]byte
	ns[0], ns[1], ns[2], ns[3] = byte(k.NS), byte(k.NS>>8), byte(k.NS>>16), byte(k.NS>>24) //nolint:gosec // G115: intentional little-endian byte split
	_, _ = h.Write(ns[:])
	_, _ = h.Write(k.Hash[:])
	return &idx.shards[byte(h.Sum64())] //nolint:gosec // G115: intentional low-byte shard pick (mod 256)
}

// Get returns the published BlockRef for k. The ref is returned WITHOUT
// acquiring it — the caller (store.Get) Acquires under its own protocol.
// Expired-TTL blocks are still returned: expiry means eviction-ELIGIBLE, not
// invisible (lazy expiry; the evictor reclaims).
func (idx *Index) Get(k Key) (*BlockRef, bool) {
	sh := idx.shardOf(k)
	sh.mu.RLock()
	ref, ok := sh.m[k]
	sh.mu.RUnlock()
	return ref, ok
}

// Put inserts ref for k unless a block is already published there; it returns
// the existing ref (and inserted=false) on a lost race — the caller decides
// OK_EXISTS vs ERR_IMMUTABLE_CONFLICT and must free its extent.
func (idx *Index) Put(k Key, ref *BlockRef) (existing *BlockRef, inserted bool) {
	sh := idx.shardOf(k)
	sh.mu.Lock()
	defer sh.mu.Unlock()
	if cur, ok := sh.m[k]; ok {
		return cur, false
	}
	sh.m[k] = ref
	return ref, true
}

// Delete removes k from the index and returns the unpublished ref. The caller
// drops the index reference (Release) and frees on zero — outside the shard
// lock.
func (idx *Index) Delete(k Key) (*BlockRef, bool) {
	sh := idx.shardOf(k)
	sh.mu.Lock()
	ref, ok := sh.m[k]
	if ok {
		delete(sh.m, k)
	}
	sh.mu.Unlock()
	return ref, ok
}

// WithShardLock runs fn while holding k's shard WRITE lock, passing the
// current ref (nil if absent). It is the mutation envelope for PinFlags and
// the lifecycle verbs (PinFlags is non-atomic BY the shard-lock convention).
func (idx *Index) WithShardLock(k Key, fn func(ref *BlockRef)) {
	sh := idx.shardOf(k)
	sh.mu.Lock()
	fn(sh.m[k])
	sh.mu.Unlock()
}

// DeleteIf gates and removes k under ONE shard-lock hold: gate inspects the
// ref (PinFlags safe — lock held); a non-OK gate status aborts the removal.
// Returns the removed ref (nil if absent or gated) and the status. The
// caller drops the index reference OUTSIDE the lock: the global order
// (shard lock first, then allocMu) would PERMIT freeing inside, but a
// reader's Release also frees without any shard lock, so keeping every
// Free outside the shard lock keeps one story for the Week-4 evictor.
func (idx *Index) DeleteIf(k Key, gate func(*BlockRef) protocol.Status) (*BlockRef, protocol.Status) {
	sh := idx.shardOf(k)
	sh.mu.Lock()
	defer sh.mu.Unlock()
	ref, ok := sh.m[k]
	if !ok {
		return nil, protocol.StatusNotFound
	}
	if st := gate(ref); st != protocol.StatusOK {
		return nil, st
	}
	delete(sh.m, k)
	return ref, protocol.StatusOK
}

// Range calls fn for every published ref until fn returns false. Takes each
// shard's read lock in turn; fn must not call back into the index.
func (idx *Index) Range(fn func(k Key, ref *BlockRef) bool) {
	for i := range idx.shards {
		sh := &idx.shards[i]
		sh.mu.RLock()
		for k, ref := range sh.m {
			if !fn(k, ref) {
				sh.mu.RUnlock()
				return
			}
		}
		sh.mu.RUnlock()
	}
}

// ExistsPrefix mirrors the wire contract (§3.2): the count of consecutive
// hits from position 0, and per-key statuses only when the bitmap was
// negotiated. Without the bitmap the walk stops at the first miss (nothing
// left to compute); with it every key is probed. Expired blocks count as
// present (lazy expiry).
func (idx *Index) ExistsPrefix(ns uint32, keys [][32]byte, withBitmap bool) (nConsecutive uint32, perKey []bool) {
	if withBitmap {
		perKey = make([]bool, len(keys))
	}
	consecutiveDone := false
	for i, kh := range keys {
		if consecutiveDone && !withBitmap {
			break
		}
		_, ok := idx.Get(Key{NS: ns, Hash: kh})
		if ok && !consecutiveDone {
			nConsecutive++
		} else if !ok {
			consecutiveDone = true
		}
		if withBitmap {
			perKey[i] = ok
		}
	}
	return nConsecutive, perKey
}
