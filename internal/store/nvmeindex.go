package store

import (
	"encoding/binary"
	"hash/maphash"
	"sync"
	"sync/atomic"

	"github.com/kvstash/kvblockd/internal/protocol"
	"github.com/kvstash/kvblockd/internal/store/dram"
	"github.com/kvstash/kvblockd/internal/store/nvme"
)

// nvmeRef is one NVMe-resident block's metadata — the same lifecycle
// conventions as dram.BlockRef (lease/TTL/access are atomics, PinFlags
// mutates only under the owning shard's write lock), with a device Loc in
// place of an arena extent. There is no refcount: device bytes are pinned
// during a read by the segment's read-hold (nvme.Volume), not per block.
type nvmeRef struct {
	Loc  nvme.Loc
	Len  uint32
	XXH3 uint64
	// S3 marks that a byte-identical copy of this block's SEGMENT lives on
	// the cold tier (spill-ack). Loc addresses both: the local file while
	// it exists, the S3 object after reclaim retires the local copy.
	S3 atomic.Bool
	// S3Only marks the retire-flip: the local segment is gone and the
	// tenant charge moved NVMe→S3 — a removal must refund the S3 side.
	// Set only inside s3FlipRetired's shard-locked walk, so it linearizes
	// against deleteIf (a racing DELETE either sees the flip or removes
	// the entry before the flip can charge it).
	S3Only atomic.Bool

	LeaseUntil atomic.Int64
	TTLUntil   atomic.Int64
	// LastAccess is the last NVMe GET-hit (unix nanos), 0 until the first
	// hit after demotion — the promotion window tracker, deliberately NOT
	// seeded from the DRAM ref (a fresh demotion must take two real NVMe
	// hits to earn promotion).
	LastAccess atomic.Int64
	PinFlags   uint8 // guarded by the nvme index shard lock
}

func (r *nvmeRef) leased(now int64) bool { return r.LeaseUntil.Load() > now }
func (r *nvmeRef) pinned() bool          { return r.PinFlags != 0 }
func (r *nvmeRef) hardPinned() bool      { return r.PinFlags&nvPinHardBit != 0 }

const (
	nvPinSoftBit = 1 << 0
	nvPinHardBit = 1 << 1
)

const nvmeIndexShards = 256

type nvmeShard struct {
	mu sync.RWMutex
	m  map[dram.Key]*nvmeRef
}

// nvmeIndex mirrors the dram Index's sharding (seeded maphash — the same
// hash-flood posture). LOCK ORDER: a dram shard lock may nest an nvme shard
// lock (CompleteDemotion's publish); never the reverse.
type nvmeIndex struct {
	seed   maphash.Seed
	shards [nvmeIndexShards]nvmeShard
}

func newNvmeIndex() *nvmeIndex {
	idx := &nvmeIndex{seed: maphash.MakeSeed()}
	for i := range idx.shards {
		idx.shards[i].m = make(map[dram.Key]*nvmeRef)
	}
	return idx
}

func (idx *nvmeIndex) shardFor(k dram.Key) *nvmeShard {
	var h maphash.Hash
	h.SetSeed(idx.seed)
	var ns [4]byte
	binary.LittleEndian.PutUint32(ns[:], k.NS)
	_, _ = h.Write(ns[:])
	_, _ = h.Write(k.Hash[:])
	return &idx.shards[h.Sum64()%nvmeIndexShards]
}

func (idx *nvmeIndex) get(k dram.Key) *nvmeRef {
	s := idx.shardFor(k)
	s.mu.RLock()
	ref := s.m[k]
	s.mu.RUnlock()
	return ref
}

func (idx *nvmeIndex) contains(k dram.Key) bool { return idx.get(k) != nil }

// put inserts or overwrites (a re-demotion after promotion refreshes the
// Loc — later write wins, matching recovery's later-segID-wins rule).
func (idx *nvmeIndex) put(k dram.Key, ref *nvmeRef) {
	s := idx.shardFor(k)
	s.mu.Lock()
	s.m[k] = ref
	s.mu.Unlock()
}

// deleteIf removes the entry when gate answers StatusOK — same single-hold
// discipline as dram's Index.DeleteIf. Returns the removed ref (nil when
// absent or refused) and the gate's status.
func (idx *nvmeIndex) deleteIf(k dram.Key, gate func(ref *nvmeRef) protocol.Status) (*nvmeRef, protocol.Status) {
	s := idx.shardFor(k)
	s.mu.Lock()
	defer s.mu.Unlock()
	ref, ok := s.m[k]
	if !ok {
		return nil, protocol.StatusNotFound
	}
	if st := gate(ref); st != protocol.StatusOK {
		return nil, st
	}
	delete(s.m, k)
	return ref, protocol.StatusOK
}

// withShardLock runs fn under the shard write lock (PinFlags mutation);
// fn receives nil when the key is absent.
func (idx *nvmeIndex) withShardLock(k dram.Key, fn func(ref *nvmeRef)) {
	s := idx.shardFor(k)
	s.mu.Lock()
	fn(s.m[k])
	s.mu.Unlock()
}

// rangeAll visits every entry (shard RLock held during the visit — keep fn
// cheap; stop early by returning false).
func (idx *nvmeIndex) rangeAll(fn func(k dram.Key, ref *nvmeRef) bool) {
	for i := range idx.shards {
		s := &idx.shards[i]
		s.mu.RLock()
		for k, ref := range s.m {
			if !fn(k, ref) {
				s.mu.RUnlock()
				return
			}
		}
		s.mu.RUnlock()
	}
}

// stats sums blocks and bytes.
func (idx *nvmeIndex) stats() (blocks int, bytes int64) {
	idx.rangeAll(func(_ dram.Key, ref *nvmeRef) bool {
		blocks++
		bytes += int64(ref.Len)
		return true
	})
	return blocks, bytes
}
