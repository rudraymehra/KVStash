package eviction

import (
	"encoding/binary"
	"math/rand/v2"
	"sync"
	"sync/atomic"
)

// SampledLRU is the second policy — the proof that Policy is a real
// interface, not S3-FIFO's private shape. Redis-style: sample K entries,
// evict the least-recently-touched. It self-samples from ITS OWN table
// (fed by the same Touch events that update the store's LastAccess), so it
// needs no index-sampling surface and no store types.
type SampledLRU struct {
	seed uint64

	mu      sync.Mutex // COW rebuilds
	domains atomic.Pointer[map[uint32]*lruDomain]
}

// sampleK is the victim sample size per selection round (the classic
// approximation quality knob; 32 ≈ within a few percent of true LRU).
const sampleK = 32

// NewSampledLRU builds the policy.
func NewSampledLRU() *SampledLRU {
	p := &SampledLRU{seed: rand.Uint64()} //nolint:gosec // G404: grind-resistance seed, not crypto
	m := make(map[uint32]*lruDomain)
	p.domains.Store(&m)
	return p
}

func (p *SampledLRU) Name() string { return "sampled-lru" }

type lruDomain struct {
	table [tableShards]lruShard
	bytes atomic.Int64
}

type lruShard struct {
	mu sync.RWMutex
	m  map[Key]*lruEntry
}

type lruEntry struct {
	key  Key
	size int64
	last atomic.Int64 // unix nanos of the latest touch
}

func (p *SampledLRU) shardIdx(k Key) int {
	return int(((binary.LittleEndian.Uint64(k.Hash[8:16]) ^ p.seed) * 0x9E3779B97F4A7C15) >> 60)
}

func (p *SampledLRU) domain(ns uint32) *lruDomain {
	return (*p.domains.Load())[ns]
}

func (p *SampledLRU) ensureDomain(ns uint32) *lruDomain {
	if d := p.domain(ns); d != nil {
		return d
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	old := *p.domains.Load()
	if d := old[ns]; d != nil {
		return d
	}
	d := &lruDomain{}
	for i := range d.table {
		d.table[i].m = make(map[Key]*lruEntry)
	}
	next := make(map[uint32]*lruDomain, len(old)+1)
	for k, v := range old {
		next[k] = v
	}
	next[ns] = d
	p.domains.Store(&next)
	return d
}

// Admit records a resident block (no-op for keys already tracked).
func (p *SampledLRU) Admit(k Key, size int64, now int64) {
	if size <= 0 {
		return
	}
	d := p.ensureDomain(k.NS)
	e := &lruEntry{key: k, size: size}
	e.last.Store(now)
	ts := &d.table[p.shardIdx(k)]
	ts.mu.Lock()
	if _, exists := ts.m[k]; exists {
		ts.mu.Unlock()
		return
	}
	ts.m[k] = e
	ts.mu.Unlock()
	d.bytes.Add(size)
}

// Touch records an access: pointer load + sharded map read + atomic
// store. Zero allocations (the GET hot path).
func (p *SampledLRU) Touch(k Key, now int64) {
	d := p.domain(k.NS)
	if d == nil {
		return
	}
	ts := &d.table[p.shardIdx(k)]
	ts.mu.RLock()
	e := ts.m[k]
	ts.mu.RUnlock()
	if e != nil {
		e.last.Store(now)
	}
}

// Remove drops a key the store removed on its own. Unknown keys no-op.
func (p *SampledLRU) Remove(k Key) {
	d := p.domain(k.NS)
	if d == nil {
		return
	}
	ts := &d.table[p.shardIdx(k)]
	ts.mu.Lock()
	e := ts.m[k]
	delete(ts.m, k)
	ts.mu.Unlock()
	if e != nil {
		d.bytes.Add(-e.size)
	}
}

// Victims samples K entries per round (random shard start + Go's runtime-
// randomized map iteration — the Redis trick) and evicts the stalest,
// repeating until need is covered or the domain is empty.
func (p *SampledLRU) Victims(ns uint32, need int64, _ int64, dst []Candidate) []Candidate {
	d := p.domain(ns)
	if d == nil || need <= 0 {
		return dst
	}
	var got int64
	for got < need {
		victim := p.sampleStalest(d)
		if victim == nil {
			break
		}
		ts := &d.table[p.shardIdx(victim.key)]
		ts.mu.Lock()
		// Re-check: a concurrent Remove may have raced our sample.
		if ts.m[victim.key] != victim {
			ts.mu.Unlock()
			continue
		}
		delete(ts.m, victim.key)
		ts.mu.Unlock()
		d.bytes.Add(-victim.size)
		dst = append(dst, Candidate{Key: victim.key, Size: victim.size})
		got += victim.size
	}
	return dst
}

// sampleStalest scans up to sampleK entries starting at a random shard and
// returns the least-recently-touched, or nil if the domain is empty.
func (p *SampledLRU) sampleStalest(d *lruDomain) *lruEntry {
	var (
		best     *lruEntry
		bestLast int64
		seen     int
	)
	start := int(rand.Uint32()) % tableShards //nolint:gosec // sampling, not crypto
	for i := 0; i < tableShards && seen < sampleK; i++ {
		ts := &d.table[(start+i)%tableShards]
		ts.mu.RLock()
		for _, e := range ts.m { // map iteration order is runtime-randomized
			if last := e.last.Load(); best == nil || last < bestLast {
				best, bestLast = e, last
			}
			if seen++; seen >= sampleK {
				break
			}
		}
		ts.mu.RUnlock()
	}
	return best
}

// Usage reports each domain's resident bytes.
func (p *SampledLRU) Usage(dst []DomainUsage) []DomainUsage {
	for ns, d := range *p.domains.Load() {
		if b := d.bytes.Load(); b > 0 {
			dst = append(dst, DomainUsage{NS: ns, Bytes: b})
		}
	}
	return dst
}
