package eviction

import (
	"encoding/binary"
	"math/rand/v2"
	"sync"
	"sync/atomic"
)

// S3FIFO is the default policy (Yang et al., "FIFO queues are all you
// need for cache eviction", SOSP'23 shape): per-tenant domains, each a
// small probationary FIFO (~10% of resident bytes), a main FIFO, and a
// ghost ring of small-evictee fingerprints. One-hit wonders die in small
// without polluting main; a ghost hit on re-insert routes straight to main.
//
// Concurrency layout (all leaves of the daemon lock graph):
//   - domains: copy-on-write map behind an atomic.Pointer — Touch reads it
//     with a plain load; rebuilds (new tenant) happen under mu and are rare.
//   - per-domain table: 16-way sharded map[Key]*entry — Touch's only
//     structure (RLock + map read + atomic freq CAS: zero allocations).
//   - per-domain qmu: guards the queues, byte counters, and ghost — taken
//     only by Admit/Remove/Victims (Put/Delete/pressure cadence, never GET).
//     Internal order inside a domain: qmu → tableShard.mu (fixed, and Touch
//     takes only tableShard.mu, so no cycle).
type S3FIFO struct {
	seed     uint64
	ghostMax int // per-domain ghost ring ceiling (0 = floor only)

	mu      sync.Mutex // guards domains rebuilds (COW)
	domains atomic.Pointer[map[uint32]*s3Domain]
}

// NewS3FIFO builds the policy. ghostMax caps each domain's ghost ring; the
// ring starts at the 1024 floor and grows with the main queue's entry-count
// high-watermark, never past ghostMax (see maybeGrowGhost).
func NewS3FIFO(ghostMax int) *S3FIFO {
	p := &S3FIFO{seed: rand.Uint64(), ghostMax: ghostMax} //nolint:gosec // G404: grind-resistance seed, not crypto
	m := make(map[uint32]*s3Domain)
	p.domains.Store(&m)
	return p
}

func (p *S3FIFO) Name() string { return "s3fifo" }

const tableShards = 16

type s3Domain struct {
	table [tableShards]tableShard

	qmu        sync.Mutex
	small      fifoQ
	main       fifoQ
	smallBytes int64
	mainBytes  int64
	mainCount  int
	mainHi     int // main entry-count high-watermark → ghost sizing
	ghost      ghostRing
}

type tableShard struct {
	mu sync.RWMutex
	m  map[Key]*entry
}

// Queue states. qUnqueued MUST be the zero value: Admit publishes the entry
// in the table BEFORE taking qmu to queue it, and a Remove racing into that
// window must see "not queued yet" — with qSmall as the zero value it would
// unlink a never-linked entry and corrupt the FIFO (the reproduced evictor
// panic). The handshake: Remove tombstones an unqueued entry (qDead) instead
// of unlinking; Admit's queue step sees qDead and drops the entry.
const (
	qUnqueued = iota
	qSmall
	qMain
	qDead
)

type entry struct {
	key   Key
	size  int64
	prev  *entry // intrusive links + where: guarded by domain.qmu
	next  *entry
	where uint8
	freq  atomic.Int32 // 0..3 (2-bit semantics); Touch's only write
}

// fingerprint mixes the key's leading hash bytes with the policy seed
// (the ghost ring's identity).
func (p *S3FIFO) fingerprint(k Key) uint64 {
	return binary.LittleEndian.Uint64(k.Hash[0:8]) ^ p.seed ^ (uint64(k.NS) * 0x9E3779B97F4A7C15)
}

// shardIdx picks a table shard from the key's own hash bytes (already
// content-hash quality) mixed with the seed.
func (p *S3FIFO) shardIdx(k Key) int {
	return int(((binary.LittleEndian.Uint64(k.Hash[8:16]) ^ p.seed) * 0x9E3779B97F4A7C15) >> 60)
}

// domain returns the tenant's domain, or nil (Touch/Remove on unknown ns).
func (p *S3FIFO) domain(ns uint32) *s3Domain {
	return (*p.domains.Load())[ns]
}

// ensureDomain returns the tenant's domain, building it via COW on first
// sight (namespaces number a handful and appear once each).
func (p *S3FIFO) ensureDomain(ns uint32) *s3Domain {
	if d := p.domain(ns); d != nil {
		return d
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	old := *p.domains.Load()
	if d := old[ns]; d != nil { // lost the build race
		return d
	}
	d := &s3Domain{}
	for i := range d.table {
		d.table[i].m = make(map[Key]*entry)
	}
	next := make(map[uint32]*s3Domain, len(old)+1)
	for k, v := range old {
		next[k] = v
	}
	next[ns] = d
	p.domains.Store(&next)
	return d
}

// Touch records an access: pointer load + sharded map read + atomic freq
// bump. Zero allocations (the GET hot path).
func (p *S3FIFO) Touch(k Key, _ int64) {
	d := p.domain(k.NS)
	if d == nil {
		return
	}
	ts := &d.table[p.shardIdx(k)]
	ts.mu.RLock()
	e := ts.m[k]
	ts.mu.RUnlock()
	if e == nil {
		return
	}
	if f := e.freq.Load(); f < 3 {
		e.freq.CompareAndSwap(f, f+1) // lossy under contention — freq is a heuristic
	}
}

// Admit records a resident block: ghost hit → main (it was evicted once
// before proving itself — the scan-resistance route), miss → small. A key
// the policy already tracks is a no-op (defensive: double-Admit must not
// corrupt the queues).
func (p *S3FIFO) Admit(k Key, size int64, _ int64) {
	if size <= 0 {
		return
	}
	d := p.ensureDomain(k.NS)
	ts := &d.table[p.shardIdx(k)]

	e := &entry{key: k, size: size}
	ts.mu.Lock()
	if _, exists := ts.m[k]; exists {
		ts.mu.Unlock()
		return
	}
	ts.m[k] = e
	ts.mu.Unlock()

	d.qmu.Lock()
	if e.where == qDead {
		// A Remove raced into the publish window and tombstoned us (it
		// already took the table entry) — the block is gone; never queue.
		d.qmu.Unlock()
		return
	}
	if d.ghost.contains(p.fingerprint(k)) {
		e.where = qMain
		d.main.pushTail(e)
		d.mainBytes += size
		d.mainCount++
		if d.mainCount > d.mainHi {
			d.mainHi = d.mainCount
			p.maybeGrowGhost(d)
		}
	} else {
		e.where = qSmall
		d.small.pushTail(e)
		d.smallBytes += size
	}
	d.qmu.Unlock()
}

// maybeGrowGhost tracks the main queue's entry-count high-watermark and
// grows the ghost ring toward it (power of two, capped at ghostMax).
// ghostMax==0 really does mean "floor only" — no growth, matching the
// factory doc; callers wanting scale pass the arena-derived ceiling.
// Caller holds qmu.
func (p *S3FIFO) maybeGrowGhost(d *s3Domain) {
	if p.ghostMax <= 0 {
		return // floor only
	}
	want := d.mainHi
	if want > p.ghostMax {
		want = p.ghostMax
	}
	if want <= d.ghost.capacity() {
		return
	}
	capacity := ghostFloor
	for capacity < want {
		capacity <<= 1
	}
	if capacity > p.ghostMax {
		capacity = p.ghostMax
	}
	d.ghost.grow(capacity)
}

// Remove drops a key the store removed on its own (DELETE, expired sweep,
// emergency sweep). Unknown keys no-op.
func (p *S3FIFO) Remove(k Key) {
	d := p.domain(k.NS)
	if d == nil {
		return
	}
	ts := &d.table[p.shardIdx(k)]
	ts.mu.Lock()
	e := ts.m[k]
	delete(ts.m, k)
	ts.mu.Unlock()
	if e == nil {
		return
	}
	d.qmu.Lock()
	switch e.where {
	case qSmall:
		d.small.unlink(e)
		d.smallBytes -= e.size
	case qMain:
		d.main.unlink(e)
		d.mainBytes -= e.size
		d.mainCount--
	case qUnqueued:
		// Racing Admit hasn't queued it yet: tombstone; its queue step
		// sees qDead and drops the entry (see the state-constant doc).
	}
	e.where = qDead
	d.qmu.Unlock()
}

// Victims runs the S3-FIFO scan for tenant ns until candidates cover need
// bytes (or the domain runs dry). Dequeued candidates leave the policy —
// the store evicts them or hands them back via Admit.
func (p *S3FIFO) Victims(ns uint32, need int64, _ int64, dst []Candidate) []Candidate {
	d := p.domain(ns)
	if d == nil || need <= 0 {
		return dst
	}
	var got int64

	d.qmu.Lock()
	defer d.qmu.Unlock()
	// Bound one pass: every visit either moves an entry between queues,
	// burns a freq point, or expels — a freq-3 small entry costs at most 5
	// visits (1 small + 3 main decrements + 1 eviction), so 6N+8 covers
	// every reachable schedule with headroom; freq monotonicity guarantees
	// termination regardless.
	budget := 6*(d.small.count+d.main.count) + 8
	for got < need && budget > 0 {
		budget--
		// Source pick: drain small while it exceeds its 10% share.
		fromSmall := d.smallBytes > (d.smallBytes+d.mainBytes)/10
		if d.small.count == 0 && d.main.count == 0 {
			break
		}
		if fromSmall && d.small.count == 0 {
			fromSmall = false
		}
		if !fromSmall && d.main.count == 0 {
			fromSmall = true
		}
		if fromSmall {
			e := d.small.popHead()
			d.smallBytes -= e.size
			if e.freq.Load() > 1 {
				// Proved itself in probation: promote (freq kept).
				e.where = qMain
				d.main.pushTail(e)
				d.mainBytes += e.size
				d.mainCount++
				if d.mainCount > d.mainHi {
					d.mainHi = d.mainCount
					p.maybeGrowGhost(d)
				}
				continue
			}
			// One-hit wonder: evict + remember in ghost.
			d.ghost.push(p.fingerprint(e.key))
			dst = p.expel(d, e, dst)
			got += e.size
		} else {
			e := d.main.popHead()
			if e.freq.Load() > 0 {
				e.freq.Add(-1)
				d.main.pushTail(e) // second chance
				continue
			}
			d.mainBytes -= e.size
			d.mainCount--
			dst = p.expel(d, e, dst) // main evictions do not ghost
			got += e.size
		}
	}
	return dst
}

// expel finalizes an eviction candidate: table delete + candidate append.
// Caller holds qmu; tableShard.mu nests under it by the fixed domain order.
func (p *S3FIFO) expel(d *s3Domain, e *entry, dst []Candidate) []Candidate {
	e.where = qDead
	ts := &d.table[p.shardIdx(e.key)]
	ts.mu.Lock()
	delete(ts.m, e.key)
	ts.mu.Unlock()
	return append(dst, Candidate{Key: e.key, Size: e.size})
}

// Usage reports each domain's resident bytes (small + main).
func (p *S3FIFO) Usage(dst []DomainUsage) []DomainUsage {
	for ns, d := range *p.domains.Load() {
		d.qmu.Lock()
		b := d.smallBytes + d.mainBytes
		d.qmu.Unlock()
		if b > 0 {
			dst = append(dst, DomainUsage{NS: ns, Bytes: b})
		}
	}
	return dst
}

// ghostStats exposes (size, capacity) of a domain's ghost ring for the
// bound-invariant tests.
func (p *S3FIFO) ghostStats(ns uint32) (size, capacity int) {
	d := p.domain(ns)
	if d == nil {
		return 0, 0
	}
	d.qmu.Lock()
	defer d.qmu.Unlock()
	return d.ghost.size(), d.ghost.capacity()
}

// ---------------------------------------------------------------------------

// fifoQ is an intrusive doubly-linked FIFO. Guarded by the domain's qmu.
type fifoQ struct {
	head, tail *entry
	count      int
}

func (q *fifoQ) pushTail(e *entry) {
	e.prev, e.next = q.tail, nil
	if q.tail != nil {
		q.tail.next = e
	} else {
		q.head = e
	}
	q.tail = e
	q.count++
}

func (q *fifoQ) popHead() *entry {
	e := q.head
	q.head = e.next
	if q.head != nil {
		q.head.prev = nil
	} else {
		q.tail = nil
	}
	e.prev, e.next = nil, nil
	q.count--
	return e
}

func (q *fifoQ) unlink(e *entry) {
	if e.prev != nil {
		e.prev.next = e.next
	} else {
		q.head = e.next
	}
	if e.next != nil {
		e.next.prev = e.prev
	} else {
		q.tail = e.prev
	}
	e.prev, e.next = nil, nil
	q.count--
}
