// Package modeltest is the model-based correctness harness: a rapid state
// machine drives the REAL dram tier and a brutally-dumb reference model
// (plain maps, full byte copies) through the same randomized op stream,
// checking the store's answers against the model's after every step.
//
// The store is LOSSY under pressure (eviction) while the model is not, so
// the harness uses RECONCILIATION rather than prediction: after a pressure
// pass every unprotected model block is marked maybeGone; the next real
// observation of that key resolves it — a hit must be byte-identical (a
// wrong-bytes hit is ALWAYS a bug, invariant I1), a miss deletes the model
// copy. A PROTECTED block (hard pin, live lease, held reader ref) observed
// missing fails immediately (I2). This asymmetry is what lets a lossy
// store have a strict oracle.
//
// Invariants (the permanent safety net every later tier must pass):
//
//	I1  GET is byte-identical or NOT_FOUND — never partial/stale/cross-key.
//	I2  hard pins, live leases, and read-held blocks survive pressure.
//	I3  store accounting (Stats) stays consistent with the model's view.
//	I4  namespace isolation is absolute (same hash, different ns, different
//	    payloads — each GET returns its own tenant's bytes).
//	I5  a held GetRef view stays byte-identical until its release fires —
//	    deletes and pressure never recycle an extent under a live view.
package modeltest

import (
	"github.com/kvstash/kvblockd/internal/eviction"
)

// mBlock is the model's copy of one committed block.
type mBlock struct {
	data       []byte // full copy — dumb on purpose
	xxh3       uint64
	leaseUntil int64
	ttlUntil   int64
	soft, hard bool
	heldRefs   int  // live GetRef holds the machine keeps open
	maybeGone  bool // a pressure pass may have taken it; reconcile on sight
}

// model is the reference: plain maps, no tiers, no eviction of its own.
type model struct {
	now       int64
	blocks    map[eviction.Key]*mBlock
	pinned    map[uint32]int64 // mirrors the per-ns pinned-bytes ledger
	pinnedCap int64
}

func newModel(startNanos, pinnedCap int64) *model {
	return &model{
		now:       startNanos,
		blocks:    make(map[eviction.Key]*mBlock),
		pinned:    make(map[uint32]int64),
		pinnedCap: pinnedCap,
	}
}

// protected mirrors the evictor's gate from the model's side: blocks the
// pressure pass must never take.
func (m *model) protected(b *mBlock) bool {
	return b.hard || b.leaseUntil > m.now || b.heldRefs > 0
}

// markPressure flags every unprotected block as maybe-evicted (soft pins
// included — the emergency pass may take them).
func (m *model) markPressure() {
	for _, b := range m.blocks {
		if !m.protected(b) {
			b.maybeGone = true
		}
	}
}

// reconcileMiss handles a real-store miss for k: legal only when the model
// agrees it's absent or a pressure pass could have taken it. Returns false
// when the miss violates I1/I2 (the block was protected or never marked).
func (m *model) reconcileMiss(k eviction.Key) bool {
	b, ok := m.blocks[k]
	if !ok {
		return true
	}
	if !b.maybeGone || m.protected(b) {
		return false
	}
	delete(m.blocks, k)
	return true
}

// insert replaces the model block for k (fresh publish: all protection
// state resets — the store built a brand-new BlockRef).
func (m *model) insert(k eviction.Key, data []byte, sum uint64) {
	cp := make([]byte, len(data))
	copy(cp, data)
	m.blocks[k] = &mBlock{data: cp, xxh3: sum}
}
