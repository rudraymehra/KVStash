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
	"fmt"

	"github.com/kvstash/kvblockd/internal/eviction"
	"github.com/kvstash/kvblockd/internal/protocol"
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
	// anyOf is set on a resurrected ghost whose exact surviving content is
	// unknown until the first byte-carrying observation: the block must
	// match ONE of these historical contents (see model.resolveBytes).
	anyOf []histRec
	// pinCharge is what the model believes the store's pinned-bytes ledger
	// was debited for THIS block's hard pin. For a pinned-down block it is
	// exact (len(data)); a hard pin on an anyOf ghost charges the block's
	// TRUE size, which the model cannot know yet — pinChargeUnknown marks
	// it, the I3 check bounds it by the candidate size range, and the next
	// byte-carrying observation retro-tightens the ledger (resolveBytes).
	pinCharge        int64
	pinChargeUnknown bool
}

// candidateSizeRange bounds a ghost's possible true size across its
// historical candidates (I3's slack for unknown pin charges).
func candidateSizeRange(anyOf []histRec) (lo, hi int64) {
	for i, h := range anyOf {
		sz := int64(len(h.data))
		if i == 0 || sz < lo {
			lo = sz
		}
		if sz > hi {
			hi = sz
		}
	}
	return lo, hi
}

// model is the reference: plain maps, no tiers, no eviction of its own.
//
// Resurrection (the tiered machine's crash semantics, a ladder finding):
// an NVMe DELETE is not crash-durable — recovery replays footers and
// checkpoints, so a deleted key whose bytes ever reached a segment may
// come BACK after a crash, carrying any content that was ever legitimately
// committed under it. The model therefore keeps a per-key HISTORY of every
// committed content and an epoch-stamped ghost set: after a crash
// (epoch++), an observation of a key the model thinks absent is legal iff
// the key was deleted in an earlier epoch and the served bytes match some
// historical content (I1 still binds byte-identity — resurrection may
// never invent bytes).
type model struct {
	now       int64
	blocks    map[eviction.Key]*mBlock
	pinned    map[uint32]int64 // mirrors the per-ns pinned-bytes ledger
	pinnedCap int64

	epoch   int                        // crash counter
	ghosts  map[eviction.Key]int       // deleted keys → epoch of deletion
	history map[eviction.Key][]histRec // every content ever committed under the key
}

type histRec struct {
	data []byte
	xxh3 uint64
}

func newModel(startNanos, pinnedCap int64) *model {
	return &model{
		now:       startNanos,
		blocks:    make(map[eviction.Key]*mBlock),
		pinned:    make(map[uint32]int64),
		pinnedCap: pinnedCap,
		ghosts:    make(map[eviction.Key]int),
		history:   make(map[eviction.Key][]histRec),
	}
}

// noteDeleted records a successful DELETE: the key may resurrect after a
// LATER crash (never before one — index removal is immediate).
func (m *model) noteDeleted(k eviction.Key) {
	delete(m.blocks, k)
	m.ghosts[k] = m.epoch
}

// crashed applies simulate_crash's whole model transition (also driven
// directly by the deterministic crash-survivor regression): the epoch bump
// makes deleted-key ghosts resurrectable from here on, and every surviving
// block reverts to the anyOf form — DRAM contents vanished, protection
// state (leases/pins — memory-only) vanished, and recovery may resurface
// ANY committed-and-persisted content for a key, not just the latest.
// Force-delete + re-put under one key composes with the non-crash-durable
// NVMe DELETE: the pre-delete bytes legally come back while the re-put
// (DRAM-only at the kill) is legally lost. The deep gauntlet found this
// after 795 walks: the model pinned the key to its LATEST content and
// called the resurrected older version an I1 violation. So every surviving
// block reverts to the anyOf form the ghost path already uses; the next
// byte-carrying observation pins it down. I1 keeps its teeth — served
// bytes outside the key's committed history still fail.
func (m *model) crashed() {
	m.epoch++
	for k, b := range m.blocks {
		b.maybeGone = true
		b.leaseUntil = 0
		b.ttlUntil = 0
		b.soft, b.hard = false, false
		b.pinCharge, b.pinChargeUnknown = 0, false
		b.data, b.xxh3, b.anyOf = nil, 0, m.history[k]
	}
	m.pinned = map[uint32]int64{}
}

// resurrectable reports whether an observation of the model-absent key k is
// explainable as post-crash resurrection.
func (m *model) resurrectable(k eviction.Key) bool {
	e, ok := m.ghosts[k]
	return ok && e < m.epoch
}

// materializeGhost re-admits a resurrected key with as-yet-unknown content:
// the block carries the key's full history as candidates (anyOf); the next
// byte-carrying observation (GET) pins it down. Protection state starts
// clean — it died with the process.
func (m *model) materializeGhost(k eviction.Key) *mBlock {
	b := &mBlock{anyOf: m.history[k], maybeGone: true}
	m.blocks[k] = b
	return b
}

// resolveBytes checks served bytes against the block: a pinned-down block
// demands ITS content; an anyOf block accepts any historical content and
// pins down to it. ns is the block's namespace — pinning down a ghost that
// is hard-pinned with an unknown charge retro-tightens the pinned-bytes
// ledger to the now-known true size. Returns false on an I1 violation.
func (m *model) resolveBytes(ns uint32, b *mBlock, data []byte, sum uint64) bool {
	if b.anyOf == nil {
		return sum == b.xxh3 && bytesEqual(data, b.data)
	}
	for _, h := range b.anyOf {
		if sum == h.xxh3 && bytesEqual(data, h.data) {
			b.data, b.xxh3, b.anyOf = h.data, h.xxh3, nil
			if b.hard && b.pinChargeUnknown {
				b.pinCharge = int64(len(b.data))
				b.pinChargeUnknown = false
				m.pinned[ns] += b.pinCharge
			}
			return true
		}
	}
	return false
}

// applyPut reconciles one PUT's observed status against the model and
// applies the resulting transition — the write-once truth table the machine
// asserts after every put action. Returns "" when the outcome is legal,
// else the invariant-violation message (the machine Fatalfs it verbatim).
// Extracted from the rapid closure so the deterministic crash-survivor
// regression drives the SAME oracle the random walks use.
func (m *model) applyPut(k eviction.Key, data []byte, sum uint64, st protocol.Status) string {
	e := m.blocks[k]
	switch {
	case e == nil:
		switch {
		case st == protocol.StatusOK:
			m.insert(k, data, sum)
		case st == protocol.StatusErrQuotaBytes: // full arena, no evictor: legal
		case st == protocol.StatusOKExists && m.resurrectable(k):
			// A crash resurrected the deleted key with OUR sum: the resident
			// content is (xxh3-)identical to this put.
			b := m.materializeGhost(k)
			if !m.resolveBytes(k.NS, b, data, sum) {
				return fmt.Sprintf("put resurrected-dup: OK_EXISTS but %v matches no history", k)
			}
			b.maybeGone = false
		case st == protocol.StatusErrImmutableConflict && m.resurrectable(k):
			// Resurrected with some OTHER historical content — the write-once
			// alarm is correct; content pins down at the next GET.
			m.materializeGhost(k).maybeGone = false
		default:
			return fmt.Sprintf("put fresh: %s", st)
		}
	case e.anyOf != nil:
		// Unresolved post-crash block (a crash SURVIVOR the model kept, or a
		// materialized ghost no observation has pinned down): its resident
		// content — if the key survived at all — is SOME member of the key's
		// committed history, unknown until byte-carrying evidence arrives.
		// e.xxh3 is 0 here (a placeholder, not knowledge), so the dup/conflict
		// split below would misclassify EVERY put as conflicting content and
		// reject a legal OK_EXISTS — the 2026-07-22 CI model-soak failure:
		// put C → demote to NVMe → crash → recovery resurfaces C → put C
		// again is a digest-verified write-once hit the harness called a
		// violation (schedule-dependent because survival depends on whether
		// the record reached the file before the kill). Same semantics the
		// e==nil ghost arm above already grants; survivors get them too.
		switch {
		case st == protocol.StatusOKExists:
			// The store vouches a RESIDENT block carries this put's digest:
			// legal iff the bytes are one of the key's committed contents
			// (crash recovery may resurface any of them — never invented
			// ones), and the match pins the block down exactly like a GET.
			if !m.resolveBytes(k.NS, e, data, sum) {
				return fmt.Sprintf("put survivor-dup: OK_EXISTS but %v matches no committed content", k)
			}
			e.maybeGone = false // the write-once gate answered on a resident block
		case st == protocol.StatusErrImmutableConflict:
			// Resident with some OTHER historical content — the write-once
			// alarm is correct; the content pins down at the next
			// byte-carrying observation. The refusal itself proves residency.
			e.maybeGone = false
		case st == protocol.StatusOK && e.maybeGone:
			m.insert(k, data, sum) // did not survive the crash: fresh insert
		case st == protocol.StatusErrQuotaBytes && e.maybeGone:
			delete(m.blocks, k) // gone AND the arena is full
		default:
			return fmt.Sprintf("put unresolved: %s (maybeGone=%v)", st, e.maybeGone)
		}
	case e.xxh3 == sum: // duplicate content
		switch {
		case st == protocol.StatusOKExists: // present (maybeGone stays — bytes unverified)
		case st == protocol.StatusOK && e.maybeGone: // was evicted: fresh insert
			m.insert(k, data, sum)
		case st == protocol.StatusErrQuotaBytes && e.maybeGone: // evicted AND arena full
			delete(m.blocks, k)
		default:
			return fmt.Sprintf("put dup: %s (maybeGone=%v)", st, e.maybeGone)
		}
	default: // conflicting content
		switch {
		case st == protocol.StatusErrImmutableConflict:
		case st == protocol.StatusOK && e.maybeGone:
			m.insert(k, data, sum)
		case st == protocol.StatusErrQuotaBytes && e.maybeGone:
			delete(m.blocks, k)
		default:
			return fmt.Sprintf("put conflict: %s (maybeGone=%v)", st, e.maybeGone)
		}
	}
	return ""
}

func bytesEqual(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
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
// state resets — the store built a brand-new BlockRef) and records the
// content in the key's resurrection history.
func (m *model) insert(k eviction.Key, data []byte, sum uint64) {
	cp := make([]byte, len(data))
	copy(cp, data)
	m.insertShared(k, cp, sum)
}

// insertShared is insert WITHOUT the defensive copy: the caller promises
// data is immutable for the model's lifetime. The machine's junk fill puts
// ONE constant pattern under thousands of monotonically-fresh keys — per-key
// copies of it (blocks + permanent resurrection history) were the harness's
// dominant memory cost under -race, not anything the oracle needed.
func (m *model) insertShared(k eviction.Key, data []byte, sum uint64) {
	m.blocks[k] = &mBlock{data: data, xxh3: sum}
	for _, h := range m.history[k] {
		if h.xxh3 == sum && bytesEqual(h.data, data) {
			return // content already on record
		}
	}
	m.history[k] = append(m.history[k], histRec{data: data, xxh3: sum})
}
