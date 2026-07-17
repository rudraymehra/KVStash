// Package eviction provides the pluggable eviction policies the DRAM tier
// consults under memory pressure: S3-FIFO (default — scan-resistant via a
// small probationary queue plus a ghost ring of recently-evicted
// fingerprints) and sampled-LRU (the second policy that proves the Policy
// interface is real). Policies are pure bookkeeping: they never touch block
// payloads, never call back into the store, and their internal locks are
// LEAVES of the daemon's lock graph — the store never invokes a policy
// method while holding an index shard lock or the allocator mutex, so no
// lock-order cycle is constructible.
//
// A policy's answers are ADVISORY. The store re-gates every victim candidate
// under the index shard write lock (refcount/lease/pin ladder) and hands a
// candidate that proves protected back via Admit — the re-admission
// contract. Candidates can therefore be arbitrarily stale; every staleness
// mode resolves to a no-op or a re-admission, never a correctness hazard.
//
// TTL note (the timing-wheel deferral, recorded): expiry is enforced lazily
// by the store's evictor (expired blocks are the preferred victims of every
// pressure pass) rather than by a timer wheel. Revisit gate: build the
// 4×256 hierarchical wheel if a soak shows expired-resident bytes exceeding
// ~10% of the arena between pressure events, or the expired-sweep costing
// more than ~5ms per pass.
package eviction

import "fmt"

// Key identifies a block: a structural mirror of the store's key so this
// package imports nothing from the tier. Conversion is a value literal.
type Key struct {
	NS   uint32
	Hash [32]byte
}

// Candidate is one advisory eviction victim.
type Candidate struct {
	Key  Key
	Size int64 // payload bytes as reported at Admit
}

// DomainUsage reports one tenant domain's policy-view resident bytes.
type DomainUsage struct {
	NS    uint32
	Bytes int64
}

// Policy is the pluggable eviction brain. All methods are safe for
// concurrent use. Touch MUST NOT allocate — it sits on the GET hot path.
type Policy interface {
	// Admit records a block that became resident, called after the index
	// publish (never on a lost Put race). Also the re-admission path for a
	// gate-failed candidate. Admitting a key the policy already tracks is a
	// no-op. Zero-size blocks are never admitted (they own no extent).
	Admit(k Key, size int64, now int64)
	// Touch records an access. Unknown keys are a no-op. Allocation-free.
	Touch(k Key, now int64)
	// Remove records an index removal the policy did NOT nominate (client
	// DELETE, expired sweep, emergency sweep). Unknown keys are a no-op.
	Remove(k Key)
	// Victims appends candidates for tenant ns until their summed Size
	// reaches need or the domain is exhausted, reusing dst's backing array.
	// Candidates are DEQUEUED from the policy as a side effect: a candidate
	// the store successfully evicts needs no Remove call; one that fails
	// the store's gate must be handed back via Admit.
	Victims(ns uint32, need int64, now int64, dst []Candidate) []Candidate
	// Usage appends per-domain resident bytes into dst (pressure
	// attribution: the watermark splits an eviction batch across tenants
	// proportionally to these numbers).
	Usage(dst []DomainUsage) []DomainUsage
	// Name reports the policy's config name.
	Name() string
}

// New builds a policy by its config name. ghostCap bounds each tenant
// domain's ghost ring (S3-FIFO only); 0 means the caller resolved no
// explicit cap and the ring stays at its floor until Admits grow it —
// callers normally pass the arena-derived ceiling (see cmd/kvblockd).
func New(name string, ghostCap int) (Policy, error) {
	switch name {
	case "s3fifo":
		return NewS3FIFO(ghostCap), nil
	case "sampled-lru":
		return NewSampledLRU(), nil
	default:
		return nil, fmt.Errorf("eviction: unknown policy %q (want s3fifo or sampled-lru)", name)
	}
}
