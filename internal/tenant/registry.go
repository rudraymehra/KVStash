// Package tenant is the multi-tenancy spine: the namespace registry
// (identity + hashed bearer tokens + per-tier byte quotas) and the quota
// accountant the store tiers charge against.
//
// Identity is structural, not an ACL: every index key already carries the
// namespace id (dram.Key{NS, Hash}), so cross-tenant reuse is impossible by
// construction — the registry's job is only to bind a connection to its id
// at HELLO and to say how many bytes each tenant may hold per tier.
package tenant

import (
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"fmt"
	"os"
	"sync"

	"gopkg.in/yaml.v3"
)

// Tier indexes the quota triple. Order is wire-stable (Stats, admin API).
type Tier int

const (
	TierDRAM Tier = iota
	TierNVMe
	TierS3
	tierCount
)

func (t Tier) String() string {
	switch t {
	case TierDRAM:
		return "dram"
	case TierNVMe:
		return "nvme"
	case TierS3:
		return "s3"
	default:
		return fmt.Sprintf("tier(%d)", int(t))
	}
}

// Namespace is one tenant. TokenHash is SHA-256 of the bearer token — the
// registry never retains plaintext (a leaked registry file must not leak
// credentials). Quota[t] == 0 means unlimited for that tier.
type Namespace struct {
	ID        uint32
	Name      string
	TokenHash [32]byte
	Quota     [tierCount]int64
	PinQuota  int64
}

// Registry is the immutable-after-load namespace table plus the small
// mutation surface the admin socket uses. Reads are lock-free-ish (RLock);
// Authenticate scans by name with constant-time hash compares.
//
// LOCK ORDER: r.mu is a LEAF, and so is the accountant's (Quotas) — never
// hold one across the other. In particular an Each callback must not call
// into Quotas: Each holds r.mu.R, and Quotas.Reload holds its own lock while
// reading this registry — the inversion is a three-party deadlock.
type Registry struct {
	mu     sync.RWMutex
	byName map[string]*Namespace
	byID   map[uint32]*Namespace
}

// regFile is the on-disk schema (namespaces.yaml). `token` (plaintext) is
// accepted and hashed at load — prefer `token_sha256` so the file itself
// carries no credential.
type regFile struct {
	Namespaces []struct {
		Name        string `yaml:"name"`
		ID          uint32 `yaml:"id"`
		Token       string `yaml:"token"`
		TokenSHA256 string `yaml:"token_sha256"`
		QuotaDRAM   int64  `yaml:"quota_dram"`
		QuotaNVMe   int64  `yaml:"quota_nvme"`
		QuotaS3     int64  `yaml:"quota_s3"`
		PinQuota    int64  `yaml:"pin_quota"`
	} `yaml:"namespaces"`
}

// LoadRegistry reads the namespaces file. An empty path yields an empty
// registry — every HELLO then fails auth (a server with no tenants accepts
// no one; secure by default).
func LoadRegistry(path string) (*Registry, error) {
	r := &Registry{byName: make(map[string]*Namespace), byID: make(map[uint32]*Namespace)}
	if path == "" {
		return r, nil
	}
	b, err := os.ReadFile(path) //nolint:gosec // G304: operator-supplied config path
	if err != nil {
		return nil, fmt.Errorf("namespaces: %w", err)
	}
	var f regFile
	if err := yaml.Unmarshal(b, &f); err != nil {
		return nil, fmt.Errorf("namespaces %s: %w", path, err)
	}
	for _, e := range f.Namespaces {
		ns := &Namespace{
			ID: e.ID, Name: e.Name,
			Quota:    [tierCount]int64{e.QuotaDRAM, e.QuotaNVMe, e.QuotaS3},
			PinQuota: e.PinQuota,
		}
		switch {
		case e.TokenSHA256 != "":
			h, herr := hex.DecodeString(e.TokenSHA256)
			if herr != nil || len(h) != sha256.Size {
				return nil, fmt.Errorf("namespaces %s: %q token_sha256 must be 64 hex chars", path, e.Name)
			}
			copy(ns.TokenHash[:], h)
		case e.Token != "":
			ns.TokenHash = sha256.Sum256([]byte(e.Token))
		default:
			return nil, fmt.Errorf("namespaces %s: entry %q needs token or token_sha256", path, e.Name)
		}
		if err := r.Add(ns); err != nil {
			return nil, fmt.Errorf("namespaces %s: %w", path, err)
		}
	}
	return r, nil
}

// NewRegistry builds a registry in memory (tests, single-tenant bring-up).
// The token is plaintext and hashed here.
func NewRegistry(name string, id uint32, token string) *Registry {
	r := &Registry{byName: make(map[string]*Namespace), byID: make(map[uint32]*Namespace)}
	_ = r.Add(&Namespace{ID: id, Name: name, TokenHash: sha256.Sum256([]byte(token))})
	return r
}

// Add registers a namespace (load path + the admin socket).
func (r *Registry) Add(ns *Namespace) error {
	if ns.Name == "" || ns.ID == 0 {
		return fmt.Errorf("namespace %q needs a non-empty name and a nonzero id", ns.Name)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, dup := r.byName[ns.Name]; dup {
		return fmt.Errorf("duplicate namespace %q", ns.Name)
	}
	if _, dup := r.byID[ns.ID]; dup {
		return fmt.Errorf("duplicate namespace id %d (%q)", ns.ID, ns.Name)
	}
	r.byName[ns.Name] = ns
	r.byID[ns.ID] = ns
	return nil
}

// SetQuota updates one tier quota (admin socket). Returns false for an
// unknown namespace.
func (r *Registry) SetQuota(name string, tier Tier, bytes int64) bool {
	if tier < 0 || tier >= tierCount {
		return false
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	ns, ok := r.byName[name]
	if !ok {
		return false
	}
	ns.Quota[tier] = bytes
	return true
}

// Authenticate resolves (name, presented-token) to a namespace id. The
// compare is constant-time over SHA-256 digests; an unknown namespace runs
// a dummy compare so its timing matches a bad-token reject (no name-probing
// oracle). Auth is connection-scoped — one HELLO, never per-request.
func (r *Registry) Authenticate(name string, token []byte) (id uint32, ok bool) {
	digest := sha256.Sum256(token)
	r.mu.RLock()
	ns, found := r.byName[name]
	var want [32]byte
	if found {
		want = ns.TokenHash
	}
	r.mu.RUnlock()
	match := subtle.ConstantTimeCompare(digest[:], want[:]) == 1
	if !found || !match {
		return 0, false
	}
	return ns.ID, true
}

// Lookup returns the namespace for an id (quota wiring, per-tenant metrics).
func (r *Registry) Lookup(id uint32) (*Namespace, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	ns, ok := r.byID[id]
	return ns, ok
}

// QuotaSnapshot copies the namespace's quota triple under the registry lock —
// the accountant reads limits through this, never via a bare ns.Quota read
// racing SetQuota's locked write. LOCK ORDER: r.mu is a LEAF — the caller
// must not hold the accountant's q.mu here (the reg↔quotas inversion once
// deadlocked the scrape/admin planes).
func (r *Registry) QuotaSnapshot(id uint32) ([tierCount]int64, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if ns, ok := r.byID[id]; ok {
		return ns.Quota, true
	}
	return [tierCount]int64{}, false
}

// ValidatePinQuotas rejects any namespace whose pin_quota override exceeds
// maxBytes (the DRAM arena) or is negative — an override above the arena
// would promise more pinnable bytes than exist, silently unbounding the cap.
// The registry itself is arena-ignorant, so the daemon calls this with its
// config after load; the admin add path enforces the same bound.
func (r *Registry) ValidatePinQuotas(maxBytes int64) error {
	if maxBytes <= 0 {
		return nil
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	for _, ns := range r.byID {
		if ns.PinQuota < 0 || ns.PinQuota > maxBytes {
			return fmt.Errorf("namespace %q: pin_quota %d must be in [0, dram_arena_bytes %d]",
				ns.Name, ns.PinQuota, maxBytes)
		}
	}
	return nil
}

// PinQuotaFor returns the namespace's own pinned-bytes cap (0 = none set —
// the daemon's global pinned_bytes_cap applies). This is dram.Params.PinCapFor:
// the field was parsed, stored, and listed long before anything ENFORCED it.
func (r *Registry) PinQuotaFor(id uint32) int64 {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if ns, ok := r.byID[id]; ok {
		return ns.PinQuota
	}
	return 0
}

// Each calls fn for every namespace in unspecified order (metrics labels,
// admin listing).
func (r *Registry) Each(fn func(*Namespace)) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	for _, ns := range r.byID {
		fn(ns)
	}
}
