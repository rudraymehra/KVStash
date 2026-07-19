package server

import (
	"github.com/kvstash/kvblockd/internal/tenant"
)

// Namespaces is the server's view of the tenant registry: (name, token) →
// namespace id at HELLO, connection-scoped, never per-request (PROTOCOL.md
// §3.1). It is a thin adapter — identity, hashed tokens, and per-tier
// quotas live in internal/tenant.
type Namespaces struct {
	reg *tenant.Registry
}

// LoadNamespaces reads the namespaces/tokens file (see tenant.LoadRegistry
// for the schema: plaintext `token` is hashed at load; `token_sha256` keeps
// credentials out of the file). An empty path yields an empty table — every
// HELLO then fails auth (a server with no tenants; secure by default).
func LoadNamespaces(path string) (*Namespaces, error) {
	reg, err := tenant.LoadRegistry(path)
	if err != nil {
		return nil, err
	}
	return &Namespaces{reg: reg}, nil
}

// NewNamespaces builds a single-tenant table in memory (tests, bring-up).
func NewNamespaces(name string, id uint32, token string) *Namespaces {
	return &Namespaces{reg: tenant.NewRegistry(name, id, token)}
}

// FromRegistry wraps an already-built registry (main.go shares one instance
// between auth, quotas, and the admin surface).
func FromRegistry(reg *tenant.Registry) *Namespaces { return &Namespaces{reg: reg} }

// Registry exposes the underlying tenant registry (quota wiring, admin).
func (n *Namespaces) Registry() *tenant.Registry { return n.reg }

// Authenticate resolves (name, token) to a namespace id — constant-time
// over SHA-256 digests, uniform timing for unknown names.
func (n *Namespaces) Authenticate(name string, token []byte) (id uint32, ok bool) {
	return n.reg.Authenticate(name, token)
}
