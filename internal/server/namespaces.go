package server

import (
	"crypto/subtle"
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

// Namespaces maps a (namespace, bearer-token) pair to a numeric namespace id
// for the connection's lifetime. Auth is connection-scoped (one HELLO), never
// per-request (PROTOCOL.md §3.1). The full tenant system (quotas, QoS) is
// later; this is the minimum: name→id + constant-time token compare.
type Namespaces struct {
	byName map[string]nsEntry
}

type nsEntry struct {
	id    uint32
	token string
}

// nsFile is the on-disk schema (namespaces.yaml).
type nsFile struct {
	Namespaces []struct {
		Name  string `yaml:"name"`
		ID    uint32 `yaml:"id"`
		Token string `yaml:"token"`
	} `yaml:"namespaces"`
}

// LoadNamespaces reads the namespaces/tokens file. An empty path yields an
// empty table (every HELLO then fails auth — a server with no tenants).
func LoadNamespaces(path string) (*Namespaces, error) {
	ns := &Namespaces{byName: make(map[string]nsEntry)}
	if path == "" {
		return ns, nil
	}
	b, err := os.ReadFile(path) //nolint:gosec // G304: operator-supplied config path
	if err != nil {
		return nil, fmt.Errorf("namespaces: %w", err)
	}
	var f nsFile
	if err := yaml.Unmarshal(b, &f); err != nil {
		return nil, fmt.Errorf("namespaces %s: %w", path, err)
	}
	for _, e := range f.Namespaces {
		if e.Name == "" || e.ID == 0 {
			return nil, fmt.Errorf("namespaces %s: entry %q needs a non-empty name and a nonzero id", path, e.Name)
		}
		if _, dup := ns.byName[e.Name]; dup {
			return nil, fmt.Errorf("namespaces %s: duplicate namespace %q", path, e.Name)
		}
		ns.byName[e.Name] = nsEntry{id: e.ID, token: e.Token}
	}
	return ns, nil
}

// NewNamespaces builds a table in memory (tests, single-tenant bring-up).
func NewNamespaces(name string, id uint32, token string) *Namespaces {
	return &Namespaces{byName: map[string]nsEntry{name: {id: id, token: token}}}
}

// Authenticate resolves (name, token) to a namespace id. The token compare is
// constant-time so a timing side channel can't probe valid tokens; an unknown
// namespace runs a dummy compare so its timing matches a bad-token reject.
func (n *Namespaces) Authenticate(name string, token []byte) (id uint32, ok bool) {
	e, found := n.byName[name]
	want := e.token
	if !found {
		want = "" // still run a compare below to keep timing uniform
	}
	match := subtle.ConstantTimeCompare(token, []byte(want)) == 1
	if !found || !match {
		return 0, false
	}
	return e.id, true
}
