package tenant

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeReg(t *testing.T, body string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "ns.yaml")
	if err := os.WriteFile(p, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestLoadRegistryPlaintextTokenIsHashed(t *testing.T) {
	r, err := LoadRegistry(writeReg(t, `
namespaces:
  - { name: a, id: 1, token: "secret-a", quota_dram: 100, quota_nvme: 200, quota_s3: 300, pin_quota: 7 }
`))
	if err != nil {
		t.Fatal(err)
	}
	ns, ok := r.Lookup(1)
	if !ok {
		t.Fatal("lookup")
	}
	if ns.TokenHash != sha256.Sum256([]byte("secret-a")) {
		t.Fatal("plaintext token not hashed at load")
	}
	if ns.Quota != [3]int64{100, 200, 300} || ns.PinQuota != 7 {
		t.Fatalf("quotas: %+v pin=%d", ns.Quota, ns.PinQuota)
	}
}

func TestLoadRegistryHashedToken(t *testing.T) {
	h := sha256.Sum256([]byte("tok-b"))
	r, err := LoadRegistry(writeReg(t, `
namespaces:
  - { name: b, id: 2, token_sha256: "`+hex.EncodeToString(h[:])+`" }
`))
	if err != nil {
		t.Fatal(err)
	}
	if id, ok := r.Authenticate("b", []byte("tok-b")); !ok || id != 2 {
		t.Fatalf("auth against token_sha256: id=%d ok=%v", id, ok)
	}
	if _, ok := r.Authenticate("b", []byte("tok-wrong")); ok {
		t.Fatal("wrong token accepted")
	}
}

func TestLoadRegistryRejects(t *testing.T) {
	for name, body := range map[string]string{ //nolint:gosec // G101: throwaway test fixtures, not credentials
		"no-token":     "namespaces:\n  - { name: x, id: 1 }\n",
		"zero-id":      "namespaces:\n  - { name: x, id: 0, token: t }\n",
		"empty-name":   "namespaces:\n  - { name: \"\", id: 1, token: t }\n",
		"dup-name":     "namespaces:\n  - { name: x, id: 1, token: t }\n  - { name: x, id: 2, token: t }\n",
		"dup-id":       "namespaces:\n  - { name: x, id: 1, token: t }\n  - { name: y, id: 1, token: t }\n",
		"bad-hash-hex": "namespaces:\n  - { name: x, id: 1, token_sha256: zz }\n",
		"short-hash":   "namespaces:\n  - { name: x, id: 1, token_sha256: abcd }\n",
	} {
		if _, err := LoadRegistry(writeReg(t, body)); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

func TestAuthenticateUnknownNamespaceIsUniformReject(t *testing.T) {
	r := NewRegistry("a", 1, "tok")
	if _, ok := r.Authenticate("nope", []byte("tok")); ok {
		t.Fatal("unknown namespace authenticated")
	}
	if _, ok := r.Authenticate("a", nil); ok {
		t.Fatal("empty token authenticated")
	}
	if id, ok := r.Authenticate("a", []byte("tok")); !ok || id != 1 {
		t.Fatalf("valid auth failed: id=%d ok=%v", id, ok)
	}
}

func TestEmptyRegistryAcceptsNoOne(t *testing.T) {
	r, err := LoadRegistry("")
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := r.Authenticate("", nil); ok {
		t.Fatal("empty registry authenticated an empty pair")
	}
}

func TestValidatePinQuotas(t *testing.T) {
	r, err := LoadRegistry(writeReg(t, `
namespaces:
  - { name: sane, id: 1, token: t, pin_quota: 1024 }
  - { name: greedy, id: 2, token: t, pin_quota: 2048 }
`))
	if err != nil {
		t.Fatal(err)
	}
	if err := r.ValidatePinQuotas(2048); err != nil {
		t.Fatalf("in-bounds pin quotas rejected: %v", err)
	}
	if err := r.ValidatePinQuotas(2047); err == nil {
		t.Fatal("pin_quota above the arena accepted — the override would unbound the cap")
	} else if !strings.Contains(err.Error(), "greedy") {
		t.Fatalf("rejection does not name the offender: %v", err)
	}
	if err := r.ValidatePinQuotas(0); err != nil {
		t.Fatalf("unknown arena (0) must skip the check: %v", err)
	}
}

func TestSetQuotaAndEach(t *testing.T) {
	r := NewRegistry("a", 1, "tok")
	if !r.SetQuota("a", TierNVMe, 42) {
		t.Fatal("SetQuota on known ns failed")
	}
	if r.SetQuota("nope", TierNVMe, 1) {
		t.Fatal("SetQuota on unknown ns succeeded")
	}
	seen := 0
	r.Each(func(ns *Namespace) {
		seen++
		if ns.Quota[TierNVMe] != 42 {
			t.Fatalf("quota not applied: %+v", ns.Quota)
		}
	})
	if seen != 1 {
		t.Fatalf("Each visited %d", seen)
	}
}
