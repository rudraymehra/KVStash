package server

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/kvstash/kvblockd/internal/tenant"
)

// adminArenaBytes stands in for dram_arena_bytes (the pin_quota ceiling).
const adminArenaBytes = 1 << 20

func adminUp(t *testing.T) (string, *tenant.Registry, *tenant.Quotas) {
	t.Helper()
	reg := tenant.NewRegistry("a", 1, "tok-a")
	q := tenant.NewQuotas(reg)
	a := NewAdminServer(reg, q, adminArenaBytes)
	ctx, cancel := context.WithCancel(context.Background())
	addr, wait, err := a.Serve(ctx, "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { cancel(); wait() })
	return addr, reg, q
}

func TestAdminLoopbackOnly(t *testing.T) {
	a := NewAdminServer(tenant.NewRegistry("a", 1, "t"), nil, adminArenaBytes)
	if _, _, err := a.Serve(context.Background(), "0.0.0.0:0"); err == nil {
		t.Fatal("non-loopback admin bind accepted")
	}
}

// TestAdminAddRejectsPervesePinQuota: a pin_quota above the arena would
// promise more pinnable bytes than exist (the per-ns override REPLACES the
// global cap — an oversize one silently unbounds it); negatives are nonsense.
func TestAdminAddRejectsPervesePinQuota(t *testing.T) {
	addr, reg, _ := adminUp(t)
	h := sha256.Sum256([]byte("tok-x"))
	for name, pin := range map[string]int64{
		"over-arena": adminArenaBytes + 1,
		"negative":   -1,
	} {
		body, _ := json.Marshal(map[string]any{
			"name": name, "id": 9, "token_sha256": hex.EncodeToString(h[:]),
			"pin_quota": pin,
		})
		resp, err := http.Post(fmt.Sprintf("http://%s/v1/namespace", addr), "application/json", bytes.NewReader(body)) //nolint:noctx // test-local
		if err != nil {
			t.Fatal(err)
		}
		_, _ = io.Copy(io.Discard, resp.Body)
		_ = resp.Body.Close()
		if resp.StatusCode != http.StatusBadRequest {
			t.Fatalf("%s pin_quota accepted: %d", name, resp.StatusCode)
		}
		if _, ok := reg.Lookup(9); ok {
			t.Fatalf("%s: namespace landed in the registry despite the reject", name)
		}
	}
}

func TestAdminNamespaceAddAndQuotaSet(t *testing.T) {
	addr, reg, q := adminUp(t)

	h := sha256.Sum256([]byte("tok-b"))
	body, _ := json.Marshal(map[string]any{
		"name": "b", "id": 2, "token_sha256": hex.EncodeToString(h[:]),
		"quota_dram": 1024,
	})
	resp, err := http.Post(fmt.Sprintf("http://%s/v1/namespace", addr), "application/json", bytes.NewReader(body)) //nolint:noctx // test-local
	if err != nil {
		t.Fatal(err)
	}
	_, _ = io.Copy(io.Discard, resp.Body)
	_ = resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("namespace add: %d", resp.StatusCode)
	}
	if id, ok := reg.Authenticate("b", []byte("tok-b")); !ok || id != 2 {
		t.Fatalf("added namespace does not authenticate: %d %v", id, ok)
	}

	// Duplicate → 409, never a silent overwrite.
	resp, err = http.Post(fmt.Sprintf("http://%s/v1/namespace", addr), "application/json", bytes.NewReader(body)) //nolint:noctx // test-local
	if err != nil {
		t.Fatal(err)
	}
	_, _ = io.Copy(io.Discard, resp.Body)
	_ = resp.Body.Close()
	if resp.StatusCode != http.StatusConflict {
		t.Fatalf("duplicate add: %d", resp.StatusCode)
	}

	// Quota set reaches the accountant through Reload.
	qb, _ := json.Marshal(map[string]any{"name": "b", "tier": "dram", "bytes": 4096})
	resp, err = http.Post(fmt.Sprintf("http://%s/v1/quota", addr), "application/json", bytes.NewReader(qb)) //nolint:noctx // test-local
	if err != nil {
		t.Fatal(err)
	}
	_, _ = io.Copy(io.Discard, resp.Body)
	_ = resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("quota set: %d", resp.StatusCode)
	}
	_ = q.Charge(2, tenant.TierDRAM, 1) // touch the domain so limits snapshot
	if got := q.Limit(2, tenant.TierDRAM); got != 4096 {
		t.Fatalf("accountant limit after set: %d", got)
	}

	// Listing never leaks token material.
	resp, err = http.Get(fmt.Sprintf("http://%s/v1/namespaces", addr)) //nolint:noctx // test-local
	if err != nil {
		t.Fatal(err)
	}
	b, _ := io.ReadAll(resp.Body)
	_ = resp.Body.Close()
	if strings.Contains(strings.ToLower(string(b)), "token") {
		t.Fatalf("listing leaks token material: %s", b)
	}
	if !strings.Contains(string(b), `"name":"b"`) {
		t.Fatalf("listing missing namespace: %s", b)
	}
}
