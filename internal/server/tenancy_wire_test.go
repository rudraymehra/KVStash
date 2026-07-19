package server_test

import (
	"bytes"
	"context"
	"crypto/sha256"
	"strings"
	"testing"
	"time"

	"github.com/kvstash/kvblockd/internal/config"
	"github.com/kvstash/kvblockd/internal/server"
	"github.com/kvstash/kvblockd/internal/store/dram"
	"github.com/kvstash/kvblockd/internal/tenant"
	"github.com/kvstash/kvblockd/pkg/client"
)

// twoTenantServer boots a daemon with tenants a(1) and b(2) over one shared
// arena; b carries a tight DRAM quota when quotaB > 0.
func twoTenantServer(t *testing.T, quotaB int64) string {
	t.Helper()
	arena, err := dram.NewArena(32<<20, false)
	if err != nil {
		t.Fatal(err)
	}
	reg := tenant.NewRegistry("a", 1, "tok-a")
	if err := reg.Add(&tenant.Namespace{
		ID: 2, Name: "b",
		TokenHash: sha256.Sum256([]byte("tok-b")),
		Quota:     [3]int64{quotaB, 0, 0},
	}); err != nil {
		t.Fatal(err)
	}
	st := dram.New(arena, dram.Params{
		LeaseDefaultMS: 5000, LeaseMaxMS: 60000, Quotas: tenant.NewQuotas(reg),
	})
	cfg := config.Default()
	cfg.ListenAddr = "127.0.0.1:0"
	srv := server.New(cfg, st, server.FromRegistry(reg))
	ctx, cancel := context.WithCancel(context.Background())
	addr, err := srv.Start(ctx)
	if err != nil {
		cancel()
		_ = st.Close()
		t.Fatal(err)
	}
	t.Cleanup(func() {
		cancel()
		dctx, dcancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer dcancel()
		srv.Drain(dctx)
		_ = st.Close()
	})
	return addr
}

func dialAs(t *testing.T, addr, ns, tok string) *client.Client {
	t.Helper()
	c, err := client.Dial(context.Background(), addr, client.Options{
		Streams: 1, Namespace: ns, Token: tok, DialTimeout: 5 * time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(c.Close)
	return c
}

func tkey(b byte) [32]byte { var k [32]byte; k[0], k[31] = b, 0xEE; return k }

// TestCrossTenantIsolationNoExistenceOracle: the SAME 32-byte key committed
// by A and B holds DISTINCT bytes (identity = (namespace, key)); and every
// cross-tenant probe — EXISTS, GET, DELETE — answers exactly like a miss,
// NEVER "forbidden": tenant B cannot even learn that A's key exists.
func TestCrossTenantIsolationNoExistenceOracle(t *testing.T) {
	addr := twoTenantServer(t, 0)
	ca := dialAs(t, addr, "a", "tok-a")
	cb := dialAs(t, addr, "b", "tok-b")
	ctx := context.Background()

	k := tkey(0x11)
	blobA := bytes.Repeat([]byte{0xAA}, 4096)
	blobB := bytes.Repeat([]byte{0xBB}, 8192)
	if err := ca.Put(ctx, k, blobA); err != nil {
		t.Fatal(err)
	}
	if err := cb.Put(ctx, k, blobB); err != nil {
		t.Fatalf("same key, different tenant must be a DISTINCT block: %v", err)
	}

	into := make([][]byte, 1)
	if _, err := ca.BatchGet(ctx, [][32]byte{k}, into); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(into[0], blobA) {
		t.Fatal("tenant A read tenant B's bytes (or vice versa)")
	}
	into2 := make([][]byte, 1)
	if _, err := cb.BatchGet(ctx, [][32]byte{k}, into2); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(into2[0], blobB) {
		t.Fatal("tenant B read tenant A's bytes")
	}

	// B probes a key ONLY A holds: every answer must be a plain miss.
	onlyA := tkey(0x22)
	if err := ca.Put(ctx, onlyA, blobA); err != nil {
		t.Fatal(err)
	}
	if n, _, err := cb.BatchExists(ctx, [][32]byte{onlyA}); err != nil || n != 0 {
		t.Fatalf("existence oracle: B sees A's key via EXISTS (n=%d err=%v)", n, err)
	}
	got := make([][]byte, 1)
	sts, err := cb.BatchGet(ctx, [][32]byte{onlyA}, got)
	if err != nil {
		t.Fatal(err)
	}
	if len(got[0]) != 0 {
		t.Fatal("B fetched A's bytes")
	}
	_ = sts
	del, err := cb.Delete(ctx, [][32]byte{onlyA}, true)
	if err != nil {
		t.Fatal(err)
	}
	if del[0].String() != "NOT_FOUND" && !strings.Contains(del[0].String(), "NOT_FOUND") {
		t.Fatalf("cross-tenant DELETE answered %s, want the indistinguishable NOT_FOUND", del[0])
	}
	// And A's block is untouched by B's force-delete attempt.
	if n, _, err := ca.BatchExists(ctx, [][32]byte{onlyA}); err != nil || n != 1 {
		t.Fatalf("A's block damaged by B's delete: n=%d err=%v", n, err)
	}
}

// TestTenantQuotaOnTheWire: B (quota = 3 blocks) is refused its 4th block at
// BEGIN with ERR_QUOTA_BYTES while A (unlimited) keeps writing; deleting one
// of B's blocks refunds exactly one block of headroom.
func TestTenantQuotaOnTheWire(t *testing.T) {
	const blk = 64 << 10
	addr := twoTenantServer(t, 3*blk)
	ca := dialAs(t, addr, "a", "tok-a")
	cb := dialAs(t, addr, "b", "tok-b")
	ctx := context.Background()
	blob := bytes.Repeat([]byte{0x5A}, blk)

	for i := 0; i < 3; i++ {
		if err := cb.Put(ctx, tkey(byte(0x30+i)), blob); err != nil {
			t.Fatalf("B put %d inside quota: %v", i, err)
		}
	}
	err := cb.Put(ctx, tkey(0x3F), blob)
	if err == nil {
		t.Fatal("B's 4th block admitted over a 3-block quota")
	}
	if !strings.Contains(err.Error(), "QUOTA") && !strings.Contains(err.Error(), "BUSY") {
		t.Fatalf("over-quota PUT surfaced as %v, want ERR_QUOTA_BYTES (BEGIN) or ERR_BUSY (commit)", err)
	}
	// A is unaffected by B's ceiling.
	if err := ca.Put(ctx, tkey(0x77), blob); err != nil {
		t.Fatalf("unlimited tenant A blocked by B's quota: %v", err)
	}
	// Refund exactness: delete one B block → exactly one more fits.
	if _, err := cb.Delete(ctx, [][32]byte{tkey(0x30)}, true); err != nil {
		t.Fatal(err)
	}
	if err := cb.Put(ctx, tkey(0x3F), blob); err != nil {
		t.Fatalf("B put after refund: %v", err)
	}
	if err := cb.Put(ctx, tkey(0x3E), blob); err == nil {
		t.Fatal("refund minted extra headroom (two admits after one delete)")
	}
}
