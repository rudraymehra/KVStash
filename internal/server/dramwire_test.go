package server_test

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"testing"
	"time"

	"github.com/kvstash/kvblockd/internal/config"
	"github.com/kvstash/kvblockd/internal/protocol"
	"github.com/kvstash/kvblockd/internal/server"
	"github.com/kvstash/kvblockd/internal/store/dram"
	"github.com/kvstash/kvblockd/pkg/client"
)

// startDRAMServer boots a server over a real arena-backed DRAM tier. Cleanup
// enforces the drain-then-close ordering rule: srv.Drain waits every conn's
// writer (which fires every in-flight GET release), so the kvbdebug refcount
// assert inside Store.Close is meaningful — a leaked release panics the test.
func startDRAMServer(t *testing.T, arenaBytes int64, p dram.Params) (addr string, cleanup func()) {
	t.Helper()
	arena, err := dram.NewArena(arenaBytes, false)
	if err != nil {
		t.Fatal(err)
	}
	st := dram.New(arena, p)
	cfg := config.Default()
	cfg.ListenAddr = "127.0.0.1:0"
	cfg.StreamTimeoutMS = 5000
	ns := server.NewNamespaces("tenant-a", 7, testToken)
	srv := server.New(cfg, st, ns)

	ctx, cancel := context.WithCancel(context.Background())
	a, err := srv.Start(ctx)
	if err != nil {
		cancel()
		_ = st.Close()
		t.Fatal(err)
	}
	return a, func() {
		cancel()
		dctx, dcancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer dcancel()
		srv.Drain(dctx)
		if err := st.Close(); err != nil {
			t.Error(err)
		}
	}
}

func dramDefaults() dram.Params {
	return dram.Params{LeaseDefaultMS: 5000, LeaseMaxMS: 60000}
}

// dramStats decodes the fields the wire tests assert on.
func dramStats(t *testing.T, c *client.Client) (store string, blocks int, arenaFree int64) {
	t.Helper()
	raw, err := c.Stats(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	var doc struct {
		Store          string `json:"store"`
		Blocks         int    `json:"blocks"`
		ArenaFreeBytes int64  `json:"arena_free_bytes"`
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatalf("stats decode: %v (%s)", err, raw)
	}
	return doc.Store, doc.Blocks, doc.ArenaFreeBytes
}

// TestDRAMWireLifecycle drives the §3.3/§3.5 lease ladder over real
// connections: GET auto-leases, a plain DELETE is refused with ERR_LEASED,
// RELEASE drops the lease, and the delete then passes.
func TestDRAMWireLifecycle(t *testing.T) {
	addr, cleanup := startDRAMServer(t, 64<<20, dramDefaults())
	defer cleanup()
	c := dialClient(t, addr, 2)
	defer c.Close()
	ctx := context.Background()

	k := key(0xD1)
	blob := bytes.Repeat([]byte{0x3C}, 1<<20)
	if err := c.Put(ctx, k, blob); err != nil {
		t.Fatal(err)
	}

	into := make([][]byte, 1)
	sts, err := c.BatchGet(ctx, [][32]byte{k}, into)
	if err != nil || sts[0] != protocol.StatusOK {
		t.Fatalf("get: %v %v", sts, err)
	}
	if !bytes.Equal(into[0], blob) {
		t.Fatal("wrong bytes over the wire")
	}

	per, err := c.Delete(ctx, [][32]byte{k}, false)
	if err != nil {
		t.Fatal(err)
	}
	if per[0] != protocol.StatusErrLeased {
		t.Fatalf("delete under the auto-lease: got %s, want ERR_LEASED", per[0])
	}

	if per, err = c.TouchLease(ctx, [][32]byte{k}, protocol.LeaseRelease, 0); err != nil || per[0] != protocol.StatusOK {
		t.Fatalf("lease release: %v %v", per, err)
	}
	if per, err = c.Delete(ctx, [][32]byte{k}, false); err != nil || per[0] != protocol.StatusOK {
		t.Fatalf("clean delete: %v %v", per, err)
	}

	store, blocks, _ := dramStats(t, c)
	if store != "dram" || blocks != 0 {
		t.Fatalf("stats after delete: store=%q blocks=%d", store, blocks)
	}
}

// TestDRAMWirePin drives §3.6 over the wire: a hard pin survives F_FORCE,
// the per-namespace pinned-bytes cap answers ERR_PIN_QUOTA, and UNPIN
// releases both the block and its quota charge.
func TestDRAMWirePin(t *testing.T) {
	p := dramDefaults()
	p.PinnedBytesCap = 1 << 20 // room for exactly one 1 MiB block
	addr, cleanup := startDRAMServer(t, 32<<20, p)
	defer cleanup()
	c := dialClient(t, addr, 2)
	defer c.Close()
	ctx := context.Background()

	k1, k2 := key(0xE1), key(0xE2)
	blob := bytes.Repeat([]byte{0x7E}, 1<<20)
	for _, k := range [][32]byte{k1, k2} {
		if err := c.Put(ctx, k, blob); err != nil {
			t.Fatal(err)
		}
	}

	per, err := c.Pin(ctx, [][32]byte{k1}, protocol.PinHard)
	if err != nil || per[0] != protocol.StatusOK {
		t.Fatalf("hard pin: %v %v", per, err)
	}
	// F_FORCE overrides leases and soft pins — never a hard pin.
	if per, err = c.Delete(ctx, [][32]byte{k1}, true); err != nil || per[0] != protocol.StatusErrPinned {
		t.Fatalf("forced delete of a hard pin: %v %v", per, err)
	}
	// §3.6: soft pins are QUOTA-FREE — they succeed even with the hard-pin
	// cap exhausted by k1 (under quota emergency they are dropped, never
	// rejected).
	if per, err = c.Pin(ctx, [][32]byte{k2}, protocol.PinSoft); err != nil || per[0] != protocol.StatusOK {
		t.Fatalf("soft pin at a full cap: %v %v", per, err)
	}
	// A soft→hard UPGRADE passes the quota gate and is refused at the cap.
	if per, err = c.Pin(ctx, [][32]byte{k2}, protocol.PinHard); err != nil || per[0] != protocol.StatusErrPinQuota {
		t.Fatalf("over-cap soft→hard upgrade: %v %v", per, err)
	}
	// UNPIN refunds the quota; the upgrade then fits, and cleanup passes.
	if per, err = c.Pin(ctx, [][32]byte{k1}, protocol.Unpin); err != nil || per[0] != protocol.StatusOK {
		t.Fatalf("unpin: %v %v", per, err)
	}
	if per, err = c.Pin(ctx, [][32]byte{k2}, protocol.PinHard); err != nil || per[0] != protocol.StatusOK {
		t.Fatalf("upgrade after refund: %v %v", per, err)
	}
	if per, err = c.Pin(ctx, [][32]byte{k2}, protocol.Unpin); err != nil || per[0] != protocol.StatusOK {
		t.Fatalf("unpin k2: %v %v", per, err)
	}
	for _, k := range [][32]byte{k1, k2} {
		if per, err = c.Delete(ctx, [][32]byte{k}, false); err != nil || per[0] != protocol.StatusOK {
			t.Fatalf("delete after unpin: %v %v", per, err)
		}
	}
}

// TestDRAMWireFMoreSplitReleases is the release-aggregation crux over the
// wire: a 20 MiB response split across F_MORE frames holds one reader ref per
// packed block, each frame's combined release fires after its writev, and a
// forced delete of everything returns the arena to fully free — proving no
// release leaked and none fired early (the extents outlived the flush).
func TestDRAMWireFMoreSplitReleases(t *testing.T) {
	const arenaBytes = 64 << 20
	addr, cleanup := startDRAMServer(t, arenaBytes, dramDefaults())
	defer cleanup()
	// Propose the §4 floor so a 20 MiB payload MUST split (floor is 16 MiB).
	c, err := client.Dial(context.Background(), addr, client.Options{
		Streams: 1, Namespace: "tenant-a", Token: testToken,
		MaxFrameLen: protocol.FloorMaxFrameLen,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	ctx := context.Background()

	const n = 20
	keys := make([][32]byte, n)
	blobs := make([][]byte, n)
	for i := range keys {
		binary.LittleEndian.PutUint32(keys[i][:], uint32(i)+500) //nolint:gosec // G115: test index
		blobs[i] = bytes.Repeat([]byte{byte(i + 1)}, 1<<20)
		if err := c.Put(ctx, keys[i], blobs[i]); err != nil {
			t.Fatal(err)
		}
	}
	into := make([][]byte, n)
	statuses, err := c.BatchGet(ctx, keys, into)
	if err != nil {
		t.Fatal(err)
	}
	for i := range keys {
		if statuses[i] != protocol.StatusOK || !bytes.Equal(into[i], blobs[i]) {
			t.Fatalf("key %d: status %s or wrong bytes", i, statuses[i])
		}
	}

	// Force past the auto-leases; every delete must succeed.
	per, err := c.Delete(ctx, keys, true)
	if err != nil {
		t.Fatal(err)
	}
	for i, st := range per {
		if st != protocol.StatusOK {
			t.Fatalf("forced delete %d: %s", i, st)
		}
	}

	// The frame releases fire on the server's writer goroutine after each
	// writev — concurrent with our reads — so poll briefly for the last one.
	deadline := time.Now().Add(5 * time.Second)
	for {
		store, blocks, free := dramStats(t, c)
		if store == "dram" && blocks == 0 && free == arenaBytes {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("extents not all freed: blocks=%d arena_free=%d want free=%d — a GET release leaked", blocks, free, arenaBytes)
		}
		time.Sleep(5 * time.Millisecond)
	}
}
