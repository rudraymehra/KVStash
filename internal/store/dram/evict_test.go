package dram_test

import (
	"bytes"
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/zeebo/xxh3"

	"github.com/kvstash/kvblockd/internal/eviction"
	"github.com/kvstash/kvblockd/internal/protocol"
	"github.com/kvstash/kvblockd/internal/store/dram"
	"github.com/kvstash/kvblockd/internal/tenant"
)

// evictStore boots a tier with a fake clock and an attached (but not
// started) policy — the deterministic shape: eviction only via EvictNow.
func evictStore(t *testing.T, arenaBytes int64, policyName string) (*dram.Store, *int64) {
	t.Helper()
	cur := int64(1_000_000_000_000)
	arena, err := dram.NewArena(arenaBytes, false)
	if err != nil {
		t.Fatal(err)
	}
	pol, err := eviction.New(policyName, 4096)
	if err != nil {
		t.Fatal(err)
	}
	s := dram.New(arena, dram.Params{
		LeaseDefaultMS: 5000, LeaseMaxMS: 60000,
		Now: func() int64 { return cur },
	})
	s.AttachPolicy(pol)
	t.Cleanup(func() { _ = s.Close() })
	return s, &cur
}

func evKey(b byte) [32]byte {
	var k [32]byte
	k[0], k[9], k[31] = b, b, 0xE7
	return k
}

func evPut(t *testing.T, s *dram.Store, ns uint32, k [32]byte, n int) []byte {
	t.Helper()
	blob := bytes.Repeat([]byte{k[0] ^ byte(ns)}, n) //nolint:gosec // G115: byte mixing
	if st := s.Put(ns, k, blob, xxh3.Hash(blob)); st != protocol.StatusOK {
		t.Fatalf("put: %s", st)
	}
	return blob
}

func statsDoc(t *testing.T, s *dram.Store) map[string]any {
	t.Helper()
	var doc map[string]any
	if err := json.Unmarshal(s.Stats(), &doc); err != nil {
		t.Fatal(err)
	}
	return doc
}

// TestEvictionLadderSurvival is the pairwise protection table through a
// REAL eviction pass: hard pins, live leases, and read-held blocks survive
// arbitrarily heavy pressure; a plain block does not; a lease lapsing
// re-enables eviction.
func TestEvictionLadderSurvival(t *testing.T) {
	for _, policy := range []string{"s3fifo", "sampled-lru"} {
		t.Run(policy, func(t *testing.T) {
			s, clk := evictStore(t, 8<<20, policy) // 8 MiB arena
			const blk = 1 << 20

			hard, leased, held, plain := evKey(1), evKey(2), evKey(3), evKey(4)
			evPut(t, s, 1, hard, blk)
			evPut(t, s, 1, leased, blk)
			heldWant := evPut(t, s, 1, held, blk)
			evPut(t, s, 1, plain, blk)

			if st := s.PinOp(1, hard, protocol.PinHard); st != protocol.StatusOK {
				t.Fatal(st)
			}
			if st := s.TouchLease(1, leased, protocol.LeaseGrant, 60_000); st != protocol.StatusOK {
				t.Fatal(st)
			}
			view, sum, release, ok := s.GetRef(1, held)
			if !ok || sum != xxh3.Hash(heldWant) {
				t.Fatal("getref")
			}
			// The GetRef auto-leased `held` too; advance past that lease so
			// ONLY the refcount protects it (the sharper claim). The 60s
			// explicit lease on `leased` survives the advance.
			*clk += int64(30_000) * int64(time.Millisecond)

			// Fill to the wall so the watermark is far exceeded, then evict.
			for i := byte(10); ; i++ {
				var k [32]byte
				k[0], k[9] = i, i
				blob := bytes.Repeat([]byte{i}, blk)
				if st := s.Put(1, k, blob, xxh3.Hash(blob)); st != protocol.StatusOK {
					break
				}
			}
			s.EvictNow()

			if !s.Contains(1, hard) {
				t.Fatal("hard pin evicted")
			}
			if !s.Contains(1, leased) {
				t.Fatal("leased block evicted mid-lease")
			}
			if !s.Contains(1, held) {
				t.Fatal("read-held block evicted")
			}
			// I5's core: the held view is byte-stable across the pass.
			if !bytes.Equal(view, heldWant) {
				t.Fatal("held view torn by eviction")
			}
			release()

			// The plain block must be evictable — under pressure SOMETHING
			// unprotected went; assert the evictor freed real bytes.
			if got := statsDoc(t, s); got["evictions_total"].(float64) == 0 {
				t.Fatal("no evictions under pressure")
			}

			// Lease lapse re-enables eviction: advance past 60s, pressure again.
			*clk += int64(61_000) * int64(time.Millisecond)
			s.EvictNow()
			if s.Contains(1, leased) {
				// Not necessarily evicted (need may be zero now) — force the
				// question: delete must now see it unprotected too.
				if st := s.Delete(1, leased, false); st != protocol.StatusOK {
					t.Fatalf("lapsed lease still protecting: %s", st)
				}
			}
			if !s.Contains(1, hard) {
				t.Fatal("hard pin fell to a second pass")
			}
		})
	}
}

// TestExpiredFirst: expired-TTL blocks are the preferred victims — an
// expired block dies while a fresher unexpired one of equal standing
// survives a small-need pass.
func TestExpiredFirst(t *testing.T) {
	s, clk := evictStore(t, 8<<20, "s3fifo")
	const blk = 1 << 20
	expired, fresh := evKey(1), evKey(2)
	evPut(t, s, 2, expired, blk)
	evPut(t, s, 2, fresh, blk)
	if st := s.TouchLease(2, expired, protocol.TouchRecency, 1000); st != protocol.StatusOK {
		t.Fatal(st) // ttl 1s
	}
	*clk += int64(10_000) * int64(time.Millisecond) // ttl lapses

	// Fill to trip the watermark, then one pass.
	for i := byte(10); ; i++ {
		var k [32]byte
		k[0], k[9] = i, i
		blob := bytes.Repeat([]byte{i}, blk)
		if st := s.Put(2, k, blob, xxh3.Hash(blob)); st != protocol.StatusOK {
			break
		}
	}
	s.EvictNow()
	if s.Contains(2, expired) && !s.Contains(2, fresh) {
		t.Fatal("evictor preferred a fresh block over an expired one")
	}
	if s.Contains(2, expired) {
		t.Fatal("expired block survived a pressure pass")
	}
}

// TestEmergencySoftPinYield: soft pins survive NORMAL pressure (a first
// pass with unprotected blocks available never touches them) but yield in
// the genuine quota emergency — when everything else resident is lease/
// hard-pin protected and the tier still cannot get under the watermark.
// Hard pins and leases never yield.
func TestEmergencySoftPinYield(t *testing.T) {
	s, _ := evictStore(t, 4<<20, "s3fifo") // 4 MiB arena
	const blk = 1 << 20
	soft, hard, hard2, leased := evKey(1), evKey(2), evKey(3), evKey(4)
	evPut(t, s, 1, soft, blk)
	evPut(t, s, 1, hard, blk)
	evPut(t, s, 1, hard2, blk)
	evPut(t, s, 1, leased, blk)
	// First: soft pins are NOT normal-pressure victims. With nothing else
	// evictable and no emergency yet declared... force the state below.
	if st := s.PinOp(1, soft, protocol.PinSoft); st != protocol.StatusOK {
		t.Fatal(st)
	}
	if st := s.PinOp(1, hard, protocol.PinHard); st != protocol.StatusOK {
		t.Fatal(st)
	}
	if st := s.PinOp(1, hard2, protocol.PinHard); st != protocol.StatusOK {
		t.Fatal(st)
	}
	if st := s.TouchLease(1, leased, protocol.LeaseGrant, 60_000); st != protocol.StatusOK {
		t.Fatal(st)
	}
	// Arena 100% full; every block protected except the soft pin. The
	// normal policy pass refuses it (re-admission), so pass 3 declares the
	// quota emergency and the soft pin — and ONLY the soft pin — yields.
	s.EvictNow()
	if !s.Contains(1, hard) || !s.Contains(1, hard2) {
		t.Fatal("hard pin evicted in emergency")
	}
	if !s.Contains(1, leased) {
		t.Fatal("leased block evicted in emergency")
	}
	if s.Contains(1, soft) {
		t.Fatal("soft pin survived the quota emergency (§6: it must yield)")
	}
}

// TestEvictorPutRetry: with the evictor STARTED, an arena-full Put succeeds
// by reclaiming — the wall becomes backpressure only when nothing can go.
func TestEvictorPutRetry(t *testing.T) {
	s, _ := evictStore(t, 4<<20, "s3fifo")
	stop := s.StartEvictor(context.Background(), dram.EvictorConfig{
		WatermarkPct: 95, BatchPct: 20, Interval: time.Hour, // ticker inert; the synchronous retry is under test
	})
	defer stop()
	const blk = 1 << 20
	// Fill the arena completely.
	for i := byte(1); i <= 4; i++ {
		var k [32]byte
		k[0], k[9] = i, i
		blob := bytes.Repeat([]byte{i}, blk)
		if st := s.Put(1, k, blob, xxh3.Hash(blob)); st != protocol.StatusOK {
			break
		}
	}
	// One more block: without the evictor this is ERR_QUOTA_BYTES (the
	// Week-3 wall test); with it, reclamation makes room.
	fresh := evKey(0xAA)
	blob := bytes.Repeat([]byte{0xAA}, blk)
	if st := s.Put(1, fresh, blob, xxh3.Hash(blob)); st != protocol.StatusOK {
		t.Fatalf("put at the wall with a live evictor: %s", st)
	}
	if !s.Contains(1, fresh) {
		t.Fatal("retried put not visible")
	}
}

// TestGetRefAllocParity: attaching a policy adds ZERO allocations to the
// GET hot path (the enforceable form of the zero-alloc gate — GetRef's
// release closure is the pre-existing baseline).
func TestGetRefAllocParity(t *testing.T) {
	measure := func(s *dram.Store, key [32]byte) float64 {
		return testing.AllocsPerRun(2000, func() {
			_, _, release, ok := s.GetRef(1, key)
			if !ok {
				t.Fatal("miss")
			}
			release()
		})
	}
	arena1, err := dram.NewArena(8<<20, false)
	if err != nil {
		t.Fatal(err)
	}
	bare := dram.New(arena1, dram.Params{LeaseDefaultMS: 5000, LeaseMaxMS: 60000})
	defer bare.Close()
	k := evKey(1)
	blob := bytes.Repeat([]byte{1}, 4096)
	if st := bare.Put(1, k, blob, xxh3.Hash(blob)); st != protocol.StatusOK {
		t.Fatal(st)
	}
	baseline := measure(bare, k)

	for _, policy := range []string{"s3fifo", "sampled-lru"} {
		t.Run(policy, func(t *testing.T) {
			s, _ := evictStore(t, 8<<20, policy)
			evPut(t, s, 1, k, 4096)
			if got := measure(s, k); got != baseline {
				t.Fatalf("GetRef allocs with %s = %.1f, baseline %.1f — the policy hook allocated", policy, got, baseline)
			}
		})
	}
	if baseline > 2 {
		t.Fatalf("GetRef baseline allocs = %.1f — a closure regression predates the policy", baseline)
	}
}

// TestEvictionStatsShape: the new Stats fields exist and move.
func TestEvictionStatsShape(t *testing.T) {
	s, _ := evictStore(t, 4<<20, "s3fifo")
	evPut(t, s, 1, evKey(1), 1<<20)
	doc := statsDoc(t, s)
	if doc["live_allocs"].(float64) != 1 {
		t.Fatalf("live_allocs = %v, want 1", doc["live_allocs"])
	}
	if doc["max_allocs"].(float64) <= 0 || doc["evictions_total"].(float64) != 0 {
		t.Fatalf("stats shape: %v", doc)
	}
}

// TestNoPhantomReadmit is the regression for the phantom-candidate leak: a
// policy candidate whose key was concurrently deleted must be DROPPED (not
// re-admitted), or the policy loops it forever, inflating Usage and eating
// the eviction budget. Simulated by admitting a key straight into the
// policy that the store never held.
func TestNoPhantomReadmit(t *testing.T) {
	s, _ := evictStore(t, 4<<20, "s3fifo")
	pol, err := eviction.New("s3fifo", 4096)
	if err != nil {
		t.Fatal(err)
	}
	s.AttachPolicy(pol)
	var phantom [32]byte
	phantom[0], phantom[31] = 0x66, 0x66
	pol.Admit(eviction.Key{NS: 5, Hash: phantom}, 1<<20, 0) // never in the index

	// Fill ns 5 past the watermark so the policy pass runs against it.
	const blk = 1 << 20
	for i := byte(10); ; i++ {
		var k [32]byte
		k[0], k[9] = i, i
		blob := bytes.Repeat([]byte{i}, blk)
		if st := s.Put(5, k, blob, xxh3.Hash(blob)); st != protocol.StatusOK {
			break
		}
	}
	s.EvictNow()
	s.EvictNow() // a second pass: the phantom must not still be circulating

	usage := pol.Usage(nil)
	var ns5 int64
	for _, u := range usage {
		if u.NS == 5 {
			ns5 = u.Bytes
		}
	}
	// The phantom's 1 MiB must be gone from the policy's books. Real
	// resident blocks may remain (under the watermark now).
	var doc struct {
		Bytes int64 `json:"bytes"`
	}
	if err := json.Unmarshal(s.Stats(), &doc); err != nil {
		t.Fatal(err)
	}
	if ns5 > doc.Bytes {
		t.Fatalf("policy books %d bytes for ns5 but the store holds only %d — the phantom was re-admitted", ns5, doc.Bytes)
	}
}

// TestZeroLenFloodBounded is the regression for the confirmed empty-block
// DoS: zero-length blocks (legal, extent-less) used to escape every
// eviction trigger — the index grew ~181 B/block without bound. They now
// carry a nominal liveAllocs slot, trip the count trigger, and the
// emergency sweep's COUNT goal reclaims them.
func TestZeroLenFloodBounded(t *testing.T) {
	s, _ := evictStore(t, 64<<20, "s3fifo") // pool = max(units/16,1024) = 1024 slots
	sum := xxh3.Hash(nil)
	for i := 0; i < 1024; i++ {
		var k [32]byte
		k[0], k[1], k[2] = byte(i), byte(i>>8), 0x5A
		if st := s.Put(3, k, nil, sum); st != protocol.StatusOK {
			t.Fatalf("zero-len put %d: %s", i, st)
		}
	}
	before := statsDoc(t, s)["live_allocs"].(float64)
	if before != 1024 {
		t.Fatalf("live_allocs = %v, want 1024 (nominal slots)", before)
	}
	s.EvictNow()
	after := statsDoc(t, s)["live_allocs"].(float64)
	if after >= before {
		t.Fatalf("zero-len flood not reclaimed: live_allocs %v -> %v", before, after)
	}
}

// TestOverQuotaTenantEvictedFirst: A at 120% of its DRAM quota, B within
// budget — under watermark pressure every victim until A is back inside
// its quota must be A's, and B's blocks all survive the pass.
func TestOverQuotaTenantEvictedFirst(t *testing.T) {
	cur := int64(1_000_000_000_000)
	// 9 x 64 KiB blocks on a 600 KiB arena = ~96% occupancy: over the 95%
	// watermark, so EvictNow's pass genuinely runs; the pass frees to the
	// 90% floor (~36 KiB), all of which must come from the over-quota
	// tenant.
	arena, err := dram.NewArena(600<<10, false)
	if err != nil {
		t.Fatal(err)
	}
	pol, err := eviction.New("s3fifo", 4096)
	if err != nil {
		t.Fatal(err)
	}
	reg := tenant.NewRegistry("a", 1, "ta")
	if err := reg.Add(&tenant.Namespace{ID: 2, Name: "b", TokenHash: [32]byte{1}}); err != nil {
		t.Fatal(err)
	}
	reg.SetQuota("a", tenant.TierDRAM, 5*64<<10) // A's quota: 5 blocks
	q := tenant.NewQuotas(reg)
	s := dram.New(arena, dram.Params{
		LeaseDefaultMS: 100, LeaseMaxMS: 60000,
		Now: func() int64 { return cur }, Quotas: q,
	})
	s.AttachPolicy(pol)
	t.Cleanup(func() { _ = s.Close() })

	// A stores 5 blocks inside quota; B stores 4 (unlimited). Then A's
	// quota is TIGHTENED to 2 blocks — A is now at 250%, no eviction ran.
	blk := 64 << 10
	for i := 0; i < 5; i++ {
		evPut(t, s, 1, evKey(byte(0x10+i)), blk)
	}
	for i := 0; i < 4; i++ {
		evPut(t, s, 2, evKey(byte(0x40+i)), blk)
	}
	cur += int64(200 * time.Millisecond) // auto-leases lapse: everything evictable
	reg.SetQuota("a", tenant.TierDRAM, int64(2*blk))
	q.Reload()

	before := q.Usage(1, tenant.TierDRAM)
	s.EvictNow()

	// The pass only needs ~36 KiB to reach the floor — but every byte of
	// it must be A's (the over-quota-first round), so A shrinks...
	if got := q.Usage(1, tenant.TierDRAM); got >= before {
		t.Fatalf("over-quota tenant A lost nothing: %d -> %d", before, got)
	}
	for i := 0; i < 4; i++ {
		if !s.Contains(2, evKey(byte(0x40+i))) {
			t.Fatalf("within-budget tenant B lost block %d while A was over quota", i)
		}
	}
}
