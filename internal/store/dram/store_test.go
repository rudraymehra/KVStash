package dram_test

import (
	"bytes"
	"encoding/json"
	"sync"
	"testing"

	"github.com/zeebo/xxh3"

	"github.com/kvstash/kvblockd/internal/protocol"
	"github.com/kvstash/kvblockd/internal/store/dram"
	"github.com/kvstash/kvblockd/internal/store/storetest"
)

// newTierStore boots a small arena-backed tier (no hugepages — CI-portable)
// and closes the arena on test cleanup.
func newTierStore(t *testing.T, arenaBytes int64) *dram.Store {
	t.Helper()
	arena, err := dram.NewArena(arenaBytes, false)
	if err != nil {
		t.Fatal(err)
	}
	s := dram.New(arena, dram.Params{LeaseDefaultMS: 5000, LeaseMaxMS: 60000, PinnedBytesCap: 0})
	t.Cleanup(func() { _ = s.Close() })
	return s
}

// TestDRAMConformance: the tier must satisfy the exact semantics ramstub does.
func TestDRAMConformance(t *testing.T) {
	storetest.RunConformance(t, func(t *testing.T) storetest.Store {
		return newTierStore(t, 64<<20)
	})
}

func k(b byte) [32]byte {
	var kk [32]byte
	kk[0], kk[3] = b, b
	return kk
}

// TestTwoPhaseVisibility pins the atomic flip: a block is invisible until Put
// returns, and the staging buffer is reusable immediately after.
func TestTwoPhaseVisibility(t *testing.T) {
	s := newTierStore(t, 64<<20)
	blob := bytes.Repeat([]byte{0x77}, 1<<20)
	sum := xxh3.Hash(blob)

	// Invisible before Put.
	if s.Contains(1, k(1)) {
		t.Fatal("visible before Put")
	}
	// Hammer EXISTS concurrently while committing: every observation must be
	// all-or-nothing (either miss, or full content served).
	stop := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			select {
			case <-stop:
				return
			default:
			}
			if data, gotSum, ok := s.Get(1, k(1)); ok {
				if gotSum != sum || len(data) != len(blob) || data[0] != 0x77 || data[len(data)-1] != 0x77 {
					t.Error("partial/corrupt block observed mid-commit")
					return
				}
			}
		}
	}()
	if st := s.Put(1, k(1), blob, sum); st != protocol.StatusOK {
		t.Fatal(st)
	}
	close(stop)
	wg.Wait()

	// Staging buffer reusable after Put (the tier copies into the arena).
	for i := range blob {
		blob[i] = 0xFF
	}
	data, _, ok := s.Get(1, k(1))
	if !ok || data[0] != 0x77 {
		t.Fatal("committed content changed when the staging buffer was reused")
	}
}

// TestRefcountSurvivesDelete pins the crux: a live reader reference defers
// the extent free across a concurrent DELETE, and the release triggers it.
func TestRefcountSurvivesDelete(t *testing.T) {
	s := newTierStore(t, 64<<20)
	blob := bytes.Repeat([]byte{0x42}, 1<<20)
	sum := xxh3.Hash(blob)
	if st := s.Put(1, k(2), blob, sum); st != protocol.StatusOK {
		t.Fatal(st)
	}

	freeBefore := arenaFree(t, s)

	// Reader holds a reference (the wire GET shape).
	data, gotSum, release, ok := s.GetRef(1, k(2))
	if !ok || gotSum != sum {
		t.Fatal("GetRef miss")
	}
	if st := s.Delete(1, k(2), true); st != protocol.StatusOK { // force past the auto-lease
		t.Fatalf("delete: %s", st)
	}
	// Deleted from the index... but the extent must NOT be freed yet.
	if s.Contains(1, k(2)) {
		t.Fatal("still visible after delete")
	}
	// freeBefore was sampled AFTER the Put, so a correctly-deferred free means
	// the number is UNCHANGED here (the extent still charged to the block).
	if got := arenaFree(t, s); got != freeBefore {
		t.Fatalf("extent freed under a live reader ref: free %d, want unchanged %d", got, freeBefore)
	}
	// The bytes are still intact under the held view.
	if data[0] != 0x42 || data[len(data)-1] != 0x42 {
		t.Fatal("bytes torn under a live reference")
	}
	// Release: NOW the extent frees (+1 MiB back).
	release()
	if got := arenaFree(t, s); got != freeBefore+(1<<20) {
		t.Fatalf("extent not freed after release: free %d, want %d", got, freeBefore+(1<<20))
	}
}

// arenaFree extracts arena_free_bytes from Stats (avoids exporting internals).
func arenaFree(t *testing.T, s *dram.Store) int64 {
	t.Helper()
	var doc struct {
		ArenaFreeBytes int64 `json:"arena_free_bytes"`
	}
	if err := json.Unmarshal(s.Stats(), &doc); err != nil {
		t.Fatal(err)
	}
	return doc.ArenaFreeBytes
}

// TestArenaFullGraceful: a full arena answers ERR_QUOTA_BYTES and keeps
// serving reads (the hard-but-clean wall until the Week-4 evictor).
func TestArenaFullGraceful(t *testing.T) {
	s := newTierStore(t, 8<<20) // small arena
	blob := bytes.Repeat([]byte{9}, 1<<20)
	var stored int
	for i := 0; i < 64; i++ {
		var kk [32]byte
		kk[0], kk[1] = byte(i), 0xEE
		st := s.Put(1, kk, blob, xxh3.Hash(blob))
		if st == protocol.StatusOK {
			stored++
			continue
		}
		if st != protocol.StatusErrQuotaBytes {
			t.Fatalf("unexpected status at capacity: %s", st)
		}
		break
	}
	if stored == 0 || stored > 8 {
		t.Fatalf("stored %d 1MiB blocks in an 8MiB arena", stored)
	}
	// Reads still work at the wall.
	var kk [32]byte
	kk[1] = 0xEE
	if _, _, ok := s.Get(1, kk); !ok {
		t.Fatal("read failed at the capacity wall")
	}
	// Delete frees space → a new Put fits again.
	if st := s.Delete(1, kk, true); st != protocol.StatusOK {
		t.Fatal(st)
	}
	var fresh [32]byte
	fresh[0] = 0xFD
	if st := s.Put(1, fresh, blob, xxh3.Hash(blob)); st != protocol.StatusOK {
		t.Fatalf("put after delete at the wall: %s", st)
	}
}

// TestDeleteGatingThroughStore drives the lifecycle gates through the Store
// surface (the wire shape): auto-lease → ERR_LEASED, hard pin → ERR_PINNED
// even forced, unpin+release → clean delete.
func TestDeleteGatingThroughStore(t *testing.T) {
	s := newTierStore(t, 16<<20)
	blob := []byte("protected")
	if st := s.Put(1, k(9), blob, xxh3.Hash(blob)); st != protocol.StatusOK {
		t.Fatal(st)
	}
	// A GET auto-leases → non-forced delete must answer ERR_LEASED.
	if _, _, ok := s.Get(1, k(9)); !ok {
		t.Fatal("get")
	}
	if st := s.Delete(1, k(9), false); st != protocol.StatusErrLeased {
		t.Fatalf("leased delete: got %s", st)
	}
	// Hard pin: force delete still refused.
	if st := s.PinOp(1, k(9), protocol.PinHard); st != protocol.StatusOK {
		t.Fatal(st)
	}
	if st := s.Delete(1, k(9), true); st != protocol.StatusErrPinned {
		t.Fatalf("hard-pinned force delete: got %s", st)
	}
	// Unpin + release lease → delete passes.
	if st := s.PinOp(1, k(9), protocol.Unpin); st != protocol.StatusOK {
		t.Fatal(st)
	}
	if st := s.TouchLease(1, k(9), protocol.LeaseRelease, 0); st != protocol.StatusOK {
		t.Fatal(st)
	}
	if st := s.Delete(1, k(9), false); st != protocol.StatusOK {
		t.Fatalf("clean delete: got %s", st)
	}
}

// TestPinCapForConstructorPath: the per-namespace pin_quota override wired
// through Params (the path main.go actually travels — a post-construction
// field write would bypass New exactly like the withDefaults regression the
// ladder caught once already).
func TestPinCapForConstructorPath(t *testing.T) {
	arena, err := dram.NewArena(16<<20, false)
	if err != nil {
		t.Fatal(err)
	}
	s := dram.New(arena, dram.Params{
		LeaseDefaultMS: 5000, LeaseMaxMS: 60000,
		PinnedBytesCap: 1 << 20,
		PinCapFor: func(ns uint32) int64 {
			if ns == 1 {
				return 64 << 10
			}
			return 0
		},
	})
	t.Cleanup(func() { _ = s.Close() })

	blob := make([]byte, 128<<10) // 128 KiB: over ns1's 64 KiB override, under the global 1 MiB
	if st := s.Put(1, k(200), blob, xxh3.Hash(blob)); st != protocol.StatusOK {
		t.Fatal(st)
	}
	if st := s.Put(2, k(201), blob, xxh3.Hash(blob)); st != protocol.StatusOK {
		t.Fatal(st)
	}
	if st := s.PinOp(1, k(200), protocol.PinHard); st != protocol.StatusErrPinQuota {
		t.Fatalf("ns1 override ignored on the wire path: got %s, want ERR_PIN_QUOTA", st)
	}
	if st := s.PinOp(2, k(201), protocol.PinHard); st != protocol.StatusOK {
		t.Fatalf("ns2 global fallback: %s", st)
	}
}
