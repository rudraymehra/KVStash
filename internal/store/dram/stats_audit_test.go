package dram

import (
	"testing"

	"github.com/zeebo/xxh3"

	"github.com/kvstash/kvblockd/internal/eviction"
	"github.com/kvstash/kvblockd/internal/protocol"
)

// TestIndexStatsCountersMatchWalk audits the funnel-maintained blocks/bytes
// counters (they replaced an O(total blocks) Range under every shard lock
// per scrape) against the walk they replaced, across every mutation funnel:
// Put, Delete, DeleteIf (the evictor's removal), and lost-race Put.
func TestIndexStatsCountersMatchWalk(t *testing.T) {
	arena, err := NewArena(16<<20, false)
	if err != nil {
		t.Fatal(err)
	}
	s := New(arena, Params{LeaseDefaultMS: 5, LeaseMaxMS: 60000})
	t.Cleanup(func() { _ = s.Close() })

	audit := func(phase string) {
		t.Helper()
		var walkBlocks, walkBytes int64
		s.index.Range(func(_ Key, ref *BlockRef) bool {
			walkBlocks++
			walkBytes += int64(ref.Len)
			return true
		})
		blocks, bytes := s.index.Stats()
		if blocks != walkBlocks || bytes != walkBytes {
			t.Fatalf("%s: counters (%d blocks, %d bytes) drifted from the walk (%d, %d)",
				phase, blocks, bytes, walkBlocks, walkBytes)
		}
	}

	key := func(i int) (k [32]byte) {
		k[0], k[1], k[2] = byte(i), byte(i>>8), 0xA5 //nolint:gosec // G115: deliberate wrap — test key pattern
		return k
	}
	blob := func(i, n int) []byte {
		b := make([]byte, n)
		for j := range b {
			b[j] = byte(i ^ j) //nolint:gosec // G115: deliberate wrap — test pattern
		}
		return b
	}

	for i := 0; i < 64; i++ {
		b := blob(i, 4096+(i%5)*512)
		if st := s.Put(1, key(i), b, xxh3.Hash(b)); st != protocol.StatusOK {
			t.Fatalf("put %d: %s", i, st)
		}
	}
	audit("after puts")

	// Duplicate puts lose the race (OK_EXISTS) — counters must not move.
	before, _ := s.index.Stats()
	for i := 0; i < 8; i++ {
		b := blob(i, 4096+(i%5)*512)
		if st := s.Put(1, key(i), b, xxh3.Hash(b)); st != protocol.StatusOKExists {
			t.Fatalf("dup put %d: %s", i, st)
		}
	}
	if after, _ := s.index.Stats(); after != before {
		t.Fatalf("lost-race puts moved the block counter %d → %d", before, after)
	}
	audit("after duplicate puts")

	for i := 0; i < 64; i += 2 {
		if st := s.Delete(1, key(i), false); st != protocol.StatusOK {
			t.Fatalf("delete %d: %s", i, st)
		}
	}
	audit("after deletes")

	// Evictor removals go through DeleteIf — the third funnel.
	s.EvictNow()
	audit("after eviction")
}

// TestStatsSnapshotSurfacesGhostBytes: the policy PRODUCES GhostBytes but a
// number no stats document carries is invisible to every operator surface —
// the snapshot must sum it (scrape gauge kvb_eviction_ghost_bytes).
func TestStatsSnapshotSurfacesGhostBytes(t *testing.T) {
	arena, err := NewArena(16<<20, false)
	if err != nil {
		t.Fatal(err)
	}
	s := New(arena, Params{LeaseDefaultMS: 5, LeaseMaxMS: 60000})
	t.Cleanup(func() { _ = s.Close() })

	if got := s.StatsSnapshot().EvictionGhostBytes; got != 0 {
		t.Fatalf("ghost bytes %d with no policy attached, want 0", got)
	}

	pol := eviction.NewS3FIFO(4096)
	s.AttachPolicy(pol)
	k := eviction.Key{NS: 3}
	k.Hash[0] = 0x7E
	pol.Admit(k, 100, 0)
	if v := pol.Victims(3, 100, 0, nil); len(v) != 1 {
		t.Fatalf("fixture: eviction did not ghost the key: %+v", v)
	}
	if got := s.StatsSnapshot().EvictionGhostBytes; got <= 0 {
		t.Fatalf("ghost bytes %d with a materialized ghost ring, want > 0", got)
	}
}
