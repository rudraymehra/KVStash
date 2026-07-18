package dram_test

import (
	"bytes"
	"sync/atomic"
	"testing"

	"github.com/zeebo/xxh3"

	"github.com/kvstash/kvblockd/internal/eviction"
	"github.com/kvstash/kvblockd/internal/protocol"
	"github.com/kvstash/kvblockd/internal/store/dram"
)

// fake clock helper: a store whose lease/TTL decisions the test drives.
func newClockStore(t *testing.T, cur *atomic.Int64) *dram.Store {
	t.Helper()
	arena, err := dram.NewArena(16<<20, false)
	if err != nil {
		t.Fatal(err)
	}
	s := dram.New(arena, dram.Params{
		LeaseDefaultMS: 100, LeaseMaxMS: 60000,
		Now: cur.Load,
	})
	t.Cleanup(func() { _ = s.Close() })
	return s
}

const msNanos = int64(1_000_000)

func TestRefForDemotionSemantics(t *testing.T) {
	var cur atomic.Int64
	cur.Store(1)
	s := newClockStore(t, &cur)

	blob := bytes.Repeat([]byte{0x42}, 32<<10)
	sum := xxh3.Hash(blob)
	if st := s.Put(1, k(1), blob, sum); st != protocol.StatusOK {
		t.Fatal(st)
	}
	// Two client GETs → hits=2; each grants an auto-lease.
	for i := 0; i < 2; i++ {
		_, _, rel, ok := s.GetRef(1, k(1))
		if !ok {
			t.Fatal("GetRef miss")
		}
		rel()
	}

	data, gotSum, hits, rel, ok := s.RefForDemotion(1, k(1))
	if !ok {
		t.Fatal("RefForDemotion refused a resident block")
	}
	if gotSum != sum || !bytes.Equal(data, blob) {
		t.Fatal("demotion view mismatch")
	}
	if hits != 2 {
		t.Fatalf("hits = %d, want 2 (demotion reads must not count)", hits)
	}
	rel()

	// A second demotion ref: hits unchanged — RefForDemotion is not a hit.
	_, _, hits2, rel2, _ := s.RefForDemotion(1, k(1))
	if hits2 != 2 {
		t.Fatalf("hits after demotion read = %d, want 2", hits2)
	}
	rel2()

	// Missing and zero-length blocks refuse.
	if _, _, _, _, ok := s.RefForDemotion(1, k(9)); ok {
		t.Fatal("missing block accepted")
	}
	if st := s.Put(1, k(2), nil, xxh3.Hash(nil)); st != protocol.StatusOK {
		t.Fatal(st)
	}
	if _, _, _, _, ok := s.RefForDemotion(1, k(2)); ok {
		t.Fatal("zero-length block accepted for demotion")
	}
}

func TestCompleteDemotionGateMatrix(t *testing.T) {
	blob := bytes.Repeat([]byte{0x33}, 16<<10)
	sum := xxh3.Hash(blob)

	setup := func(t *testing.T) (*dram.Store, *atomic.Int64) {
		var cur atomic.Int64
		cur.Store(1)
		s := newClockStore(t, &cur)
		if st := s.Put(1, k(1), blob, sum); st != protocol.StatusOK {
			t.Fatal(st)
		}
		return s, &cur
	}

	t.Run("happy path publishes inside the gate and removes", func(t *testing.T) {
		s, _ := setup(t)
		published := false
		if !s.CompleteDemotion(1, k(1), sum, func(ref *dram.BlockRef) {
			published = true
			if ref.Len != uint32(len(blob)) || ref.XXH3 != sum { //nolint:gosec // G115: test blob ≪ 4 GiB
				t.Error("publish saw the wrong ref")
			}
		}) {
			t.Fatal("gate refused an evictable block")
		}
		if !published {
			t.Fatal("publish never ran")
		}
		if s.Contains(1, k(1)) {
			t.Fatal("block still resident after demotion")
		}
	})

	t.Run("xxh3 mismatch refuses (content swapped mid-queue)", func(t *testing.T) {
		s, _ := setup(t)
		if s.CompleteDemotion(1, k(1), sum^1, nil) {
			t.Fatal("gate passed a stale digest")
		}
		if !s.Contains(1, k(1)) {
			t.Fatal("refused demotion removed the block")
		}
	})

	t.Run("auto-lease from a GET refuses until expiry", func(t *testing.T) {
		s, cur := setup(t)
		_, _, rel, _ := s.GetRef(1, k(1)) // leases for 100ms
		rel()
		if s.CompleteDemotion(1, k(1), sum, nil) {
			t.Fatal("gate passed a leased block")
		}
		cur.Add(200 * msNanos) // lease lapses
		if !s.CompleteDemotion(1, k(1), sum, nil) {
			t.Fatal("gate refused after lease expiry")
		}
	})

	t.Run("read-held refuses (refcount > 1)", func(t *testing.T) {
		s, cur := setup(t)
		_, _, rel, _ := s.GetRef(1, k(1))
		cur.Add(200 * msNanos) // lease lapsed, but the view is still held
		if s.CompleteDemotion(1, k(1), sum, nil) {
			t.Fatal("gate passed a read-held block")
		}
		rel()
		if !s.CompleteDemotion(1, k(1), sum, nil) {
			t.Fatal("gate refused after release")
		}
	})

	t.Run("pin refuses", func(t *testing.T) {
		s, _ := setup(t)
		if st := s.PinOp(1, k(1), protocol.PinSoft); st != protocol.StatusOK {
			t.Fatal(st)
		}
		if s.CompleteDemotion(1, k(1), sum, nil) {
			t.Fatal("gate passed a pinned block")
		}
	})

	t.Run("vanished block returns false", func(t *testing.T) {
		s, _ := setup(t)
		if st := s.Delete(1, k(1), true); st != protocol.StatusOK {
			t.Fatal(st)
		}
		if s.CompleteDemotion(1, k(1), sum, nil) {
			t.Fatal("gate passed a deleted block")
		}
	})
}

// spyPolicy records Remove calls — CompleteDemotion must NEVER fire
// policy.Remove (Victims already dequeued the key; a Remove here would
// erase a concurrent re-Put's Admit).
type spyPolicy struct {
	eviction.Policy
	removes atomic.Int32
}

func (p *spyPolicy) Remove(k eviction.Key) {
	p.removes.Add(1)
	p.Policy.Remove(k)
}

func TestCompleteDemotionNeverCallsPolicyRemove(t *testing.T) {
	var cur atomic.Int64
	cur.Store(1)
	s := newClockStore(t, &cur)
	inner, err := eviction.New("s3fifo", 128)
	if err != nil {
		t.Fatal(err)
	}
	spy := &spyPolicy{Policy: inner}
	s.AttachPolicy(spy)

	blob := bytes.Repeat([]byte{0x11}, 8<<10)
	sum := xxh3.Hash(blob)
	if st := s.Put(1, k(1), blob, sum); st != protocol.StatusOK {
		t.Fatal(st)
	}
	if !s.CompleteDemotion(1, k(1), sum, func(*dram.BlockRef) {}) {
		t.Fatal("demotion refused")
	}
	if n := spy.removes.Load(); n != 0 {
		t.Fatalf("CompleteDemotion fired policy.Remove %d times", n)
	}
}
