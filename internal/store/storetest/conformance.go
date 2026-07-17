// Package storetest is the shared Store conformance suite: every block-store
// implementation (ramstub today, the DRAM tier, later NVMe/S3) must pass the
// same semantics — write-once, namespace isolation, prefix probing, delete,
// stats shape, and the ownership CONTRACT (a store may alias or copy the
// committed bytes; either way the committed CONTENT is what Get returns).
package storetest

import (
	"bytes"
	"strings"
	"testing"

	"github.com/zeebo/xxh3"

	"github.com/kvstash/kvblockd/internal/protocol"
)

// Store is a local structural mirror of the server's Store interface (no
// import of internal/server — avoids a dependency cycle and keeps this suite
// usable from any store package).
type Store interface {
	ExistsPrefix(ns uint32, keys [][32]byte, withBitmap bool) (uint32, []protocol.Status)
	Get(ns uint32, key [32]byte) ([]byte, uint64, bool)
	Put(ns uint32, key [32]byte, data []byte, xxh3 uint64) protocol.Status
	Contains(ns uint32, key [32]byte) bool
	Delete(ns uint32, key [32]byte, force bool) protocol.Status
	Stats() []byte
}

func key(b byte) [32]byte {
	var k [32]byte
	k[0], k[1] = b, b
	return k
}

// put stores data (computing its digest) and asserts the expected status.
func put(t *testing.T, s Store, ns uint32, k [32]byte, data []byte, want protocol.Status) {
	t.Helper()
	if got := s.Put(ns, k, data, xxh3.Hash(data)); got != want {
		t.Fatalf("Put: got %s, want %s", got, want)
	}
}

// RunConformance drives the full semantics table against a fresh store from
// the factory. Call it from each implementation's _test.go.
func RunConformance(t *testing.T, newStore func(t *testing.T) Store) {
	t.Run("WriteOnceMatrix", func(t *testing.T) {
		s := newStore(t)
		blobA := bytes.Repeat([]byte{0xA1}, 4096)
		put(t, s, 7, key(1), blobA, protocol.StatusOK)
		// Same bytes → idempotent hit.
		put(t, s, 7, key(1), blobA, protocol.StatusOKExists)
		// Different bytes under the same key → corruption alarm; original intact.
		blobB := bytes.Repeat([]byte{0xB2}, 4096)
		if got := s.Put(7, key(1), blobB, xxh3.Hash(blobB)); got != protocol.StatusErrImmutableConflict {
			t.Fatalf("conflicting re-put: got %s", got)
		}
		data, sum, ok := s.Get(7, key(1))
		if !ok || sum != xxh3.Hash(blobA) || !bytes.Equal(data, blobA) {
			t.Fatal("original block damaged by a conflicting re-put")
		}
	})

	t.Run("OwnershipContract", func(t *testing.T) {
		// The CONTRACT is ownership TRANSFER: Put takes the slice and the
		// caller must never touch it again (the server hands over its staging
		// buffer). A store may alias it (ramstub) or copy it (dram); either
		// way the committed content — verified against an independent copy
		// the store never saw — is what Get returns. Deliberately NOT tested:
		// mutating the buffer after Put, which the interface forbids.
		s := newStore(t)
		orig := bytes.Repeat([]byte{0x5A}, 8192)
		staged := append([]byte(nil), orig...)
		put(t, s, 1, key(2), staged, protocol.StatusOK)
		data, _, ok := s.Get(1, key(2))
		if !ok || !bytes.Equal(data, orig) {
			t.Fatal("committed content mismatch")
		}
	})

	t.Run("EmptyBlock", func(t *testing.T) {
		// §3.4: total_len=0 is legal; the GET descriptor is status=OK, len=0.
		s := newStore(t)
		put(t, s, 1, key(9), nil, protocol.StatusOK)
		if !s.Contains(1, key(9)) {
			t.Fatal("empty block not visible")
		}
		data, sum, ok := s.Get(1, key(9))
		if !ok || len(data) != 0 || sum != xxh3.Hash(nil) {
			t.Fatalf("empty-block get: ok=%v len=%d", ok, len(data))
		}
		n, _ := s.ExistsPrefix(1, [][32]byte{key(9)}, false)
		if n != 1 {
			t.Fatal("empty block breaks the prefix probe")
		}
		// force: the Get above auto-leases on lease-aware stores (§3.3).
		if st := s.Delete(1, key(9), true); st != protocol.StatusOK {
			t.Fatalf("empty-block delete: %s", st)
		}
	})

	t.Run("NamespaceIsolation", func(t *testing.T) {
		s := newStore(t)
		put(t, s, 1, key(3), []byte("tenant-1-block"), protocol.StatusOK)
		if s.Contains(2, key(3)) {
			t.Fatal("block visible across namespaces")
		}
		put(t, s, 2, key(3), []byte("tenant-2-block!"), protocol.StatusOK)
		d1, _, _ := s.Get(1, key(3))
		d2, _, _ := s.Get(2, key(3))
		if bytes.Equal(d1, d2) {
			t.Fatal("namespaces share a block")
		}
	})

	t.Run("ExistsPrefix", func(t *testing.T) {
		s := newStore(t)
		put(t, s, 1, key(1), []byte("a"), protocol.StatusOK)
		put(t, s, 1, key(2), []byte("b"), protocol.StatusOK)
		put(t, s, 1, key(4), []byte("d"), protocol.StatusOK)
		keys := [][32]byte{key(1), key(2), key(3), key(4)}
		n, per := s.ExistsPrefix(1, keys, true)
		if n != 2 {
			t.Fatalf("nConsecutive = %d, want 2", n)
		}
		want := []protocol.Status{protocol.StatusOK, protocol.StatusOK, protocol.StatusNotFound, protocol.StatusOK}
		for i := range want {
			if per[i] != want[i] {
				t.Fatalf("perKey[%d] = %s, want %s", i, per[i], want[i])
			}
		}
		if _, per := s.ExistsPrefix(1, keys, false); per != nil {
			t.Fatal("bitmap returned without negotiation")
		}
	})

	t.Run("Delete", func(t *testing.T) {
		s := newStore(t)
		put(t, s, 1, key(5), []byte("x"), protocol.StatusOK)
		if st := s.Delete(1, key(5), false); st != protocol.StatusOK {
			t.Fatalf("delete: %s", st)
		}
		if s.Contains(1, key(5)) {
			t.Fatal("block survives delete")
		}
		if st := s.Delete(1, key(5), false); st != protocol.StatusNotFound {
			t.Fatalf("re-delete: %s", st)
		}
		// Write-once slot released: a DIFFERENT-bytes put now succeeds.
		put(t, s, 1, key(5), []byte("fresh bytes"), protocol.StatusOK)
	})

	t.Run("StatsShape", func(t *testing.T) {
		s := newStore(t)
		put(t, s, 1, key(6), bytes.Repeat([]byte{1}, 100), protocol.StatusOK)
		put(t, s, 1, key(7), bytes.Repeat([]byte{2}, 50), protocol.StatusOK)
		doc := string(s.Stats())
		if !strings.Contains(doc, `"blocks":2`) || !strings.Contains(doc, `"bytes":150`) {
			t.Fatalf("stats: %s", doc)
		}
	})
}
