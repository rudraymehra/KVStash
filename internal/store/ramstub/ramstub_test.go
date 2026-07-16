package ramstub

import (
	"bytes"
	"strings"
	"testing"

	"github.com/kvstash/kvblockd/internal/protocol"
)

func k(b byte) [32]byte {
	var key [32]byte
	key[0] = b // first byte drives shard selection
	key[1] = b
	return key
}

// TestPutWriteOnce pins the §13 write-once matrix: first Put OK, same-checksum
// re-Put OK_EXISTS, different-checksum re-Put ERR_IMMUTABLE_CONFLICT with the
// original bytes untouched.
func TestPutWriteOnce(t *testing.T) {
	s := New()
	orig := []byte("block-A")
	if st := s.Put(7, k(1), orig, 111); st != protocol.StatusOK {
		t.Fatalf("first put: %s", st)
	}
	if st := s.Put(7, k(1), orig, 111); st != protocol.StatusOKExists {
		t.Fatalf("idempotent re-put: %s", st)
	}
	if st := s.Put(7, k(1), []byte("block-B"), 222); st != protocol.StatusErrImmutableConflict {
		t.Fatalf("conflicting re-put: %s", st)
	}
	data, sum, ok := s.Get(7, k(1))
	if !ok || sum != 111 || !bytes.Equal(data, orig) {
		t.Fatalf("original block damaged by conflict: ok=%v sum=%d data=%q", ok, sum, data)
	}
}

// TestPutTransfersOwnership pins the no-copy contract: Put keeps the exact
// slice it was handed (ownership transfer — the commit path discards its
// staging extent, so copying would double every PUT's memory traffic). The
// flip side, "caller must not reuse the slice", is the documented obligation.
func TestPutTransfersOwnership(t *testing.T) {
	s := New()
	staging := []byte("handed-over")
	s.Put(1, k(2), staging, 42)
	data, _, ok := s.Get(1, k(2))
	if !ok || &data[0] != &staging[0] {
		t.Fatal("store copied the staging buffer (ownership contract regressed to copy)")
	}
}

// TestNamespaceIsolation: the same key in two namespaces is two blocks.
func TestNamespaceIsolation(t *testing.T) {
	s := New()
	s.Put(1, k(3), []byte("tenant-1"), 10)
	if s.Contains(2, k(3)) {
		t.Fatal("block visible across namespaces")
	}
	if st := s.Put(2, k(3), []byte("tenant-2"), 20); st != protocol.StatusOK {
		t.Fatalf("same key, other namespace: %s", st)
	}
}

// TestExistsPrefix: consecutive count stops at the first miss and never
// resumes; bitmap reports every key independently.
func TestExistsPrefix(t *testing.T) {
	s := New()
	s.Put(1, k(1), []byte("a"), 1)
	s.Put(1, k(2), []byte("b"), 2)
	s.Put(1, k(4), []byte("d"), 4)
	keys := [][32]byte{k(1), k(2), k(3), k(4)}

	n, per := s.ExistsPrefix(1, keys, true)
	if n != 2 {
		t.Fatalf("n_consecutive = %d, want 2 (miss at index 2 ends the run)", n)
	}
	want := []protocol.Status{protocol.StatusOK, protocol.StatusOK, protocol.StatusNotFound, protocol.StatusOK}
	for i := range want {
		if per[i] != want[i] {
			t.Fatalf("perKey[%d] = %s, want %s", i, per[i], want[i])
		}
	}
	if _, per := s.ExistsPrefix(1, keys, false); per != nil {
		t.Fatal("bitmap returned without withBitmap")
	}
}

// TestDelete: present → OK then gone; absent → NOT_FOUND.
func TestDelete(t *testing.T) {
	s := New()
	s.Put(1, k(5), []byte("x"), 5)
	if st := s.Delete(1, k(5), false); st != protocol.StatusOK {
		t.Fatalf("delete: %s", st)
	}
	if s.Contains(1, k(5)) {
		t.Fatal("block survives delete")
	}
	if st := s.Delete(1, k(5), false); st != protocol.StatusNotFound {
		t.Fatalf("re-delete: %s", st)
	}
}

// TestStats: block/byte counts reflect the contents.
func TestStats(t *testing.T) {
	s := New()
	s.Put(1, k(6), make([]byte, 100), 6)
	s.Put(1, k(7), make([]byte, 50), 7)
	doc := string(s.Stats())
	if !strings.Contains(doc, `"blocks":2`) || !strings.Contains(doc, `"bytes":150`) {
		t.Fatalf("stats: %s", doc)
	}
}
