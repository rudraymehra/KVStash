package target

import (
	"bytes"
	"context"
	"path/filepath"
	"testing"

	"github.com/kvstash/kvblockd/bench/kvbench/internal/gen"
)

func k(i int) [32]byte { return gen.Keyspace{Seed: 9}.Key(i) }

func blobs(n, sz int) ([][32]byte, [][]byte) {
	keys := make([][32]byte, n)
	bs := make([][]byte, n)
	for i := range keys {
		keys[i] = k(i)
		bs[i] = make([]byte, sz)
		gen.FillPayload(bs[i], 9, keys[i])
	}
	return keys, bs
}

func TestTilingMath(t *testing.T) {
	ctx := context.Background()
	m := NewMem(0, 4) // cap 4 forces tiling
	keys, bs := blobs(10, 128)

	// Direct over-cap call must fail (the pkg/client contract shape).
	if _, err := m.BatchPut(ctx, keys, bs); err == nil {
		t.Fatal("over-cap batch accepted by the double")
	}
	if _, err := TiledPut(ctx, m, LimitOf(m), keys, bs); err != nil {
		t.Fatal(err)
	}

	// Exists early-exit: delete key 5 → consecutive prefix is 5, and the
	// probe must stop at the tile containing the break (tiles of 4: probes
	// tiles [0..3] full-hit, [4..7] hits 1 → stop; [8..9] never probed).
	m.mu.Lock()
	delete(m.m, keys[5])
	m.mu.Unlock()
	n, err := TiledExists(ctx, m, 4, keys)
	if err != nil {
		t.Fatal(err)
	}
	if n != 5 {
		t.Fatalf("consecutive=%d, want 5", n)
	}

	dst := make([][]byte, len(keys))
	sts, err := TiledGet(ctx, m, 4, keys, dst)
	if err != nil {
		t.Fatal(err)
	}
	if len(sts) != 10 {
		t.Fatalf("%d statuses", len(sts))
	}
	for i, st := range sts {
		want := OK
		if i == 5 {
			want = Miss
		}
		if st != want {
			t.Fatalf("key %d: status %d want %d", i, st, want)
		}
		if st == OK && !bytes.Equal(dst[i], bs[i]) {
			t.Fatalf("key %d: bytes differ", i)
		}
	}
}

func TestMemCapacityFIFO(t *testing.T) {
	ctx := context.Background()
	m := NewMem(512, 0) // 4 × 128-byte blobs fit
	keys, bs := blobs(6, 128)
	if _, err := m.BatchPut(ctx, keys, bs); err != nil {
		t.Fatal(err)
	}
	// FIFO: keys 0,1 evicted; 2..5 resident.
	dst := make([][]byte, 6)
	sts, err := m.BatchGet(ctx, keys, dst)
	if err != nil {
		t.Fatal(err)
	}
	for i, st := range sts {
		want := OK
		if i < 2 {
			want = Miss
		}
		if st != want {
			t.Fatalf("key %d: %d want %d", i, st, want)
		}
	}
}

func TestMemFlipIsCaughtByVerify(t *testing.T) {
	ctx := context.Background()
	m := NewMem(0, 0)
	keys, bs := blobs(1, 4096)
	if _, err := m.BatchPut(ctx, keys, bs); err != nil {
		t.Fatal(err)
	}
	m.FlipByteOnce(keys[0], 100)
	dst := make([][]byte, 1)
	if _, err := m.BatchGet(ctx, keys, dst); err != nil {
		t.Fatal(err)
	}
	if gen.VerifyPayload(dst[0], 9, keys[0]) {
		t.Fatal("flipped blob passed verification — the oracle is vacuous")
	}
	// Second read is pristine again (flip-ONCE).
	if _, err := m.BatchGet(ctx, keys, dst); err != nil {
		t.Fatal(err)
	}
	if !gen.VerifyPayload(dst[0], 9, keys[0]) {
		t.Fatal("pristine re-read rejected")
	}
}

func TestFSRoundTrip(t *testing.T) {
	ctx := context.Background()
	fs, err := OpenFS(FSOptions{Dir: filepath.Join(t.TempDir(), "blocks"), BlobBytes: 4096, Workers: 4})
	if err != nil {
		t.Fatal(err)
	}
	keys, bs := blobs(8, 4096)
	if _, err := fs.BatchPut(ctx, keys, bs); err != nil {
		t.Fatal(err)
	}
	// Write-once: a second put is an Exists no-op (moved no bytes) — the
	// distinction goodput accounting relies on.
	if sts, err := fs.BatchPut(ctx, keys[:1], bs[:1]); err != nil || sts[0] != Exists {
		t.Fatalf("re-put: %v %v (want Exists)", sts, err)
	}
	if Exists.Wrote() || !OK.Wrote() {
		t.Fatal("Wrote() semantics inverted")
	}
	n, err := fs.BatchExists(ctx, keys)
	if err != nil || n != 8 {
		t.Fatalf("exists=%d err=%v", n, err)
	}
	dst := make([][]byte, 8)
	sts, err := fs.BatchGet(ctx, keys, dst)
	if err != nil {
		t.Fatal(err)
	}
	for i, st := range sts {
		if st != OK || !bytes.Equal(dst[i], bs[i]) {
			t.Fatalf("key %d: st=%d equal=%v", i, st, bytes.Equal(dst[i], bs[i]))
		}
	}
	// Missing key is a Miss, not an error.
	miss := [][32]byte{k(99)}
	mdst := make([][]byte, 1)
	sts, err = fs.BatchGet(ctx, miss, mdst)
	if err != nil || sts[0] != Miss {
		t.Fatalf("missing: %v %v", sts, err)
	}
	// Blob-size validation.
	if _, err := OpenFS(FSOptions{Dir: t.TempDir(), BlobBytes: 1000}); err == nil {
		t.Fatal("unaligned blob accepted")
	}
}
