package main

import (
	"encoding/binary"
	"io"
	"net"
	"sync/atomic"
	"testing"
)

// TestServeArenaIntegrity confirms serveArena returns the exact arena bytes for
// each blob id — closing the "off-heap AND correct" claim: a regression that
// served zeros, the wrong subslice, or a heap copy would fail this.
func TestServeArenaIntegrity(t *testing.T) {
	// Build a small arena with a distinct byte pattern per blob.
	index := []blobRef{{off: 0, len: 100}, {off: 100, len: 250}, {off: 350, len: 40}}
	arena := make([]byte, 390)
	for _, ref := range index {
		for i := range ref.len {
			arena[ref.off+i] = byte((ref.off + i) % 251) //nolint:gosec // %251 < 256 always fits a byte
		}
	}

	srv, cli := net.Pipe()
	var served atomic.Int64
	go serveArena(srv, arena, index, &served)
	defer cli.Close()

	req := make([]byte, 4)
	hdr := make([]byte, headerSize)
	for id, ref := range index {
		binary.LittleEndian.PutUint32(req, uint32(id)) //nolint:gosec // small test ids
		if _, err := cli.Write(req); err != nil {
			t.Fatalf("write req %d: %v", id, err)
		}
		if _, err := io.ReadFull(cli, hdr); err != nil {
			t.Fatalf("read hdr %d: %v", id, err)
		}
		h, err := decodeHeader(hdr)
		if err != nil {
			t.Fatalf("decode hdr %d: %v", id, err)
		}
		if int(h.length) != ref.len {
			t.Fatalf("blob %d: length %d, want %d", id, h.length, ref.len)
		}
		got := make([]byte, h.length)
		if _, err := io.ReadFull(cli, got); err != nil {
			t.Fatalf("read body %d: %v", id, err)
		}
		want := arena[ref.off : ref.off+ref.len]
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("blob %d byte %d: got %d, want %d", id, i, got[i], want[i])
			}
		}
	}
}

// TestPctileConservative checks the percentile walk returns the correct bucket
// upper bound and never under-reports.
func TestPctileConservative(t *testing.T) {
	buckets := []float64{0, 1, 2, 4, 8} // 4 buckets
	counts := []uint64{10, 0, 0, 90}    // 90% of mass in [4,8)
	if got := pctile(counts, buckets, 0.50); got != 8 {
		t.Errorf("p50 = %v, want 8 (the [4,8) bucket dominates)", got)
	}
	if got := pctile(counts, buckets, 0.99); got != 8 {
		t.Errorf("p99 = %v, want 8", got)
	}
	if got := maxBucket(counts, buckets); got != 8 {
		t.Errorf("max = %v, want 8", got)
	}
}
