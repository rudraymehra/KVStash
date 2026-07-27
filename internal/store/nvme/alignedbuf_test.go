package nvme

import (
	"context"
	"testing"

	"github.com/zeebo/xxh3"
)

func TestBufPoolClassesAndReuse(t *testing.T) {
	p := newBufPool(uint32(recordSpan(2560<<10)), 2) //nolint:gosec // G115: 2.5 MiB span
	defer p.Close()

	// Class selection: request sizes land in the smallest class that fits.
	small, err := p.Get(4096)
	if err != nil {
		t.Fatal(err)
	}
	if len(small) != 128<<10 {
		t.Fatalf("small class = %d, want 128KiB", len(small))
	}
	mid, err := p.Get(200 << 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(mid) != 1<<20 {
		t.Fatalf("mid class = %d, want 1MiB", len(mid))
	}
	big, err := p.Get(2 << 20)
	if err != nil {
		t.Fatal(err)
	}
	if uint64(len(big)) != roundUpAlign(recordSpan(2560<<10)) {
		t.Fatalf("big class = %d", len(big))
	}

	// O_DIRECT usability: page-aligned length, writable.
	for _, b := range [][]byte{small, mid, big} {
		if len(b)%recordAlign != 0 {
			t.Fatalf("buffer len %d not 4KiB-multiple", len(b))
		}
		b[0], b[len(b)-1] = 0xAB, 0xCD
	}

	// Reuse: a returned buffer comes back on the next Get of its class.
	p.Put(small)
	small2, err := p.Get(4096)
	if err != nil {
		t.Fatal(err)
	}
	if small2[0] != 0xAB {
		t.Fatal("free list did not reuse the returned buffer")
	}
	p.Put(mid)
	p.Put(big)
	p.Put(small2)

	// Oversize request is a loud error, not a silent alloc.
	if _, err := p.Get(uint32(recordSpan(2560<<10)) + recordAlign); err == nil { //nolint:gosec // G115: test size
		t.Fatal("oversize Get succeeded")
	}
}

func TestBufPoolOverflowAndForeign(t *testing.T) {
	p := newBufPool(1<<20, 1)
	defer p.Close()

	a, _ := p.Get(1 << 20)
	b, _ := p.Get(1 << 20)
	p.Put(a) // retained (cap 1)
	p.Put(b) // overflow → munmapped, must not panic or block

	// Foreign-sized slice: munmapped path, no panic.
	f, err := mmapBuf(8192)
	if err != nil {
		t.Fatal(err)
	}
	p.Put(f)

	// Small maxSpan collapses classes without duplicates or zero classes.
	tiny := newBufPool(4096, 1)
	defer tiny.Close()
	tb, err := tiny.Get(4096)
	if err != nil {
		t.Fatal(err)
	}
	tiny.Put(tb)
}

// TestBufPoolCounters pins the observability of the retain cliff: every
// free-list miss counts on pool_alloc_total, every overflow munmap on
// pool_overflow_munmap_total — the pair that makes "outstanding responses
// exceeded the retain bound" visible on a scrape instead of a mystery
// syscall storm.
func TestBufPoolCounters(t *testing.T) {
	p := newBufPool(1<<20, 2)
	defer p.Close()

	a, _ := p.Get(1 << 20)
	b, _ := p.Get(1 << 20)
	c, _ := p.Get(1 << 20)
	m := map[string]int64{}
	p.statsInto(m)
	if m["pool_alloc_total"] != 3 || m["pool_overflow_munmap_total"] != 0 {
		t.Fatalf("after 3 cold Gets: %v", m)
	}
	p.Put(a)
	p.Put(b)
	p.Put(c) // retain 2 → this one munmaps
	m = map[string]int64{}
	p.statsInto(m)
	if m["pool_overflow_munmap_total"] != 1 {
		t.Fatalf("overflow munmap uncounted: %v", m)
	}
	// A warm Get reuses the free list — allocs must not move.
	d, _ := p.Get(1 << 20)
	p.Put(d)
	m = map[string]int64{}
	p.statsInto(m)
	if m["pool_alloc_total"] != 3 {
		t.Fatalf("warm Get counted as an alloc: %v", m)
	}
}

// TestReadBufRetainFollowsOutstanding pins the sizing change: the free-list
// bound follows expected outstanding transport-held buffers (readq depth +
// in-flight responses = 8×ReadWorkers by default; operator-overridable),
// not 2×ReadWorkers — the old bound made every read past 32 outstanding an
// mmap/munmap pair.
func TestReadBufRetainFollowsOutstanding(t *testing.T) {
	dir := t.TempDir()
	p := testParams(t, dir)
	p.ReadWorkers = 2
	v, _, _, err := OpenVolume(p)
	if err != nil {
		t.Fatal(err)
	}
	for _, ch := range v.free() {
		if cap(ch) != 8*p.ReadWorkers {
			t.Fatalf("default retain = %d, want %d (8×ReadWorkers)", cap(ch), 8*p.ReadWorkers)
		}
	}
	_ = v.Close()

	p.Dir = t.TempDir()
	p.ReadBufRetain = 5
	v2, _, _, err := OpenVolume(p)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = v2.Close() }()
	for _, ch := range v2.free() {
		if cap(ch) != 5 {
			t.Fatalf("explicit retain = %d, want 5", cap(ch))
		}
	}
}

// free exposes the pool's free lists to the sizing test.
func (v *Volume) free() []chan []byte { return v.pool.free }

// TestPoolFailureReadsBusyNotCorrupt pins the error taxonomy: a read whose
// buffer the pool cannot provide (here: a span beyond the largest class —
// the same return path as mmap ENOMEM) must answer ReadBusy, NOT
// ReadCorrupt. ReadCorrupt's contract self-heals (deletes) the index entry,
// so the old classification turned transient memory pressure into
// permanent eviction of healthy on-disk blocks.
func TestPoolFailureReadsBusyNotCorrupt(t *testing.T) {
	v, _, _ := openTestVolume(t, t.TempDir())
	defer func() { _ = v.Close() }()
	blob := testPayload(1, 4096)
	done := make(chan Loc, 1)
	if !v.Append(AppendReq{
		NS: 1, Key: testKey(1), XXH3: xxh3.Hash(blob), Data: blob,
		OnWritten: func(loc Loc, ok bool) {
			if !ok {
				t.Error("append failed")
			}
			done <- loc
		},
	}) {
		t.Fatal("append refused")
	}
	loc := <-done

	// Same segment, impossible span: pool.Get fails before any device I/O
	// (a full alignment step past MaxBlobLen, so the span rounds PAST the
	// largest class instead of into it).
	huge := loc
	huge.Len = v.p.MaxBlobLen + 2*recordAlign
	if _, _, st := v.Read(context.Background(), huge, 1, testKey(1), 42); st != ReadBusy {
		t.Fatalf("pool-failure read = %d, want ReadBusy (%d) — ReadCorrupt would self-heal-delete a healthy block", st, ReadBusy)
	}
	// The honest read still serves.
	mustRead(t, v, loc, 1, xxh3.Hash(blob), 4096)
}
