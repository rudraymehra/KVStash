package nvme

import "testing"

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
