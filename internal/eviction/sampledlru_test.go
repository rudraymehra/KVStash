package eviction

import "testing"

// TestSampledLRUOrder: with a population under the sample size K the scan
// is exhaustive, so eviction order is EXACT LRU — deterministic to assert.
func TestSampledLRUOrder(t *testing.T) {
	p := NewSampledLRU()
	a, b, c := ek(7, 1), ek(7, 2), ek(7, 3)
	p.Admit(a, 100, 10)
	p.Admit(b, 100, 20)
	p.Admit(c, 100, 30)
	p.Touch(a, 40) // a becomes the freshest; b is now the stalest

	v := p.Victims(7, 100, 0, nil)
	if len(v) != 1 || v[0].Key != b {
		t.Fatalf("stalest first: %+v, want [b]", v)
	}
	v = p.Victims(7, 100, 0, v[:0])
	if len(v) != 1 || v[0].Key != c {
		t.Fatalf("second stalest: %+v, want [c]", v)
	}
	if u := p.Usage(nil); len(u) != 1 || u[0].Bytes != 100 {
		t.Fatalf("usage: %+v, want ns7=100 (only a left)", u)
	}
}

// TestSampledLRURemoveAndIsolation: Remove drops bookkeeping; victims stay
// tenant-scoped.
func TestSampledLRURemoveAndIsolation(t *testing.T) {
	p := NewSampledLRU()
	p.Admit(ek(1, 1), 50, 1)
	p.Admit(ek(1, 2), 60, 2)
	p.Admit(ek(2, 1), 70, 3)
	p.Remove(ek(1, 1))
	if u := p.Usage(nil); len(u) != 2 {
		t.Fatalf("usage domains: %+v", u)
	}
	v := p.Victims(1, 1000, 0, nil)
	if len(v) != 1 || v[0].Key != ek(1, 2) {
		t.Fatalf("victims ns1: %+v", v)
	}
	if v := p.Victims(2, 1000, 0, nil); len(v) != 1 || v[0].Key.NS != 2 {
		t.Fatalf("victims ns2: %+v", v)
	}
	p.Remove(ek(9, 9)) // unknown domain: no-op
}

// TestSampledLRUDoubleAdmitIsNoop mirrors the S3-FIFO defensive contract.
func TestSampledLRUDoubleAdmitIsNoop(t *testing.T) {
	p := NewSampledLRU()
	p.Admit(ek(3, 1), 100, 1)
	p.Admit(ek(3, 1), 100, 2)
	if u := p.Usage(nil); len(u) != 1 || u[0].Bytes != 100 {
		t.Fatalf("double admit: %+v", u)
	}
}
