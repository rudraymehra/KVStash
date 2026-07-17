package eviction

import (
	"sync"
	"testing"
	"time"
)

func ek(ns uint32, b byte) Key {
	var k Key
	k.NS = ns
	k.Hash[0], k.Hash[9] = b, b // byte 9 lands in the shardIdx window
	return k
}

// keysOf projects candidate keys for order-insensitive assertions.
func keysOf(cands []Candidate) map[Key]bool {
	m := make(map[Key]bool, len(cands))
	for _, c := range cands {
		m[c.Key] = true
	}
	return m
}

// TestS3FIFOScriptedTrace walks the published algorithm with hand-computed
// expected evictions: one-hit wonders die in small without touching main;
// a twice-touched probationer promotes; a ghost-remembered re-insert goes
// straight to main; main entries get freq second chances.
func TestS3FIFOScriptedTrace(t *testing.T) {
	p := NewS3FIFO(0)
	a, b, c := ek(7, 1), ek(7, 2), ek(7, 3)

	p.Admit(a, 100, 0)
	p.Admit(b, 100, 0)
	p.Admit(c, 100, 0) // all probationary: small=[a b c] 300B, main empty

	p.Touch(b, 0)
	p.Touch(b, 0) // freq(b)=2 → earns promotion

	// Round 1: small over its 10% share → scan small. Head a has freq 0:
	// evicted (and ghosted). Exactly [a].
	v := p.Victims(7, 100, 0, nil)
	if len(v) != 1 || v[0].Key != a || v[0].Size != 100 {
		t.Fatalf("round 1: %+v, want exactly [a]", v)
	}

	// Round 2: head b (freq 2) promotes to main; head c (freq 0) evicts.
	v = p.Victims(7, 100, 0, v[:0])
	if len(v) != 1 || v[0].Key != c {
		t.Fatalf("round 2: %+v, want exactly [c] (b must promote, not evict)", v)
	}
	if u := p.Usage(nil); len(u) != 1 || u[0].Bytes != 100 {
		t.Fatalf("usage after rounds: %+v, want ns7=100 (b resident in main)", u)
	}

	// Ghost memory: a was evicted from small, so its re-insert routes to
	// MAIN (scan resistance), joining b.
	p.Admit(a, 100, 0)
	d := p.domain(7)
	d.qmu.Lock()
	mainCount, smallCount := d.main.count, d.small.count
	d.qmu.Unlock()
	if mainCount != 2 || smallCount != 0 {
		t.Fatalf("ghost re-insert: main=%d small=%d, want main=2 small=0", mainCount, smallCount)
	}

	// Main scan: b entered main with freq 2 (kept on promote) → two
	// second chances; a entered with freq 0 → first eviction candidate.
	v = p.Victims(7, 100, 0, v[:0])
	if len(v) != 1 || v[0].Key != a {
		t.Fatalf("main scan: %+v, want [a] (b holds freq credit)", v)
	}

	// Drain: only b remains; needs 2 more scan visits to burn freq 2→0.
	v = p.Victims(7, 1000, 0, v[:0])
	if len(v) != 1 || v[0].Key != b {
		t.Fatalf("drain: %+v, want [b]", v)
	}
	if u := p.Usage(nil); len(u) != 0 {
		t.Fatalf("usage after drain: %+v, want empty", u)
	}
}

// TestS3FIFORemoveUnlinks: a store-side DELETE removes the entry from
// whichever queue holds it; it never resurfaces as a victim.
func TestS3FIFORemoveUnlinks(t *testing.T) {
	p := NewS3FIFO(0)
	a, b := ek(1, 1), ek(1, 2)
	p.Admit(a, 50, 0)
	p.Admit(b, 50, 0)
	p.Remove(a)
	if u := p.Usage(nil); len(u) != 1 || u[0].Bytes != 50 {
		t.Fatalf("usage after remove: %+v", u)
	}
	v := p.Victims(1, 1000, 0, nil)
	if got := keysOf(v); got[a] || !got[b] {
		t.Fatalf("victims after remove: %+v, want only b", v)
	}
	p.Remove(ek(1, 99)) // unknown key: no-op
	p.Remove(ek(9, 1))  // unknown domain: no-op
}

// TestS3FIFOReAdmission pins the re-admission contract: a gate-failed
// small-evictee (just ghosted) re-enters MAIN — a proved-protected block
// upgrades instead of thrashing through small.
func TestS3FIFOReAdmission(t *testing.T) {
	p := NewS3FIFO(0)
	a := ek(3, 1)
	p.Admit(a, 100, 0)
	v := p.Victims(3, 100, 0, nil)
	if len(v) != 1 || v[0].Key != a {
		t.Fatalf("setup eviction: %+v", v)
	}
	// The store's gate refused (leased, say) → hand back.
	p.Admit(a, 100, 0)
	d := p.domain(3)
	d.qmu.Lock()
	inMain := d.main.count == 1 && d.small.count == 0
	d.qmu.Unlock()
	if !inMain {
		t.Fatal("re-admitted gate-failed candidate did not land in main")
	}
}

// TestS3FIFODoubleAdmitIsNoop: defensive — double-Admit must not corrupt
// the queues or double-count bytes.
func TestS3FIFODoubleAdmitIsNoop(t *testing.T) {
	p := NewS3FIFO(0)
	a := ek(2, 1)
	p.Admit(a, 100, 0)
	p.Admit(a, 100, 0)
	if u := p.Usage(nil); len(u) != 1 || u[0].Bytes != 100 {
		t.Fatalf("double admit: %+v, want ns2=100 once", u)
	}
	if v := p.Victims(2, 1000, 0, nil); len(v) != 1 {
		t.Fatalf("double admit queued twice: %+v", v)
	}
}

// TestS3FIFOPerTenantDomains: victims for one tenant never name another's.
func TestS3FIFOPerTenantDomains(t *testing.T) {
	p := NewS3FIFO(0)
	p.Admit(ek(1, 1), 100, 0)
	p.Admit(ek(2, 1), 100, 0)
	if u := p.Usage(nil); len(u) != 2 {
		t.Fatalf("usage domains: %+v", u)
	}
	v := p.Victims(1, 1000, 0, nil)
	if len(v) != 1 || v[0].Key.NS != 1 {
		t.Fatalf("victims ns1: %+v (must never name another tenant's block)", v)
	}
	if u := p.Usage(nil); len(u) != 1 || u[0].NS != 2 {
		t.Fatalf("ns2 untouched by ns1 pressure: %+v", u)
	}
}

// TestGhostRingBounds: the ring never exceeds capacity, oldest entries
// fall out FIFO, and grow keeps the newest.
func TestGhostRingBounds(t *testing.T) {
	var g ghostRing
	for i := 0; i < ghostFloor*3; i++ {
		g.push(uint64(i))
	}
	if g.size() != ghostFloor || g.capacity() != ghostFloor {
		t.Fatalf("size=%d cap=%d, want both %d", g.size(), g.capacity(), ghostFloor)
	}
	if g.contains(0) {
		t.Fatal("oldest fingerprint survived FIFO overwrite")
	}
	if !g.contains(uint64(ghostFloor*3 - 1)) {
		t.Fatal("newest fingerprint missing")
	}
	g.grow(ghostFloor * 2)
	if g.capacity() != ghostFloor*2 || g.size() != ghostFloor {
		t.Fatalf("after grow: cap=%d size=%d", g.capacity(), g.size())
	}
	if !g.contains(uint64(ghostFloor*3 - 1)) {
		t.Fatal("grow lost the newest entry")
	}
}

// TestGhostAutoGrowCapped: mainHi drives ring growth, clamped at ghostMax.
func TestGhostAutoGrowCapped(t *testing.T) {
	p := NewS3FIFO(2048)
	d := p.ensureDomain(5)
	d.qmu.Lock()
	d.ghost.push(1) // materialize the floor ring
	d.mainHi = 100_000
	p.maybeGrowGhost(d)
	capacity := d.ghost.capacity()
	d.qmu.Unlock()
	if capacity != 2048 {
		t.Fatalf("ghost capacity %d, want the 2048 ghostMax clamp", capacity)
	}
}

// TestS3FIFOAdmitRemoveRace is the regression for the reproduced evictor
// panic: Admit publishes the table entry BEFORE queueing it, and a racing
// Remove used to unlink the never-queued entry (zero-value where == the old
// qSmall), corrupting the FIFO. The tombstone handshake (qUnqueued/qDead)
// closes it; this hammer drove the old code to a nil-deref in <1ms.
func TestS3FIFOAdmitRemoveRace(t *testing.T) {
	p := NewS3FIFO(4096)
	var wg sync.WaitGroup
	stop := make(chan struct{})
	for w := 0; w < 4; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := 0; ; i++ {
				select {
				case <-stop:
					return
				default:
				}
				k := ek(1, byte(i^w)) //nolint:gosec // G115: byte mixing
				p.Admit(k, 100, 0)
				p.Remove(k)
			}
		}(w)
	}
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			select {
			case <-stop:
				return
			default:
			}
			// The old corruption made count>0 with head==nil → popHead panic.
			_ = p.Victims(1, 1<<20, 0, nil)
		}
	}()
	time.Sleep(500 * time.Millisecond)
	close(stop)
	wg.Wait()
	// Queue/byte bookkeeping must still be coherent: drain fully.
	d := p.domain(1)
	if d != nil {
		_ = p.Victims(1, 1<<40, 0, nil)
		d.qmu.Lock()
		defer d.qmu.Unlock()
		if d.small.count != 0 || d.main.count != 0 || d.smallBytes != 0 || d.mainBytes != 0 {
			t.Fatalf("post-race bookkeeping: small=%d/%dB main=%d/%dB",
				d.small.count, d.smallBytes, d.main.count, d.mainBytes)
		}
	}
}
