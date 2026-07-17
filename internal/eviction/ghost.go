package eviction

// ghostRing remembers fingerprints of blocks recently evicted from the
// SMALL queue: a bounded FIFO ring of uint64s plus a membership set —
// pointer-free, so ghost memory is strictly bounded and GC-invisible in
// practice. A ghost hit at Admit means "this key was evicted before it
// proved itself" and routes the re-insert straight to MAIN (the S3-FIFO
// scan-resistance mechanism).
//
// Fingerprints are the key's first 8 hash bytes XOR a per-policy random
// seed: keys are already strong content hashes, so 64 bits suffice
// (collision odds at 128K resident ghosts ≈ 4e-10 per lookup), and the
// seed keeps the structure grind-resistant on principle.
type ghostRing struct {
	ring []uint64
	set  map[uint64]struct{}
	head int // next overwrite position when full
	n    int // live entries
}

// ghostFloor is the minimum ring capacity once any entry lands: cold
// domains still get scan-resistance for ~8 KiB.
const ghostFloor = 1024

func (g *ghostRing) contains(fp uint64) bool {
	_, ok := g.set[fp]
	return ok
}

// push records fp, evicting the oldest fingerprint when full. The ring
// lazily initializes at the floor; growth is maybeGrowGhost's job.
func (g *ghostRing) push(fp uint64) {
	if g.ring == nil {
		g.ring = make([]uint64, ghostFloor)
		g.set = make(map[uint64]struct{}, ghostFloor)
	}
	if g.contains(fp) {
		return // already remembered; refreshing order isn't worth a scan
	}
	if g.n == len(g.ring) {
		delete(g.set, g.ring[g.head])
		g.ring[g.head] = fp
		g.head = (g.head + 1) % len(g.ring)
	} else {
		g.ring[(g.head+g.n)%len(g.ring)] = fp
		g.n++
	}
	g.set[fp] = struct{}{}
}

// grow rebuilds the ring at newCap (power-of-two rounded by the caller),
// keeping the newest entries. No-op if newCap ≤ current capacity. grow can
// run before any push (a promotion trips ghost sizing first), so it must
// materialize the membership set itself — rapid found the nil-map panic.
func (g *ghostRing) grow(newCap int) {
	if newCap <= len(g.ring) {
		return
	}
	if g.set == nil {
		g.set = make(map[uint64]struct{}, newCap)
	}
	fresh := make([]uint64, newCap)
	kept := 0
	for i := 0; i < g.n; i++ { // oldest → newest
		fresh[kept] = g.ring[(g.head+i)%len(g.ring)]
		kept++
	}
	g.ring = fresh
	g.head = 0
	g.n = kept
	// set is unchanged (same members).
}

// size reports live entries (the harness's ghost-bound invariant reads it).
func (g *ghostRing) size() int { return g.n }

func (g *ghostRing) capacity() int { return len(g.ring) }
