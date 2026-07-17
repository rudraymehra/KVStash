package modeltest

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"testing"
	"time"

	"github.com/zeebo/xxh3"
	"pgregory.net/rapid"

	"github.com/kvstash/kvblockd/internal/eviction"
	"github.com/kvstash/kvblockd/internal/protocol"
	"github.com/kvstash/kvblockd/internal/store/dram"
)

const (
	arenaBytes     = 8 << 20
	leaseDefaultMS = 500
	leaseMaxMS     = 60_000
	pinnedCap      = 1 << 20
	startNanos     = int64(1_700_000_000_000_000_000)
	junkSize       = 256 << 10
)

// poolHash returns the i'th pooled key hash. The pool is SHARED across
// namespaces (same hash, different ns, different payloads) — I4's teeth.
func poolHash(i int) [32]byte {
	var h [32]byte
	h[0] = byte(i)     //nolint:gosec // G115: pool index 0..47
	h[9] = byte(i * 7) //nolint:gosec // G115: byte mixing — lands in the policy shard-index window
	h[31] = 0x9D
	return h
}

func junkHash(seq int) [32]byte {
	var h [32]byte
	h[0] = 0xEE
	binary.LittleEndian.PutUint64(h[1:9], uint64(seq)) //nolint:gosec // test counter
	h[9] = byte(seq)                                   //nolint:gosec // G115: byte mixing
	return h
}

type heldRef struct {
	key     eviction.Key
	blk     *mBlock // the GENERATION we held — a force-delete + re-put maps the key to a new block
	view    []byte
	snap    []byte
	release func()
}

type storeStats struct {
	Blocks         int              `json:"blocks"`
	Bytes          int64            `json:"bytes"`
	ArenaBytes     int64            `json:"arena_bytes"`
	ArenaFreeBytes int64            `json:"arena_free_bytes"`
	PinnedBytes    map[string]int64 `json:"pinned_bytes"`
	EvictionsTotal uint64           `json:"evictions_total"`
	LiveAllocs     int64            `json:"live_allocs"`
}

// TestStoreModel is the harness: the real tier vs the reference model over
// a randomized op stream, invariants after every step, both policies drawn.
func TestStoreModel(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		policyName := rapid.SampledFrom([]string{"s3fifo", "sampled-lru"}).Draw(rt, "policy")
		pol, err := eviction.New(policyName, 8192)
		if err != nil {
			rt.Fatal(err)
		}
		cur := startNanos
		arena, err := dram.NewArena(arenaBytes, false)
		if err != nil {
			rt.Fatal(err)
		}
		s := dram.New(arena, dram.Params{
			LeaseDefaultMS: leaseDefaultMS,
			LeaseMaxMS:     leaseMaxMS,
			PinnedBytesCap: pinnedCap,
			Now:            func() int64 { return cur },
		})
		s.AttachPolicy(pol) // evictor NOT started: eviction only via the explicit pressure op
		m := newModel(startNanos, pinnedCap)
		var held []heldRef
		junkSeq := 0
		defer func() {
			for _, h := range held {
				h.release()
			}
			_ = s.Close()
		}()

		nsGen := rapid.SampledFrom([]uint32{1, 2, 3})
		idxGen := rapid.IntRange(0, 47)
		sizeGen := rapid.SampledFrom([]int{0, 1, 100, 4096, 4097, 65536, 262144})
		drawKey := func(rt *rapid.T) (eviction.Key, uint32, [32]byte) {
			ns := nsGen.Draw(rt, "ns")
			h := poolHash(idxGen.Draw(rt, "ki"))
			return eviction.Key{NS: ns, Hash: h}, ns, h
		}
		mustReconcileMiss := func(k eviction.Key, what string) {
			if !m.reconcileMiss(k) {
				rt.Fatalf("%s: %v missing but the model holds it %s", what, k,
					map[bool]string{true: "PROTECTED", false: "un-marked"}[m.protected(m.blocks[k])])
			}
		}
		// grantLease mirrors the store's §3.3 auto-lease on every OK read.
		grantLease := func(b *mBlock) { b.leaseUntil = m.now + int64(leaseDefaultMS)*int64(time.Millisecond) }

		rt.Repeat(map[string]func(*rapid.T){
			"put": func(rt *rapid.T) {
				k, ns, h := drawKey(rt)
				size := sizeGen.Draw(rt, "size")
				fill := rapid.Byte().Draw(rt, "fill")
				data := bytes.Repeat([]byte{fill ^ byte(ns)}, size) //nolint:gosec // G115: byte mixing
				sum := xxh3.Hash(data)
				st := s.Put(ns, h, data, sum)
				e := m.blocks[k]
				switch {
				case e == nil:
					switch st {
					case protocol.StatusOK:
						m.insert(k, data, sum)
					case protocol.StatusErrQuotaBytes: // full arena, no evictor: legal
					default:
						rt.Fatalf("put fresh: %s", st)
					}
				case e.xxh3 == sum: // duplicate content
					switch {
					case st == protocol.StatusOKExists: // present (maybeGone stays — bytes unverified)
					case st == protocol.StatusOK && e.maybeGone: // was evicted: fresh insert
						m.insert(k, data, sum)
					case st == protocol.StatusErrQuotaBytes && e.maybeGone: // evicted AND arena full
						delete(m.blocks, k)
					default:
						rt.Fatalf("put dup: %s (maybeGone=%v)", st, e.maybeGone)
					}
				default: // conflicting content
					switch {
					case st == protocol.StatusErrImmutableConflict:
					case st == protocol.StatusOK && e.maybeGone:
						m.insert(k, data, sum)
					case st == protocol.StatusErrQuotaBytes && e.maybeGone:
						delete(m.blocks, k)
					default:
						rt.Fatalf("put conflict: %s (maybeGone=%v)", st, e.maybeGone)
					}
				}
			},
			"get": func(rt *rapid.T) {
				k, ns, h := drawKey(rt)
				data, sum, ok := s.Get(ns, h)
				if !ok {
					mustReconcileMiss(k, "GET")
					return
				}
				e := m.blocks[k]
				if e == nil {
					rt.Fatalf("GET hit for a key the model never stored: %v", k)
					return
				}
				// I1: byte-identical, right checksum, never cross-key/stale.
				if sum != e.xxh3 || !bytes.Equal(data, e.data) {
					rt.Fatalf("I1: GET bytes/sum mismatch for %v (len got %d want %d)", k, len(data), len(e.data))
				}
				e.maybeGone = false // byte-verified present
				grantLease(e)
			},
			"getref_hold": func(rt *rapid.T) {
				if len(held) >= 8 {
					return
				}
				k, ns, h := drawKey(rt)
				view, sum, release, ok := s.GetRef(ns, h)
				if !ok {
					mustReconcileMiss(k, "GETREF")
					return
				}
				e := m.blocks[k]
				if e == nil {
					rt.Fatalf("GETREF hit for unknown key %v", k)
					return
				}
				if sum != e.xxh3 || !bytes.Equal(view, e.data) {
					rt.Fatalf("I1: GETREF view mismatch for %v", k)
				}
				e.maybeGone = false
				e.heldRefs++
				grantLease(e)
				snap := make([]byte, len(view))
				copy(snap, view)
				held = append(held, heldRef{key: k, blk: e, view: view, snap: snap, release: release})
			},
			"getref_release": func(rt *rapid.T) {
				if len(held) == 0 {
					return
				}
				i := rapid.IntRange(0, len(held)-1).Draw(rt, "hi")
				hr := held[i]
				// I5: the view is byte-stable for the whole hold, whatever
				// deletes/pressure happened meanwhile.
				if !bytes.Equal(hr.view, hr.snap) {
					rt.Fatalf("I5: held view mutated for %v", hr.key)
				}
				hr.release()
				hr.blk.heldRefs-- // the generation we held — a re-put under the same key is a DIFFERENT block
				held = append(held[:i], held[i+1:]...)
			},
			"exists": func(rt *rapid.T) {
				k, ns, h := drawKey(rt)
				n, _ := s.ExistsPrefix(ns, [][32]byte{h}, false)
				if n == 0 {
					mustReconcileMiss(k, "EXISTS")
					return
				}
				e := m.blocks[k]
				if e == nil {
					rt.Fatalf("EXISTS hit for unknown key %v", k)
					return
				}
				// Presence proven at this instant: an un-marked block that
				// later vanishes without pressure is a store bug (the
				// sticky-maybeGone hole the review closed).
				e.maybeGone = false
			},
			"pin": func(rt *rapid.T) {
				k, ns, h := drawKey(rt)
				sub := rapid.SampledFrom([]uint8{protocol.PinSoft, protocol.PinHard, protocol.Unpin}).Draw(rt, "sub")
				st := s.PinOp(ns, h, sub)
				e := m.blocks[k]
				if st == protocol.StatusNotFound {
					mustReconcileMiss(k, "PIN")
					return
				}
				if e == nil {
					rt.Fatalf("PIN %d answered %s for unknown key", sub, st)
					return
				}
				e.maybeGone = false // the pin op reached a resident block
				sz := int64(len(e.data))
				switch sub {
				case protocol.PinSoft:
					if st != protocol.StatusOK {
						rt.Fatalf("soft pin: %s", st)
					}
					if e.hard {
						m.pinned[k.NS] -= sz // downgrade refunds
					}
					e.soft, e.hard = true, false
				case protocol.PinHard:
					switch {
					case e.hard:
						if st != protocol.StatusOK {
							rt.Fatalf("hard re-pin: %s", st)
						}
					case m.pinned[k.NS]+sz > m.pinnedCap:
						if st != protocol.StatusErrPinQuota {
							rt.Fatalf("over-cap hard pin: %s", st)
						}
					default:
						if st != protocol.StatusOK {
							rt.Fatalf("hard pin: %s", st)
						}
						m.pinned[k.NS] += sz
						e.hard, e.soft = true, false
					}
				case protocol.Unpin:
					if st != protocol.StatusOK {
						rt.Fatalf("unpin: %s", st)
					}
					if e.hard {
						m.pinned[k.NS] -= sz
					}
					e.soft, e.hard = false, false
				}
			},
			"lease": func(rt *rapid.T) {
				k, ns, h := drawKey(rt)
				sub := rapid.SampledFrom([]uint8{protocol.LeaseGrant, protocol.LeaseRelease, protocol.TouchRecency}).Draw(rt, "sub")
				ttl := rapid.SampledFrom([]uint32{0, 100, 5000, 120_000}).Draw(rt, "ttl")
				st := s.TouchLease(ns, h, sub, ttl)
				if st == protocol.StatusNotFound {
					mustReconcileMiss(k, "TOUCH_LEASE")
					return
				}
				e := m.blocks[k]
				if e == nil {
					rt.Fatalf("TOUCH_LEASE %s for unknown key", st)
					return
				}
				if st != protocol.StatusOK {
					rt.Fatalf("touch_lease sub %d: %s", sub, st)
				}
				e.maybeGone = false // the lifecycle op reached a resident block
				switch sub {
				case protocol.LeaseGrant:
					d := ttl
					if d == 0 {
						d = leaseDefaultMS
					}
					if d > leaseMaxMS {
						d = leaseMaxMS
					}
					e.leaseUntil = m.now + int64(d)*int64(time.Millisecond)
				case protocol.LeaseRelease:
					e.leaseUntil = 0
				case protocol.TouchRecency:
					if ttl > 0 {
						e.ttlUntil = m.now + int64(ttl)*int64(time.Millisecond)
					}
				}
			},
			"delete": func(rt *rapid.T) {
				k, ns, h := drawKey(rt)
				force := rapid.Bool().Draw(rt, "force")
				st := s.Delete(ns, h, force)
				e := m.blocks[k]
				if e == nil {
					if st != protocol.StatusNotFound {
						rt.Fatalf("delete absent: %s", st)
					}
					return
				}
				if st == protocol.StatusNotFound {
					mustReconcileMiss(k, "DELETE")
					return
				}
				// The §3.7 truth table, mirrored.
				var want protocol.Status
				switch {
				case e.hard:
					want = protocol.StatusErrPinned
				case force:
					want = protocol.StatusOK
				case e.leaseUntil > m.now:
					want = protocol.StatusErrLeased
				case e.soft:
					want = protocol.StatusErrPinned
				default:
					want = protocol.StatusOK
				}
				if st != want {
					rt.Fatalf("delete gating: got %s want %s (hard=%v soft=%v leased=%v force=%v)",
						st, want, e.hard, e.soft, e.leaseUntil > m.now, force)
				}
				if st == protocol.StatusOK {
					delete(m.blocks, k) // soft pins carried no charge; hard never deletes
				} else {
					e.maybeGone = false // the gate answered: the block is resident
				}
			},
			"clock_advance": func(rt *rapid.T) {
				d := rapid.Int64Range(int64(time.Millisecond), int64(2*time.Hour)).Draw(rt, "d")
				cur += d
				m.now = cur
			},
			"pressure": func(rt *rapid.T) {
				// Junk-fill toward the wall, then one deterministic pass.
				for i := 0; i < 40; i++ {
					h := junkHash(junkSeq)
					junkSeq++
					data := bytes.Repeat([]byte{0xEE}, junkSize)
					sum := xxh3.Hash(data)
					st := s.Put(9, h, data, sum)
					if st == protocol.StatusOK {
						m.insert(eviction.Key{NS: 9, Hash: h}, data, sum)
						continue
					}
					if st != protocol.StatusErrQuotaBytes {
						rt.Fatalf("junk put: %s", st)
					}
					break
				}
				s.EvictNow()
				m.markPressure()
				// I2 probe: every protected block must still be resident.
				for k, b := range m.blocks {
					if m.protected(b) && !s.Contains(k.NS, k.Hash) {
						rt.Fatalf("I2: protected block %v gone after pressure (hard=%v leased=%v held=%d)",
							k, b.hard, b.leaseUntil > m.now, b.heldRefs)
					}
				}
			},
			"": func(rt *rapid.T) {
				var doc storeStats
				if err := json.Unmarshal(s.Stats(), &doc); err != nil {
					rt.Fatal(err)
				}
				// I3: accounting consistency with the model's view.
				definite, total := 0, 0
				var definiteBytes int64
				for _, b := range m.blocks {
					total++
					if !b.maybeGone {
						definite++
						definiteBytes += int64(len(b.data))
					}
				}
				if doc.Blocks < definite || doc.Blocks > total {
					rt.Fatalf("I3: store blocks=%d outside model [%d, %d]", doc.Blocks, definite, total)
				}
				if doc.Bytes < definiteBytes {
					rt.Fatalf("I3: store bytes=%d < model definite %d", doc.Bytes, definiteBytes)
				}
				if doc.ArenaFreeBytes < 0 || doc.ArenaFreeBytes > doc.ArenaBytes {
					rt.Fatalf("I3: arena accounting: %+v", doc)
				}
				for _, ns := range []uint32{1, 2, 3, 9} {
					want := m.pinned[ns] // zero when unpinned — the store must agree BOTH ways
					if got := doc.PinnedBytes[fmt.Sprint(ns)]; got != want {
						rt.Fatalf("I3: pinned_bytes ns%d = %d, model %d (a leaked or lost charge)", ns, got, want)
					}
				}
				// An extent-leak mutation (index entry removed, extent kept)
				// makes live extents outnumber index entries. The allowance:
				// a DELETED block still held by a GetRef legitimately keeps
				// its extent until the release fires (the whichever-Release-
				// hits-zero rule) — the deep gate itself caught the
				// unallowanced form of this bound as a false positive.
				if doc.LiveAllocs > int64(doc.Blocks)+int64(len(held)) {
					rt.Fatalf("I3: live_allocs %d > blocks %d + held %d — extent leak",
						doc.LiveAllocs, doc.Blocks, len(held))
				}
				// I5-light: one held view spot check.
				if len(held) > 0 {
					hr := held[len(held)-1]
					if !bytes.Equal(hr.view, hr.snap) {
						rt.Fatalf("I5: held view for %v drifted", hr.key)
					}
				}
				// I4: for every pooled hash resident in ≥2 namespaces
				// (byte-verified copies only), each GET returns its OWN
				// tenant's payload.
				for i := 0; i < 48; i++ {
					h := poolHash(i)
					var present []eviction.Key
					for _, ns := range []uint32{1, 2, 3} {
						k := eviction.Key{NS: ns, Hash: h}
						if b := m.blocks[k]; b != nil && !b.maybeGone {
							present = append(present, k)
						}
					}
					if len(present) < 2 {
						continue
					}
					for _, k := range present {
						data, _, ok := s.Get(k.NS, k.Hash)
						if !ok {
							rt.Fatalf("I4: %v vanished outside pressure", k)
						}
						e := m.blocks[k]
						if !bytes.Equal(data, e.data) {
							rt.Fatalf("I4: cross-namespace payload leak on hash %d ns %d", i, k.NS)
						}
						grantLease(e) // the probe's own GETs auto-lease; mirror them
					}
					break // one shared hash per step is enough
				}
			},
		})
	})
}
