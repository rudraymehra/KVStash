package modeltest

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"testing"
	"time"

	"github.com/zeebo/xxh3"
	"pgregory.net/rapid"

	"github.com/kvstash/kvblockd/internal/eviction"
	"github.com/kvstash/kvblockd/internal/protocol"
	"github.com/kvstash/kvblockd/internal/store"
	"github.com/kvstash/kvblockd/internal/store/dram"
	"github.com/kvstash/kvblockd/internal/store/nvme"
	"github.com/kvstash/kvblockd/internal/tenant"
)

const (
	arenaBytes     = 8 << 20
	leaseDefaultMS = 500
	leaseMaxMS     = 60_000
	pinnedCap      = 1 << 20
	startNanos     = int64(1_700_000_000_000_000_000)
	junkSize       = 256 << 10

	// Two-tier machine geometry: 1 MiB segments so a handful of demotions
	// rotate + seal + checkpoint; a generous volume budget so reclaim fires
	// occasionally (cache-legal loss) without dominating the walk.
	segBytes   = 1 << 20
	volMax     = 64 << 20
	maxBlobLen = 512 << 10
	// Three-tier machine: a TIGHT volume budget so reclaim retires spilled
	// segments constantly — the retire-flip and cold-read paths carry real
	// traffic instead of firing once per walk.
	volMaxS3 = 4 << 20
)

// fakeS3 is the deterministic cold tier: DemoteSegment uploads INLINE
// (onUp fires before it returns — spillPass completes flips synchronously,
// so the async queue can never race the model), and objects live in a map
// that survives simulate_crash the way real S3 outlives the daemon process.
// The real async spiller/restorer have their own gofakes3 wall in
// internal/store/s3spill; the machine wants determinism, not queue coverage.
type fakeS3 struct {
	objects    map[uint64][]byte
	spilled    uint64
	putErrors  uint64
	rangedGets uint64
	restores   uint64
}

func newFakeS3() *fakeS3 { return &fakeS3{objects: map[uint64][]byte{}} }

// fakeSpill / fakeRestore split the two backend interfaces over one object
// store (their Stats signatures differ).
type (
	fakeSpill   struct{ *fakeS3 }
	fakeRestore struct{ *fakeS3 }
)

func (f fakeSpill) DemoteSegment(segID uint64, size int64, open func() (io.ReadSeekCloser, error), onUp func(uint64, bool)) bool {
	r, err := open()
	if err != nil {
		f.putErrors++
		onUp(segID, false)
		return true
	}
	b, rerr := io.ReadAll(r)
	_ = r.Close()
	if rerr != nil || int64(len(b)) != size {
		f.putErrors++
		onUp(segID, false)
		return true
	}
	f.objects[segID] = b // idempotent overwrite — a post-crash re-spill writes the same bytes
	f.spilled++
	onUp(segID, true)
	return true
}

func (f fakeSpill) Drop(_ context.Context, segID uint64) error {
	delete(f.objects, segID)
	return nil
}

func (f fakeSpill) Stats() (spilled, dropped, putErrors uint64) {
	return f.spilled, 0, f.putErrors
}

func (f fakeRestore) ReadRange(_ context.Context, segID uint64, off, n int64, dst []byte) error {
	obj, ok := f.objects[segID]
	if !ok || off < 0 || n < 0 || off+n > int64(len(obj)) {
		return fmt.Errorf("fakeS3: no bytes for seg %d [%d,+%d)", segID, off, n)
	}
	f.rangedGets++
	copy(dst, obj[off:off+n])
	return nil
}

// RestoreSegment streams the whole object inline — DemoteNow's restore pass
// completes deterministically, so the machine's cold 2nd hits exercise the
// real adopt/flip-back/GC path against the model's invariants.
func (f fakeRestore) RestoreSegment(_ context.Context, segID uint64, sink func(io.Reader) error) error {
	obj, ok := f.objects[segID]
	if !ok {
		return fmt.Errorf("fakeS3: no object for seg %d", segID)
	}
	if err := sink(bytes.NewReader(obj)); err != nil {
		return err
	}
	f.restores++
	return nil
}

func (f fakeRestore) Stats() (rangedGets, restores uint64) {
	return f.rangedGets, f.restores
}

// sutStore is what both kinds under test satisfy (the model is tier-blind).
type sutStore interface {
	ExistsPrefix(ns uint32, keys [][32]byte, withBitmap bool) (uint32, []protocol.Status)
	Get(ns uint32, key [32]byte) ([]byte, uint64, bool)
	GetRef(ns uint32, key [32]byte) ([]byte, uint64, func(), bool)
	Put(ns uint32, key [32]byte, data []byte, xxh3 uint64) protocol.Status
	Contains(ns uint32, key [32]byte) bool
	Delete(ns uint32, key [32]byte, force bool) protocol.Status
	TouchLease(ns uint32, key [32]byte, sub uint8, ttlMS uint32) protocol.Status
	PinOp(ns uint32, key [32]byte, sub uint8) protocol.Status
	Stats() []byte
	Close() error
}

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
	Blocks         int              `json:"blocks"` // DRAM-resident (the top-level doc IS the dram tier)
	Bytes          int64            `json:"bytes"`
	ArenaBytes     int64            `json:"arena_bytes"`
	ArenaFreeBytes int64            `json:"arena_free_bytes"`
	PinnedBytes    map[string]int64 `json:"pinned_bytes"`
	EvictionsTotal uint64           `json:"evictions_total"`
	LiveAllocs     int64            `json:"live_allocs"`
	Nvme           *struct {
		Blocks int   `json:"blocks"`
		Bytes  int64 `json:"bytes"`
	} `json:"nvme"` // present only on the tiered machine
	S3 *struct {
		Blocks int   `json:"blocks"`
		Bytes  int64 `json:"bytes"`
	} `json:"s3"` // present only on the three-tier machine
}

// residentBlocks/residentBytes are the store's whole-cache view across tiers.
func (d *storeStats) residentBlocks() int {
	n := d.Blocks
	if d.Nvme != nil {
		n += d.Nvme.Blocks
	}
	if d.S3 != nil {
		n += d.S3.Blocks
	}
	return n
}

func (d *storeStats) residentBytes() int64 {
	n := d.Bytes
	if d.Nvme != nil {
		n += d.Nvme.Bytes
	}
	if d.S3 != nil {
		n += d.S3.Bytes
	}
	return n
}

// TestStoreModel is the harness: the real store vs the reference model over
// a randomized op stream, invariants after every step. Both policies AND
// both store kinds are drawn — "dram" is the single-tier machine, "tiered"
// stacks a real NVMe volume underneath and adds crash/recovery (I6).
func TestStoreModel(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		policyName := rapid.SampledFrom([]string{"s3fifo", "sampled-lru"}).Draw(rt, "policy")
		kind := rapid.SampledFrom([]string{"dram", "tiered", "s3"}).Draw(rt, "kind")
		cur := startNanos
		// The cold tier outlives crashes (real S3 outlives the process); one
		// object store per walk, rebuilt stores keep pointing at it.
		coldTier := newFakeS3()

		volDir, err := os.MkdirTemp("", "kvb-modeltest-*")
		if err != nil {
			rt.Fatal(err)
		}
		defer func() { _ = os.RemoveAll(volDir) }()

		// mkSUT builds a fresh store over volDir — called once at start and
		// again after every simulate_crash (recovery is the point).
		var s sutStore
		var ds *dram.Store
		var tt *store.Tiered
		// Tenant quotas: ns 1 runs TIGHT (forces the refusal + refund paths
		// constantly), ns 2 loose, ns 3/9 unlimited. The accountant is
		// process state — mkSUT rebuilds it on every (re)start and recovery
		// re-seeds the NVMe side, exactly like the daemon.
		var quotas *tenant.Quotas
		mkQuotas := func() *tenant.Quotas {
			reg := tenant.NewRegistry("ns1", 1, "t1")
			if err := reg.Add(&tenant.Namespace{ID: 2, Name: "ns2", TokenHash: [32]byte{2}}); err != nil {
				rt.Fatal(err)
			}
			reg.SetQuota("ns1", tenant.TierDRAM, 128<<10) // tight: two 64K blocks, refuses the 256K draw
			reg.SetQuota("ns1", tenant.TierNVMe, 256<<10)
			reg.SetQuota("ns2", tenant.TierDRAM, 1<<20)
			return tenant.NewQuotas(reg)
		}
		mkSUT := func() {
			pol, perr := eviction.New(policyName, 8192)
			if perr != nil {
				rt.Fatal(perr)
			}
			arena, aerr := dram.NewArena(arenaBytes, false)
			if aerr != nil {
				rt.Fatal(aerr)
			}
			quotas = mkQuotas()
			ds = dram.New(arena, dram.Params{
				LeaseDefaultMS: leaseDefaultMS,
				LeaseMaxMS:     leaseMaxMS,
				PinnedBytesCap: pinnedCap,
				Now:            func() int64 { return cur },
				Quotas:         quotas,
			})
			ds.AttachPolicy(pol) // evictor NOT started: pressure only via the explicit op
			if kind == "dram" {
				s, tt = ds, nil
				return
			}
			budget := int64(volMax)
			if kind == "s3" {
				budget = volMaxS3 // tight: reclaim retires spilled segments constantly
			}
			vol, _, ents, verr := nvme.OpenVolume(nvme.VolumeParams{
				Dir: volDir, SegmentBytes: segBytes, MaxBytes: budget,
				ReadWorkers: 2, CkptEverySegs: 2, MaxBlobLen: maxBlobLen,
				Now: func() int64 { return cur },
			})
			if verr != nil {
				rt.Fatal(verr)
			}
			p := store.Params{
				LeaseDefaultMS: leaseDefaultMS, LeaseMaxMS: leaseMaxMS,
				PromoteWindow: time.Hour, // wide window; promotions happen via GETs (no background loops started)
				Now:           func() int64 { return cur },
				Quotas:        quotas,
			}
			if kind == "s3" {
				p.Spill = fakeSpill{coldTier}
				p.Restore = fakeRestore{coldTier}
				p.S3ReadTimeout = time.Second
				// Aggressive demotion (drain the arena 50%→10% per pass): a
				// single pressure action pushes multiple segments through
				// seal→spill→reclaim, so retire-flips and cold reads carry
				// real traffic inside ordinary-length walks instead of
				// needing ~9 pressure draws to first fill the tight budget.
				p.DemoteWatermarkPct = 50
				p.DemoteBatchPct = 40
			}
			tt = store.NewTiered(ds, pol, []*nvme.Volume{vol}, nil,
				[][]nvme.RecoveredEntry{ents}, p)
			s = tt
		}
		mkSUT()

		m := newModel(startNanos, pinnedCap)
		var held []heldRef
		junkSeq := 0
		// ONE junk pattern, shared by every junk key in the model
		// (insertShared) — per-key copies were the harness's memory ceiling.
		junkData := bytes.Repeat([]byte{0xEE}, junkSize)
		junkSum := xxh3.Hash(junkData)
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
		// ghostOrFatal resolves a store observation of a key the model thinks
		// absent: legal ONLY as post-crash resurrection of a deleted key
		// (NVMe DELETE is not crash-durable — recovery may replay it; the
		// ladder's confirmed model/store divergence). Returns the
		// materialized block, or fails the run.
		ghostOrFatal := func(k eviction.Key, what string) *mBlock {
			if !m.resurrectable(k) {
				rt.Fatalf("%s hit for a key the model never stored: %v", what, k)
				return nil
			}
			return m.materializeGhost(k)
		}
		// unknownPinRange bounds the store-ledger charge the model can't see:
		// hard pins on resurrected ghosts whose true size hasn't pinned down
		// yet (candidate sizes bound it). Zero/zero when none exist — the
		// common case, where every pinned-bytes check stays exact.
		unknownPinRange := func(ns uint32) (lo, hi int64) {
			for kk, b := range m.blocks {
				if kk.NS != ns || !b.hard || !b.pinChargeUnknown {
					continue
				}
				l, h := candidateSizeRange(b.anyOf)
				lo += l
				hi += h
			}
			return
		}
		// grantLease mirrors the store's §3.3 auto-lease on every OK read —
		// MONOTONIC, like the store's extendLease: a 5s auto-grant never
		// truncates a live longer lease (the divergence here was the deep
		// gate's 180-second find: model said the lease lapsed, the store
		// correctly still held it).
		grantLease := func(b *mBlock) {
			if v := m.now + int64(leaseDefaultMS)*int64(time.Millisecond); v > b.leaseUntil {
				b.leaseUntil = v
			}
		}

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
					switch {
					case st == protocol.StatusOK:
						m.insert(k, data, sum)
					case st == protocol.StatusErrQuotaBytes: // full arena, no evictor: legal
					case st == protocol.StatusOKExists && m.resurrectable(k):
						// A crash resurrected the deleted key with OUR sum:
						// the resident content is (xxh3-)identical to this put.
						b := m.materializeGhost(k)
						if !m.resolveBytes(k.NS, b, data, sum) {
							rt.Fatalf("put resurrected-dup: OK_EXISTS but %v matches no history", k)
						}
						b.maybeGone = false
					case st == protocol.StatusErrImmutableConflict && m.resurrectable(k):
						// Resurrected with some OTHER historical content —
						// the write-once alarm is correct; content pins down
						// at the next GET.
						m.materializeGhost(k).maybeGone = false
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
					e = ghostOrFatal(k, "GET")
				}
				// I1: byte-identical (to the block's content, or to ONE
				// historical content for a resurrected ghost), right
				// checksum, never cross-key/stale/invented.
				if !m.resolveBytes(k.NS, e, data, sum) {
					rt.Fatalf("I1: GET bytes/sum mismatch for %v (%d bytes served)", k, len(data))
				}
				e.maybeGone = false // byte-verified present
				grantLease(e)
			},
			"get_junk": func(rt *rapid.T) {
				// The junk fill is what actually sinks through the tiers (it
				// dominates demoted bytes, so flipped segments are mostly
				// junk) — without reads against it the cold path carries no
				// GET traffic and the s3 machine verifies nothing.
				if junkSeq == 0 {
					return
				}
				seq := rapid.IntRange(0, junkSeq-1).Draw(rt, "jseq")
				h := junkHash(seq)
				k := eviction.Key{NS: 9, Hash: h}
				data, sum, ok := s.Get(9, h)
				if !ok {
					mustReconcileMiss(k, "GET-junk")
					return
				}
				e := m.blocks[k]
				if e == nil {
					e = ghostOrFatal(k, "GET-junk")
				}
				if !m.resolveBytes(k.NS, e, data, sum) {
					rt.Fatalf("I1: GET-junk bytes/sum mismatch for %v (%d bytes served)", k, len(data))
				}
				e.maybeGone = false
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
					e = ghostOrFatal(k, "GETREF")
				}
				if !m.resolveBytes(k.NS, e, view, sum) {
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
					e = ghostOrFatal(k, "EXISTS")
				}
				// Presence proven at this instant: an un-marked block that
				// later vanishes without pressure is a store bug (the
				// sticky-maybeGone hole the review closed). A resurrected
				// ghost keeps anyOf until bytes pin it down.
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
					e = ghostOrFatal(k, "PIN")
				}
				e.maybeGone = false // the pin op reached a resident block
				if tt != nil && sub == protocol.PinHard && st == protocol.StatusErrBusy {
					return // NVMe-resident hard pin promotes first; a full arena refuses (retryable) — pin unapplied
				}
				// refundHard returns a hard pin's charge to the ledger. An
				// unknown charge (ghost never pinned down) contributed
				// nothing to the ledger — clearing the flag nets zero on
				// both sides of the mirror, since the store refunds exactly
				// what it charged.
				refundHard := func() {
					if !e.hard {
						return
					}
					if e.pinChargeUnknown {
						e.pinChargeUnknown = false
					} else {
						m.pinned[k.NS] -= e.pinCharge
					}
					e.pinCharge = 0
				}
				sz := int64(len(e.data))
				switch sub {
				case protocol.PinSoft:
					if st != protocol.StatusOK {
						rt.Fatalf("soft pin: %s", st)
					}
					refundHard() // downgrade refunds
					e.soft, e.hard = true, false
				case protocol.PinHard:
					switch {
					case e.hard:
						if st != protocol.StatusOK {
							rt.Fatalf("hard re-pin: %s", st)
						}
					case e.anyOf != nil:
						// Resurrected ghost: the store charged the block's
						// TRUE size, which the model can't know yet — and
						// so can't predict the pin-quota verdict either.
						// Both outcomes are explainable; I3 bounds the
						// window and resolveBytes retro-tightens it.
						switch st {
						case protocol.StatusOK:
							e.hard, e.soft = true, false
							e.pinChargeUnknown = true
						case protocol.StatusErrPinQuota: // pin unapplied
						default:
							rt.Fatalf("hard pin on resurrected ghost: %s", st)
						}
					default:
						lo, hi := unknownPinRange(k.NS)
						switch {
						case m.pinned[k.NS]+lo+sz > m.pinnedCap:
							if st != protocol.StatusErrPinQuota {
								rt.Fatalf("over-cap hard pin: %s", st)
							}
						case m.pinned[k.NS]+hi+sz > m.pinnedCap:
							// Unknown ghost charges straddle the cap — the
							// store's verdict depends on sizes the model
							// can't see yet. Both outcomes are explainable.
							if st == protocol.StatusOK {
								m.pinned[k.NS] += sz
								e.pinCharge = sz
								e.hard, e.soft = true, false
							} else if st != protocol.StatusErrPinQuota {
								rt.Fatalf("cap-straddling hard pin: %s", st)
							}
						default:
							if st != protocol.StatusOK {
								rt.Fatalf("hard pin: %s", st)
							}
							m.pinned[k.NS] += sz
							e.pinCharge = sz
							e.hard, e.soft = true, false
						}
					}
				case protocol.Unpin:
					if st != protocol.StatusOK {
						rt.Fatalf("unpin: %s", st)
					}
					refundHard()
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
					e = ghostOrFatal(k, "TOUCH_LEASE")
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
					if v := m.now + int64(d)*int64(time.Millisecond); v > e.leaseUntil {
						e.leaseUntil = v // grant/extend — never shorten (mirrors extendLease)
					}
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
					if st == protocol.StatusNotFound {
						return
					}
					// A non-NotFound answer for a model-absent key: legal
					// only as a resurrected ghost (post-crash replay).
					e = ghostOrFatal(k, "DELETE")
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
					m.noteDeleted(k) // soft pins carried no charge; hard never deletes; a later CRASH may resurrect
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
				// Junk-fill toward the wall, then one deterministic pass:
				// eviction on the dram machine; demotion + reclaim + eviction
				// on the tiered one (the same lossy-cache posture — the model
				// marks unprotected blocks maybeGone either way; demoted
				// blocks simply KEEP answering from the other tier).
				for i := 0; i < 40; i++ {
					h := junkHash(junkSeq)
					junkSeq++
					st := s.Put(9, h, junkData, junkSum)
					if st == protocol.StatusOK {
						m.insertShared(eviction.Key{NS: 9, Hash: h}, junkData, junkSum)
						continue
					}
					if st != protocol.StatusErrQuotaBytes {
						rt.Fatalf("junk put: %s", st)
					}
					break
				}
				if tt != nil {
					tt.DemoteNow() // demote pass + reclaim pass, synchronous
				}
				ds.EvictNow()
				m.markPressure()
				// I2 probe: every protected block must still be resident.
				for k, b := range m.blocks {
					if m.protected(b) && !s.Contains(k.NS, k.Hash) {
						rt.Fatalf("I2: protected block %v gone after pressure (hard=%v leased=%v held=%d)",
							k, b.hard, b.leaseUntil > m.now, b.heldRefs)
					}
				}
			},
			"simulate_crash": func(rt *rapid.T) {
				if tt == nil {
					return // single-tier machine has no crash story (nothing persists)
				}
				if len(held) > 0 {
					return // a crash with live reader views is the harness's own UB, not the store's
				}
				tt.CrashForTest() // volumes drop fds mid-flight, arena unmapped — SIGKILL semantics
				mkSUT()           // reopen the SAME volume dir: real recovery runs
				m.crashed()       // deleted-key ghosts become resurrectable from here on
				// The model after a crash: DRAM contents vanished, protection
				// state (leases/pins — memory-only) vanished, and recovery may
				// resurface ANY committed-and-persisted content for a key —
				// not just the latest. Force-delete + re-put under one key
				// composes with the non-crash-durable NVMe DELETE: the
				// pre-delete bytes legally come back while the re-put (DRAM-
				// only at the kill) is legally lost. The deep gauntlet found
				// this after 795 walks: the model pinned the key to its
				// LATEST content and called the resurrected older version an
				// I1 violation. So every surviving block reverts to the anyOf
				// form the ghost path already uses; the next byte-carrying
				// observation pins it down. I1 keeps its teeth — served bytes
				// outside the key's committed history still fail.
				for k, b := range m.blocks {
					b.maybeGone = true
					b.leaseUntil = 0
					b.ttlUntil = 0
					b.soft, b.hard = false, false
					b.pinCharge, b.pinChargeUnknown = 0, false
					b.data, b.xxh3, b.anyOf = nil, 0, m.history[k]
				}
				m.pinned = map[uint32]int64{}
				// I6: the recovered index matches storage — every entry the
				// store still vouches for must read back checksum-clean.
				if bad := tt.Scrub(); bad != 0 {
					rt.Fatalf("I6: post-recovery scrub found %d unservable indexed blocks", bad)
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
				// Dual residency (a promoted block counted in BOTH tiers) can
				// push the resident count past the model's key count — the
				// upper bound doubles on the tiered machine, honestly.
				maxResident := total
				if doc.Nvme != nil {
					maxResident = 2 * total
				}
				if got := doc.residentBlocks(); got < definite || got > maxResident {
					rt.Fatalf("I3: store blocks=%d outside model [%d, %d]", got, definite, maxResident)
				}
				if got := doc.residentBytes(); got < definiteBytes {
					rt.Fatalf("I3: store bytes=%d < model definite %d", got, definiteBytes)
				}
				if doc.ArenaFreeBytes < 0 || doc.ArenaFreeBytes > doc.ArenaBytes {
					rt.Fatalf("I3: arena accounting: %+v", doc)
				}
				// I3-quota: the DRAM ledger is STRICTLY bounded by the quota
				// on this direct-Put path (no BEGIN reserve, so no slack) and
				// never negative. NVMe may legally exceed its quota (recovery
				// over-seed + never-failing transfers); the kvbdebug build
				// separately panics on any underflow.
				for _, nsq := range []uint32{1, 2} {
					if lim := quotas.Limit(nsq, tenant.TierDRAM); lim > 0 {
						if u := quotas.Usage(nsq, tenant.TierDRAM); u > lim {
							rt.Fatalf("I3q: ns%d dram usage %d exceeds quota %d", nsq, u, lim)
						}
					}
					for _, tr := range []tenant.Tier{tenant.TierDRAM, tenant.TierNVMe, tenant.TierS3} {
						if u := quotas.Usage(nsq, tr); u < 0 {
							rt.Fatalf("I3q: ns%d %s usage negative: %d", nsq, tr, u)
						}
					}
				}
				for _, ns := range []uint32{1, 2, 3, 9} {
					// Exact when no unknown ghost charges exist (the common
					// case); bounded by candidate sizes when a hard pin
					// landed on a not-yet-pinned-down resurrected block.
					lo, hi := unknownPinRange(ns)
					want, got := m.pinned[ns], doc.PinnedBytes[fmt.Sprint(ns)]
					if got < want+lo || got > want+hi {
						rt.Fatalf("I3: pinned_bytes ns%d = %d, model %d..%d (a leaked or lost charge)", ns, got, want+lo, want+hi)
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
						// Only byte-verified, pinned-down copies participate —
						// an unresolved resurrected ghost has no bytes to
						// compare yet.
						if b := m.blocks[k]; b != nil && !b.maybeGone && b.anyOf == nil {
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

// TestColdRestoreFires is the deterministic companion the random walks
// cannot promise: it drives the exact 2nd-cold-hit sequence against the
// SAME fakes the machine uses and asserts the backend's restore counter
// actually moved — so "restore never triggers" (a dead trigger wire, a
// promote-window regression) can never pass this package silently.
func TestColdRestoreFires(t *testing.T) {
	cur := startNanos
	coldTier := newFakeS3()
	pol, err := eviction.New("s3fifo", 8192)
	if err != nil {
		t.Fatal(err)
	}
	arena, err := dram.NewArena(arenaBytes, false)
	if err != nil {
		t.Fatal(err)
	}
	ds := dram.New(arena, dram.Params{
		LeaseDefaultMS: leaseDefaultMS, LeaseMaxMS: leaseMaxMS,
		Now: func() int64 { return cur },
	})
	ds.AttachPolicy(pol)
	vol, _, ents, err := nvme.OpenVolume(nvme.VolumeParams{
		Dir: t.TempDir(), SegmentBytes: segBytes, MaxBytes: volMaxS3,
		ReadWorkers: 2, CkptEverySegs: 2, MaxBlobLen: maxBlobLen,
		Now: func() int64 { return cur },
	})
	if err != nil {
		t.Fatal(err)
	}
	tt := store.NewTiered(ds, pol, []*nvme.Volume{vol}, nil,
		[][]nvme.RecoveredEntry{ents}, store.Params{
			LeaseDefaultMS: leaseDefaultMS, LeaseMaxMS: leaseMaxMS,
			PromoteWindow: time.Hour,
			Now:           func() int64 { return cur },
			Spill:         fakeSpill{coldTier}, Restore: fakeRestore{coldTier},
			S3ReadTimeout:      time.Second,
			DemoteWatermarkPct: 50, DemoteBatchPct: 40,
		})
	defer func() { _ = tt.Close() }()

	key := func(i int) [32]byte {
		var k [32]byte
		k[0], k[1], k[31] = byte(i), byte(i>>8), 0xD7 //nolint:gosec // G115: test index mixing
		return k
	}
	blob := func(i int) []byte {
		return bytes.Repeat([]byte{byte(i), byte(0x91 ^ i)}, junkSize/2) //nolint:gosec // G115: test payload pattern
	}
	// Fill past the tight volume budget so reclaim retire-flips spilled
	// segments — the machine's own three-tier pressure shape.
	for i := 0; i < 48; i++ {
		if st := tt.Put(3, key(i), blob(i), xxh3.Hash(blob(i))); st != protocol.StatusOK {
			t.Fatalf("put %d: %s", i, st)
		}
		cur += int64(time.Second)
		tt.DemoteNow()
	}
	// Find a cold-served key (its GET is the 1st hit, arming the window).
	cold := -1
	for i := 0; i < 48 && cold < 0; i++ {
		if _, _, rel, tier, st := tt.GetRefTier(3, key(i)); st == protocol.StatusOK {
			if tier == "s3" {
				cold = i
			}
			rel()
		}
	}
	if cold < 0 {
		t.Fatal("no key served from the cold tier — the fill never flipped a segment")
	}
	// The 2nd cold hit ≥ the min gap later + one synchronous pass = restore.
	for try := 0; try < 10; try++ {
		cur += int64(20 * time.Millisecond)
		_, _, rel, tier, st := tt.GetRefTier(3, key(cold))
		if st != protocol.StatusOK {
			t.Fatalf("cold key vanished (try %d): %s", try, st)
		}
		rel()
		if tier != "s3" {
			break // restored home
		}
		tt.DemoteNow() // drains the restore queue synchronously
	}
	if _, restores := (fakeRestore{coldTier}).Stats(); restores == 0 {
		t.Fatal("the 2nd cold hit never triggered a whole-segment restore (restores=0)")
	}
	// And the restored key serves locally again, byte-identical.
	data, _, rel, tier, st := tt.GetRefTier(3, key(cold))
	if st != protocol.StatusOK || tier != "nvme" {
		t.Fatalf("post-restore get: %s tier=%q, want OK nvme", st, tier)
	}
	if !bytes.Equal(data, blob(cold)) {
		rel()
		t.Fatal("post-restore bytes differ")
	}
	rel()
}
