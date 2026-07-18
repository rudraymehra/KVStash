package store

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"sync"
	"sync/atomic"
	"time"

	"github.com/kvstash/kvblockd/internal/eviction"
	"github.com/kvstash/kvblockd/internal/protocol"
	"github.com/kvstash/kvblockd/internal/store/dram"
	"github.com/kvstash/kvblockd/internal/store/nvme"
)

// Params tunes the tier orchestration.
type Params struct {
	DemoteWatermarkPct int    // demote at DRAM used ≥ this % (default 90 — below the evictor's 95)
	DemoteBatchPct     int    // demote down to (watermark − batch)% (default 5)
	AdmitMinHits       uint32 // blocks with fewer lifetime GETs are evicted, not written to NVMe (default 1)
	// PromoteWindow: a 2nd NVMe hit within this window promotes the block
	// back to DRAM. ≤0 = promotion DISABLED — zero genuinely means never
	// (the ladder caught withDefaults silently turning 0 into a minute;
	// callers wanting the default pass it explicitly, as config does).
	PromoteWindow  time.Duration
	Interval       time.Duration // demote/reclaim poll cadence (default 100ms)
	LeaseDefaultMS uint32        // mirrors dram.Params — NVMe hits auto-lease with the same clamp
	LeaseMaxMS     uint32
	Now            func() int64 // unix nanos; nil = real time (the fake-clock seam)
}

func (p Params) withDefaults() Params {
	if p.DemoteWatermarkPct == 0 {
		p.DemoteWatermarkPct = 90
	}
	if p.DemoteBatchPct == 0 {
		p.DemoteBatchPct = 5
	}
	if p.AdmitMinHits == 0 {
		p.AdmitMinHits = 1
	}
	// PromoteWindow deliberately has NO default: ≤0 = never promote.
	if p.Interval == 0 {
		p.Interval = 100 * time.Millisecond
	}
	if p.Now == nil {
		p.Now = func() int64 { return time.Now().UnixNano() }
	}
	return p
}

// Tiered is the DRAM→NVMe orchestrator: it implements the server's full
// Store surface by routing through the DRAM tier first and the NVMe index
// second, and owns the demotion / promotion / reclaim machinery. With no
// volumes it degrades to a transparent dram passthrough (DRAM-only configs
// never construct one — main.go wires dram.Store directly, byte-for-byte
// today's behavior).
//
// LOCK ORDER (extends dram's): dram shard lock → nvme index shard lock →
// (allocMu | policy | volume internals) — never inverted. GETs take only
// shard RLocks and the volume's segment read-hold.
type Tiered struct {
	d    *dram.Store
	pol  eviction.Policy
	vols []*nvme.Volume
	idx  *nvmeIndex
	p    Params
	now  func() int64

	kick    chan struct{}   // CanStore pressure nudge
	promote chan promoteReq // best-effort 2nd-hit promotions (drop on full)

	stopOnce sync.Once
	stopped  chan struct{}
	loopWG   sync.WaitGroup

	// Pass scratch (single demoter goroutine — same rationale as dram's).
	scUsages []eviction.DomainUsage
	scCands  []eviction.Candidate

	demotions    atomic.Uint64
	demoteDrops  atomic.Uint64 // queue-full / write-failed / gate-refused demotions
	dedupSkips   atomic.Uint64 // dual-resident: bytes already on NVMe, append skipped
	promotions   atomic.Uint64
	reclaims     atomic.Uint64
	reclaimSkips atomic.Uint64
	readBusy     atomic.Uint64
	checksumErrs atomic.Uint64

	recoveredBlocks int
	recoverySecs    float64
}

type promoteReq struct {
	k   dram.Key
	ref *nvmeRef
}

// NewTiered assembles the orchestrator over an already-built dram tier and
// opened volumes, seeding the NVMe index from recovery's findings. The
// policy must be the SAME instance attached to the dram store (victims are
// nominated once, dequeued once).
func NewTiered(d *dram.Store, pol eviction.Policy, vols []*nvme.Volume, reports []*nvme.RecoveryReport, recovered [][]nvme.RecoveredEntry, p Params) *Tiered {
	t := &Tiered{
		d:       d,
		pol:     pol,
		vols:    vols,
		idx:     newNvmeIndex(),
		p:       p.withDefaults(),
		kick:    make(chan struct{}, 1),
		promote: make(chan promoteReq, 64),
		stopped: make(chan struct{}),
	}
	t.now = t.p.Now
	for _, rep := range reports {
		if rep != nil {
			t.recoveredBlocks += rep.BlocksRecovered
			t.recoverySecs += rep.Duration.Seconds()
		}
	}
	for _, ents := range recovered {
		for _, e := range ents {
			ref := &nvmeRef{Loc: e.Loc, Len: e.Loc.Len, XXH3: e.XXH3}
			t.idx.put(dram.Key{NS: e.NS, Hash: e.Key}, ref)
		}
	}
	return t
}

// Start launches the demoter/reclaimer and promoter loops. The returned
// stop func cancels and WAITS — call it after Drain, before Close.
func (t *Tiered) Start(ctx context.Context) (stop func()) {
	ctx, cancel := context.WithCancel(ctx)
	t.loopWG.Add(2)
	go func() {
		defer t.loopWG.Done()
		tick := time.NewTicker(t.p.Interval)
		defer tick.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-tick.C:
			case <-t.kick:
			}
			t.demotePass(false)
			t.reclaimPass()
		}
	}()
	go func() {
		defer t.loopWG.Done()
		for {
			select {
			case <-ctx.Done():
				return
			case req := <-t.promote:
				t.promoteOne(req)
			}
		}
	}()
	return func() {
		t.stopOnce.Do(func() {
			cancel()
			t.loopWG.Wait()
			close(t.stopped)
		})
	}
}

func (t *Tiered) volumeFor(key [32]byte) *nvme.Volume {
	return t.vols[binary.LittleEndian.Uint64(key[:8])%uint64(len(t.vols))]
}

// ---------------------------------------------------------------------------
// The server Store surface.

// ExistsPrefix merges both indexes — pure map lookups, NO device I/O ever
// (the spy test enforces it): the EXISTS p99 contract survives tiering.
func (t *Tiered) ExistsPrefix(ns uint32, keys [][32]byte, withBitmap bool) (uint32, []protocol.Status) {
	var perKey []protocol.Status
	if withBitmap {
		perKey = make([]protocol.Status, len(keys))
	}
	var n uint32
	counting := true
	for i, key := range keys {
		hit := t.d.Contains(ns, key) || t.idx.contains(dram.Key{NS: ns, Hash: key})
		if hit && counting {
			n++
		} else {
			counting = false
		}
		if withBitmap {
			if hit {
				perKey[i] = protocol.StatusOK
			} else {
				perKey[i] = protocol.StatusNotFound
			}
		}
	}
	return n, perKey
}

// Get is the copying interface method (conformance/tooling parity).
func (t *Tiered) Get(ns uint32, key [32]byte) ([]byte, uint64, bool) {
	data, sum, rel, _, st := t.GetRefTier(ns, key)
	if st != protocol.StatusOK {
		return nil, 0, false
	}
	out := make([]byte, len(data))
	copy(out, data)
	rel()
	return out, sum, true
}

// GetRef is the legacy zero-copy extension (tier dropped) — the server
// prefers GetRefTier.
func (t *Tiered) GetRef(ns uint32, key [32]byte) (data []byte, xxh3 uint64, release func(), ok bool) {
	data, xxh3, release, _, st := t.GetRefTier(ns, key)
	return data, xxh3, release, st == protocol.StatusOK
}

// GetRefTier is the tier-routed zero-copy read: DRAM first; on miss the
// NVMe index, a bounded synchronous device read (verified before a byte
// escapes), auto-lease, and the 2nd-hit promotion probe. Status is OK /
// NOT_FOUND / ERR_BUSY (reader pool saturated — retryable, per-key).
func (t *Tiered) GetRefTier(ns uint32, key [32]byte) (data []byte, xxh3 uint64, release func(), tier string, st protocol.Status) {
	if data, sum, rel, ok := t.d.GetRef(ns, key); ok {
		return data, sum, rel, "dram", protocol.StatusOK
	}
	k := dram.Key{NS: ns, Hash: key}
	e := t.idx.get(k)
	if e == nil {
		return nil, 0, nil, "", protocol.StatusNotFound
	}
	vol := t.volumeFor(key)
	blob, rel, rst := vol.Read(e.Loc, ns, key, e.XXH3)
	switch rst {
	case nvme.ReadOK:
		// served — lease + promotion probe continue below the switch
	case nvme.ReadBusy:
		t.readBusy.Add(1)
		return nil, 0, nil, "", protocol.StatusErrBusy
	case nvme.ReadGone:
		// Segment retired under us — the retire already removed (or will
		// remove) the entries; a plain miss keeps the index self-consistent.
		return nil, 0, nil, "", protocol.StatusNotFound
	case nvme.ReadCorrupt:
		// Device rot: self-heal the entry (identity-gated) and miss. The
		// block is never served unverified.
		t.checksumErrs.Add(1)
		_, _ = t.idx.deleteIf(k, func(ref *nvmeRef) protocol.Status {
			if ref == e {
				return protocol.StatusOK
			}
			return protocol.StatusErrBusy // a newer entry replaced it — keep
		})
		return nil, 0, nil, "", protocol.StatusNotFound
	}

	now := t.now()
	// The §3.3 auto-lease + the promotion probe run under the shard lock so
	// they ORDER against the reclaim gate's shard-locked lease read (the
	// ladder's lease-vs-reclaim HIGH): the lease either lands before the
	// gate (segment skipped) or the entry is already removed (nil — no
	// promise minted). The bytes in flight are safe either way via the
	// segment read-hold.
	var prev int64
	present := false
	t.idx.withShardLock(k, func(ref *nvmeRef) {
		if ref != e {
			return // removed or replaced mid-read — do not lease a dangling ref
		}
		present = true
		extendNvmeLease(ref, now+int64(t.clampLeaseMS(0))*int64(time.Millisecond))
		prev = ref.LastAccess.Swap(now)
	})
	// 2nd hit within the window promotes — but a frame-boundary re-fetch
	// (the transport re-GETs a block that didn't fit the frame) arrives
	// within microseconds and must not count as a second touch (ladder
	// finding): require a minimum gap between qualifying hits.
	const promoteMinGap = int64(10 * time.Millisecond)
	if present && t.p.PromoteWindow > 0 && prev != 0 &&
		now-prev <= int64(t.p.PromoteWindow) && now-prev >= promoteMinGap {
		select { // best-effort: promotion is an optimization, drop on full
		case t.promote <- promoteReq{k: k, ref: e}:
		default:
		}
	}
	return blob, e.XXH3, rel, "nvme", protocol.StatusOK
}

// Put lands in DRAM (the write tier), after the write-once check consults
// BOTH homes — a block resident only on NVMe still answers OK_EXISTS /
// ERR_IMMUTABLE_CONFLICT exactly like a DRAM-resident one.
//
// The pre-check and d.Put span two lock domains, so a demotion's publish
// can slip between them (the ladder's write-once TOCTOU HIGH): the NVMe
// index was empty at the check, the dram entry moved to NVMe during d.Put,
// and a conflicting insert would be acked OK with the OLD bytes surviving
// on flash. The post-insert re-check closes it: on divergence the fresh
// insert is withdrawn and the caller gets the conflict the write-once
// contract promises.
func (t *Tiered) Put(ns uint32, key [32]byte, data []byte, xxh3 uint64) protocol.Status {
	k := dram.Key{NS: ns, Hash: key}
	if e := t.idx.get(k); e != nil {
		if e.XXH3 == xxh3 {
			return protocol.StatusOKExists
		}
		return protocol.StatusErrImmutableConflict
	}
	st := t.d.Put(ns, key, data, xxh3)
	if st == protocol.StatusOK {
		if e := t.idx.get(k); e != nil && e.XXH3 != xxh3 {
			_ = t.d.Delete(ns, key, true) // withdraw our fresh, unprotected insert
			return protocol.StatusErrImmutableConflict
		}
	}
	return st
}

// Contains consults both indexes (PUT BEGIN short-circuit).
func (t *Tiered) Contains(ns uint32, key [32]byte) bool {
	return t.d.Contains(ns, key) || t.idx.contains(dram.Key{NS: ns, Hash: key})
}

// Delete gates the NVMe entry by the same §3.7 truth table, then the DRAM
// tier. An NVMe-side refusal returns early (nothing touched); a DRAM-side
// refusal after the NVMe copy dropped is cache-legal (a copy may vanish any
// time; the surviving DRAM block keeps its protections). The segment bytes
// stay until reclaim — NVMe DELETE is not crash-durable (doc'd; a replay
// may resurrect the key, which the crash contract permits).
func (t *Tiered) Delete(ns uint32, key [32]byte, force bool) protocol.Status {
	k := dram.Key{NS: ns, Hash: key}
	removed, nvSt := t.deleteNvme(k, force)
	if nvSt == protocol.StatusErrLeased || nvSt == protocol.StatusErrPinned {
		return nvSt
	}
	dSt := t.d.Delete(ns, key, force)
	if dSt != protocol.StatusNotFound {
		return dSt
	}
	if removed {
		return protocol.StatusOK
	}
	return protocol.StatusNotFound
}

func (t *Tiered) deleteNvme(k dram.Key, force bool) (removed bool, st protocol.Status) {
	ref, st := t.idx.deleteIf(k, func(ref *nvmeRef) protocol.Status {
		return t.canDeleteNvme(ref, force)
	})
	return ref != nil, st
}

// canDeleteNvme mirrors lifecycle.canDelete on an nvmeRef (§3.7 order:
// hard pin always refuses; force overrides lease + soft pin). Caller holds
// the nvme shard lock (PinFlags read).
func (t *Tiered) canDeleteNvme(ref *nvmeRef, force bool) protocol.Status {
	if ref.hardPinned() {
		return protocol.StatusErrPinned
	}
	if force {
		return protocol.StatusOK
	}
	if ref.leased(t.now()) {
		return protocol.StatusErrLeased
	}
	if ref.pinned() {
		return protocol.StatusErrPinned
	}
	return protocol.StatusOK
}

// ---------------------------------------------------------------------------
// Lifecycle extension (TOUCH_LEASE / PIN on either tier).

func (t *Tiered) clampLeaseMS(ttlMS uint32) uint32 {
	if ttlMS == 0 {
		ttlMS = t.p.LeaseDefaultMS
	}
	if ttlMS > t.p.LeaseMaxMS {
		ttlMS = t.p.LeaseMaxMS
	}
	return ttlMS
}

func extendNvmeLease(ref *nvmeRef, until int64) {
	for {
		cur := ref.LeaseUntil.Load()
		if cur >= until {
			return
		}
		if ref.LeaseUntil.CompareAndSwap(cur, until) {
			return
		}
	}
}

// TouchLease routes §3.5 to whichever tier holds the block (DRAM wins on
// dual residency — it answers first).
func (t *Tiered) TouchLease(ns uint32, key [32]byte, sub uint8, ttlMS uint32) protocol.Status {
	if st := t.d.TouchLease(ns, key, sub, ttlMS); st != protocol.StatusNotFound {
		return st
	}
	// All NVMe lifecycle mutations run under the shard lock: they must
	// ORDER against the reclaim gate, and an entry the gate just removed
	// answers NOT_FOUND instead of minting a protection promise on a
	// dangling ref (the ladder's lease-vs-reclaim HIGH).
	now := t.now()
	st := protocol.StatusNotFound
	t.idx.withShardLock(dram.Key{NS: ns, Hash: key}, func(e *nvmeRef) {
		if e == nil {
			return
		}
		switch sub {
		case protocol.TouchRecency:
			if ttlMS > 0 {
				e.TTLUntil.Store(now + int64(ttlMS)*int64(time.Millisecond))
			}
			// Recency on the NVMe tier is FIFO-reclaimed — a metadata touch
			// does NOT count as a promotion hit (GETs do).
			st = protocol.StatusOK
		case protocol.LeaseGrant:
			extendNvmeLease(e, now+int64(t.clampLeaseMS(ttlMS))*int64(time.Millisecond))
			st = protocol.StatusOK
		case protocol.LeaseRelease:
			e.LeaseUntil.Store(0)
			st = protocol.StatusOK
		default:
			st = protocol.StatusErrUnsupported
		}
	})
	return st
}

// PinOp routes §3.6. Soft pin / unpin mutate the NVMe entry in place; a
// HARD pin on an NVMe-resident block promotes it to DRAM first and pins
// there — the per-namespace pinned-bytes ledger stays single (dram's), the
// cap stays unbypassable, and "must survive" honestly means DRAM residency.
// A promotion that cannot land (DRAM full, eviction can't help) answers
// ERR_BUSY — retryable, no new wire status.
func (t *Tiered) PinOp(ns uint32, key [32]byte, sub uint8) protocol.Status {
	if st := t.d.PinOp(ns, key, sub); st != protocol.StatusNotFound {
		return st
	}
	k := dram.Key{NS: ns, Hash: key}
	e := t.idx.get(k)
	if e == nil {
		return protocol.StatusNotFound
	}
	switch sub {
	case protocol.PinSoft, protocol.Unpin:
		st := protocol.StatusNotFound
		t.idx.withShardLock(k, func(ref *nvmeRef) {
			if ref == nil {
				return
			}
			if sub == protocol.PinSoft {
				ref.PinFlags = nvPinSoftBit
			} else {
				ref.PinFlags = 0
			}
			st = protocol.StatusOK
		})
		return st
	case protocol.PinHard:
		if st := t.promoteSync(k, e); st != protocol.StatusOK {
			return st
		}
		return t.d.PinOp(ns, key, sub)
	default:
		return protocol.StatusErrUnsupported
	}
}

// CanStore probes the WRITE tier (DRAM) and nudges the demoter — demotion,
// not just eviction, is how this store makes room.
func (t *Tiered) CanStore(n uint32) bool {
	ok := t.d.CanStore(n)
	if !ok {
		select {
		case t.kick <- struct{}{}:
		default:
		}
	}
	return ok
}

// ---------------------------------------------------------------------------

// Stats is the dram document (top level — existing assertions and the
// scrape collector keep working) plus an "nvme" sub-document.
func (t *Tiered) Stats() []byte {
	var doc map[string]any
	if err := json.Unmarshal(t.d.Stats(), &doc); err != nil {
		doc = map[string]any{"schema": 1, "store": "dram", "error": "dram stats decode failed"}
	}
	blocks, bytes := t.idx.stats()
	volStats := make(map[string]int64, 8)
	for _, v := range t.vols {
		v.StatsInto(volStats)
	}
	nv := map[string]any{
		"blocks":             blocks,
		"bytes":              bytes,
		"demotions_total":    t.demotions.Load(),
		"demote_drops_total": t.demoteDrops.Load(),
		"dedup_skips_total":  t.dedupSkips.Load(),
		"promotions_total":   t.promotions.Load(),
		// reclaims_total comes from the volume merge below — the tiered
		// counter counted the SAME events and silently shadowed it (ladder
		// finding); t.reclaims stays for internal assertions only.
		"reclaim_skips_total":   t.reclaimSkips.Load(),
		"read_busy_total":       t.readBusy.Load(),
		"checksum_errors_total": t.checksumErrs.Load(),
		"recovered_blocks":      t.recoveredBlocks,
		"recovery_seconds":      t.recoverySecs,
	}
	for k, v := range volStats {
		nv[k] = v
	}
	doc["nvme"] = nv
	b, err := json.Marshal(doc)
	if err != nil {
		return t.d.Stats()
	}
	return b
}

// Scrub re-reads and re-verifies EVERY NVMe-resident block — invariant I6's
// teeth: after a crash-recovery the index must match storage exactly (a bad
// read here is an indexed-but-unservable block).
func (t *Tiered) Scrub() (bad int) {
	type item struct {
		k dram.Key
		e *nvmeRef
	}
	var items []item
	t.idx.rangeAll(func(k dram.Key, ref *nvmeRef) bool {
		items = append(items, item{k, ref})
		return true
	})
	for _, it := range items {
		blob, rel, st := t.volumeFor(it.k.Hash).Read(it.e.Loc, it.k.NS, it.k.Hash, it.e.XXH3)
		switch st {
		case nvme.ReadOK:
			rel()
			_ = blob
		case nvme.ReadBusy:
			// Retry once synchronously — scrub runs on quiet stores.
			time.Sleep(time.Millisecond)
			if blob2, rel2, st2 := t.volumeFor(it.k.Hash).Read(it.e.Loc, it.k.NS, it.k.Hash, it.e.XXH3); st2 == nvme.ReadOK {
				rel2()
				_ = blob2
			} else {
				bad++
			}
		default:
			bad++
		}
	}
	return bad
}

// DemoteNow runs one synchronous demotion pass (modeltest's deterministic
// pressure handle) — it WAITS for every accepted append's completion, then
// runs a reclaim pass.
func (t *Tiered) DemoteNow() int {
	n := t.demotePass(true)
	t.reclaimPass()
	return n
}

// Close shuts the volumes down first (writer drain + readers), then the
// DRAM tier. The caller has already stopped the loops (stop func) and
// drained the server.
func (t *Tiered) Close() error {
	for _, v := range t.vols {
		_ = v.Close()
	}
	return t.d.Close()
}

// CrashForTest abandons everything as SIGKILL would: loops must already be
// stopped; volumes drop their queues and fds without sync/seal; the DRAM
// tier's contents simply vanish with the process (arena unmapped).
func (t *Tiered) CrashForTest() {
	for _, v := range t.vols {
		v.CrashForTest()
	}
	_ = t.d.Close()
}
