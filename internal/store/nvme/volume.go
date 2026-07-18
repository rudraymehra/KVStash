package nvme

import (
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"
)

// VolumeParams configures one volume (one directory, normally one device).
type VolumeParams struct {
	Dir            string
	SegmentBytes   int64 // fixed fallocated size of every segment file
	MaxBytes       int64 // reclaim pressure reference for this volume
	SyncEveryBytes int64 // group-commit cadence (fdatasync ledger)
	ReadWorkers    int
	CkptEverySegs  int    // checkpoint every N seals; 0 = never
	MaxBlobLen     uint32 // record payload cap (mirrors the wire limit)
	Backend        IOBackend
	Now            func() int64 // unix nanos; nil = real time
	Logger         *slog.Logger
}

// AppendReq is one demotion write. Data is typically an arena view whose
// refcount the CALLER holds until OnWritten fires — the writer copies it
// into the staging buffer and never retains it.
type AppendReq struct {
	NS   uint32
	Key  [32]byte
	XXH3 uint64
	Data []byte
	// OnWritten runs on the writer goroutine with no volume locks held.
	// ok=false means the record was not written (ENOSPC/rotation failure);
	// the Loc is only meaningful when ok.
	OnWritten func(loc Loc, ok bool)
}

// ReadStatus classifies a Volume.Read outcome.
type ReadStatus uint8

const (
	ReadOK      ReadStatus = iota
	ReadBusy               // reader pool saturated — retryable
	ReadGone               // segment dying/retired — treat as miss, keep index self-consistent
	ReadCorrupt            // header/nskey/xxh3 verify failed or device error — caller should self-heal the index entry
)

// segment is one on-disk segment file. Sealed segments are immutable;
// the (single) active segment is written only by the writer goroutine.
// size is the FILE's own geometry — recovered segments keep the size they
// were created with, so retuning nvme_segment_bytes never orphans them.
type segment struct {
	id      uint32
	f       File
	path    string
	size    int64
	sealed  bool
	dataEnd uint32        // bytes of record region (writer-owned until sealed)
	entries []footerEntry // sealed: the footer table; active: accumulated so far
	dying   atomic.Bool   // reclaim in progress — new reads refused
	reads   atomic.Int64  // in-flight reads (reclaim waits for drain)
}

// acquireRead registers an in-flight read unless the segment is dying.
// The increment-then-recheck closes the race with RetireBegin.
func (s *segment) acquireRead() bool {
	if s.dying.Load() {
		return false
	}
	s.reads.Add(1)
	if s.dying.Load() {
		s.reads.Add(-1)
		return false
	}
	return true
}

func (s *segment) releaseRead() { s.reads.Add(-1) }

// Volume is one NVMe volume: an append-only segment log + writer goroutine +
// bounded reader pool. It owns no key index — the tiered store does.
type Volume struct {
	p       VolumeParams
	backend IOBackend
	log     *slog.Logger
	pool    *bufPool
	now     func() int64

	mu       sync.RWMutex // guards segs/active/nextID/readOnly/sealsSinceCkpt/ckptSeq
	segs     map[uint32]*segment
	active   *segment
	nextID   uint32
	readOnly bool

	sealsSinceCkpt int
	ckptSeq        uint64

	reqs       chan AppendReq
	writerStop chan struct{} // closed by Close/CrashForTest — writer exits
	writerDone chan struct{}
	readq      chan readReq  // never closed — readStop is the shutdown signal
	readStop   chan struct{} // closed at shutdown; unblocks queued Read callers
	readerWG   sync.WaitGroup

	used     atomic.Int64 // fallocated bytes across live segments
	appended atomic.Uint64
	seals    atomic.Uint64
	ckpts    atomic.Uint64
	enospc   atomic.Uint64
	reclaims atomic.Uint64
	crashed  atomic.Bool
	closed   atomic.Bool
}

// OpenVolume recovers the directory (checkpoint + footer scan + tail
// truncation), starts the writer and reader pool, and returns what recovery
// found so the tiered store can rebuild its index.
func OpenVolume(p VolumeParams) (*Volume, *RecoveryReport, []RecoveredEntry, error) {
	if p.Backend == nil {
		p.Backend = DefaultBackend()
	}
	if p.Logger == nil {
		p.Logger = slog.Default()
	}
	if p.Now == nil {
		p.Now = func() int64 { return time.Now().UnixNano() }
	}
	if p.ReadWorkers <= 0 {
		p.ReadWorkers = 16
	}
	if p.SyncEveryBytes <= 0 {
		p.SyncEveryBytes = 8 << 20
	}
	if p.MaxBlobLen == 0 {
		p.MaxBlobLen = 32 << 20
	}
	minSeg := MinSegmentBytes(p.MaxBlobLen)
	if p.SegmentBytes < minSeg || p.SegmentBytes%recordAlign != 0 || p.SegmentBytes >= int64(^uint32(0)) {
		return nil, nil, nil, fmt.Errorf("nvme: segment_bytes %d invalid (need aligned, ≥%d for max blob %d, <4GiB)",
			p.SegmentBytes, minSeg, p.MaxBlobLen)
	}
	if err := os.MkdirAll(p.Dir, 0o750); err != nil {
		return nil, nil, nil, fmt.Errorf("nvme: mkdir volume dir: %w", err)
	}

	v := &Volume{
		p:          p,
		backend:    p.Backend,
		log:        p.Logger,
		now:        p.Now,
		pool:       newBufPool(uint32(recordSpan(p.MaxBlobLen)), 2*p.ReadWorkers), //nolint:gosec // G115: span < 4 GiB (validated above)
		segs:       make(map[uint32]*segment),
		reqs:       make(chan AppendReq, 128),
		writerStop: make(chan struct{}),
		writerDone: make(chan struct{}),
		readq:      make(chan readReq, 4*p.ReadWorkers),
		readStop:   make(chan struct{}),
	}

	report, entries, err := v.recoverDir()
	if err != nil {
		v.pool.Close()
		return nil, nil, nil, err
	}
	if err := v.openActive(); err != nil {
		v.closeSegments()
		v.pool.Close()
		return nil, nil, nil, err
	}

	go v.writerLoop()
	for i := 0; i < p.ReadWorkers; i++ {
		v.readerWG.Add(1)
		go v.readerLoop()
	}
	return v, report, entries, nil
}

// Append hands one record to the writer. Non-blocking: false means the
// queue is full or the volume is read-only/closed — the caller re-admits
// the victim and drops the demotion (fire-and-forget posture).
func (v *Volume) Append(r AppendReq) bool {
	if v.closed.Load() {
		return false
	}
	v.mu.RLock()
	ro := v.readOnly
	v.mu.RUnlock()
	if ro {
		return false
	}
	select {
	case v.reqs <- r:
		return true
	default:
		return false
	}
}

// Read serves one record synchronously through the bounded reader pool,
// verifying magic, nskey, and xxh3 before a byte escapes. release returns
// the pooled buffer AND the segment read-hold; it must be called exactly
// once on ReadOK.
func (v *Volume) Read(loc Loc, ns uint32, key [32]byte, want uint64) (data []byte, release func(), st ReadStatus) {
	if v.closed.Load() {
		return nil, nil, ReadGone
	}
	v.mu.RLock()
	seg := v.segs[loc.SegmentID]
	v.mu.RUnlock()
	if seg == nil {
		return nil, nil, ReadGone
	}
	if !seg.acquireRead() {
		return nil, nil, ReadGone
	}
	req := readReq{seg: seg, loc: loc, ns: ns, key: key, want: want, done: make(chan readResult, 1)}
	select {
	case v.readq <- req:
	default:
		seg.releaseRead()
		return nil, nil, ReadBusy
	}
	// readStop unblocks a caller whose request was queued when shutdown
	// drained the pool (the ladder's Read-vs-Close hang/panic trap). A
	// worker that already picked the request up replies on done regardless
	// (buffered — its send never blocks; a raced buf is impossible because
	// queued requests own no buffer yet).
	var res readResult
	select {
	case res = <-req.done:
	case <-v.readStop:
		seg.releaseRead()
		return nil, nil, ReadGone
	}
	if res.st != ReadOK {
		seg.releaseRead()
		return nil, nil, res.st
	}
	rel := func() {
		v.pool.Put(res.buf)
		seg.releaseRead()
	}
	return res.data, rel, ReadOK
}

// UsedBytes is the fallocated on-disk footprint of live segments.
func (v *Volume) UsedBytes() int64 { return v.used.Load() }

// MaxBytes is this volume's configured share.
func (v *Volume) MaxBytes() int64 { return v.p.MaxBytes }

// SegmentBytes exposes the fixed segment size (reclaim sizing).
func (v *Volume) SegmentBytes() int64 { return v.p.SegmentBytes }

// StatsInto merges this volume's counters into m (summed across volumes by
// the tiered store's Stats).
func (v *Volume) StatsInto(m map[string]int64) {
	v.mu.RLock()
	m["segments"] += int64(len(v.segs))
	v.mu.RUnlock()
	m["used_bytes"] += v.used.Load()
	m["max_bytes"] += v.p.MaxBytes
	m["appended_total"] += int64(v.appended.Load()) //nolint:gosec // G115: counters
	m["seals_total"] += int64(v.seals.Load())       //nolint:gosec // G115: counters
	m["checkpoints_total"] += int64(v.ckpts.Load()) //nolint:gosec // G115: counters
	m["enospc_total"] += int64(v.enospc.Load())     //nolint:gosec // G115: counters
	m["reclaims_total"] += int64(v.reclaims.Load()) //nolint:gosec // G115: counters
}

// Close stops the writer (STAGED records are flushed and synced; records
// still in the queue get their OnWritten fired with ok=false — the caller's
// re-admission path runs, no arena ref leaks), stops the readers, and
// closes every segment. The tiered store stops its loops before calling.
func (v *Volume) Close() error {
	if !v.closed.CompareAndSwap(false, true) {
		return nil
	}
	close(v.writerStop)
	<-v.writerDone
	v.failQueuedAppends()
	close(v.readStop)
	v.readerWG.Wait()
	v.closeSegments()
	v.pool.Close()
	return nil
}

// failQueuedAppends drains the append queue after the writer exited, firing
// every abandoned request's callback with ok=false. Without this, the
// demoter's arena reader-refs held across the queue would leak at shutdown
// (the ladder's abandoned-OnWritten MED).
func (v *Volume) failQueuedAppends() {
	for {
		select {
		case req := <-v.reqs:
			req.OnWritten(Loc{}, false)
		default:
			return
		}
	}
}

// CrashForTest abandons everything mid-flight — no seal, no sync, no drain —
// then closes the fds. The next OpenVolume on the same dir exercises real
// recovery. Callers (modeltest) ensure no reads are in flight.
func (v *Volume) CrashForTest() {
	if !v.closed.CompareAndSwap(false, true) {
		return
	}
	v.crashed.Store(true)
	close(v.writerStop)
	<-v.writerDone
	v.failQueuedAppends() // parent-side bookkeeping survives a crash; only the DISK state is abandoned
	close(v.readStop)
	v.readerWG.Wait()
	v.mu.Lock()
	for _, s := range v.segs {
		_ = s.f.Close() // no sync, no seal — crash semantics
	}
	v.segs = map[uint32]*segment{}
	v.active = nil
	v.mu.Unlock()
	v.pool.Close()
}

func (v *Volume) closeSegments() {
	v.mu.Lock()
	defer v.mu.Unlock()
	for _, s := range v.segs {
		if err := s.f.Close(); err != nil {
			v.log.Warn("nvme: segment close", "path", s.path, "err", err)
		}
	}
	v.segs = map[uint32]*segment{}
	v.active = nil
}

func segPath(dir string, id uint32) string {
	return filepath.Join(dir, fmt.Sprintf("seg-%08d.kvbs", id))
}

// openActive creates the fresh active segment (fallocate + fsync + dir sync
// — size metadata is durable before any record lands).
func (v *Volume) openActive() error {
	v.mu.Lock()
	defer v.mu.Unlock()
	id := v.nextID
	v.nextID++
	path := segPath(v.p.Dir, id)
	f, err := v.backend.Open(path, true)
	if err != nil {
		return err
	}
	if err := f.Preallocate(v.p.SegmentBytes); err != nil {
		_ = f.Close()
		return err
	}
	if err := f.Datasync(); err != nil {
		_ = f.Close()
		return err
	}
	if err := SyncDir(v.p.Dir); err != nil {
		_ = f.Close()
		return err
	}
	s := &segment{id: id, f: f, path: path, size: v.p.SegmentBytes}
	v.segs[id] = s
	v.active = s
	v.used.Add(v.p.SegmentBytes)
	return nil
}

// TryRecoverWrites re-arms a write-dead volume once conditions may have
// changed (reclaim freed space, transient FS error passed): if the volume
// is read-only with NO active segment (a failed rotation — the ladder's
// permanently-write-dead MED), it retries opening one. Called from the
// tiered store's maintenance tick.
func (v *Volume) TryRecoverWrites() {
	if v.closed.Load() {
		return
	}
	v.mu.RLock()
	ro, active := v.readOnly, v.active
	v.mu.RUnlock()
	if !ro || active != nil {
		return
	}
	if err := v.openActive(); err != nil {
		return // still failing — stay read-only, retry next tick
	}
	v.clearReadOnly()
	v.log.Info("nvme: volume writable again after failed rotation", "dir", v.p.Dir)
}
