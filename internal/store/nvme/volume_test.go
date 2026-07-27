package nvme

import (
	"bytes"
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/zeebo/xxh3"
	"go.uber.org/goleak"
)

func TestMain(m *testing.M) {
	goleak.VerifyTestMain(m)
}

// testParams: small blobs + small segments so a handful of appends exercise
// rotation, sealing, and checkpoints.
func testParams(t *testing.T, dir string) VolumeParams {
	t.Helper()
	return VolumeParams{
		Dir:            dir,
		SegmentBytes:   256 << 10,
		MaxBytes:       8 << 20,
		SyncEveryBytes: 64 << 10,
		ReadWorkers:    2,
		CkptEverySegs:  2,
		MaxBlobLen:     64 << 10,
	}
}

func openTestVolume(t *testing.T, dir string) (*Volume, *RecoveryReport, []RecoveredEntry) {
	t.Helper()
	v, rep, ents, err := OpenVolume(testParams(t, dir))
	if err != nil {
		t.Fatal(err)
	}
	return v, rep, ents
}

func testKey(i int) (k [32]byte) {
	copy(k[:], fmt.Sprintf("key-%06d", i))
	return k
}

func testPayload(i, n int) []byte {
	p := make([]byte, n)
	for j := range p {
		p[j] = byte(i + j) //nolint:gosec // G115: deliberate wrap — deterministic test pattern
	}
	return p
}

// appendWait pushes one record (namespace 1) through the writer and waits
// for its ack.
func appendWait(t *testing.T, v *Volume, i, n int) (Loc, uint64) {
	t.Helper()
	p := testPayload(i, n)
	sum := xxh3.Hash(p)
	done := make(chan Loc, 1)
	ok := v.Append(AppendReq{
		NS: 1, Key: testKey(i), XXH3: sum, Data: p,
		OnWritten: func(loc Loc, wok bool) {
			if !wok {
				t.Errorf("append %d failed", i)
			}
			done <- loc
		},
	})
	if !ok {
		t.Fatalf("append %d refused (queue full/read-only)", i)
	}
	select {
	case loc := <-done:
		return loc, sum
	case <-time.After(10 * time.Second):
		t.Fatalf("append %d: OnWritten never fired", i)
		return Loc{}, 0
	}
}

func mustRead(t *testing.T, v *Volume, loc Loc, i int, sum uint64, wantLen int) {
	t.Helper()
	data, rel, st := v.Read(context.Background(), loc, 1, testKey(i), sum)
	if st != ReadOK {
		t.Fatalf("read %d: status %d", i, st)
	}
	defer rel()
	if len(data) != wantLen || !bytes.Equal(data, testPayload(i, wantLen)) {
		t.Fatalf("read %d: payload mismatch (%d bytes)", i, len(data))
	}
}

func TestVolumeAppendReadRoundTrip(t *testing.T) {
	v, rep, ents := openTestVolume(t, t.TempDir())
	defer func() { _ = v.Close() }()
	if rep.SegmentsScanned != 0 || len(ents) != 0 {
		t.Fatalf("fresh dir recovered something: %+v", rep)
	}

	sizes := []int{0, 1, 4040, 10 << 10, 60 << 10} // empty block legal
	locs := make([]Loc, len(sizes))
	sums := make([]uint64, len(sizes))
	for i, n := range sizes {
		locs[i], sums[i] = appendWait(t, v, i, n)
		if locs[i].Offset%recordAlign != 0 {
			t.Fatalf("record %d at unaligned offset %d", i, locs[i].Offset)
		}
	}
	for i, n := range sizes {
		mustRead(t, v, locs[i], i, sums[i], n)
	}

	// Wrong expectations must never serve bytes.
	if _, _, st := v.Read(context.Background(), locs[1], 2, testKey(1), sums[1]); st != ReadCorrupt {
		t.Fatalf("cross-namespace read: %d, want corrupt", st)
	}
	if _, _, st := v.Read(context.Background(), locs[1], 1, testKey(1), sums[1]^1); st != ReadCorrupt {
		t.Fatalf("wrong-sum read: %d, want corrupt", st)
	}
	if _, _, st := v.Read(context.Background(), Loc{SegmentID: 99, Offset: 0, Len: 8}, 1, testKey(1), 1); st != ReadGone {
		t.Fatalf("unknown segment read: %d, want gone", st)
	}
}

func TestVolumeRotationAndSeal(t *testing.T) {
	v, _, _ := openTestVolume(t, t.TempDir())
	defer func() { _ = v.Close() }()

	// 60 KiB payloads in 256 KiB segments: rotation every ~3 records.
	var lastSeg uint32
	for i := 0; i < 12; i++ {
		loc, _ := appendWait(t, v, i, 60<<10)
		lastSeg = loc.SegmentID
	}
	if lastSeg == 0 {
		t.Fatal("no rotation happened")
	}
	if v.seals.Load() == 0 {
		t.Fatal("no seal recorded")
	}
	id, entries, ok := v.OldestSealed(0)
	if !ok || id != 0 || len(entries) == 0 {
		t.Fatalf("OldestSealed: id=%d ok=%v entries=%d", id, ok, len(entries))
	}
	// Sealed records stay readable.
	e := entries[0]
	data, rel, st := v.Read(context.Background(), Loc{SegmentID: id, Offset: e.Off, Len: e.Len}, e.NS, e.Key, e.XXH3)
	if st != ReadOK {
		t.Fatalf("sealed read: %d", st)
	}
	rel()
	_ = data
}

func TestVolumeCleanReopen(t *testing.T) {
	dir := t.TempDir()
	v, _, _ := openTestVolume(t, dir)
	locs := make([]Loc, 8)
	sums := make([]uint64, 8)
	for i := range locs {
		locs[i], sums[i] = appendWait(t, v, i, 30<<10)
	}
	if err := v.Close(); err != nil {
		t.Fatal(err)
	}

	v2, rep, ents := openTestVolume(t, dir)
	defer func() { _ = v2.Close() }()
	if len(ents) != 8 {
		t.Fatalf("clean reopen recovered %d blocks, want 8 (report %+v)", len(ents), rep)
	}
	byKey := map[[32]byte]RecoveredEntry{}
	for _, e := range ents {
		byKey[e.Key] = e
	}
	for i := range locs {
		e, ok := byKey[testKey(i)]
		if !ok {
			t.Fatalf("block %d lost on clean reopen", i)
		}
		mustRead(t, v2, e.Loc, i, sums[i], 30<<10)
	}
	if rep.BytesTruncated != 0 {
		t.Fatalf("clean reopen truncated %d bytes", rep.BytesTruncated)
	}
}

func TestVolumeCrashReopen(t *testing.T) {
	dir := t.TempDir()
	v, _, _ := openTestVolume(t, dir)
	n := 10
	sums := make([]uint64, n)
	for i := 0; i < n; i++ {
		_, sums[i] = appendWait(t, v, i, 20<<10)
	}
	v.CrashForTest() // no seal, no final sync

	v2, rep, ents := openTestVolume(t, dir)
	defer func() { _ = v2.Close() }()
	// Every acked record was pwritten before OnWritten fired; the same-kernel
	// page cache makes them all visible. All 10 must come back verified.
	if len(ents) != n {
		t.Fatalf("crash reopen recovered %d blocks, want %d (report %+v)", len(ents), n, rep)
	}
	for _, e := range ents {
		data, rel, st := v2.Read(context.Background(), e.Loc, e.NS, e.Key, e.XXH3)
		if st != ReadOK {
			t.Fatalf("recovered block read: %d", st)
		}
		rel()
		_ = data
	}
}

// gateBackend wraps the default backend; while armed, every ReadAt parks on
// gate (announcing itself on entered) — a stand-in for an NVMe controller
// stall. Writes and recovery reads pass through untouched (arm after setup).
type gateBackend struct {
	inner   IOBackend
	armed   atomic.Bool
	gate    chan struct{}
	entered chan struct{}
}

func newGateBackend() *gateBackend {
	return &gateBackend{
		inner:   DefaultBackend(),
		gate:    make(chan struct{}),
		entered: make(chan struct{}, 16),
	}
}

func (b *gateBackend) Open(path string, forWrite bool) (File, error) {
	f, err := b.inner.Open(path, forWrite)
	if err != nil {
		return nil, err
	}
	return &gateFile{File: f, b: b}, nil
}

type gateFile struct {
	File
	b *gateBackend
}

func (f *gateFile) ReadAt(p []byte, off int64) error {
	if f.b.armed.Load() {
		f.b.entered <- struct{}{}
		<-f.b.gate
	}
	return f.File.ReadAt(p, off)
}

// TestReadCancelReleasesHoldAndBuffer pins the ctx leg of Volume.Read: with
// the single worker wedged in a device stall, (a) a QUEUED read whose ctx
// fires returns immediately instead of blocking behind the stall, and (b)
// the stalled read's own cancellation abandons it to the worker, which —
// once the device answers — must put the pooled buffer back and drop BOTH
// segment read-holds (exactly-one-owner via the claimed flag). Before the
// ctx plumb both callers blocked unboundedly and pinned their holds, which
// also wedged RetireFinish's drain loop.
func TestReadCancelReleasesHoldAndBuffer(t *testing.T) {
	dir := t.TempDir()
	p := testParams(t, dir)
	p.ReadWorkers = 1
	gb := newGateBackend()
	p.Backend = gb
	v, _, _, err := OpenVolume(p)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = v.Close() }()

	locA, sumA := appendWait(t, v, 1, 8<<10)
	locB, sumB := appendWait(t, v, 2, 8<<10)
	gb.armed.Store(true)

	// Read A occupies the only worker and parks inside the device stall.
	ctxA, cancelA := context.WithCancel(context.Background())
	defer cancelA()
	aDone := make(chan ReadStatus, 1)
	go func() {
		_, _, st := v.Read(ctxA, locA, 1, testKey(1), sumA)
		aDone <- st
	}()
	<-gb.entered // the worker is now wedged mid-pread

	// Read B sits QUEUED behind the stall; its cancellation must answer NOW.
	ctxB, cancelB := context.WithCancel(context.Background())
	bDone := make(chan ReadStatus, 1)
	go func() {
		_, _, st := v.Read(ctxB, locB, 1, testKey(2), sumB)
		bDone <- st
	}()
	cancelB()
	select {
	case st := <-bDone:
		if st != ReadGone {
			t.Fatalf("cancelled queued read: %d, want ReadGone", st)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("cancelled queued read still blocked behind the device stall")
	}

	// Cancel A mid-pread: the caller unblocks; ownership of the buffer and
	// the read-hold moves to the worker.
	cancelA()
	select {
	case st := <-aDone:
		if st != ReadGone {
			t.Fatalf("cancelled in-flight read: %d, want ReadGone", st)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("cancelled in-flight read never returned")
	}

	// Un-stall the device: the worker completes A (cleans up as the claim
	// loser) and drains B's queued request (releasing its hold unread).
	gb.armed.Store(false)
	close(gb.gate)
	v.mu.RLock()
	segA, segB := v.segs[locA.SegmentID], v.segs[locB.SegmentID]
	v.mu.RUnlock()
	deadline := time.Now().Add(5 * time.Second)
	for segA.reads.Load() != 0 || segB.reads.Load() != 0 {
		if time.Now().After(deadline) {
			t.Fatalf("read-holds leaked after cancellation: segA=%d segB=%d",
				segA.reads.Load(), segB.reads.Load())
		}
		time.Sleep(time.Millisecond)
	}

	// The tier still serves both records — nothing was corrupted or stuck.
	mustRead(t, v, locA, 1, sumA, 8<<10)
	mustRead(t, v, locB, 2, sumB, 8<<10)
}

func TestVolumeDualResidencyLatestWins(t *testing.T) {
	// The same key appended twice (demote → promote → re-demote shape):
	// recovery must surface exactly one entry, the later segID.
	dir := t.TempDir()
	v, _, _ := openTestVolume(t, dir)

	// Fill past one rotation so the two copies land in different segments.
	appendWait(t, v, 7, 60<<10)
	for i := 100; i < 104; i++ {
		appendWait(t, v, i, 60<<10)
	}
	loc2, _ := appendWait(t, v, 7, 60<<10) // second copy of key 7
	if err := v.Close(); err != nil {
		t.Fatal(err)
	}

	v2, _, ents := openTestVolume(t, dir)
	defer func() { _ = v2.Close() }()
	var got []RecoveredEntry
	for _, e := range ents {
		if e.Key == testKey(7) {
			got = append(got, e)
		}
	}
	if len(got) != 1 {
		t.Fatalf("key 7 recovered %d times, want 1", len(got))
	}
	if got[0].Loc.SegmentID != loc2.SegmentID {
		t.Fatalf("recovery kept segID %d, want the later %d", got[0].Loc.SegmentID, loc2.SegmentID)
	}
}

// TestAppendCloseNeverAbandonsCallbacks: every Append that returned true
// must get its OnWritten fired by the time Close returns — an accepted
// request slipping past failQueuedAppends leaks the demoter's arena
// reader-ref forever. The lock now enforces the ordering the prose contract
// ("the tiered store stops its loops first") used to merely describe.
//
// Two phases: a DETERMINISTIC one (appendDelayForTest parks one Append
// inside its check→send window while Close runs to completion — without
// appendMu that Append sends into the already-drained queue and returns
// true with a callback nobody will ever fire), then the original stress
// loop as a belt over schedules the hook does not model.
func TestAppendCloseNeverAbandonsCallbacks(t *testing.T) {
	{
		v, _, _ := openTestVolume(t, t.TempDir())
		appendDelayForTest.Store(int64(100 * time.Millisecond))
		var accepted, answered atomic.Int64
		done := make(chan struct{})
		p := testPayload(0, 512)
		go func() {
			defer close(done)
			ok := v.Append(AppendReq{
				NS: 1, Key: testKey(1), XXH3: xxh3.Hash(p), Data: p,
				OnWritten: func(Loc, bool) { answered.Add(1) },
			})
			if ok {
				accepted.Add(1)
			}
		}()
		time.Sleep(20 * time.Millisecond) // the appender is parked inside the window
		if err := v.Close(); err != nil {
			t.Fatal(err)
		}
		<-done
		appendDelayForTest.Store(0)
		if a, ans := accepted.Load(), answered.Load(); ans != a {
			t.Fatalf("deterministic window: %d accepted but %d callbacks — Close ran through Append's check/send gap", a, ans)
		}
	}
	for iter := 0; iter < 40; iter++ {
		v, _, _ := openTestVolume(t, t.TempDir())
		var accepted, answered atomic.Int64
		stop := make(chan struct{})
		var wg sync.WaitGroup
		for g := 0; g < 8; g++ {
			wg.Add(1)
			go func(g int) {
				defer wg.Done()
				p := testPayload(g, 512)
				for i := 0; ; i++ {
					select {
					case <-stop:
						return
					default:
					}
					ok := v.Append(AppendReq{
						NS: 1, Key: testKey(g*1_000_000 + i), XXH3: xxh3.Hash(p), Data: p,
						OnWritten: func(Loc, bool) { answered.Add(1) },
					})
					if ok {
						accepted.Add(1)
					}
				}
			}(g)
		}
		time.Sleep(2 * time.Millisecond)
		if err := v.Close(); err != nil {
			t.Fatal(err)
		}
		close(stop)
		wg.Wait()
		if a, ans := accepted.Load(), answered.Load(); ans != a {
			t.Fatalf("iter %d: %d accepted appends but %d callbacks — %d arena refs leaked past Close",
				iter, a, ans, a-ans)
		}
	}
}

// TestCheckpointDoesNotStallAppends: the checkpoint's O(total blocks) file
// I/O must run OFF the writer goroutine — inline it stalls every append
// (and drops demotions once the 128-deep queue fills) for the checkpoint's
// full duration on every cadence tick.
func TestCheckpointDoesNotStallAppends(t *testing.T) {
	ckptIODelayForTest.Store(int64(600 * time.Millisecond))
	defer ckptIODelayForTest.Store(0)

	v, _, _ := openTestVolume(t, t.TempDir()) // CkptEverySegs: 2
	defer func() { _ = v.Close() }()

	worst := time.Duration(0)
	for i := 0; i < 12; i++ { // ~3 records/segment → ≥4 seals → ≥1 checkpoint due
		start := time.Now()
		appendWait(t, v, i, 64<<10)
		if el := time.Since(start); el > worst {
			worst = el
		}
	}
	if worst > 300*time.Millisecond {
		t.Fatalf("append stalled %v behind checkpoint I/O — the checkpoint is running on the writer goroutine", worst)
	}
	waitForCkpt(t, v)
}

// waitForCkpt blocks until at least one checkpoint completed — the cadence
// checkpoint's file I/O is asynchronous, so fixtures must wait, not probe.
func waitForCkpt(t *testing.T, v *Volume) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for v.ckpts.Load() == 0 {
		if time.Now().After(deadline) {
			t.Fatal("no checkpoint ever completed")
		}
		time.Sleep(2 * time.Millisecond)
	}
}
