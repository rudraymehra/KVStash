package nvme

import (
	"bytes"
	"context"
	"io"
	"os"
	"testing"

	"github.com/zeebo/xxh3"
)

// Regression tests for confirmed recovery/checkpoint defects. Each pins a
// fixed bug with the exact scenario that exposed it.

// TestCheckpointRoundTripAcrossRestarts pins a data-loss bug: a
// checkpoint-trusted segment recovered with a nil entry table poisoned the
// NEXT checkpoint into dropping every one of its blocks — sealed, fsync'd
// data silently lost on the SECOND restart. All prior tests covered only a
// single restart, so it shipped green.
func TestCheckpointRoundTripAcrossRestarts(t *testing.T) {
	dir := t.TempDir()

	// Generation 1: enough seals to guarantee a checkpoint (every 2 seals),
	// then a clean close.
	v1, _, _ := openTestVolume(t, dir)
	const n = 15 // 60 KiB blocks in 256 KiB segments → ~5 seals → ≥2 ckpts
	sums := make([]uint64, n)
	for i := 0; i < n; i++ {
		_, sums[i] = appendWait(t, v1, i, 60<<10)
	}
	waitForCkpt(t, v1)
	if err := v1.Close(); err != nil {
		t.Fatal(err)
	}

	// Generation 2 (restart #1): the low segments are checkpoint-TRUSTED.
	// Append enough NEW blocks to trigger another checkpoint — the one that
	// used to erase the trusted segments.
	v2, _, ents2 := openTestVolume(t, dir)
	if len(ents2) != n {
		t.Fatalf("restart #1 recovered %d blocks, want %d", len(ents2), n)
	}
	for i := 100; i < 100+9; i++ { // ≥3 more seals → ≥1 new checkpoint
		appendWait(t, v2, i, 60<<10)
	}
	waitForCkpt(t, v2)
	if err := v2.Close(); err != nil {
		t.Fatal(err)
	}

	// Generation 3 (restart #2): EVERY generation-1 block must still be
	// indexed and byte-identical. Before the fix: most vanished here.
	v3, _, ents3 := openTestVolume(t, dir)
	defer func() { _ = v3.Close() }()
	byKey := map[[32]byte]RecoveredEntry{}
	for _, e := range ents3 {
		byKey[e.Key] = e
	}
	for i := 0; i < n; i++ {
		e, ok := byKey[testKey(i)]
		if !ok {
			t.Fatalf("BLOCKER regression: generation-1 block %d lost after the second restart", i)
		}
		mustRead(t, v3, e.Loc, i, sums[i], 60<<10)
	}
	for i := 100; i < 109; i++ {
		if _, ok := byKey[testKey(i)]; !ok {
			t.Fatalf("generation-2 block %d lost", i)
		}
	}
}

// TestCheckpointExcludesDyingCoverage pins a related data-loss bug: a segment
// mid-retire is excluded from the checkpoint's entries, so it must also cap
// maxSealedSegID — otherwise an ABORTED retire leaves a live segment that
// the next recovery "trusts" with zero entries.
func TestCheckpointExcludesDyingCoverage(t *testing.T) {
	dir := t.TempDir()
	v, _, _ := openTestVolume(t, dir)
	sums := make([]uint64, 12)
	for i := range sums {
		_, sums[i] = appendWait(t, v, i, 60<<10) // several sealed segments
	}
	id, entries, ok := v.OldestSealed(0)
	if !ok || len(entries) == 0 {
		t.Fatal("nothing sealed")
	}
	if !v.RetireBegin(id) {
		t.Fatal("retire begin refused")
	}
	// Checkpoint written WHILE the oldest segment is dying…
	if err := v.writeCheckpoint(); err != nil {
		t.Fatal(err)
	}
	// …and the retire then ABORTS (a lease landed): the segment stays live.
	v.RetireAbort(id)
	v.CrashForTest()

	v2, _, ents := openTestVolume(t, dir)
	defer func() { _ = v2.Close() }()
	byKey := map[[32]byte]RecoveredEntry{}
	for _, e := range ents {
		byKey[e.Key] = e
	}
	for i := range sums {
		e, ok := byKey[testKey(i)]
		if !ok {
			t.Fatalf("B1b regression: block %d lost after ckpt-during-aborted-retire + crash", i)
		}
		mustRead(t, v2, e.Loc, i, sums[i], 60<<10)
	}
}

// spillReclaimSetup drives the spill→reclaim shape shared by the adopt
// regression tests: fills v past two seals, captures the oldest sealed
// segment's byte-identical object (what the uploader would PUT), reclaims it,
// and writes the checkpoint that now covers its id with ZERO entries.
func spillReclaimSetup(t *testing.T, v *Volume) (id uint32, entries []footerEntry, obj []byte) {
	t.Helper()
	for i := 0; i < 9; i++ {
		appendWait(t, v, i, 60<<10) // ≥2 sealed segments in 256 KiB geometry
	}
	id, entries, ok := v.OldestSealed(0)
	if !ok || len(entries) == 0 {
		t.Fatal("nothing sealed to spill")
	}
	f, size, err := v.OpenSegmentReadOnly(id)
	if err != nil {
		t.Fatal(err)
	}
	obj = make([]byte, size)
	if _, err := io.ReadFull(f, obj); err != nil {
		t.Fatal(err)
	}
	_ = f.Close()
	v.MarkSpilled(id)
	if !v.RetireBegin(id) {
		t.Fatal("retire begin refused")
	}
	if err := v.RetireFinish(id); err != nil {
		t.Fatal(err)
	}
	// The checkpoint AFTER the reclaim: its maxSealedSegID still covers id
	// (a higher sealed segment survives) but it holds no entries for it.
	if err := v.writeCheckpoint(); err != nil {
		t.Fatal(err)
	}
	return id, entries, obj
}

// assertAdoptedRecovered asserts every pre-reclaim entry of the adopted
// segment came back at the same id and reads verified.
func assertAdoptedRecovered(t *testing.T, v *Volume, id uint32, entries []footerEntry, ents []RecoveredEntry) {
	t.Helper()
	byKey := map[[32]byte]RecoveredEntry{}
	for _, e := range ents {
		byKey[e.Key] = e
	}
	for _, want := range entries {
		got, ok := byKey[want.Key]
		if !ok {
			t.Fatalf("HIGH regression: adopted segment %d lost a block after crash-restart", id)
		}
		if got.Loc.SegmentID != id {
			t.Fatalf("adopted block recovered from segment %d, want %d", got.Loc.SegmentID, id)
		}
		data, rel, st := v.Read(context.Background(), got.Loc, want.NS, want.Key, want.XXH3)
		if st != ReadOK {
			t.Fatalf("adopted block read: %d", st)
		}
		rel()
		_ = data
	}
}

// TestAdoptedSegmentSurvivesCrashBeforeCheckpoint pins the adopt-side hole in
// the checkpoint trust boundary: AdoptSegment legally re-creates a segment
// whose id the newest checkpoint already covers with ZERO entries (the id was
// reclaimed before that checkpoint was written). A crash before any newer
// checkpoint used to recover the adopted segment "trusted" with an empty
// entry table — every restored block silently vanished and the retained S3
// object was orphaned. Recovery must key trust on checkpoint MEMBERSHIP and
// footer-scan a covered-but-absent segment.
func TestAdoptedSegmentSurvivesCrashBeforeCheckpoint(t *testing.T) {
	dir := t.TempDir()
	p := testParams(t, dir)
	// Checkpoints disabled: the crash lands in the exact window where the
	// adopt published (rename + dir sync durable) but no post-adopt
	// checkpoint exists yet — the stale one on disk stays newest.
	p.CkptEverySegs = 0
	v, _, _, err := OpenVolume(p)
	if err != nil {
		t.Fatal(err)
	}
	id, entries, obj := spillReclaimSetup(t, v)

	// Lazy restore brings the segment home; then crash.
	if err := v.AdoptSegment(id, bytes.NewReader(obj)); err != nil {
		t.Fatal(err)
	}
	v.CrashForTest()

	v2, _, ents := openTestVolume(t, dir)
	defer func() { _ = v2.Close() }()
	assertAdoptedRecovered(t, v2, id, entries, ents)
}

// TestAdoptSegmentWritesCheckpoint pins the window-shrink half of the same
// fix: a successful adopt immediately re-checkpoints, so the adopted entries
// are in the newest checkpoint and the common recovery path stays
// checkpoint-trusted (no footer scan needed to keep them).
func TestAdoptSegmentWritesCheckpoint(t *testing.T) {
	dir := t.TempDir()
	v, _, _ := openTestVolume(t, dir) // CkptEverySegs=2: post-adopt ckpt enabled
	id, entries, obj := spillReclaimSetup(t, v)

	if err := v.AdoptSegment(id, bytes.NewReader(obj)); err != nil {
		t.Fatal(err)
	}
	ckEntries, ckMax, _, ok := loadNewestCkpt(dir, func(string, ...any) {})
	if !ok {
		t.Fatal("no checkpoint after adopt")
	}
	adopted := 0
	for _, ce := range ckEntries {
		if ce.SegID == id {
			adopted++
		}
	}
	if adopted != len(entries) || ckMax < id {
		t.Fatalf("post-adopt checkpoint holds %d/%d entries for segment %d (maxSealed %d)",
			adopted, len(entries), id, ckMax)
	}
	v.CrashForTest()

	v2, _, ents := openTestVolume(t, dir)
	defer func() { _ = v2.Close() }()
	assertAdoptedRecovered(t, v2, id, entries, ents)
}

// TestSameSegmentLatestWins pins a fixed recovery bug: two generations of one
// key inside the SAME segment (delete + re-put before rotation) must
// recover to the LATER offset — the old code kept the first footer entry
// and served superseded bytes.
func TestSameSegmentLatestWins(t *testing.T) {
	dir := t.TempDir()
	v, _, _ := openTestVolume(t, dir)
	// Two records under the SAME key, same segment, different content.
	oldLoc, _ := appendWait(t, v, 7, 10<<10)
	p := testPayload(7777, 10<<10) // different bytes for the same key
	done := make(chan Loc, 1)
	if !v.Append(AppendReq{
		NS: 1, Key: testKey(7), XXH3: xxh3Hash(p), Data: p,
		OnWritten: func(loc Loc, ok bool) {
			if !ok {
				t.Error("second append failed")
			}
			done <- loc
		},
	}) {
		t.Fatal("append refused")
	}
	newLoc := <-done
	if newLoc.SegmentID != oldLoc.SegmentID {
		t.Fatalf("fixture: generations landed in different segments (%d vs %d)", oldLoc.SegmentID, newLoc.SegmentID)
	}
	v.CrashForTest()

	v2, _, ents := openTestVolume(t, dir)
	defer func() { _ = v2.Close() }()
	var got *RecoveredEntry
	for i := range ents {
		if ents[i].Key == testKey(7) {
			if got != nil {
				t.Fatal("key recovered twice")
			}
			got = &ents[i]
		}
	}
	if got == nil {
		t.Fatal("key lost")
	}
	if got.Loc.Offset != newLoc.Offset {
		t.Fatalf("HIGH regression: recovery kept offset %d (older), want %d (later)", got.Loc.Offset, newLoc.Offset)
	}
	data, rel, st := v2.Read(context.Background(), got.Loc, 1, testKey(7), xxh3Hash(p))
	if st != ReadOK {
		t.Fatalf("later-generation read: %d", st)
	}
	defer rel()
	if len(data) != len(p) || data[0] != p[0] {
		t.Fatal("later-generation bytes wrong")
	}
}

// TestGeometryChangeSurvivesRetune pins a fixed recovery bug: retuning
// nvme_segment_bytes must NOT delete existing sealed segments — geometry is
// per-file, only genuinely partial creates are dropped.
func TestGeometryChangeSurvivesRetune(t *testing.T) {
	dir := t.TempDir()
	v, _, _ := openTestVolume(t, dir) // 256 KiB segments
	sums := make([]uint64, 8)
	for i := range sums {
		_, sums[i] = appendWait(t, v, i, 60<<10)
	}
	if err := v.Close(); err != nil {
		t.Fatal(err)
	}

	p := testParams(t, dir)
	p.SegmentBytes = 512 << 10 // the operator doubles the segment size
	v2, _, ents, err := OpenVolume(p)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = v2.Close() }()
	byKey := map[[32]byte]RecoveredEntry{}
	for _, e := range ents {
		byKey[e.Key] = e
	}
	for i := range sums {
		e, ok := byKey[testKey(i)]
		if !ok {
			t.Fatalf("HIGH regression: block %d wiped by a segment-size retune", i)
		}
		mustRead(t, v2, e.Loc, i, sums[i], 60<<10)
	}
}

// TestJunkFilenamesIgnored pins a fixed recovery bug: non-canonical names that
// Sscanf would happily parse (seg-0.kvbs aliasing seg-00000000.kvbs) must
// be ignored, not double-counted.
func TestJunkFilenamesIgnored(t *testing.T) {
	dir := t.TempDir()
	v, _, _ := openTestVolume(t, dir)
	appendWait(t, v, 0, 30<<10)
	if err := v.Close(); err != nil {
		t.Fatal(err)
	}
	for _, junk := range []string{"seg-0.kvbs", "seg-007.kvbs", "seg-junk.kvbs", "ckpt-1.kvbi.tmp"} {
		if err := os.WriteFile(dir+"/"+junk, []byte("garbage"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	v2, _, ents := openTestVolume(t, dir)
	defer func() { _ = v2.Close() }()
	if len(ents) != 1 {
		t.Fatalf("junk filenames changed recovery: %d entries, want 1", len(ents))
	}
	// used must equal exactly one real segment (the recovered one) plus the
	// fresh active — no double count from the alias.
	if got := v2.UsedBytes(); got != 2*testParams(t, dir).SegmentBytes {
		t.Fatalf("MED regression: used=%d, want exactly 2 segments' worth (%d)", got, 2*testParams(t, dir).SegmentBytes)
	}
	if _, err := os.Stat(dir + "/ckpt-1.kvbi.tmp"); !os.IsNotExist(err) {
		t.Fatal("orphaned .tmp checkpoint not pruned at open")
	}
}

// TestCloseFiresQueuedAppendCallbacks pins a fixed shutdown leak: appends still in
// the queue when Close runs must get their OnWritten(ok=false) — the
// demoter's arena refs leak otherwise.
func TestCloseFiresQueuedAppendCallbacks(t *testing.T) {
	v, _, _ := openTestVolume(t, t.TempDir())
	fired := make(chan bool, 256)
	accepted := 0
	for i := 0; i < 200; i++ {
		if v.Append(AppendReq{
			NS: 1, Key: testKey(i), XXH3: 1, Data: testPayload(i, 30<<10),
			OnWritten: func(_ Loc, ok bool) { fired <- ok },
		}) {
			accepted++
		}
	}
	if err := v.Close(); err != nil {
		t.Fatal(err)
	}
	if got := len(fired); got != accepted {
		t.Fatalf("MED regression: %d/%d accepted appends got a callback across Close", got, accepted)
	}
}

func xxh3Hash(b []byte) uint64 { return xxh3.Hash(b) }
