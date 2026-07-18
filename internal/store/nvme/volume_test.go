package nvme

import (
	"bytes"
	"fmt"
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
	data, rel, st := v.Read(loc, 1, testKey(i), sum)
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
	if _, _, st := v.Read(locs[1], 2, testKey(1), sums[1]); st != ReadCorrupt {
		t.Fatalf("cross-namespace read: %d, want corrupt", st)
	}
	if _, _, st := v.Read(locs[1], 1, testKey(1), sums[1]^1); st != ReadCorrupt {
		t.Fatalf("wrong-sum read: %d, want corrupt", st)
	}
	if _, _, st := v.Read(Loc{SegmentID: 99, Offset: 0, Len: 8}, 1, testKey(1), 1); st != ReadGone {
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
	id, entries, ok := v.OldestSealed()
	if !ok || id != 0 || len(entries) == 0 {
		t.Fatalf("OldestSealed: id=%d ok=%v entries=%d", id, ok, len(entries))
	}
	// Sealed records stay readable.
	e := entries[0]
	data, rel, st := v.Read(Loc{SegmentID: id, Offset: e.Off, Len: e.Len}, e.NS, e.Key, e.XXH3)
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
		data, rel, st := v2.Read(e.Loc, e.NS, e.Key, e.XXH3)
		if st != ReadOK {
			t.Fatalf("recovered block read: %d", st)
		}
		rel()
		_ = data
	}
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
