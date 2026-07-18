package nvme

import (
	"bytes"
	"os"
	"testing"
)

// The torn-write matrix (SPEC-3 §2): every corruption shape recovery must
// survive without ever surfacing a corrupt block. Each case builds a real
// volume, crashes it, damages bytes with raw buffered writes, reopens, and
// asserts exactly what survived.
//
// Layout used by buildTornFixture: 5 × 20 KiB records appended, NO seal
// (CrashForTest) — all five live in the unsealed tail of segment 0, each
// record spanning 6 aligned units (24 KiB).

func buildTornFixture(t *testing.T) (dir string, locs []Loc, sums []uint64) {
	t.Helper()
	dir = t.TempDir()
	v, _, _ := openTestVolume(t, dir)
	locs = make([]Loc, 5)
	sums = make([]uint64, 5)
	for i := range locs {
		locs[i], sums[i] = appendWait(t, v, i, 20<<10)
	}
	v.CrashForTest()
	return dir, locs, sums
}

// damage flips bytes at off in segment 0 with plain buffered IO.
func damage(t *testing.T, dir string, off int64, b []byte) {
	t.Helper()
	f, err := os.OpenFile(segPath(dir, 0), os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = f.Close() }()
	if _, err := f.WriteAt(b, off); err != nil {
		t.Fatal(err)
	}
}

// reopenAndIndex recovers the dir and returns key→entry.
func reopenAndIndex(t *testing.T, dir string) (*Volume, *RecoveryReport, map[[32]byte]RecoveredEntry) {
	t.Helper()
	v, rep, ents := openTestVolume(t, dir)
	byKey := map[[32]byte]RecoveredEntry{}
	for _, e := range ents {
		byKey[e.Key] = e
	}
	return v, rep, byKey
}

// assertRecovered checks that exactly the records in `want` survived, all
// byte-identical, and nothing else.
func assertRecovered(t *testing.T, v *Volume, byKey map[[32]byte]RecoveredEntry, sums []uint64, want []int) {
	t.Helper()
	wantSet := map[int]bool{}
	for _, i := range want {
		wantSet[i] = true
	}
	for i := range sums {
		e, ok := byKey[testKey(i)]
		if wantSet[i] != ok {
			t.Fatalf("record %d: recovered=%v want=%v", i, ok, wantSet[i])
		}
		if ok {
			mustRead(t, v, e.Loc, i, sums[i], 20<<10)
		}
	}
	if len(byKey) != len(want) {
		t.Fatalf("recovered %d records, want %d", len(byKey), len(want))
	}
}

func TestTornWriteMatrix(t *testing.T) {
	span := int64(recordSpan(20 << 10)) //nolint:gosec // G115: 24 KiB per record

	cases := []struct {
		name     string
		hurt     func(t *testing.T, dir string, locs []Loc)
		want     []int // record indices that must survive
		wantTorn bool  // BytesTruncated must be > 0 (damage LOOKS torn, not clean-end)
	}{
		{
			name: "clean tail — everything survives",
			hurt: func(t *testing.T, dir string, locs []Loc) {},
			want: []int{0, 1, 2, 3, 4},
		},
		{
			name: "torn header of record 3 — truncate there",
			hurt: func(t *testing.T, dir string, locs []Loc) {
				damage(t, dir, int64(locs[3].Offset), []byte{0xDE, 0xAD, 0xBE, 0xEF})
			},
			want:     []int{0, 1, 2},
			wantTorn: true,
		},
		{
			name: "torn payload byte of record 2 — xxh3 catches, truncate there",
			hurt: func(t *testing.T, dir string, locs []Loc) {
				damage(t, dir, int64(locs[2].Offset)+recordHdrSize+777, []byte{0xFF})
			},
			want:     []int{0, 1},
			wantTorn: true,
		},
		{
			// A ZEROED header is indistinguishable from the clean end of
			// written data (fallocated zeros) — recovery stops there without
			// counting a tear. Records 2–4 are intact on disk but must stay
			// dead: truncate-at-first-bad, no scan-past.
			name: "valid record AFTER a zeroed one is NOT resurrected",
			hurt: func(t *testing.T, dir string, locs []Loc) {
				damage(t, dir, int64(locs[1].Offset), make([]byte, recordHdrSize))
			},
			want: []int{0},
		},
		{
			name: "hostile length in record 0's header — reject before any read",
			hurt: func(t *testing.T, dir string, locs []Loc) {
				damage(t, dir, int64(locs[0].Offset)+40, []byte{0xFF, 0xFF, 0xFF, 0x7F})
			},
			want:     []int{},
			wantTorn: true,
		},
		{
			name: "zeroed first unit of record 0 — clean-looking end, nothing served",
			hurt: func(t *testing.T, dir string, locs []Loc) {
				damage(t, dir, int64(locs[0].Offset), make([]byte, recordAlign))
			},
			want: []int{},
		},
		{
			// Damage confined to record 2's PADDING zone plus record 3's
			// header: padding carries no data (xxh3 covers the payload), so
			// record 2 legitimately survives; the tear truncates at 3.
			name: "pad-zone damage spares its record, torn next header truncates",
			hurt: func(t *testing.T, dir string, locs []Loc) {
				garbage := bytes.Repeat([]byte{0xAA}, 200) // non-zero: must read as TORN, not clean end
				damage(t, dir, int64(locs[2].Offset)+span-100, garbage)
			},
			want:     []int{0, 1, 2},
			wantTorn: true,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			dir, locs, sums := buildTornFixture(t)
			c.hurt(t, dir, locs)
			v, rep, byKey := reopenAndIndex(t, dir)
			defer func() { _ = v.Close() }()
			assertRecovered(t, v, byKey, sums, c.want)
			if c.wantTorn && rep.BytesTruncated == 0 {
				t.Fatal("visibly torn damage but BytesTruncated == 0")
			}
			if !c.wantTorn && rep.BytesTruncated != 0 {
				t.Fatalf("clean-looking case counted %d truncated bytes", rep.BytesTruncated)
			}
		})
	}
}

func TestSealedFooterDamageFallsBackToScan(t *testing.T) {
	// Seal segment 0 by overflowing it, then flip a footer byte: the seal
	// CRC fails, recovery treats it as an unsealed tail, and the records
	// come back via the forward scan + re-seal.
	dir := t.TempDir()
	v, _, _ := openTestVolume(t, dir)
	sums := make([]uint64, 6)
	locs := make([]Loc, 6)
	for i := range locs {
		locs[i], sums[i] = appendWait(t, v, i, 60<<10) // rotates after ~3
	}
	if v.seals.Load() == 0 {
		t.Fatal("fixture never sealed")
	}
	if err := v.Close(); err != nil {
		t.Fatal(err)
	}

	// Flip one byte inside segment 0's trailer CRC.
	damage(t, dir, testParams(t, dir).SegmentBytes-4, []byte{0xFF})

	v2, _, ents := openTestVolume(t, dir)
	defer func() { _ = v2.Close() }()
	byKey := map[[32]byte]RecoveredEntry{}
	for _, e := range ents {
		byKey[e.Key] = e
	}
	for i := range locs {
		e, ok := byKey[testKey(i)]
		if !ok {
			t.Fatalf("record %d lost after footer damage (scan fallback failed)", i)
		}
		mustRead(t, v2, e.Loc, i, sums[i], 60<<10)
	}
}

func TestSealedRecordBodyDamageSelfHeals(t *testing.T) {
	// A CRC-valid footer with a rotted record body: recovery indexes it
	// (the footer vouched), but the READ path's xxh3 verify refuses to
	// serve it — never a corrupt byte out.
	dir := t.TempDir()
	v, _, _ := openTestVolume(t, dir)
	for i := 0; i < 6; i++ {
		appendWait(t, v, i, 60<<10)
	}
	if err := v.Close(); err != nil {
		t.Fatal(err)
	}

	v2, _, ents := openTestVolume(t, dir)
	defer func() { _ = v2.Close() }()
	var seg0 *RecoveredEntry
	for i := range ents {
		if ents[i].Loc.SegmentID == 0 {
			seg0 = &ents[i]
			break
		}
	}
	if seg0 == nil {
		t.Fatal("no record recovered in sealed segment 0")
	}
	// Rot one payload byte behind recovery's back... via a second damage
	// pass + reopen (the fds are open — damage the NEXT incarnation).
	if err := v2.Close(); err != nil {
		t.Fatal(err)
	}
	damage(t, dir, int64(seg0.Loc.Offset)+recordHdrSize+5, []byte{0xEE})

	v3, _, ents3 := openTestVolume(t, dir)
	defer func() { _ = v3.Close() }()
	for _, e := range ents3 {
		if e.Key != seg0.Key {
			continue
		}
		if _, _, st := v3.Read(e.Loc, e.NS, e.Key, e.XXH3); st != ReadCorrupt {
			t.Fatalf("rotted sealed record served: status %d, want ReadCorrupt", st)
		}
		return
	}
	t.Fatal("rotted record vanished from a CRC-valid footer (unexpected)")
}

func TestCheckpointInterplay(t *testing.T) {
	// Build enough seals for a checkpoint, then damage segment 0's footer:
	// the ckpt (which covers it) must carry recovery — proving the
	// trusted-sealed fast path actually reads from the checkpoint.
	dir := t.TempDir()
	v, _, _ := openTestVolume(t, dir)
	n := 15 // ≥4 rotations at 3 records/segment → ≥1 checkpoint (every 2 seals)
	sums := make([]uint64, n)
	for i := 0; i < n; i++ {
		_, sums[i] = appendWait(t, v, i, 60<<10)
	}
	if v.ckpts.Load() == 0 {
		t.Fatal("fixture wrote no checkpoint")
	}
	if err := v.Close(); err != nil {
		t.Fatal(err)
	}

	damage(t, dir, testParams(t, dir).SegmentBytes-4, []byte{0xAA}) // footer CRC of seg 0

	v2, _, ents := openTestVolume(t, dir)
	byKey := map[[32]byte]RecoveredEntry{}
	for _, e := range ents {
		byKey[e.Key] = e
	}
	found := 0
	for i := 0; i < n; i++ {
		if e, ok := byKey[testKey(i)]; ok {
			mustRead(t, v2, e.Loc, i, sums[i], 60<<10)
			found++
		}
	}
	if found != n {
		t.Fatalf("ckpt-backed recovery found %d/%d blocks", found, n)
	}
	if err := v2.Close(); err != nil {
		t.Fatal(err)
	}

	// Now ALSO delete every checkpoint: recovery falls back to footer scans;
	// seg 0's damaged footer sends it through the tail scan instead — the
	// blocks must STILL come back (slower path, same truth).
	for _, s := range listCkptSeqs(dir) {
		if err := os.Remove(ckptPath(dir, s)); err != nil {
			t.Fatal(err)
		}
	}
	v3, _, ents3 := openTestVolume(t, dir)
	defer func() { _ = v3.Close() }()
	byKey3 := map[[32]byte]RecoveredEntry{}
	for _, e := range ents3 {
		byKey3[e.Key] = e
	}
	for i := 0; i < n; i++ {
		e, ok := byKey3[testKey(i)]
		if !ok {
			t.Fatalf("record %d lost without checkpoints", i)
		}
		mustRead(t, v3, e.Loc, i, sums[i], 60<<10)
	}
}

func TestCheckpointCorruptFallsBack(t *testing.T) {
	dir := t.TempDir()
	v, _, _ := openTestVolume(t, dir)
	n := 15
	sums := make([]uint64, n)
	for i := 0; i < n; i++ {
		_, sums[i] = appendWait(t, v, i, 60<<10)
	}
	if v.ckpts.Load() == 0 {
		t.Fatal("fixture wrote no checkpoint")
	}
	if err := v.Close(); err != nil {
		t.Fatal(err)
	}

	// Flip a byte in every checkpoint: CRC fails, recovery must fall back
	// to pure footer scans and lose nothing.
	for _, s := range listCkptSeqs(dir) {
		f, err := os.OpenFile(ckptPath(dir, s), os.O_WRONLY, 0)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := f.WriteAt([]byte{0x5A}, 30); err != nil {
			t.Fatal(err)
		}
		_ = f.Close()
	}

	v2, _, ents := openTestVolume(t, dir)
	defer func() { _ = v2.Close() }()
	byKey := map[[32]byte]RecoveredEntry{}
	for _, e := range ents {
		byKey[e.Key] = e
	}
	for i := 0; i < n; i++ {
		e, ok := byKey[testKey(i)]
		if !ok {
			t.Fatalf("record %d lost with corrupt checkpoints", i)
		}
		mustRead(t, v2, e.Loc, i, sums[i], 60<<10)
	}
}

func TestCheckpointReferencingReclaimedSegment(t *testing.T) {
	// Checkpoint written, then its oldest segment reclaimed, then crash:
	// recovery must DROP the stale ckpt entries (missing file) instead of
	// minting phantom locations.
	dir := t.TempDir()
	v, _, _ := openTestVolume(t, dir)
	n := 15
	for i := 0; i < n; i++ {
		appendWait(t, v, i, 60<<10)
	}
	if v.ckpts.Load() == 0 {
		t.Fatal("fixture wrote no checkpoint")
	}
	id, entries, ok := v.OldestSealed()
	if !ok {
		t.Fatal("nothing sealed")
	}
	if !v.RetireBegin(id) {
		t.Fatal("retire begin refused")
	}
	if err := v.RetireFinish(id); err != nil {
		t.Fatal(err)
	}
	v.CrashForTest()

	v2, _, ents := openTestVolume(t, dir)
	defer func() { _ = v2.Close() }()
	for _, e := range ents {
		if e.Loc.SegmentID == id {
			t.Fatalf("phantom entry into reclaimed segment %d: %+v", id, e)
		}
		data, rel, st := v2.Read(e.Loc, e.NS, e.Key, e.XXH3)
		if st != ReadOK {
			t.Fatalf("surviving entry unreadable: %d", st)
		}
		rel()
		_ = data
	}
	_ = entries
}
