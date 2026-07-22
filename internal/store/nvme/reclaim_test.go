package nvme

import (
	"testing"
	"time"
)

func TestReclaimRetireProtocol(t *testing.T) {
	v, _, _ := openTestVolume(t, t.TempDir())
	defer func() { _ = v.Close() }()

	sums := make([]uint64, 9)
	locs := make([]Loc, 9)
	for i := range locs {
		locs[i], sums[i] = appendWait(t, v, i, 60<<10) // ≥2 sealed segments
	}
	id, entries, ok := v.OldestSealed(0)
	if !ok || len(entries) == 0 {
		t.Fatal("nothing sealed to reclaim")
	}

	// Hold a read across the retire — the transport-holds-a-view shape.
	e := entries[0]
	data, rel, st := v.Read(Loc{SegmentID: id, Offset: e.Off, Len: e.Len}, e.NS, e.Key, e.XXH3)
	if st != ReadOK {
		t.Fatalf("pre-retire read: %d", st)
	}

	if !v.RetireBegin(id) {
		t.Fatal("RetireBegin refused")
	}
	if v.RetireBegin(id) {
		t.Fatal("double RetireBegin succeeded")
	}
	// New reads on a dying segment refuse.
	if _, _, st := v.Read(Loc{SegmentID: id, Offset: e.Off, Len: e.Len}, e.NS, e.Key, e.XXH3); st != ReadGone {
		t.Fatalf("read on dying segment: %d, want gone", st)
	}
	// Abort restores service.
	v.RetireAbort(id)
	if d2, rel2, st := v.Read(Loc{SegmentID: id, Offset: e.Off, Len: e.Len}, e.NS, e.Key, e.XXH3); st != ReadOK {
		t.Fatalf("read after abort: %d", st)
	} else {
		rel2()
		_ = d2
	}

	// Retire for real; RetireFinish must WAIT for the held read.
	if !v.RetireBegin(id) {
		t.Fatal("re-RetireBegin refused")
	}
	finished := make(chan error, 1)
	go func() { finished <- v.RetireFinish(id) }()
	select {
	case <-finished:
		t.Fatal("RetireFinish completed while a read was still held")
	case <-time.After(50 * time.Millisecond):
	}
	// The held view is still byte-valid (fd survives the unlink).
	if data[0] != testPayload(0, 60<<10)[0] {
		t.Fatal("held view corrupted during retire")
	}
	rel()
	if err := <-finished; err != nil {
		t.Fatalf("RetireFinish: %v", err)
	}

	// The segment is gone: reads say so, space is back, writes may resume.
	if _, _, st := v.Read(Loc{SegmentID: id, Offset: e.Off, Len: e.Len}, e.NS, e.Key, e.XXH3); st != ReadGone {
		t.Fatalf("read after retire: %d, want gone", st)
	}
	if v.reclaims.Load() != 1 {
		t.Fatalf("reclaims counter %d", v.reclaims.Load())
	}
	// Un-reclaimed segments still serve.
	last := len(locs) - 1
	if locs[last].SegmentID == id {
		t.Fatal("fixture error: newest record was in the reclaimed segment")
	}
	mustRead(t, v, locs[last], last, sums[last], 60<<10)
}

func TestRetireRefusesActiveAndUnknown(t *testing.T) {
	v, _, _ := openTestVolume(t, t.TempDir())
	defer func() { _ = v.Close() }()
	appendWait(t, v, 0, 10<<10) // one record — segment 0 is ACTIVE, unsealed
	if v.RetireBegin(0) {
		t.Fatal("RetireBegin accepted the active segment")
	}
	if v.RetireBegin(42) {
		t.Fatal("RetireBegin accepted an unknown segment")
	}
	if err := v.RetireFinish(42); err == nil {
		t.Fatal("RetireFinish accepted an unbegun segment")
	}
}
