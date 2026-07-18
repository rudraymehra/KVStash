package nvme

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/zeebo/xxh3"
	"golang.org/x/sys/unix"
)

// faultBackend wraps the real backend and fails WriteAt with ENOSPC once
// armed — the disk-full drill.
type faultBackend struct {
	inner IOBackend
	armed atomic.Bool
}

type faultFile struct {
	File
	b *faultBackend
}

func (fb *faultBackend) Open(path string, forWrite bool) (File, error) {
	f, err := fb.inner.Open(path, forWrite)
	if err != nil {
		return nil, err
	}
	return &faultFile{File: f, b: fb}, nil
}

func (ff *faultFile) WriteAt(p []byte, off int64) error {
	if ff.b.armed.Load() {
		return unix.ENOSPC
	}
	return ff.File.WriteAt(p, off)
}

func TestWriterENOSPCFlipsReadOnly(t *testing.T) {
	dir := t.TempDir()
	fb := &faultBackend{inner: DefaultBackend()}
	p := testParams(t, dir)
	p.Backend = fb
	v, _, _, err := OpenVolume(p)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = v.Close() }()

	loc, sum := appendWait(t, v, 0, 20<<10) // healthy write first

	fb.armed.Store(true)
	failed := make(chan bool, 1)
	ok := v.Append(AppendReq{
		NS: 1, Key: testKey(1), XXH3: 1, Data: testPayload(1, 20<<10),
		OnWritten: func(_ Loc, wok bool) { failed <- wok },
	})
	if !ok {
		t.Fatal("append refused before the fault fired")
	}
	select {
	case wok := <-failed:
		if wok {
			t.Fatal("OnWritten reported ok through ENOSPC")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("OnWritten never fired on ENOSPC")
	}
	if v.enospc.Load() == 0 {
		t.Fatal("enospc counter untouched")
	}
	// The volume is read-only now: new appends refuse at the gate.
	deadline := time.Now().Add(2 * time.Second)
	for v.Append(AppendReq{NS: 1, Key: testKey(2), XXH3: 2, Data: testPayload(2, 1024), OnWritten: func(Loc, bool) {}}) {
		if time.Now().After(deadline) {
			t.Fatal("appends still accepted after ENOSPC flip")
		}
		time.Sleep(10 * time.Millisecond)
	}
	// Reads keep working through it all.
	mustRead(t, v, loc, 0, sum, 20<<10)
	fb.armed.Store(false)
}

func TestWriterGroupCommitBatches(t *testing.T) {
	v, _, _ := openTestVolume(t, t.TempDir())
	defer func() { _ = v.Close() }()

	// Fire a burst without waiting: the writer drains greedily; every ack
	// must arrive and every record read back — batch boundaries invisible.
	const n = 40
	var wg sync.WaitGroup
	locs := make([]Loc, n)
	sums := make([]uint64, n)
	oks := make([]bool, n)
	for i := 0; i < n; i++ {
		p := testPayload(i, 3<<10)
		sums[i] = xxh3.Hash(p)
		wg.Add(1)
		idx := i
		accepted := v.Append(AppendReq{
			NS: 1, Key: testKey(idx), XXH3: sums[idx], Data: p,
			OnWritten: func(loc Loc, ok bool) {
				locs[idx], oks[idx] = loc, ok
				wg.Done()
			},
		})
		if !accepted {
			wg.Done() // bounded queue may refuse under burst — legal drop
		}
	}
	wg.Wait()
	served := 0
	for i := 0; i < n; i++ {
		if !oks[i] {
			continue
		}
		mustRead(t, v, locs[i], i, sums[i], 3<<10)
		served++
	}
	if served == 0 {
		t.Fatal("burst produced zero written records")
	}
	if v.appended.Load() != uint64(served) { //nolint:gosec // G115: test count
		t.Fatalf("appended counter %d, want %d", v.appended.Load(), served)
	}
}

func TestWriterRejectsOversizedBlob(t *testing.T) {
	dir := t.TempDir()
	v, _, _ := openTestVolume(t, dir)
	defer func() { _ = v.Close() }()
	res := make(chan bool, 1)
	ok := v.Append(AppendReq{
		NS: 1, Key: testKey(0), XXH3: 1,
		Data:      make([]byte, int(testParams(t, dir).MaxBlobLen)+1),
		OnWritten: func(_ Loc, wok bool) { res <- wok },
	})
	if !ok {
		return // refused at the gate — also acceptable
	}
	if wok := <-res; wok {
		t.Fatal("oversized blob written")
	}
}

// TestWriterBatchNeverCollidesWithFooter pins the staged-bytes accounting
// bug the tiered scrub caught: a greedy batch of records must fit the
// segment INCLUDING unflushed staging, or the seal trailer overwrites the
// last record's tail. Burst 60 KiB records into 256 KiB segments (4 stage
// as one batch) and every acked record must read back verified — across
// the seals the burst forces.
func TestWriterBatchNeverCollidesWithFooter(t *testing.T) {
	v, _, _ := openTestVolume(t, t.TempDir())
	defer func() { _ = v.Close() }()

	const n = 12
	var wg sync.WaitGroup
	locs := make([]Loc, n)
	oks := make([]bool, n)
	sums := make([]uint64, n)
	for i := 0; i < n; i++ {
		p := testPayload(i, 60<<10)
		sums[i] = xxh3.Hash(p)
		idx := i
		wg.Add(1)
		if !v.Append(AppendReq{
			NS: 1, Key: testKey(idx), XXH3: sums[idx], Data: p,
			OnWritten: func(loc Loc, ok bool) {
				locs[idx], oks[idx] = loc, ok
				wg.Done()
			},
		}) {
			wg.Done()
		}
	}
	wg.Wait()
	for i := 0; i < n; i++ {
		if !oks[i] {
			continue
		}
		// The record's span must never reach into the trailer chunk.
		if int64(locs[i].Offset)+int64(recordSpan(locs[i].Len)) > v.p.SegmentBytes-recordAlign { //nolint:gosec // G115: test values
			t.Fatalf("record %d at %d spans into the footer region", i, locs[i].Offset)
		}
		mustRead(t, v, locs[i], i, sums[i], 60<<10)
	}
}
