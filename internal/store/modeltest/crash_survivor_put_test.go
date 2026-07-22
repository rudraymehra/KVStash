package modeltest

import (
	"bytes"
	"testing"
	"time"

	"github.com/zeebo/xxh3"

	"github.com/kvstash/kvblockd/internal/eviction"
	"github.com/kvstash/kvblockd/internal/protocol"
	"github.com/kvstash/kvblockd/internal/store"
	"github.com/kvstash/kvblockd/internal/store/dram"
	"github.com/kvstash/kvblockd/internal/store/nvme"
)

// TestPutCrashSurvivorOKExists is the deterministic regression for the
// 2026-07-22 CI model-soak failure ("put conflict: OK_EXISTS
// (maybeGone=true)", seed 12773142929788551604, linux/amd64): a block is
// committed, demoted to NVMe (DemoteNow waits for the append — the record
// bytes are IN the segment file before it returns), the process crashes
// without sync/seal, and recovery resurfaces the entry from the unsealed
// tail. Re-putting the IDENTICAL content then answers OK_EXISTS from the
// write-once pre-check — a fully legal outcome (the block genuinely
// survived, carrying content the key committed) that the oracle rejected:
// simulate_crash reverts every surviving model block to the anyOf form
// (data=nil, xxh3=0), and the put truth table classified anyOf blocks by
// e.xxh3(=0) != sum as "conflicting content", where OK_EXISTS had no arm.
//
// The random walk only reproduced when the crash landed AFTER the record
// reached the file (schedule/platform dependent — CI linux hit it, darwin
// failfile replays did not, because survival is decided by writer-goroutine
// progress at CrashForTest, which the recorded draws cannot capture). This
// test removes the timing from the equation: OnWritten completion is a
// happens-before edge to the crash, so the survivor is guaranteed.
func TestPutCrashSurvivorOKExists(t *testing.T) {
	cur := startNanos
	volDir := t.TempDir()
	coldTier := newFakeS3() // s3-kind wiring, mirroring the failing walk

	// mk mirrors the machine's mkSUT (kind=="s3") over a SHARED volDir so the
	// rebuild after CrashForTest runs real recovery. DemoteWatermarkPct 1
	// makes the single resident block clear the watermark deterministically.
	mk := func() (*dram.Store, *store.Tiered) {
		pol, err := eviction.New("sampled-lru", 8192) // the failing walk's policy draw
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
			Dir: volDir, SegmentBytes: segBytes, MaxBytes: volMaxS3,
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
				DemoteWatermarkPct: 1, DemoteBatchPct: 40,
			})
		return ds, tt
	}
	ds, tt := mk()

	m := newModel(startNanos, pinnedCap)
	// The failing walk's protagonist: ns=2, pool key 3, content
	// fill 0x2 ^ ns 0x2 = 0x00 (sized to clear the demote watermark; the
	// walk's 4097-byte draw is too small to trip a deterministic pass).
	k := eviction.Key{NS: 2, Hash: poolHash(3)}
	data := bytes.Repeat([]byte{0x00}, 262144)
	sum := xxh3.Hash(data)

	// Commit (the walk's line-5001 put) — through the SAME oracle.
	if msg := m.applyPut(k, data, sum, tt.Put(k.NS, k.Hash, data, sum)); msg != "" {
		t.Fatalf("first put: %s", msg)
	}
	// Demote it. demotePass(wait=true) blocks on the append's OnWritten, so
	// the record's WriteAt has RETURNED before DemoteNow does — the bytes are
	// in the page cache under the segment file, which a same-OS reopen (crash
	// recovery) reads back even though nothing was synced.
	cur += int64(time.Second)
	tt.DemoteNow()
	if ds.Contains(k.NS, k.Hash) {
		t.Fatal("setup: block still DRAM-resident after DemoteNow — demotion never ran")
	}
	if !tt.Contains(k.NS, k.Hash) {
		t.Fatal("setup: block vanished during demotion")
	}

	// SIGKILL + real recovery, the machine's simulate_crash verbatim.
	tt.CrashForTest()
	_, tt2 := mk()
	defer func() { _ = tt2.Close() }()
	m.crashed() // every surviving model block reverts to the anyOf form

	// The failing step: re-put the IDENTICAL bytes. The recovered write-once
	// pre-check must see the survivor and answer OK_EXISTS — if this ever
	// misses, recovery lost a completed append and this test must fail loudly
	// (it would also mean the regression can no longer exercise the oracle).
	st := tt2.Put(k.NS, k.Hash, data, sum)
	if st != protocol.StatusOKExists {
		t.Fatalf("re-put of surviving content: %s, want OK_EXISTS (recovery lost the record)", st)
	}
	// The oracle must accept the legal survivor hit. RED before the anyOf
	// arm existed: "put conflict: OK_EXISTS (maybeGone=true)" — the exact CI
	// failure.
	if msg := m.applyPut(k, data, sum, st); msg != "" {
		t.Fatalf("oracle rejected a legal crash-survivor OK_EXISTS: %s", msg)
	}
	// And it must have PINNED the block down (OK_EXISTS is digest evidence,
	// same as a GET): anyOf resolved to the re-put content.
	if b := m.blocks[k]; b == nil || b.anyOf != nil || b.xxh3 != sum || !bytes.Equal(b.data, data) {
		t.Fatal("oracle did not pin the survivor down to the re-put content")
	}

	// Teeth stay in: OK_EXISTS vouching for bytes the key NEVER committed is
	// still an I1-grade violation, crash or no crash — the fix must tolerate
	// exactly the legal outcome, nothing more.
	m2 := newModel(startNanos, pinnedCap)
	m2.insert(k, data, sum)
	m2.crashed()
	other := bytes.Repeat([]byte{0xA5}, 64)
	if msg := m2.applyPut(k, other, xxh3.Hash(other), protocol.StatusOKExists); msg == "" {
		t.Fatal("oracle accepted OK_EXISTS for bytes outside the key's committed history")
	}
}
