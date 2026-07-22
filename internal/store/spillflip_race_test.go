package store

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/kvstash/kvblockd/internal/eviction"
	"github.com/kvstash/kvblockd/internal/protocol"
	"github.com/kvstash/kvblockd/internal/store/dram"
	"github.com/kvstash/kvblockd/internal/store/nvme"

	"github.com/zeebo/xxh3"
)

// capSpill captures DemoteSegment requests instead of uploading; the TEST
// decides when onUp fires (standing in for the async s3spill worker).
type capSpill struct {
	mu   sync.Mutex
	reqs []capReq
	objs map[uint64][]byte
}

type capReq struct {
	segID uint64
	onUp  func(uint64, bool)
}

func (c *capSpill) DemoteSegment(segID uint64, _ int64, open func() (io.ReadSeekCloser, error), onUp func(uint64, bool)) bool {
	r, err := open()
	if err == nil {
		b, _ := io.ReadAll(r)
		_ = r.Close()
		c.mu.Lock()
		c.objs[segID] = b
		c.mu.Unlock()
	}
	c.mu.Lock()
	c.reqs = append(c.reqs, capReq{segID: segID, onUp: onUp})
	c.mu.Unlock()
	return true
}
func (c *capSpill) Drop(context.Context, uint64) error { return nil }
func (c *capSpill) Stats() (a, b, d uint64)            { return 0, 0, 0 }

func (c *capSpill) ReadRange(_ context.Context, segID uint64, off, n int64, dst []byte) error {
	c.mu.Lock()
	obj, ok := c.objs[segID]
	c.mu.Unlock()
	if !ok || off+n > int64(len(obj)) {
		return fmt.Errorf("no object %d", segID)
	}
	copy(dst, obj[off:off+n])
	return nil
}

func (c *capSpill) RestoreSegment(_ context.Context, segID uint64, sink func(io.Reader) error) error {
	c.mu.Lock()
	obj, ok := c.objs[segID]
	c.mu.Unlock()
	if !ok {
		return fmt.Errorf("no object %d", segID)
	}
	return sink(bytes.NewReader(obj))
}

type capRestore struct{ *capSpill }

func (capRestore) Stats() (a, b uint64) { return 0, 0 }

func sortCapReqs(reqs []capReq) {
	for i := 1; i < len(reqs); i++ {
		for j := i; j > 0 && reqs[j].segID < reqs[j-1].segID; j-- {
			reqs[j], reqs[j-1] = reqs[j-1], reqs[j]
		}
	}
}

// TestSpillAckVsReclaimFlip hunts the window between the spill-ack and a
// concurrent reclaim: the ack must set every ref's S3 flag BEFORE
// vol.MarkSpilled, and the retire-flip must abort rather than finish when
// it meets an entry whose flag (or protection) says no. Pre-fix, a reclaim
// reading IsSpilled==true mid-ack ran s3FlipRetired against bare refs and
// SKIPPED them — accounting drift (S3Only never set) or permanent ghosts
// (EXISTS true, GET NOT_FOUND, entry never removed). Probabilistic hunt:
// finding NOTHING passes; finding ANY drift or ghost is the regression.
func TestSpillAckVsReclaimFlip(t *testing.T) {
	trials := 20
	if testing.Short() {
		trials = 6
	}
	var driftTotal, ghostTotal int
	for trial := 0; trial < trials; trial++ {
		drift, ghost := spillFlipTrial(t, trial)
		driftTotal += drift
		ghostTotal += ghost
		if driftTotal > 0 && ghostTotal > 0 {
			break
		}
	}
	if driftTotal == 0 && ghostTotal == 0 {
		return // window not hit / nothing strandable — the invariant held
	}
	t.Fatalf("RACE CONFIRMED: drift=%d ghost=%d (flip ran against refs the spill-ack hadn't flagged)",
		driftTotal, ghostTotal)
}

func spillFlipTrial(t *testing.T, trial int) (drift, ghost int) {
	dir, err := os.MkdirTemp("", "kvbflip-*")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)

	cur := time.Now().UnixNano()
	arena, err := dram.NewArena(4<<20, false)
	if err != nil {
		t.Fatal(err)
	}
	pol, _ := eviction.New("s3fifo", 4096)
	ds := dram.New(arena, dram.Params{LeaseDefaultMS: 1, LeaseMaxMS: 2, Now: func() int64 { return cur }})
	ds.AttachPolicy(pol)
	defer ds.Close()

	vol, _, ents, err := nvme.OpenVolume(nvme.VolumeParams{
		Dir: dir, SegmentBytes: 64 << 10, MaxBytes: 32 << 20,
		ReadWorkers: 2, MaxBlobLen: 8 << 10,
		Now: func() int64 { return cur },
	})
	if err != nil {
		t.Fatal(err)
	}
	back := &capSpill{objs: map[uint64][]byte{}}
	tt := NewTiered(ds, pol, []*nvme.Volume{vol}, nil, [][]nvme.RecoveredEntry{ents}, Params{
		DemoteWatermarkPct: 1, DemoteBatchPct: 1,
		LeaseDefaultMS: 1, LeaseMaxMS: 2,
		Now:   func() int64 { return cur },
		Spill: back, Restore: capRestore{back},
	})
	defer tt.Close()

	// Fill: 2 KiB blocks -> ~28 records per 64 KiB segment; enough for ~3
	// sealed segments after demotion.
	blob := make([]byte, 2<<10)
	var keys [][32]byte
	for i := 0; i < 120; i++ {
		var k [32]byte
		k[0], k[1], k[2] = byte(i), byte(i>>8), byte(trial) //nolint:gosec // G115: test key mixing
		k[3] = 0xAB
		copy(blob, k[:])
		if st := tt.Put(1, k, blob, xxh3.Hash(blob)); st != protocol.StatusOK {
			t.Fatalf("put %d: %v", i, st)
		}
		keys = append(keys, k)
	}
	cur += int64(10 * time.Millisecond) // leases lapse
	tt.demotePass(true)                 // all victims -> volume, segments seal
	tt.spillPass()                      // enqueue sealed segments into capSpill

	back.mu.Lock()
	reqs := append([]capReq(nil), back.reqs...)
	back.mu.Unlock()
	if len(reqs) == 0 {
		return 0, 0
	}

	// The race: the spill-ack (worker) vs reclaimSegment (demoter), started
	// together per captured segment. reclaimSegment always takes the OLDEST
	// sealed segment, so pair in ascending segID order. Noise goroutines
	// oversubscribe the Ps so the ack worker can lose its P mid-ack.
	sortCapReqs(reqs)
	stopNoise := make(chan struct{})
	for i := 0; i < runtime.GOMAXPROCS(0); i++ {
		go func() {
			x := 0
			for {
				select {
				case <-stopNoise:
					return
				default:
					x++
				}
			}
		}()
	}
	defer close(stopNoise)
	for _, rq := range reqs {
		var wg sync.WaitGroup
		wg.Add(2)
		go func(r capReq) {
			defer wg.Done()
			r.onUp(r.segID, true) // per-ref S3 flag walk + MarkSpilled
		}(rq)
		go func(seed int) {
			defer wg.Done()
			for i := 0; i < seed%97; i++ { // randomized arrival vs the ack
				runtime.Gosched()
			}
			for i := 0; i < 512; i++ {
				if out, _ := tt.reclaimSegment(vol, 0); out == reclaimRetired {
					return
				}
			}
		}(trial*7919 + int(rq.segID)*131) //nolint:gosec // G115: schedule seed mixing, small segIDs
		wg.Wait()
	}

	// Post-mortem: any ref still pointing at a RETIRED segment must be a
	// completed flip (S3 + S3Only) — anything else is stranded state.
	for _, k := range keys {
		ref := tt.idx.get(dram.Key{NS: 1, Hash: k})
		if ref == nil {
			continue
		}
		if _, _, err := vol.OpenSegmentReadOnly(ref.Loc.SegmentID); err == nil {
			continue // still local (unreclaimed segment) — not interesting
		}
		if !ref.S3.Load() {
			ghost++ // unreadable forever, still indexed
			continue
		}
		if !ref.S3Only.Load() {
			drift++ // readable via S3 but flip accounting never ran
		}
	}
	return drift, ghost
}
