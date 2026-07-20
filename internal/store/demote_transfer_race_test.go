package store

import (
	"fmt"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/kvstash/kvblockd/internal/eviction"
	"github.com/kvstash/kvblockd/internal/protocol"
	"github.com/kvstash/kvblockd/internal/store/dram"
	"github.com/kvstash/kvblockd/internal/store/nvme"
	"github.com/kvstash/kvblockd/internal/tenant"

	"github.com/zeebo/xxh3"
)

// TestDemotePublishVsDeleteQuotaLeak hammers DELETE against demotion
// completion: CompleteDemotion publishes the NVMe index entry (idx.put,
// under the shard locks), and the NVMe-side quota charge must land inside
// the SAME nvme shard hold — when it instead landed in a Transfer call AFTER
// CompleteDemotion returned, a DELETE slipping between them refunded an
// uncharged NVMe balance (underflow-healed to 0 in release; PANIC under
// kvbdebug) and the late Transfer then minted a permanent phantom NVMe
// charge. The index is fully drained before each measurement, so any
// residue is the leak.
func TestDemotePublishVsDeleteQuotaLeak(t *testing.T) {
	dir, err := os.MkdirTemp("", "kvbleak-*")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)

	cur := time.Now().UnixNano()
	arena, err := dram.NewArena(8<<20, false)
	if err != nil {
		t.Fatal(err)
	}
	reg := tenant.NewRegistry("a", 1, "tok") // no limits: unlimited, usage still tracked
	q := tenant.NewQuotas(reg)
	pol, _ := eviction.New("s3fifo", 65536)
	ds := dram.New(arena, dram.Params{LeaseDefaultMS: 1, LeaseMaxMS: 2, Now: func() int64 { return cur }, Quotas: q})
	ds.AttachPolicy(pol)
	defer ds.Close()

	vol, _, ents, err := nvme.OpenVolume(nvme.VolumeParams{
		Dir: dir, SegmentBytes: 4 << 20, MaxBytes: 1 << 30,
		ReadWorkers: 2, MaxBlobLen: 64 << 10,
		Now: func() int64 { return cur },
	})
	if err != nil {
		t.Fatal(err)
	}
	tt := NewTiered(ds, pol, []*nvme.Volume{vol}, nil, [][]nvme.RecoveredEntry{ents}, Params{
		DemoteWatermarkPct: 1, DemoteBatchPct: 1,
		LeaseDefaultMS: 1, LeaseMaxMS: 2,
		Now:    func() int64 { return cur },
		Quotas: q,
	})
	defer tt.Close()

	const nKeys = 64
	blob := make([]byte, 8<<10)
	budget := 8 * time.Second
	if testing.Short() {
		budget = 2 * time.Second
	}
	deadline := time.Now().Add(budget)
	round := 0
	for time.Now().Before(deadline) {
		round++
		var keys [][32]byte
		for i := 0; i < nKeys; i++ {
			var k [32]byte
			copy(k[:], fmt.Sprintf("R%06d-K%03d", round, i))
			k[31] = 0xCD
			copy(blob, k[:])
			if st := tt.Put(1, k, blob, xxh3.Hash(blob)); st != protocol.StatusOK {
				t.Fatalf("put: %v", st)
			}
			keys = append(keys, k)
		}
		cur += int64(10 * time.Millisecond) // leases lapse — everything demotable

		stop := make(chan struct{})
		var wg sync.WaitGroup
		for w := 0; w < 3; w++ {
			wg.Add(1)
			go func(off int) {
				defer wg.Done()
				for {
					select {
					case <-stop:
						return
					default:
					}
					for i := off; i < len(keys); i += 3 {
						_ = tt.Delete(1, keys[i], true)
					}
				}
			}(w)
		}
		tt.demotePass(true) // waits for every accepted append's OnWritten
		close(stop)
		wg.Wait()
		for _, k := range keys {
			_ = tt.Delete(1, k, true)
		}
		if blocks, bytes := tt.idx.stats(); blocks == 0 && bytes == 0 {
			if nv := q.Usage(1, tenant.TierNVMe); nv != 0 {
				t.Fatalf("QUOTA LEAK after %d rounds: %d phantom NVMe bytes charged with an EMPTY nvme index", round, nv)
			}
		}
	}
	blocks, bytes := tt.idx.stats()
	nv := q.Usage(1, tenant.TierNVMe)
	t.Logf("%d rounds; final index: %d blocks/%d bytes; nvme ledger: %d", round, blocks, bytes, nv)
	if blocks == 0 && nv != 0 {
		t.Fatalf("QUOTA LEAK: %d phantom NVMe bytes with an empty index", nv)
	}
}
