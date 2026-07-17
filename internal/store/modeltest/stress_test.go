package modeltest

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"math/rand/v2"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/zeebo/xxh3"

	"github.com/kvstash/kvblockd/internal/eviction"
	"github.com/kvstash/kvblockd/internal/protocol"
	"github.com/kvstash/kvblockd/internal/store/dram"
)

// stressPayload derives a block's full content from its identity — any GET
// hit can be verified byte-for-byte with no shared bookkeeping.
func stressPayload(ns uint32, i int, size int) []byte {
	out := make([]byte, size)
	seed := uint64(ns)*0x9E3779B97F4A7C15 + uint64(i) //nolint:gosec // deterministic content
	var b [8]byte
	for off := 0; off < size; off += 8 {
		binary.LittleEndian.PutUint64(b[:], seed^uint64(off)) //nolint:gosec // as above
		copy(out[off:], b[:])
	}
	return out
}

func stressKey(ns uint32, i int) [32]byte {
	var k [32]byte
	binary.LittleEndian.PutUint32(k[0:4], uint32(i)) //nolint:gosec // test index
	binary.LittleEndian.PutUint32(k[4:8], ns)
	k[9] = byte(i) //nolint:gosec // G115: byte mixing
	return k
}

// TestStressConcurrentEviction: rapid is sequential, so this is the
// concurrency leg — 8 workers hammering Get/Put/Delete plus a ref-holder
// goroutine, with the watermark evictor LIVE at an aggressive cadence.
// Every GET hit is byte-verified (I1 under concurrency); every held view is
// snapshot-compared at release (I5 under concurrency); Close's kvbdebug
// refcount assert is the final gate.
func TestStressConcurrentEviction(t *testing.T) {
	dur := 8 * time.Second
	if testing.Short() {
		dur = 2 * time.Second
	}
	arena, err := dram.NewArena(16<<20, false)
	if err != nil {
		t.Fatal(err)
	}
	pol, err := eviction.New("s3fifo", 8192)
	if err != nil {
		t.Fatal(err)
	}
	s := dram.New(arena, dram.Params{LeaseDefaultMS: 50, LeaseMaxMS: 60_000})
	s.AttachPolicy(pol)
	stop := s.StartEvictor(context.Background(), dram.EvictorConfig{
		WatermarkPct: 80, BatchPct: 10, Interval: 5 * time.Millisecond,
	})

	const (
		nWorkers = 8
		keySpace = 512
		blkSize  = 64 << 10 // 512 × 64 KiB = 32 MiB working set over a 16 MiB arena
	)
	var (
		wg       sync.WaitGroup
		stopFlag atomic.Bool
		gets     atomic.Int64
		verifyOK atomic.Int64
	)
	for w := 0; w < nWorkers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			rng := rand.New(rand.NewPCG(uint64(w), 42)) //nolint:gosec // reproducible-ish load shape
			for !stopFlag.Load() {
				ns := uint32(1 + rng.IntN(2)) //nolint:gosec // G115: 1..2
				i := rng.IntN(keySpace)
				k := stressKey(ns, i)
				switch rng.IntN(10) {
				case 0: // delete (mixed force/plain — plain exercises the lease gate)
					_ = s.Delete(ns, k, i%2 == 0)
				case 1, 2, 3: // put
					data := stressPayload(ns, i, blkSize)
					st := s.Put(ns, k, data, xxh3.Hash(data))
					switch st {
					case protocol.StatusOK, protocol.StatusOKExists, protocol.StatusErrQuotaBytes, protocol.StatusErrBusy:
					default:
						t.Errorf("put: %s", st)
						return
					}
				default: // get + verify
					data, _, ok := s.Get(ns, k)
					gets.Add(1)
					if ok {
						if !bytes.Equal(data, stressPayload(ns, i, blkSize)) {
							t.Errorf("I1 under concurrency: torn/foreign bytes ns=%d i=%d", ns, i)
							return
						}
						verifyOK.Add(1)
					}
				}
			}
		}(w)
	}
	// The ref-holder: snapshot at acquire, compare at release (I5).
	wg.Add(1)
	go func() {
		defer wg.Done()
		rng := rand.New(rand.NewPCG(99, 7)) //nolint:gosec // as above
		for !stopFlag.Load() {
			ns := uint32(1 + rng.IntN(2)) //nolint:gosec // G115: 1..2
			i := rng.IntN(keySpace)
			view, _, release, ok := s.GetRef(ns, stressKey(ns, i))
			if !ok {
				continue
			}
			snap := make([]byte, len(view))
			copy(snap, view)
			time.Sleep(time.Duration(1+rng.IntN(10)) * time.Millisecond)
			if !bytes.Equal(view, snap) {
				t.Error("I5 under concurrency: held view mutated during the hold")
				release()
				return
			}
			release()
		}
	}()

	time.Sleep(dur)
	stopFlag.Store(true)
	wg.Wait()
	stop() // evictor halted before Close — the drain-before-unmap rule

	if t.Failed() {
		return
	}
	var doc storeStats
	if err := json.Unmarshal(s.Stats(), &doc); err != nil {
		t.Fatal(err)
	}
	if doc.EvictionsTotal == 0 {
		t.Fatal("no evictions during the stress window — the pressure loop is broken")
	}
	t.Logf("stress: %d gets (%d verified hits), %d evictions, %d resident",
		gets.Load(), verifyOK.Load(), doc.EvictionsTotal, doc.Blocks)
	if err := s.Close(); err != nil { // kvbdebug: asserts zero leaked reader refs
		t.Fatal(err)
	}
}
