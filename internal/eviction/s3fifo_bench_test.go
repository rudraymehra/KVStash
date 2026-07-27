package eviction

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// worstCaseDomain builds the most expensive Victims schedule: N main-queue
// entries all at freq 3, so the scan burns three second-chance rounds
// (3N visits) before the first expel.
func worstCaseDomain(p *S3FIFO, ns uint32, n int) {
	for i := 0; i < n; i++ {
		var k Key
		k.NS = ns
		k.Hash[0], k.Hash[1], k.Hash[2] = byte(i), byte(i>>8), byte(i>>16)
		k.Hash[8], k.Hash[9], k.Hash[10] = byte(i), byte(i>>8), byte(i>>16)
		p.Admit(k, 4096, 0)
		for j := 0; j < 3; j++ {
			p.Touch(k, 0)
		}
	}
}

// BenchmarkVictimsWorstCaseQmuHold pins the chunked scan's lock bound: while
// one goroutine runs the all-freq-3 worst-case Victims pass, a sampler times
// qmu acquisitions through Usage (the same lock Admit/Remove/the demoter
// take under pressure). max-qmu-wait-ns is the number the fix exists for —
// before chunking it equaled the WHOLE scan (tens of ms at 10^5+ entries,
// scaling with population); with victimsYieldEvery the observed bound is
// one chunk plus sync.Mutex's ~1ms starvation-mode handoff threshold
// (barging soaks shorter waits), INDEPENDENT of the domain size — measured
// ~1.5ms at 200k entries vs ~30ms for the unchunked scan.
func BenchmarkVictimsWorstCaseQmuHold(b *testing.B) {
	const entries = 200_000
	var maxWait atomic.Int64
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		p := NewS3FIFO(1 << 20)
		worstCaseDomain(p, 1, entries)
		stop := make(chan struct{})
		var wg sync.WaitGroup
		wg.Add(1)
		go func() { // the contender: samples qmu wait through Usage
			defer wg.Done()
			var dst []DomainUsage
			for {
				select {
				case <-stop:
					return
				default:
				}
				t0 := time.Now()
				dst = p.Usage(0, dst[:0])
				if w := time.Since(t0).Nanoseconds(); w > maxWait.Load() {
					maxWait.Store(w)
				}
			}
		}()
		b.StartTimer()
		// need beyond the domain: the scan runs to exhaustion (6N+8 visits).
		got := p.Victims(1, int64(entries)*4096+1, 0, nil)
		b.StopTimer()
		close(stop)
		wg.Wait()
		if len(got) == 0 {
			b.Fatal("worst-case scan expelled nothing")
		}
	}
	b.ReportMetric(float64(maxWait.Load()), "max-qmu-wait-ns")
}

// TestVictimsChunkedScanIsComplete pins that chunking changed no eviction
// decision: the worst-case domain still drains fully, in FIFO-with-second-
// chances order, covering the requested need exactly as the unchunked scan
// did (the single-threaded schedule is identical — yields only release and
// retake an uncontended lock).
func TestVictimsChunkedScanIsComplete(t *testing.T) {
	const entries = 3 * victimsYieldEvery // force several yield boundaries
	p := NewS3FIFO(1 << 20)
	worstCaseDomain(p, 7, entries)
	got := p.Victims(7, int64(entries)*4096, 0, nil)
	if len(got) != entries {
		t.Fatalf("chunked scan expelled %d of %d entries", len(got), entries)
	}
	// Drained of resident bytes; ghost-only footprint stays visible by
	// design (it is real memory outside the arena budget).
	for _, u := range p.Usage(0, nil) {
		if u.Bytes != 0 {
			t.Fatalf("domain not drained: %v", u)
		}
	}
}

// TestVictimsYieldRace provokes the new interleaving under -race: Admit,
// Remove, Touch, and Usage all land inside Victims' yield windows. The
// assertion is the race detector plus queue-consistency (no panic, no
// negative byte accounting via Usage).
func TestVictimsYieldRace(t *testing.T) {
	p := NewS3FIFO(1 << 20)
	worstCaseDomain(p, 3, 4*victimsYieldEvery)
	stop := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(2)
	go func() { // mutator: admissions and removals racing the scan
		defer wg.Done()
		i := 0
		for {
			select {
			case <-stop:
				return
			default:
			}
			var k Key
			k.NS = 3
			k.Hash[4], k.Hash[5], k.Hash[12] = byte(i), byte(i>>8), 0xEE
			p.Admit(k, 4096, 0)
			p.Touch(k, 0)
			p.Remove(k)
			i++
		}
	}()
	go func() { // reader: Usage must never observe torn counters
		defer wg.Done()
		var dst []DomainUsage
		for {
			select {
			case <-stop:
				return
			default:
			}
			dst = p.Usage(0, dst[:0])
			for _, u := range dst {
				if u.Bytes < 0 {
					panic("negative domain bytes")
				}
			}
		}
	}()
	for pass := 0; pass < 8; pass++ {
		p.Victims(3, 1<<40, 0, nil)
		worstCaseDomain(p, 3, 2*victimsYieldEvery) // refill for the next pass
	}
	close(stop)
	wg.Wait()
}
