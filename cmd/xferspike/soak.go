package main

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net"
	"os"
	"runtime"
	"runtime/metrics"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/sys/unix"
)

// gcPauseMetric is the current name for the GC stop-the-world pause-latency
// histogram. The older /gc/pauses:seconds is deprecated in favor of this.
const gcPauseMetric = "/sched/pauses/total/gc:seconds"

// soak blob size band = the vLLM contiguous-block sizes kvblockd will store.
var soakBlobSizes = []int{440 << 10, 1 << 20, 2560 << 10} // 0.44, 1, 2.5 MiB

// soakResult is the A2 JSON line: proof that blob bytes live off the Go heap
// (heap_alloc_bytes stays tiny while rss_bytes tracks the arena) plus the GC
// pause distribution measured during the load window.
type soakResult struct {
	Mode           string  `json:"mode"`
	ArenaBytes     int64   `json:"arena_bytes"`
	Blobs          int     `json:"blobs"`
	HugePages      bool    `json:"hugepages"`
	SoakStreams    int     `json:"soak_streams"`
	DurationS      float64 `json:"duration_s"`
	BytesServed    int64   `json:"bytes_served"`
	HeapAllocBytes uint64  `json:"heap_alloc_bytes"`
	RSSBytes       int64   `json:"rss_bytes"`
	GCPausesSeen   uint64  `json:"gc_pauses_observed"`
	// Valid is true only when the window carried enough real work + pauses to
	// trust the percentiles. A gate MUST check valid==true before scoring p99 —
	// otherwise an empty/degenerate run's absent percentiles could read as 0.
	Valid bool `json:"valid"`
	// Percentiles are pointers with omitempty: absent (not 0) when !Valid, so a
	// naive "p99 < 5" gate can't false-PASS on a run that measured nothing.
	GCPauseP50ms  *float64 `json:"gc_pause_p50_ms,omitempty"`
	GCPauseP99ms  *float64 `json:"gc_pause_p99_ms,omitempty"`
	GCPauseP999ms *float64 `json:"gc_pause_p999_ms,omitempty"`
	GCPauseMaxms  *float64 `json:"gc_pause_max_ms,omitempty"`
	Note          string   `json:"note,omitempty"`
}

func (r soakResult) print() {
	if err := json.NewEncoder(os.Stdout).Encode(r); err != nil {
		fmt.Fprintln(os.Stderr, "xferspike: encode soak result:", err)
	}
}

type blobRef struct {
	off int
	len int
}

// runSoak is the A2 rig. It mmaps an off-heap arena of immutable blobs, then
// serves them over loopback via writev (zero blob bytes on the Go heap) under
// concurrent load, measuring GC pause percentiles across the window.
func runSoak(cfg config) error {
	if cfg.arenaBytes < soakBlobSizes[len(soakBlobSizes)-1] {
		return fmt.Errorf("arena-bytes %d too small; need at least %d", cfg.arenaBytes, soakBlobSizes[len(soakBlobSizes)-1])
	}
	if cfg.soakStreams < 1 {
		return fmt.Errorf("soak-streams must be >= 1")
	}

	arena, unmap, huge, err := mmapArena(cfg.arenaBytes)
	if err != nil {
		return fmt.Errorf("mmap arena: %w", err)
	}
	defer func() {
		if cerr := unmap(); cerr != nil {
			fmt.Fprintln(os.Stderr, "xferspike: munmap:", cerr)
		}
	}()

	// Carve deterministic blobs into the arena, touching every page so RSS
	// reflects the true footprint. Only the small index plus per-client buffers
	// live on the Go heap; the blob bytes stay in the mmap (the A2 point).
	index := carveArena(arena)
	fmt.Fprintf(os.Stderr, "xferspike soak: arena=%d MiB, blobs=%d, hugepages=%v\n",
		cfg.arenaBytes>>20, len(index), huge)

	// Loopback server that serves blob-id GETs from arena subslices via writev.
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return fmt.Errorf("listen: %w", err)
	}
	defer ln.Close()

	var bytesServed atomic.Int64
	var srvWG sync.WaitGroup
	go func() {
		for {
			conn, aerr := ln.Accept()
			if aerr != nil {
				return
			}
			tune(conn, cfg)
			srvWG.Add(1)
			go func() {
				defer srvWG.Done()
				serveArena(conn, arena, index, &bytesServed)
			}()
		}
	}()

	// Start the pause-histogram window right before the load.
	before := readGCPauses()

	ctx, cancel := context.WithTimeout(context.Background(), cfg.duration)
	defer cancel()

	var churnStop atomic.Bool
	go churn(&churnStop) // realistic small-object GC pressure

	wallStart := time.Now()
	var cliWG sync.WaitGroup
	for i := 0; i < cfg.soakStreams; i++ {
		cliWG.Add(1)
		go func(seed uint64) {
			defer cliWG.Done()
			soakClient(ctx, ln.Addr().String(), cfg, len(index), seed)
		}(uint64(i) + 1)
	}
	cliWG.Wait()
	wall := time.Since(wallStart)

	// Stop accepting and drain in-flight serveArena goroutines BEFORE the
	// deferred unmap frees the arena they read from. Clients have all closed,
	// so every serveArena loop hits EOF and returns promptly — no new conns can
	// be accepted after Close, so this can't deadlock on the WaitGroup.
	_ = ln.Close()
	srvWG.Wait()

	churnStop.Store(true)
	after := readGCPauses()

	p50, p99, p999, maxp, seen := pausePercentiles(before, after)

	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)

	served := bytesServed.Load()
	res := soakResult{
		Mode:           "soak",
		ArenaBytes:     int64(cfg.arenaBytes),
		Blobs:          len(index),
		HugePages:      huge,
		SoakStreams:    cfg.soakStreams,
		DurationS:      wall.Seconds(),
		BytesServed:    served,
		HeapAllocBytes: ms.HeapAlloc,
		RSSBytes:       maxRSSBytes(),
		GCPausesSeen:   seen,
	}
	if huge {
		res.Note = "hugepages in use: rss_bytes may exclude hugetlb pages on Linux; trust heap_alloc_bytes as the off-heap proof"
	}

	// Percentiles are only trustworthy with a real workload AND enough pauses.
	// Leaving them absent (nil pointers) when invalid prevents a gate from
	// reading a degenerate run's missing p99 as 0 and false-PASSing.
	const minValidPauses = 100
	switch {
	case cfg.duration <= 0:
		res.Note = joinNote(res.Note, "non-positive duration: no measurement window")
	case served == 0:
		res.Note = joinNote(res.Note, "no bytes served: the writev-from-arena path did not run; any pause figure would be churn-only noise")
	case seen < minValidPauses:
		res.Note = joinNote(res.Note, fmt.Sprintf("insufficient GC pauses (%d < %d) for a trustworthy percentile; increase --duration/--arena-bytes", seen, minValidPauses))
	case math.IsInf(p99, 1) || math.IsInf(p999, 1) || math.IsInf(maxp, 1):
		res.Note = joinNote(res.Note, "a pause exceeded the largest finite histogram bucket")
	default:
		res.Valid = true
		p50ms, p99ms, p999ms, maxms := p50*1e3, p99*1e3, p999*1e3, maxp*1e3
		res.GCPauseP50ms, res.GCPauseP99ms = &p50ms, &p99ms
		res.GCPauseP999ms, res.GCPauseMaxms = &p999ms, &maxms
		if seen < 1000 {
			res.Note = joinNote(res.Note, "p999 is only meaningful at >=1000 pauses; treat gc_pause_max_ms as the tail here")
		}
	}
	res.print()
	return nil
}

// joinNote concatenates note fragments with "; ".
func joinNote(a, b string) string {
	if a == "" {
		return b
	}
	return a + "; " + b
}

// carveArena fills the arena with deterministic blobs (cycling the size band),
// touching each blob's first byte of every page so the pages are resident.
func carveArena(arena []byte) []blobRef {
	var index []blobRef
	off := 0
	i := 0
	for {
		sz := soakBlobSizes[i%len(soakBlobSizes)]
		if off+sz > len(arena) {
			break
		}
		// Touch one byte per 4 KiB page to fault it in and make RSS honest.
		for p := off; p < off+sz; p += 4096 {
			arena[p] = byte(i)
		}
		index = append(index, blobRef{off: off, len: sz})
		off += sz
		i++
	}
	return index
}

// serveArena reads 4-byte blob-id requests and replies header+blob via writev,
// sending the blob straight from the arena subslice (no heap copy).
func serveArena(conn net.Conn, arena []byte, index []blobRef, bytesServed *atomic.Int64) {
	defer conn.Close()
	req := make([]byte, 4)
	hdr := make([]byte, headerSize)
	for {
		if _, err := io.ReadFull(conn, req); err != nil {
			return
		}
		id := int(binary.LittleEndian.Uint32(req))
		if id < 0 || id >= len(index) {
			return
		}
		ref := index[id]
		putHeader(hdr, frameHeader{magic: magic, seq: uint64(id), length: uint32(ref.len)}) //nolint:gosec // G115: ref.len is a soakBlobSizes entry, all << MaxUint32
		bufs := net.Buffers{hdr, arena[ref.off : ref.off+ref.len]}
		n, err := bufs.WriteTo(conn)
		bytesServed.Add(n)
		if err != nil {
			return
		}
	}
}

// soakClient requests random blob-ids until ctx expires, reusing one buffer.
func soakClient(ctx context.Context, addr string, cfg config, nBlobs int, seed uint64) {
	var d net.Dialer
	conn, err := d.DialContext(ctx, "tcp", addr)
	if err != nil {
		return
	}
	defer conn.Close()
	tune(conn, cfg)

	deadline, hasDeadline := ctx.Deadline()
	req := make([]byte, 4)
	hdr := make([]byte, headerSize)
	buf := make([]byte, soakBlobSizes[len(soakBlobSizes)-1]) // max blob
	rng := seed

	for ctx.Err() == nil {
		rng = rng*6364136223846793005 + 1442695040888963407 // LCG (Knuth MMIX constants)
		id := uint32(rng>>33) % uint32(nBlobs)              //nolint:gosec // G115: nBlobs is a slice length (<< MaxUint32); modulo keeps id in range
		binary.LittleEndian.PutUint32(req, id)
		if hasDeadline {
			_ = conn.SetDeadline(deadline)
		}
		if _, err := conn.Write(req); err != nil {
			return
		}
		if _, err := io.ReadFull(conn, hdr); err != nil {
			return
		}
		h, err := decodeHeader(hdr)
		if err != nil || int(h.length) > len(buf) {
			return
		}
		if _, err := io.ReadFull(conn, buf[:h.length]); err != nil {
			return
		}
	}
}

// churn allocates and drops small objects to give the GC realistic work,
// modeling per-request metadata churn while blob bytes stay off-heap.
func churn(stop *atomic.Bool) {
	var sink [][]byte
	for !stop.Load() {
		for i := 0; i < 1000; i++ {
			sink = append(sink, make([]byte, 256))
		}
		if len(sink) > 100000 {
			sink = nil // drop → garbage for the GC to collect
		}
	}
}

// readGCPauses returns a deep copy of the GC pause histogram now, or nil if the
// metric is unavailable or not a histogram (defensive; go1.26 always has it).
func readGCPauses() *metrics.Float64Histogram {
	s := []metrics.Sample{{Name: gcPauseMetric}}
	metrics.Read(s)
	if s[0].Value.Kind() != metrics.KindFloat64Histogram {
		return nil
	}
	h := s[0].Value.Float64Histogram()
	// Deep-copy Counts (the runtime may reuse the backing array on next Read).
	counts := make([]uint64, len(h.Counts))
	copy(counts, h.Counts)
	return &metrics.Float64Histogram{Counts: counts, Buckets: h.Buckets}
}

// pausePercentiles diffs two histogram snapshots and returns p50/p99/p999/max
// (seconds) of pauses that occurred between them, plus the pause count. Returns
// zeros if the snapshots are missing, mis-shaped, or a counter went backwards
// (a histogram reset — never subtract into a uint64 underflow).
func pausePercentiles(before, after *metrics.Float64Histogram) (p50, p99, p999, max float64, total uint64) {
	if before == nil || after == nil ||
		len(before.Counts) != len(after.Counts) ||
		len(after.Buckets) != len(after.Counts)+1 {
		return 0, 0, 0, 0, 0
	}
	buckets := after.Buckets
	diff := make([]uint64, len(after.Counts))
	for i := range after.Counts {
		if after.Counts[i] < before.Counts[i] {
			return 0, 0, 0, 0, 0 // counter reset; bail rather than underflow
		}
		diff[i] = after.Counts[i] - before.Counts[i]
		total += diff[i]
	}
	if total == 0 {
		return 0, 0, 0, 0, 0
	}
	return pctile(diff, buckets, 0.50), pctile(diff, buckets, 0.99),
		pctile(diff, buckets, 0.999), maxBucket(diff, buckets), total
}

// maxBucket returns the upper bound of the highest bucket with any count — the
// observed worst-case pause (conservative; +Inf if it lands in the top bucket).
func maxBucket(counts []uint64, buckets []float64) float64 {
	for i := len(counts) - 1; i >= 0; i-- {
		if counts[i] > 0 {
			return buckets[i+1]
		}
	}
	return 0
}

// pctile returns the upper bound of the first bucket whose cumulative count
// reaches p*total — conservative for a kill-gate (never under-reports). A result
// in the top (+Inf) bucket returns +Inf, signalling "over any finite threshold".
func pctile(counts []uint64, buckets []float64, p float64) float64 {
	var total uint64
	for _, c := range counts {
		total += c
	}
	target := p * float64(total)
	var cum uint64
	for i := range counts { // len(buckets) == len(counts)+1
		cum += counts[i]
		if float64(cum) >= target {
			ub := buckets[i+1]
			if math.IsInf(ub, 1) {
				return math.Inf(1)
			}
			return ub
		}
	}
	return buckets[len(buckets)-1]
}

// maxRSSBytes returns peak resident set size in bytes (portable: ru_maxrss is
// bytes on darwin, kilobytes on linux — rssMul is set per-platform).
func maxRSSBytes() int64 {
	var ru unix.Rusage
	if err := unix.Getrusage(unix.RUSAGE_SELF, &ru); err != nil {
		return 0
	}
	return ru.Maxrss * rssMul
}
