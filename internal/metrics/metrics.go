// Package metrics is the daemon's observability surface: a dedicated
// Prometheus registry carrying the kvb_* instrument set, and the ops HTTP
// endpoint mounting /metrics, /healthz (503 until the daemon is ready), and
// /debug/pprof — the zero-alloc proof's capture point.
//
// The store stays decoupled: tier gauges are read at scrape time from the
// store's Stats() JSON document (the same document the wire STATS verb
// serves), so no store package ever imports Prometheus.
package metrics

import (
	"context"
	"encoding/json"
	"errors"
	"net"
	"net/http"
	"net/http/pprof"
	"strconv"
	"sync/atomic"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/collectors"
	"github.com/prometheus/client_golang/prometheus/promhttp"

	"github.com/kvstash/kvblockd/internal/protocol"
)

// Set owns the registry and every kvb_* instrument. Its recording methods
// satisfy the server's Recorder seam; nil-checking there means a daemon
// without metrics pays nothing.
type Set struct {
	reg   *prometheus.Registry
	ready atomic.Bool

	opSeconds *prometheus.HistogramVec
	hits      *prometheus.CounterVec
	misses    *prometheus.CounterVec
	bytes     *prometheus.CounterVec
	evictions prometheus.Counter
}

// New builds the registry. stats, when non-nil, is the store's Stats()
// document source — tier gauges (blocks, bytes, arena occupancy, pinned
// bytes) are decoded from it at scrape time.
func New(stats func() []byte) *Set {
	s := &Set{
		reg: prometheus.NewRegistry(),
		opSeconds: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Name: "kvb_op_seconds",
			Help: "Server-side request handling time by verb (decode + store + queue-to-writer; excludes the socket flush).",
			// Native histogram: sparse high-resolution buckets, no fixed schema.
			NativeHistogramBucketFactor:     1.1,
			NativeHistogramMaxBucketNumber:  160,
			NativeHistogramMinResetDuration: time.Hour,
		}, []string{"op"}),
		hits: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "kvb_hits_total",
			Help: "BATCH_GET per-key hits by tier and namespace id.",
		}, []string{"tier", "ns"}),
		misses: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "kvb_misses_total",
			Help: "BATCH_GET per-key misses by namespace id.",
		}, []string{"ns"}),
		bytes: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "kvb_bytes_total",
			Help: "Block payload bytes by direction (in = committed PUTs, out = served GETs) and namespace id.",
		}, []string{"dir", "ns"}),
		evictions: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "kvb_evictions_total",
			Help: "Blocks evicted under pressure (registered now; the evictor increments it).",
		}),
	}
	s.reg.MustRegister(s.opSeconds, s.hits, s.misses, s.bytes, s.evictions)
	s.reg.MustRegister(collectors.NewGoCollector()) // GC pause/heap — the launch-day GC defense reads these
	if stats != nil {
		s.reg.MustRegister(&storeCollector{stats: stats})
	}
	return s
}

// SetReady flips /healthz from 503 to 200. Call once the arena is prefaulted
// and the listener is accepting.
func (s *Set) SetReady() { s.ready.Store(true) }

// ---------------------------------------------------------------------------
// The server Recorder seam (structural — the server package never imports us).

// opNames is a fixed table: bounded label cardinality by construction.
var opNames = map[protocol.Opcode]string{
	protocol.OpNop:         "nop",
	protocol.OpHello:       "hello",
	protocol.OpBatchExists: "batch_exists",
	protocol.OpBatchGet:    "batch_get",
	protocol.OpPutStream:   "put_stream",
	protocol.OpTouchLease:  "touch_lease",
	protocol.OpPin:         "pin",
	protocol.OpDelete:      "delete",
	protocol.OpStats:       "stats",
}

func opName(op protocol.Opcode) string {
	if n, ok := opNames[op]; ok {
		return n
	}
	return "unknown"
}

// Op records one served request's handling time.
func (s *Set) Op(op protocol.Opcode, seconds float64) {
	s.opSeconds.WithLabelValues(opName(op)).Observe(seconds)
}

// GetResult records one BATCH_GET's per-key outcomes and payload bytes out.
// The tier label is fixed to "dram" for now: the Recorder seam carries no
// tier and neither does refGetter — when NVMe/S3 GETs land (ruling #4), the
// seam gains a tier argument or hits from those tiers would mislabel.
func (s *Set) GetResult(ns uint32, hits, misses, bytesOut int) {
	nsl := nsLabel(ns)
	if hits > 0 {
		s.hits.WithLabelValues("dram", nsl).Add(float64(hits))
	}
	if misses > 0 {
		s.misses.WithLabelValues(nsl).Add(float64(misses))
	}
	if bytesOut > 0 {
		s.bytes.WithLabelValues("out", nsl).Add(float64(bytesOut))
	}
}

// PutCommitted records one committed block's payload bytes in.
func (s *Set) PutCommitted(ns uint32, n int) {
	s.bytes.WithLabelValues("in", nsLabel(ns)).Add(float64(n))
}

func nsLabel(ns uint32) string { return strconv.FormatUint(uint64(ns), 10) }

// ---------------------------------------------------------------------------
// Scrape-time store gauges.

var (
	descBlocks = prometheus.NewDesc("kvb_blocks",
		"Committed blocks resident in the tier.", []string{"tier"}, nil)
	descStoreBytes = prometheus.NewDesc("kvb_store_bytes",
		"Committed block payload bytes resident in the tier.", []string{"tier"}, nil)
	descArenaBytes = prometheus.NewDesc("kvb_arena_bytes",
		"DRAM arena occupancy by state (total / free / largest_free_region).", []string{"state"}, nil)
	descPinned = prometheus.NewDesc("kvb_pinned_bytes",
		"Pinned bytes charged against the namespace's pin quota.", []string{"ns"}, nil)
)

// storeCollector decodes the store's Stats() JSON at scrape time. A document
// that fails to decode yields no samples (never a scrape error — the wire
// STATS verb is the debugging fallback).
type storeCollector struct {
	stats func() []byte
}

func (c *storeCollector) Describe(ch chan<- *prometheus.Desc) {
	ch <- descBlocks
	ch <- descStoreBytes
	ch <- descArenaBytes
	ch <- descPinned
}

func (c *storeCollector) Collect(ch chan<- prometheus.Metric) {
	var doc struct {
		Store            string             `json:"store"`
		Blocks           float64            `json:"blocks"`
		Bytes            float64            `json:"bytes"`
		ArenaBytes       float64            `json:"arena_bytes"`
		ArenaFreeBytes   float64            `json:"arena_free_bytes"`
		LargestFreeBytes float64            `json:"largest_free_region_bytes"`
		PinnedBytes      map[string]float64 `json:"pinned_bytes"`
	}
	if err := json.Unmarshal(c.stats(), &doc); err != nil {
		return
	}
	tier := doc.Store
	if tier == "" {
		tier = "unknown"
	}
	ch <- prometheus.MustNewConstMetric(descBlocks, prometheus.GaugeValue, doc.Blocks, tier)
	ch <- prometheus.MustNewConstMetric(descStoreBytes, prometheus.GaugeValue, doc.Bytes, tier)
	if doc.ArenaBytes > 0 {
		ch <- prometheus.MustNewConstMetric(descArenaBytes, prometheus.GaugeValue, doc.ArenaBytes, "total")
		ch <- prometheus.MustNewConstMetric(descArenaBytes, prometheus.GaugeValue, doc.ArenaFreeBytes, "free")
		ch <- prometheus.MustNewConstMetric(descArenaBytes, prometheus.GaugeValue, doc.LargestFreeBytes, "largest_free_region")
	}
	for ns, b := range doc.PinnedBytes {
		ch <- prometheus.MustNewConstMetric(descPinned, prometheus.GaugeValue, b, ns)
	}
}

// ---------------------------------------------------------------------------
// The ops HTTP endpoint.

// Handler mounts /metrics, /healthz, and /debug/pprof on a fresh mux (never
// http.DefaultServeMux — nothing else leaks onto the ops port).
func (s *Set) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.HandlerFor(s.reg, promhttp.HandlerOpts{}))
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		if !s.ready.Load() {
			http.Error(w, "starting", http.StatusServiceUnavailable)
			return
		}
		_, _ = w.Write([]byte("ok\n"))
	})
	mux.HandleFunc("/debug/pprof/", pprof.Index)
	mux.HandleFunc("/debug/pprof/cmdline", pprof.Cmdline)
	mux.HandleFunc("/debug/pprof/profile", pprof.Profile)
	mux.HandleFunc("/debug/pprof/symbol", pprof.Symbol)
	mux.HandleFunc("/debug/pprof/trace", pprof.Trace)
	return mux
}

// Serve runs the ops endpoint until ctx is cancelled, then shuts it down
// gracefully. The listener is bound synchronously (so ":0" callers can read
// the address) and serving continues in the background; the returned wait
// func BLOCKS until shutdown completes — shutdown itself is driven only by
// the ctx cancel, so cancel before waiting.
func (s *Set) Serve(ctx context.Context, addr string) (bound string, wait func(), err error) {
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return "", nil, err
	}
	srv := &http.Server{Handler: s.Handler(), ReadHeaderTimeout: 5 * time.Second}
	done := make(chan struct{})
	go func() {
		defer close(done)
		if serr := srv.Serve(ln); serr != nil && !errors.Is(serr, http.ErrServerClosed) {
			_ = serr // the ops port dying must never take the data plane with it
		}
	}()
	go func() { //nolint:gosec // G118: shutdown must outlive the cancelled ctx; the fresh timeout context is the point
		<-ctx.Done()
		shCtx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		_ = srv.Shutdown(shCtx)
	}()
	return ln.Addr().String(), func() { <-done }, nil
}
