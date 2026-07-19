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
	"github.com/kvstash/kvblockd/internal/tenant"
)

// Set owns the registry and every kvb_* instrument. Its recording methods
// satisfy the server's Recorder seam; nil-checking there means a daemon
// without metrics pays nothing.
type Set struct {
	reg   *prometheus.Registry
	ready atomic.Bool

	// Per-tenant quota view (SetTenants); nil = single-tenant, no series.
	tenants *tenantView

	opSeconds *prometheus.HistogramVec
	hits      *prometheus.CounterVec
	misses    *prometheus.CounterVec
	bytes     *prometheus.CounterVec
	getBusy   *prometheus.CounterVec
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
		getBusy: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "kvb_get_busy_total",
			Help: "BATCH_GET per-key ERR_BUSY descriptors (NVMe reader saturation; retryable).",
		}, []string{"ns"}),
	}
	s.reg.MustRegister(s.opSeconds, s.hits, s.misses, s.bytes, s.getBusy)
	s.reg.MustRegister(collectors.NewGoCollector()) // GC pause/heap — the launch-day GC defense reads these
	// process_cpu_seconds_total / process_resident_memory_bytes — the ONLY
	// cross-host way a benchmark client (on node A) reads the daemon's CPU
	// (node B). client_golang emits these on both Linux and darwin.
	s.reg.MustRegister(collectors.NewProcessCollector(collectors.ProcessCollectorOpts{}))
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

// GetResult records one tier's share of a BATCH_GET's per-key outcomes and
// payload bytes out (the seam gained the tier argument when NVMe GETs
// landed — hits from that tier no longer mislabel as "dram").
func (s *Set) GetResult(ns uint32, tier string, hits, misses, bytesOut int) {
	nsl := nsLabel(ns)
	if hits > 0 {
		s.hits.WithLabelValues(tier, nsl).Add(float64(hits))
	}
	if misses > 0 {
		s.misses.WithLabelValues(nsl).Add(float64(misses))
	}
	if bytesOut > 0 {
		s.bytes.WithLabelValues("out", nsl).Add(float64(bytesOut))
	}
}

// GetBusy counts per-key ERR_BUSY GET descriptors (NVMe reader saturation —
// the retryable backpressure signal).
func (s *Set) GetBusy(ns uint32, n int) {
	if n > 0 {
		s.getBusy.WithLabelValues(nsLabel(ns)).Add(float64(n))
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
	// Evictions live in the store (which never imports Prometheus), so the
	// counter is scrape-time const, fed from the Stats JSON like the gauges.
	descEvictions = prometheus.NewDesc("kvb_evictions_total",
		"Blocks evicted under memory pressure.", []string{"tier"}, nil)
	descLiveAllocs = prometheus.NewDesc("kvb_live_allocs",
		"Live arena extents (the allocator node-pool watermark numerator).", []string{"tier"}, nil)
	descMaxAllocs = prometheus.NewDesc("kvb_max_allocs",
		"Allocator node-pool capacity (live extents ceiling).", []string{"tier"}, nil)

	// NVMe tier scrape-time metrics, fed from the "nvme" sub-document a
	// tiered store adds to its Stats() JSON. A DRAM-only store has no such
	// sub-document and emits none of these — scrape output is byte-identical
	// to the pre-tiering daemon.
	descNvmeSegments = prometheus.NewDesc("kvb_nvme_segments",
		"Live NVMe segment files across volumes.", nil, nil)
	descNvmeUsed = prometheus.NewDesc("kvb_nvme_used_bytes",
		"Fallocated bytes across live NVMe segments.", nil, nil)
	descNvmeMax = prometheus.NewDesc("kvb_nvme_max_bytes",
		"Configured NVMe capacity across volumes.", nil, nil)
	descNvmeDemotions = prometheus.NewDesc("kvb_nvme_demotions_total",
		"Blocks demoted DRAM→NVMe (bytes written and index moved).", nil, nil)
	descNvmeDemoteDrops = prometheus.NewDesc("kvb_nvme_demote_drops_total",
		"Demotions dropped (queue full, write failure, or publish-gate refusal). Endurance-gate deletions are counted separately in kvb_nvme_admit_refusals_total.", nil, nil)
	descNvmeAdmitRefusals = prometheus.NewDesc("kvb_nvme_admit_refusals_total",
		"Blocks with fewer than nvme_admit_min_hits lifetime GETs, DELETED at the demote watermark instead of written to flash (SSD endurance).", nil, nil)
	descNvmeDedup = prometheus.NewDesc("kvb_nvme_dedup_skips_total",
		"Demotions completed WITHOUT an SSD write (bytes already resident).", nil, nil)
	descNvmePromotions = prometheus.NewDesc("kvb_nvme_promotions_total",
		"Blocks promoted NVMe→DRAM (2nd hit in window or hard pin).", nil, nil)
	descNvmeReclaims = prometheus.NewDesc("kvb_nvme_reclaims_total",
		"Whole segments reclaimed (FIFO).", nil, nil)
	descNvmeReclaimSkips = prometheus.NewDesc("kvb_nvme_reclaim_skips_total",
		"Reclaim rounds that skipped a segment holding protected blocks.", nil, nil)
	descNvmeReadBusy = prometheus.NewDesc("kvb_nvme_read_busy_total",
		"Device reads refused by reader-pool saturation (per-key ERR_BUSY).", nil, nil)
	descNvmeChecksumErrs = prometheus.NewDesc("kvb_nvme_checksum_errors_total",
		"Device reads that failed verification and self-healed the index (never served).", nil, nil)
	descNvmeRecovered = prometheus.NewDesc("kvb_nvme_recovered_blocks",
		"Blocks recovered at the last startup (checkpoint + footer + tail scan).", nil, nil)
	descNvmeRecoverySecs = prometheus.NewDesc("kvb_nvme_recovery_seconds",
		"Wall time of the last startup recovery across volumes.", nil, nil)
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
	ch <- descEvictions
	ch <- descLiveAllocs
	ch <- descMaxAllocs
	ch <- descNvmeSegments
	ch <- descNvmeUsed
	ch <- descNvmeMax
	ch <- descNvmeDemotions
	ch <- descNvmeDemoteDrops
	ch <- descNvmeAdmitRefusals
	ch <- descNvmeDedup
	ch <- descNvmePromotions
	ch <- descNvmeReclaims
	ch <- descNvmeReclaimSkips
	ch <- descNvmeReadBusy
	ch <- descNvmeChecksumErrs
	ch <- descNvmeRecovered
	ch <- descNvmeRecoverySecs
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
		EvictionsTotal   float64            `json:"evictions_total"`
		LiveAllocs       float64            `json:"live_allocs"`
		MaxAllocs        float64            `json:"max_allocs"`
		// Present only on a tiered store (nil otherwise — a DRAM-only
		// scrape stays byte-identical).
		Nvme *struct {
			Blocks        float64 `json:"blocks"`
			Bytes         float64 `json:"bytes"`
			Segments      float64 `json:"segments"`
			UsedBytes     float64 `json:"used_bytes"`
			MaxBytes      float64 `json:"max_bytes"`
			Demotions     float64 `json:"demotions_total"`
			DemoteDrops   float64 `json:"demote_drops_total"`
			AdmitRefusals float64 `json:"admit_refusals_total"`
			DedupSkips    float64 `json:"dedup_skips_total"`
			Promotions    float64 `json:"promotions_total"`
			Reclaims      float64 `json:"reclaims_total"`
			ReclaimSkips  float64 `json:"reclaim_skips_total"`
			ReadBusy      float64 `json:"read_busy_total"`
			ChecksumErrs  float64 `json:"checksum_errors_total"`
			RecoveredBlks float64 `json:"recovered_blocks"`
			RecoverySecs  float64 `json:"recovery_seconds"`
		} `json:"nvme"`
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
	ch <- prometheus.MustNewConstMetric(descEvictions, prometheus.CounterValue, doc.EvictionsTotal, tier)
	if doc.MaxAllocs > 0 {
		ch <- prometheus.MustNewConstMetric(descLiveAllocs, prometheus.GaugeValue, doc.LiveAllocs, tier)
		ch <- prometheus.MustNewConstMetric(descMaxAllocs, prometheus.GaugeValue, doc.MaxAllocs, tier)
	}
	if nv := doc.Nvme; nv != nil {
		ch <- prometheus.MustNewConstMetric(descBlocks, prometheus.GaugeValue, nv.Blocks, "nvme")
		ch <- prometheus.MustNewConstMetric(descStoreBytes, prometheus.GaugeValue, nv.Bytes, "nvme")
		ch <- prometheus.MustNewConstMetric(descNvmeSegments, prometheus.GaugeValue, nv.Segments)
		ch <- prometheus.MustNewConstMetric(descNvmeUsed, prometheus.GaugeValue, nv.UsedBytes)
		ch <- prometheus.MustNewConstMetric(descNvmeMax, prometheus.GaugeValue, nv.MaxBytes)
		ch <- prometheus.MustNewConstMetric(descNvmeDemotions, prometheus.CounterValue, nv.Demotions)
		ch <- prometheus.MustNewConstMetric(descNvmeDemoteDrops, prometheus.CounterValue, nv.DemoteDrops)
		ch <- prometheus.MustNewConstMetric(descNvmeAdmitRefusals, prometheus.CounterValue, nv.AdmitRefusals)
		ch <- prometheus.MustNewConstMetric(descNvmeDedup, prometheus.CounterValue, nv.DedupSkips)
		ch <- prometheus.MustNewConstMetric(descNvmePromotions, prometheus.CounterValue, nv.Promotions)
		ch <- prometheus.MustNewConstMetric(descNvmeReclaims, prometheus.CounterValue, nv.Reclaims)
		ch <- prometheus.MustNewConstMetric(descNvmeReclaimSkips, prometheus.CounterValue, nv.ReclaimSkips)
		ch <- prometheus.MustNewConstMetric(descNvmeReadBusy, prometheus.CounterValue, nv.ReadBusy)
		ch <- prometheus.MustNewConstMetric(descNvmeChecksumErrs, prometheus.CounterValue, nv.ChecksumErrs)
		ch <- prometheus.MustNewConstMetric(descNvmeRecovered, prometheus.GaugeValue, nv.RecoveredBlks)
		ch <- prometheus.MustNewConstMetric(descNvmeRecoverySecs, prometheus.GaugeValue, nv.RecoverySecs)
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

// ---------------------------------------------------------------------------
// Per-tenant quota series (Week-9 tenancy).

var (
	descTenantBytes = prometheus.NewDesc("kvb_tenant_bytes",
		"Resident bytes charged to a tenant's tier quota.", []string{"namespace", "tier"}, nil)
	descTenantQuota = prometheus.NewDesc("kvb_tenant_quota_bytes",
		"Configured tier quota for a tenant (0 = unlimited).", []string{"namespace", "tier"}, nil)
)

// tenantView reads the registry + accountant at scrape time. Cardinality is
// #namespaces × 3 tiers — namespaces are operator-registered (tens, never
// per-key), and the smoke test pins it.
type tenantView struct {
	reg *tenant.Registry
	q   *tenant.Quotas
}

// SetTenants attaches the per-tenant quota series. Call once before Serve.
func (s *Set) SetTenants(reg *tenant.Registry, q *tenant.Quotas) {
	if reg == nil || q == nil {
		return
	}
	s.tenants = &tenantView{reg: reg, q: q}
	s.reg.MustRegister(s.tenants)
}

func (v *tenantView) Describe(ch chan<- *prometheus.Desc) {
	ch <- descTenantBytes
	ch <- descTenantQuota
}

func (v *tenantView) Collect(ch chan<- prometheus.Metric) {
	tiers := []tenant.Tier{tenant.TierDRAM, tenant.TierNVMe, tenant.TierS3}
	v.reg.Each(func(ns *tenant.Namespace) {
		for _, t := range tiers {
			ch <- prometheus.MustNewConstMetric(descTenantBytes, prometheus.GaugeValue,
				float64(v.q.Usage(ns.ID, t)), ns.Name, t.String())
			ch <- prometheus.MustNewConstMetric(descTenantQuota, prometheus.GaugeValue,
				float64(v.q.Limit(ns.ID, t)), ns.Name, t.String())
		}
	})
}
