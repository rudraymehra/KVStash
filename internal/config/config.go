// Package config loads and validates the kvblockd daemon configuration.
//
// Precedence, lowest to highest: built-in defaults → YAML file → environment
// (the fixed KVBLOCKD_* table) → command-line overrides. Validation never
// silently clamps: a config that violates a PROTOCOL.md §4 floor is an error
// the operator sees, not a surprise the peer negotiates around.
package config

import (
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"

	"github.com/kvstash/kvblockd/internal/protocol"
	"github.com/kvstash/kvblockd/internal/store/nvme"
)

// Config is the daemon configuration. Byte sizes are plain integers (no size
// suffix parsing — example.yaml carries the arithmetic in comments).
type Config struct {
	// ListenAddr is the data-plane TCP listener ("host:port").
	ListenAddr string `yaml:"listen_addr"`
	// MetricsAddr serves the ops endpoint (/metrics, /healthz, /debug/pprof);
	// empty disables it. AdminAddr serves the namespace/quota admin surface —
	// LOOPBACK ONLY (enforced at bind: shell trust, same boundary as editing
	// the namespaces file); empty disables it.
	AdminAddr   string `yaml:"admin_addr"`
	MetricsAddr string `yaml:"metrics_addr"`

	// MaxConns is a global accept cap — a cheap DoS floor. Per-tenant
	// connection accounting is the tenant package's job, not this field's.
	MaxConns int `yaml:"max_conns"`

	// PROTOCOL.md §4 negotiated-limit ceilings this server offers. Floors are
	// enforced by Validate against the protocol package's constants.
	MaxBatchKeys  uint32 `yaml:"max_batch_keys"`
	MaxFrameLen   uint32 `yaml:"max_frame_len"`
	MaxBlobLen    uint32 `yaml:"max_blob_len"`
	InitialCredit uint32 `yaml:"initial_credit"`

	// StreamTimeoutMS is the PUT_STREAM inactivity reaper (§5; floor 5000).
	StreamTimeoutMS uint32 `yaml:"stream_timeout_ms"`
	// LeaseDefaultMS / LeaseMaxMS are the §3.1 lease parameters.
	LeaseDefaultMS uint32 `yaml:"lease_default_ms"`
	LeaseMaxMS     uint32 `yaml:"lease_max_ms"`

	// NamespacesPath points at the namespaces/tokens YAML (server-side auth).
	NamespacesPath string `yaml:"namespaces_path"`

	// SockSndBuf / SockRcvBuf are per-connection socket buffer requests in
	// bytes. 16 MiB is the value the A1 rig saturated 50 GbE with; the kernel
	// silently clamps on untuned hosts (the transport logs the effective size).
	SockSndBuf int `yaml:"sock_sndbuf"`
	SockRcvBuf int `yaml:"sock_rcvbuf"`

	// WriteChunkBytes caps how many payload bytes one writev syscall covers
	// within a coalesced flush (0 = unchunked). ~1 MiB keeps the kernel
	// copy pipeline overlapped with the receiver; see transport.Config.
	WriteChunkBytes int `yaml:"write_chunk_bytes"`

	// DramArenaBytes sizes the DRAM tier's arena mapping (the hot-tier block
	// capacity). It must hold at least one max_blob_len block.
	DramArenaBytes int64 `yaml:"dram_arena_bytes"`
	// DramHugepages requests explicit hugepages for the arena (Linux
	// MAP_HUGETLB; falls back to THP-eligible mappings elsewhere — see
	// the dram arena docs).
	DramHugepages bool `yaml:"dram_hugepages"`
	// PinnedBytesCap caps per-namespace pinned bytes (§3.6 ERR_PIN_QUOTA);
	// 0 = unlimited.
	PinnedBytesCap int64 `yaml:"pinned_bytes_cap"`

	// EvictionPolicy selects the pressure policy: "s3fifo" (default),
	// "sampled-lru", or "none" (the Week-3 hard-wall behavior).
	EvictionPolicy string `yaml:"eviction_policy"`
	// EvictionWatermarkPct triggers an eviction batch at this arena
	// occupancy; EvictionBatchPct is how far below it one batch frees.
	EvictionWatermarkPct int `yaml:"eviction_watermark_pct"`
	EvictionBatchPct     int `yaml:"eviction_batch_pct"`
	// EvictionGhostEntries caps each tenant's S3-FIFO ghost ring;
	// 0 = auto (arena-derived: one fingerprint per conceivable block).
	EvictionGhostEntries int `yaml:"eviction_ghost_entries"`

	// NvmePaths lists the NVMe tier's volume directories (one per device is
	// the intended shape); empty = DRAM-only, byte-for-byte the pre-tiering
	// behavior. Blocks are placed across volumes by key hash.
	NvmePaths []string `yaml:"nvme_paths"`
	// NvmeMaxBytes is the total NVMe budget across volumes (split evenly);
	// REQUIRED > 0 when nvme_paths is set — reclaim needs a reference.
	NvmeMaxBytes int64 `yaml:"nvme_max_bytes"`
	// NvmeSegmentBytes is the fixed fallocated size of every segment file
	// (256 MiB default; [4 MiB, 4 GiB), multiple of 4096, and large enough
	// for one max_blob_len record plus the seal footer).
	NvmeSegmentBytes int64 `yaml:"nvme_segment_bytes"`
	// NvmeReadWorkers bounds each volume's device-read pool (saturation
	// answers per-key ERR_BUSY, never a hang).
	NvmeReadWorkers int `yaml:"nvme_read_workers"`
	// NvmeDemoteWatermarkPct triggers demotion at this DRAM occupancy —
	// strictly below eviction_watermark_pct so demotion normally beats
	// eviction to the punch; NvmeDemoteBatchPct is how far below it one
	// pass demotes.
	NvmeDemoteWatermarkPct int `yaml:"nvme_demote_watermark_pct"`
	NvmeDemoteBatchPct     int `yaml:"nvme_demote_batch_pct"`
	// NvmeAdmitMinHits: blocks with fewer than this many lifetime GETs are
	// DELETED at the demote watermark (90% by default — BEFORE the evictor's
	// 95%) rather than written to flash (SSD endurance). A pure-ingest
	// workload's never-read blocks melt under the default — set 0 to admit
	// everything, and watch kvb_nvme_admit_refusals_total either way. The
	// default (1) lives HERE; explicit 0 is honored end-to-end.
	NvmeAdmitMinHits uint32 `yaml:"nvme_admit_min_hits"`
	// NvmePromoteWindowMS: a 2nd NVMe hit within this window promotes the
	// block back to DRAM; 0 disables promotion.
	NvmePromoteWindowMS uint32 `yaml:"nvme_promote_window_ms"`
	// NvmeSyncEveryBytes is the group-commit cadence (fdatasync ledger).
	NvmeSyncEveryBytes int64 `yaml:"nvme_sync_every_bytes"`
	// S3 cold tier (inert until s3_bucket is set; needs the NVMe tier).
	// Credentials come from the ambient AWS chain (env/shared-config/IMDS) —
	// never from this file.
	S3Bucket           string `yaml:"s3_bucket"`
	S3Region           string `yaml:"s3_region"`
	S3NodeID           string `yaml:"s3_node_id"`           // namespaces object keys; REQUIRED with s3_bucket
	S3EndpointOverride string `yaml:"s3_endpoint_override"` // MinIO-compatible targets
	S3PathStyle        bool   `yaml:"s3_path_style"`
	S3SpillQueue       int    `yaml:"s3_spill_queue"`     // bounded write-back depth (default 8; explicit 0 also means 8 — a zero-depth queue is not a configuration)
	S3ReadTimeoutMS    uint32 `yaml:"s3_read_timeout_ms"` // cold ranged-GET deadline (default 2000; explicit 0 also means 2000 — an unbounded cold read is not a configuration)

	// NvmeCkptEverySegments writes an index checkpoint every N seals
	// (0 = never; recovery falls back to footer scans).
	NvmeCkptEverySegments int `yaml:"nvme_ckpt_every_segments"`
}

// Overrides are the command-line flags an operator actually needs at launch;
// nil pointer = not set. Deliberately short.
type Overrides struct {
	ListenAddr     *string
	MetricsAddr    *string
	NamespacesPath *string
	MaxConns       *int
}

// Default returns the built-in defaults (PROTOCOL.md §4 defaults for the
// wire-visible limits).
func Default() Config {
	return Config{
		ListenAddr:      ":9440",
		AdminAddr:       "127.0.0.1:9441",
		MetricsAddr:     "127.0.0.1:9442",
		MaxConns:        1024,
		MaxBatchKeys:    protocol.DefaultMaxBatchKeys,
		MaxFrameLen:     protocol.DefaultMaxFrameLen,
		MaxBlobLen:      protocol.DefaultMaxBlobLen,
		InitialCredit:   protocol.DefaultInitialCredit,
		StreamTimeoutMS: protocol.DefaultStreamTimeoutMS,
		LeaseDefaultMS:  protocol.DefaultLeaseMS,
		LeaseMaxMS:      protocol.MaxLeaseMS,
		SockSndBuf:      16 << 20,
		SockRcvBuf:      16 << 20,
		WriteChunkBytes: 1 << 20,
		DramArenaBytes:  1 << 30,   // 1 GiB hot tier
		PinnedBytesCap:  128 << 20, // 128 MiB per namespace

		EvictionPolicy:       "s3fifo",
		EvictionWatermarkPct: 95, // Mooncake numbers: evict at 95%…
		EvictionBatchPct:     5,  // …freeing 5% per batch

		// NVMe tier defaults (inert until nvme_paths is set).
		NvmePaths:              []string{}, // non-nil so `nvme_paths: []` in YAML round-trips to the default
		NvmeSegmentBytes:       256 << 20,  // CacheLib-Navy-informed geometry
		NvmeReadWorkers:        16,
		NvmeDemoteWatermarkPct: 90,
		NvmeDemoteBatchPct:     5,
		NvmeAdmitMinHits:       1,
		NvmePromoteWindowMS:    60000,
		NvmeSyncEveryBytes:     8 << 20,
		NvmeCkptEverySegments:  8,

		// S3 tier defaults (inert until s3_bucket is set). Visible HERE, not
		// in a downstream clamp — the AdmitMinHits lesson.
		S3SpillQueue:    8,
		S3ReadTimeoutMS: 2000,
	}
}

// envTable is the fixed environment layer: variable → setter. One loop, no
// reflection, no surprises.
var envTable = []struct {
	name string
	set  func(*Config, string) error
}{
	{"KVBLOCKD_LISTEN_ADDR", func(c *Config, v string) error { c.ListenAddr = v; return nil }},
	{"KVBLOCKD_METRICS_ADDR", func(c *Config, v string) error { c.MetricsAddr = v; return nil }},
	{"KVBLOCKD_NAMESPACES", func(c *Config, v string) error { c.NamespacesPath = v; return nil }},
	{"KVBLOCKD_MAX_CONNS", func(c *Config, v string) error {
		n, err := strconv.Atoi(v)
		if err != nil {
			return fmt.Errorf("KVBLOCKD_MAX_CONNS: %w", err)
		}
		c.MaxConns = n
		return nil
	}},
	{"KVBLOCKD_DRAM_ARENA_BYTES", func(c *Config, v string) error {
		n, err := strconv.ParseInt(v, 10, 64)
		if err != nil {
			return fmt.Errorf("KVBLOCKD_DRAM_ARENA_BYTES: %w", err)
		}
		c.DramArenaBytes = n
		return nil
	}},
	{"KVBLOCKD_DRAM_HUGEPAGES", func(c *Config, v string) error {
		b, err := strconv.ParseBool(v)
		if err != nil {
			return fmt.Errorf("KVBLOCKD_DRAM_HUGEPAGES: %w", err)
		}
		c.DramHugepages = b
		return nil
	}},
	{"KVBLOCKD_PINNED_BYTES_CAP", func(c *Config, v string) error {
		n, err := strconv.ParseInt(v, 10, 64)
		if err != nil {
			return fmt.Errorf("KVBLOCKD_PINNED_BYTES_CAP: %w", err)
		}
		c.PinnedBytesCap = n
		return nil
	}},
	{"KVBLOCKD_NVME_PATHS", func(c *Config, v string) error {
		c.NvmePaths = c.NvmePaths[:0]
		for _, p := range strings.Split(v, ",") {
			if p = strings.TrimSpace(p); p != "" {
				c.NvmePaths = append(c.NvmePaths, p)
			}
		}
		return nil
	}},
	{"KVBLOCKD_NVME_MAX_BYTES", func(c *Config, v string) error {
		n, err := strconv.ParseInt(v, 10, 64)
		if err != nil {
			return fmt.Errorf("KVBLOCKD_NVME_MAX_BYTES: %w", err)
		}
		c.NvmeMaxBytes = n
		return nil
	}},
}

// Load builds the effective configuration. path == "" means "no file" (the
// daemon runs on defaults); a non-empty path that cannot be read or parsed is
// an error — an explicitly configured file must exist. The returned Config is
// already validated.
func Load(path string, ov Overrides) (Config, error) {
	c := Default()

	if path != "" {
		f, err := os.Open(path) //nolint:gosec // G304: path is the operator's own --config flag
		if err != nil {
			return Config{}, fmt.Errorf("config: %w", err)
		}
		defer f.Close()
		dec := yaml.NewDecoder(f)
		dec.KnownFields(true) // a typo'd key fails loudly, never silently defaults
		// An empty or comments-only file decodes to io.EOF: that is the
		// documented "empty file yields the defaults" case, not an error.
		if err := dec.Decode(&c); err != nil && !errors.Is(err, io.EOF) {
			return Config{}, fmt.Errorf("config %s: %w", path, err)
		}
	}

	for _, e := range envTable {
		if v, ok := os.LookupEnv(e.name); ok {
			if err := e.set(&c, v); err != nil {
				return Config{}, err
			}
		}
	}

	if ov.ListenAddr != nil {
		c.ListenAddr = *ov.ListenAddr
	}
	if ov.MetricsAddr != nil {
		c.MetricsAddr = *ov.MetricsAddr
	}
	if ov.NamespacesPath != nil {
		c.NamespacesPath = *ov.NamespacesPath
	}
	if ov.MaxConns != nil {
		c.MaxConns = *ov.MaxConns
	}

	if err := c.Validate(); err != nil {
		return Config{}, fmt.Errorf("config: %w", err)
	}
	return c, nil
}

// Validate checks the configuration against the PROTOCOL.md floors and basic
// sanity. It mutates nothing: a wrong config is an error, not a clamp.
func (c Config) Validate() error {
	var errs []error
	check := func(cond bool, format string, args ...any) {
		if !cond {
			errs = append(errs, fmt.Errorf(format, args...))
		}
	}

	check(c.ListenAddr != "", "listen_addr: the data-plane address must be set")
	for _, a := range []struct{ name, addr string }{
		{"listen_addr", c.ListenAddr},
		{"admin_addr", c.AdminAddr},
		{"metrics_addr", c.MetricsAddr},
	} {
		if a.addr == "" {
			continue
		}
		if _, err := net.ResolveTCPAddr("tcp", a.addr); err != nil {
			errs = append(errs, fmt.Errorf("%s %q: %w", a.name, a.addr, err))
		}
	}

	check(c.MaxConns >= 1, "max_conns %d: must be >= 1", c.MaxConns)
	check(c.MaxBatchKeys >= protocol.FloorMaxBatchKeys,
		"max_batch_keys %d: below the §4 floor %d", c.MaxBatchKeys, protocol.FloorMaxBatchKeys)
	check(c.MaxFrameLen >= protocol.FloorMaxFrameLen,
		"max_frame_len %d: below the §4 floor %d", c.MaxFrameLen, protocol.FloorMaxFrameLen)
	check(c.MaxBlobLen >= protocol.FloorMaxBlobLen,
		"max_blob_len %d: below the §4 floor %d", c.MaxBlobLen, protocol.FloorMaxBlobLen)
	check(c.InitialCredit >= protocol.FloorInitialCredit,
		"initial_credit %d: below the §4 floor %d", c.InitialCredit, protocol.FloorInitialCredit)
	check(c.MaxBlobLen <= c.MaxFrameLen,
		"max_blob_len %d exceeds max_frame_len %d", c.MaxBlobLen, c.MaxFrameLen)
	check(c.StreamTimeoutMS >= 5000,
		"stream_timeout_ms %d: §5 says not negotiable below 5s", c.StreamTimeoutMS)
	check(c.LeaseDefaultMS >= 1 && c.LeaseDefaultMS <= c.LeaseMaxMS,
		"lease_default_ms %d: must be in [1, lease_max_ms %d]", c.LeaseDefaultMS, c.LeaseMaxMS)
	check(c.LeaseMaxMS <= protocol.MaxLeaseMS,
		"lease_max_ms %d: exceeds the protocol clamp %d", c.LeaseMaxMS, protocol.MaxLeaseMS)
	check(c.SockSndBuf >= 0, "sock_sndbuf %d: must be >= 0 (0 = OS default)", c.SockSndBuf)
	check(c.SockRcvBuf >= 0, "sock_rcvbuf %d: must be >= 0 (0 = OS default)", c.SockRcvBuf)
	check(c.WriteChunkBytes >= 0, "write_chunk_bytes %d: must be >= 0 (0 = unchunked)", c.WriteChunkBytes)
	check(c.DramArenaBytes > 0, "dram_arena_bytes %d: must be > 0", c.DramArenaBytes)
	check(c.DramArenaBytes < 1<<44,
		"dram_arena_bytes %d: must be below 16 TiB (the allocator's uint32 granule addressing)", c.DramArenaBytes)
	check(c.DramArenaBytes >= int64(c.MaxBlobLen),
		"dram_arena_bytes %d: must hold at least one max_blob_len (%d) block", c.DramArenaBytes, c.MaxBlobLen)
	check(c.PinnedBytesCap >= 0 && c.PinnedBytesCap <= c.DramArenaBytes,
		"pinned_bytes_cap %d: must be in [0, dram_arena_bytes %d]", c.PinnedBytesCap, c.DramArenaBytes)
	switch c.EvictionPolicy {
	case "s3fifo", "sampled-lru", "none":
	default:
		errs = append(errs, fmt.Errorf("eviction_policy %q: want s3fifo, sampled-lru, or none", c.EvictionPolicy))
	}
	check(c.EvictionWatermarkPct >= 50 && c.EvictionWatermarkPct <= 99,
		"eviction_watermark_pct %d: must be in [50, 99]", c.EvictionWatermarkPct)
	check(c.EvictionBatchPct >= 1 && c.EvictionBatchPct < c.EvictionWatermarkPct,
		"eviction_batch_pct %d: must be in [1, eviction_watermark_pct %d)", c.EvictionBatchPct, c.EvictionWatermarkPct)
	check(c.EvictionGhostEntries >= 0, "eviction_ghost_entries %d: must be >= 0 (0 = auto)", c.EvictionGhostEntries)

	if c.S3Bucket != "" {
		check(len(c.NvmePaths) > 0,
			"s3_bucket set but nvme_paths empty — the cold tier spills SEALED NVMe segments")
		check(c.S3NodeID != "",
			"s3_bucket set but s3_node_id empty — object keys must be node-namespaced")
	}
	if len(c.NvmePaths) > 0 {
		// Paths are compared NORMALIZED: "/a" vs "/a/" (or "./x" vs "x")
		// naming the same directory would put two volumes on one segment
		// log — interleaved appends destroy the data (ladder finding).
		// Nested paths are rejected for the same reason. Symlink aliases
		// remain the operator's responsibility (documented; paths may not
		// exist yet at validation time). NOTE also that key→volume
		// placement is positional: reordering or removing entries reroutes
		// every key (mass invalidation) — append only.
		cleaned := make([]string, 0, len(c.NvmePaths))
		seen := map[string]bool{}
		for _, p := range c.NvmePaths {
			check(p != "", "nvme_paths: empty path entry")
			cp := filepath.Clean(p)
			check(!seen[cp], "nvme_paths: duplicate path %q (normalized %q)", p, cp)
			seen[cp] = true
			cleaned = append(cleaned, cp)
		}
		for i := range cleaned {
			for j := range cleaned {
				if i != j && strings.HasPrefix(cleaned[i], cleaned[j]+string(filepath.Separator)) {
					errs = append(errs, fmt.Errorf("nvme_paths: %q is nested inside %q — volumes must not overlap", cleaned[i], cleaned[j]))
				}
			}
		}
		nVols := int64(len(c.NvmePaths))
		// The EXACT bound OpenVolume enforces — the ladder caught the two
		// layers computing slightly different minima (a config in the gap
		// validated, then failed at startup).
		minSeg := nvme.MinSegmentBytes(c.MaxBlobLen)
		check(c.NvmeSegmentBytes >= minSeg && c.NvmeSegmentBytes%4096 == 0 && c.NvmeSegmentBytes < int64(^uint32(0)),
			"nvme_segment_bytes %d: must be a 4096-multiple in [%d (one max_blob_len record + footer), 4 GiB)",
			c.NvmeSegmentBytes, minSeg)
		check(c.NvmeMaxBytes > 2*c.NvmeSegmentBytes*nVols,
			"nvme_max_bytes %d: must exceed 2 segments per volume (%d) — reclaim needs headroom",
			c.NvmeMaxBytes, 2*c.NvmeSegmentBytes*nVols)
		check(c.NvmeReadWorkers >= 1 && c.NvmeReadWorkers <= 1024,
			"nvme_read_workers %d: must be in [1, 1024]", c.NvmeReadWorkers)
		check(c.NvmeDemoteWatermarkPct >= 50 && c.NvmeDemoteWatermarkPct < c.EvictionWatermarkPct,
			"nvme_demote_watermark_pct %d: must be in [50, eviction_watermark_pct %d) — demotion runs below eviction",
			c.NvmeDemoteWatermarkPct, c.EvictionWatermarkPct)
		check(c.NvmeDemoteBatchPct >= 1 && c.NvmeDemoteBatchPct < c.NvmeDemoteWatermarkPct,
			"nvme_demote_batch_pct %d: must be in [1, nvme_demote_watermark_pct %d)",
			c.NvmeDemoteBatchPct, c.NvmeDemoteWatermarkPct)
		check(c.NvmeSyncEveryBytes > 0,
			"nvme_sync_every_bytes %d: must be > 0 (the group-commit ledger)", c.NvmeSyncEveryBytes)
		check(c.NvmeCkptEverySegments >= 0,
			"nvme_ckpt_every_segments %d: must be >= 0 (0 = never)", c.NvmeCkptEverySegments)
		check(c.EvictionPolicy != "none",
			"nvme_paths set with eviction_policy \"none\": the demoter needs a policy to nominate victims")
	}

	return errors.Join(errs...)
}

// WireLimits is the config→protocol bridge: the §4 ceilings this server
// offers at HELLO negotiation.
func (c Config) WireLimits() protocol.Limits {
	return protocol.Limits{
		MaxBatchKeys:  c.MaxBatchKeys,
		MaxFrameLen:   c.MaxFrameLen,
		MaxBlobLen:    c.MaxBlobLen,
		InitialCredit: c.InitialCredit,
	}
}
