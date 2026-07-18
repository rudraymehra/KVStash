package stats

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"runtime"
	"runtime/debug"
	"sync"
	"time"
)

// CellRecord is one JSONL line — the unit every chart and table is built
// from. Conventions follow the repo's precedents (xferspike's result
// struct, soak's interval records, BENCHMARKS.md): snake_case, explicit
// units in names, DECIMAL GB/s (bytes/1e9), payload-only goodput, and the
// ratio-vs-same-rig-ceiling as the quotable number.
type CellRecord struct {
	SchemaVersion int    `json:"schema_version"` // 1
	Kind          string `json:"kind"`           // sweep | replay | fill | verify
	TS            string `json:"ts"`             // RFC3339 UTC
	GitSHA        string `json:"git_sha,omitempty"`
	Rig           string `json:"rig,omitempty"`
	Store         string `json:"store"` // kvblockd | redis | valkey | nvmefs | mem
	GOOS          string `json:"goos"`  // zipf float determinism is per-platform — provenance
	GOARCH        string `json:"goarch"`
	Seed          uint64 `json:"seed"`

	Cell CellMeta `json:"cell"`

	WarmupS   float64 `json:"warmup_s"`
	DurationS float64 `json:"duration_s"` // measured window

	Ops map[string]OpSummary `json:"ops"` // keyed by Op.String()

	OpsPerS           float64 `json:"ops_per_s"`
	GoodputGBytesS    float64 `json:"goodput_gbytes_s"` // payload bytes only, decimal
	CeilingGBytesS    float64 `json:"ceiling_gbytes_s,omitempty"`
	RatioVsCeiling    float64 `json:"ratio_vs_ceiling,omitempty"`
	ClosedCeilingOpsS float64 `json:"closed_ceiling_ops_s,omitempty"` // the rate sweep's denominator

	CPU   CPUSample  `json:"cpu"`
	Sched SchedStats `json:"sched,omitzero"`

	HitRate      float64  `json:"hit_rate,omitempty"` // replay OUTPUT — never an input
	ErrorsTotal  int64    `json:"errors_total"`
	VerifyFails  int64    `json:"verify_fails"`
	HgrmPaths    []string `json:"hgrm_paths,omitempty"`
	SaturatedRun bool     `json:"saturated,omitempty"` // excluded from repeatability checks
}

// CellMeta is the cell's identity + provenance knobs.
type CellMeta struct {
	ID            string  `json:"id"`
	BlobBytes     int     `json:"blob_bytes"`
	BatchKeys     int     `json:"batch_keys"`
	Streams       int     `json:"streams"`
	Mix           string  `json:"mix"`
	Skew          string  `json:"skew"`
	Mode          string  `json:"mode"` // closed | open
	RateOpsS      float64 `json:"rate_ops_s,omitempty"`
	RateFrac      float64 `json:"rate_frac_of_ceiling,omitempty"`
	CapacityBytes int64   `json:"capacity_bytes,omitempty"` // replay provenance
	Policy        string  `json:"policy,omitempty"`
	Trace         string  `json:"trace,omitempty"`
	SpeedupF      float64 `json:"speedup,omitempty"`
}

// SchedStats is the open-loop scheduler's own health (the coordinated-
// omission bookkeeping made visible).
type SchedStats struct {
	MaxLagUs  float64 `json:"max_lag_us,omitempty"`
	P99LagUs  float64 `json:"p99_lag_us,omitempty"`
	Saturated bool    `json:"saturated,omitempty"`
}

// NewCellRecord stamps the invariant provenance fields, including the VCS
// revision from the build (git_sha) so every result line is traceable to a
// commit — the schema documents it, so it must be populated.
func NewCellRecord(kind, store string, seed uint64) *CellRecord {
	return &CellRecord{
		SchemaVersion: 1,
		Kind:          kind,
		TS:            time.Now().UTC().Format(time.RFC3339),
		GitSHA:        gitSHA(),
		Store:         store,
		GOOS:          runtime.GOOS,
		GOARCH:        runtime.GOARCH,
		Seed:          seed,
	}
}

var (
	gitOnce sync.Once
	gitRev  string
)

func gitSHA() string {
	gitOnce.Do(func() {
		if info, ok := debug.ReadBuildInfo(); ok {
			for _, s := range info.Settings {
				if s.Key == "vcs.revision" {
					gitRev = s.Value
					if len(gitRev) > 12 {
						gitRev = gitRev[:12]
					}
					return
				}
			}
		}
	})
	return gitRev
}

// Writer emits one JSON object per line. Path "" writes to stdout.
type Writer struct {
	f   *os.File
	enc *json.Encoder
	own bool
}

// NewWriter opens (appending) the JSONL sink.
func NewWriter(path string) (*Writer, error) {
	if path == "" {
		return &Writer{f: os.Stdout, enc: json.NewEncoder(os.Stdout)}, nil
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644) //nolint:gosec // G304/G302: operator-chosen results path, non-secret data
	if err != nil {
		return nil, fmt.Errorf("stats: open jsonl %s: %w", path, err)
	}
	return &Writer{f: f, enc: json.NewEncoder(f), own: true}, nil
}

// Write emits one record.
func (w *Writer) Write(rec *CellRecord) error { return w.enc.Encode(rec) }

// Close closes an owned file (stdout is left alone).
func (w *Writer) Close() error {
	if w.own {
		return w.f.Close()
	}
	return nil
}

var _ io.Closer = (*Writer)(nil)
