// Package stats owns kvbench's measurement pipeline: HDR histograms per
// (op, cell), the .hgrm export, the JSONL cell records, and CPU accounting
// for the client and the daemon under test.
//
// Histogram choice: github.com/HdrHistogram/hdrhistogram-go (bench-only
// dep). The in-repo soak histogram is 64 log buckets at factor 1.25 — up to
// +25% quantization on a reported percentile, which cannot meet the
// benchmark spec's "open-loop p99 repeatable within 2%" acceptance gate or
// its 3-significant-figure .hgrm export. Recording is per-worker private
// histograms (RecordValue is not goroutine-safe) merged at cell end.
package stats

import (
	"fmt"
	"io"
	"time"

	hdrhistogram "github.com/HdrHistogram/hdrhistogram-go"
)

// Op indexes the recorded operation classes.
type Op uint8

const (
	OpGet Op = iota
	OpPut
	OpExists
	// OpLag is the open-loop scheduler's dispatch lateness (actual start −
	// scheduled start): the queue-growth signal that separates "the store
	// is slow" from "the harness fell behind".
	OpLag
	opCount
)

var opNames = [opCount]string{"get", "put", "exists", "lag"}

func (o Op) String() string { return opNames[o] }

// Histogram bounds: 1µs .. 60s at 3 significant figures (SPEC-4's
// hdrhistogram.New(1_000, 60_000_000_000, 3), values in nanoseconds).
const (
	histMinNs = 1_000
	histMaxNs = 60_000_000_000
	histSigF  = 3
)

// Recorder is ONE worker's private histogram set — never shared across
// goroutines.
type Recorder struct {
	h      [opCount]*hdrhistogram.Histogram
	errs   [opCount]int64
	counts [opCount]int64
}

// NewRecorder builds a per-worker recorder.
func NewRecorder() *Recorder {
	r := &Recorder{}
	for i := range r.h {
		r.h[i] = hdrhistogram.New(histMinNs, histMaxNs, histSigF)
	}
	return r
}

// Observe records one latency sample (clamped into histogram range —
// clamping is visible as Max pegging at 60s, never a lost sample).
func (r *Recorder) Observe(op Op, d time.Duration) {
	ns := d.Nanoseconds()
	if ns < histMinNs {
		ns = histMinNs
	}
	if ns > histMaxNs {
		ns = histMaxNs
	}
	_ = r.h[op].RecordValue(ns) // in-range by the clamp above
	r.counts[op]++
}

// CountError tallies a failed op (not recorded as a latency sample).
func (r *Recorder) CountError(op Op) { r.errs[op]++ }

// Summary is the merged view of every worker's recorder for one cell.
type Summary struct {
	h      [opCount]*hdrhistogram.Histogram
	errs   [opCount]int64
	counts [opCount]int64
}

// Merge combines per-worker recorders into one Summary.
func Merge(rs []*Recorder) *Summary {
	s := &Summary{}
	for i := range s.h {
		s.h[i] = hdrhistogram.New(histMinNs, histMaxNs, histSigF)
	}
	for _, r := range rs {
		for i := range s.h {
			s.h[i].Merge(r.h[i])
			s.errs[i] += r.errs[i]
			s.counts[i] += r.counts[i]
		}
	}
	return s
}

// OpSummary is one op class's percentile digest, in microseconds (the
// repo's reporting unit — soak's JSONL uses µs too).
type OpSummary struct {
	N      int64   `json:"n"`
	Errors int64   `json:"errors"`
	MeanUs float64 `json:"mean_us"`
	P50Us  float64 `json:"p50_us"`
	P90Us  float64 `json:"p90_us"`
	P99Us  float64 `json:"p99_us"`
	P999Us float64 `json:"p999_us"`
	MaxUs  float64 `json:"max_us"`
}

// Op digests one op class. An empty class reports ZEROS — hdr's
// ValueAtQuantile on an empty histogram returns the low bound, which would
// read as a (dishonest) sub-microsecond percentile.
func (s *Summary) Op(op Op) OpSummary {
	if s.counts[op] == 0 {
		return OpSummary{Errors: s.errs[op]}
	}
	h := s.h[op]
	toUs := func(ns int64) float64 { return float64(ns) / 1e3 }
	return OpSummary{
		N:      s.counts[op],
		Errors: s.errs[op],
		MeanUs: h.Mean() / 1e3,
		P50Us:  toUs(h.ValueAtQuantile(50)),
		P90Us:  toUs(h.ValueAtQuantile(90)),
		P99Us:  toUs(h.ValueAtQuantile(99)),
		P999Us: toUs(h.ValueAtQuantile(99.9)),
		MaxUs:  toUs(h.Max()),
	}
}

// WriteHGRM emits the classic HdrHistogram percentile-distribution table
// ("Value Percentile TotalCount 1/(1-Percentile)", value column in µs) —
// the format hdr plotting tools consume.
func (s *Summary) WriteHGRM(op Op, w io.Writer) error {
	h := s.h[op]
	if _, err := fmt.Fprintf(w, "%12s %14s %10s %14s\n\n", "Value", "Percentile", "TotalCount", "1/(1-Percentile)"); err != nil {
		return err
	}
	total := h.TotalCount()
	if total == 0 {
		return nil
	}
	for _, br := range h.CumulativeDistribution() {
		q := br.Quantile / 100.0
		inv := 1.0 / (1.0 - q)
		if q >= 1.0 {
			inv = 0 // sentinel matching hgrm convention for the 100th percentile row
		}
		if _, err := fmt.Fprintf(w, "%12.3f %2.12f %10d %14.2f\n",
			float64(br.ValueAt)/1e3, q, br.Count, inv); err != nil {
			return err
		}
	}
	_, err := fmt.Fprintf(w, "#[Mean = %12.3f, Total count = %d]\n", h.Mean()/1e3, total)
	return err
}
