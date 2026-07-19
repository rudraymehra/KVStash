// Command s3probe measures the daemon's REAL S3 code paths against a live
// bucket: whole-segment uploads through s3spill.Spiller (the spill path,
// serially awaited so each sample is one clean upload) and ranged reads
// through s3spill.Restorer.ReadRange (the cold-GET path), at representative
// KV-block sizes. Output is JSONL (one sample per line) plus a stderr
// summary — the DESIGN.md S3-economics numbers come from here.
//
// Credentials come from the ambient AWS chain (env vars on the rig box).
// Run in-region (us-east-1 box → us-east-1 bucket) or the numbers measure
// your WAN, not S3.
package main

import (
	"context"
	"crypto/rand"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/kvstash/kvblockd/internal/store/s3spill"
)

func main() {
	bucket := flag.String("bucket", "", "target bucket (required)")
	region := flag.String("region", "us-east-1", "bucket region")
	node := flag.String("node", "e2e", "s3 node_id namespace for object keys")
	segMB := flag.Int("seg-mb", 8, "segment object size (MiB) — mirrors nvme_segment_bytes")
	puts := flag.Int("puts", 10, "segment uploads to sample")
	reads := flag.Int("reads", 30, "ranged reads per block size")
	sizesCSV := flag.String("sizes", "65536,262144,1048576,2621440", "cold-read block sizes (bytes, csv)")
	out := flag.String("out", "s3probe-results.jsonl", "JSONL output path")
	flag.Parse()
	if *bucket == "" {
		fmt.Fprintln(os.Stderr, "s3probe: -bucket is required")
		os.Exit(2)
	}
	if err := run(*bucket, *region, *node, *segMB, *puts, *reads, *sizesCSV, *out); err != nil {
		fmt.Fprintln(os.Stderr, "s3probe:", err)
		os.Exit(1)
	}
}

type sample struct {
	Op    string  `json:"op"` // "put_segment" | "ranged_get"
	Bytes int64   `json:"bytes"`
	MS    float64 `json:"ms"`
	Err   string  `json:"err,omitempty"`
}

func run(bucket, region, node string, segMB, puts, reads int, sizesCSV, out string) error {
	ctx := context.Background()
	cfg := s3spill.Config{Bucket: bucket, Region: region, NodeID: node, OpTimeout: 60 * time.Second}
	api, err := s3spill.NewClient(ctx, cfg)
	if err != nil {
		return err
	}
	sp := s3spill.NewSpiller(api, cfg, 2)
	defer sp.Close()
	re := s3spill.NewRestorer(api, cfg)

	segBytes := int64(segMB) << 20
	seg, err := os.CreateTemp("", "s3probe-seg-*")
	if err != nil {
		return err
	}
	defer os.Remove(seg.Name())
	if _, err := io.CopyN(seg, rand.Reader, segBytes); err != nil {
		return err
	}
	_ = seg.Close()

	f, err := os.Create(out) //nolint:gosec // G304: operator-chosen results path
	if err != nil {
		return err
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	emit := func(s sample) { _ = enc.Encode(s) }

	// PUT phase: serial spill uploads — each DemoteSegment is awaited before
	// the next, so a sample is one whole-object PutObject with no queueing.
	putMS := make([]float64, 0, puts)
	for i := 0; i < puts; i++ {
		done := make(chan bool, 1)
		start := time.Now()
		ok := sp.DemoteSegment(
			uint64(i+1), segBytes, //nolint:gosec // G115: small loop index
			func() (io.ReadSeekCloser, error) {
				return os.Open(seg.Name()) //nolint:gosec // G304: our own temp file
			},
			func(_ uint64, up bool) { done <- up },
		)
		if !ok {
			return fmt.Errorf("put %d: enqueue refused", i)
		}
		up := <-done
		ms := float64(time.Since(start).Microseconds()) / 1000
		s := sample{Op: "put_segment", Bytes: segBytes, MS: ms}
		if !up {
			s.Err = "upload failed"
		}
		emit(s)
		if up {
			putMS = append(putMS, ms)
		}
	}
	if len(putMS) == 0 {
		return fmt.Errorf("no segment upload succeeded — check credentials/bucket")
	}
	summarize("put_segment", segBytes, putMS)

	// GET phase: the daemon's exact cold-read call, random offsets inside
	// random uploaded objects.
	for _, szs := range strings.Split(sizesCSV, ",") {
		sz, perr := strconv.ParseInt(strings.TrimSpace(szs), 10, 64)
		if perr != nil || sz <= 0 || sz > segBytes {
			return fmt.Errorf("bad size %q", szs)
		}
		buf := make([]byte, sz)
		getMS := make([]float64, 0, reads)
		for i := 0; i < reads; i++ {
			segID := uint64(1 + i%len(putMS)) //nolint:gosec // G115: small counts
			maxOff := segBytes - sz
			off := (int64(i) * 2654435761) % (maxOff + 1) //nolint:gosec // deterministic offset scatter
			start := time.Now()
			rerr := re.ReadRange(ctx, segID, off, sz, buf)
			ms := float64(time.Since(start).Microseconds()) / 1000
			s := sample{Op: "ranged_get", Bytes: sz, MS: ms}
			if rerr != nil {
				s.Err = rerr.Error()
			} else {
				getMS = append(getMS, ms)
			}
			emit(s)
		}
		if len(getMS) == 0 {
			return fmt.Errorf("all %d-byte ranged reads failed", sz)
		}
		summarize("ranged_get", sz, getMS)
	}
	return nil
}

func summarize(op string, bytes int64, ms []float64) {
	sort.Float64s(ms)
	pct := func(p float64) float64 { return ms[int(p*float64(len(ms)-1))] }
	mbps := float64(bytes) / (pct(0.5) / 1000) / (1 << 20)
	fmt.Fprintf(os.Stderr, "[s3probe] %-12s %9dB n=%-3d p50=%8.1fms p95=%8.1fms p99=%8.1fms max=%8.1fms (~%.0f MiB/s at p50)\n",
		op, bytes, len(ms), pct(0.5), pct(0.95), pct(0.99), ms[len(ms)-1], mbps)
}
