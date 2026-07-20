// Package s3spill is the cold tier's transport: one SEALED NVMe segment
// (footer included, byte-for-byte) = one S3 object, written whole and read
// back by exact byte range. The index never leaves the node — S3 stores
// bytes, never metadata — so a cold GET is exactly one ranged GetObject.
//
// The SeaweedFS trick, and why coalescing wins: S3 has no offset writes,
// which fits an append-only sealed segment perfectly, and one 256 MiB
// object costs ONE PutObject where per-block objects would cost ~100 —
// request-cost math is the design (economics in docs/DESIGN.md).
package s3spill

import (
	"context"
	"fmt"
	"io"
	"time"

	awsv2 "github.com/aws/aws-sdk-go-v2/aws"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/s3"
)

// S3API is the seam the spiller and restorer call through — the SDK client
// satisfies it; tests substitute gofakes3-backed or synthetic
// implementations. Every call carries a caller-owned context deadline.
type S3API interface {
	PutObject(ctx context.Context, in *s3.PutObjectInput, opts ...func(*s3.Options)) (*s3.PutObjectOutput, error)
	GetObject(ctx context.Context, in *s3.GetObjectInput, opts ...func(*s3.Options)) (*s3.GetObjectOutput, error)
	HeadObject(ctx context.Context, in *s3.HeadObjectInput, opts ...func(*s3.Options)) (*s3.HeadObjectOutput, error)
	DeleteObject(ctx context.Context, in *s3.DeleteObjectInput, opts ...func(*s3.Options)) (*s3.DeleteObjectOutput, error)
}

// Config selects the bucket and endpoint. EndpointOverride + PathStyle serve
// MinIO/gofakes3-compatible targets; empty means real AWS S3.
type Config struct {
	Bucket           string
	Region           string
	NodeID           string // namespaces the object keys: kvblockd/<node_id>/segments/…
	EndpointOverride string
	PathStyle        bool
	// OpTimeout bounds every S3 call (0 = 30s). Cold GETs pass their own
	// tighter deadline via context; this is the outer safety net.
	OpTimeout time.Duration
}

func (c Config) withDefaults() Config {
	if c.OpTimeout == 0 {
		c.OpTimeout = 30 * time.Second
	}
	return c
}

// NewClient builds the real SDK client from ambient credentials (env,
// shared config, IMDS) — no credentials ever live in kvblockd config files.
func NewClient(ctx context.Context, cfg Config) (S3API, error) {
	if cfg.Bucket == "" {
		return nil, fmt.Errorf("s3spill: bucket required")
	}
	awsCfg, err := awsconfig.LoadDefaultConfig(ctx, awsconfig.WithRegion(cfg.Region))
	if err != nil {
		return nil, fmt.Errorf("s3spill: aws config: %w", err)
	}
	return s3.NewFromConfig(awsCfg, func(o *s3.Options) {
		if cfg.EndpointOverride != "" {
			o.BaseEndpoint = &cfg.EndpointOverride
		}
		o.UsePathStyle = cfg.PathStyle
		// The SDK's default flexible checksums emit trailing sums that break
		// plain-HTTP MinIO/gofakes3 targets (segment bodies are seekable now
		// — the ReadSeekCloser seam exists for SigV4 — but the trailing-sum
		// incompatibility is transport-shaped, not body-shaped). When-
		// required keeps integrity via ETag + the segment's own footer xxh3s.
		o.RequestChecksumCalculation = awsv2.RequestChecksumCalculationWhenRequired
	}), nil
}

// segKey is the object-key layout — stable, listable by prefix per node.
func segKey(nodeID string, segID uint64) string {
	return fmt.Sprintf("kvblockd/%s/segments/seg-%08d.seg", nodeID, segID)
}

// drainClose fully drains and closes an S3 body (connection reuse hygiene).
func drainClose(rc io.ReadCloser) {
	if rc != nil {
		_, _ = io.Copy(io.Discard, rc)
		_ = rc.Close()
	}
}
