package gen

import (
	"fmt"
	"strings"
)

// The sweep grid (SPEC-4 §2.2). Blob sizes are EXACT byte counts, 4 KiB
// multiples, printed everywhere — "0.44 MiB" is a band label, 462848 is
// the number (the 16-token vLLM block for an 8B-class model; 2.5 MiB for
// 70B-class). Sizes are fixed per run, never mixed (the fixed-size-chunk
// invariant every store in the comparison shares).
const (
	BlobSmall = 462848  // "0.44 MiB" band: 113 × 4096
	BlobMid   = 1 << 20 // 1 MiB
	BlobLarge = 2621440 // 2.5 MiB: 640 × 4096
)

// Mix is the op mix of a cell.
type Mix string

const (
	MixGet  Mix = "get" // the headline
	MixPut  Mix = "put"
	Mix9010 Mix = "9010" // 90% GET / 10% PUT
)

// Skew is the key-popularity distribution.
type Skew string

const (
	SkewUniform Skew = "uniform"
	SkewZipf    Skew = "zipf099" // YCSB zipfian θ=0.99
)

// Cell is one grid point.
type Cell struct {
	BlobBytes int
	BatchKeys int
	Streams   int
	Mix       Mix
	Skew      Skew
}

// ID is the stable cell identifier used in filenames, JSONL, and --filter.
func (c Cell) ID() string {
	return fmt.Sprintf("b%d_k%d_s%d_%s_%s", c.BlobBytes, c.BatchKeys, c.Streams, c.Mix, c.Skew)
}

// GridConfig selects a subset of the full grid.
type GridConfig struct {
	Headline bool   // {BlobSmall, BlobLarge} × batch 32 × all streams × GET × uniform (the chart cells)
	Filter   string // substring match on Cell.ID(); empty = all
}

var (
	allBlobs   = []int{BlobSmall, BlobMid, BlobLarge}
	allBatches = []int{1, 8, 32}
	allStreams = []int{1, 2, 4, 8, 16, 32, 64}
	allMixes   = []Mix{MixGet, MixPut, Mix9010}
	allSkews   = []Skew{SkewUniform, SkewZipf}
)

// Grid enumerates the requested cells in deterministic order.
func Grid(cfg GridConfig) []Cell {
	blobs, batches, mixes, skews := allBlobs, allBatches, allMixes, allSkews
	if cfg.Headline {
		blobs = []int{BlobSmall, BlobLarge}
		batches = []int{32}
		mixes = []Mix{MixGet}
		skews = []Skew{SkewUniform}
	}
	var out []Cell
	for _, b := range blobs {
		for _, k := range batches {
			for _, s := range allStreams {
				for _, m := range mixes {
					for _, sk := range skews {
						c := Cell{BlobBytes: b, BatchKeys: k, Streams: s, Mix: m, Skew: sk}
						if cfg.Filter != "" && !strings.Contains(c.ID(), cfg.Filter) {
							continue
						}
						out = append(out, c)
					}
				}
			}
		}
	}
	return out
}
