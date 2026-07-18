package gen

import (
	"math"
	"math/rand/v2"
)

// Zipfian is the YCSB/Gray-et-al bounded zipfian generator over [0, n) with
// exponent theta ∈ (0, 1).
//
// Why not the stdlib: math/rand/v2's NewZipf implements the s>1 rejection
// sampler and RETURNS NIL for s ≤ 1 — the benchmark spec's zipf(s=0.99)
// (the standard YCSB skew) is unreachable with it. This is the classic
// "Quickly Generating Billion-Record Synthetic Databases" construction:
// zeta(n, θ) precomputed once per cell (O(n), ~10ms for 10^6 keys), each
// draw O(1).
//
// Determinism: draws come from the caller's PCG stream — bit-identical
// across runs on one platform. Cross-ARCHITECTURE bit-identity of the
// float math is not guaranteed; every JSONL record carries goos/goarch so
// repeatability claims stay scoped (documented in the schema).
type Zipfian struct {
	n     uint64
	theta float64
	alpha float64
	zetan float64
	eta   float64
	z2    float64 // zeta(2, θ)
	r     *rand.Rand
}

// NewZipfian builds the generator. Panics on n == 0 or theta outside (0,1)
// — misuse is a harness bug, not a runtime condition.
func NewZipfian(r *rand.Rand, n uint64, theta float64) *Zipfian {
	if n == 0 || theta <= 0 || theta >= 1 {
		panic("gen: NewZipfian needs n > 0 and theta in (0,1)")
	}
	z := &Zipfian{n: n, theta: theta, r: r}
	z.zetan = zeta(n, theta)
	z.z2 = zeta(2, theta)
	z.alpha = 1.0 / (1.0 - theta)
	z.eta = (1.0 - math.Pow(2.0/float64(n), 1.0-theta)) / (1.0 - z.z2/z.zetan)
	return z
}

// Next draws the next rank in [0, n) from the generator's own stream —
// rank 0 is the hottest key.
func (z *Zipfian) Next() uint64 { return z.Draw(z.r.Float64()) }

// Draw maps ONE uniform u ∈ [0,1) to a rank — a pure function of u and the
// precomputed constants, so the sweep can feed it per-EVENT PCG streams and
// keep the op sequence index-deterministic across any worker interleaving.
func (z *Zipfian) Draw(u float64) uint64 {
	uz := u * z.zetan
	if uz < 1.0 {
		return 0
	}
	if uz < 1.0+math.Pow(0.5, z.theta) {
		return 1
	}
	v := uint64(float64(z.n) * math.Pow(z.eta*u-z.eta+1.0, z.alpha)) //nolint:gosec // G115: clamped to [0,n) below
	if v >= z.n {
		// Float edge (u→1⁻, pow rounding to 1.0) can hit exactly n — clamp
		// so callers never index one past the pool (the ladder's false-miss
		// edge). Standard YCSB impls clamp here too.
		v = z.n - 1
	}
	return v
}

func zeta(n uint64, theta float64) float64 {
	var sum float64
	for i := uint64(1); i <= n; i++ {
		sum += 1.0 / math.Pow(float64(i), theta)
	}
	return sum
}
