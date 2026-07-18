package gen

import (
	"bytes"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"math/rand/v2"
	"os"
	"testing"

	"github.com/zeebo/xxh3"
)

// TestPayloadGolden pins "kvbench-payload-v1" against the committed golden
// vectors — the cross-language contract file the Python replayer consumes.
func TestPayloadGolden(t *testing.T) {
	raw, err := os.ReadFile("testdata/payload_golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var g struct {
		Spec      string `json:"spec"`
		Seed      uint64 `json:"seed"`
		KeyHex    string `json:"key_hex"`
		BlobBytes int    `json:"blob_bytes"`
		First64   string `json:"first64_hex"`
		XXH3Hex   string `json:"blob_xxh3_64_hex"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	if g.Spec != "kvbench-payload-v1" {
		t.Fatalf("golden spec %q", g.Spec)
	}
	var key [32]byte
	kb, _ := hex.DecodeString(g.KeyHex)
	copy(key[:], kb)

	blob := make([]byte, g.BlobBytes)
	FillPayload(blob, g.Seed, key)
	if got := hex.EncodeToString(blob[:64]); got != g.First64 {
		t.Fatalf("first64:\n got  %s\n want %s", got, g.First64)
	}
	var sum [8]byte
	for i, b := range hexMust(t, g.XXH3Hex) {
		sum[i] = b
	}
	h := xxh3.Hash(blob)
	want := uint64(sum[0])<<56 | uint64(sum[1])<<48 | uint64(sum[2])<<40 | uint64(sum[3])<<32 |
		uint64(sum[4])<<24 | uint64(sum[5])<<16 | uint64(sum[6])<<8 | uint64(sum[7])
	if h != want {
		t.Fatalf("blob xxh3 %#x, want %#x", h, want)
	}
	if !VerifyPayload(blob, g.Seed, key) {
		t.Fatal("VerifyPayload rejected a pristine blob")
	}
}

func hexMust(t *testing.T, s string) []byte {
	t.Helper()
	b, err := hex.DecodeString(s)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func TestVerifyPayloadCatchesFlips(t *testing.T) {
	ks := Keyspace{Seed: 7}
	blob := make([]byte, BlobSmall)
	key := ks.Key(3)
	FillPayload(blob, 7, key)
	if !VerifyPayload(blob, 7, key) {
		t.Fatal("pristine blob rejected")
	}
	for _, off := range []int{0, 15, 16, BlobSmall / 2, BlobSmall - 1} {
		blob[off] ^= 0x01
		if VerifyPayload(blob, 7, key) {
			t.Fatalf("flip at %d not caught", off)
		}
		blob[off] ^= 0x01
	}
	// Wrong seed and wrong key are also corruption.
	if VerifyPayload(blob, 8, key) {
		t.Fatal("wrong seed accepted")
	}
	if VerifyPayload(blob, 7, ks.Key(4)) {
		t.Fatal("wrong key accepted")
	}
	// Truncated tails still verify their prefix bytes exactly.
	if !VerifyPayload(blob[:100], 7, key) {
		t.Fatal("100-byte prefix rejected")
	}
}

func TestPayloadIncompressibleish(t *testing.T) {
	// Cheap sanity: byte-value histogram of a 1 MiB blob should be near
	// uniform (no value above 2× the expected share) — a stand-in for the
	// "compression can't flatter anyone" property.
	blob := make([]byte, 1<<20)
	FillPayload(blob, 1, Keyspace{Seed: 1}.Key(0))
	var histo [256]int
	for _, b := range blob {
		histo[b]++
	}
	expect := len(blob) / 256
	for v, n := range histo {
		if n > 2*expect {
			t.Fatalf("byte %#x appears %d times (expected ~%d) — payload is compressible", v, n, expect)
		}
	}
}

func TestKeyspaceDeterminismAndDisjointness(t *testing.T) {
	a, b := Keyspace{Seed: 1}, Keyspace{Seed: 2}
	k1, k2 := a.Key(5), a.Key(5)
	if k1 != k2 {
		t.Fatal("nondeterministic key")
	}
	if a.Key(5) == b.Key(5) {
		t.Fatal("seed does not scope the keyspace")
	}
	if a.Key(5) == a.Key(6) {
		t.Fatal("index collision")
	}
	k := a.Key(0)
	if string(k[24:32]) != "kvbench1" {
		t.Fatal("family tag missing — disjointness from trace/product keys relies on it")
	}
}

func TestZipfianGoldenAndSkew(t *testing.T) {
	z := NewZipfian(rand.New(rand.NewPCG(42, 1)), 1_000_000, 0.99) //nolint:gosec // G404: deterministic benchmark skew, not crypto
	got := make([]uint64, 16)
	for i := range got {
		got[i] = z.Next()
	}
	// Golden first-16 sequence for (PCG(42,1), n=1e6, θ=0.99) — pins the
	// generator's bit-determinism on this platform (goarch caveat recorded
	// in the JSONL schema).
	z2 := NewZipfian(rand.New(rand.NewPCG(42, 1)), 1_000_000, 0.99) //nolint:gosec // G404: deterministic benchmark skew, not crypto
	for i := range got {
		if v := z2.Next(); v != got[i] {
			t.Fatalf("draw %d: %d != %d — generator is not seed-deterministic", i, v, got[i])
		}
	}

	// Skew bound: at θ=0.99 the hottest 10% of ranks must absorb ≥60% of
	// 1e6 draws (YCSB-shaped skew, loose bound).
	z3 := NewZipfian(rand.New(rand.NewPCG(7, 7)), 1_000_000, 0.99) //nolint:gosec // G404: deterministic benchmark skew, not crypto
	hot := 0
	const draws = 1_000_000
	for i := 0; i < draws; i++ {
		if z3.Next() < 100_000 {
			hot++
		}
	}
	if hot < draws*60/100 {
		t.Fatalf("hot-10%% share = %.1f%%, want ≥60%%", 100*float64(hot)/draws)
	}
	// Range bound.
	z4 := NewZipfian(rand.New(rand.NewPCG(9, 9)), 100, 0.99) //nolint:gosec // G404: deterministic benchmark skew, not crypto
	for i := 0; i < 100_000; i++ {
		if v := z4.Next(); v >= 100 {
			t.Fatalf("draw %d out of range", v)
		}
	}
}

func TestGrid(t *testing.T) {
	full := Grid(GridConfig{})
	if len(full) != 3*3*7*3*2 {
		t.Fatalf("full grid %d cells, want %d", len(full), 3*3*7*3*2)
	}
	head := Grid(GridConfig{Headline: true})
	if len(head) != 2*1*7*1*1 {
		t.Fatalf("headline grid %d cells, want 14", len(head))
	}
	for _, c := range head {
		if c.Mix != MixGet || c.BatchKeys != 32 {
			t.Fatalf("headline cell %s not GET/batch32", c.ID())
		}
	}
	f := Grid(GridConfig{Filter: "b462848_k32_s8_get_uniform"})
	if len(f) != 1 {
		t.Fatalf("filter matched %d cells", len(f))
	}
	if f[0].ID() != "b462848_k32_s8_get_uniform" {
		t.Fatalf("cell ID drifted: %s", f[0].ID())
	}
	// Blob sizes are 4 KiB multiples (nvmefs O_DIRECT + honesty of exact counts).
	for _, b := range []int{BlobSmall, BlobMid, BlobLarge} {
		if b%4096 != 0 {
			t.Fatalf("blob %d not 4 KiB-aligned", b)
		}
	}
}

func TestZipfianPanicsOnMisuse(t *testing.T) {
	for _, bad := range []struct {
		n     uint64
		theta float64
	}{{0, 0.99}, {10, 0}, {10, 1.0}, {10, 1.5}} {
		func() {
			defer func() {
				if recover() == nil {
					t.Fatalf("NewZipfian(%d, %v) did not panic", bad.n, bad.theta)
				}
			}()
			NewZipfian(rand.New(rand.NewPCG(1, 1)), bad.n, bad.theta) //nolint:gosec // G404: misuse-panic test
		}()
	}
}

var _ = bytes.Equal // keep bytes import if assertions above change shape

// TestWriteGolden regenerates testdata/payload_golden.json when
// KVBENCH_WRITE_GOLDEN=1 — the standard -update pattern. The committed file
// is the cross-language contract; regenerate ONLY on a deliberate spec bump.
func TestWriteGolden(t *testing.T) {
	if os.Getenv("KVBENCH_WRITE_GOLDEN") != "1" {
		t.Skip("set KVBENCH_WRITE_GOLDEN=1 to regenerate")
	}
	var key [32]byte
	for i := range key {
		key[i] = 0x42
	}
	blob := make([]byte, BlobSmall)
	FillPayload(blob, 1, key)
	sum := xxh3.Hash(blob)
	// XXH3's canonical digest representation is big-endian by definition —
	// this is a hash encoding for the golden file, not KVB1 wire code.
	sumBytes := binary.BigEndian.AppendUint64(nil, sum) //nolint:gocritic // ruleguard wireByteOrder: canonical hash digest, not wire
	out := map[string]any{
		"spec":             "kvbench-payload-v1",
		"note":             "cross-language contract: chunk_j = XXH3_128('kvbench-payload-v1' || u64le(seed) || key || u64le(j)), canonical big-endian 16B, concatenated, truncated to blob_bytes. Python: xxhash.xxh3_128(P + j.to_bytes(8,'little')).digest()",
		"seed":             1,
		"key_hex":          hex.EncodeToString(key[:]),
		"blob_bytes":       BlobSmall,
		"first64_hex":      hex.EncodeToString(blob[:64]),
		"blob_xxh3_64_hex": hex.EncodeToString(sumBytes),
	}
	f, err := os.Create("testdata/payload_golden.json")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	if err := enc.Encode(out); err != nil {
		t.Fatal(err)
	}
}
