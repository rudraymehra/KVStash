package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/kvstash/kvblockd/bench/kvbench/internal/kvops"
)

// cmdConvert turns a Bailian/Mooncake trace file into .kvops. The
// acceptance discipline is COUNT-EXACTNESS: --expect-requests (from the
// published trace numbers) makes a silent line drop a hard failure.
func cmdConvert(_ context.Context, args []string) error {
	fs := flag.NewFlagSet("convert", flag.ExitOnError)
	var (
		format   = fs.String("format", "", "bailian | mooncake")
		in       = fs.String("in", "", "trace file (JSONL)")
		out      = fs.String("out", "", ".kvops output path")
		name     = fs.String("trace", "", "trace name (key-derivation field + provenance)")
		blob     = fs.Int("blob-bytes", 462848, "bytes-per-block model parameter (recorded in the header)")
		expected = fs.Int64("expect-requests", 0, "published request count; mismatch = converter corruption (0 = skip)")
	)
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *format == "" || *in == "" || *out == "" || *name == "" {
		return fmt.Errorf("convert needs --format --in --out --trace")
	}
	if *blob <= 0 || *blob >= 1<<32 {
		return fmt.Errorf("--blob-bytes %d out of range", *blob)
	}
	if len(*name) > 1<<16 {
		return fmt.Errorf("--trace name too long (%d bytes)", len(*name))
	}

	inF, err := os.Open(*in) //nolint:gosec // G304: operator-chosen trace file
	if err != nil {
		return err
	}
	defer func() { _ = inF.Close() }()
	sum := sha256.New()
	tee := io.TeeReader(inF, sum)

	outF, err := os.Create(*out) //nolint:gosec // G304: operator-chosen output
	if err != nil {
		return err
	}
	defer func() { _ = outF.Close() }()

	// Provenance is stamped after the fact via the meta we can compute up
	// front; the source sha256 streams alongside the conversion.
	w, err := kvops.NewWriter(outF, uint32(*blob), kvops.Meta{ //nolint:gosec // G115: blob sizes are small positive
		Trace: *name, Converter: *format,
	})
	if err != nil {
		return err
	}

	var st kvops.ConvStats
	switch *format {
	case "bailian":
		st, err = kvops.ConvertBailian(tee, *name, w)
	case "mooncake":
		st, err = kvops.ConvertMooncake(tee, *name, w)
	default:
		return fmt.Errorf("unknown format %q", *format)
	}
	if err != nil {
		return err
	}
	if err := outF.Sync(); err != nil {
		return err
	}
	fmt.Printf("convert: %s → %s  requests=%d keys=%d source_sha256=%s\n",
		*in, *out, st.Requests, st.KeysTotal, hex.EncodeToString(sum.Sum(nil)))
	if *expected > 0 && st.Requests != *expected {
		return fmt.Errorf("COUNT MISMATCH: converted %d requests, published count is %d — the converter is wrong, stop", st.Requests, *expected)
	}
	return nil
}
