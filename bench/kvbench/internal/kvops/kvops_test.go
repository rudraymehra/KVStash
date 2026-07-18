package kvops

import (
	"bytes"
	"encoding/hex"
	"errors"
	"io"
	"strings"
	"testing"

	"pgregory.net/rapid"
)

func TestRoundTripProperty(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		nRecs := rapid.IntRange(0, 50).Draw(rt, "n")
		type rec struct {
			ts   uint64
			keys [][32]byte
		}
		var recs []rec
		ts := rapid.Uint64Range(0, 1<<40).Draw(rt, "ts0")
		for i := 0; i < nRecs; i++ {
			n := rapid.IntRange(1, 70).Draw(rt, "nk")
			keys := make([][32]byte, n)
			for j := range keys {
				keys[j][0] = byte(rapid.IntRange(0, 255).Draw(rt, "kb")) //nolint:gosec // G115: range [0,255]
				keys[j][31] = byte(j)
			}
			ts += rapid.Uint64Range(0, 1e6).Draw(rt, "dt")
			recs = append(recs, rec{ts, keys})
		}

		var buf bytes.Buffer
		w, err := NewWriter(&buf, 462848, Meta{Trace: "prop", Converter: "test", Requests: int64(nRecs)})
		if err != nil {
			rt.Fatal(err)
		}
		for _, r := range recs {
			if err := w.Write(r.ts, r.keys); err != nil {
				rt.Fatal(err)
			}
		}
		if err := w.Flush(); err != nil {
			rt.Fatal(err)
		}

		rd, err := NewReader(bytes.NewReader(buf.Bytes()))
		if err != nil {
			rt.Fatal(err)
		}
		if rd.Header().BlobBytes != 462848 || rd.Header().Meta.Trace != "prop" ||
			rd.Header().Meta.KeyDerivation != KeyDerivation {
			rt.Fatalf("header mangled: %+v", rd.Header())
		}
		var got Record
		for i, want := range recs {
			if err := rd.Next(&got); err != nil {
				rt.Fatalf("rec %d: %v", i, err)
			}
			if got.TSMicros != want.ts || len(got.Keys) != len(want.keys) {
				rt.Fatalf("rec %d mismatch", i)
			}
			for j := range want.keys {
				if got.Keys[j] != want.keys[j] {
					rt.Fatalf("rec %d key %d mismatch", i, j)
				}
			}
		}
		if err := rd.Next(&got); !errors.Is(err, io.EOF) {
			rt.Fatalf("want clean EOF, got %v", err)
		}
		if rd.Records() != int64(nRecs) {
			rt.Fatalf("Records()=%d", rd.Records())
		}
	})
}

func TestReaderRejectsTears(t *testing.T) {
	var buf bytes.Buffer
	w, _ := NewWriter(&buf, 4096, Meta{Trace: "tear"})
	keys := [][32]byte{{1}, {2}}
	if err := w.Write(100, keys); err != nil {
		t.Fatal(err)
	}
	if err := w.Write(200, keys); err != nil {
		t.Fatal(err)
	}
	_ = w.Flush()
	whole := buf.Bytes()

	// Truncate mid-record (torn keys) — must be an explicit error, not EOF.
	rd, err := NewReader(bytes.NewReader(whole[:len(whole)-16]))
	if err != nil {
		t.Fatal(err)
	}
	var rec Record
	if err := rd.Next(&rec); err != nil {
		t.Fatal(err)
	}
	if err := rd.Next(&rec); err == nil || errors.Is(err, io.EOF) {
		t.Fatalf("torn record read as %v", err)
	}

	// Timestamp regression rejected at write AND read.
	if err := w.Write(50, keys); err == nil {
		t.Fatal("writer accepted a timestamp regression")
	}
	// Bad magic.
	bad := append([]byte("NOPE"), whole[4:]...)
	if _, err := NewReader(bytes.NewReader(bad)); err == nil {
		t.Fatal("bad magic accepted")
	}
}

func TestTraceKeyGolden(t *testing.T) {
	// Pinned vectors — the Python replayer must reproduce these (domain
	// "kvbench-trace-v1\x00", u32-LE length-prefixed fields).
	got := TraceKey("bailian-A", "12345")
	const want = "de31d5a2461718aae9d2894c1d1c05a4d860bdb89d4dbe8aaefde695732736e4"
	if hex.EncodeToString(got[:]) != want {
		// Regenerate deliberately only on a spec bump.
		t.Fatalf("TraceKey drifted:\n got  %x\n want %s", got, want)
	}
	// Distinctness across fields vs concatenation ambiguity.
	if TraceKey("a", "bc") == TraceKey("ab", "c") {
		t.Fatal("length-prefixing failed — field-boundary ambiguity")
	}
	if TraceKey("t", "1", "2") == TraceKey("t", "12") {
		t.Fatal("subidx ambiguity")
	}
}

func TestConvertBailianFixture(t *testing.T) {
	// A 5-line fixture with known counts — the count-exactness gate shape.
	fixture := strings.Join([]string{
		`{"timestamp": 1.0, "hash_ids": [1, 2, 3]}`,
		`{"timestamp": 1.5, "hash_ids": [1, 2, 3, 4]}`,
		`{"timestamp": 2.0, "hash_ids": [9]}`,
		`{"timestamp": 2.0, "hash_ids": [1]}`,
		`{"timestamp": 3.25, "hash_ids": [5, 6]}`,
	}, "\n")
	var buf bytes.Buffer
	w, _ := NewWriter(&buf, 462848, Meta{Trace: "fixtA", Converter: "bailian"})
	st, err := ConvertBailian(strings.NewReader(fixture), "fixtA", w)
	if err != nil {
		t.Fatal(err)
	}
	if st.Requests != 5 || st.KeysTotal != 11 {
		t.Fatalf("counts: %+v (want 5 requests, 11 keys)", st)
	}
	rd, err := NewReader(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	var rec Record
	if err := rd.Next(&rec); err != nil {
		t.Fatal(err)
	}
	if rec.TSMicros != 1_000_000 || len(rec.Keys) != 3 {
		t.Fatalf("rec0: ts=%d n=%d", rec.TSMicros, len(rec.Keys))
	}
	// Shared prefix: hash_id 1 in line 1 and line 2 derive the SAME key
	// (reuse structure preserved — the whole point of trace replay).
	k1 := rec.Keys[0]
	if err := rd.Next(&rec); err != nil {
		t.Fatal(err)
	}
	if rec.Keys[0] != k1 {
		t.Fatal("hash_id reuse broken across lines")
	}
}

func TestConvertMooncakeFixture(t *testing.T) {
	fixture := strings.Join([]string{
		`{"timestamp": 1000, "hash_ids": [7]}`,
		`{"timestamp": 2000, "hash_ids": [7, 8]}`,
	}, "\n")
	var buf bytes.Buffer
	w, _ := NewWriter(&buf, 462848, Meta{Trace: "fixtM", Converter: "mooncake"})
	st, err := ConvertMooncake(strings.NewReader(fixture), "fixtM", w)
	if err != nil {
		t.Fatal(err)
	}
	if st.Requests != 2 || st.KeysTotal != 3*32 {
		t.Fatalf("counts: %+v (want 2 requests, 96 keys)", st)
	}
	rd, _ := NewReader(bytes.NewReader(buf.Bytes()))
	var rec Record
	if err := rd.Next(&rec); err != nil {
		t.Fatal(err)
	}
	if rec.TSMicros != 1_000_000 || len(rec.Keys) != 32 {
		t.Fatalf("rec0: ts=%d n=%d", rec.TSMicros, len(rec.Keys))
	}
	// 512-token semantics: sub-keys of one hash_id are distinct and
	// deterministic.
	if rec.Keys[0] == rec.Keys[1] {
		t.Fatal("sub-keys collide")
	}
	first := rec.Keys[0]
	if err := rd.Next(&rec); err != nil {
		t.Fatal(err)
	}
	if rec.Keys[0] != first {
		t.Fatal("hash_id 7's sub-key 0 differs across lines")
	}
}
