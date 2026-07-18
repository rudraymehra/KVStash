// Package kvops defines the canonical binary op-stream both the Go and
// Python replayers consume — one format, one op sequence, so a Redis
// baseline replays EXACTLY the ops kvblockd replayed.
//
// Layout (little-endian, the repo's wire convention):
//
//	header: magic "KVOP" (4) | version u16 = 1 | reserved u16 = 0
//	      | blob_bytes u32 | meta_len u32 | meta[meta_len]
//	        (meta = JSON: trace, converter, source_sha256, requests,
//	         keys_total, key_derivation = "kvbench-trace-v1")
//	record: ts_us u64 | n_keys u16 | keys n×32 bytes
//	        (one record per request's full prefix chain, timestamp order)
package kvops

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
)

const (
	magic      = "KVOP"
	version    = 1
	hdrFixed   = 4 + 2 + 2 + 4 + 4
	recFixed   = 8 + 2
	maxChain   = 65535    // n_keys is u16
	maxMetaLen = 16 << 20 // sanity cap on the header's meta blob
)

// Meta is the header's provenance document.
type Meta struct {
	Trace         string `json:"trace"`
	Converter     string `json:"converter"`
	SourceSHA256  string `json:"source_sha256,omitempty"`
	Requests      int64  `json:"requests"`
	KeysTotal     int64  `json:"keys_total"`
	KeyDerivation string `json:"key_derivation"` // "kvbench-trace-v1"
}

// Header is the parsed file header.
type Header struct {
	Version   uint16
	BlobBytes uint32
	Meta      Meta
}

// Record is one request's chain. Keys' backing array is REUSED across
// Reader.Next calls — copy if retaining.
type Record struct {
	TSMicros uint64
	Keys     [][32]byte
}

// Writer streams records out.
type Writer struct {
	w       *bufio.Writer
	lastTS  uint64
	started bool
}

// NewWriter emits the header immediately.
func NewWriter(w io.Writer, blobBytes uint32, meta Meta) (*Writer, error) {
	meta.KeyDerivation = KeyDerivation
	mb, err := json.Marshal(meta)
	if err != nil {
		return nil, err
	}
	if len(mb) > maxMetaLen {
		// The reader enforces this cap; refuse to WRITE a file no reader
		// will accept (the ladder's asymmetry L4).
		return nil, fmt.Errorf("kvops: meta %d bytes exceeds the %d cap", len(mb), maxMetaLen)
	}
	bw := bufio.NewWriterSize(w, 1<<20)
	var hdr [hdrFixed]byte
	copy(hdr[0:4], magic)
	binary.LittleEndian.PutUint16(hdr[4:6], version)
	binary.LittleEndian.PutUint32(hdr[8:12], blobBytes)
	binary.LittleEndian.PutUint32(hdr[12:16], uint32(len(mb))) //nolint:gosec // G115: meta is small JSON
	if _, err := bw.Write(hdr[:]); err != nil {
		return nil, err
	}
	if _, err := bw.Write(mb); err != nil {
		return nil, err
	}
	return &Writer{w: bw}, nil
}

// Write appends one record (timestamps must be nondecreasing).
func (w *Writer) Write(tsMicros uint64, keys [][32]byte) error {
	if len(keys) == 0 || len(keys) > maxChain {
		return fmt.Errorf("kvops: chain of %d keys (max %d)", len(keys), maxChain)
	}
	if w.started && tsMicros < w.lastTS {
		return fmt.Errorf("kvops: timestamp regression %d < %d", tsMicros, w.lastTS)
	}
	w.started, w.lastTS = true, tsMicros
	var rec [recFixed]byte
	binary.LittleEndian.PutUint64(rec[0:8], tsMicros)
	binary.LittleEndian.PutUint16(rec[8:10], uint16(len(keys))) //nolint:gosec // G115: bounded by maxChain above
	if _, err := w.w.Write(rec[:]); err != nil {
		return err
	}
	for i := range keys {
		if _, err := w.w.Write(keys[i][:]); err != nil {
			return err
		}
	}
	return nil
}

// Flush drains the buffer (call once at the end).
func (w *Writer) Flush() error { return w.w.Flush() }

// Reader streams records in, enforcing header validity, timestamp order,
// and clean truncation errors (a torn file must never silently shorten a
// benchmark).
type Reader struct {
	r      *bufio.Reader
	hdr    Header
	lastTS uint64
	nRead  int64
	keys   [][32]byte // reused backing
}

// NewReader parses and validates the header.
func NewReader(r io.Reader) (*Reader, error) {
	br := bufio.NewReaderSize(r, 1<<20)
	var hdr [hdrFixed]byte
	if _, err := io.ReadFull(br, hdr[:]); err != nil {
		return nil, fmt.Errorf("kvops: header: %w", err)
	}
	if string(hdr[0:4]) != magic {
		return nil, fmt.Errorf("kvops: bad magic %q", hdr[0:4])
	}
	if v := binary.LittleEndian.Uint16(hdr[4:6]); v != version {
		return nil, fmt.Errorf("kvops: version %d (want %d)", v, version)
	}
	metaLen := binary.LittleEndian.Uint32(hdr[12:16])
	if metaLen > maxMetaLen {
		return nil, fmt.Errorf("kvops: meta %d bytes exceeds the sanity cap", metaLen)
	}
	mb := make([]byte, metaLen)
	if _, err := io.ReadFull(br, mb); err != nil {
		return nil, fmt.Errorf("kvops: meta: %w", err)
	}
	out := &Reader{r: br}
	out.hdr.Version = version
	out.hdr.BlobBytes = binary.LittleEndian.Uint32(hdr[8:12])
	if err := json.Unmarshal(mb, &out.hdr.Meta); err != nil {
		return nil, fmt.Errorf("kvops: meta json: %w", err)
	}
	return out, nil
}

// Header returns the parsed header.
func (r *Reader) Header() Header { return r.hdr }

// Records returns how many records Next has delivered.
func (r *Reader) Records() int64 { return r.nRead }

// Next fills rec with the next record. io.EOF at a CLEAN end; any tear is
// an explicit error.
func (r *Reader) Next(rec *Record) error {
	var fixed [recFixed]byte
	if _, err := io.ReadFull(r.r, fixed[:]); err != nil {
		if err == io.EOF {
			return io.EOF // clean end (record boundary)
		}
		return fmt.Errorf("kvops: torn record header at #%d: %w", r.nRead, err)
	}
	ts := binary.LittleEndian.Uint64(fixed[0:8])
	n := int(binary.LittleEndian.Uint16(fixed[8:10]))
	if n == 0 {
		return fmt.Errorf("kvops: empty chain at #%d", r.nRead)
	}
	if r.nRead > 0 && ts < r.lastTS {
		return fmt.Errorf("kvops: timestamp regression at #%d", r.nRead)
	}
	if cap(r.keys) < n {
		r.keys = make([][32]byte, n)
	}
	r.keys = r.keys[:n]
	for i := 0; i < n; i++ {
		if _, err := io.ReadFull(r.r, r.keys[i][:]); err != nil {
			return fmt.Errorf("kvops: torn keys at #%d: %w", r.nRead, err)
		}
	}
	r.lastTS = ts
	r.nRead++
	rec.TSMicros = ts
	rec.Keys = r.keys
	return nil
}
