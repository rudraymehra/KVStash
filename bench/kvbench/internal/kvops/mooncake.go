package kvops

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"strconv"
)

// ConvertMooncake turns one Mooncake FAST25 trace file (JSONL) into
// .kvops. Mooncake blocks are 512 tokens; each hash_id splits into 32
// sub-keys TraceKey(trace, id, i) — 16-token-equivalent blobs preserving
// prefix-chain semantics (a 512-token hit = 32 consecutive sub-hits).
//
// Line shape (fields we consume):
//
//	{"timestamp": <ms int>, "hash_ids": [int, ...], ...}
const mooncakeSubKeys = 32

// ConvertMooncake streams in → out.
func ConvertMooncake(in io.Reader, traceName string, out *Writer) (ConvStats, error) {
	sc := bufio.NewScanner(in)
	sc.Buffer(make([]byte, 1<<20), 64<<20)
	var st ConvStats
	var line struct {
		Timestamp uint64   `json:"timestamp"`
		HashIDs   []uint64 `json:"hash_ids"`
	}
	keys := make([][32]byte, 0, 1024)
	for sc.Scan() {
		raw := sc.Bytes()
		if len(raw) == 0 {
			continue
		}
		line.Timestamp, line.HashIDs = 0, line.HashIDs[:0]
		if err := json.Unmarshal(raw, &line); err != nil {
			return st, fmt.Errorf("mooncake: line %d: %w", st.Requests+1, err)
		}
		if len(line.HashIDs) == 0 {
			return st, fmt.Errorf("mooncake: line %d has no hash_ids", st.Requests+1)
		}
		// Reject before the 32× amplification: a hostile line of ~5M ids
		// would drive a multi-GB allocation before Writer.Write's chain cap
		// fires (the ladder's memory-amplification MED).
		if len(line.HashIDs)*mooncakeSubKeys > maxChain {
			return st, fmt.Errorf("mooncake: line %d expands to %d keys (max %d)",
				st.Requests+1, len(line.HashIDs)*mooncakeSubKeys, maxChain)
		}
		keys = keys[:0]
		for _, id := range line.HashIDs {
			ids := strconv.FormatUint(id, 10)
			for i := 0; i < mooncakeSubKeys; i++ {
				keys = append(keys, TraceKey(traceName, ids, strconv.Itoa(i)))
			}
		}
		if err := out.Write(line.Timestamp*1000, keys); err != nil { // ms → µs
			return st, fmt.Errorf("mooncake: line %d: %w", st.Requests+1, err)
		}
		st.Requests++
		st.KeysTotal += int64(len(keys))
	}
	if err := sc.Err(); err != nil {
		return st, err
	}
	return st, out.Flush()
}
