package kvops

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"strconv"
)

// ConvertBailian turns one Alibaba Bailian usage-trace file (JSONL; the
// qwen-bailian-usagetraces-anon repo) into .kvops. Each trace line's
// 16-token hash_ids map 1:1 to one blob each (bytes-per-block is the
// RUN's model parameter, recorded in the kvops header, not here).
//
// Line shape (fields we consume):
//
//	{"timestamp": <seconds float>, "hash_ids": [int, ...], ...}
//
// The converter is COUNT-EXACT: requests out == lines in (the §11
// acceptance gate demands reproducing the published request counts
// exactly; a silently skipped line is converter corruption).

// ConvStats reports what a conversion produced — checked against the
// published trace counts by the convert subcommand.
type ConvStats struct {
	Requests  int64
	KeysTotal int64
}

// ConvertBailian streams in → out.
func ConvertBailian(in io.Reader, traceName string, out *Writer) (ConvStats, error) {
	sc := bufio.NewScanner(in)
	sc.Buffer(make([]byte, 1<<20), 64<<20) // hash_id lists can be long
	var st ConvStats
	var line struct {
		Timestamp float64  `json:"timestamp"`
		HashIDs   []uint64 `json:"hash_ids"`
	}
	keys := make([][32]byte, 0, 256)
	for sc.Scan() {
		raw := sc.Bytes()
		if len(raw) == 0 {
			continue
		}
		line.Timestamp, line.HashIDs = 0, line.HashIDs[:0]
		if err := json.Unmarshal(raw, &line); err != nil {
			return st, fmt.Errorf("bailian: line %d: %w", st.Requests+1, err)
		}
		if len(line.HashIDs) == 0 {
			return st, fmt.Errorf("bailian: line %d has no hash_ids", st.Requests+1)
		}
		if len(line.HashIDs) > maxChain {
			return st, fmt.Errorf("bailian: line %d has %d hash_ids (max %d)",
				st.Requests+1, len(line.HashIDs), maxChain)
		}
		if line.Timestamp < 0 {
			return st, fmt.Errorf("bailian: line %d has a negative timestamp", st.Requests+1)
		}
		keys = keys[:0]
		for _, id := range line.HashIDs {
			keys = append(keys, TraceKey(traceName, strconv.FormatUint(id, 10)))
		}
		tsUs := uint64(line.Timestamp * 1e6) //nolint:gosec // G115: timestamp checked ≥ 0 above
		if err := out.Write(tsUs, keys); err != nil {
			return st, fmt.Errorf("bailian: line %d: %w", st.Requests+1, err)
		}
		st.Requests++
		st.KeysTotal += int64(len(keys))
	}
	if err := sc.Err(); err != nil {
		return st, err
	}
	return st, out.Flush()
}
