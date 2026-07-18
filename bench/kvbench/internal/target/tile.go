package target

import "context"

// The caller-tiles rule: pkg/client (and any capped driver) rejects batches
// over the negotiated max; these helpers split chains/batches into tiles.
// lim ≤ 0 means unbounded (one call).

// TiledExists probes the chain tile by tile and STOPS at the first tile
// whose consecutive-hit count is shorter than the tile — later tiles cannot
// extend a broken prefix, so the round trips are saved (the adaptive
// replayer's hot path).
func TiledExists(ctx context.Context, t Target, lim int, chain [][32]byte) (int, error) {
	if lim <= 0 || len(chain) <= lim {
		return t.BatchExists(ctx, chain)
	}
	total := 0
	for off := 0; off < len(chain); off += lim {
		end := min(off+lim, len(chain))
		k, err := t.BatchExists(ctx, chain[off:end])
		if err != nil {
			return total, err
		}
		total += k
		if k < end-off {
			break // prefix broken inside this tile
		}
	}
	return total, nil
}

// TiledGet fetches keys in tiles, filling the matching dst windows.
func TiledGet(ctx context.Context, t Target, lim int, keys [][32]byte, dst [][]byte) ([]Status, error) {
	if lim <= 0 || len(keys) <= lim {
		return t.BatchGet(ctx, keys, dst)
	}
	out := make([]Status, 0, len(keys))
	for off := 0; off < len(keys); off += lim {
		end := min(off+lim, len(keys))
		st, err := t.BatchGet(ctx, keys[off:end], dst[off:end])
		if err != nil {
			return out, err
		}
		out = append(out, st...)
	}
	return out, nil
}

// TiledPut stores keys in tiles.
func TiledPut(ctx context.Context, t Target, lim int, keys [][32]byte, blobs [][]byte) ([]Status, error) {
	if lim <= 0 || len(keys) <= lim {
		return t.BatchPut(ctx, keys, blobs)
	}
	out := make([]Status, 0, len(keys))
	for off := 0; off < len(keys); off += lim {
		end := min(off+lim, len(keys))
		st, err := t.BatchPut(ctx, keys[off:end], blobs[off:end])
		if err != nil {
			return out, err
		}
		out = append(out, st...)
	}
	return out, nil
}
