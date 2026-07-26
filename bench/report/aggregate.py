#!/usr/bin/env python3
"""Cross-run aggregate for Chart-2 TTFT JSONL (methodology rule 7: median of
>=3 runs with min/max — no single-run numbers).

Input: N JSONL files, one per INDEPENDENT run of the same config (independent
HF job submissions = independent engine boots; bench/rigs/hf-gpu/submit-n.sh
produces `<tag>-run1.jsonl .. -runN.jsonl`). Records are the `kind:"ttft"`
shape run_ttft.py emits (one record per (arm, length) with ttft_p50_ms /
ttft_p95_ms over that run's reps).

Output: one summary JSONL line per cell — keyed (model, arm, series,
target_prefix_tokens, kv_cache_dtype) so a Llama file and a Qwen file can
never median into one cell — with median/min/max/spread% ACROSS runs for p50
and p95, the per-run values themselves, and the warm arm's verification
status. Records predating the dtype stamp default to "auto-bf16" (the
pre-stamp era ran vLLM's auto dtype). Mixing dtypes WITHIN one (model, arm,
series, target) group is refused outright — the same fp8 disclosure rule
plot.py enforces: an fp8 run may never median into a bf16 cell, split the
inputs per dtype. A human table goes to stderr; stdout stays
machine-readable.

Exit nonzero if any cell's p50 spread exceeds --tolerance (default 10%).

WHY 10% AND NOT THE 2% QUIET-RIG THROUGHPUT GATE: the 2% repeatability gate
(bench/METHODOLOGY.md; `report --check-repeat --tolerance 0.02`) governs
closed-loop steady-state throughput on a quiet dedicated pair — a 30s mean
over millions of ops with the daemon already warm. A GPU TTFT cell is the
median of 5 single-shot requests against a FRESHLY BOOTED engine on a shared
HF Jobs host: run-to-run variance includes engine boot state (CUDA graph and
allocator layout), host neighbors we cannot pin, and ms-scale timer
granularity on sub-100ms warm cells. n>=3 independent boots exists to catch
config drift and instability, not to enforce throughput-grade tightness;
cells that disagree by more than 10% are flagged for a rerun instead of
being silently medianed.

Usage:
  python3 bench/report/aggregate.py bench/results/rig-e/<tag>-run*.jsonl \
      [--tolerance 10] [--out summary.jsonl]
"""

import argparse
import json
import os
import sys


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return 0.0
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def load_cells(paths):
    """{(model, arm, series, target, kv_cache_dtype): [(record, source), ...]}.

    Refuses (hard exit) when one (model, arm, series, target) group spans
    more than one kv_cache_dtype — the fp8 disclosure rule, same as plot.py:
    an fp8 run must never median into a bf16 cell."""
    cells = {}
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("kind") != "ttft":
                    continue
                x = r.get("target_prefix_tokens") or r.get("prefix_tokens")
                if not x or r.get("arm") not in ("cold", "warm", "baseline"):
                    continue
                key = (r.get("model", "?"), r["arm"],
                       r.get("series", r["arm"]), x,
                       r.get("kv_cache_dtype", "auto-bf16"))
                cells.setdefault(key, []).append((r, os.path.basename(path)))
    groups = {}
    for key in cells:
        groups.setdefault(key[:4], set()).add(key[4])
    mixed = {g: sorted(d) for g, d in groups.items() if len(d) > 1}
    if mixed:
        for (model, arm, series, x), dts in sorted(mixed.items()):
            print(f"  mixed dtypes: {model} {arm}/{series} @{x}: {dts}",
                  file=sys.stderr)
        sys.exit("refusing to aggregate: input records mix engine KV dtypes "
                 "within the same (model, arm, series, target) cell — every "
                 "run medianed into a cell must share one kv-cache dtype "
                 "(fp8 disclosure rule, as plot.py; split the inputs per dtype)")
    return cells


def spread_stats(values):
    """(json_dict, raw_spread_pct). The JSONL carries the rounded spread for
    humans; the tolerance gate compares the RAW value — a 10.004% spread must
    fail a 10% gate, not round down to 10.00 and pass."""
    med = _median(values)
    lo, hi = min(values), max(values)
    spread = ((hi - lo) / med * 100.0) if med > 0 else 0.0
    return {"median_ms": med, "min_ms": lo, "max_ms": hi,
            "spread_pct": round(spread, 2), "per_run_ms": values}, spread


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="one JSONL file per independent run")
    ap.add_argument("--tolerance", type=float, default=10.0,
                    help="max allowed p50 spread%% (max-min)/median across runs "
                         "per cell (default 10 — see the docstring for why this "
                         "is looser than the 2%% quiet-rig throughput gate)")
    ap.add_argument("--out", default="",
                    help="also write the summary JSONL here (stdout regardless)")
    args = ap.parse_args()

    cells = load_cells(args.inputs)
    if not cells:
        sys.exit("no kind:'ttft' records found in the inputs")

    out_lines = []
    over = []
    print(f"{'model':40s} {'arm':9s} {'tokens':>7s} {'runs':>4s} "
          f"{'p50 med ms':>11s} {'min':>9s} {'max':>9s} {'spread%':>8s}",
          file=sys.stderr)
    for (model, arm, series, x, dtype), rs in sorted(cells.items()):
        p50s = [r.get("ttft_p50_ms", r.get("ttft_ms", 0.0)) for r, _ in rs]
        p95s = [r.get("ttft_p95_ms", 0.0) for r, _ in rs]
        p50, p50_spread_raw = spread_stats(p50s)
        p95, _ = spread_stats(p95s)
        rec = {
            "kind": "ttft-aggregate",
            "model": model, "arm": arm, "series": series,
            "target_prefix_tokens": x,
            "kv_cache_dtype": dtype,
            "runs": len(rs),
            "reps_per_run": [r.get("reps", 0) for r, _ in rs],
            "ttft_p50_ms": p50,
            "ttft_p95_ms": p95,
            "sources": [src for _, src in rs],
        }
        if arm == "warm":
            # A single unverified run poisons the cell — surfaced, never hidden.
            rec["warm_hits_verified_all_runs"] = all(
                r.get("warm_hits_verified") is True for r, _ in rs)
        line = json.dumps(rec, sort_keys=True)
        print(line)
        out_lines.append(line)
        flag = ""
        # Gate on the RAW spread, never the display-rounded one.
        if p50_spread_raw > args.tolerance:
            over.append((model, arm, x, p50_spread_raw))
            flag = "  << OVER TOLERANCE"
        if arm == "warm" and not rec["warm_hits_verified_all_runs"]:
            flag += "  !! warm_hits_verified_all_runs=false"
        print(f"{model:40s} {arm:9s} {x:>7d} {len(rs):>4d} "
              f"{p50['median_ms']:>11.1f} {p50['min_ms']:>9.1f} "
              f"{p50['max_ms']:>9.1f} {p50['spread_pct']:>8.2f}{flag}",
              file=sys.stderr)
    if args.out:
        with open(args.out, "w") as f:
            f.write("\n".join(out_lines) + "\n")

    n_single = sum(1 for rs in cells.values() if len(rs) < 3)
    if n_single:
        print(f"NOTE: {n_single}/{len(cells)} cells have <3 runs — below the "
              "median-of-3 rule (methodology rule 7); the chart must stay "
              "labeled accordingly.", file=sys.stderr)
    if over:
        print(f"FAIL: {len(over)} cell(s) exceed the {args.tolerance:.0f}% p50 "
              "spread tolerance across runs:", file=sys.stderr)
        for model, arm, x, sp in over:
            print(f"  {model} {arm} @{x}: {sp:.2f}%", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
