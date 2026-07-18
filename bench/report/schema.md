# bench result schema v1

Every kvbench run (and the Python baseline drivers) emits **one JSON object
per line** (JSONL). `plot.py` renders both headline charts from committed
JSONL alone — a stranger with the results file regenerates the images.

## Conventions (repo-wide, non-negotiable)

- **Decimal GB/s** everywhere: `bytes / 1e9 / seconds` (so 12 GB/s = 96
  Gbit/s vs iperf3 — never binary GiB/s).
- **Payload-only goodput**: `goodput_gbytes_s` counts block bytes moved, not
  framing/headers.
- **Ratio-vs-ceiling is the quotable number**: `ratio_vs_ceiling` =
  goodput ÷ the same-rig iperf3/fio ceiling (`ceiling_gbytes_s`).
- Latencies in **microseconds**, from **scheduled** send time for open-loop
  (coordinated-omission-safe); closed-loop latencies are labeled `mode:
  "closed"`.
- `hit_rate` is an **OUTPUT** of replay (the store evicts itself), never an
  input.
- Provenance on every line: `git_sha`, `goos`, `goarch` (zipf/float
  determinism is per-platform), `seed`, `rig`.

## Fields

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | int | 1 |
| `kind` | string | sweep \| replay \| fill \| verify |
| `ts` | RFC3339 | run time (UTC) |
| `git_sha`, `rig` | string | provenance |
| `store` | string | kvblockd \| redis \| valkey \| nvmefs \| redis-py \| mooncake |
| `goos`, `goarch`, `seed` | | reproducibility |
| `cell` | object | `id, blob_bytes, batch_keys, streams, mix, skew, mode, rate_ops_s, rate_frac_of_ceiling, capacity_bytes, policy, trace, speedup` |
| `warmup_s`, `duration_s` | float | windows (measured = duration) |
| `ops` | map | per op (`get`/`put`/`exists`) → `{n, errors, mean_us, p50_us, p90_us, p99_us, p999_us, max_us}` |
| `ops_per_s` | float | |
| `goodput_gbytes_s` | float | payload-only, decimal |
| `ceiling_gbytes_s`, `ratio_vs_ceiling` | float | same-rig ceiling + ratio |
| `closed_ceiling_ops_s` | float | the open-loop sweep's denominator |
| `cpu` | object | `client_cores, daemon_cores, daemon_rss_bytes, daemon_source` |
| `sched` | object | `max_lag_us, p99_lag_us, saturated` (open-loop queue-growth) |
| `hit_rate` | float | replay output |
| `errors_total`, `verify_fails` | int | |
| `hgrm_paths` | []string | side-car HDR dumps |
| `saturated` | bool | excluded from the 2% repeatability gate |

## Gates keyed off this schema

- `kvbench report --check-repeat a.jsonl b.jsonl --tolerance 0.02` — pairs
  open-loop, non-saturated cells by `(store, cell.id, rate_frac)` and fails
  if any GET `p99_us` diverges >2% (SPEC-4 §11).
- Chart 1 reads `goodput_gbytes_s` + `ratio_vs_ceiling` + `cpu` from
  `mode:"closed"` headline cells (median of 3 runs).
- Chart 2 reads `hit_rate` (x) + GET `p50_us`/`p99_us` (y) from `kind:
  "replay"` runs at each seeded hit-rate point.
