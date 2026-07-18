# kvblockd benchmark methodology — the 12 honesty rules

Every number kvblockd publishes obeys these rules (SPEC-4 §9). They exist so
the charts survive HN-comment-section forensics: each rule closes a specific
way benchmarks lie.

1. **Repro shipped with every number.** Every figure ships with its repro
   script, instance type, spot price, and `VERSIONS.lock` hash; anyone
   re-runs Chart 1 for ~$12.
2. **Ceiling always shown.** No throughput number without the same-rig
   iperf3 (or fio) ceiling beside it — absolute GB/s *and* %-of-ceiling.
3. **Two-client rule.** Baselines get both the ecosystem-default client
   (redis-py) and the strongest client we can find (go-redis zero-copy), as
   separate labeled bars. The gap between them is the client-side story.
4. **Best-effort tuning for competitors.** Baseline configs are published;
   if a Mooncake/Valkey maintainer suggests a better config we re-run and
   update — a standing README offer. An unreproducible bar ships as
   "unreproducible, configs+errors attached", never silently dropped.
5. **Coordinated-omission-safe latency only.** Open-loop, scheduled-time
   accounting for every p99/p999 claim; closed-loop latencies are labeled
   as such.
6. **Hit-rate-swept curves always.** Single-hit-rate TTFT claims are banned.
   The "when NOT to use kvblockd" region is printed on the headline chart.
7. **Median of ≥3 runs with min/max.** No single-run numbers; warm-state and
   fill-level disclosed.
8. **Payload-only goodput, corruption-checked.** Incompressible payloads;
   every GET is checksum-verified — a benchmark that returns wrong bytes is a
   failed benchmark.
9. **Production traces over synthetic** wherever the claim is about hit rates
   or policies (Bailian, Mooncake FAST25). Synthetic only for pure transport
   cells.
10. **Never compare against RDMA-tier numbers** (WEKA/VAST/Mooncake-RDMA)
    except in a clearly separate "different league" table. Every claim is
    scoped to TCP/Ethernet.
11. **Emulated links always disclosed on-chart** (the tc netem/tbf classes);
    Mac numbers never leave dev docs.
12. **Raw artifacts published** (JSONL, `.hgrm`, fio/iperf3 logs) — the
    charts regenerate from them alone.

## How the harness enforces them mechanically

| Rule | Enforcement |
|---|---|
| 2, 12 | `ratio_vs_ceiling` in every JSONL line; `plot.py` renders from JSONL alone |
| 3 | `redis-py` bar (`baselines/redis_py_driver`) + `go-redis` bar (`--target redis`) |
| 5 | `loadgen` open-loop measures from **scheduled** send time (CO-safety unit test) |
| 6 | Chart 2 shades the sub-crossover region; single-point TTFT has no code path |
| 7 | `report --check-repeat --tolerance 0.02` — the executable 2% gate |
| 8 | deterministic `kvbench-payload-v1` blobs; `verify` regenerates + byte-compares (length-checked). Note: `verify` corruption-checks the **kvblockd** path; baseline bars quote throughput, and every kvblockd GET is xxh3-verified in-line unless `--noverify` isolates that cost |
| 9 | trace converters with `--expect-requests` count-exactness |
| 11 | the GPU rig stamps `gpu`/`model`/`tc_link` into each JSONL line; `plot.py` chart2 reads them into the conditions box (never hardcoded) |
