# kvblockd benchmarks

> Prior raw transport/DRAM/NVMe gate numbers live in `bench/BENCHMARKS.md`
> (the running scoreboard); this file is the methodology-first launch view.
> Every headline claim's exact conditions and falsification line:
> [docs/CLAIMS.md](CLAIMS.md).

**Methodology first.** Read `bench/METHODOLOGY.md` (the 12 honesty rules)
before any number below. Every figure is absolute GB/s *and* %-of-same-rig-
ceiling; latencies are coordinated-omission-safe (open-loop, scheduled-time);
payloads are incompressible; kvblockd GETs are xxh3-checksum-verified in-line
(and `kvbench verify` regenerates + length-checks any stored blob); hit rates
are outputs of trace replay, never inputs. Raw JSONL + `.hgrm` + iperf3/fio
logs live in `bench/results/` and the charts regenerate from them alone
(`bench/report/plot.py`).

Scope: **TCP / commodity Ethernet only.** We never compare against RDMA-tier
systems (WEKA/VAST/Mooncake-RDMA) except in a clearly separate "different
league" note.

---

## The harness

`bench/kvbench` (`sweep | replay | fill | verify | convert | report`) drives
every store behind one `Target` interface, so kvblockd, Redis 7 / Valkey 8
(go-redis zero-copy), redis-py (the LMCache-shipped path), and an NVMe-fs
floor all replay the SAME op stream. The one-command local acceptance gate is
`bench/kvbench/loopback.sh` (grid sweep + injected-flip catch + converter
count-exactness + Go↔Python op-sequence parity). It exercises the
`report --check-repeat` gate but tolerates laptop jitter; the **hard 2%
repeatability gate runs on the quiet rig** (`report --check-repeat
--tolerance 0.02`), where scheduler noise isn't a factor.

---

## Chart 1 — throughput vs the field (measured 2026-07-19, Rig T)

GET-only, batch 32, closed-loop, warmed, **median of 3** at each store's
best stream count, on 2× c6in.8xlarge (50 GbE) in a cluster placement
group. iperf3 ceiling measured first on the same pair: **49.8 Gbit/s =
6.225 GB/s** — drawn on the chart. Raw JSONL: `bench/results/rig-t/`;
render: `python bench/report/plot.py chart1 --in bench/results/rig-t/*.jsonl`.
kvblockd ran the full 7-point stream curve; baselines ran streams {8,32,64}
(single-threaded Redis costs ~4 min/cell — median-of-3 held on every
published cell; disclosed). xxh3 verification ON for kvblockd.

| Store | 0.44 MiB GB/s | %-ceiling | 2.5 MiB GB/s | %-ceiling |
|---|---|---|---|---|
| **kvblockd (DRAM)** | **6.22** | **100%** | **6.23** | **100%** |
| Valkey 8 (go-redis zero-copy) | 2.38 | 38% | 1.83 | 29% |
| Redis 7 (go-redis zero-copy) | 2.26 | 36% | 1.81 | 29% |
| Redis 7 (redis-py 8.0.1, the LMCache-shipped client) | 0.88 | 14% | 0.83 | 13% |

NVMe-resident bars (same-host, i7i.8xlarge tier session, 2026-07-19):
one-volume storm **5.22 GB/s**, two-volume **10.57 GB/s** (mixed
DRAM+NVMe pool, disclosed; fio read ceiling 4.48 GB/s/device); post-kill-9
warm storm 10.58 GB/s. Mooncake-TCP: not run this session (timebox);
standing re-run offer per methodology rule 4.

**≥10× vs redis-py gate: 7.1× at the best comparable cell (6.22 vs 0.88),
5.6× median-of-everything — BELOW 10× on this link, reported honestly.**
The gate is *ceiling-limited here, not software-limited*: kvblockd sits at
**100% of the 50 GbE wire** and cannot score higher on it, while redis-py
uses 14% of the same wire. On the measured 100 GbE pair (c7gn session
below, 12.67 GB/s), the same client-bound redis-py bar implies ~14× — the
multiple is a property of the link. A full-matrix 100 GbE re-run is the
certification path for the ≥10× headline.

Prior measured transport ceilings (this rig family):

- 100 GbE (c7gn pair): kvblockd GET **12.67 GB/s ≈ 102% of the iperf3
  ceiling, verify ON** — the 10+ GB/s target, measured.
- 50 GbE (c6in pair): xferspike 6.27 GB/s = 100% of the iperf3 ceiling at 0.79 cores; kvblockd GET 6.37 GB/s ≈ 102% (see bench/BENCHMARKS.md for the split).
- DRAM-tier same-host gates (c7i): 0.96–0.97× the raw-GET same-shape ceiling;
  EXISTS p99 705 µs under 8 saturated lanes; zero blob-band allocs on the GET
  path.
- NVMe (i4i, A3): device ceiling 2.99 GB/s, Go threadpool 98.3% of it — the
  literal ≥6 GB/s line is not printable on AWS instance-store hardware; the
  i7i tier session quotes %-of-ceiling (A3 stays open pending faster
  hardware).

---

## Chart 2 — TTFT: reload from kvblockd vs recompute (measured 2026-07-26)

NVIDIA A10G (HF Jobs a10g-large), Llama-3.1-8B-Instruct bf16, vLLM 0.25.1 +
the native connector (`vllm_kvblockd` 0.1.0), **unshaped loopback link —
disclosed on-chart**. Two-phase isolation: populate → **vLLM restart** →
measure, `--no-enable-prefix-caching` in both phases, so a warm hit's KV has
exactly one possible source: kvblockd over TCP into a fresh engine. TTFT =
first SSE token; p50 of n=5 reps per point, warmup discarded. Raw JSONL:
`bench/results/rig-e/chart2-ttft-{run5,baseline}.jsonl`; render exactly:
`python bench/report/plot.py chart2 --in bench/results/rig-e/chart2-ttft-run5.jsonl bench/results/rig-e/chart2-ttft-baseline.jsonl`
— NOT a wildcard: `rig-e/` keeps the superseded run3/run4 files for history
(run3's warm arm is flagged UNVERIFIED; run4 predates the load-path rework),
and a wildcard render would median them into the current cells.
Honesty note against methodology rule 7: this table is ONE run of n=5 reps
per point (per-rep values published in `ttft_all_ms`); an independent
same-config re-run is queued with the next paid session, and until it lands
the table is labeled single-run rather than silently treated as final.

<!-- TODO(multi-run reruns): when the n>=3 independent submissions land
     (bench/rigs/hf-gpu/submit-n.sh -> bench/results/rig-e/<tag>-run{1..N}.jsonl),
     replace the single-run honesty note above with the n>=3 protocol:
     median across runs charted with min/max whiskers (plot.py), per-cell
     spread gated at <=10% by bench/report/aggregate.py — looser than the 2%
     quiet-rig throughput gate because each run is a fresh engine boot on a
     shared GPU host (boot-state + neighbor variance), not a 30s steady-state
     mean; aggregate.py's docstring carries the full justification. Quote the
     measured spread% here. The same marker covers the Qwen long-context
     table below (its "single-run disclosures apply as above" line). -->

**n=3 independent runs** (each its own engine boots; median across runs, per-run
values and min/max in the aggregate JSONL; worst p50 spread 2.8%). The cold
arm runs the connector's **gathered-slots** store path (0.3.x): misses stage
through batched GPU gathers + async DMA into pinned slots, and the TCP put
happens off the critical path — the store path each run took is stamped in
its own receipts (`connector store path: gathered-slots`).

| prefix | recompute (no connector)² | recompute, connector on¹ | **kvblockd reload** | vs pure | vs serving¹ |
|---|---|---|---|---|---|
| 1k  | 269 ms  | 286 ms  | **78 ms**  | **3.4×** | 3.7× |
| 4k  | 1070 ms | 1124 ms | **178 ms** | **6.0×** | 6.3× |
| 8k  | 2212 ms | 2318 ms | **304 ms** | **7.3×** | 7.6× |
| 16k | 5045 ms | 5260 ms | **576 ms** | **8.8×** | 9.1× |

² baseline is n=1 (its measured rep spread is <1%; an n=3 baseline rides the
next paid session). Movements vs the previous n=3 table, stated plainly:
the cold arm dropped a further ~30% — **store-on-miss overhead is now
1.04–1.06× vs pure recompute** (was 1.45–1.51×), i.e. filling the cache
costs ~4–6% of a request, receipt-attributed to the gathered-slots staging
path that replaced ~2×L×blocks blocking host copies with batched gathers
and async pinned DMA. The "vs serving" column shrank as a RESULT (the
serving baseline got faster); vs-pure is the stable comparison. The warm
arm recovered ~4% of the earlier wave's regression (599 → 576 at 16k,
spread 1.7%) after the per-recv deadline re-arm was hoisted off the hot
loop; the pipelined-load wave targets a floor well below it.

The warm column improved 3.6–4.8× between run 4 and run 5 by rebuilding the
connector's load path (pinned staging, chunked DMA, sharded drain — commit
8b29f83); the recompute columns are unchanged within noise, as they must be.
Effective reload bandwidth at 16k: ~3.8 GB/s through a Python client with
per-block xxh3 verification on. Run-5 hit verification uses the exact-count
gate: every rep's `kvb_hits_total` delta equals its expected block count
(16k: four reps at 1023/1023 and one at 1024/1024 — that rep's prompt
calibrated one filler unit longer, and the gate tracks each rep's own
measured token count, not a nominal target), recorded per-rep in the JSONL.

¹ the cold arm of the two-phase run serves WITH the connector, so every miss
also pays the synchronous store-on-miss write — the steady-state serving
shape. The no-connector column is the separate control run. Quote the
conservative "vs pure" column unless the serving context is explicit.

**Attribution is arithmetic, not inferred:** `kvb_hits_total` grew by
exactly each rep's own expected block count at every length (16k: 1023+
1023+1023+1024+1023 = 5116 hits — one rep's prompt calibrated one block
longer, and the gate is per-rep against the measured token count, never
a nominal target). A warm rep that fails to grow the counter is
recorded `warm_hits_verified: false`, drawn red, and fails the job — the
harness cannot emit an unattributed warm number (`run_ttft.py --selftest`
proves the gates; an earlier run's 16× "speedup" with zero store hits was
rejected under exactly this rule and never published).

**What this does and does not show.** Reloading KV from kvblockd beats
recompute at every measured length on this box — the first independent,
methodology-open measurement of TCP KV-reload vs recompute we know of (every
prior figure in this category is vendor-claimed). The link is loopback:
network transfer time is near zero, and a real NIC adds wire time (~0.7 s
for the 16k prefix's ~2 GB at 25 GbE — still well inside the 5 s recompute
budget, but that number must be measured, not asserted). The tc-shaped
25/50 Gbps points, the hit-rate sweep, and the Bailian-trace A4 datapoint
remain with the AWS rig (`bench/rigs/aws-gpu/`), gated on GPU quota.

**A4 status:** the loopback evidence is a PASS at every length at 100%
prefix hit; the pre-registered A4 verdict (measured Bailian hit-rate band,
emulated link) stays OPEN until the AWS rig runs.

### The long-context cells (measured 2026-07-27, same rig, Qwen2.5-7B-Instruct)

Same harness, same gates, KV-lighter model (56 KiB/token GQA-4) at its
NATIVE 32k context — no rope scaling, no config overrides. Raw JSONL:
`bench/results/rig-e/chart2-ttft-qwen32k{,-baseline}.jsonl`; chart:
`chart2-qwen32k.png`. Every warm rep passed the exact-count gate — 1024
blocks per rep @16k, 1999–2000 per rep @32k (per-prompt calibration lands
within a token, so reps differ by one block; the gate is exact against each
rep's own measured count) — and every record is path-stamped `chunked-slab`.

**n=3 independent runs**, gathered-slots cold arm + pipelined-slab warm arm
(both stamped in every run's receipts), same protocol as above:

| prefix | recompute (no connector)² | recompute, connector on¹ | **kvblockd reload** | vs pure | vs serving¹ |
|---|---|---|---|---|---|
| 16k | 4,588 ms | 4,701 ms | **300 ms** (294–319) | **15.3×** | 15.7× |
| 32k | 10,923 ms | 10,941 ms | **535 ms** (522–536) | **20.4×** | 20.5× |

**The fp8 cells (kv_cache_dtype=fp8_e4m3, n=3, token-identity certified):**
halving the KV bytes halves the reload — every arm of these runs (baseline,
cold, warm) ran the SAME dtype, and each run's certification is machine-
checked before any warm number exists: 8 equivalence prompts recorded on
engine #1, recorded TWICE to exclude kernel-nondeterministic prompts (one
was, in one run — disclosed in that run's summary record), then replayed
token-identical on the restarted engine.

| prefix | fp8 recompute | **fp8 reload** | vs same-dtype recompute | vs bf16 recompute |
|---|---|---|---|---|
| 16k | 4,536 ms | **216 ms** (213–217) | **21.0×** | 21.2× |
| 32k | 10,538 ms | **399 ms** (398–406) | **26.4×** | 27.4× |

Raw: `bench/results/rig-e/chart2-ttft-qwen-fp8-{run1,run2,run3,baseline}.jsonl`.
The bf16 column is context, not the claim: it compares across dtypes (a
user choosing fp8 changes their numerics regardless of any store).

Store-on-miss overhead on this model is **1.00–1.02×** — at 56 KiB/token
the gathered store path makes filling the cache effectively free, which is
also why the "vs serving" column nearly equals "vs pure" (the serving
baseline stopped paying a cache tax, not the reload getting slower).
Worst-case ratios from the slow end of the ranges: 4,588/319 = 14.4× at
16k (spread 8.6%, within the 10% gate); 10,923/536 = 20.4× at 32k (2.7%).
The previous n=3 table (369/685 ms, 12.4×/15.9×, write-behind cold arm)
and the earlier single-run cells (321/636 ms) are superseded by this one.

Why the multiple grows vs the Llama table above: the speedup is
`prefill(L) / reload(L)` — prefill grows superlinearly with context while
reload grows linearly with KV bytes, and this model carries less than half
the KV per token. Same physics every vendor's 100k-context headline runs
on; stated here with the formula instead of hidden behind it. The same
loopback and single-run disclosures apply as above.

### The faster-GPU cells (measured 2026-07-27, NVIDIA A100-SXM4-80GB)

Same harness, same gates, same models, HF Jobs `a100-large`. Published
precisely because the multiple SHRINKS here: `speedup = prefill(L) /
reload(L)`, prefill got ~3.4× faster on this GPU while reload is bound by
KV bytes, not FLOPs. A vendor picking the slow-GPU/long-context corner can
print any multiple it likes; both corners are on this page. Raw JSONL:
`bench/results/rig-e/chart2-ttft-a100-{llama-run1,llama-baseline,qwen32k-run1,qwen-baseline}.jsonl`.
Single run of n=5 reps per point (the n≥3 protocol above rides the next
paid session); every warm rep passed the exact-count hit gate and is
path-stamped `chunked-slab`.

| model @ prefix | recompute (no connector)² | recompute, connector on¹ | **kvblockd reload** | vs pure | vs serving¹ |
|---|---|---|---|---|---|
| Llama-8B @ 8k   | 650 ms   | 1,812 ms | **304 ms** | **2.1×** | 6.0× |
| Llama-8B @ 16k  | 1,470 ms | 3,836 ms | **582 ms** | **2.5×** | 6.6× |
| Qwen-7B @ 16k   | 1,355 ms | 2,818 ms | **300 ms** | **4.5×** | 9.4× |
| Qwen-7B @ 32k   | 3,215 ms | 6,090 ms | **567 ms** | **5.7×** | 10.7× |

These are the fastest absolute reloads we have measured (Qwen 16k prefix
resident in 300 ms), and the smallest multiples — both statements are true
and both are the point. Below ~4k on this GPU the multiple approaches 1×
and reload stops being worth the trip (see "When NOT to use kvblockd").
Operational disclosure: the first A100 Llama attempt aborted itself — this
GPU misses fast enough that the write-behind queue's 1 GiB default
overflowed and the populate gate refused to continue on `dropped=2890`
rather than publish a silently-thinner warm set; the run above uses the
now-configurable 4 GiB queue and recorded `dropped=0 failed=0`.

---

## When NOT to use kvblockd

Below the crossover hit rate, recompute is cheaper than a remote fetch —
don't deploy a remote KV cache for that workload. From this table's own
measured numbers, today's break-even is stark: with the current synchronous
store-on-miss path, `h·warm + (1−h)·cold_with_connector = pure_recompute`
solves to **h ≈ 46–55% at every length** — below roughly half your requests
hitting, installing the connector makes mean TTFT *worse*. Write-behind
stores (next connector release) collapse the miss overhead and move that
break-even to a few percent. The hit-rate-swept chart with the crossover
region drawn is pre-registered for the shaped-link rig; until it exists, no
single-hit-rate number here should be read as a deployment recommendation.
The `bench/e2e/economics.py` model dollarizes the crossover ($/GB moved vs
$/GPU-sec saved per hit, same-AZ vs cross-AZ) and now derives its measured
sections from these tables' committed medians (`bench/results/rig-e/`):
**4.4/4.2/10.2 GPU-s saved per hit** (Llama@16k, Qwen@16k, Qwen@32k —
floored from the raw 4.49/4.27/10.29, never rounded up; the floored column
is the quotable one), the measured ~46% sync-store break-even, and the
PROJECTED few-percent write-behind break-even as an explicitly separate
mode.

---

## Reproduce

```bash
# Local acceptance (no cloud):
bench/kvbench/loopback.sh

# Chart 1 (~$12 spot):
bench/rigs/aws-transport/provision.sh && bench/rigs/aws-transport/run-chart1.sh && \
  bench/rigs/aws-transport/teardown.sh
python bench/report/plot.py chart1 --in bench/results/rig-t/*.jsonl --out chart1.png

# Chart 2 (HF Jobs, a10g-large — the rig that produced the published table):
DRY_RUN=1 bench/rigs/hf-gpu/submit.sh       # inspect the exact job first
MODEL=meta-llama/Llama-3.1-8B-Instruct bench/rigs/hf-gpu/submit.sh   # two-phase run
BASELINE_ONLY=1 MODEL=meta-llama/Llama-3.1-8B-Instruct bench/rigs/hf-gpu/submit.sh
python bench/report/plot.py chart2 \
  --in bench/results/rig-e/chart2-ttft-run5.jsonl bench/results/rig-e/chart2-ttft-baseline.jsonl \
  --out chart2.png
# (bench/rigs/aws-gpu/ is the future shaped-link runbook, not this chart's source.)
```
