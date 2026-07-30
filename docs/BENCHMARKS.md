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
first SSE token; p50 of n=5 reps per point per run, **median across n=3
independent runs** (each run its own two-phase job with fresh engine
boots). Raw JSONL: `bench/results/rig-e/chart2-ttft-a10g-final-run{1,2,3}.jsonl`
(warm+cold) with `chart2-ttft-baseline.jsonl` (pure recompute); render
exactly:
`python bench/report/plot.py chart2 --in bench/results/rig-e/chart2-ttft-a10g-final-run1.jsonl bench/results/rig-e/chart2-ttft-a10g-final-run2.jsonl bench/results/rig-e/chart2-ttft-a10g-final-run3.jsonl bench/results/rig-e/chart2-ttft-baseline.jsonl`
— NOT a `rig-e/` wildcard: the directory keeps every superseded run for
history (run3's warm arm is flagged UNVERIFIED; run4 predates the load-path
rework; run5 and the `a10g-postwave-run{1,2,3}` table are superseded below),
and a wildcard render would median history into the current cells.
Cross-run protocol (methodology rule 7): per-cell p50 spread gated at ≤10%
by `bench/report/aggregate.py` — looser than the 2% quiet-rig throughput
gate because each run is a fresh engine boot on a shared GPU host
(boot-state + neighbor variance), not a 30 s steady-state mean;
aggregate.py's docstring carries the full justification. Worst cell in this
table: **7.4%** (16k warm); per-run values and min/max are in the aggregate
output.

The warm arm runs the connector's **pipelined-slab** load path
(double-buffered slab halves on a dedicated copy stream) and the cold arm
its **gathered-slots** store path (0.3.x): misses stage through batched GPU
gathers + async DMA into pinned slots, and the TCP put happens off the
critical path. The load path is stamped in every committed record
(`path=pipelined-slab` in the `connector` field); the store path is
attributed in each run's job log (`kvblockd store path: gathered-slots`) —
a log line, not a committed-JSONL stamp, stated so the evidence chain is
honest about which claims the committed artifacts can and cannot carry.

| prefix | recompute (no connector)² | recompute, connector on¹ | **kvblockd reload** | vs pure | vs serving¹ |
|---|---|---|---|---|---|
| 1k  | 269 ms  | 285 ms  | **78 ms**  | **3.4×** | 3.6× |
| 4k  | 1070 ms | 1125 ms | **178 ms** | **6.0×** | 6.3× |
| 8k  | 2212 ms | 2321 ms | **264 ms** | **8.4×** | 8.8× |
| 16k | 5045 ms | 5261 ms | **497 ms** | **10.2×** | 10.6× |

² baseline is n=1 (its measured rep spread is <1%; an n=3 baseline rides the
next paid session). Movements vs the previous n=3 table
(`a10g-postwave-run{1,2,3}`: 78/178/304/576), stated plainly: the warm arm
improved a further 13–14% at 8k/16k (304 → 264, 576 → 497) —
receipt-attributed to the pipelined-slab load path that replaced
chunked-slab's serial copy-then-scatter with overlapped halves — and is
unchanged within noise at 1k/4k, where one slab half already covers the
transfer. The recompute columns are unchanged within noise, as they must
be. **Store-on-miss overhead stays 1.04–1.06× vs pure recompute** (was
1.45–1.51× before the gathered-slots wave): filling the cache costs ~4–6%
of a request. The "vs serving" column stays close to "vs pure" as a RESULT
(the serving baseline stopped paying a cache tax); vs-pure is the stable
comparison.

The warm column's earlier history, every file still committed: run 4's
per-block loads → run 5's chunked-slab rebuild (pinned staging, chunked
DMA, sharded drain — commit 8b29f83; a 3.6–4.8× improvement) → the
gathered-slots store wave (`a10g-postwave`, load path still chunked-slab)
→ the pipelined-slab table above. Effective reload bandwidth at 16k: ~4.2 GB/s through a Python client
with per-block xxh3 verification on. Hit verification uses the exact-count
gate in every run: each rep's `kvb_hits_total` delta must equal its own
expected block count (a rep whose prompt calibrates one filler unit longer
is gated at 1024, not a nominal 1023 — the gate tracks each rep's measured
token count), recorded per-rep in the JSONL.

¹ the cold arm of the two-phase run serves WITH the connector, so every miss
also pays the synchronous store-on-miss write — the steady-state serving
shape. The no-connector column is the separate control run. Quote the
conservative "vs pure" column unless the serving context is explicit.

**Attribution is arithmetic, not inferred:** `kvb_hits_total` grew by
exactly each rep's own expected block count at every length in every run —
per rep in each run's JSONL as `pairs[].hit_delta`, gated against that
rep's `expected_hit_blocks` (never a nominal target), with
`kvb_hit_delta_total` as the record's run total. A
warm rep that fails to grow the counter is
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
`bench/results/rig-e/chart2-ttft-qwen-final-run{1,2,3}.jsonl` with
`chart2-ttft-qwen32k-baseline.jsonl` (the superseded single-run
`chart2-ttft-qwen32k.jsonl` stays for history); chart: `chart2-qwen32k.png`.
Every warm rep passed the exact-count gate — 1024 blocks per rep @16k,
1999–2000 per rep @32k (per-prompt calibration lands within a token, so
reps differ by one block; the gate is exact against each rep's own measured
count).

**n=3 independent runs**, gathered-slots cold arm + pipelined-slab warm arm
(load path stamped `path=pipelined-slab` in every committed record; the
store path is attributed in each run's job log), same protocol as above:

| prefix | recompute (no connector)² | recompute, connector on¹ | **kvblockd reload** | vs pure | vs serving¹ |
|---|---|---|---|---|---|
| 16k | 4,588 ms | 4,701 ms | **300 ms** (294–319) | **15.3×** | 15.7× |
| 32k | 10,923 ms | 10,941 ms | **535 ms** (522–536) | **20.4×** | 20.4× |

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
loopback disclosure applies as above; baselines are n=1 as above.

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

### The long-context cells (measured 2026-07-29, EC2 g5.2xlarge, single A10G — Rig G)

Same harness, same gates, driven onto a rented EC2 box by `bench/rigs/aws-gpu/`
(the HF-Jobs measurement core reused byte-for-byte; `rig=ec2-g5-a10g` stamped
per record). Model: Qwen2.5-7B-Instruct-1M served in its model-card-sanctioned
standard-attention mode — `dual_chunk_attention_config` and the companion
`sparse_attention_config.json` are removed from a local snapshot (DCA has no
backend on vLLM 0.25.x; the card certifies standard attention ≤ 262,144
tokens), the IDENTICAL snapshot serves both arms, and the surgery is stamped
`model_surgery=` in every record. fp8_e4m3 KV, `max_num_batched_tokens=8192`
frozen across every engine boot, loopback (disclosed), per-rep exact-count hit
gate, token-identity certification around every restart.

| prefix | recompute (no connector)² | **kvblockd reload** | vs pure | best rep |
|---|---|---|---|---|
| 64k  | 27,666 ms  | **886 ms** (n=3, 0.8%) | **31.2×** | 31.8× |
| 96k  | 51,820 ms  | **1,317 ms** (n=2, 0.5%) | **39.4×** | 39.6× |
| 131k | 82,883 ms  | **1,787 ms** (n=2, 1.9%) | **46.4×** | 47.0× |
| 160k | 120,831 ms | **2,228 ms** (n=1, reps within 1.4% of median; 2.7% peak-to-peak) | **54.2×** | 55.0× |

² baselines are one run × (2+1) reps (rep spreads ≤0.1%); calibration on the
same box first reproduced the published HF-Jobs 16k/32k fp8 cells within
8–13% (233.7 ms / 451.3 ms warm; ~2.4 points of the 32k gap is prompt length
— this cell ran 32,768 tokens vs the published 32,000), proving the ported
stack before any long cell was measured. Spreads above are peak-to-peak of
run medians over the cell median, the same convention throughout. Raw JSONL: `bench/results/rig-g/chart2-ttft-a10g-long-*.jsonl`.

**Why some cells are n<3, stated plainly:** the token-identity gate REFUSED
four run attempts across three run slots (96k run3, 128k run3 twice, 160k
run2) — in each refusal a DIFFERENT certification prompt diverged on engine
replay (and the kernel-determinism control excluded 1–4 prompts per run,
typically 1–2, disclosed in each record's token_identity stamp and in the
banked refusal records: `bench/results/rig-g/refusals/` carries each refused
run's EQUIVJSONL — the 128k retry overwrote its first attempt's log under
the same tag, so the second attempt's records are the ones banked). The
refused runs' warm timings were never recorded; the store's byte delivery was xxh3-verified
exact in refused and certified runs alike. We publish the refusals rather
than loosen a pre-registered gate. 192k was attempted and REFUSED BY VRAM at
boot (vLLM: max feasible 179,392 tokens on 24 GB at these settings — the
boot-log excerpt is banked at `bench/results/rig-g/refusals/192k-vram-refusal.log`)
— 160k is this GPU's top rung, and the refusal cost $0.20.

The multiple grows with context by the same physics as every table above
(`speedup = prefill(L)/reload(L)`); the GPU is deliberately modest and the
GPU class is disclosed on every chart — a faster GPU shrinks the multiple
(see the A100 rows). The entire session that produced this table cost ~$4.7
of on-demand EC2 (3.72 box-hours × $1.212 + storage; the per-arm ledger is
banked at `bench/results/rig-g/session-ledger.csv`).

### The quarter-million-token cells (measured 2026-07-29 UTC, EC2 g6e.2xlarge, single L40S — Rig G)

The pre-registered L40S session, run hours after the A10G table above — the
same harness, gates, model surgery, fp8, loopback, and `mnbt=8192` freeze.
Calibration first: the L40S prefills **2.8× faster** than the A10G at
16k/32k (measured; the pre-registration's largest unknown), which per the
disclosed formula shrinks every multiple — and stretches the reachable
context to the model card's full certified window.

| prefix | recompute (no connector)² | **kvblockd reload** | vs pure | best rep |
|---|---|---|---|---|
| 131k | 26,621 ms | **1,224 ms** (n=1) | **21.7×** | 21.9× |
| 160k | 37,678 ms | **1,510 ms** (n=1) | **25.0×** | 25.1× |
| **262,144** | **85,693 ms** | **2,474 ms** (n=2, 0.3%) | **34.6×** | 34.7× |

² baseline: one run × (3+1) reps per point, five-point curve banked
(`chart2-ttft-l40s-base-fp8-run1.jsonl`). The 262,144 cell runs at
the model card's certified standard-attention ceiling — with a disclosed
tokenizer overshoot: the prompt landed at 262,146 prefix tokens (+2 over
target; the baseline row +4), visible in every banked record — resume a
quarter-million-token session in 2.5 s. A third 256k run was refused by the
token-identity gate (the fifth refusal of the campaign, a different prompt
again; record banked under `refusals/`); the two certified runs agree
within 0.3%. Read the two GPU tables together: the A10G carries the largest
multiples (slower prefill), the L40S the largest contexts (more VRAM,
faster prefill) — both corners published, per the pre-registration, with
the same store underneath.

### The real-network cells (measured 2026-07-30, two nodes over a real NIC — Rig G)

The loopback caveat, retired. A separate network-optimized store node
(r6in.2xlarge) ran kvblockd; a g6e.2xlarge GPU node ran vLLM and the
connector; every reload byte crossed a real VPC NIC between them. The
harness refuses to enter this mode unless the store is on another host
(a 127.x / same-host address dies at boot), and each run's log proves it
entered external-daemon mode or the run is voided. iperf3 ceiling measured
FIRST on the same pair, store→GPU (the reload direction), banked as its own
artifact and quoted beside every cell. **bf16** (fp8 kernels are nondeterministic on these GPUs and the
token-identity gate rightly refused them — the refusal record is banked
under `refusals/`; bf16 certified cleanly, and the larger payload makes the
wire matter *more*, not less). Qwen2.5-7B-1M, standard-attention mode,
per-rep exact-count hit gate, token-identity certified.

| context | link (iperf3 ceiling) | recompute² | **kvblockd reload** | vs pure |
|---|---|---|---|---|
| 32k  | ~19 Gbit burst (2.40 GB/s) | 3,608 ms  | **968 ms**    | **3.7×** |
| 128k | ~19 Gbit burst (2.40 GB/s) | 30,562 ms | **4,030 ms**  | **7.6×** |
| 128k | shaped 5 Gbit (0.61 GB/s)  | 30,562 ms | **12,586 ms** | **2.4×** |

² bf16 pure-recompute baseline on the GPU node, one run × (2+1) reps.
Raw JSONL: `bench/results/rig-g/chart2-ttft-twonode-bf16-*.jsonl`. The iperf3
ceilings are banked as `bench/results/rig-g/iperf3-twonode-{unshaped,shaped5g}-store2gpu.json`
(2.40 / 0.61 GB/s, store→GPU); tc state rides each record's `tc_link` stamp.

**What this shows, stated plainly.** Reload beats recompute over a real
network at every point, and the multiple **compresses as the link narrows**
— 7.6× on a burst link, 2.4× when shaped to 5 Gbit — because reload is
wire-bound while recompute is not (`speedup = prefill / reload`, and reload
= KV bytes / link rate). This is the honest deployable number the loopback
tables (10–54×) could not give: a customer reproduces *these* on their own
NIC. The effective reload rate (~1.9 GB/s against the banked 2.40 GB/s
ceiling at 32k) localizes the bottleneck to the connector, not the store:
both the reload timing and the iperf3 ceiling it is measured against are
committed artifacts, so the ratio is checkable, not asserted.

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
