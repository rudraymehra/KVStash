# Rig E — GPU TTFT rig (g5.xlarge, A10G)

**Status: NOT YET RUN — requires GPU access.** Nothing in this directory has
produced numbers; `bench/results/rig-e/` does not exist yet. Everything below
is the written plan for the first session.

## What it measures

Chart #2: **TTFT vs KV-cache hit rate** through the real stack — vLLM +
LMCache + `lmcache_kvblockd` + a kvblockd daemon — answering the only
question that matters for a remote KV cache: *at what hit rate does fetching
beat recomputing, and by how much?* Series: recompute baseline,
LMCache + Redis 7, LMCache + kvblockd at 25 Gbps and 50 Gbps emulated links.

## Hardware

- 1x `g5.xlarge` (NVIDIA A10G 24 GB, 4 vCPU). Llama-3.1-8B-Instruct fits in
  bf16 with a capped `--max-model-len`. The GPU class is disclosed on-chart;
  an A100/H100-class re-run would be a separate, labeled session.
- kvblockd runs on the same host; the network is emulated with `tc` (tbf)
  at 25 and 50 Gbps classes, and the class is written into every JSONL line
  (emulated links are always disclosed — methodology rule 11).
- G-instance vCPU quota is often 0 on a fresh AWS account; request the
  increase first (`Service Quotas → EC2 → Running On-Demand G and VT
  instances`, ≥4 vCPU — usually auto-approved in minutes).

## Discipline before any long metered run

No long sweep until the pipeline yields **3 stable points on the same box**.
Cuts pre-applied to keep the session small: 4 hit-rate points
{0, 25, 50, 90}%, 2 runs + a spot-check third, and the LMCache-local-CPU
series dropped from the headline chart.

## Runbook

The provision/setup/sweep/teardown scripts are authored at session start on
the box (they depend on the AMI's exact CUDA/driver versions) and committed
alongside the results.

1. `provision.sh` — g5.xlarge spot with on-demand fallback, Deep Learning
   AMI (CUDA 12.4), tagged `kvbench=gpu`, `trap`-guarded teardown.
2. `setup.sh` — pins from `bench/VERSIONS.lock`: vLLM 0.25.1, LMCache 0.5.1,
   `lmcache_kvblockd` + a kvblockd static binary, CUDA torch (reuses the
   `bench/e2e/cpu/install.sh` shape with the GPU index). Apply the `tc`
   link classes.
3. `ttft_sweep.sh` — for each hit-rate point: pre-seed kvblockd with that %
   of each prompt's prefix blocks (via the connector's key-parity module),
   flush vLLM's local prefix cache, drive `inference-perf` shared-prefix,
   record p50/p99 TTFT per series.
4. **Bailian datapoint** — replay `qwen_traceA` at realistic arrival rates;
   record the hit rate the store actually achieves (expect 54–62%) and the
   TTFT there. That is the quotable real-trace number.
5. nsys capture while the GPU is up — verify fetch/compute overlap
   (see `docs/notes/` for the overlap hooks).
6. `run.sh` emits a PASS/FAIL line — remote TCP fetch beats recompute at the
   measured Bailian hit rate → PASS; **publish the curve either way**, with
   the shaded "recompute wins here" region.
7. `economics.py` update — measured $/GB moved vs $/GPU-sec saved at the
   measured hit rates (inputs to `bench/e2e/economics.py`).
8. `teardown.sh` — terminate everything tagged `kvbench=gpu`; verify zero
   tagged resources remain.

`run.sh` stamps `gpu`/`model`/`vllm`/`lmcache`/`tc_link` into every JSONL
line so `plot.py` reads the conditions box from data, never hardcoded.

All JSONL → `bench/results/rig-e/`; render with
`python bench/report/plot.py chart2 --in bench/results/rig-e/*.jsonl --out chart2.png`.
