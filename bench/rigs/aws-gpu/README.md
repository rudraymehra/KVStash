# Rig E — the A4 GPU session (AWS g5.xlarge, A10G)

Chart #2: TTFT vs hit rate through the real vLLM + LMCache + kvblockd stack.
**Budget-driven substitution (Rudray, 2026-07-18):** RunPod is out (no
budget); the GPU rig runs on the existing AWS account, `g5.xlarge` (A10G
24 GB, ~$0.35–0.50/hr spot, ~$1.006 OD). Llama-3.1-8B-Instruct fits in bf16
with a capped `--max-model-len`. The GPU class is disclosed on-chart; an
A100-class headline re-run is deferred pending budget (an AWS Activate grant
would fund it).

## C-21 gate (honored)

- **No long metered run until the pipeline yields 3 stable points** on the
  same box; a second cheap g5 day is the pre-authorized buffer.
- Ranked cuts pre-applied: **4 hit-rate points {0, 25, 50, 90}%**, 2 runs +
  spot-check third, **LMCache-local-CPU series dropped** from the headline
  chart.

## Quota check (do this first)

G-instance vCPU quota is often 0 on a fresh account. Request the increase
FIRST (`Service Quotas → EC2 → Running On-Demand G and VT instances`, ask for
≥4 vCPU) — usually auto-approved in minutes. The offered second AWS account
is the fallback.

## Status

This is a **runbook**, not committed scripts. The
provision/setup/ttft_sweep/run/teardown scripts below are authored at
session start on the g5 box (they depend on the AMI's exact CUDA/driver
versions), then committed alongside the results. `run.sh` will stamp
`gpu`/`model`/`vllm`/`lmcache`/`tc_link` into every JSONL line so
`plot.py` reads the conditions box from data (not hardcoded).

## Runbook

1. `provision.sh` — g5.xlarge spot w/ OD fallback, Deep Learning AMI
   (CUDA 12.4), tagged `kvbench=gpu`, `trap`-guarded teardown.
2. `setup.sh` — pins from `bench/VERSIONS.lock`: vLLM 0.25.1, LMCache 0.5.1,
   `lmcache_kvblockd` + `kvblockd` static binary, CUDA torch (reuses the
   `bench/e2e/cpu/install.sh` shape with the GPU index). kvblockd on the same
   host; **loopback tc-throttled to 25 and 50 Gbps** classes (`tc qdisc … tbf`)
   — the script is committed and the class is written into each JSONL line.
3. `ttft_sweep.sh` — for each hit-rate point, pre-seed kvblockd with that %
   of each prompt's prefix blocks (via the connector's key-parity module), flush
   vLLM's local prefix cache, drive `inference-perf` shared-prefix, record
   p50/p99 TTFT per series {recompute, LMCache+Redis 7, LMCache+kvblockd@25G,
   @50G}.
4. **Bailian datapoint** — replay `qwen_traceA` at realistic arrival rates;
   record the hit rate the store actually achieves (expect 54–62%) and the
   TTFT there. **That is the A4 number.**
5. A5 nsys overlap check while the GPU is metered (C-17).
6. `run.sh` emits the **A4 PASS/FAIL line** — remote TCP fetch beats
   recompute at the measured Bailian hit rate → PASS; **publish the curve
   either way** with the shaded "recompute wins here" region.
7. `economics.py` update — measured $/GB moved vs $/GPU-sec saved per hit at
   the measured hit rates (A7 inputs).
8. `teardown.sh` — zero tagged resources.

All JSONL → `bench/results/rig-e/`; render with
`python bench/report/plot.py chart2 --in bench/results/rig-e/*.jsonl --out chart2.png`.
