# kvblockd benchmarks

> Prior raw transport/DRAM/NVMe gate numbers live in `bench/BENCHMARKS.md`
> (the running scoreboard); this file is the methodology-first launch view.

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

## Chart 1 — throughput vs the field (TO FILL from Rig T)

GET-only, batch 32, best streams per store, warmed, median of 3, on 2×
c6in.8xlarge in a cluster placement group, with the measured iperf3 ceiling
drawn in. Bars: kvblockd (DRAM) · kvblockd (NVMe-resident) · Mooncake TCP ·
Valkey 8 (go-redis) · Redis 7 (go-redis) · Redis 7 (redis-py) · local NVMe-fs
floor.

| Store | 0.44 MiB GB/s | %-ceiling | cores/GB·s⁻¹ | 2.5 MiB GB/s |
|---|---|---|---|---|
| kvblockd (DRAM) | _T_ | _T_ | _T_ | _T_ |
| … | | | | |

**≥10× vs redis-py gate:** _verdict from run-chart1.sh_ (median of ≥3, measured, not massaged).

Prior measured transport ceilings (Weeks 1–6, this rig family):

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

## Chart 2 — TTFT vs hit rate (TO FILL from Rig E)

A10G/g5.xlarge, Llama-3.1-8B-Instruct, vLLM 0.25.1 + LMCache 0.5.1 +
lmcache_kvblockd, tc-emulated 25/50 Gbps link (disclosed on-chart). Series:
vLLM recompute · LMCache+Redis 7 remote · LMCache+kvblockd@25G · @50G. Hit-
rate points {0, 25, 50, 90}% by prefix-block seeding, plus the **Bailian
production-trace datapoint at the measured hit rate (54–62% band) — that is
the A4 number**, not a 90% fantasy point.

**A4 verdict:** _remote TCP fetch beats recompute at the measured Bailian hit
rate → PASS / FAIL, curve published either way with the shaded "recompute
wins here" region._

---

## When NOT to use kvblockd

Below the crossover hit rate (drawn on Chart 2), recompute is cheaper than a
remote fetch — don't deploy a remote KV cache for that workload. The
`bench/e2e/economics.py` model dollarizes the crossover ($/GB moved vs
$/GPU-sec saved per hit, same-AZ vs cross-AZ) at the measured hit rates.

---

## Reproduce

```bash
# Local acceptance (no cloud):
bench/kvbench/loopback.sh

# Chart 1 (~$12 spot):
bench/rigs/aws-transport/provision.sh && bench/rigs/aws-transport/run-chart1.sh && \
  bench/rigs/aws-transport/teardown.sh
python bench/report/plot.py chart1 --in bench/results/rig-t/*.jsonl --out chart1.png

# Chart 2 (AWS g5.xlarge):
# Rig E is a RUNBOOK (bench/rigs/aws-gpu/README.md); its provision/seed/sweep/teardown
# scripts are finalized at session start on the g5 box, then committed with the results.
python bench/report/plot.py chart2 --in bench/results/rig-e/*.jsonl --out chart2.png
```
