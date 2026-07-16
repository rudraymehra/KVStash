# kvblockd — Design & kill-gate results

Kill-gate verdicts and load-bearing design decisions. Kill-gates are pre-registered
(A1–A8); a FAIL executes its written consequence.

## A1 — transport ceiling: can Go on plain TCP saturate the NIC?

**Gate:** ≥12 GB/s loopback AND ≥85% of the measured iperf3 ceiling on a cloud pair.
**Result: PASS on both halves.**

**Loopback (recorded first):** 14.1 GB/s peak on the dev machine (`docs/notes/a1-log.md`).

**Cloud rig (2026-07-16):** 2× c6in.8xlarge (50 Gbps ENA), us-east-1, cluster
placement group, private-IP traffic. Tuning applied before any measurement
(`bench/rigs/sysctl-esnet.conf` + `tune.sh`): BBR + fq, 256 MiB max socket
buffers, MTU 9001. Spot capacity was unavailable; both nodes ran on-demand.

Ceiling first, same discipline every rig follows — the denominator is measured,
never assumed:

| iperf3 `-P` | 8 | 16 | 32 | 64 |
|---|---|---|---|---|
| Gbit/s | 49.5 | 49.8 | 49.8 | 49.8 |

**Ceiling: 49.8 Gbit/s** — the NIC's full rated 50 Gbps. Raw:
`bench/rigs/aws-transport/iperf-ceiling.txt`.

xferspike sweep (streams {8,16,32,64} × frame {1,4,16} MiB, 30 s cells, 16 MiB
SO_SNDBUF; raw JSONL: `bench/rigs/aws-transport/xferspike-results.jsonl`):

| streams \ frame | 1 MiB | 4 MiB | 16 MiB |
|---|---|---|---|
| 8 | 49.74 | 49.58 | 49.58 |
| 16 | 49.87 | 49.90 | 49.86 |
| 32 | 50.02 | 50.01 | 49.99 |
| 64 | 50.14 | 50.16 | **50.17** |

(Gbit/s, decimal.) Best cell = **50.17 Gbit/s = 6.27 GB/s = ~101% of the iperf3
ceiling** at **0.77–0.91 sender CPU cores**. Every cell in the matrix clears the
85% gate; the curve is flat because the NIC, not the software, is the limit.

**Honest caveats:**
1. This box is 50 GbE, so the *absolute* number is NIC-capped at 6.25 GB/s
   decimal; the 10+ GB/s headline requires 100 GbE hardware. What A1 proves is
   that the Go/TCP data path is not the bottleneck at line rate with <1 core to
   spare — the software has headroom, the NIC ran out.
2. iperf3 and xferspike agree within ~0.7% (xferspike slightly above — run-to-run
   variance, both saturated); we quote %-of-ceiling, not GB/s alone.

> **Verdict (Rudray, hand-written):** _[pending — explain what the flat matrix +
> sub-core CPU means for the wedge, in your own words]_

## A2 — GC pause under off-heap arenas

**Gate:** GC pause p99 < 5 ms while serving from a large off-heap arena.
**Result: PASS.**

**Mechanism proof (Mac, 512 MiB arena):** p99 0.92 ms (`docs/notes/a2-log.md`).

**Cloud run (2026-07-15):** c7g.xlarge (4 vCPU Graviton, 7.6 GiB RAM), Amazon
Linux 2023 arm64, `xferspike --mode=soak --arena-bytes=$((5<<30))
--soak-streams=8`. Arena sized 5 GiB (not the planned 8 GiB: the box has
7.6 GiB total; an 8 GiB arena cannot fit beside the OS + heap).

| window | GC pauses observed | p50 | p99 | p999 | max | heap_alloc | bytes served |
|---|---|---|---|---|---|---|---|
| 30 s | 3,304 | 0.057 ms | 5.24 ms | 12.6 ms | 12.6 ms | 59 MB | 334 GB |
| 60 s | 6,473 | 0.057 ms | 4.19 ms | 10.5 ms | 12.6 ms | 45 MB | 658 GB |
| **300 s** | **32,584** | **0.057 ms** | **4.19 ms** | 10.5 ms | 21.0 ms | **57 MB** | **3.3 TB** |

**p99 = 4.19 ms < 5 ms on the 32,584-pause sample → PASS.** The 30 s window's
5.24 ms is small-sample tail noise (it settles with 10× the pauses — percentile
tails from short runs are not evidence). The off-heap thesis held exactly:
`heap_alloc` stayed at ~57 MB while RSS tracked the 5 GiB arena and 3.3 TB
crossed the wire.

**Tail attribution (measured, not guessed):** `GODEBUG=gctrace=1` shows true
stop-the-world segments (sweep-term / mark-term) almost all <0.5 ms; the 10–21 ms
p999/max tail is scheduler rendezvous on 4 oversubscribed shared-tenant cores
(17 goroutines, CPU steal), not GC work. Percentiles are reported as histogram
bucket *ceilings* (conservative).

**Honest caveats:**
1. 5-minute window, not the planned 24 h — pause behavior is proven; multi-day
   *stability* (leak/drift) is not. A 24 h soak remains optional follow-up.
2. Shared-tenant 4-vCPU box inflates the measured tail; production hardware
   with dedicated cores will read lower, not higher.

> **Verdict (Rudray, hand-written):** _[pending — why doesn't a 5 GiB cache
> stress Go's GC? Where do the blob bytes live, and what does the GC actually
> scan?]_

## A3 — NVMe from Go: is the Go I/O path the bottleneck?

**Gate (as pre-registered):** ≥6 GB/s per NVMe device from Go (threadpool
fallback accepted per the recorded giouring/Go-1.26 decision).
**Result against the gate as literally written: NOT MET on this hardware —
2.94 GB/s measured vs the 6.0 figure.** Kill-gates are pre-registered and are
not renegotiated after seeing data, so no "pass" is recorded here. What the
evidence *does* establish, fully disclosed below: the device's own hardware
ceiling is 2.99 GB/s (no software can print 6 on an i4i Nitro SSD), and Go
reaches 98.3% of that ceiling — the software is provably not the bottleneck.
**Disposition (gate owner's call, pending):** either (a) record a dated gate
amendment — the 6 GB/s criterion was calibrated to PCIe4-class local drives,
amend to "≥95% of the measured fio device ceiling" — or (b) re-run on hardware
whose device ceiling is ≥6 GB/s. Until one is recorded, A3 stays open.

**Rig (2026-07-16):** 1× i4i.8xlarge, us-east-1, 2× 3,750 GB AWS Nitro SSD
(instance store), XFS, O_DIRECT, 32 GiB fully-written test file per device.

**fio ceiling first (the denominator):**

| config | result |
|---|---|
| read 128k qd32 ×1 job, raw device | 2.97 GB/s |
| read 1m qd64 ×4 jobs, raw device | 2.99 GB/s |
| both devices in parallel (1m qd32 ×2 jobs each) | 2.98 + 2.98 = **5.95 GB/s** |

**Per-device hardware ceiling: ~2.99 GB/s** — an AWS Nitro SSD limit (i4i's
devices are not PegaFlow-class PCIe4 drives; the gate's "6 GB/s/device" figure
was calibrated to the latter). Deeper queues and more jobs move nothing: the
device is util-saturated at 99.7%.

**nvmeprobe (Go, threadpool engine, O_DIRECT), matrix + aggregate** (raw JSONL:
`bench/rigs/aws-nvme/nvmeprobe-results.jsonl`):

| op | bs | qd | GB/s | % of fio ceiling | CPU cores |
|---|---|---|---|---|---|
| read | 128k | 8 | 2.93 | 98.0% | 0.23 |
| read | 128k | 32 | 2.94 | 98.3% | 0.24 |
| read | 1m | 8 | 2.94 | 98.3% | 0.12 |
| read | 1m | 32 | 2.94 | 98.3% | 0.13 |
| write | 128k/1m | 8/32 | 2.33 | — | 0.11–0.27 |
| **read, both devices parallel** | 1m | 32 | **5.86 aggregate** | **98.5%** | 0.25 total |

Zero `io_errors` in every cell.

**Reading:** Go's pinned-thread pread pool reaches **98.3% of the fio ceiling
per device and 98.5% aggregate**, at 0.1–0.25 cores. The software is not the
bottleneck on any config; "≥6 GB/s per device" is a hardware-shopping decision
(a single modern PCIe4/5 drive), not a software risk. The NVMe tier's IOBackend
seam keeps an io_uring engine pluggable if faster devices ever expose a syscall
ceiling (v1.1 spike, per the recorded decision in
`bench/microbench/nvmeprobe/io_linux.go`).

**Honest caveats:**
1. No config on THIS hardware can print 6 GB/s from one device — we report
   %-of-ceiling as the verdict basis, same discipline as A1, and say so plainly.
2. io_uring remains unmeasured (giouring is dead on Go 1.26); the threadpool
   number is the shipping baseline, not the theoretical best.

> **Verdict (Rudray, hand-written):** _[pending — why is %-of-fio-ceiling the
> honest gate here, and what would you buy to serve 6 GB/s per device?]_


## A7 — same-AZ economics: does loading beat recompute, and does it pencil out?

**Gate:** same-AZ KV-cache fetch cost < recompute cost at ≥40% hit rate.
**Verdict: PASS** (same-AZ, GQA/MLA models). Model: `bench/e2e/economics.py` (every formula printed).

**The break-even identity:** loading beats recompute above `B* = bytes_per_token × prefill_rate` — literally the rate the GPU *produces* KV during prefill. It's context-length-independent for O(n) prefill (the token count cancels).

*(Break-even = decimal GB/s, matching iperf3 / the transport rig. Regenerated from `bench/e2e/economics.py` — do not hand-edit.)*

| Model | KV bytes/token | Break-even B* |
|---|---|---|
| Llama-3.1-8B (GQA) | 131,072 B (0.125 MiB) | ~1.97 GB/s |
| Llama-3-70B (GQA) | 327,680 B (0.312 MiB) | ~0.66 GB/s |
| DeepSeek-V3 (MLA) | 70,272 B (0.067 MiB) | ~0.56 GB/s |
| *Llama-70B [MHA-miscount]* | *2.5 MiB* | *~2.62 GB/s* |

GQA/MLA models break even at ~0.5–2.0 GB/s; kvblockd's 10+ GB/s target clears it with large margin. The commonly-cited "2.5 MB/token / ~2 GB/s SLO" (Cake, Tensormesh) is calibrated to the **MHA miscount** — real GQA Llama-3-70B is 8× smaller. We use GQA/MLA counts and never the 2.5 MB figure.

**Cost:** same-AZ private-IP transfer is **$0/GB**, so a hit's cost is ~zero while recompute burns GPU-seconds ($0.002–0.015/hit reusing 80K tokens on an A100). PASS at every hit rate ≥40%.

**Honest caveats (recorded, not hidden):**
1. **Same-AZ private IP only.** Cross-AZ transfer is $0.01/GB each way and *exceeds* recompute for cheap-to-prefill small models ($0.098 vs $0.002 for 8B) — the deployment guide must mandate same-AZ; a public/Elastic IP silently triggers $0.01/GB even within one AZ.
2. Use GQA/MLA byte counts, never the MHA 2.5 MB figure (8× overstatement).
3. Below ~40% hit rate, amortized infra cost may not clear recompute.

Sources: LMCache (arXiv 2510.09665), Cake (arXiv 2410.03065), Tensormesh Redis blog, DeepSeek-V3 (arXiv 2505.09343), AWS EC2 data-transfer pricing, RunPod pricing. The transport rig has since run (§A1: 6.27 GB/s NIC-limited on 50 GbE) — every GQA/MLA break-even in the table above clears with margin at the measured cloud figure.
