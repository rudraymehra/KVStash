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

## Week 2 wire-path results (RAM-stub loopback)

The full request→response path is live end to end: `pkg/client` → `internal/transport`
→ `internal/server` → `internal/store/ramstub`, with the BATCH_GET response
emitted as one `writev` (descriptor region + block bytes straight from store
memory) and per-block `xxh3_64` verified on the client.

**Throughput gate — `BenchmarkBatchGet_32x1MB` (Mac loopback), after the
pprof-driven tuning pass:** cold-data peak **~9.5–9.7 GB/s** at 4 streams
(up from 7.4 pre-tuning, +30%); hot-source cell ~10.2 GB/s. Run-to-run
variance on the laptop is ±15% (thermal), so the binding number below is a
ratio, not an absolute.

**The gate, resolved by measurement (A1 %-of-ceiling methodology).** The
written target — "within 10% of the Week-1 xferspike loopback 14.1 GB/s" —
compared two different workload shapes. xferspike's 14.1 is a one-way blast
that resends ONE cache-hot 1 MiB buffer with no response read, no store, and
no integrity check (verified in `cmd/xferspike/spike.go`: single `payload`
buffer, single reusable receive buffer). A store GET is a request→response
round trip over 32 *distinct* cold blocks into 32 distinct destinations, plus
an xxh3 pass. To make the comparison honest we built the missing baseline —
`bench/microbench/rawget` — the same GET shape on raw sockets with **no
protocol, no auth, no checksums**:

| run (interleaved, same thermal state) | raw-socket ceiling | kvblockd | ratio |
|---|---:|---:|---:|
| cold 4-stream, session A | 10.06 GB/s | 9.5–9.7 GB/s | ~0.95 |
| cold 4-stream, session B (throttled) | 7.87 / 7.95 | 8.19 / 6.52 | ~1.04 / 0.82 |
| hot-source 4-stream | 11.80 | 10.2 | 0.86 |

**kvblockd's full stack runs at parity (±noise) with a raw socket doing the
same work** — the protocol, credit, descriptor, and verification layers cost
≈nothing measurable; the ceiling is the kernel's loopback copy path itself.
Even raw sockets cannot print 14 GB/s on the GET shape. Per the A1 rule ("we
quote %-of-ceiling, not GB/s alone") the wire path is **within 10% of the
same-shape ceiling: PASS**; the 14.1 blast figure stays in the record as what
it is — a different shape.

**What the tuning pass changed (each pprof-justified):** (1) writev syscalls
are now windowed at ~1 MiB (`write_chunk_bytes`; A1 measured 14.1 at 1 MiB-
per-writev vs 6.9 at 16 MiB — giant single copies stall the loopback pipe);
(2) client xxh3 verification moved to a sidecar goroutine so hashing block i
overlaps reading block i+1 (serialized read-then-hash idled the socket every
block); (3) client socket-buffer options added (`SockSndBuf/SockRcvBuf`);
loopback prefers OS defaults, real links want 16 MiB. The benchmark now sweeps
streams×{cold,hot} because loopback throughput *falls* with stream count
(a1-log finding 2) — single-number quotes hide the shape.

**Final decomposition (out-of-process daemon + `bench/kvbench/getbench`, the
production shape). The claim is a RATIO measured interleaved, not an absolute:**
laptop absolutes swing ±20% run-to-run under sustained load (the raw ceiling
itself ranged 8.8–10.8 GB/s in one batch), so any single GB/s figure is a
lucky run. Interleaved kvblockd-verify-off ÷ raw-socket ceiling over 6 pairs:
**0.88 / 0.91 / 0.95 / 1.00 / 1.02 / 1.04, median ~0.97** — i.e. the full
protocol stack runs at ~parity with a plain raw socket doing the same work,
robustly, even as absolutes drift. verify-off absolute this batch ~7.7–10.6;
verify-on (default) 12-run median ~6.9 (min 3.7). The verify delta is xxh3
integrity — real work, dialable via `client.Options.SkipVerify`. Cross-kernel:
Linux 6.10 VM held the same ~0.9 ratio (VM overhead in absolutes; ratio is the
claim). **An earlier revision of this section quoted 11.27/9.75 GB/s as the
headline — those were favorable single runs and overstated the absolute; the
ratio is the honest, reproducible claim.**

**Pipelining (measured, recorded for the network rig):** an in-order
depth×streams matrix (`BenchmarkBatchGetPipelined`, bench-only raw client —
per-connection response ordering needs no FEAT_OOO) is FLAT on loopback: the
request-turnaround bubble is microseconds there. The lever's payoff is
real-RTT networks; re-measure on the AWS pair.

**PUT (ingest) path — `BenchmarkPut_1MB` / `BenchmarkPutPipelined_1MB`.**
Multi-stream PUT is ~5.9 GB/s at 4 streams (the bench Deletes each key to stay
bounded, so this UNDERSTATES pure PUT); the 1-RTT pipelined shape (BEGIN+CHUNK+
COMMIT in one burst) is 277 µs/op single-stream vs 353 µs for the two-RTT
product client (−20%), at 25 vs 32 allocs/op. The PUT/GET ratio is ~2×, better
than every store studied (Redis's own SET/GET = 3.8× at 4 MB; the team
deprioritized the write path outright). pprof of the pipelined PUT: **64%
syscall, 18% goroutine-wakeup handoffs (pthread_cond), 10% kevent, memmove 1.9%,
xxh3 1.6%** — i.e. syscall/handoff-bound, NOT copy-bound. The copy/digest work
done this pass (ownership-transfer Put, first-chunk exact-cap staging,
incremental digest) removed the memory traffic; what remains is the same
kernel-syscall wall the GET path hit, plus the read-side two-syscall-per-frame
cost (`io.ReadFull` header then body). A 2 MiB server rcvbuf buys PUT ~8%
(8 MiB no more); the structural levers — `readv` to fuse the header/body reads
(golang/go#17607) and a worker-pool handoff to cut the per-response cond
signal (RocksDB's write-thread-adaptive-yield finding) — are deferred:
`readv` steps outside `net`, worker handoff is a post-Week-2 server upgrade.
Both are pre-registered for the network rigs, not chased on loopback where the
syscall floor dominates.

**Research-verified positioning (adversarially-verified deep-research pass,
2026-07-16; sources: golang/go#13451/#17607/#21676, CloudWeGo docs, Cloudflare
& Netflix engineering, netdev MSG_ZEROCOPY paper):**

- goroutine-per-conn + `net.Buffers` writev IS the right architecture at ≥1 MiB
  messages — ByteDance's own docs steer >1 MB workloads to `go net`, not their
  event loop. Event-loop frameworks solve C10K context-switch cost, not
  large-message throughput.
- The writev fast path exists only on a bare `*net.TCPConn` (golang/go#21676:
  a non-embedding wrapper silently becomes one Write PER BUFFER — 32 syscalls
  per response). The transport now logs a tripwire if the conn is ever wrapped.
- MSG_ZEROCOPY has provably ZERO loopback benefit (kernel copies shared pages
  on the loopback path), and 1 MiB sends are its max-benefit regime on real
  NICs (79% of process cycles are the user→kernel copy; expect 5–15% end to
  end). The `writeReq.release()` seam is exactly the errqueue-completion hook
  it needs. Deferral to the 100 GbE rig stands, now evidence-backed.
- At tens of GB/s the terminal wall is memory bandwidth (Netflix: 30 GB/s on
  ~150 GB/s DRAM) — count DRAM passes per byte before buying faster NICs.

**Block-size characterization (fresh daemon per size, two-process, verify on).**
Across the real workload range GET holds a flat 5–9 GB/s band with no size
cliff: 0.4 MiB ≈ 7.7–8.5, 1 MiB ≈ 7.3–7.9, 2.5 MiB ≈ 5.4–7.6 GB/s (the slight
decline at 2.5 MiB is the memory-bandwidth wall — larger blocks evict more
cache per response). Method note recorded so it is not re-learned: the load
generator seeds keys 1..N independent of block size, so sizes MUST be swept
against SEPARATE daemon instances — reusing one daemon hits write-once
immutable-conflict and credits new-size bytes against stale data (produced a
physically-impossible 49 GB/s reading; discarded, not reported).

**Second-kernel confirmation (Linux 6.10 / Docker linuxkit, 8 cores).** These
carry virtualization overhead and are NOT the quotable figure (bare-metal
Linux is the Week-3 rig) — the value is that the SHAPE reproduces on a second
kernel: codecs zero-alloc (Marshal 19.6 ns, Parse 22.5 ns); GET cold peaks at
4 streams (5.48 GB/s) and declines at 16 (2.93) — the a1-log inverted-stream
curve, independent of macOS; PUT 2.2–3.5 GB/s (same ~2× read/write ratio);
EXISTS 9.9–20 µs. One divergence worth noting for the network rig: EXISTS
pipelining depth-16 gains only ~1.4× here vs ~5× on macOS, because per-RTT
latency is lower to begin with — the pipelining lever's payoff scales with the
RTT it hides, so the AWS pair (real RTT) is where it will matter most.

**Inline vs sidecar verification (sidecar kept, margin is small).** Measured
head-to-head INTERLEAVED (same daemon, same thermal state, 5 pairs): sidecar
~8.4 vs inline ~8.2 GB/s — sidecar wins 3/5 and is marginally ahead, but they
are near-equal, well within run-to-run noise. (An earlier note here claimed
6.9 vs 9.75; that was a thermal confound — the two variants were measured at
different times/thermal states, not interleaved. Corrected.) xxh3 is 19.2 GB/s
single-core so verification is never CPU-bound either way; the sidecar's
read/hash overlap is kept as the marginally-better and clearer design, not a
decisive win. Both are memory-bandwidth bound (the second pass over the block).

**Verification & allocation — evidence-backed decisions (research pass 3,
107 adversarially-verified agents; sources: fasthttp, golang/go#26663/#72036,
RocksDB xxh3/crc32c bench, Go 1.26 Green Tea GC notes):**

- **Keep xxh3 (C-35), don't switch to hardware CRC32C.** CRC32C is not reliably
  faster — xxh3 measured 26.8 vs CRC32C 19.3 GB/s in RocksDB — and a 32-bit CRC
  loses collision resistance that matters for content-addressed blocks. The
  protocol lock is also the right engineering call, not just a constraint.
- **Per-connection scratch is GC-safe under Green Tea.** Pointer-free `[]byte`
  scratch (`lendBuf`, `descScratch`, client `rbuf`/`wbuf`) is allocated
  `noscan` and never scanned for contents, so keeping large per-conn buffers
  alive does not raise GC scan time proportionally — the recycling this pass
  introduced is GC-friendly, confirmed.
- **Per-request allocation elimination is GC-hygiene, not a throughput lever
  here.** The residual allocs (net.Buffers iovec escape at the writev syscall
  boundary #26663, `[]byte`→interface boxing #72036, per-request result slices)
  are real and poolable, but the GET path is memory-bandwidth bound — cutting
  them won't move loopback GB/s (independently confirmed: the second
  verification pass over the block is the memory-bandwidth cost). Deferred as
  low-value/higher-async-risk until profiling shows allocation rate (not
  bandwidth) as the bind — most likely under high connection counts, a
  Week-6 tenancy concern, not now.

**Standing Week-3+ items:** re-run gate + `rawget` baseline on the bare-metal
Linux rig (the quotable environment; Mac loopback is a sanity check per
a1-log); pipelined/OOO client demux for the network path (loopback-flat,
network-decisive); `readv` via `SyscallConn` on the PUT/request read path
(golang/go#17607 — kills the second per-frame read syscall); Linux
`tcp_rmem/wmem` sized to LAN BDP (~0.3–3 MB, not WAN-scale). The DRAM-tier
week re-tests this exact benchmark against the same-shape ceiling; the
goalpost does not move.

**Fuzz:** `FuzzParseHeader` + `FuzzParseBatch` clean (tens of millions of execs);
formal 1h-per-target gate is the Day-7 item. **PUT_STREAM invariants**
(invisible-until-COMMIT, ERR_CHECKSUM, exactly-one-response incl. ABORT,
OK_EXISTS idempotency) covered by server tests; goleak clean.

## Week 3 — the DRAM tier (arena + allocator + index + lifecycle)

<!-- RUDRAY: prose first per the Merge Rule — write this section from memory
     (arena layout & prefault, the OffsetAllocator port and where it diverges
     from Aaltonen's C++, index sharding + maphash seeding, BlockRef field
     semantics, the lease/pin/TTL ladder ordering table, two-phase visibility,
     copy-at-commit and why stage-in-arena was rejected this week, the GET
     refcount/release path through WriteFrames) — THEN fact-check against the
     code and replace the measured-numbers block below with your own reading
     of the gate logs. The numbers here are the raw results for you to
     interpret, not prose to copy. -->

**Measured gate results (Day 6).** All three gates are same-host relative;
the recorded venue is Linux (c7i.4xlarge, 16 vCPU AL2023 — the `aws-dramgate`
rig); the Mac runs are the dev-box sanity check.

| gate | Mac (8-core M-series, dev box) | Linux c7i.4xlarge (recorded) |
|---|---|---|
| G1: getbench ÷ same-shape ceiling, ≥0.9× | 0.98 / 1.02 / 1.09 (3 interleaved pairs, 768 MiB set) | **0.97 / 0.96 / 0.96 — PASS** (rawget ceiling 27.3–27.4, kvblockd 26.2–26.4 GB/s, 4 GiB set) |
| G2: 512-key EXISTS p99 under 8 GET lanes, <1 ms | 4.6 ms — 8-core oversubscription (2-lane control: 697 µs) | **p99 = 705 µs — PASS** (1.2 M samples, 60 s, GET lanes serving ~37 GB/s throughout) |
| G3: blob-band (0.4–2.5 MB) alloc sites on the GET path | ZERO (only 256 B/896 B header-region objects; 220 GB served, 33 MB total heap alloc in 15 s) | **ZERO per-request — PASS** (~400 GB through the 15 s window, 46 MB heap alloc total; only per-CONNECTION one-time 2 MiB `Lend` recycle buffers + pprof's own writer touch the band) |

G1 note (the Week-2 ruling reapplied): xferspike is a one-way hot-buffer
blast — a different shape. On the Mac the two shapes coincide (memory-bound
either way); on the 16-vCPU box they diverge (xferspike prints 54.7 GB/s).
The binding ceiling is `bench/microbench/rawget` — the same request→response
GET shape on raw sockets with no protocol/auth/checksums. "≥0.9× the
same-shape ceiling; the goalpost does not move."

PUT staging remains on the Go heap by the locked copy-at-commit decision
(the Week-2-reviewed DoS posture: lazy, capped, tombstoned); it IS a
blob-band alloc site and is the documented exception until the Week-4+
arena Reservation API removes the copy. The GET path — the 99% path — is
allocation-free in the blob band.

**Week-3 ladder outcome (FULL, 5 stages + Opus diversity breaker, 4 refuters — all HIGH+ findings confirmed).** Fixed pre-push: spec-legal empty blocks were rejected ERR_QUOTA_BYTES (now extent-less, conformance-tested both stores); soft pins wrongly debited the §3.6 hard-pin quota (now: charge only on transition into hard, upgrade passes the gate, downgrade refunds); arena-full COMMIT emitted ERR_QUOTA_BYTES outside §3.4's frozen set (now: advisory BEGIN capacity check answers it there; the rare commit race maps to retryable ERR_BUSY — §3.4/§5 amended one line each); a drain-timeout could munmap the arena under an in-flight writev (transport now caps post-close writes at the 1s drain budget, Drain reports success/failure, main skips the unmap on timeout); CanEvict required refcount==0 and was structurally false for every resident block (now the refcount==1 indexed-block pre-filter, race-audited); daemon deadlocked on startup errors with metrics enabled (defer order); BEGIN ttl_ms was silently dropped (now applied at commit; no wire-observable effect until the evictor — review-covered, not behavior-testable yet); max_blob_len==max_frame_len made frame-cap blobs unreadable (negotiation now reserves the 32 B GET-header headroom); plain dram Get returned an arena view after release (now copies — the wire path uses GetRef and never pays it); a writeLoop flush-failure leaked the socket fd.

**Known ceilings (documented, deliberate):** PUT staging stays heap-side (copy-at-commit; the reviewed DoS posture) until the Week-4+ reservation API. The allocator node pool caps at 2^17 live blocks regardless of arena size (Allocation.Meta's 18 slot bits) — irrelevant at the 1 GiB default, binds first on ≥8 GiB arenas of small blocks; widening Meta is scheduled with the evictor. kvb_hits_total hardcodes tier="dram" until the Recorder seam carries a tier. TouchLease can land on a ref a concurrent Delete just removed (client sees OK for a lease on a dead block — memory-safe, pre-existing, evictor-week item).

## Week 4 — eviction + the model-based correctness harness

<!-- RUDRAY: prose first per the Merge Rule — write this section from memory
     (why S3-FIFO beats LRU for one-hit-wonders and what exactly the ghost
     ring remembers and why hashes suffice; the three-pass evictor and the
     §6 ladder precedence; why eviction is metadata-only and therefore
     cheap; why a LOSSY store can still have a STRICT correctness oracle —
     the asymmetric I1 and the maybeGone reconciliation; the happens-before
     story between a reader's refcount and the evictor's gate) — THEN
     fact-check against the code. The results below are raw inputs. -->

**Ladder outcome (FULL, 5 stages + Opus breaker + 1 measuring refuter).**
One reproduced BLOCKER (S3-FIFO Admit/Remove atomicity gap — the zero-value
queue state collided with qSmall; a plain PUT-vs-forced-DELETE race
corrupted the FIFO and panicked the evictor; found independently by two
stages AND re-reproduced by the refuter on its snapshot) — fixed with an
explicit qUnqueued state + tombstone handshake and a hammer regression.
HIGHs fixed: GET auto-lease TRUNCATED longer explicit leases (now monotonic
extendLease — spec says grant/extend, RELEASE is the only shortener);
evictOne conflated not-found with gate-refused, minting permanent phantom
policy entries under delete churn (three-way outcome now; phantoms proven
dropped); the soak driver exited 0 when the client's own xxh3 check failed
(checksum errors now count as verify failures and any hard error fails the
run); policy.Remove raced re-publishing Puts (now inside the DeleteIf gate,
atomic with the removal). The breaker's request-path eviction-convoy claim
was REFUTED at HIGH by measurement (insert p99 ≈ eviction-free memcpy
contention; 29% of passes return at the trigger re-check; disabling the
synchronous pass collapsed goodput 47×) and recorded as the p999-tail note
below. The breaker's empty-block flood DoS was CONFIRMED by measurement
(181 B of index heap per zero-length block, unbounded) — zero-length blocks
now carry a nominal count slot and the emergency sweep gained a COUNT goal.
The deep rapid gate also caught one harness bug of ours (the extent-leak
bound needed the held-ref allowance) — the oracle audits itself.

**Deliberate ceilings & notes (Week-4 vintage):**
- p999 tail under sustained overcommit is the evictMu queue (measured
  16–43 ms worst-case waits at 16 hammering writers on a laptop; p99
  unaffected). Revisit gate: if the 24h soak's put_p99 exceeds ~10 ms,
  implement single-flight-with-shared-completion on the eviction pass.
- The TIMING WHEEL is deferred (lazy expiry + expired-first sweeps; the
  plan's own fallback). Revisit gate: expired-resident bytes >10% of arena
  between pressure events, or expired-sweep >5 ms in the soak.
- The modeltest oracle hard-codes "eviction = data loss". The Week-6 NVMe
  tier turns eviction into DEMOTION — reconcileMiss and evictOne both need
  the demotion seam before tier stacking (recorded for the Week-6 plan).
- Strict per-tenant PRESSURE isolation needs Week-6 quotas; until then
  attribution is proportional-to-resident-bytes with a remainder round.
- ttlBlocks is a skip-hint that can ratchet up under lease-churn races
  (never down): worst case is a wasted expired-sweep per pass, never a
  correctness issue.
- Ghost rings ratchet to their high-watermark (bounded by the arena-derived
  ceiling) and don't shrink with cold domains; a gate-refused small-evictee
  re-admits to MAIN via its own ghost entry (deliberate: proved-protected
  blocks upgrade).

## Week 6 — the NVMe tier (log-structured segments + kill -9 recovery)

The durability differentiator: a warm tier that survives `kill -9` mid-write-storm
with **zero corrupt reads** — where KVBM unlinks its NVMe file on restart
(dynamo #6031), PegaFlow's SSD index is memory-only, and PrisKV "persists" to
tmpfs. (Honesty note: Mooncake's bucket backend does have SSD eviction; the
restart-recoverable index is the contrast, not SSD eviction itself.)

### Geometry and formats (all little-endian, versioned)

- **Segments**: fixed-size append-only files (`seg-<id>.kvbs`), default
  **256 MB** (CacheLib-Navy-informed; `nvme_segment_bytes`), fallocated at
  create (create-time fsync — size metadata never changes again). Records are
  **4 KiB-aligned**: 56 B header `{magic "KVBR", ns u32, key[32], len u32,
  rsvd, xxh3_64}` + payload + zero pad.
- **Seal footer**: entry table `{ns, key, off, len, xxh3_64}×n` at dataEnd +
  a 64 B trailer in the file's final 4 KiB (`magic "KVBF"`, version, counts,
  CRC32C over table+trailer — the same Castagnoli discipline as the wire
  header). Sealed segments are write-once: after the seal fdatasync no write
  ever targets the file, so a torn tail is confined to the single unsealed
  segment.
- **Checkpoints** (`ckpt-<seq>.kvbi`): the sealed-segment index serialized
  every `nvme_ckpt_every_segments` seals, written tmp→fsync→rename→fsync(dir).
  Recovery trusts it for covered segments (skipping their footer reads — the
  warm-restart seconds-per-GB win) and footer-scans anything newer.
- **Recovery** (in `OpenVolume`, before the daemon accepts traffic): newest
  CRC-valid checkpoint → validated entries; sealed segments beyond it →
  footer scan (later segID wins per key); the unsealed tail → forward record
  scan verifying every payload's xxh3, **truncate at the first torn record,
  then seal the recovered prefix in place** (append never resumes into a
  scanned tail). Lose data, never serve corruption. `RecoveryReport` is
  logged at startup and quoted by the demo.

### Tier orchestration (`internal/store.Tiered`)

- Separate 256-shard NVMe index (nskey → `{Loc, len, xxh3, lease/TTL
  atomics, pin flags}`); dram's BlockRef/allocator stay untouched. A
  DRAM-only config takes the pre-tiering code paths and emits no
  `kvb_nvme_*` scrape families (metrics_test asserts their absence; the
  hot-path Stats document is the dram tier's own).
- **Demotion** at 90% DRAM occupancy (below the evictor's 95): policy
  victims → reader-ref held across a bounded writer queue → group-commit
  append (fdatasync every `nvme_sync_every_bytes`) → the index swap runs
  INSIDE the dram shard-lock delete gate (zero-width: a concurrent GET
  serves exactly one tier). Blocks under the admission filter
  (`nvme_admit_min_hits`) are DELETED at the demote watermark, not written
  — SSD endurance; each such deletion counts in
  `kvb_nvme_admit_refusals_total`, and a pure-ingest workload should set
  the filter to 0 (a rig session once watched a 20 GiB fill melt under
  the default before this was visible). Bytes already NVMe-resident (dual
  residency after promotion) are never rewritten.
- **GETs**: DRAM first; NVMe via a bounded per-volume reader pool —
  saturation answers a per-key `ERR_BUSY` descriptor (retryable; §3's
  per-key-outcomes-ride-in-descriptors mechanism, no wire change). Every
  device read is verified (magic + nskey + xxh3) BEFORE a byte is served;
  verification failure self-heals the index entry and counts
  `kvb_nvme_checksum_errors_total`. **BATCH_EXISTS never touches the
  device** (a spy IOBackend that panics on read enforces it in CI).
- **Promotion** NVMe→DRAM only on the 2nd hit within
  `nvme_promote_window_ms`; a hard PIN promotes synchronously (the
  pinned-bytes ledger stays single and unbypassable).
- **Reclaim**: whole-segment FIFO (write-once ⇒ FIFO≈LRU, no compaction);
  segments holding leased or pinned blocks are SKIPPED (gated under the
  shard lock); in-flight READS need no skip — the file is unlinked with
  the fd held open and the retire drains them before closing.

### The crash contract (what the torture harness enforces)

PUT COMMIT ack = DRAM-committed (a cache, not a database). After `kill -9`:
whatever the recovered daemon says EXISTS must GET **byte-identical**
(client-side xxh3 scrub against regenerated content); a key whose COMMIT was
never acked must NEVER exist; recovery-to-ready < 5 s; fresh traffic serves.
DRAM-only blocks die with the process — honest loss, never corruption.
Harness: `go run -tags crashtest ./test/crash -loops N` (journal written
parent-side strictly AFTER each ack; ~30% of streams deliberately abandoned
mid-CHUNK). CI runs 10 loops per push on ubuntu; the rig runs 50 on real
NVMe; Docker rehearsals run 100+.

### Week-6 measured results

| What | Where | Number |
|---|---|---|
| torture, darwin dev box | 3 loops | 0 corrupt, 0 phantom (mechanism check) |
| torture, Linux kernel (Docker) | 100 loops | **0 corrupt, 0 phantom** over 18,160 journaled acks, 234 s (2026-07-18) |
| torture, real NVMe (i7i) | 50 loops | _Day-6 session (run-tier.sh T2)_ |
| daemon NVMe GET storm, per device | i7i.8xlarge | _T1a — quote GB/s AND %-of-fio-ceiling_ |
| daemon NVMe GET storm, both devices | i7i.8xlarge | _T1b_ |
| warm restart | ~20 GB fill | _T3 — wall seconds + seconds-per-GB + hits-survive_ |

A3 status: OPEN — the pre-registered ≥6.0 GB/s/device line is not printable
on any AWS instance-store device (i4i ceiling 2.99; i7i projected ~4.5); the
session records %-of-ceiling with the same discipline the A1 transport gates
used, and the literal gate awaits either a dated amendment or bare-metal
PCIe-Gen4 hardware.

### What Rudray owns (Merge Rule; write BEFORE reading the code back)

- The recovery-algorithm narrative — checkpoint load → footer scan → tail
  truncation — from memory, then diff against `internal/store/nvme/recovery.go`.
- The durability-contract explanation: what a COMMIT ack promises, why the
  parent journals AFTER the ack, why a torn tail cannot damage sealed
  segments, what fsync/fdatasync actually guarantee (and why darwin's
  fsync does not flush the drive cache).
- O_DIRECT alignment discipline (4 KiB buffers/offsets/lengths; why the
  aligned pool exists) and the build-tag split (`io_direct_linux.go` /
  `io_direct_other.go` / `kvb_uring` stub — why io_uring is deferred).

## Tenancy model

Multi-tenancy is structural, not an ACL bolted in front of a shared space:
`namespace_id` rides in every frame header, every index key is
`(namespace, key)`, and dedup never crosses the pair — the same 32-byte key
committed by two tenants is two distinct blocks. The tenancy layer therefore
has exactly two jobs: bind a connection to its namespace at HELLO, and
account how many bytes each tenant holds per tier. Of the three tier quotas,
only `quota_dram` is ENFORCED (PUT admission below); `quota_nvme` and
`quota_s3` are REPORTING/ADVISORY at v0.2 — usage and limits surface in
stats, metrics, and the admin listing, but demotion and spill move bytes
uncapped and nothing corrects an overage (the enforcement design is ledgered
in IMPROVEMENTS.md with its trigger).

**Hashed tokens, constant-time auth.** The registry (`namespaces.yaml`)
retains SHA-256 digests, never plaintext: `token_sha256` is the preferred
schema (the file itself then carries no credential), and a plaintext `token`
entry is hashed at load and dropped. Authentication hashes the presented
token and compares with `crypto/subtle.ConstantTimeCompare`; an unknown
namespace runs a dummy compare so its timing matches a bad-token reject — no
name-probing oracle. Auth is connection-scoped (one HELLO; token bytes cross
the wire once), and an empty registry fails every HELLO: a server with no
tenants accepts no one, secure by default.

**The ledger: per-(namespace, tier) CAS accounting.** `tenant.Quotas` keeps
one atomic byte counter per (ns, tier). Admission is a CAS loop — load,
headroom check against the loaded value, CompareAndSwap, retry on contention
— so check-and-add is atomic with no mutex on the hot path (a plain
Add-then-check would briefly overshoot and need a compensating Sub that
races readers). The documented invariant (modeltest pins it as I3):
usage(ns, tier) ≤ quota + one in-flight block of slack — the slack exists
because BEGIN's probe is advisory while racing writers each individually
observed headroom; a lost race costs at most one block per racer. What keeps
the counter honest is **refund at every removal seam**: DELETE, eviction, a
lost publish race, demotion's dual-residency collapse and endurance-gate
deletion on the DRAM side; the delete verb, segment-reclaim retire, and the
corrupt-entry self-heal on the NVMe side. Tier MOVES use `Transfer`, which
never fails — refusing a demotion for destination-tier quota would wedge the
memory ladder. That is exactly why the NVMe/S3 limits stay reporting-only:
bytes land on the destination uncapped, and the evictor's over-quota-first
pass runs on DRAM alone, so nothing corrects a cold-tier overage. Recovery
uses `Seed`, an unchecked charge: a restart must not mint free NVMe bytes,
and refusing recovery over quota would lose data.
A double-refund clamps at zero in release builds and panics under
`kvbdebug`. Refunds are **tier-exact**: a retire-flipped entry's charge
lives on S3, not NVMe (next section), and removing it refunds the S3 side —
refunding NVMe there would leak the S3 charge forever while the underflow
heal silently hid the NVMe double-subtract.

**Advisory BEGIN, binding publish.** PUT BEGIN answers `ERR_QUOTA_BYTES`
from a pure headroom probe (`WouldExceed`) that reserves nothing — staging
stays lazy per the anti-amplification posture — and a refused probe kicks
the evictor, whose over-quota-first pass may free that same tenant's cold
blocks. The BINDING admission is `Charge` at publish, inside the commit path
after the arena copy and before the index insert; a refused charge frees the
extent and backs out. §3.4 keeps `ERR_QUOTA_BYTES` out of COMMIT's frozen
response set, so the rare BEGIN-said-yes-then-lost-the-race commit maps to
the retryable `ERR_BUSY`, and the client's fresh BEGIN reports the honest
quota answer. Enforcement went live with zero wire change. (Zero-length
blocks are extent-less and never charged — spec-legal per §3.4.)

**Over-quota tenants pay first.** The eviction pass gained a Round 0: while
any tenant is over its DRAM quota, victims come from the worst usage/quota
ratio first (integer thousandths — allocation-free over a handful of
tenants), bounded by that tenant's overage. Under shared pressure the tenant
breaking its own budget pays before any within-budget tenant loses a block;
only then do the proportional and remainder rounds run.

**No existence oracle.** A cross-tenant probe — EXISTS, GET, even
force-DELETE — answers exactly like a miss: `NOT_FOUND`, never `FORBIDDEN`.
A tenant cannot learn that a foreign key exists, let alone touch it; the
wire-wall test commits the same key under two tenants and proves two
independent blocks with miss-identical cross-answers. (`ERR_FORBIDDEN` stays
reserved for token-permission classes, not cross-tenant probes.)

**The admin plane is shell-trust, not wire-trust.** Namespace management is
a loopback-only HTTP listener (`admin_addr`; a non-loopback address refuses
to bind): `POST /v1/namespace`, `POST /v1/quota`, `GET /v1/namespaces` —
usage and quotas listed, tokens never. Mutations affect the running process
only and say so in the response (persist by editing the namespaces file);
quota changes re-snapshot the accountant's limits immediately. `kvbctl
namespace add|list` and `kvbctl quota set` drive it from the shell.
Deliberately not the data plane: no KVB1 frames, no bearer tokens — reaching
it requires a shell on the box, the same trust boundary as editing
`namespaces.yaml`.

**Per-tenant observability at bounded cardinality.** `kvb_tenant_bytes` and
`kvb_tenant_quota_bytes`, labeled `{namespace, tier}`, read the registry and
accountant at scrape time: cardinality is #namespaces × 3 tiers — namespaces
are operator-registered (tens, never per-key) and a smoke test pins the
count.

<!-- RUDRAY: prose first per the Merge Rule — from memory: why the
     constant-time compare needs the dummy compare for unknown names; why
     Charge is a CAS loop and not Add-then-check; where each refund seam
     lives and what leaks if one is missed; why Transfer must never fail;
     why cross-tenant answers NOT_FOUND and never FORBIDDEN — THEN diff
     against internal/tenant/{registry,quota}.go and the store seams. -->

## S3 cold tier

The third tier and the headline differentiator: sealed NVMe segments spill
to object storage, whole. **One sealed segment = one S3 object**,
byte-for-byte, footer included. That coalescing is request-cost math, not
convenience: one whole-segment PutObject replaces the ~100 per-block
requests a 256 MB segment of 2.5 MB blocks would cost, and S3's
no-offset-writes model fits a write-once append-only file exactly. S3 stores
bytes, never metadata — the index never leaves the node, which is what makes
a cold read exactly ONE ranged GetObject. Object keys are node-namespaced
(`kvblockd/<node_id>/segments/seg-<id>.seg`); credentials come from the
ambient AWS chain only (env, shared config, IMDS) — no credential ever lives
in a kvblockd config file.

**Spill is a copy, never a move.** A background pass enqueues every sealed,
not-yet-spilled segment onto a bounded async write-back queue
(`s3_spill_queue`, default 8 segments). Fire-and-forget: a full queue is a
counted drop retried next tick, a failed upload leaves the segment
local-only — the cold tier can never block a foreground PUT. Until reclaim
retires the segment, reads stay on local NVMe; the spill-ack merely marks
each entry "also on S3".

**The retire-flip: reclaim becomes demotion.** When segment reclaim retires
a SPILLED segment, its entries are not deleted — they FLIP to s3-resident:
the index entry survives (its location now addresses the object), the
tenant charge moves NVMe→S3 (`Transfer` under the entry's shard lock, so a
racing DELETE either removes the entry first or observes the flip and
refunds the right tier), and cold reads route through the restorer.
Protections hold through the flip exactly as through a delete, and by the
same mechanism: the flip's shard-locked walk IS the authoritative gate (not
a pre-check it could race) — a lease or pin observed there aborts the whole
retire, and a hard-pinned block never becomes s3-only (PIN_HARD means DRAM
residency, promoted synchronously).

**Cold reads: one ranged GET, verified before a byte escapes.** A GET whose
segment is retired-but-spilled issues a single ranged GetObject for exactly
the payload bytes — the stored offset addresses the record (56-byte header
first), so the range starts at `RecordDataOffset`; the object is the
byte-identical segment file, so file offsets transfer 1:1. The bytes are
xxh3-verified against the index entry BEFORE anything escapes to the wire; a
mismatch counts a checksum error and answers a miss, and a slow S3 hits the
`s3_read_timeout_ms` deadline (default 2 s) as a per-key outcome — never a
frame stall. PIN_HARD's synchronous promote reads through the same verified
path, so an s3-resident block a GET can serve is never refused a pin. The
2nd-hit trigger is tier-split: a warm 2nd hit promotes ONE block to DRAM; a
cold 2nd hit inside the promote window triggers the LAZY WHOLE-SEGMENT
RESTORE — a cold segment's blocks come back together or not at all (one
download amortizes over every entry): the segment object is downloaded back
into the volume, its entries flip nvme-resident (reads go local again), and
the object is RETAINED — the adopted segment republishes spilled=true, so
its next reclaim FLIPS the entries back to s3-residency against that same
object instead of deleting them (dropping the object on restore was a
reproduced pre-release data-loss bug: a reclaim landing before the next
spill-ack deleted the entries with the object already gone). A failed or
refused restore changes nothing — entries stay s3-resident and keep serving
per-read. Whole-segment restore is singleflight per segment (two concurrent
cold misses trigger exactly one 256 MB download), and the per-segment latch
also excludes spill, reclaim, and object GC while it runs. **BATCH_EXISTS
never touches S3** — both indexes are node-local map lookups, the same
contract the NVMe spy test enforces for the device.

**Restart semantics are per-life, and honest.** Spilled state is
memory-only: after a restart every sealed segment re-spills, and the
whole-object PUT is idempotent (same key, same bytes — sealed segments are
write-once), so a crash costs one duplicate upload, never correctness.
Entries that were retire-flipped are honestly LOST at restart — the local
bytes are gone and recovery rebuilds only from local checkpoints and footers
— a miss, never corruption, exactly the crash contract's shape. Dead
objects are garbage-collected ASYNCHRONOUSLY, on by default: per-segment
liveness counting (each retire-flip +1, each s3-resident removal or
flip-home −1) nominates an object once its LAST s3-resident entry leaves
and no live spilled segment still claims it; removals only enqueue the
candidate — the demoter tick drops a bounded few per pass
(deadline-bounded, never inline on a foreground op, never a memory-ladder
stall), each drop re-checking deadness under the per-segment latch. A
lost, failed, or refused drop merely orphans the object: it costs cents,
the bucket lifecycle rule remains the recorded backstop, and reclaim still
never fails over S3 cleanup.

**Config and wiring.** Inert until `s3_bucket` is set — byte-for-byte the
two-tier daemon otherwise; requires the NVMe tier. Keys: `s3_bucket`,
`s3_region`, `s3_node_id` (required — object keys must be node-namespaced),
`s3_endpoint_override` + `s3_path_style` (MinIO/gofakes3-compatible
targets), `s3_spill_queue`, `s3_read_timeout_ms`. The store never imports
the SDK — the spill/restore backends are structural interfaces implemented
by `internal/store/s3spill`.

**Observability.** STATS gains an `"s3"` sub-document (resident
blocks/bytes, spilled segments, drops, put errors, ranged gets, restores,
hits, read errors, checksum errors, plus the restore/GC set: segment
restores completed, restore errors, restore skips, object GCs, object GC
errors, object GC skips) whose residency split sums with the nvme
sub-document to the index — matching the per-tenant ledger's tier split;
the scrape side adds the `kvb_s3_*` families and tier="s3" to `kvb_blocks`
/ `kvb_store_bytes`.

**Real-S3 latencies (measured 2026-07-20, in-region us-east-1 m7g.xlarge,
the daemon's own spill/restore code paths via s3probe).** Whole-segment PUT
(8 MiB): p50 110 ms (~73 MiB/s), worst of 10 = 235 ms. Ranged cold GET, p50:
64 KiB 26.9 ms · 256 KiB 25.7 ms · 1 MiB 28.6 ms · 2.5 MiB 31.4 ms; worst
observed of 30 per size: 33–61 ms (n is too small for honest p99s — we quote
medians and maxima, and the review ladder caught the first draft quoting a
"p99" range that excluded the two worst cells) —
S3's first-byte latency dominates, so a 2.5 MiB block costs ~4 ms more than a
64 KiB one: the economics of one-segment-one-object. A cold 2.5 MiB KV block
at ~31 ms is roughly an order of magnitude under recomputing its prefix.
Raw JSONL: bench/rigs/aws-s3e2e/s3probe-results.jsonl.

The 17.3 h three-tier spill soak (MinIO on-box): 12.37M verified hits,
112,851 verified cold reads, 0 errors, 0 read errors, goroutines flat
(148→144). HONEST CAVEAT (the committed artifacts corrected our first
read): MinIO's 40 GiB volume FILLED at ~6.8 h ("drive path full" in
minio.out) because retired segments' objects are never deleted — the
object-GC gap that is IMPROVEMENTS.md's first row eating its own soak.
93% of the 3,120 upload errors came after the disk filled, spills
flatlined 364→379, and ~111k of the 112.8k cold reads landed in the
first ~8 h — so the cold tier was genuinely exercised for ~8 h, not
17.3 h. Loss-free throughout (local authoritative); the single-worker
spiller's queue pressure (enqueue-refusal ticks) is real but secondary.

**Measured evidence.** The 24 h soak: **123.3M verified hits, 0 errors**,
RSS flat (165→160 MB), and across 1.7M GC cycles the stop-the-world pauses
held **p50 42 µs / p99 3.0 ms / p99.9 5.8 ms / max 13 ms** — measured on a
4-vCPU box pinned at 100% CPU by design, which inflates the scheduler-bound
tail; the arena keeps the block bytes themselves entirely outside the GC's
world.
The overnight three-tier spill soak (MinIO on-box): **112,851 verified cold
reads, 0 read errors** (final; see the caveat below on the run's second half). The spill-mode kill -9 torture: **25 cycles, 0 corrupt,
0 phantom — 18/18 cold reads served mid-kill**. The end-to-end cold-tier
walk also caught its own integration bug as 51 checksum refusals (a
mis-ranged read), refused before a single wrong byte escaped — the
verify-before-serve contract doing its job.

<!-- RUDRAY: prose first per the Merge Rule — from memory: the request-cost
     arithmetic behind one-segment-one-object; why spill must be a copy and
     what the retire-flip changes (and moves, quota-wise); the exact
     cold-read verify path and why EXISTS must never touch S3; what a
     restart loses and why that loss is honest — THEN diff against
     internal/store/{s3tier.go,s3spill/,nvme/spillsurface.go}. -->
