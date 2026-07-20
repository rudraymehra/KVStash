# Improvements ledger

Known gaps, deferred work, and tuning headroom — with the reason each item
waits and the trigger that revives it. This file exists so a deferral never
silently becomes a forgotten defect: if it's not fixed, it's listed; if it's
listed, it has a trigger. Companion to ROADMAP.md (that's direction; this is
debt and headroom). Dates are when the entry was recorded.

## Cold tier (S3)

| Item | Why it waits | Revisit trigger |
|---|---|---|
| Volume-scoped S3 object keys: `segKey` is (node, segment-id) with no volume dimension while every volume numbers segments from 0, so multi-volume + S3 would overwrite objects (flipped entries then checksum-fail forever); config.Validate refuses the pair outright, and `spillInflight` is segID-keyed (volume-blind) under the same guarantee (2026-07-20) | Key-layout change ripples into restore addressing, recovery's spilled-marker mapping, and the object-GC design — belongs with that work, not a hotfix | First deployment needing multi-volume WITH the cold tier; land with (or before) object GC |
| NVMe/S3 quota enforcement: `quota_nvme`/`quota_s3` are REPORTING-ONLY — `Transfer`/`Seed` land bytes unchecked and no pass corrects an overage (the evictor's over-quota-first round is DRAM-only) (2026-07-20) | Enforcing at demote/spill would wedge the memory ladder (Transfer must not fail); real enforcement needs an NVMe-side over-quota reclaim preference — design work on reclaim ordering | A deployment actually selling per-tenant NVMe/S3 capacity; or the first operator report of one tenant squeezing others' cold tiers |
| Object GC: the daemon never deletes retired segments' S3 objects (`SpillBackend.Drop` has no production caller); the bucket lifecycle rule is the only cleanup (2026-07-20) | Needs per-segment liveness accounting (delete the object only when its LAST s3-resident entry is gone) — real design work, not a one-liner | Before any deployment where S3 storage cost matters; candidate to land with lazy restore |
| Lazy whole-segment restore: cold blocks are served per-read (one ranged GET each); repeated hits should pull the whole segment back to NVMe and flip entries home (2026-07-20) | Scheduled this sprint (in-repo restore machinery exists; orchestration half remains) | In the current sprint queue — do not let it slip past v1.0.0 |
| Async-spill coverage in the model harness: the machine drives a synchronous fake (deterministic); the REAL async spiller's queue/ack interleavings are covered by torture + soak, not by the model oracle (2026-07-20) | Async completion racing a model step needs a drain-barrier design in the harness | Next harness session; or immediately if a spill-adjacent bug escapes to torture |
| Spill-soak upload errors were 93% caused by the object-GC gap, not spiller throughput: MinIO's 40 GiB volume filled at ~6.8h (unbounded object growth ~4.5 GB/h), spills flatlined, and the run's last ~11h did not exercise the cold tier (2026-07-20 soak — the ladder caught the first writeup misattributing this to "spiller fell behind") | Object GC is the fix (first row above); the single-worker queue pressure is real but secondary — N workers + per-segment backoff + a distinct-segments drops counter remain worth doing | Re-run the spill soak AFTER object GC lands, with disk sized for the working set; then judge spiller throughput honestly |
| Spill-soak RSS drifted +16 MB over 17.3h (230→246 MB; soak-1 was flat) — hourly heap pprofs retained in bench/rigs/aws-s3soak/results/ (2026-07-20) | Small, possibly SDK/http buffer pools; needs a heap-diff to attribute | Heap-diff before v1.0.0; escalate if a longer soak shows monotonic growth |

## Hardware / quota gated

| Item | Why it waits | Revisit trigger |
|---|---|---|
| Chart 2 (TTFT vs vLLM recompute) — Rig E runbook staged | AWS G-family quota is 0; spot case denied, appeal + on-demand case pending | The day quota lands (or a second account) |
| vLLM `tier_manager` GPU e2e (code + unit tests done; DEFER.md in package) | GPU session costs real cash (RunPod ~$5); decided against this sprint | Before any public tier-manager announcement, or when a GPU hour is funded |
| SGLang HiCache GPU e2e (CPU suite green; rig scripts written, marked NOT-RUN) | Same GPU-budget decision; SHIP/DEFER verdict is DEFER with this exact blocker | Same as above — the written rig runs in ~4h when funded |
| 100 GbE full-matrix Chart-1 re-run (kvblockd measured 12.67 GB/s there; baselines only projected) | ~$4 and a rig session; 50 GbE matrix already saturates the wire | Before quoting the ~14× multiple anywhere formal |
| Mooncake-TCP baseline in Chart 1 | Timeboxed out of the rig session | Standing re-run offer — first request from a reviewer/partner |
| A100 datapoints | C-21 discipline: not until the cheaper pipeline produced 3 stable points | 4090 pipeline stability |

## Upstream / third-party gated

| Item | Why it waits | Revisit trigger |
|---|---|---|
| SGLang v2 controller methods (stubbed `NotImplementedError`) | Upstream interface still churning (sgl-project/sglang#18239) | A tagged SGLang release stabilizes v2 |
| NIXL native C++ plugin — beta posture | C-11 reclassified it stretch (segfault-class unknowns vs a zero-code S3-compat path that already works) | NIXL CI green across HEAD+tag for a sustained window; a partner asks for the native path |
| vLLM connector churn-watch (`SharedStorageConnector`→`ExampleConnector` rename; RFC tier-dict loading never merged) | Upstream moves faster than releases; A6 matrix + UPSTREAM.lock are the tripwire | A6 matrix failure on any new vLLM release |

## Performance headroom (unscheduled, measured)

| Item | Why it waits | Revisit trigger |
|---|---|---|
| GC stop-the-world tail: p99 3.0 ms / p99.9 5.8 ms measured on a 4-vCPU box pinned at 100% CPU (p50 42 µs; block bytes are outside the GC by design) | Tail is scheduler-bound under deliberate oversaturation; right-sized boxes shrink it for free | If a latency-sensitive deployment scrapes p99 pauses >1 ms at sane CPU headroom; then GOGC/GOMEMLIMIT tuning pass |
| Deferred wire-path levers on a real NIC (MSG_ZEROCOPY at larger blocks, sendfile on the NVMe read path, io_uring) | Loopback showed the syscall/memory-bandwidth bound is already ~0.97× raw socket; real-NIC gains need a rig, and io_uring was explicitly pushed to a hand-written learning spike (C-33) | The io_uring spike week; or a rig session showing the daemon below wire ceiling on 100 GbE+ |
| Allocator honesty ceiling: 2^17 live allocations (Allocation.Meta slot bits) — arenas above ~8 GiB of 64 KiB blocks hit the pool before capacity | Documented in code; widening Meta touches the hot allocator | First deployment with a >50 GiB arena of small blocks |
| Promotion channel is best-effort (64-deep, drops on full) | Promotion is an optimization; drops are invisible-but-uncounted | Add a counter first (cheap); revisit sizing if it's hot on real traffic |
| Soak GET latencies (p50 ~3 ms) reflect deliberate 6× oversaturation of a small box, not service latency | Working as intended — soaks measure survival, rigs measure speed | Never quote soak latencies as performance; a right-sized latency benchmark is a separate rig if ever needed |

## Process / harness

| Item | Why it waits | Revisit trigger |
|---|---|---|
| Deep model gauntlet (20000×500, -race) does not fit the 8 GB laptop's Docker VM; runs on a 16 GB Linux box (~$1) or CI dispatch instead | Physics of -race memory × long walks | Each release gate: run it on the Linux box; laptop runs the 200-step leg |
| macOS torture runs can hit TMPDIR purging under disk pressure (one non-reproducible phantom traced to this) | Environment artifact; Linux is the authoritative platform | Run macOS tortures with `-dir` on a non-TMPDIR path if it recurs |
| Calibration drills (~monthly seeded-bug recall check on the review ladder) | Cadence item | Next drill due within a month of the last ladder ledger row |
