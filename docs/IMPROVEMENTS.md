# Improvements ledger

Known gaps, deferred work, and tuning headroom — with the reason each item
waits and the trigger that revives it. This file exists so a deferral never
silently becomes a forgotten defect: if it's not fixed, it's listed; if it's
listed, it has a trigger. Companion to ROADMAP.md (that's direction; this is
debt and headroom). Dates are when the entry was recorded.

## Cold tier (S3)

| Item | Why it waits | Revisit trigger |
|---|---|---|
| Volume-scoped S3 object keys: `segKey` is (node, segment-id) with no volume dimension while every volume numbers segments from 0, so multi-volume + S3 would overwrite objects (flipped entries then checksum-fail forever); config.Validate refuses the pair outright, and `spillInflight`, the object-GC liveness map (`s3SegRefs`), and the restore/GC candidate queues are all segID-keyed (volume-blind) under the same guarantee (2026-07-20; widened 2026-07-21 when lazy restore + object GC landed volume-blind) | Key-layout change ripples into restore addressing, recovery's spilled-marker mapping, and the now-landed GC/restore liveness accounting — one coordinated change, not a hotfix | First deployment needing multi-volume WITH the cold tier |
| NVMe/S3 quota enforcement: `quota_nvme`/`quota_s3` are REPORTING-ONLY — `Transfer`/`Seed` land bytes unchecked and no pass corrects an overage (the evictor's over-quota-first round is DRAM-only) (2026-07-20) | Enforcing at demote/spill would wedge the memory ladder (Transfer must not fail); real enforcement needs an NVMe-side over-quota reclaim preference — design work on reclaim ordering | A deployment actually selling per-tenant NVMe/S3 capacity; or the first operator report of one tenant squeezing others' cold tiers |
| Async-spill coverage in the model harness: the machine drives a synchronous fake (deterministic); the REAL async spiller's queue/ack interleavings are covered by torture + soak, not by the model oracle (2026-07-20). The machine DOES now drive restore + object GC through the same synchronous fake (2026-07-21) | Async completion racing a model step needs a drain-barrier design in the harness | Next harness session; or immediately if a spill-adjacent bug escapes to torture |
| Spill-soak upload errors were 93% caused by the object-GC gap, not spiller throughput: MinIO's 40 GiB volume filled at ~6.8h (unbounded object growth ~4.5 GB/h), spills flatlined, and the run's last ~11h did not exercise the cold tier (2026-07-20 soak — the ladder caught the first writeup misattributing this to "spiller fell behind") | Object GC landed 2026-07-21 (per-segment liveness accounting; drops on last-entry removal and restore-back) — the re-run is unblocked; the single-worker queue pressure is real but secondary — N workers + per-segment backoff + a distinct-segments drops counter remain worth doing | Re-run the spill soak with disk sized for the working set — with the rig asserting the bucket-lifecycle rule exists as a precondition (the object-GC orphan backstop, next row); then judge spiller throughput honestly |
| Object-GC orphan channels are one-shot: a failed `Drop` (`object_gc_errors_total`), a full `dropq` and a latch-busy drop (`object_gc_skips_total`), and restart amnesia (the `s3SegRefs` liveness map is in-memory) each orphan the object with no in-process retry — the only backstop is a bucket-lifecycle rule the daemon cannot verify exists (2026-07-22) | Every channel is counted and an orphan costs storage, never correctness; the honest fix — a bounded retry ring re-nominating failed drops — needs care against the one-shot `s3GCQueued` dedupe, and lifecycle already reaps what it would retry | The spill-soak re-run: add the bounded retry ring for `Drop`-failed candidates first, and make the rig assert the lifecycle rule before the run counts |
| Reclaim defers to any in-flight spill/restore/GC holding a segment's `spillInflight` latch (the restore-overtake data-loss fix): a mid-upload oldest segment is owned for seconds to tens of seconds. Mitigated 2026-07-22 — latch-busy now SKIPS to the next-oldest candidate in the same pass (floor walk; ≤8 skips + 4 retires per volume per pass) instead of ending the pass, so one owned segment no longer pins the volume ≥100% while Appends refuse; deferrals stay counted (`reclaim_skips_total`, drops in `demote_drops_total`) | The residual trade is deliberate: a pass whose 8+ oldest candidates are ALL latch-owned (deep spill backlog) still frees nothing that tick and retries 100 ms later — the latch's correctness outranks a tick of reclaim latency | Soak/rig telemetry showing `reclaim_skips_total` climbing together with `demote_drops_total` while a volume sits pinned — then widen the skip budget or add spill-aware reclaim ordering |
| LOW: GC budget-slot burn — every flipped retire enqueues an object-GC candidate (deadness is re-checked at the pass, not the enqueue), so live-segment false positives each consume one of the 4 `gcDropBudget` slots per pass (2026-07-22) | Wasted slots only delay dead-object drops by 100 ms ticks; enqueue-always is what keeps the retire path free of GC decisions and GC I/O | If `object_gc_skips_total` climbs on real traffic, or dead objects age out via the lifecycle rule instead of the GC |
| LOW: restore/reclaim ping-pong — an adopted segment is typically the volume's oldest sealed, i.e. the FIFO reclaimer's first victim, so a hot cold segment can cycle download → flip-home → flip-out under sustained pressure, a whole-object GET per lap (2026-07-22) | The adopt already refuses without headroom and kicks the reclaimer instead, which absorbs the worst of it; real hysteresis (adopt immunity for N ticks, or re-aging the adopted id) is reclaim-policy design | Telemetry showing the same segment restoring repeatedly — `segment_restores_total` rising in lockstep with `reclaims_total` |
| LOW: garbage adopt — a restore whose entries were ALL deleted mid-download still publishes a zero-live-entry sealed segment, wasting up to one segment of volume space until reclaim sweeps it (its retire flips nothing and the object then GCs clean) (2026-07-22) | Self-corrects under pressure — the adopted segment is oldest, so the next reclaim pass retires it — and costs at most one segment; a pre-flip "any ref still homed?" recheck would add a walk to every restore for a rare race | If restore-heavy telemetry shows adopts immediately followed by their own reclaims at rate |
| LOW (pre-existing): persistent cold-object rot never self-heals — `readS3` checksum-fail counts (the s3 doc's `checksum_errors_total`) and misses but KEEPS the entry, unlike the NVMe `ReadCorrupt` path (identity-gated delete + refund), so a rotted object leaves permanent phantoms: indexed-but-unservable keys pinning S3 quota and `s3SegRefs` liveness, which in turn keeps the dead object from ever GCing (2026-07-22) | The fix is small and known — mirror ReadCorrupt's delete + `refundNvmeQuota` on the cold path (the refund helper already handles the S3 side and the liveness decrement) — but it edits the verified-read path right before a release | First nonzero cold `checksum_errors_total` in any soak or deployment; land the mirror-fix then, with a rot-injection test |
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
| Heavy external-integration CI legs have never gone green on first contact (2026-07-21): `e2e-cpu` sglang real-ABC import → `TypeError: int() ... not NoneType`; `vllm-native-cpu` full CPU serve; `nixl` meson configure (NIXL built from source). NOT on the main-push path (PR/dispatch/cron only) so main stays green; our own correctness is covered by ci.yml (unit + race + torture + the vLLM-adapter UNIT job, all green) | Each leg installs a GB-scale upstream / builds a C++ dep from the internet — fragile, drift-driven, and they last failed inside a Dependabot PR's restricted context (no secrets), which muddies the signal. Real triage needs a clean dispatch run per leg | Before relying on any adapter's live-integration claim; also add an `if: github.actor != 'dependabot[bot]'` guard so action-bump PRs stop tripping them |

## Binary size

| Item | Why it waits | Revisit trigger |
|---|---|---|
| `kvb_nos3` build tag to strip aws-sdk-go-v2 and reclaim a ~14 MB SDK-free binary (the S3 tier's SDK is ~2.5 MB of the ~21 MB release binary; the assert_static gate was raised 20→24 MB to accept the full-featured build) (2026-07-21) | Needs the SDK-importing seam (s3spill client) split behind `//go:build !kvb_nos3` with a stub for the tag, plus a goreleaser matrix leg — real work, and the full binary is still a single dependency-free static binary (the claim that matters) | When a deployment cares about the smaller binary, or before a "tiny binary" marketing push; ship it as a second goreleaser artifact |

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
