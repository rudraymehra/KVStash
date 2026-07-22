# Roadmap

Direction after v0.2.0. This file is the companion to
[IMPROVEMENTS.md](IMPROVEMENTS.md) — that ledger is debt and headroom with
revisit triggers; this one is where the project goes next and what gates
each step. Two honesty rules apply to everything below: an item gated on
external resources (GPU hours, hardware, upstream releases) carries **no
date** — it lands when its gate opens, and the gate is written down; and a
claim graduates from this file only with measured evidence behind it, the
same %-of-ceiling discipline every published number already follows
([BENCHMARKS.md](BENCHMARKS.md), [DESIGN.md](DESIGN.md)).

## Shipped (v0.1.0-rc1 → v0.2.0 → `main`)

What exists today, with the receipts:

- **All three tiers.** DRAM (GC-invisible mmap arena, lease/pin/TTL ladder,
  S3-FIFO eviction, zero blob-band allocations on the GET path) → NVMe
  (log-structured 256 MB sealed segments, checkpointed crash recovery; the
  kill -9 torture harness holds the crash contract at **100 loops, 0 corrupt,
  0 phantom over 18,160 journaled acks**) → S3 (one sealed segment = one
  object, single ranged GetObject cold reads, xxh3-verified before a byte
  escapes; measured in-region latencies in [DESIGN.md](DESIGN.md)).
- **Cold tier completed (landed on `main`, changelogged as v0.5.0).** Lazy
  whole-segment restore — repeated cold hits pull the whole segment back to
  NVMe and flip its entries home, one download amortized across every
  entry — and S3 object GC: a retired segment's object is deleted once its
  last s3-resident entry is gone (refund-driven, bounded per demoter pass,
  backstopped by the bucket lifecycle rule). This closes the gap that filled
  the first spill soak's own disk; the receipt still owed is the GC-on
  spill-soak re-run (autopsy and triggers in
  [IMPROVEMENTS.md](IMPROVEMENTS.md)).
- **Tenancy + quotas.** Namespace registry with hashed tokens and
  constant-time HELLO auth; per-(namespace, tier) CAS byte accounting charged
  at publish and refunded at every removal seam; over-quota tenants evicted
  first; no cross-tenant existence oracle. Honest scope note: the DRAM quota
  is *enforced*; `quota_nvme`/`quota_s3` are reporting-only until the
  cold-tier reclaim-ordering design lands (ledgered with its trigger in
  IMPROVEMENTS.md).
- **Integration paths** — kvblockd is reached through the connectors people
  already run ([INTEGRATIONS.md](INTEGRATIONS.md)):
  - LMCache → vLLM `RemoteConnector` (`python/lmcache_kvblockd`) — shipped,
    CPU e2e in CI, upstream pins tracked by the interface tripwire.
  - vLLM native connector + offload tier manager (`python/vllm_kvblockd`) —
    code-complete and unit-tested against the pinned upstream contract
    (`UPSTREAM.lock`, golden fingerprint vectors); the tier-manager GPU e2e
    is deliberately deferred, not faked (`python/vllm_kvblockd/DEFER.md`).
  - SGLang HiCacheStorage v1 backend (`python/sglang_kvblockd`) —
    CPU-validated, verdict **DEFER** until a GPU e2e runs; blocker and
    revisit trigger in [design/sglang-hicache-v1.1.md](design/sglang-hicache-v1.1.md).
  - NIXL native C++ plugin (`adapters/nixl`) — **beta**: golden-vector wire
    parity with the Go/Python codecs, CI-tracked, not GA.
  - S3-compatibility endpoint (`s3compat_addr`) — a minimal S3 REST subset
    on its own listener; the zero-code path for NIXL's `obj` plugin and
    vLLM's `obj` tier via `endpoint_override`. Compatibility surface, not
    the performance path.
- **Release pipeline.** Tag-driven goreleaser v2 static matrix
  (linux/amd64, linux/arm64, darwin/arm64), `assert_static.sh` gate
  (statically linked, <24 MB), SBOMs, `FROM scratch` image, `install.sh`,
  systemd unit. Latest published release: **v0.2.0**. The v0.5.0 cut (the
  completed cold tier) is changelogged ([CHANGELOG](../CHANGELOG.md)) but its
  tag and artifacts are not yet published — it becomes the latest release
  only when they exist.
- **Benchmarks with the ceiling drawn on.** GET at **12.67 GB/s ≈ 102% of
  the iperf3 ceiling on a 100 GbE pair, xxh3 verify ON**; the Chart-1 matrix
  vs Redis 7 / Valkey 8 / redis-py at 100% of a 50 GbE wire; NVMe quoted as
  %-of-fio-ceiling (98.3%); a 24 h soak at 123.3M verified hits, 0 errors.
  Methodology and raw logs: [BENCHMARKS.md](BENCHMARKS.md),
  `bench/METHODOLOGY.md`.

## Next (the v1.1 direction)

Ordered by what unblocks the most. Each item states its acceptance and its
gate; none of the externally-gated items carries a date.

### 1. GPU-validated engine integrations (vLLM tier manager, SGLang, NIXL live rig)

The CPU-side code, unit suites, and CI tripwires exist for all three; what
is missing is the same thing in each case: a funded GPU (or live-NIXL)
session. The acceptance criteria are already written down so the sessions
are execute-only:

- vLLM tier manager: `OffloadingConnector` end to end on a real GPU —
  non-zero secondary-tier hits, byte-correct outputs, clean `drain_jobs`
  under load (`python/vllm_kvblockd/DEFER.md`).
- SGLang HiCache: token-identical multi-turn output vs a no-L3 baseline,
  remote hits visible in `/metrics`; PyPI publish and docs listing follow a
  green run, never precede it
  ([design/sglang-hicache-v1.1.md](design/sglang-hicache-v1.1.md)).
- NIXL: the live round-trip against a real NIXL build, both the S3-compat
  zero-code path and the native plugin.

Gate: a funded GPU hour. The announcements do not outrun the evidence.

### 2. TLS / mTLS

v1 transport is deliberately plaintext TCP on a trusted segment, with
TLS-termination guidance in the [deployment guide](deployment-guide.md).
The protocol already reserves the upgrade seam: **`FEAT_TLS_UPGRADE`,
feature bit 4, negotiated as intersection at HELLO**
([PROTOCOL.md §10](PROTOCOL.md)). The work: in-protocol TLS upgrade behind
that bit, mTLS as the tenant-channel option, zero cost when the bit is off.
Acceptance: the wire-path gate re-run with TLS on, the overhead published
as a number rather than adjectives.

### 3. io_uring NVMe engine (experiment, spike-gated)

The measured baseline says the thread-pool engine is not the bottleneck on
current hardware: **98.3% of the fio device ceiling per device, 98.5%
aggregate, at 0.1–0.25 cores** ([DESIGN.md §A3](DESIGN.md)). The NVMe tier's
`IOBackend` seam keeps an io_uring engine pluggable (decision recorded in
`bench/microbench/nvmeprobe/io_linux.go`; giouring is dead on Go 1.26, so
bindings are part of the question). Next step is a hand-written
io_uring-vs-threadpool pread spike on hardware whose device ceiling
actually exceeds what the pool drives — the engine ships only if the spike
prints a win. No numbers, no engine.

### 4. `kvb_nos3` slim build

The full-featured binary carries aws-sdk-go-v2 for the S3 tier (the static
gate was raised to 24 MB to admit it). Planned: an S3-less build behind a
`//go:build !kvb_nos3`-style seam, shipped as a second goreleaser artifact.
The full build stays the default and the headline — a single static binary
with all three tiers is the point; the slim one is for deployments that
want the smaller footprint. Ledgered in IMPROVEMENTS.md.

### 5. Benchmark completion

- **Chart 2 — TTFT vs hit rate** through a real vLLM + LMCache + kvblockd
  stack, with the honest production-trace hit-rate band. Blocked on GPU
  quota (the rig runbook is staged); the chart publishes with the shaded
  "recompute wins here" region either way.
- **100 GbE full-matrix re-run** of Chart 1: kvblockd measured 12.67 GB/s
  there, but the baseline bars are so far only measured on 50 GbE — the
  ≥10× headline is not quoted anywhere formal until the same-rig matrix
  exists (today's honest number: 7.1× at the best comparable 50 GbE cell,
  ceiling-limited).

### Under consideration (no commitment)

- AIBrix L2 connector — candidate next integration; not scheduled.
- NVMe/S3 quota *enforcement* (today: reporting-only) — waits on the
  reclaim-ordering design; trigger in IMPROVEMENTS.md.

## Non-goals (the founding rulings, restated)

These are load-bearing decisions, not gaps. They do not move in v1.x:

- **TCP-only. No RDMA, no AF_XDP, no DPDK.** MSG_ZEROCOPY, sendfile-class
  optimizations, and io_uring are in scope — they are TCP-path work. If you
  have InfiniBand/RoCE, use Mooncake; RDMA is a different league and we say
  so on every comparison we publish.
- **Opaque sealed blocks. The server never parses tensors.** Zero
  model-shape awareness, by design: engines own layout and semantics,
  kvblockd owns bytes, checksums, and tiers. No "smart" server-side tensor
  features, ever.
- **Open-core with the tech free forever.** Everything technical — all
  tiers, all protocol features, all benchmarks — is Apache-2.0 and stays
  that way. A commercial edition, if one exists, adds fleet-ops conveniences
  (SSO, audit, QoS dashboards, SLAs). Nothing moves from free to paid.
- **No HA/replication in v1.x.** Single-node, deliberately: this is a cache
  of recomputable data — losing a node costs latency, not data.

## How this list changes

Additions and re-orderings land as PRs to this file with their rationale;
deferrals get a row and a revisit trigger in IMPROVEMENTS.md rather than
silent deletion. If a published number here stops matching the repo's
measured evidence, the number is wrong and gets fixed — file an issue.
