# Changelog

All notable changes to kvblockd will be documented in this file.
Format: Keep a Changelog (https://keepachangelog.com), SemVer after v0.1.0.

## [Unreleased]

> In flight, pending merge (not yet on main): **lazy whole-segment restore**
> (repeated cold hits pull the whole segment back to NVMe and flip its
> entries home) + **S3 object GC** (delete a retired segment's object once
> its last s3-resident entry is gone). Tracked with revisit triggers in
> `docs/IMPROVEMENTS.md`.

### Added
- `python/vllm_kvblockd`: native vLLM integration — a KVConnector-v1
  connector with no LMCache in the path, plus `KvblockdTierManager` for
  vLLM's offloading/tiering altitude. Model-fingerprinted keys pinned by
  golden vectors; the pinned upstream contract carried in `UPSTREAM.lock`;
  unit suites run against a real daemon. The tier-manager GPU e2e is
  deferred until a GPU session is funded (`DEFER.md` in-package — the
  announcement must not outrun the evidence); the heavy live-serve CI leg
  (`vllm-native-cpu`) is dispatch-gated and ledgered in
  `docs/IMPROVEMENTS.md`.
- `python/sglang_kvblockd`: SGLang HiCacheStorage v1 backend (zero-copy
  `batch_get_v1`/`batch_set_v1` into the pinned host pool, consecutive-prefix
  `batch_exists`, rank/model-fingerprinted keys with golden vectors) + CPU
  unit suite + `sglang-cpu` CI tripwire leg. Spike verdict: **DEFER** — CPU-
  validated, not GPU-validated; not published to PyPI. Blocker + revisit
  trigger in `docs/design/sglang-hicache-v1.1.md`; v2 controller methods
  stubbed pending sgl-project/sglang#18239.
- `adapters/nixl`: native C++ NIXL storage-backend plugin (**beta**) —
  `libplugin_KVBLOCKD.so` discovered via `NIXL_PLUGIN_DIR`; WRITE maps to
  `PUT_STREAM BEGIN→CHUNK→COMMIT`, READ to `BATCH_GET` tiled across a
  connection pool with per-block xxh3 verification. The NIXL-free C++
  client core (KVB1 codec, HELLO auth, all 8 verbs, credit ledger)
  byte-compares against the Go/Python golden vectors; meson build +
  `nixl.yml` CI leg. The S3-compat endpoint is the zero-code default path
  for NIXL; this plugin is the performance path — CI-tracked, not GA.
- S3-compatibility endpoint: `s3compat_addr` (default off) serves a minimal
  S3 REST subset — PutObject, GetObject with Range, HeadObject — on its own
  listener, so NIXL's `obj` plugin and vLLM's `obj` tier reach kvblockd with
  zero plugin code via `endpoint_override`. Bucket = namespace against the
  same tenant registry and constant-time token compare as HELLO (unknown
  bucket and wrong token collapse to one 403 — no enumeration oracle);
  accepts Bearer and SigV4-shaped auth (the access-key id is the token; the
  signature is deliberately not verified — a documented divergence); the
  object key is the 64-hex encoding of the 32-byte block key. A
  compatibility surface, not the performance path: every GET copies to the
  heap before `net/http`.

## [0.1.0-rc1] - 2026-07-18

_Pre-release: the foundation. Everything below first shipped in this tag
and is included in v0.2.0._

### Added
- Repository scaffold: module, license, CI, directory structure.
- `docs/PROTOCOL.md`: frozen wire protocol v1 (KVB1) — 64-byte header, 8 batch
  verb families, two-phase PUT_STREAM, credit backpressure, 23 status codes.
- `internal/protocol`: zero-alloc header + body codecs (CRC32C-protected
  header, xxh3_64 descriptors), golden vectors, and fuzz targets.
- `internal/config`: YAML+env+flag daemon configuration with floor validation.
- `internal/transport`: framed connection loops, credit ledger with byte
  conservation, buffer lending, and coalesced writev responses.
- CI: weekly long-fuzz workflow; `fuzz-short`, `kvbdebug`, and `go mod tidy`
  checks in the PR pipeline.
- Tooling: `make mutate` mutation-score gate; repo-local ruleguard rules.
- `internal/store/ramstub`: temporary sharded in-heap block store (write-once,
  namespace-scoped) implementing the server's Store surface.
- `internal/server`: connection sessions with HELLO-first auth, per-verb
  dispatch, and the two-phase PUT_STREAM state machine + inactivity reaper;
  graceful Drain.
- `pkg/client`: reference Go client — Dial/HELLO, pooled connections, and the
  batch verbs with streaming `recv_into` for GET and client-side xxh3 verify.
- `cmd/kvblockd`: daemon bring-up (config → ramstub → server), end to end.
- `bench/microbench/rawget`: raw-socket loopback baseline for the GET-shaped
  request→response path — the fair same-shape ceiling the throughput gate is
  quoted against (%-of-ceiling methodology).
- `bench/kvbench/getbench`: out-of-process BATCH_GET load generator (daemon
  and load in separate processes — the production shape).
- `pkg/client`: `Options.SkipVerify` (default off) — disables the client-side
  xxh3 pass for consumers that re-verify downstream; used to decompose
  verification cost in the gate benchmark.
- Transport: startup tripwire logging if the connection is ever not a bare
  `*net.TCPConn` (golang/go#21676 — a wrapped conn silently loses the writev
  fast path, one Write syscall per buffer).
- `internal/store/dram`: the real DRAM tier — GC-invisible mmap arena
  (optional hugepages), OffsetAllocator port, sharded index, lease/pin/TTL
  ladder, refcounted zero-copy reads, per-namespace pinned-bytes accounting.
- `internal/eviction`: S3-FIFO (+ sampled-LRU) behind a pluggable Policy
  interface; watermark evictor; scrape-time eviction counters.
- `internal/store/modeltest`: model-based correctness harness (rapid) — the
  permanent oracle every tier must pass, incl. crash/recovery + resurrection
  semantics.
- `python/kvblockd` + `python/lmcache_kvblockd`: the Python wire client and
  the LMCache RemoteConnector backend (Go↔Python key-hash parity pinned by
  golden vectors); no-GPU CPU e2e recipe + CI.
- `internal/store/nvme` + tiered orchestrator: log-structured NVMe warm tier
  (256 MiB sealed segments, group-commit fdatasync ledger, checkpointed
  crash recovery), DRAM→NVMe demotion / second-hit promotion, and the
  kill -9 torture harness (`test/crash/`) enforcing the crash contract.
- `bench/kvbench`: the benchmark harness — coordinated-omission-safe
  open-loop scheduler, HDR histograms, four target drivers, deterministic
  payloads, `.kvops` trace converters (count-exact), adaptive replay where
  hit rate is an output, and a one-command loopback acceptance gate.
- Release engineering: `--version` stamping, goreleaser v2 static matrix
  (linux amd64/arm64 + darwin arm64), `test/release/assert_static.sh`
  (statically linked + <20 MB + scratch-boot gates), SBOMs, `FROM scratch`
  Docker image, `deploy/kvblockd.service`, `scripts/install.sh`, and the
  tag-driven release workflow.

### Changed
- Transport writes are windowed at `write_chunk_bytes` (~1 MiB) per writev
  syscall within a coalesced flush — giant single copies stall the loopback
  pipe; windows keep the kernel copy pipeline overlapped.
- Client xxh3 verification overlaps the socket reads (sidecar goroutine)
  instead of serializing read-then-hash per block.
- Client `Options` gained `SockSndBuf`/`SockRcvBuf` socket-buffer requests.

### Fixed (review ladder on the wire path)
- Client: non-OK BATCH_GET responses no longer deadlock the reader (preamble
  is inspected before descriptors are awaited); F_MORE-split responses are
  reassembled; a desynchronized connection is evicted from the pool and
  replaced instead of being handed to the next caller.
- Server: BATCH_GET responses larger than the negotiated `max_frame_len` are
  split into F_MORE frames; PUT staging is bounded per connection (live-stream
  cap + staged-byte cap, lazy allocation — BEGIN reserves nothing); zero-length
  CHUNKs no longer reset the inactivity timer; tombstoned streams are reaped
  after a grace period; HELLO rejections answer as HELLO responses with
  F_FATAL; Drain no longer races new connections past its snapshot.

## [0.2.0] - 2026-07-20

### Added
- Multi-tenant byte quotas: per-(namespace, tier) accounting with CAS
  admission — charged at publish, refunded at every removal seam (delete,
  eviction, lost publish race, reclaim retire, corrupt-entry self-heal),
  transferred on tier moves, re-seeded from recovery so a restart never mints
  free bytes. PUT BEGIN answers `ERR_QUOTA_BYTES` from an advisory probe; the
  binding charge is commit-time. No wire change. Over-quota tenants are
  evicted first under shared pressure; cross-tenant probes answer
  `NOT_FOUND`, never `FORBIDDEN` (no existence oracle).
- Namespace registry with hashed bearer tokens: `namespaces.yaml` carries
  `token_sha256` (plaintext `token` accepted and hashed at load, never
  retained); constant-time authentication with uniform timing for unknown
  names; per-tier quotas + pin quota per namespace.
- Admin socket: loopback-only HTTP surface (`admin_addr`) — `POST
  /v1/namespace`, `POST /v1/quota`, `GET /v1/namespaces` (tokens never
  listed); mutations are runtime-only and say so (persist via the namespaces
  file).
- `kvbctl` admin verbs: `namespace add|list` and `quota set` against the
  admin socket, alongside the existing data-plane verbs.
- Per-tenant metrics: `kvb_tenant_bytes` + `kvb_tenant_quota_bytes` labeled
  `{namespace, tier}` — cardinality #namespaces × 3 tiers, pinned by test.
- S3 cold tier: one sealed NVMe segment = one whole S3 object (request-cost
  coalescing), spilled on a bounded async write-back queue — a copy, reads
  stay local; reclaim of a spilled segment retire-flips its entries to
  s3-resident (index entries survive, the tenant charge moves NVMe→S3); cold
  GETs are single ranged GetObject reads, xxh3-verified before a byte
  escapes; EXISTS never touches S3; restart re-spills idempotently (spilled
  state is memory-only; s3-only entries are honestly lost at restart).
  Config keys: `s3_bucket` (tier inert until set), `s3_region`, `s3_node_id`,
  `s3_endpoint_override`, `s3_path_style`, `s3_spill_queue`,
  `s3_read_timeout_ms`; credentials come from the ambient AWS chain only —
  never from kvblockd config.
- S3 observability: an `s3` sub-document in STATS (resident blocks/bytes,
  spill/drop/put-error, ranged-get/restore, hit/read-error counters) and the
  `kvb_s3_*` scrape families; `kvb_blocks`/`kvb_store_bytes` gain tier="s3".

### Changed
- Release: the static-binary size gate (`test/release/assert_static.sh`) was
  raised 20 MB → 24 MB to admit the full-featured S3-tier binary
  (aws-sdk-go-v2 is the growth); a `kvb_nos3` slim build is ledgered as
  future work in `docs/IMPROVEMENTS.md`.

### Fixed
- Tier-exact refunds: removing an s3-resident (retire-flipped) entry refunds
  the tenant's S3 charge, not NVMe — the mismatch leaked the S3 charge and
  the underflow heal masked the NVMe double-subtract.
- Cold-tier hard-pin promotion: PIN_HARD (and 2nd-hit promotion) on an
  s3-resident block promotes through the verified cold-read path instead of
  answering `NOT_FOUND` for a block GET still serves.
- `nvme_admit_min_hits: 0` genuinely means admit-everything: an internal
  default clamp silently turned an explicit 0 into 1, and a pure-ingest fill
  was deleted at the demote watermark with no counter. Endurance-gate
  deletions are now counted (`kvb_nvme_admit_refusals_total`) and the
  operator default (1) lives visibly in the config layer.
