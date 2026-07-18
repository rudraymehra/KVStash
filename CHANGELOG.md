# Changelog

All notable changes to kvblockd will be documented in this file.
Format: Keep a Changelog (https://keepachangelog.com), SemVer after v0.1.0.

## [Unreleased]
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
