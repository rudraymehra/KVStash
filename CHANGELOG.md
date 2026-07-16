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
