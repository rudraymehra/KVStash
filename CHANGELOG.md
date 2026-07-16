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
