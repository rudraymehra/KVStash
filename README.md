# kvblockd

**A single static Go binary that serves LLM KV-cache blocks at 10+ GB/s over the plain TCP you already have.** Engines (vLLM via LMCache today; native vLLM, SGLang, and NIXL adapters on `main`, validation-gated) store prefix-keyed, write-once cache blocks here and load them back faster than the GPU can recompute them — tiered DRAM → NVMe → S3, multi-tenant, and honest about every number it publishes.

> **Status: all three tiers — DRAM, NVMe, S3 — ship on `main`. Latest published release: v0.2.0** (the S3 tier landed there; lazy cold-segment restore + cold-object GC completed on `main` since). The wire protocol is frozen at v1 and the transport numbers below are measured. The two headline charts (throughput matrix vs baselines; TTFT vs hit rate on a real vLLM stack) land here from the committed benchmark harness — raw JSONL first, pictures second.

## Why

- LLM APIs price cached input tokens at steep discounts (as low as ~2% of full price, per DeepSeek's published pricing as of mid-2026) — that's how large the recompute waste is.
- The serious KV-cache stores assume RDMA fabrics, storage appliances, or a managed cloud. Most GPU fleets run plain Ethernet.
- The proven design — immutable prefix-hash blocks, two-phase commit, lease/pin, batched MB-scale I/O — doesn't need exotic hardware. This is that design for everyone else.

## 60-second quickstart

```sh
curl -fsSL https://raw.githubusercontent.com/rudraymehra/KVStash/main/scripts/install.sh | sh
kvblockd --config /usr/local/etc/kvblockd/example.yaml &   # :9440, 1 GiB DRAM arena, demo tenant
echo hello | kvbctl put -ns demo -token demo-token demo-key -
kvbctl get -ns demo -token demo-token demo-key             # → hello
```

(The installer drops the example config + demo tenant at `/usr/local/etc/kvblockd/`; manual tarball users have the same files under `config/` in the archive.)

That's a DRAM-only daemon — the recommended first run. NVMe tiering, hugepages, systemd, and real tenants: [docs/deployment-guide.md](docs/deployment-guide.md).

## Measured, not promised

**Wire path:** kvblockd served batched GETs at **12.67 GB/s (101.4 Gbit/s) — ~102% of the iperf3 ceiling on a 100 GbE pair, with end-to-end xxh3 verification ON**. On 50 GbE it saturates the NIC the same way. Methodology, raw logs, and the one-command repro scripts: [bench/BENCHMARKS.md](bench/BENCHMARKS.md) · [docs/BENCHMARKS.md](docs/BENCHMARKS.md) · [bench/METHODOLOGY.md](bench/METHODOLOGY.md). Every chart we publish draws the transport ceiling on the chart — a bar that can't be compared to the wire's physical limit isn't honest.

**Durability:** the kill -9 torture harness SIGKILLs a live daemon mid-write-storm and holds recovery to the crash contract — every acknowledged commit either survives byte-identical or is honestly gone; never corrupt, never a phantom. **100 loops on Linux: 0 corrupt, 0 phantom over 18,160 journaled acks** ([docs/DESIGN.md](docs/DESIGN.md), `test/crash/`). Run it yourself: `go run -tags crashtest ./test/crash -loops 10`.

Chart 1 (GB/s vs Redis 7 / Valkey 8 / Mooncake-TCP / NVMe-fs floor, two client classes) and Chart 2 (TTFT vs hit rate through vLLM + LMCache + kvblockd, honest Bailian-trace hit-rate band) render from committed JSONL via `bench/report/plot.py` and are published with the rig sessions.

## How it compares

<!-- Every cell cites the upstream project's own docs/issues; links re-verified at publish time. -->

| | kvblockd | [LMCache](https://github.com/LMCache/LMCache) | [Mooncake](https://github.com/kvcache-ai/Mooncake) | PegaFlow | InfiniStore | Redis / Valkey |
|---|---|---|---|---|---|---|
| Standalone TCP data plane for MB blocks | **yes — the product** | no (Python lib inside the engine process; remote stores via connectors) | TCP mode exists; designed and tuned for RDMA ([Transfer Engine docs](https://kvcache-ai.github.io/Mooncake/)) | no | RDMA-first | protocol yes; string-store semantics, not MB-block zero-copy ([LMCache #2204](https://github.com/LMCache/LMCache/issues/2204)) |
| Tenancy + quotas | **shipped v0.2.0** (namespace identity is already structural at HELLO) | no ([#2878](https://github.com/LMCache/LMCache/issues/2878): `cache_salt` ignored on the remote path) | no per-tenant quotas ([#1035](https://github.com/kvcache-ai/Mooncake/issues/1035)) | no | no | ACLs, no cache-aware quotas |
| TTL / lease / pin ladder | **yes** ([PROTOCOL.md §6](docs/PROTOCOL.md)) | TTL only | no lease/pin | no | no | TTL only |
| S3 / object tier | **yes — shipped v0.2.0**, completed on `main` (see note) | via pluggable backends | no | no | no | no |
| Warm restart after kill -9 | **yes — measured** (100-loop torture, contract above) | n/a (in-process cache) | metadata lives in etcd | no (SSD index held in memory) | no | RDB/AOF replay; no block-level crash contract |
| Single static binary, no sidecars | **yes (~21 MB, S3 tier included)** | no (pip package + engine) | no (etcd + C++ toolchain) | no | no | server + client libraries |

**Honesty notes:** tenancy + quotas shipped in v0.2.0 — hashed-token namespace registry, per-(namespace, tier) byte accounting with CAS admission (`ERR_QUOTA_BYTES` at PUT, binding charge at commit), per-namespace pin quota, over-quota-first eviction, per-tenant metrics. Open scope, stated plainly: the DRAM quota is *enforced*; NVMe/S3 quotas are accounted and reported but not yet enforced (reclaim-ordering design pending — [docs/ROADMAP.md](docs/ROADMAP.md)); per-tenant QoS stays future commercial-edition scope. The S3 tier shipped in v0.2.0 (async spill; reclaim retire-flips entries s3-resident; cold GETs are single ranged reads, xxh3-verified before a byte escapes; EXISTS never touches S3) and is completed on `main` — repeated cold hits restore the whole segment back to NVMe and flip its entries home, and cold-object GC deletes a retired segment's object once its last s3-resident entry is gone. Competitor cells describe the linked docs/issues as of writing; if we got one wrong, file an issue and the table gets fixed.

## When NOT to use kvblockd

- **You have InfiniBand/RoCE:** use Mooncake — RDMA is a different league (35–270 GB/s) and we don't pretend to play in it.
- **Your working set fits in GPU HBM:** engine-native prefix caching is free; we add a hop.
- **Your inter-node network is <10 GbE:** below ~2 GB/s deliverable bandwidth, recompute usually wins; run `bench/kvbench` and believe your own numbers.
- **You need multi-node replication/HA today:** v1 is deliberately single-node — a cache of recomputable data, so losing a node costs latency, not data.
- **Your prompts never share prefixes** (fully unique, no system prompt, no multi-turn): nothing to cache.

## v1 cut-line

TCP only (MSG_ZEROCOPY/sendfile-class optimizations in scope; RDMA/AF_XDP/DPDK out). Reached through the connectors people already run — LMCache → vLLM today; on `main`: a native vLLM connector (code-complete, GPU e2e deferred), an SGLang HiCache backend (CPU-validated, verdict DEFER until a GPU run), a native NIXL C++ plugin (beta), and the S3-compat endpoint as the zero-code NIXL/`obj` path ([docs/INTEGRATIONS.md](docs/INTEGRATIONS.md)). Blocks are opaque sealed bytes — the server never parses tensors. No HA/replication in v1. Tenancy quotas and the S3 tier both shipped in v0.2.0; the cold tier is completed on `main`.

## Security model

Identity is structural: a connection authenticates a `(namespace, token)` pair once at HELLO (constant-time compare) and lives inside that namespace — no per-request auth to get wrong, and a cross-tenant key collision is impossible by construction. A daemon with **no namespaces file accepts no one** (secure by default). Transport is plaintext TCP in v1: deploy on a trusted network segment; TLS-termination guidance lives in the [deployment guide](docs/deployment-guide.md). Report vulnerabilities via GitHub security advisories.

## License

Apache-2.0 — free forever. Everything technical (all tiers, all protocol features, all benchmarks) is and stays Apache-2.0; a future commercial edition adds only fleet-ops conveniences (SSO, audit, QoS dashboards, SLAs) and never moves free things behind a paywall. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
