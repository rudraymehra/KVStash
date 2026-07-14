# kvblockd

**A single-binary block store for LLM KV-cache — batched multi-stream TCP, DRAM→NVMe→S3 tiering, real eviction/lease/pin semantics, and per-tenant quotas. No RDMA required. No etcd. One static Go binary.**

LLM inference engines throw away gigabytes of perfectly reusable KV-cache every minute because GPU memory is small and expensive. kvblockd is the shared locker for those blocks: engines (vLLM, SGLang — via LMCache and native connectors) store prefix-keyed, write-once cache blocks here and load them back faster than the GPU can recompute them, over the plain Ethernet your cloud already has.

> ⚠️ **Status: pre-v0.1. Do not use.** This repository is in its measurement-and-scaffolding phase; the daemon does not exist yet. Follow along: transport-ceiling numbers, design docs, and honest benchmarks land here as they're produced. First tagged release target: v0.1.0.

## Why this exists (the 30-second version)

- Major LLM APIs price cached input tokens at steep discounts (as low as ~2% of the full price, per DeepSeek's published pricing as of mid-2026) — the recompute waste is that large.
- Most serious KV-cache stores today assume RDMA fabrics, storage appliances, or a managed cloud — while a large share of GPU fleets run on plain Ethernet with no equivalent option.
- The proven design (immutable prefix-hash blocks, two-phase commit, lease/pin, batched MB-scale I/O) doesn't actually require any of that hardware. This is that design, implemented for everyone else.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
