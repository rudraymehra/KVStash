# vllm-kvblockd

Native vLLM KVConnector-v1 for [kvblockd](https://github.com/rudraymehra/KVStash) —
the single-binary LLM KV-cache store (DRAM → NVMe → S3 over plain TCP).
This connector produced every published TTFT number in the repo
(receipts: `docs/BENCHMARKS.md`, `docs/CLAIMS.md`).

Quick start: `docs/INTEGRATIONS.md` → "vLLM native connector".
TP=1 only (refuses multi-GPU at boot); requires `PYTHONHASHSEED=0`.
