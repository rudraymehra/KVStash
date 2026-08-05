# kvblockd (Python client)

Synchronous Python client for
[kvblockd](https://github.com/rudraymehra/KVStash) — the single-binary
LLM KV-cache store (DRAM → NVMe → S3 over plain TCP, prefix-hash keyed
write-once blocks, per-block xxh3 verification, namespace tenancy).

The daemon itself is one Go binary — one-line install in the repo README.
This package is the wire client the vLLM/LMCache/SGLang connectors build on:
zero-copy batched GET (`batch_get_scatter`), batched EXISTS, streamed PUT,
lease/pin/TTL verbs.
