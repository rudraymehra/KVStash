# DEFERRED: GPU end-to-end for the SecondaryTierManager altitude

`tier_manager.py` (KvblockdTierManager under vLLM's `OffloadingConnector` +
`TieringOffloadingSpec`) is **code-complete and unit-tested** against the
pinned v0.25.0 contract (UPSTREAM.lock): synthetic primary memoryview,
hand-built `JobMetadata`, byte-exact round-trips, RETRY→HIT lookup
transitions, failure jobs, and `drain_jobs` semantics — all green with a real
kvblockd daemon on the other end of the wire.

What is NOT yet proven: the `OffloadingConnector` path end-to-end on a real
GPU. That path cannot run on the vLLM CPU backend (it is CUDA/ROCm/XPU only),
so the GPU e2e is deferred rather than faked. There is no silently-skipped
green: the CPU e2e gate is carried by `connector.py` alone
(`.github/workflows/vllm-native-cpu.yml`).

## Revisit trigger

Run the GPU validation session before any release or announcement that claims
the tier-manager (`OffloadingConnector`) path — the claim must not outrun the
evidence.

## Validation checklist for that session

- vLLM v0.25.x + `vllm-kvblockd` installed; kvblockd daemon co-located.
- `OffloadingConnector` + `TieringOffloadingSpec` with the `kvblockd`
  secondary tier; `kv_offload_benchmark.py`-style workload: 10k unique
  512-token requests, prefix caching off, Llama-3.1-8B (or a comparable
  model that fits the available GPU).
- PASS = non-zero secondary-tier hits, byte-correct outputs,
  `drain_jobs` clean shutdown under load, TTFT before/after captured.
- Result logged in `docs/INTEGRATIONS.md`.
