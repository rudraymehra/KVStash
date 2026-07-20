# DEFERRED: GPU end-to-end for the SecondaryTierManager altitude

`tier_manager.py` (KvblockdTierManager under vLLM's `OffloadingConnector` +
`TieringOffloadingSpec`) is **code-complete and unit-tested** against the
pinned v0.25.0 contract (UPSTREAM.lock): synthetic primary memoryview,
hand-built `JobMetadata`, byte-exact round-trips, RETRY→HIT lookup
transitions, failure jobs, and `drain_jobs` semantics — all green with a real
kvblockd daemon on the other end of the wire.

What is NOT yet proven: the `OffloadingConnector` path end-to-end on a real
GPU. It cannot run on the vLLM CPU backend (SPEC 5 §2.5 caveat — CUDA/ROCm/XPU
only), and this run had **no GPU budget**, so the GPU e2e is deferred rather
than faked. There is no silently-skipped green: the CPU e2e gate is carried by
`connector.py` alone (`.github/workflows/vllm-native-cpu.yml`).

## Revisit trigger (exact)

Rent ONE RunPod RTX 4090 secure instance (~$0.69/hr, ~2.5h wall, ≤$4 —
inside the Week-10 ≤$8 envelope) at the EARLIER of:

1. the Week-10 Day-5 slot, or
2. before any public announcement that names the tier-manager altitude
   (vLLM Slack/Discord/LinkedIn thread) — the announcement must not outrun
   the evidence.

## Validation checklist for that session (from week-10.md Day 5)

- vLLM v0.25.x + `vllm-kvblockd` installed; kvblockd daemon co-located.
- `OffloadingConnector` + `TieringOffloadingSpec` with the `kvblockd`
  secondary tier; `kv_offload_benchmark.py`-style workload: 10k unique
  512-token requests, prefix caching off, Llama-3.1-8B (fall back to a 1h
  A100 80GB ONLY if 4090 memory blocks the model — C-21 spirit).
- PASS = non-zero secondary-tier hits, byte-correct outputs,
  `drain_jobs` clean shutdown under load, TTFT before/after captured for the
  announcement thread.
- Result logged in `docs/INTEGRATIONS.md`; instance torn down same day,
  $0 residue verified.
