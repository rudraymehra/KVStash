# Gate A artifacts — is the client reload path byte-bound or block-bound?

Script: `bench/e2e/gate_a_bytes_vs_blocks.py` (pre-registered thresholds in the
script: ratio ≥ 1.70 → byte-bound, ≤ 1.30 → block-bound, for a 2× byte ratio
at equal block count).

## gate-a-blocks1024-darwin-arm64.txt

- Host: the development Mac (Apple Silicon, **8 GiB RAM** — stated because it
  is the caveat), macOS, loopback, one local kvblockd, xxh3 verify ON.
- Command: `BLOCKS=1024 REPS=3 python bench/e2e/gate_a_bytes_vs_blocks.py`
- Result: ratio **3.32×** for 2× bytes → **byte-bound** per the pre-registered
  threshold. The magnitude is NOT quotable as a scaling law: 3.32 > 2 is
  super-linear, consistent with memory-tier pressure on an 8 GiB host whose
  big arm alone allocates ~3.4 GiB (arena + payloads). The verdict
  (≥ 1.70) is the only thing this artifact supports.
- Scope, from the script's own verdict string: the CLIENT path only (recv +
  xxh3 + memcpy). It says nothing about the GPU scatter stage.

## Pre-registered follow-up (PR-10 session step 0)

Run on the session's store node (r6in.2xlarge, 64 GiB, Linux — the CPU class
the store actually ships on) at `BLOCKS=2048` and `BLOCKS=16384` (the exact
block count of the 262k arm), banked here before any GPU spend. Decision rule,
frozen in PR-10: a ratio < 1.70 at BLOCKS=16384 kills the fp8 real-NIC
session's premise and the GPU node is not provisioned.

History note: an earlier session quoted "t ~ bytes^0.98" from an uncommitted
run. No artifact of that run exists, the script never computed an exponent,
and the figure is withdrawn — these banked files are the evidence of record.
