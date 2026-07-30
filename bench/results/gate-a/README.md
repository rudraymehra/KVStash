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

## PR-10 session step 0 — MEASURED (2026-07-31, before any GPU spend)

Run on the session's store node (r6in.2xlarge, 64 GiB, Ubuntu 22.04, python
3.10, kvblockd binary from commit 406a336; script at b0c0774 — see the OOM
note) with the session's main daemon stopped for the duration:

- `gate-a-blocks2048-r6in2xl-linux.txt` — ratio **2.25×** → byte-bound.
- `gate-a-blocks16384-r6in2xl-linux.txt` — the 262k arm's exact block count:
  ratio **2.95×** → **byte-bound**. Decision rule frozen in PR-10 (< 1.70
  halts the session) does not fire; GPU provisioning authorized.
- `oom-kill-first-16384-attempt-dmesg.txt` — the first 16384 attempt (script
  at 406a336) held every reloaded value (~17 GB client RSS) and the kernel
  OOM killer took the daemon mid-read. The fix (b0c0774) streams the reload
  in 512-key batches and discards per chunk — identical in both arms, ratio
  semantics unchanged, and bounded batches are the connector's real access
  pattern. The crash evidence is banked because instrument failures are
  data too.

Ratios above 2.0 are super-linear: the big arm's working set spills further
past cache tiers than the small arm's at equal block count. The pre-registered
verdict is the threshold comparison, not the magnitude.

History note: an earlier session quoted "t ~ bytes^0.98" from an uncommitted
run. No artifact of that run exists, the script never computed an exponent,
and the figure is withdrawn — these banked files are the evidence of record.
