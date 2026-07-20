# SGLang HiCache e2e — session notes

**STATUS: NOT RUN.** No GPU session happened this sprint (no GPU budget —
the wk-12 verdict is DEFER, see `docs/design/sglang-hicache-v1.1.md`). The
scripts here are written and code-read against sglang `v0.5.15.post1` but
have never executed on a real box. Treat every pin and flag as unverified
until the first session log lands below.

## How to use this file

Week-12 Day-3 rule: log every failure **verbatim** (command, full stderr,
stack trace, sglang log excerpt) — these lines are the raw material for the
blocker doc. No paraphrasing.

## Pre-registered failure suspects (from the week file, check in this order)

1. **Page-layout offset math** in `batch_get_v1` — our ptr→offset
   translation vs the real `MHATokenToKVPoolHost.get_page_buffer_meta`
   (only the CPU fake, a verbatim port of the page_first branch, is tested).
2. **MLA vs MHA key-suffix handling** — real MLA pools may emit multi-buffer
   (Sequence) pointers under layer_first; our backend degrades those to
   misses by design. Confirm the pool is page_first on the rig.
3. **`batch_exists` called mid-prefetch** with keys the controller hasn't
   backed up yet — must return the partial prefix count, not error
   (CPU-tested; confirm under real concurrency).
4. **Dynamic-loader handshake** — `backend_class(storage_config, kwargs)`
   with kwargs as a POSITIONAL dict, and `extra_config["interface_v1"]=1`
   required for the controller to route batch_get_v1/set_v1 (code-read at
   v0.5.15.post1 `backend_factory.py` / `cache_controller.py`; never run).
5. **Pinned-pool registration** — `kv_buffer` must be CPU + contiguous;
   hybrid/anchor pools (`kv_buffer=None`) disable v1 paths.
6. `sglang[all]` install vs the pod's CUDA/driver (wk-12 risk #5: fall back
   to the community tier or defer to the wk-13 buffer).

## Session log (verbatim; newest first)

_(empty — no sessions yet)_

## Run recipe

```
bench/e2e/sglang/runpod_up.sh                    # on the pod
BASELINE=1 bench/e2e/sglang/run_multiturn.sh     # no-L3 token-identity ref
BASELINE=0 bench/e2e/sglang/run_multiturn.sh     # kvblockd L3 run
bench/e2e/sglang/runpod_down.sh                  # ALWAYS, then $0 check
```
