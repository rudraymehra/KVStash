# SGLang HiCache backend — verdict: DEFER (v1.1 revisit doc)

**Date:** 2026-07-20 · **Decision rule:** week-12 Day-4 "SHIP or
DEFER-with-blocker, no limbo" · **Branch taken:** DEFER.

## Verdict

`sglang-kvblockd` is **built and CPU-validated but not "supported"**: the
package, keymap, CPU unit suite, and CI tripwire leg merge now (the gate
requires them on BOTH branches); the PyPI publish and the
`docs/INTEGRATIONS.md` section wait for a green GPU e2e (C-4 makes the
publish unconditional on the SHIP branch only).

## The exact blocker

**No GPU budget this sprint.** The SHIP gate is pre-registered as: e2e green
on a Linux x86 RTX 4090 box — token-identical multi-turn output vs a no-L3
baseline, plus remote PUT-then-GET hits visible in kvblockd `/metrics`
(week-12 DoD). No GPU session ran, so no validated x86 run exists; per
week-12 risk #5 that reads as an honest DEFER, and per C-32 the SGLang spike
is the week's *acceptable-DEFER* item (the revenue work is not). Not a
technical failure: `bench/e2e/sglang/NOTES.md` contains zero failure lines
because nothing was attempted on GPU.

**Secondary (structural, pre-registered):** the HiCache **v2** controller
surface (`PoolTransfer`/`CacheControllerV2`, `batch_*_v2`) is still churning
upstream — [sgl-project/sglang#18239](https://github.com/sgl-project/sglang/issues/18239).
Ruling: do not build against v2; the backend stubs it (below) and targets
the **v1** interface only.

## What exists and is validated (CPU, no sglang required)

- `python/sglang_kvblockd/` — `KvblockdHiCacheStorage` (HiCacheStorage v1:
  `get/batch_get/set/batch_set/exists`, consecutive-prefix `batch_exists`,
  zero-copy `batch_get_v1`/`batch_set_v1` via whole-pool ctypes registration
  + `get_page_buffer_meta` offset translation, `clear`, `get_stats`) and
  `keymap.py` (rank-suffixed keys, model/tp/pp fingerprint folded into every
  32-byte BLAKE3 wire key; golden vectors in `tests/golden/`).
- 23-test CPU suite green against a live daemon (byte-exact zero-copy
  round-trips, consecutive-prefix semantics incl. mid-prefetch partials,
  128-key `batch_exists` p99 < 1 ms loopback, never-raise on dead daemon,
  MLA single-object + cross-TP sharing, fingerprint isolation, size-mismatch
  → miss, layout rejection, stats drain, v2 stubs).
- `.github/workflows/e2e-cpu.yml` `sglang-cpu` job — the A6-style tripwire:
  shim import-check → CPU suite → pinned `sglang==0.5.15.post1` install →
  real-ABC subclass check → suite re-run.

## What is NOT validated (the DEFER surface)

- Any run against a real SGLang server / real `HostKVCache` pools (our pool
  fakes port the upstream `page_first` `get_page_buffer_meta` math verbatim,
  but the real thing never executed against us).
- The dynamic-loader handshake in anger: `backend_class(storage_config,
  kwargs)` (positional dict) and `extra_config["interface_v1"]=1` routing —
  code-read at the `v0.5.15.post1` tag, never executed.
- MLA/pp>1/cp>1 on real models; head-split (`should_split_heads`) is
  explicitly unsupported (fingerprint-tagged so it cannot cross-hit).
- Token-identity, prefetch/backup rates, and any performance number.

## Stubs (raise `NotImplementedError` naming #18239)

`batch_exists_v2`, `batch_get_v2`, `batch_set_v2`. The registration hook
`register_mem_host_pool_v2` accepts-and-stores (so speculative-decode init
doesn't explode) but no v2 data path exists.

## Revisit trigger (whichever fires first)

1. **GPU budget line reopens** (wk-13 buffer or a later sprint): get the
   tree onto the pod first — the repo is PRIVATE, so a deploy-key/token
   `git clone` or an rsync from the laptop; there are no fetchable release
   artifacts. Then run the already-written rig —
   `bench/e2e/sglang/runpod_up.sh` (installs the pinned Go toolchain and
   builds the daemon from that tree) → `run_multiturn.sh` (baseline +
   kvblockd) → `runpod_down.sh`; ~2 sessions × 5h × $0.69/hr ≈ $7. Green ⇒
   execute the SHIP branch (tag v0.1.0, publish, INTEGRATIONS.md section,
   CHANGELOG).
2. **Upstream stabilizes v2:** #18239 resolved and `PoolTransfer` shipping
   unchanged in a tagged release for one minor cycle ⇒ re-pin, implement
   `batch_*_v2` on the frozen shape, and re-verdict.

Before any rerun: re-verify `bench/e2e/sglang/versions.env` pins against
PyPI and re-diff `hicache_storage.py` / `backend_factory.py` /
`cache_controller.py` against the notes above (the `sglang-cpu` CI job
catches subclass/import drift automatically).

## Launch command under test (for the rerun)

```
python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct \
  --enable-hierarchical-cache --hicache-ratio 2 --page-size 32 \
  --hicache-storage-backend dynamic \
  --hicache-storage-backend-extra-config '{"backend_name":"kvblockd",
    "module_path":"sglang_kvblockd.backend","class_name":"KvblockdHiCacheStorage",
    "interface_v1":1,"endpoint":"kvblockd://127.0.0.1:9400",
    "namespace":"sglang-e2e","token":"..."}'
```
