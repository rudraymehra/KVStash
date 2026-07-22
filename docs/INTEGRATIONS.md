# Integrations

## LMCache (vLLM) — kvblockd as a remote KV-cache backend

kvblockd plugs into [LMCache](https://github.com/LMCache/LMCache) as a
`RemoteConnector` via the `kvblockd://` scheme, so any vLLM deployment using
LMCache can offload prefix-cached KV blocks to a kvblockd daemon (DRAM →
NVMe → S3 tiering happens behind the wire verbs; the adapter and LMCache
never see it — opaque blocks).

### Install

```bash
pip install kvblockd lmcache-kvblockd    # from PyPI once published; until then, pip install ./python/kvblockd ./python/lmcache_kvblockd
```

### Configure

Point LMCache at a running kvblockd daemon. `lmcache.yaml`:

```yaml
chunk_size: 256
local_cpu: true
remote_url: "kvblockd://HOST:9440?namespace=lmcache&streams=4"
remote_storage_plugins: ["kvblockd"]
extra_config:
  kvblockd_token: "YOUR_TOKEN"                              # or env KVBLOCKD_TOKEN
  remote_storage_plugin.kvblockd.module_path: "lmcache_kvblockd.adapter"
  remote_storage_plugin.kvblockd.class_name: "KvblockdConnectorAdapter"
```

vLLM `--kv-transfer-config`:

```json
{"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both",
 "kv_connector_extra_config": {"lmcache_config_file": "lmcache.yaml"}}
```

> **⚠ PYTHONHASHSEED must be pinned identically on every worker.** LMCache's
> chunk-hash chain seeds from vLLM's `NONE_HASH`, which depends on
> `PYTHONHASHSEED` when the builtin hash is in play. If it differs between
> workers, two instances derive DIFFERENT keys for the same tokens and never
> share cache. Set `PYTHONHASHSEED=0` everywhere. The connector's `post_init`
> runs a determinism check and logs loudly if it's unpinned.

### Engine support matrix

Tracked by the `interface-tripwire` workflow (weekly + on demand); a rename
in either upstream turns the run red before it can reach production.

| LMCache | vLLM (import/instantiate) |
|---|---|
| **0.5.1** (pip-installable; the package pins `lmcache>=0.5.1,<0.6`) | 0.25.1, 0.24.0, 0.23.0, 0.22.1 |
| 0.5.0, 0.4.7 (interface tripwire only, via `--no-deps`) | — |

The `interface-tripwire` workflow imports the adapter against the older
LMCache releases with `--no-deps` to catch method renames early; only 0.5.1
satisfies the runtime dependency pin. The e2e (`e2e-cpu.yml`) exercises the
full stack on `facebook/opt-125m` (CPU) at the pinned
`bench/e2e/cpu/versions.env` (lmcache 0.5.1, vllm 0.25.1).

### How it behaves

- **Never raises.** A daemon that is down, slow, or killed mid-request
  surfaces as a cache miss (`None`/`0`/empty), never an exception — LMCache
  treats an exception or hang as fatal to the serving engine, a miss as
  routine. The connector's op timeout (10 s) sits below LMCache's
  `blocking_timeout_secs` so a stall becomes a miss, not an engine stall.
- **Zero-copy reads.** Blocks are stored as a 32-byte metadata prefix
  (format/dtype/shape) plus the tensor bytes; on GET the connector allocates
  the return MemoryObj from LMCache's pinned pool and the tensor bytes land
  in it directly.
- **`batched_contains` is a consecutive-prefix count**, mapped 1:1 to the
  daemon's BATCH_EXISTS `n_consecutive` — hit,hit,miss,hit answers 2, which
  is exactly what the vLLM scheduler wants.

### Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `DeterminismError` at startup | `PYTHONHASHSEED` unset/`random` — export `PYTHONHASHSEED=0` on every worker |
| No remote hits after round 2 | daemon unreachable (check `curl HOST:9442/healthz`), or a token/namespace mismatch → every op is silently a miss |
| `connection refused` in logs, serving still works | expected during a daemon restart — the connector re-dials lazily; hits resume once it's back |
| vLLM won't build on macOS arm64 | known upstream flakiness — the CI gate (ubuntu) is authoritative; on Mac, run the connector unit suite (`pytest python/lmcache_kvblockd/tests`) which exercises every line of the adapter against a real daemon without vLLM |

## Follow-on connectors (status: on `main`, validation-gated)

The strategy is fixed: kvblockd is reached through the connectors people
already run, in this order. All three follow-on connectors now have real
code merged on `main` — but merged is not GA: each row states what is
validated and what is still gated, and nothing is called supported until
its pre-registered gate is green.

| Connector | Status | Path |
|---|---|---|
| LMCache → vLLM | **shipped** (above) | `python/lmcache_kvblockd/` |
| vLLM native connector | **on `main`** — code-complete, CPU-validated in CI; GPU e2e deferred | `python/vllm_kvblockd/` |
| NIXL | **beta** (native plugin); zero-code today via the S3-compat endpoint | `adapters/nixl/` + `internal/server/s3compat.go` |
| SGLang HiCache backend | **on `main`** — CPU-validated; verdict **DEFER** until a GPU run | `python/sglang_kvblockd/` |

Per-connector honesty notes:

- **vLLM native** (`vllm-kvblockd`): a native KVConnector-v1
  (`KvblockdConnector`) plus the `KvblockdTierManager` offloading altitude.
  The connector runs end-to-end on the vLLM CPU backend in CI
  (`.github/workflows/vllm-native-cpu.yml`); the tier manager is
  code-complete and unit-tested against a real daemon, but its GPU
  end-to-end is deferred, not faked — the exact revisit trigger and pass
  criteria live in `python/vllm_kvblockd/DEFER.md`. Not on PyPI yet.
- **NIXL**: two paths. The zero-code default is the S3-compatibility
  endpoint (`s3compat_addr`, off unless configured) — NIXL's stock `obj`
  plugin (and vLLM's `obj` tier) reach kvblockd via `endpoint_override`
  with no plugin code (`internal/server/s3compat.go`). The native C++
  plugin (`libplugin_KVBLOCKD.so`) is the performance path: **beta**,
  CI-tracked (`.github/workflows/nixl.yml`), not GA — caveats in
  `adapters/nixl/README.md`.
- **SGLang** (`sglang-kvblockd`): a HiCacheStorage **v1** backend,
  CPU-validated (23-test suite against a live daemon, plus the
  `sglang-cpu` tripwire job in `e2e-cpu.yml`) — **not GPU-validated and
  not on PyPI**; the pre-registered SHIP gate and its blocker are in
  [docs/design/sglang-hicache-v1.1.md](design/sglang-hicache-v1.1.md). The
  HiCache **v2** controller methods are stubbed pending upstream
  stabilization
  ([sgl-project/sglang#18239](https://github.com/sgl-project/sglang/issues/18239)).

Version-compatibility policy: each shipped connector pins the upstream
releases it is tested against (the support matrix above); when an upstream
release breaks the interface, the tripwire workflow goes red, the matrix in
this file states the last supported pin, and the fix lands as a patch
release — the answer to "does it work with X?" is always this table, never
a guess.
