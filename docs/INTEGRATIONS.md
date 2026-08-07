# Integrations

## LMCache (vLLM) — kvblockd as a remote KV-cache backend

kvblockd plugs into [LMCache](https://github.com/LMCache/LMCache) as a
`RemoteConnector` via the `kvblockd://` scheme, so any vLLM deployment using
LMCache can offload prefix-cached KV blocks to a kvblockd daemon (DRAM →
NVMe → S3 tiering happens behind the wire verbs; the adapter and LMCache
never see it — opaque blocks).

### Install

```bash
pip install lmcache-kvblockd    # pulls the kvblockd client automatically
```

> Linux x86_64 only (verified: resolves `lmcache==0.5.2` from its manylinux
> wheel). On macOS the `lmcache` dependency has no wheel and its source
> build drags CUDA-only packages — develop the adapter on a Mac via its
> unit suite (troubleshooting table below), run the real stack on Linux.

### Configure

Point LMCache at a running kvblockd daemon. `lmcache.yaml`:

```yaml
chunk_size: 256
local_cpu: true
remote_storage_plugins: ["kvblockd"]
extra_config:
  kvblockd_token: "YOUR_TOKEN"                              # or env KVBLOCKD_TOKEN
  remote_storage_plugin.kvblockd.module_path: "lmcache_kvblockd.adapter"
  remote_storage_plugin.kvblockd.class_name: "KvblockdConnectorAdapter"
  remote_storage_plugin.kvblockd.url: "kvblockd://HOST:9440?namespace=lmcache&streams=4"
```

> **The endpoint must be `extra_config["remote_storage_plugin.kvblockd.url"]`,
> not `remote_url`.** In lmcache 0.5.x a backend created via
> `remote_storage_plugins` dials the virtual URL `plugin://kvblockd` and never
> reads `remote_url`; adapter versions before this fix only matched
> `kvblockd://`, so the backend failed connector creation, LMCache swallowed
> the error, and every put/get silently became a local-tier miss (zero bytes
> reached kvblockd). Also do NOT set `remote_url` alongside the plugin: LMCache
> would create a second, deprecated RemoteBackend and double every put.
> `python/lmcache_kvblockd/tests/test_lmcache_registration.py` pins this
> behavior against the real lmcache package in CI.

vLLM `--kv-transfer-config` — configure LMCache through its own
`lmcache.`-prefixed keys, which is the only extra-config channel the
integration reads (anything else is discarded without warning):

```json
{"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both",
 "kv_connector_extra_config": {
   "lmcache.remote_storage_plugins": "kvblockd",
   "lmcache.local_cpu": true,
   "lmcache.chunk_size": 256,
   "lmcache.extra_config": {
     "kvblockd_token": "YOUR-TOKEN",
     "remote_storage_plugin.kvblockd.module_path": "lmcache_kvblockd.adapter",
     "remote_storage_plugin.kvblockd.class_name": "KvblockdConnectorAdapter",
     "remote_storage_plugin.kvblockd.url": "kvblockd://HOST:9440?namespace=lmcache&streams=4"
   }}}
```

> **⚠ There is no `lmcache_config_file` key.** Earlier revisions of this page
> pointed at a YAML file that way. That key is read by neither vLLM 0.25.1 nor
> LMCache 0.5.1 — searching both source trees finds no occurrence — so the file
> was never loaded and LMCache silently ran on defaults with a local tier only:
> no remote backend, and not one log line to say so. To use a YAML file
> instead, export `LMCACHE_CONFIG_FILE` in the environment of the process that
> serves the model (including inside the container, if you use one); do not mix
> the two sources, since `lmcache.extra_config` replaces a file-provided one.
>
> Whichever route you choose, confirm it worked: the engine logs
> `Created remote backend for plugin: kvblockd`, and the daemon's
> `kvb_bytes_total{dir="in"}` rises on the first prefill. Silence means the
> plugin branch never ran.

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
| **0.5.1, 0.5.2** (pip-installable; the package pins `lmcache>=0.5.1,<0.6` — a fresh Linux resolve picks 0.5.2) | 0.25.1, 0.24.0, 0.23.0, 0.22.1 |
| 0.5.0, 0.4.7 (interface tripwire only, via `--no-deps`) | — |

The `interface-tripwire` workflow imports the adapter against the older
LMCache releases with `--no-deps` to catch method renames early; 0.5.1 and
0.5.2 satisfy the runtime dependency pin (0.5.2 checked by the CPU e2e
recipe's version notes). The e2e (`e2e-cpu.yml`) exercises the full stack
on `facebook/opt-125m` (CPU) at the pinned `bench/e2e/cpu/versions.env`
(lmcache 0.5.1, vllm 0.25.1).

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

## vLLM native connector — the path that produced the published TTFT numbers

`KvblockdConnector` is a native vLLM KVConnector-v1: no LMCache in the
middle, the engine talks straight to the daemon. Every published TTFT
multiple (the A10G/L40S charts, the two-node real-NIC cells) was measured
through this connector; the receipts are in [BENCHMARKS.md](BENCHMARKS.md)
and [CLAIMS.md](CLAIMS.md).

### Install

```bash
pip install vllm-kvblockd    # pulls the kvblockd client automatically
```

(Installing from a clone still works: `pip install ./python/kvblockd
./python/vllm_kvblockd`.)

### Run

Daemon side — a namespace for the engine. The quickstart installer's
`namespaces.yaml` already contains an active `demo` tenant at `id: 1`,
and the daemon refuses duplicate ids at boot — so ADD a second line
(or replace the demo one), don't copy `id: 1`:

```yaml
namespaces:
  - { name: demo, id: 1, token: "demo-token" }   # installer's quickstart tenant
  - { name: vllm, id: 2, token: YOUR-TOKEN }
```

Engine side — vLLM 0.25.x, one flag:

```bash
export PYTHONHASHSEED=0   # REQUIRED on every worker — the connector
                          # refuses to boot unpinned (key derivation must
                          # be deterministic across engine restarts)
vllm serve MODEL \
  --kv-transfer-config '{
    "kv_connector": "KvblockdConnector", "kv_role": "kv_both",
    "kv_connector_module_path": "vllm_kvblockd.connector",
    "kv_connector_extra_config": {
      "kvblockd_endpoint": "kvblockd://127.0.0.1:9440",
      "kvblockd_namespace": "vllm",
      "kvblockd_token": "YOUR-TOKEN",
      "kvblockd_streams": 8,
      "kvblockd_get_fanout": 8}}'
```

(`streams=8, get_fanout=8` is the measured real-NIC config — the
published A/B selected it; xxh3 verification defaults ON and measured
free at 131k, leave it on.)

### Confirm it worked (silence is not success)

1. Engine log at boot: `Creating v1 connector with name: KvblockdConnector`
   — the same factory line our benchmark rig gates on. No line = vLLM
   dropped the transfer config and is serving with no connector at all.
2. First prefill: the daemon's `kvb_bytes_total{dir="in"}` rises
   (`curl HOST:9442/metrics`).
3. Restart the engine (not the daemon), replay the same prompt:
   `kvb_hits_total` rises and TTFT drops — that delta is the product.

### Limits, stated plainly

- **TP=1 only**: the connector refuses multi-GPU at boot (block keys do
  not carry rank identity yet; refusing beats silently cross-loading KV
  shards between ranks).
- **`DeterminismError` at startup** = `PYTHONHASHSEED` unset — export
  `PYTHONHASHSEED=0` on every worker, same as the LMCache path above.
- Failure posture is fail-open: a down/slow daemon degrades to cache
  misses (recompute), never an engine exception.
- The connector (`KvblockdConnector`) is **GPU-validated**. The separate
  `KvblockdTierManager` offloading altitude is code-complete but its GPU
  end-to-end remains deferred, not faked — trigger and pass criteria in
  `python/vllm_kvblockd/DEFER.md`.

## Follow-on connectors (status: on `main`, validation-gated)

The strategy is fixed: kvblockd is reached through the connectors people
already run, in this order. All three follow-on connectors now have real
code merged on `main` — but merged is not GA: each row states what is
validated and what is still gated, and nothing is called supported until
its pre-registered gate is green.

| Connector | Status | Path |
|---|---|---|
| LMCache → vLLM | **shipped** (above) | `python/lmcache_kvblockd/` |
| vLLM native connector | **GPU-validated** (it produced the published TTFT numbers — section above); the TierManager altitude's GPU e2e stays deferred | `python/vllm_kvblockd/` |
| NIXL | **beta** (native plugin); zero-code today via the S3-compat endpoint | `adapters/nixl/` + `internal/server/s3compat.go` |
| SGLang HiCache backend | **on `main`** — CPU-validated; verdict **DEFER** until a GPU run | `python/sglang_kvblockd/` |

Per-connector honesty notes:

- **vLLM native** (`vllm-kvblockd`): a native KVConnector-v1
  (`KvblockdConnector`) plus the `KvblockdTierManager` offloading altitude.
  The connector is **GPU-validated** — every published TTFT cell
  (A10G, L40S, the two-node real-NIC sessions) was measured through it,
  receipts in [BENCHMARKS.md](BENCHMARKS.md)/[CLAIMS.md](CLAIMS.md) —
  and it also runs end-to-end on the vLLM CPU backend in CI
  (`.github/workflows/vllm-native-cpu.yml`). The tier manager is
  code-complete and unit-tested against a real daemon, but its GPU
  end-to-end is deferred, not faked — the exact revisit trigger and pass
  criteria live in `python/vllm_kvblockd/DEFER.md`. On PyPI as
  `vllm-kvblockd`.
- **NIXL**: two paths. The zero-code default is the S3-compatibility
  endpoint (`s3compat_addr`, off unless configured) — NIXL's stock `obj`
  plugin (and vLLM's `obj` tier) reach kvblockd via `endpoint_override`
  with no plugin code (`internal/server/s3compat.go`). The native C++
  plugin (`libplugin_KVBLOCKD.so`) is the performance path: **beta**,
  CI-tracked (`.github/workflows/nixl.yml`), not GA — caveats in
  `adapters/nixl/README.md`.
- **SGLang** (`sglang-kvblockd`): a HiCacheStorage **v1** backend,
  CPU-validated (23-test suite against a live daemon, plus the
  `sglang-cpu` tripwire job in `e2e-cpu.yml`) — **not GPU-validated**
  (on PyPI as `sglang-kvblockd`, published with that caveat stated
  here); the pre-registered SHIP gate and its blocker are in
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
