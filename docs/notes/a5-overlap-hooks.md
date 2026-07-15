# A5 — do overlap hooks exist without forking the engines?

**Gate A5:** confirm kvblockd can hide network fetch latency behind GPU compute *without* forking vLLM/LMCache — i.e., the engines already expose per-layer / async connector hooks a third-party backend can use. This underwrites the "loading overlaps with compute, so the latency hides" answer.

**Verdict: A5 PASS** — layer-wise hooks (vLLM) and async non-blocking paths (LMCache) are both reachable out-of-tree. Verified from live source, July 2026. Caveat: this confirms the hooks *exist*; the *measured* overlap (nsys profile on a real GPU) is deferred to the GPU-benchmark week — A5 is the mechanism check, not the measurement.

## vLLM — per-layer overlap hooks

`class KVConnectorBase_V1(ABC)` in `vllm/distributed/kv_transfer/kv_connector/v1/base.py` (main). The module docstring states the worker-side flow verbatim:

> start_load_kv() - starts loading all KVs (maybe async)
> wait_for_layer_load() - blocks until layer i load is done
> save_kv_layer() - starts saving KV for layer i (maybe async)
> wait_for_save() - blocks until all saves are done

The **per-layer** hooks are what enable overlap — the load for layer N can complete while the GPU is still computing earlier layers:

```python
def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None
def wait_for_layer_load(self, layer_name: str) -> None      # blocks only until THIS layer is ready
def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs) -> None
def wait_for_save(self)
def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str] | None, set[str] | None]
```

`start_load_kv` kicks off the (possibly async) whole-request load at the start of the forward pass; `wait_for_layer_load(layer_name)` then blocks per layer, so transfer of a later layer's KV overlaps with compute on earlier layers. `save_kv_layer` symmetrically starts an async single-layer save mid-forward.

Newer, lower-level path (merged on main, but churn-prone — flagged, not depended on): `AsyncLookupManager.batch_lookup(...)` in `vllm/v1/kv_offload/tiering/async_lookup.py` (non-blocking lookup), `SecondaryTierManager` in `.../tiering/base.py`, `TieringOffloadingManager.touch/complete_store` in `.../tiering/manager.py`.

Source: https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/distributed/kv_transfer/kv_connector/v1/base.py

## LMCache — async non-blocking connector paths

`class RemoteConnector` in `lmcache/v1/storage_backend/connector/base_connector.py` (dev). The async coroutines let a connector overlap remote fetch with the engine's compute; the `lookup_id`-keyed variants are the prefetch/non-blocking paths, and both are enabled by default (`support_*` return `True`):

```python
async def batched_async_contains(self, lookup_id: str, keys, pin: bool = False) -> int   # prefix hit-count + session pin
async def batched_get_non_blocking(self, lookup_id: str, keys) -> List[MemoryObj]        # the non-blocking fetch
async def batched_get(self, keys) -> List[Optional[MemoryObj]]
async def get(self, key) -> Optional[MemoryObj]
```

Abstract methods our adapter must implement: `exists, exists_sync, get, put, list, close`. `batched_async_contains` maps 1:1 onto kvblockd's `BATCH_EXISTS` + lease verb; `batched_get_non_blocking` onto a pipelined `BATCH_GET`.

Registration is out-of-tree via `remote_storage_plugins` (URL-scheme `ConnectorAdapter` + the `RemoteConnector`) — no engine fork, no pip entry-point, package just has to be importable.

Source: https://raw.githubusercontent.com/LMCache/LMCache/dev/lmcache/v1/storage_backend/connector/base_connector.py · https://docs.lmcache.ai/developer_guide/extending_lmcache/remote_storage_plugins.html

## Why this matters
Both engines let a backend (a) learn what's cached without moving data (prefix-count / lookup), and (b) stream KV back layer-by-layer or asynchronously while the GPU works — the exact seam kvblockd needs to hide TCP latency behind compute, achieved purely as an out-of-tree connector. No engine forks; the A5 tripwire (A6) watches these method names for drift.
