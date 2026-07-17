# LMCache HEAD verification — Week-5 Day-1 gate

Verified against `github.com/LMCache/LMCache@dev` and PyPI on 2026-07-17.
This is the precondition for the adapter: if the plugin surface or the
chunk-hash derivation moved, the day re-plans. Both held.

## Versions (pinned)

- lmcache latest on PyPI: **0.5.1** (releases: 0.5.0, 0.5.1). Pin `>=0.5.1,<0.6`.
- vllm latest on PyPI: **0.25.1**.
- The A6 tripwire (`.github/workflows/vllm-matrix.yml`) already pins
  lmcache {0.5.1, 0.5.0, 0.4.7} × vllm {0.25.1, 0.24.0, 0.23.0, 0.22.1};
  that matrix stays the single source of truth for drift. `versions.env`
  pins only what the e2e job installs.

## 1. RemoteConnector surface (base_connector.py) — MATCHES the A6 pin

Abstract (all async): `exists(key)->bool`, `exists_sync(key)->bool`,
`get(key)->Optional[MemoryObj]`, `put(key, memory_obj)`, `list()->List[str]`,
`close()`.

Fast-path toggles (defaults): `support_ping()->False`,
`support_batched_get()->False`, `support_batched_put()->False`,
`support_batched_contains()->False`, `support_batched_async_contains()->True`,
`support_batched_get_non_blocking()->True`.

Concrete/overridable:
- `post_init(self)` — optional init hook.
- `batched_contains(keys)->int` — "hit chunks by prefix match" (consecutive
  prefix count); raises NotImplementedError in the base → maps 1:1 to our
  BATCH_EXISTS `n_consecutive`.
- `batched_async_contains(lookup_id: str, keys, pin: bool = False)->int` —
  consecutive count; `pin` is the session-pin hint → our LeaseGrant.
- `batched_get_non_blocking(lookup_id: str, keys)->List[MemoryObj]` — returns
  ONLY the consecutive prefix of retrieved objects; **calls
  `ref_count_down()` on unreturned objects** to prevent leaks. Our impl must
  honor the same obligation.
- `remove_sync(key)->bool` — base raises NotImplementedError → our delete.
- `reshape_partial_chunk(...)` — `@NotAudit`, not needed for a whole-block store.

## 2. Registration mechanism (connector/__init__.py) — MATCHES the pin

`CreateConnector(url, loop, local_cpu_backend, config, metadata, plugin_name)`
→ `ConnectorManager` discovers adapters two ways:
1. **builtin**: modules named `*_adapter` that subclass `ConnectorAdapter`.
2. **plugin**: `config.remote_storage_plugins` list; per name, reads
   `extra_config["remote_storage_plugin.{name}.module_path"]` and
   `["...class_name"]`, dynamically imports, wraps a `RemoteConnector`
   subclass in `DynamicConnectorAdapter`. (Our adapter can also register the
   builtin way by being a `ConnectorAdapter` subclass — but plugin loading
   is the documented external path; use it.)

`ConnectorAdapter(ABC)`: `__init__(self, schema="")`, `can_parse(url) ->
url.startswith(self.schema)`, abstract `create_connector(context) ->
RemoteConnector`. So the scheme match is a plain prefix (`kvblockd://`).

`ConnectorContext` fields: `url, loop, local_cpu_backend, config, metadata,
plugin_name`. NOTE vs SPEC: it carries `local_cpu_backend` (the pinned
MemoryObj allocator) directly — the connector gets the allocator from the
context, not by importing a global. No entry-points file; discovery is
programmatic.

## 3. Chunk-hash derivation (token_database.py) — one correction to absorb

`ChunkedTokenDatabase`, default `chunk_size = 256`. Prefix chain:

```
prefix_hash = init_hash (NONE_HASH)
for chunk in token_chunks:
    prefix_hash = hash_func((prefix_hash, tuple(chunk_tokens), extra_keys))
    yield prefix_hash
```

Hash function is vLLM's, resolved at runtime (`get_hash_fn_by_name` →
`sha256_cbor` / `sha256_cbor_64bit`, fallback builtin `hash()`).
`NONE_HASH` comes from `vllm.v1.core.kv_cache_utils.init_none_hash(hash_func)`,
falls back to 0 without vLLM, and **depends on PYTHONHASHSEED** when the
builtin hash is in play — this is exactly why `startup_determinism_check()`
exists.

**CORRECTION to the plan:** chunk hashes render as **uint64 INTEGERS**
(`_normalize_hash_to_int`: first 8 bytes of a 32-byte digest, big-endian),
NOT hex strings. The `CacheEngineKey.chunk_hash` field is therefore an int;
our canonical `wire_key` serialization uses `str(chunk_hash)` (decimal),
which is unambiguous for an int. The parity test derives chunk hashes via
LMCache's own `ChunkedTokenDatabase` (not a reimplementation) and asserts
our `wire_key` matches — reimplementing vLLM's hash resolver is neither
needed nor wise; we consume LMCache's output and only own the CEK→32B map.

## 4. put payload type

`put(key, memory_obj: MemoryObj)` hands us a MemoryObj (tensor + metadata),
not pre-serialized bytes — so `meta.py`'s dtype/shape prefix is the right
shape. `local_cpu_backend` on the context is where GETs allocate the pinned
return buffer.

## Deltas from the plan (folded in)
- chunk_hash is int → parity uses LMCache's DB output, `str()` in wire_key.
- ConnectorContext gives `local_cpu_backend` directly (cleaner than the
  plan's "LMCache's LocalCPUBackend pinned pool" import assumption).
- Registration confirmed: `remote_storage_plugins` + the two extra_config
  keys, `DynamicConnectorAdapter` wrapper. adapter.py is safe to write.
