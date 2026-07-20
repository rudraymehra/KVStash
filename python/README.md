# python — kvblockd client + engine adapters

Three packages, Apache-2.0, `requires-python >= 3.10`:

| Package | What | Deps |
|---|---|---|
| `kvblockd` | KVB1 wire client (protocol codec, sync socket client, connection pool, key hashing) | `blake3`, `xxhash` |
| `lmcache-kvblockd` | LMCache `RemoteConnector` backend (`kvblockd://` scheme) | `kvblockd`, `lmcache>=0.5.1,<0.6` |
| `vllm-kvblockd` | vLLM NATIVE adapter, two altitudes: `KvblockdConnector` (`KVConnectorBase_V1`, CPU-backend e2e-gated) + `KvblockdTierManager` (`SecondaryTierManager` under `OffloadingConnector`, GPU e2e deferred — `python/vllm_kvblockd/DEFER.md`) | `kvblockd`, `numpy` (`vllm` is an extra, never a hard dep) |
| `sglang-kvblockd` | SGLang `HiCacheStorage` (v1) backend — CPU-validated, GPU e2e DEFERred (`docs/design/sglang-hicache-v1.1.md`), NOT on PyPI | `kvblockd` (sglang/torch as extras) |

## kvblockd

```python
from kvblockd.client import Client
c = Client("127.0.0.1:9440", namespace="tenant", token="secret", streams=4)
c.put(key32, blob)
vals, statuses = c.batch_get_bytes([key32])       # simple path
n, per = c.batch_exists([key32])                   # consecutive-prefix count
c.batch_get_scatter(keys, prefix_len, alloc)       # zero-copy: bytes land in your buffer
```

The wire codec (`kvblockd.protocol`) is byte-identical to the Go server's and
is tested against the same golden frames (`internal/protocol/testdata/frames`).
`kvblockd.hashing.wire_key` and Go's `pkg/client.WireKey` share
`tests/golden/hash_chain.json` — the key a client derives is portable across
languages.

## lmcache-kvblockd

The LMCache integration. See `docs/INTEGRATIONS.md` for the full setup. The
adapter is loaded via LMCache's plugin mechanism (no entry-points); the
connector never raises (failures degrade to cache misses) and reads
zero-copy into LMCache's pinned MemoryObj pool.

## vllm-kvblockd

The native vLLM integration — no LMCache in the loop. Config JSON for both
altitudes is in `python/vllm_kvblockd/src/vllm_kvblockd/__init__.py`; the
pinned upstream contract (and the SPEC-vs-merged-code delta for tier
loading) is in `python/vllm_kvblockd/UPSTREAM.lock`. Keys are a BLAKE3 chain
seeded by the config fingerprint + the request's `cache_salt` (C-14) —
golden-pinned in `python/vllm_kvblockd/tests/golden/vllm_fingerprint.json`.
Cross-instance sharing requires `PYTHONHASHSEED=0` on every process; the
adapter refuses to start unpinned and the error names the fix.

## Version-support policy

The `interface-tripwire` workflow pins the LMCache and vLLM releases the
adapters are verified against (see `docs/INTEGRATIONS.md`). For
`vllm-kvblockd`, every matrix cell (vLLM 0.25/0.24/0.23/0.22) must
import-and-instantiate BOTH altitudes; where a release lacks the
`vllm.v1.kv_offload.tiering` module, the tier-manager half of that cell is a
DOCUMENTED SKIP (printed by `bench/e2e/cpu/import_check_vllm_native.py`,
exit 0) — the connector altitude has no skips. As of the v0.25.0 pin, all
four releases carry both surfaces. `v0.1.0` promises nothing about API
stability — it is the first working release.

## Development

```bash
pip install -e ./python/kvblockd[test]
PYTHONHASHSEED=0 pytest python/kvblockd/tests            # needs the Go toolchain for the daemon fixture
pip install -e ./python/lmcache_kvblockd[test]
PYTHONHASHSEED=0 pytest python/lmcache_kvblockd/tests    # needs torch
```
