# python — kvblockd client + engine adapters

Two packages, Apache-2.0, `requires-python >= 3.10`:

| Package | What | Deps |
|---|---|---|
| `kvblockd` | KVB1 wire client (protocol codec, sync socket client, connection pool, key hashing) | `blake3`, `xxhash` |
| `lmcache-kvblockd` | LMCache `RemoteConnector` backend (`kvblockd://` scheme) | `kvblockd`, `lmcache>=0.5.1,<0.6` |

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

## Version-support policy

The `interface-tripwire` workflow pins the LMCache and vLLM releases the
adapter is verified against (see `docs/INTEGRATIONS.md`). `v0.1.0` promises
nothing about API stability — it is the first working release.

## Development

```bash
pip install -e ./python/kvblockd[test]
PYTHONHASHSEED=0 pytest python/kvblockd/tests            # needs the Go toolchain for the daemon fixture
pip install -e ./python/lmcache_kvblockd[test]
PYTHONHASHSEED=0 pytest python/lmcache_kvblockd/tests    # needs torch
```
