"""Key derivation: LMCache CacheEngineKey → the 32-byte opaque wire key the
server stores blind (T3). The canonical serialization is length-prefixed (not
separator-joined) because model names legally contain '/' and '@'; BLAKE3-256
over a domain-separated blob gives the wire key. pkg/client.WireKey (Go) uses
the identical encoding — hash_chain.json is the shared oracle.

The LMCache chunk-hash chain that FEEDS CacheEngineKey.chunk_hash is LMCache's
own (vLLM's hash func, seeded by NONE_HASH which depends on PYTHONHASHSEED).
We do not reimplement it — we consume its output and only own the CEK→32B map.
startup_determinism_check() catches a hostile PYTHONHASHSEED loudly at boot.
"""

from __future__ import annotations

import os
import subprocess
import sys

from blake3 import blake3

_DOMAIN = b"kvblockd-cek-v1\x00"


def wire_key(fields) -> bytes:
    """Map an ordered sequence of string fields to the 32-byte wire key.

    fields is the CacheEngineKey in field order: (fmt, model_name,
    str(world_size), str(worker_id), str(chunk_hash)). Each field is
    length-prefixed (u32-LE) UTF-8, so no field value can be confused with a
    separator. Deterministic and identical to Go's WireKey.
    """
    blob = bytearray(_DOMAIN)
    for f in fields:
        b = f.encode("utf-8")
        blob += len(b).to_bytes(4, "little")
        blob += b
    return blake3(bytes(blob)).digest()  # 32 bytes


class DeterminismError(RuntimeError):
    """Raised when the key-hash environment is nondeterministic — two vLLM
    instances would then derive DIFFERENT keys for the same tokens and never
    share cache. The message names the fix."""


def startup_determinism_check() -> None:
    """Assert PYTHONHASHSEED is pinned and that a hash of the same input is
    stable across a fresh subprocess. Called from the connector's post_init.
    """
    seed = os.environ.get("PYTHONHASHSEED")
    if seed in (None, "", "random"):
        raise DeterminismError(
            "PYTHONHASHSEED is not pinned; LMCache key derivation would be "
            "nondeterministic across processes and cache would never be shared. "
            "Fix: export PYTHONHASHSEED=0 (identically on every vLLM worker)."
        )
    probe = "import os; print(hash(('kvblockd-determinism-probe', 1, 2, 3)))"
    env = dict(os.environ)
    out1 = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, env=env)
    out2 = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, env=env)
    if out1.stdout.strip() != out2.stdout.strip():
        raise DeterminismError(
            f"builtin hash() is nondeterministic across subprocesses under "
            f"PYTHONHASHSEED={seed!r} ({out1.stdout.strip()} != {out2.stdout.strip()}); "
            "pin PYTHONHASHSEED=0."
        )
