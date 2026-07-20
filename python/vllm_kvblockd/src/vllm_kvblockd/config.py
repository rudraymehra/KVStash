"""Config parsing + key derivation for the vLLM native adapter.

Two derivations live here, and their bytes are pinned by
tests/golden/vllm_fingerprint.json (regenerating them silently would orphan
every block already stored under the old keys — a 100% miss storm, or worse
a cross-hit):

  FINGERPRINT (config-level, 32B): BLAKE3 over a domain-separated,
  length-prefixed encoding of the engine facts that make KV bytes
  compatible — model, block size, dtype, parallel layout. Mirrors the fields
  vLLM's own FileMapper writes to config.json for the fs tier. Two engines
  with different fingerprints can NEVER collide on a key.

  BLOCK CHAIN (request-level): seed = H(domain || fingerprint || lp(cache_salt)
  || lp(lora_name) || lp(mm_id_0) || lp(mm_id_1) ...), then
  key_i = H(domain || key_{i-1} || tokens of block i). The chain seed FOLDS IN
  vLLM's per-request cache_salt (correction C-14): two requests with the same
  tokens but different salts diverge at the seed, so every block key differs —
  salted isolation is structural, not a flag someone forgets (LMCache #2878).
  The LoRA adapter name folds in for the same reason: KV computed under one
  adapter is wrong for another, and for the base model. Every field is
  length-prefixed, and mm identifiers are prefixed INDIVIDUALLY — ids may
  contain any byte (UUIDs contain '-'), so a joined encoding is not injective.

The chain is BLAKE3 over raw token ids — it does NOT depend on Python's
builtin hash(), so the CONNECTOR's keys are deterministic regardless of
PYTHONHASHSEED. The startup determinism check is still enforced (see
require_pinned_hashseed) because vLLM's own prefix-cache and OffloadKey
chains DO depend on it, and a mixed fleet would silently never share.
"""

from __future__ import annotations

import os
import struct

from blake3 import blake3
from kvblockd.hashing import DeterminismError, startup_determinism_check

_FP_DOMAIN = b"kvblockd-vllm-fp-v1\x00"
# chain-v2: the seed layout is (salt, lora, per-id mm tail). Sharing a domain
# with the v1 layout (salt, joined-mm) could alias a v1 mm field against a v2
# lora field for identical bytes — the domain bump makes v1 blocks pure
# misses instead of possible cross-hits.
_CHAIN_DOMAIN = b"kvblockd-vllm-chain-v2\x00"
_TIER_DOMAIN = b"kvblockd-vllm-tier-v1\x00"

WIRE_KEY_LEN = 32


def _lp(b: bytes) -> bytes:
    """u32-LE length prefix — field values can never masquerade as separators
    (model names legally contain '/', '@', and anything else)."""
    return len(b).to_bytes(4, "little") + b


def _lps(s: str) -> bytes:
    return _lp(s.encode("utf-8"))


def fingerprint(fields: dict[str, object]) -> bytes:
    """32B config fingerprint over sorted key/value pairs (canonical order —
    dict insertion order must not change the bytes)."""
    blob = bytearray(_FP_DOMAIN)
    for k in sorted(fields):
        blob += _lps(k)
        blob += _lps(str(fields[k]))
    return blake3(bytes(blob)).digest()


def chain_seed(
    fp: bytes,
    cache_salt: str | None,
    mm_ids: list[str] | None = None,
    lora_name: str | None = None,
) -> bytes:
    """Request-level chain seed. cache_salt, the LoRA adapter name, and the
    multimodal identifiers are folded here so the WHOLE chain diverges (C-14).
    None and "" both encode as the empty field: an unsalted/base-model request
    has exactly one identity. mm identifiers are length-prefixed ONE BY ONE —
    each _lp field is self-delimiting, so ids containing '-' (UUIDs) or any
    other byte can never merge or split into a colliding encoding. Any new
    field must be inserted BEFORE the mm tail (or bump _CHAIN_DOMAIN): the
    variable-length tail is only unambiguous because nothing follows it."""
    if len(fp) != WIRE_KEY_LEN:
        raise ValueError(f"fingerprint must be {WIRE_KEY_LEN} bytes, got {len(fp)}")
    blob = bytearray(_CHAIN_DOMAIN)
    blob += fp
    blob += _lps(cache_salt or "")
    blob += _lps(lora_name or "")
    for mm_id in mm_ids or []:
        blob += _lps(mm_id)
    return blake3(bytes(blob)).digest()


def block_chain_keys(seed: bytes, token_ids: list[int], block_size: int) -> list[bytes]:
    """Chain keys for the FULL blocks of token_ids: key_i covers blocks
    0..i by construction (prefix property — exactly what BATCH_EXISTS's
    consecutive-prefix count answers). Trailing partial blocks get no key."""
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    n_blocks = len(token_ids) // block_size
    keys: list[bytes] = []
    prev = seed
    for i in range(n_blocks):
        chunk = token_ids[i * block_size : (i + 1) * block_size]
        try:
            tok = struct.pack(f"<{block_size}I", *chunk)
        except struct.error as e:  # negative / >u32 token id: refuse loudly,
            # a silently-wrapped id would store under a colliding key.
            raise ValueError(f"token id out of u32 range in block {i}: {e}") from e
        prev = blake3(_CHAIN_DOMAIN + prev + tok).digest()
        keys.append(prev)
    return keys


def tier_wire_key(fp: bytes, offload_key: bytes) -> bytes:
    """SecondaryTierManager altitude: vLLM owns the block hash (its OffloadKey
    already folds cache_salt via the first block's extra keys — C-14 upstream);
    we bind it to OUR config identity: H(fingerprint || offload_key)."""
    if len(fp) != WIRE_KEY_LEN:
        raise ValueError(f"fingerprint must be {WIRE_KEY_LEN} bytes, got {len(fp)}")
    return blake3(_TIER_DOMAIN + fp + _lp(bytes(offload_key))).digest()


# --- vLLM config extraction (duck-typed: works on real VllmConfig and on the
# --- SimpleNamespace stubs the A6 import check instantiates with) ---


def get_extra_config(kv_transfer_config, key: str, default):
    """kv_transfer_config.get_from_extra_config when present (real vLLM),
    plain dict access otherwise (stubs, older releases)."""
    getter = getattr(kv_transfer_config, "get_from_extra_config", None)
    if callable(getter):
        return getter(key, default)
    extra = getattr(kv_transfer_config, "kv_connector_extra_config", None) or {}
    return extra.get(key, default)


def parse_endpoint(endpoint: str) -> tuple[str, int]:
    """kvblockd://host:port (or bare host:port) -> (host, port)."""
    ep = endpoint.strip()
    if ep.startswith("kvblockd://"):
        ep = ep[len("kvblockd://") :]
    host, _, port = ep.partition(":")
    if not host or not port:
        raise ValueError(f"endpoint must be kvblockd://host:port, got {endpoint!r}")
    return host, int(port)


class AdapterConfig:
    """Everything the connector needs, pulled defensively off vllm_config."""

    __slots__ = (
        "host", "port", "namespace", "token", "streams", "verify",
        "op_timeout", "connect_timeout", "block_size", "model_name",
        "world_size", "dtype", "fingerprint",
    )

    @classmethod
    def from_vllm_config(cls, vllm_config) -> "AdapterConfig":
        ktc = getattr(vllm_config, "kv_transfer_config", None)
        endpoint = get_extra_config(ktc, "kvblockd_endpoint", "kvblockd://127.0.0.1:9440")
        c = cls()
        c.host, c.port = parse_endpoint(str(endpoint))
        c.namespace = str(get_extra_config(ktc, "kvblockd_namespace", "vllm"))
        c.token = str(
            get_extra_config(ktc, "kvblockd_token", os.environ.get("KVBLOCKD_TOKEN", ""))
        )
        c.streams = int(get_extra_config(ktc, "kvblockd_streams", 4))
        c.verify = bool(get_extra_config(ktc, "kvblockd_verify", True))
        c.op_timeout = float(get_extra_config(ktc, "kvblockd_op_timeout_s", 10.0))
        c.connect_timeout = float(get_extra_config(ktc, "kvblockd_connect_timeout_s", 5.0))

        cache = getattr(vllm_config, "cache_config", None)
        c.block_size = int(getattr(cache, "block_size", 16) or 16)
        model = getattr(vllm_config, "model_config", None)
        c.model_name = str(getattr(model, "model", "unknown-model"))
        c.dtype = str(getattr(model, "dtype", getattr(cache, "cache_dtype", "auto")))
        par = getattr(vllm_config, "parallel_config", None)
        c.world_size = int(getattr(par, "world_size", 1) or 1)
        c.fingerprint = fingerprint(
            {
                "scheme": "vllm-native-connector",
                "model_name": c.model_name,
                "block_size": c.block_size,
                "world_size": c.world_size,
                "dtype": c.dtype,
            }
        )
        return c


def tier_fingerprint_fields(offloading_spec) -> dict[str, object]:
    """Mirror FileMapper.get_run_config()'s identity fields (parallel_agnostic
    collapse included: the offloaded block is canonical TP-rank-1 form, so
    tp/pp do not partition the keyspace)."""
    vc = getattr(offloading_spec, "vllm_config", None)
    model_cfg = getattr(vc, "model_config", None)
    model = getattr(model_cfg, "model", "unknown-model")
    cache = getattr(vc, "cache_config", None)
    dtype = str(getattr(cache, "cache_dtype", "auto")).replace("torch.", "")
    if dtype in ("auto", ""):
        # "auto" is an instruction, not an identity: two engines can resolve
        # it to different dtypes, so fold what it resolves to — the model
        # dtype (vLLM's auto cache_dtype follows model_config.dtype).
        dtype = str(getattr(model_cfg, "dtype", "auto")).replace("torch.", "")
    hash_block = getattr(offloading_spec, "hash_block_size", None)
    if hash_block is None:
        hash_block = getattr(cache, "block_size", 16)
    # The blob at this altitude is one primary-tier block = the concatenation
    # over ALL KV-cache groups; a different group structure is a different
    # byte layout, so the group count partitions the keyspace (FileMapper
    # folds it into config.json for the same reason).
    groups = getattr(
        getattr(offloading_spec, "kv_cache_config", None), "kv_cache_groups", None
    )
    try:
        n_groups = len(groups) if groups is not None else 1
    except TypeError:
        n_groups = 1
    return {
        "scheme": "vllm-native-tier",
        "model_name": str(model),
        "hash_block_size": int(hash_block),
        "gpu_blocks_per_file": int(getattr(offloading_spec, "block_size_factor", 1)),
        "kv_cache_groups": n_groups,
        "tp_size": 1,
        "pp_size": 1,
        "pcp_size": 1,
        "dcp_size": 1,
        "dtype": dtype,
    }


def require_pinned_hashseed() -> None:
    """Refuse to start under an unpinned PYTHONHASHSEED, naming the fix.
    KVBLOCKD_SKIP_HASHSEED_CHECK=1 escapes it for single-process experiments
    (documented footgun: a fleet that skips this never shares cache)."""
    if os.environ.get("KVBLOCKD_SKIP_HASHSEED_CHECK") == "1":
        return
    startup_determinism_check()


__all__ = [
    "AdapterConfig",
    "DeterminismError",
    "WIRE_KEY_LEN",
    "block_chain_keys",
    "chain_seed",
    "fingerprint",
    "get_extra_config",
    "parse_endpoint",
    "require_pinned_hashseed",
    "tier_fingerprint_fields",
    "tier_wire_key",
]
