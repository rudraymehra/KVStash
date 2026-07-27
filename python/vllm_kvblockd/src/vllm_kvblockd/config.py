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
  vLLM's per-request cache_salt: two requests with the same
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

# The connector's 32B blob-prefix layout version. Canonical HERE (not in
# connector.py) so the config fingerprint can fold it without a circular
# import — the connector imports it and packs it into every blob prefix.
# Bumping it forks the keyspace (clean misses) AND fails the prefix decode
# of old blobs; both degrade to recompute, never to a wrong byte.
BLOB_VERSION = 1


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
    multimodal identifiers are folded here so the WHOLE chain diverges.
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
    already folds cache_salt via the first block's extra keys — salt isolation
    is satisfied upstream); we bind it to OUR config identity:
    H(fingerprint || offload_key)."""
    if len(fp) != WIRE_KEY_LEN:
        raise ValueError(f"fingerprint must be {WIRE_KEY_LEN} bytes, got {len(fp)}")
    return blake3(_TIER_DOMAIN + fp + _lp(bytes(offload_key))).digest()


# --- vLLM config extraction (duck-typed: works on real VllmConfig and on the
# --- SimpleNamespace stubs the CI import check instantiates with) ---


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
    ep = ep.removeprefix("kvblockd://")
    host, _, port = ep.partition(":")
    if not host or not port:
        raise ValueError(f"endpoint must be kvblockd://host:port, got {endpoint!r}")
    return host, int(port)


class AdapterConfig:
    """Everything the connector needs, pulled defensively off vllm_config."""

    __slots__ = (
        "async_lookup",
        "async_store",
        "block_size",
        "connect_timeout",
        "dtype",
        "fingerprint",
        "host",
        "kv_cache_dtype",
        "load_deadline_s",
        "lookup_timeout_s",
        "model_name",
        "namespace",
        "op_timeout",
        "port",
        "prewarm_bytes",
        "revision",
        "so_rcvbuf",
        "staging_bytes",
        "store_flush_timeout_s",
        "store_queue_bytes",
        "store_staging_bytes",
        "streams",
        "token",
        "tokenizer",
        "tokenizer_revision",
        "verify",
        "world_size",
    )

    @classmethod
    def from_vllm_config(cls, vllm_config) -> AdapterConfig:
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
        # OPT-IN SO_RCVBUF override (bytes). None/unset = leave kernel
        # receive-window autotuning alone (setting SO_RCVBUF disables it and
        # clamps at the non-netns-writable net.core.rmem_max — see the
        # client's _connect for the full rationale).
        rcvbuf = get_extra_config(ktc, "kvblockd_so_rcvbuf", None)
        c.so_rcvbuf = int(rcvbuf) if rcvbuf not in (None, "", 0, "0") else None
        # Pinned host staging CAP (bytes) for the connector's CUDA load slab:
        # the slab grows up to this and never past it; loads bigger than the
        # cap drain through it in cap-sized passes. <=0 disables the slab
        # (per-block loads only).
        c.staging_bytes = int(get_extra_config(ktc, "kvblockd_staging_bytes", 2 * 2**30))
        # Eager pinned-slab pre-warm CEILING, applied at the FIRST CUDA layout
        # capture (default: the staging cap). cudaHostAlloc of gigabytes takes
        # hundreds of ms — paying it at capture time instead of inside the
        # first measured load, and saying so in the log, keeps the
        # first-request stall out of the warm arm. The EFFECTIVE pin is
        # min(this, what the pipelined load path can ever reserve: two
        # 256MiB-capped slab halves, ~512MiB at defaults) — the prewarm never
        # pins bytes no load will touch, and this knob can only shrink it.
        c.prewarm_bytes = int(get_extra_config(ktc, "kvblockd_prewarm_bytes", c.staging_bytes))
        c.op_timeout = float(get_extra_config(ktc, "kvblockd_op_timeout_s", 10.0))
        c.connect_timeout = float(get_extra_config(ktc, "kvblockd_connect_timeout_s", 5.0))
        # Write-behind stores: wait_for_save stages OWNED copies and returns;
        # a single "kvb-store" daemon thread drains them to the daemon. False
        # = the original synchronous put loop, byte-identical, kept for A/B.
        c.async_store = bool(get_extra_config(ktc, "kvblockd_async_store", True))
        # Staged-copy budget for the store queue. Enqueue past the budget
        # never blocks the engine: the block (and, tail-skip, the rest of its
        # request) is dropped and counted in the connector's dropped_puts /
        # dropped_put_bytes counters.
        c.store_queue_bytes = int(get_extra_config(ktc, "kvblockd_store_queue_bytes", 1 << 30))
        # Pinned staging pool for the CUDA gathered-store fast path. 0 (or
        # negative) disables the pool; UNSET auto-sizes it to the queue byte
        # budget plus two slots of headroom (~1 GiB of pinned host RAM at
        # the default queue budget, allocated at the first CUDA layout
        # capture and held for the run, ON TOP of the load slab's prewarm —
        # budget both on a RAM-tight rig). A denied lease is congestion, not
        # failure: the block degrades to an IDENTICAL bytearray blob (same
        # bytes, same accounting). An explicit value overrides the auto-size.
        ssb = get_extra_config(ktc, "kvblockd_store_staging_bytes", None)
        c.store_staging_bytes = int(ssb) if ssb not in (None, "") else None
        # shutdown() waits at most this long for the queue to flush; whatever
        # is still undelivered is counted dropped and disclosed.
        c.store_flush_timeout_s = float(
            get_extra_config(ktc, "kvblockd_store_flush_timeout_s", 10.0)
        )
        # Async lookup (default OFF): get_num_new_matched_tokens answers None
        # ("ask again") while a background thread runs BATCH_EXISTS, instead
        # of blocking the scheduler step on the wire round-trip. Flag off =
        # the original synchronous lookup, unchanged.
        c.async_lookup = bool(get_extra_config(ktc, "kvblockd_async_lookup", False))
        # A pending async lookup older than this is answered as a miss and
        # pruned — a wedged resolver must never park a request forever.
        lt = get_extra_config(ktc, "kvblockd_lookup_timeout_s", None)
        c.lookup_timeout_s = float(lt) if lt not in (None, "") else c.op_timeout
        # Overall per-LOAD wall-clock ceiling. op_timeout bounds each recv,
        # but a slow daemon that keeps trickling passes every per-recv check
        # forever; past this deadline the load abandons its remaining shards,
        # flags the unfilled block ids, and degrades to a miss. <=0 disables.
        c.load_deadline_s = float(get_extra_config(ktc, "kvblockd_load_deadline_s", 30.0))

        cache = getattr(vllm_config, "cache_config", None)
        c.block_size = int(getattr(cache, "block_size", 16) or 16)
        model = getattr(vllm_config, "model_config", None)
        c.model_name = str(getattr(model, "model", "unknown-model"))
        c.dtype = str(getattr(model, "dtype", getattr(cache, "cache_dtype", "auto")))
        # Resolved KV-CACHE dtype (mirrors tier_fingerprint_fields): "auto" is
        # an instruction, not an identity — it resolves to the model dtype.
        # Without this field a bf16-KV and an fp8-KV engine over the same
        # model dtype mint IDENTICAL keys and cross-serve blobs whose 32B
        # prefix only rejects them at load time (permanent silent miss storm).
        kv_dtype = str(getattr(cache, "cache_dtype", "auto") or "auto").replace("torch.", "")
        if kv_dtype in ("auto", ""):
            kv_dtype = str(getattr(model, "dtype", "auto")).replace("torch.", "")
        c.kv_cache_dtype = kv_dtype
        # Tokenizer identity: the same model path served through two different
        # tokenizers yields different token-id streams — same ids, different
        # text. vLLM's ModelConfig exposes .tokenizer (defaults to the model
        # path) and .tokenizer_revision; fall back to model path + revision.
        tok = getattr(model, "tokenizer", None)
        c.tokenizer = str(tok) if tok else c.model_name
        rev = (getattr(model, "tokenizer_revision", None)
               or getattr(model, "revision", None))
        c.tokenizer_revision = str(rev) if rev else ""
        # The MODEL WEIGHTS revision is its own identity field, not just the
        # tokenizer_revision fallback above: same model path + same tokenizer
        # revision but different weights produce different KV bytes for
        # identical token ids — sharing keys across them serves
        # stale-weights KV with zero errors.
        wrev = getattr(model, "revision", None)
        c.revision = str(wrev) if wrev else ""
        par = getattr(vllm_config, "parallel_config", None)
        c.world_size = int(getattr(par, "world_size", 1) or 1)
        # Refuse multi-GPU outright: the key identity has no per-rank
        # component, so at TP>1 every rank derives IDENTICAL keys for
        # DIFFERENT KV shards. The blob-prefix drift armor is blind to it
        # (shards are shape-symmetric), the first rank's put wins the
        # write-once dedup, and every other rank then loads the winner's
        # heads — silent garbage attention state with zero errors. Refusing
        # to boot beats corrupting quietly (same posture as
        # require_pinned_hashseed). Lifting this needs rank folded into the
        # worker-side chain plus a scheduler-side lookup story — a design
        # change, not a patch. Adversarially verified 2026-07-27: two
        # world_size=2 configs differing only in rank fingerprint
        # byte-identically.
        if c.world_size > 1:
            raise ValueError(
                f"kvblockd connector: world_size={c.world_size} is unsupported — "
                "block keys carry no rank identity, and tensor-parallel ranks "
                "would silently cross-load each other's KV shards. Run TP=1, "
                "or use the OffloadingConnector altitude (tier_manager) once "
                "its GPU validation lands."
            )
        # Every field here partitions the keyspace; adding one turns existing
        # blocks into clean misses (write-once cache repopulates) — so add
        # facts that change KV BYTES or TOKEN IDS, and nothing else. The
        # attention backend name is deliberately NOT folded: vLLM selects it
        # at worker init (get_attn_backend), after this constructor runs, so
        # it is not cheaply available here — the 32B blob prefix refuses a
        # cross-layout scatter if a backend flip ever changes the page bytes.
        c.fingerprint = fingerprint(
            {
                "scheme": "vllm-native-connector",
                "model_name": c.model_name,
                "block_size": c.block_size,
                "world_size": c.world_size,
                "dtype": c.dtype,
                "kv_cache_dtype": c.kv_cache_dtype,
                "tokenizer": c.tokenizer,
                "tokenizer_revision": c.tokenizer_revision,
                "revision": c.revision,
                "blob_version": BLOB_VERSION,
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
    "BLOB_VERSION",
    "WIRE_KEY_LEN",
    "AdapterConfig",
    "DeterminismError",
    "block_chain_keys",
    "chain_seed",
    "fingerprint",
    "get_extra_config",
    "parse_endpoint",
    "require_pinned_hashseed",
    "tier_fingerprint_fields",
    "tier_wire_key",
]
