"""SGLang HiCache key → kvblockd 32-byte wire key.

SGLang hands the backend *logical page keys*: chained-SHA-256 hex strings
(``sglang.srt.mem_cache.utils.get_hash_str``), one per page. We do not
reimplement that chain — we consume its output, exactly as the LMCache
adapter consumes LMCache's chunk hashes (same philosophy, same reused
``kvblockd.hashing.wire_key`` construction, same golden-vector discipline).

One logical page maps to 1 or 2 *physical* objects in kvblockd:

  MHA : ``{key}_{tp_rank}_{pp_rank}_k`` and ``{key}_{tp_rank}_{pp_rank}_v``
        (each TP rank holds its own K/V head shard → per-rank objects)
  MLA : ``{key}_{pp_rank}_k``
        (KV is replicated across TP ranks → one object, shared by all ranks)

and the 32-byte wire key is

  key32 = BLAKE3( "sglang-hicache-v1" ‖ model_fingerprint ‖ full_key_str )

via the length-prefixed, domain-separated ``kvblockd.hashing.wire_key``
encoding (u32-LE length before each field — no separator ambiguity). The
fingerprint folds model_name + tp/pp geometry into every key so two
deployments with different sharding can NEVER cross-hit inside a shared
namespace — a cross-hit would be silent KV corruption, not a cache bug.

Fingerprint rules (deliberate, asymmetric):
  - MHA folds tp_size (a ``_0_0_k`` shard under tp=2 holds different head
    slices than under tp=4 — must not collide).
  - MLA omits tp_size (pages are TP-invariant; folding it would only
    forbid legitimate sharing across TP widths).
  - pp_size is always folded (the same pp_rank owns different layer
    ranges under different pp widths).
  - context-parallel (attn_cp_size>1) and head-split (should_split_heads)
    deployments get extra fingerprint tags so they cannot collide with
    plain deployments; the head-split tag folds tp_lcm_size (the virtual
    rank geometry) so two head-split deployments with different lcm
    widths cannot collide with each other either.

``tests/golden/sglang_keymap.json`` pins all of this byte-for-byte.
"""

from __future__ import annotations

from dataclasses import dataclass

from kvblockd.hashing import wire_key

# First wire_key field: domain-separates SGLang keys from LMCache
# CacheEngineKey keys (whose first field is the CEK fmt, e.g. "vllm")
# inside the same BLAKE3 construction / shared Go oracle.
SCHEME = "sglang-hicache-v1"


@dataclass(frozen=True)
class KeyScheme:
    """Everything key derivation needs, frozen at backend construction."""

    fingerprint: str
    tp_rank: int
    pp_rank: int
    is_mla: bool
    cp_tag: str = ""  # "_cp{rank}" when attn_cp_size > 1, else ""


def _cfg_get(cfg, name: str, default):
    """Duck-typed config access: HiCacheStorageConfig, SimpleNamespace, or dict."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def scheme_from_config(cfg) -> KeyScheme:
    """Build the KeyScheme from an SGLang HiCacheStorageConfig (duck-typed;
    every field is optional so pre-0.5.x configs and test stubs both work)."""
    model = _cfg_get(cfg, "model_name", None) or "unknown-model"
    tp_rank = int(_cfg_get(cfg, "tp_rank", 0))
    tp_size = int(_cfg_get(cfg, "tp_size", 1))
    pp_rank = int(_cfg_get(cfg, "pp_rank", 0))
    pp_size = int(_cfg_get(cfg, "pp_size", 1))
    is_mla = bool(_cfg_get(cfg, "is_mla_model", False))
    cp_rank = int(_cfg_get(cfg, "attn_cp_rank", 0))
    cp_size = int(_cfg_get(cfg, "attn_cp_size", 1))
    split_heads = bool(_cfg_get(cfg, "should_split_heads", False))
    tp_lcm_size = int(_cfg_get(cfg, "tp_lcm_size", 0))

    if is_mla:
        fingerprint = f"{model}|pp{pp_size}|mla"
    else:
        fingerprint = f"{model}|tp{tp_size}|pp{pp_size}|mha"
    if cp_size > 1:
        fingerprint += f"|cp{cp_size}"
    if split_heads:
        # Head-split replication stores per-*virtual*-rank shards whose
        # geometry is set by tp_lcm_size; we don't implement that layout —
        # tag it (lcm width folded) so a head-split deployment can never
        # cross-hit a plain rank-scoped one, nor a head-split one with a
        # different lcm width.
        fingerprint += f"|hs{tp_lcm_size}"

    cp_tag = f"_cp{cp_rank}" if cp_size > 1 else ""
    return KeyScheme(fingerprint=fingerprint, tp_rank=tp_rank, pp_rank=pp_rank,
                     is_mla=is_mla, cp_tag=cp_tag)


def multiplier(scheme: KeyScheme) -> int:
    """Physical objects per logical page (2 for MHA k+v, 1 for MLA)."""
    return 1 if scheme.is_mla else 2


def physical_suffixes(scheme: KeyScheme) -> tuple[str, ...]:
    """Per-page suffixes, in the SAME order the host pool's
    get_page_buffer_meta emits regions for one page (k then v for MHA)."""
    if scheme.is_mla:
        return (f"_{scheme.pp_rank}{scheme.cp_tag}_k",)
    base = f"_{scheme.tp_rank}_{scheme.pp_rank}{scheme.cp_tag}"
    return (f"{base}_k", f"{base}_v")


def page_suffix(scheme: KeyScheme) -> str:
    """Suffix for the generic (non-zero-copy) whole-flat-page object. Distinct
    from the k/v suffixes: v1 and generic layouts differ, and a shared suffix
    could hand a whole page to a k-region reader (the size check would catch
    it, but a distinct key never gets there)."""
    if scheme.is_mla:
        return f"_{scheme.pp_rank}{scheme.cp_tag}_pg"
    return f"_{scheme.tp_rank}_{scheme.pp_rank}{scheme.cp_tag}_pg"


def physical_keys(scheme: KeyScheme, keys) -> list[str]:
    """Flatten logical page keys into interleaved physical key strings:
    [k0_k, k0_v, k1_k, k1_v, …] — aligned index-for-index with the pointer
    list get_page_buffer_meta returns for the same pages."""
    suffixes = physical_suffixes(scheme)
    return [f"{k}{s}" for k in keys for s in suffixes]


def wire_key32(scheme: KeyScheme, full_key: str) -> bytes:
    """Map one full (suffixed) key string to the 32-byte kvblockd wire key."""
    return wire_key([SCHEME, scheme.fingerprint, full_key])


def wire_keys(scheme: KeyScheme, keys) -> list[bytes]:
    """physical_keys + wire_key32 in one hop (the hot-path helper)."""
    return [wire_key32(scheme, k) for k in physical_keys(scheme, keys)]
