"""Key-derivation suite. The golden file pins the exact bytes:
changing them orphans every stored block, so a diff here is a migration,
never a refactor. One vector is ALSO recomputed from first principles
(raw blake3 calls) so the goldens can't just be parroting config.py."""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path
from typing import ClassVar

import pytest
from blake3 import blake3

from vllm_kvblockd.config import (
    AdapterConfig,
    DeterminismError,
    block_chain_keys,
    chain_seed,
    fingerprint,
    parse_endpoint,
    require_pinned_hashseed,
    tier_fingerprint_fields,
    tier_wire_key,
)

_GOLDEN = Path(__file__).resolve().parent / "golden" / "vllm_fingerprint.json"


def _load():
    return json.loads(_GOLDEN.read_text())


def test_chain_matches_goldens():
    doc = _load()
    assert doc["scheme"] == "kvblockd-vllm-v2"
    for v in doc["vectors"]:
        fp = fingerprint(v["fingerprint_fields"])
        assert fp.hex() == v["fingerprint_hex"], v["fingerprint_fields"]
        seed = chain_seed(fp, v["cache_salt"], v["mm_ids"], v["lora_name"])
        assert seed.hex() == v["seed_hex"], (v["cache_salt"], v["mm_ids"], v["lora_name"])
        keys = block_chain_keys(seed, v["token_ids"], v["block_size"])
        assert [k.hex() for k in keys] == v["block_keys_hex"]


def test_tier_wire_key_matches_goldens():
    t = _load()["tier"]
    fp = fingerprint(t["fingerprint_fields"])
    assert fp.hex() == t["fingerprint_hex"]
    for ok_hex, wk_hex in zip(t["offload_keys_hex"], t["wire_keys_hex"]):
        assert tier_wire_key(fp, bytes.fromhex(ok_hex)).hex() == wk_hex
    # group index is part of the identity: same hash, different group != same key
    assert t["wire_keys_hex"][0] != t["wire_keys_hex"][1]


def test_first_principles_recompute():
    """Rebuild vector[1] (tenant-a) and vector[10] (mm ids) from raw blake3 —
    documents the exact v2 derivation independent of config.py's helpers:
    seed = H(domain, fp, lp(salt), lp(lora), lp(mm_0), lp(mm_1), ...)."""
    doc = _load()

    def lp(b: bytes) -> bytes:
        return len(b).to_bytes(4, "little") + b

    def fp_of(fields) -> bytes:
        blob = b"kvblockd-vllm-fp-v1\x00"
        for k in sorted(fields):
            blob += lp(k.encode()) + lp(str(fields[k]).encode())
        return blake3(blob).digest()

    v = doc["vectors"][1]
    assert v["cache_salt"] == "tenant-a" and v["lora_name"] is None
    fp = fp_of(v["fingerprint_fields"])
    assert fp.hex() == v["fingerprint_hex"]

    seed = blake3(
        b"kvblockd-vllm-chain-v2\x00" + fp + lp(b"tenant-a") + lp(b"")
    ).digest()
    assert seed.hex() == v["seed_hex"]

    prev, keys = seed, []
    b = v["block_size"]
    for i in range(len(v["token_ids"]) // b):
        chunk = v["token_ids"][i * b : (i + 1) * b]
        prev = blake3(
            b"kvblockd-vllm-chain-v2\x00" + prev + struct.pack(f"<{b}I", *chunk)
        ).digest()
        keys.append(prev.hex())
    assert keys == v["block_keys_hex"]

    v = doc["vectors"][10]
    assert v["mm_ids"] == ["a-b"]
    fp = fp_of(v["fingerprint_fields"])
    seed = blake3(
        b"kvblockd-vllm-chain-v2\x00" + fp + lp(b"") + lp(b"") + lp(b"a-b")
    ).digest()
    assert seed.hex() == v["seed_hex"]


def test_cache_salt_diverges_whole_chain():
    """cache_salt isolation: same tokens, different salts -> EVERY key differs
    (isolation is structural — there is no block index at which salted chains
    re-converge)."""
    fp = fingerprint({"m": "x"})
    toks = list(range(64))
    plain = block_chain_keys(chain_seed(fp, None, []), toks, 16)
    a = block_chain_keys(chain_seed(fp, "tenant-a", []), toks, 16)
    b = block_chain_keys(chain_seed(fp, "tenant-b", []), toks, 16)
    for i in range(4):
        assert len({plain[i], a[i], b[i]}) == 3
    # None and "" are the same identity: exactly one unsalted keyspace.
    assert chain_seed(fp, None, []) == chain_seed(fp, "", [])
    # mm identifiers are a further axis.
    assert chain_seed(fp, None, ["img-1"]) != chain_seed(fp, None, [])


def test_lora_name_diverges_whole_chain():
    """KV computed under a LoRA adapter must never serve another adapter or
    the base model — the name folds into the seed like cache_salt does."""
    fp = fingerprint({"m": "x"})
    toks = list(range(32))
    base = block_chain_keys(chain_seed(fp, None, [], None), toks, 16)
    a = block_chain_keys(chain_seed(fp, None, [], "adapter-a"), toks, 16)
    b = block_chain_keys(chain_seed(fp, None, [], "adapter-b"), toks, 16)
    for i in range(2):
        assert len({base[i], a[i], b[i]}) == 3
    # None and "" are both the base model: exactly one unadapted keyspace.
    assert chain_seed(fp, None, [], None) == chain_seed(fp, None, [], "")
    # ...and lora composes with salt (independent axes).
    assert chain_seed(fp, "s", [], "adapter-a") != chain_seed(fp, "s", [], None)


def test_mm_id_encoding_is_injective():
    """Per-id length prefixes: ids containing '-' (UUIDs) can never merge or
    split into a colliding encoding — the join-based encoding did."""
    fp = fingerprint({"m": "x"})
    assert chain_seed(fp, None, ["a-b"]) != chain_seed(fp, None, ["a", "b"])
    assert chain_seed(fp, None, ["ab", "c"]) != chain_seed(fp, None, ["a", "bc"])
    assert chain_seed(fp, None, [""]) != chain_seed(fp, None, [])


def test_prefix_property():
    """key_i depends only on blocks 0..i — a shared prompt prefix shares keys,
    which is what makes BATCH_EXISTS's consecutive-prefix count meaningful."""
    fp = fingerprint({"m": "x"})
    seed = chain_seed(fp, "s", [])
    short = block_chain_keys(seed, list(range(32)), 16)
    long = block_chain_keys(seed, list(range(32)) + [7, 7, 7], 16)
    assert long[:2] == short
    # ...and a diverging block diverges everything after it.
    fork = block_chain_keys(seed, list(range(31)) + [999, 7], 16)
    assert fork[0] == short[0] and fork[1] != short[1]


def test_partial_blocks_get_no_key():
    fp = fingerprint({"m": "x"})
    seed = chain_seed(fp, None, [])
    assert block_chain_keys(seed, list(range(15)), 16) == []
    assert len(block_chain_keys(seed, list(range(17)), 16)) == 1


def test_token_out_of_u32_range_refused():
    seed = chain_seed(fingerprint({"m": "x"}), None, [])
    with pytest.raises(ValueError):
        block_chain_keys(seed, [-1] * 16, 16)
    with pytest.raises(ValueError):
        block_chain_keys(seed, [2**32] * 16, 16)


def test_fingerprint_is_order_insensitive_and_length_prefixed():
    assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})
    # length-prefixing: field boundaries can't be forged by crafted values
    assert fingerprint({"a": "xy", "b": "z"}) != fingerprint({"a": "x", "b": "yz"})


def test_parse_endpoint():
    assert parse_endpoint("kvblockd://10.0.0.5:9440") == ("10.0.0.5", 9440)
    assert parse_endpoint("localhost:1234") == ("localhost", 1234)
    with pytest.raises(ValueError):
        parse_endpoint("kvblockd://noport")


def test_adapter_config_from_stub():
    class KTC:
        kv_connector_extra_config: ClassVar[dict] = {
            "kvblockd_endpoint": "kvblockd://127.0.0.1:19440",
            "kvblockd_namespace": "ns1",
            "kvblockd_token": "t",
            "kvblockd_streams": 2,
        }

        def get_from_extra_config(self, key, default):
            return self.kv_connector_extra_config.get(key, default)

    class VC:
        kv_transfer_config = KTC()
        cache_config = type("C", (), {"block_size": 16, "cache_dtype": "auto"})()
        model_config = type("M", (), {"model": "facebook/opt-125m", "dtype": "torch.bfloat16"})()
        parallel_config = type("P", (), {"world_size": 1})()

    cfg = AdapterConfig.from_vllm_config(VC())
    assert (cfg.host, cfg.port, cfg.namespace, cfg.streams) == ("127.0.0.1", 19440, "ns1", 2)
    assert len(cfg.fingerprint) == 32
    # the fingerprint is a pure function of the engine facts
    assert cfg.fingerprint == AdapterConfig.from_vllm_config(VC()).fingerprint


def test_tier_fields_resolve_auto_dtype_and_fold_groups():
    """cache_dtype='auto' is an instruction, not an identity: the tier
    fingerprint must fold what it resolves to (the model dtype). The KV-cache
    group count partitions the keyspace (a different group structure is a
    different primary-block byte layout)."""
    from types import SimpleNamespace

    spec = SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(model="m", dtype="torch.float16"),
            cache_config=SimpleNamespace(cache_dtype="auto", block_size=16),
        ),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[object(), object()]),
        hash_block_size=16,
        block_size_factor=1,
    )
    fields = tier_fingerprint_fields(spec)
    assert fields["dtype"] == "float16"  # resolved, torch. prefix stripped
    assert fields["kv_cache_groups"] == 2

    # An explicit cache_dtype is folded as-is; absent group info counts as 1.
    spec.vllm_config.cache_config.cache_dtype = "fp8_e4m3"
    del spec.kv_cache_config
    fields = tier_fingerprint_fields(spec)
    assert fields["dtype"] == "fp8_e4m3"
    assert fields["kv_cache_groups"] == 1
    # Group structure partitions: same config otherwise -> different key.
    a = fingerprint(fields)
    b = fingerprint({**fields, "kv_cache_groups": 2})
    assert tier_wire_key(a, b"k" * 36) != tier_wire_key(b, b"k" * 36)


def _vc(cache_dtype="auto", model_dtype="torch.bfloat16", tokenizer=None,
        tokenizer_revision=None, revision=None, model="facebook/opt-125m",
        calculate_kv_scales=None, extra=None):
    """Stub VllmConfig with the knobs the fingerprint-completion tests turn."""

    class KTC:
        kv_connector_extra_config: ClassVar[dict] = {"kvblockd_token": "t",
                                                     **(extra or {})}

        def get_from_extra_config(self, key, default):
            return self.kv_connector_extra_config.get(key, default)

    mattrs = {"model": model, "dtype": model_dtype}
    if tokenizer is not None:
        mattrs["tokenizer"] = tokenizer
    if tokenizer_revision is not None:
        mattrs["tokenizer_revision"] = tokenizer_revision
    if revision is not None:
        mattrs["revision"] = revision

    cattrs = {"block_size": 16, "cache_dtype": cache_dtype}
    if calculate_kv_scales is not None:
        cattrs["calculate_kv_scales"] = calculate_kv_scales

    class VC:
        kv_transfer_config = KTC()
        cache_config = type("C", (), cattrs)()
        model_config = type("M", (), mattrs)()
        parallel_config = type("P", (), {"world_size": 1})()

    return VC()


def test_fingerprint_folds_resolved_kv_cache_dtype():
    """DR-3 regression: a bf16-KV and an fp8-KV engine over the same model
    dtype minted IDENTICAL keys. The RESOLVED kv-cache dtype now forks the
    keyspace, and 'auto' resolves to the model dtype — an instruction, not an
    identity (mirrors tier_fingerprint_fields)."""
    auto = AdapterConfig.from_vllm_config(_vc(cache_dtype="auto"))
    fp8 = AdapterConfig.from_vllm_config(_vc(cache_dtype="fp8_e4m3"))
    assert auto.kv_cache_dtype == "bfloat16"  # resolved, torch. prefix stripped
    assert fp8.kv_cache_dtype == "fp8_e4m3"
    assert auto.fingerprint != fp8.fingerprint
    # ...and auto == the dtype it resolves to: one identity, not two.
    explicit = AdapterConfig.from_vllm_config(_vc(cache_dtype="bfloat16"))
    assert auto.fingerprint == explicit.fingerprint


def test_fp8_alias_normalized_before_fingerprinting():
    """'fp8' and 'fp8_e4m3' are the SAME e4m3 KV cache in vLLM (the bare
    alias resolves to e4m3 on CUDA). Before the alias table they minted
    DISJOINT keyspaces — two fleets with identical semantics that never share
    one block, a 100%-miss footgun with zero errors."""
    alias = AdapterConfig.from_vllm_config(_vc(cache_dtype="fp8"))
    canon = AdapterConfig.from_vllm_config(_vc(cache_dtype="fp8_e4m3"))
    assert alias.kv_cache_dtype == "fp8_e4m3"  # one canonical spelling
    assert alias.fingerprint == canon.fingerprint
    # e5m2 stays its own identity: different byte semantics, different keys.
    e5m2 = AdapterConfig.from_vllm_config(_vc(cache_dtype="fp8_e5m2"))
    assert e5m2.fingerprint != canon.fingerprint


def test_kv_cache_dtype_allowlist_refuses_scale_carrying_dtypes():
    """The connector's store path is a raw uint8 page copy: a dtype whose
    numeric meaning lives partly in auxiliary scale buffers (per-token int8,
    fp8_inc, fp8_ds_mla, ...) would round-trip its payload bytes but not its
    VALUE. Refused loudly at boot, never served wrong."""
    for bad in ("int8", "fp8_inc", "fp8_ds_mla"):
        with pytest.raises(ValueError, match="kv_cache_dtype"):
            AdapterConfig.from_vllm_config(_vc(cache_dtype=bad))
    # The full supported set boots: float32 kept per the skeptic — the CPU
    # rigs (bench/e2e/cpu) resolve auto -> float32 model dtype.
    for ok in ("auto", "float16", "bfloat16", "fp8", "fp8_e4m3", "fp8_e5m2"):
        AdapterConfig.from_vllm_config(_vc(cache_dtype=ok))
    cpu = AdapterConfig.from_vllm_config(
        _vc(cache_dtype="auto", model_dtype="torch.float32"))
    assert cpu.kv_cache_dtype == "float32"


def test_calculate_kv_scales_refused_at_boot():
    """calculate_kv_scales derives fp8 scales from the FIRST forward pass and
    bakes them into the page bytes with no scale metadata in the blob — two
    engine boots then produce byte-INCOMPATIBLE blobs under IDENTICAL keys,
    invisible to the shape-symmetric blob prefix. Refuse at CONFIG time
    (same posture as the world_size guard), never fall back quietly."""
    with pytest.raises(ValueError, match="calculate_kv_scales"):
        AdapterConfig.from_vllm_config(
            _vc(cache_dtype="fp8_e4m3", calculate_kv_scales=True))
    # vLLM's default (False, static scale=1.0 / checkpoint scales) boots.
    cfg = AdapterConfig.from_vllm_config(
        _vc(cache_dtype="fp8_e4m3", calculate_kv_scales=False))
    assert cfg.kv_cache_dtype == "fp8_e4m3"


def test_codec_knob_defaults_raw_and_refuses_everything_else():
    """Item 6 keystone: the blob-prefix codec FIELD exists for forward compat,
    but no codec serde has landed — every non-raw codec is refused at boot
    (behind the pre-registered lossy quality gate, not before it), and codec +
    engine-fp8 stacking is refused with its OWN reason (double quantization is
    unvalidated anywhere). Codec identity must never enter key derivation."""
    cfg = AdapterConfig.from_vllm_config(_vc())
    assert cfg.codec == "raw"
    # The knob never partitions the keyspace (namespaces are the tool).
    assert cfg.fingerprint == AdapterConfig.from_vllm_config(
        _vc(extra={"kvblockd_codec": "raw"})).fingerprint
    with pytest.raises(ValueError, match="quality gate"):
        AdapterConfig.from_vllm_config(_vc(extra={"kvblockd_codec": "fp8-cast"}))
    with pytest.raises(ValueError, match="double quantization"):
        AdapterConfig.from_vllm_config(
            _vc(cache_dtype="fp8_e4m3", extra={"kvblockd_codec": "fp8-cast"}))


def test_codec_refusal_cites_a_quality_gate_that_actually_exists():
    """The non-raw-codec refusal points the operator at the pre-registered
    codec quality gate in docs/CLAIMS.md. A pointer to a section that does
    not exist is a broken promise — the honesty story depends on the gate
    being written down BEFORE any codec measurement, so assert the cited
    section exists and pre-registers the load-bearing conditions: fixed
    prompt set, deterministic exact-match scoring with no LLM judge, a pass
    threshold at 16k and 32k, and lossy arms as separately-labeled rows."""
    claims = Path(__file__).resolve().parents[3] / "docs" / "CLAIMS.md"
    assert claims.is_file(), f"refusal message cites {claims}, which is missing"
    text = claims.read_text(encoding="utf-8")
    assert re.search(r"^##.*Codec quality gate", text, re.MULTILINE), (
        "docs/CLAIMS.md has no 'Codec quality gate' section, but the codec "
        "refusal message cites one")
    for needle in (
        "Fixed prompt set",
        "exact-match",
        "NO LLM judge",
        "16384",
        "32768",
        "separately-labeled rows",
    ):
        assert needle in text, (
            f"codec quality gate in docs/CLAIMS.md lost its pre-registered "
            f"condition: {needle!r}")


def test_fingerprint_folds_tokenizer_identity():
    """Same model weights behind a different tokenizer (or revision) produce
    different token-id streams for the same text — sharing keys across them
    serves the right ids for the wrong words."""
    base = AdapterConfig.from_vllm_config(_vc())
    other_tok = AdapterConfig.from_vllm_config(_vc(tokenizer="org/other-tokenizer"))
    other_rev = AdapterConfig.from_vllm_config(_vc(tokenizer_revision="beefcafe"))
    assert base.tokenizer == "facebook/opt-125m"  # falls back to the model path
    assert other_tok.tokenizer == "org/other-tokenizer"
    assert len({base.fingerprint, other_tok.fingerprint, other_rev.fingerprint}) == 3


def test_fingerprint_folds_model_revision_as_its_own_field():
    """The WEIGHTS revision is its own identity field, never just the
    tokenizer_revision fallback: two engines pinning the SAME tokenizer
    revision but different weights revisions produce different KV bytes for
    identical token ids — sharing keys across them serves stale-weights KV."""
    a = AdapterConfig.from_vllm_config(_vc(tokenizer_revision="tok-rev"))
    b = AdapterConfig.from_vllm_config(_vc(tokenizer_revision="tok-rev", revision="v2"))
    assert a.fingerprint != b.fingerprint, "weights revision did not fork the keyspace"
    assert a.tokenizer_revision == b.tokenizer_revision == "tok-rev"
    assert (a.revision, b.revision) == ("", "v2")
    # ...and the tokenizer_revision fallback to the weights revision is
    # unchanged (absent tokenizer_revision still inherits it).
    c = AdapterConfig.from_vllm_config(_vc(revision="v2"))
    assert c.tokenizer_revision == "v2" and c.revision == "v2"


def test_fingerprint_folds_blob_version(monkeypatch):
    """The connector's blob layout version is part of the config identity: a
    format bump must fork the keyspace (clean misses), not hand old-format
    blobs to a new-format decoder. One canonical constant, config-owned."""
    import vllm_kvblockd.config as cfg_mod
    from vllm_kvblockd import connector as conn_mod

    assert conn_mod.BLOB_VERSION == cfg_mod.BLOB_VERSION  # single source
    before = AdapterConfig.from_vllm_config(_vc()).fingerprint
    monkeypatch.setattr(cfg_mod, "BLOB_VERSION", cfg_mod.BLOB_VERSION + 1)
    after = AdapterConfig.from_vllm_config(_vc()).fingerprint
    assert before != after


def test_hashseed_check_rejects_unpinned(monkeypatch):
    monkeypatch.delenv("KVBLOCKD_SKIP_HASHSEED_CHECK", raising=False)
    monkeypatch.setenv("PYTHONHASHSEED", "random")
    with pytest.raises(DeterminismError):
        require_pinned_hashseed()
    # the escape hatch works (single-process experiments)
    monkeypatch.setenv("KVBLOCKD_SKIP_HASHSEED_CHECK", "1")
    require_pinned_hashseed()  # must not raise


def test_multi_gpu_refused_at_boot():
    """world_size>1 must refuse to construct: block keys carry no rank
    identity, so TP ranks would silently cross-load each other's KV shards
    (adversarially verified: rank is not in the fingerprint). Refusing beats
    corrupting — same posture as the hashseed determinism check."""
    import pytest

    from vllm_kvblockd.config import AdapterConfig

    class KTC:
        kv_connector_extra_config: ClassVar[dict] = {"kvblockd_token": "t"}

        def get_from_extra_config(self, key, default):
            return self.kv_connector_extra_config.get(key, default)

    class VC:
        kv_transfer_config = KTC()
        cache_config = type("C", (), {"block_size": 16, "cache_dtype": "auto"})()
        model_config = type("M", (), {"model": "m", "dtype": "torch.bfloat16"})()
        parallel_config = type("P", (), {"world_size": 2})()

    with pytest.raises(ValueError, match="world_size=2.*rank"):
        AdapterConfig.from_vllm_config(VC())


def _vc_extra(**extra):
    """Stub VllmConfig whose extra config carries the quick-perf knobs."""

    class KTC:
        kv_connector_extra_config: ClassVar[dict] = {"kvblockd_token": "t", **extra}

        def get_from_extra_config(self, key, default):
            return self.kv_connector_extra_config.get(key, default)

    class VC:
        kv_transfer_config = KTC()
        cache_config = type("C", (), {"block_size": 16, "cache_dtype": "auto"})()
        model_config = type("M", (), {"model": "m", "dtype": "torch.bfloat16"})()
        parallel_config = type("P", (), {"world_size": 1})()

    return VC()


def test_get_fanout_knob_parses_and_validates():
    """kvblockd_get_fanout: unset = None (the client keeps today's clamped
    default of 4); explicit values are range-checked at BOOT — 8 is the hard
    stop (iperf: no gain past 8 flows) and the fan-out may never exceed the
    pool (each shard needs its own pooled connection)."""
    assert AdapterConfig.from_vllm_config(_vc_extra()).get_fanout is None
    cfg = AdapterConfig.from_vllm_config(
        _vc_extra(kvblockd_get_fanout=8, kvblockd_streams=8))
    assert cfg.get_fanout == 8 and cfg.streams == 8
    with pytest.raises(ValueError, match="out of range"):
        AdapterConfig.from_vllm_config(
            _vc_extra(kvblockd_get_fanout=9, kvblockd_streams=16))
    with pytest.raises(ValueError, match="out of range"):
        AdapterConfig.from_vllm_config(_vc_extra(kvblockd_get_fanout=0))
    with pytest.raises(ValueError, match="kvblockd_streams"):
        AdapterConfig.from_vllm_config(_vc_extra(kvblockd_get_fanout=8))  # streams=4


def test_admission_gate_knobs_default_inert():
    """Item 12 ships INERT: every gate knob defaults to off/flat-compat —
    today's behavior until the short-prefix sweep locates a negative region."""
    cfg = AdapterConfig.from_vllm_config(_vc_extra())
    assert cfg.min_hit_tokens == 0
    assert cfg.recompute_ms_per_token == 0.0
    assert cfg.load_deadline_per_block_s == 0.0
    assert cfg.load_deadline_cap_s == 0.0
    cfg = AdapterConfig.from_vllm_config(_vc_extra(
        kvblockd_min_hit_tokens=512,
        kvblockd_load_deadline_per_block_s=0.05,
        kvblockd_load_deadline_cap_s=60.0,
    ))
    assert cfg.min_hit_tokens == 512
    assert cfg.load_deadline_per_block_s == 0.05
    assert cfg.load_deadline_cap_s == 60.0
    for bad in ({"kvblockd_min_hit_tokens": -1},
                {"kvblockd_recompute_ms_per_token": -0.1},
                {"kvblockd_load_deadline_per_block_s": -1.0}):
        with pytest.raises(ValueError, match="must be >= 0"):
            AdapterConfig.from_vllm_config(_vc_extra(**bad))


def test_recompute_knob_refused_until_plumbed_cross_role():
    """kvblockd_recompute_ms_per_token: the load-throughput EMA it needs is
    worker-role-only and the gate is scheduler-role-only (separate objects/
    processes in vLLM), so a nonzero value would silently admit every hit —
    the parse refuses it LOUDLY instead. Explicit 0 stays accepted (inert)."""
    with pytest.raises(ValueError, match="not yet plumbed cross-role"):
        AdapterConfig.from_vllm_config(
            _vc_extra(kvblockd_recompute_ms_per_token=0.25))
    cfg = AdapterConfig.from_vllm_config(
        _vc_extra(kvblockd_recompute_ms_per_token=0))
    assert cfg.recompute_ms_per_token == 0.0


def test_store_dedupe_knobs_parse():
    """Item 8 ships INERT: keys default 0 (today's self-healing re-put —
    a nonzero default would leave an acked-then-evicted block missing for
    up to the TTL); the multi-turn arm enables it explicitly."""
    cfg = AdapterConfig.from_vllm_config(_vc_extra())
    assert cfg.store_dedupe_keys == 0
    assert cfg.store_dedupe_ttl_s == 30.0
    cfg = AdapterConfig.from_vllm_config(_vc_extra(
        kvblockd_store_dedupe_keys=4096, kvblockd_store_dedupe_ttl_s=5.0))
    assert cfg.store_dedupe_keys == 4096 and cfg.store_dedupe_ttl_s == 5.0
    with pytest.raises(ValueError, match="must be >= 0"):
        AdapterConfig.from_vllm_config(_vc_extra(kvblockd_store_dedupe_keys=-1))


def test_pipeline_half_bytes_knob_parses_and_validates():
    """kvblockd_pipeline_half_bytes: default 256MiB (the pre-knob constant —
    every banked number was measured at it); explicit values re-shape the
    pipelined pass count on rigs with pinned-RAM headroom; <=0 is refused at
    boot (disabling the pipeline is the staging knob's job, and a silent 0
    would just skip the pipelined path with no disclosure)."""
    assert AdapterConfig.from_vllm_config(_vc_extra()).pipeline_half_bytes == 256 * 2**20
    cfg = AdapterConfig.from_vllm_config(
        _vc_extra(kvblockd_pipeline_half_bytes=512 * 2**20))
    assert cfg.pipeline_half_bytes == 512 * 2**20
    with pytest.raises(ValueError, match="must be > 0"):
        AdapterConfig.from_vllm_config(_vc_extra(kvblockd_pipeline_half_bytes=0))
    with pytest.raises(ValueError, match="must be > 0"):
        AdapterConfig.from_vllm_config(_vc_extra(kvblockd_pipeline_half_bytes=-1))
