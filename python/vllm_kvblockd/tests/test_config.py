"""Key-derivation suite. The golden file pins the exact bytes (C-14):
changing them orphans every stored block, so a diff here is a migration,
never a refactor. One vector is ALSO recomputed from first principles
(raw blake3 calls) so the goldens can't just be parroting config.py."""

from __future__ import annotations

import json
import struct
from pathlib import Path

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
    """C-14: same tokens, different salts -> EVERY key differs (isolation is
    structural — there is no block index at which salted chains re-converge)."""
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
        kv_connector_extra_config = {
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


def test_hashseed_check_rejects_unpinned(monkeypatch):
    monkeypatch.delenv("KVBLOCKD_SKIP_HASHSEED_CHECK", raising=False)
    monkeypatch.setenv("PYTHONHASHSEED", "random")
    with pytest.raises(DeterminismError):
        require_pinned_hashseed()
    # the escape hatch works (single-process experiments)
    monkeypatch.setenv("KVBLOCKD_SKIP_HASHSEED_CHECK", "1")
    require_pinned_hashseed()  # must not raise
