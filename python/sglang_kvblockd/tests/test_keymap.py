"""Keymap golden-vector + property tests. No daemon, no torch, no sglang —
this leg must pass in the barest CI environment (blake3 only)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace as NS

from sglang_kvblockd import keymap

_GOLDEN = Path(__file__).resolve().parent / "golden" / "sglang_keymap.json"


def _load():
    return json.loads(_GOLDEN.read_text())


def _scheme_for(case):
    cfg = None if case["config"] is None else NS(**case["config"])
    return keymap.scheme_from_config(cfg)


def test_golden_fingerprints_and_suffixes():
    doc = _load()
    assert doc["scheme"] == keymap.SCHEME
    for case in doc["cases"]:
        sch = _scheme_for(case)
        assert sch.fingerprint == case["fingerprint"], case["name"]
        assert list(keymap.physical_suffixes(sch)) == case["suffixes"], case["name"]
        assert keymap.page_suffix(sch) == case["page_suffix"], case["name"]


def test_golden_wire_keys():
    doc = _load()
    for case in doc["cases"]:
        sch = _scheme_for(case)
        for v in case["vectors"]:
            pks = keymap.physical_keys(sch, [v["logical_key"]])
            assert pks == v["physical_keys"], (case["name"], v["logical_key"])
            got = [keymap.wire_key32(sch, k).hex() for k in pks]
            assert got == v["wire_keys_hex"], (case["name"], v["logical_key"])
            page = keymap.wire_key32(
                sch, v["logical_key"] + keymap.page_suffix(sch)
            ).hex()
            assert page == v["page_wire_key_hex"], (case["name"], v["logical_key"])


def test_wire_key_is_32_bytes():
    sch = keymap.scheme_from_config(None)
    assert len(keymap.wire_key32(sch, "x")) == 32


def test_mha_interleaves_k_then_v_per_page():
    sch = keymap.scheme_from_config(
        NS(model_name="m", tp_rank=1, tp_size=2, pp_rank=0, pp_size=1,
           is_mla_model=False))
    flat = keymap.physical_keys(sch, ["a", "b"])
    # Must align index-for-index with get_page_buffer_meta's [k,v] per page.
    assert flat == ["a_1_0_k", "a_1_0_v", "b_1_0_k", "b_1_0_v"]
    assert keymap.multiplier(sch) == 2


def test_mla_single_object_no_tp_in_key_or_fingerprint():
    a = keymap.scheme_from_config(
        NS(model_name="m", tp_rank=0, tp_size=2, pp_rank=1, pp_size=2,
           is_mla_model=True))
    b = keymap.scheme_from_config(
        NS(model_name="m", tp_rank=3, tp_size=8, pp_rank=1, pp_size=2,
           is_mla_model=True))
    # MLA KV is TP-invariant: different tp geometry MUST still share keys.
    assert a.fingerprint == b.fingerprint
    assert keymap.physical_keys(a, ["h"]) == keymap.physical_keys(b, ["h"]) == ["h_1_k"]
    assert keymap.multiplier(a) == 1


def test_mismatched_deployments_cannot_cross_hit():
    base = dict(model_name="m", tp_rank=0, pp_rank=0, pp_size=1, is_mla_model=False)
    tp2 = keymap.scheme_from_config(NS(tp_size=2, **base))
    tp4 = keymap.scheme_from_config(NS(tp_size=4, **base))
    other_model = keymap.scheme_from_config(
        NS(model_name="m2", tp_rank=0, tp_size=2, pp_rank=0, pp_size=1,
           is_mla_model=False))
    k = "deadbeef" * 8
    keys = {keymap.wire_key32(s, k + "_0_0_k") for s in (tp2, tp4, other_model)}
    assert len(keys) == 3  # every geometry/model islands its keyspace


def test_headsplit_folds_tp_lcm_size():
    # Head-split shards are laid out by the VIRTUAL rank geometry
    # (tp_lcm_size): two head-split deployments with different lcm widths
    # must island their keyspaces, and neither may touch the plain one.
    base = dict(model_name="m", tp_rank=0, tp_size=2, pp_rank=0, pp_size=1,
                is_mla_model=False)
    plain = keymap.scheme_from_config(NS(**base))
    hs8 = keymap.scheme_from_config(
        NS(should_split_heads=True, tp_lcm_size=8, **base))
    hs16 = keymap.scheme_from_config(
        NS(should_split_heads=True, tp_lcm_size=16, **base))
    assert len({plain.fingerprint, hs8.fingerprint, hs16.fingerprint}) == 3


def test_pp_size_always_folded():
    # Same pp_rank under different pp widths holds DIFFERENT layer ranges.
    base = dict(model_name="m", tp_rank=0, tp_size=1, pp_rank=0, is_mla_model=False)
    pp1 = keymap.scheme_from_config(NS(pp_size=1, **base))
    pp2 = keymap.scheme_from_config(NS(pp_size=2, **base))
    assert pp1.fingerprint != pp2.fingerprint


def test_length_prefixing_blocks_boundary_shifts():
    # fingerprint/full_key boundary must not be forgeable by string shifts.
    sch_a = keymap.KeyScheme(fingerprint="m|tp1", tp_rank=0, pp_rank=0, is_mla=False)
    sch_b = keymap.KeyScheme(fingerprint="m|tp1_0", tp_rank=0, pp_rank=0, is_mla=False)
    assert keymap.wire_key32(sch_a, "_0abc") != keymap.wire_key32(sch_b, "abc")


def test_generic_page_suffix_never_collides_with_kv_suffixes():
    for cfg in (NS(model_name="m", tp_rank=0, tp_size=1, pp_rank=0, pp_size=1,
                   is_mla_model=False),
                NS(model_name="m", tp_rank=0, tp_size=1, pp_rank=0, pp_size=1,
                   is_mla_model=True)):
        sch = keymap.scheme_from_config(cfg)
        assert keymap.page_suffix(sch) not in keymap.physical_suffixes(sch)


def test_dict_config_and_none_config_work():
    d = keymap.scheme_from_config({"model_name": "m", "tp_size": 2, "tp_rank": 1})
    assert d.fingerprint == "m|tp2|pp1|mha" and d.tp_rank == 1
    n = keymap.scheme_from_config(None)
    assert n.fingerprint == "unknown-model|tp1|pp1|mha"
