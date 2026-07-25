"""Hash parity: the committed golden vectors must round-trip through
kvblockd.hashing.wire_key. The same golden file is the oracle for the Go
client (pkg/client/hashchain_test.go), so a mismatch here means the two
implementations have drifted apart.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from kvblockd.hashing import DeterminismError, startup_determinism_check, wire_key

_GOLDEN = Path(__file__).resolve().parent / "golden" / "hash_chain.json"


def _load():
    return json.loads(_GOLDEN.read_text())


def test_wire_key_matches_goldens():
    doc = _load()
    assert doc["scheme"] == "kvblockd-cek-v1"
    for v in doc["vectors"]:
        assert wire_key(v["fields"]).hex() == v["wire_key_hex"], v["fields"]


def test_wire_key_is_length_prefixed_not_joined():
    # A model name containing the field boundary must not collide with a
    # differently-split pair — the reason we length-prefix.
    a = wire_key(["vllm", "org/model", "1", "0", "5"])
    b = wire_key(["vllm", "org", "model/1", "0", "5"])  # same bytes if naively joined
    assert a != b


def test_determinism_check_rejects_random_seed(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "random")
    with pytest.raises(DeterminismError):
        startup_determinism_check()


@pytest.mark.skipif(os.environ.get("PYTHONHASHSEED") in (None, "", "random"),
                    reason="determinism check needs a pinned PYTHONHASHSEED")
def test_determinism_check_passes_when_pinned():
    startup_determinism_check()  # must not raise


