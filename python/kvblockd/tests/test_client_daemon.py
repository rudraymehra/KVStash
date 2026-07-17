"""Client ↔ real daemon round-trips: every verb against a live kvblockd."""

from __future__ import annotations

import hashlib

import pytest

from kvblockd import protocol as p
from kvblockd.client import Client
from kvblockd.errors import ConnectionLost


def _key(seed: str) -> bytes:
    return hashlib.blake2b(seed.encode(), digest_size=32).digest()


@pytest.fixture
def client(daemon):
    c = Client(daemon["addr"], namespace=daemon["namespace"], token=daemon["token"], streams=2)
    yield c
    c.close()


def test_put_get_roundtrip(client):
    k = _key("alpha")
    blob = b"\x11" * (1 << 20)
    assert client.put(k, blob) in (p.Status.OK, p.Status.OK_EXISTS)
    vals, sts = client.batch_get_bytes([k])
    assert sts[0] == p.Status.OK
    assert vals[0] == blob


def test_put_idempotent(client):
    k = _key("beta")
    blob = b"beta-block"
    assert client.put(k, blob) == p.Status.OK
    assert client.put(k, blob) == p.Status.OK_EXISTS  # write-once idempotent hit


def test_batch_exists_consecutive(client):
    ks = [_key(f"e{i}") for i in range(3)]
    client.put(ks[0], b"a")
    client.put(ks[1], b"b")
    # ks[2] absent → consecutive prefix stops at 2.
    n, per = client.batch_exists(ks)
    assert n == 2
    if per is not None:
        assert per[0] and per[1] and not per[2]


def test_scatter_zero_copy(client):
    k = _key("scatter")
    prefix = b"KVM1" + b"\x00" * 12  # 16-byte fake metadata prefix
    body = b"\xab" * 4096
    client.put(k, prefix + body)
    seen = {}

    def alloc(idx, pfx, body_len):
        seen["prefix"] = pfx
        buf = bytearray(body_len)
        seen["buf"] = buf
        return memoryview(buf)

    sts = client.batch_get_scatter([k], prefix_len=16, alloc=alloc)
    assert sts[0] == p.Status.OK
    assert seen["prefix"] == prefix
    assert bytes(seen["buf"]) == body


def test_delete_and_touch(client):
    k = _key("del")
    client.put(k, b"x")
    # A GET auto-leases → non-forced delete is ERR_LEASED; release then delete.
    client.batch_get_bytes([k])
    assert client.delete([k], force=False)[0] == p.Status.ERR_LEASED
    assert client.touch_lease([k], p.LEASE_RELEASE)[0] == p.Status.OK
    assert client.delete([k], force=False)[0] == p.Status.OK


def test_stats_json(client):
    doc = client.stats()
    assert b'"store":"dram"' in doc


def test_bad_token_rejected(daemon):
    with pytest.raises((ConnectionLost, p.StatusError, Exception)):
        Client(daemon["addr"], namespace=daemon["namespace"], token="wrong", streams=1)


def test_oversize_batch_splits(client):
    # More keys than a small negotiated cap would still work (the client tiles).
    ks = [_key(f"big{i}") for i in range(300)]
    for k in ks[:5]:
        client.put(k, b"present")
    n, _ = client.batch_exists(ks)
    assert n == 5  # first 5 present, 6th absent breaks the prefix


def test_tiled_get_global_indices(client):
    # >max_batch_keys (512) keys force tiling. Present blocks live in BOTH
    # tiles with DISTINCT payloads; each must come back at its correct GLOBAL
    # index. The pre-fix bug reused tile-local indices → tile 2 overwrote
    # tile 1's slot 0 (cross-key corruption). Regression for the BLOCKER.
    n = 600
    ks = [_key(f"tile{i}") for i in range(n)]
    present = {0: b"\x01" * 4096, 300: b"\x02" * 8192, 513: b"\x03" * 2048, 599: b"\x04" * 1024}
    for i, blob in present.items():
        client.put(ks[i], blob)
    vals, sts = client.batch_get_bytes(ks)
    assert len(vals) == n and len(sts) == n
    for i in range(n):
        if i in present:
            assert sts[i] == p.Status.OK, f"index {i} missing"
            assert vals[i] == present[i], f"index {i}: wrong bytes (tile collision?)"
        else:
            assert sts[i] == p.Status.NOT_FOUND, f"index {i} unexpectedly present"


def test_pool_survives_verb_after_status_error(client):
    # A StatusError keeps the conn in sync (re-pooled); the pool must not
    # leak the slot. Fire a delete that draws ERR_LEASED (in-sync status),
    # then keep using the client — no hang, slot returned.
    k = _key("poolslot")
    client.put(k, b"z")
    client.batch_get_bytes([k])  # auto-leases
    assert client.delete([k], force=False)[0] == p.Status.ERR_LEASED
    # If the slot leaked, this would hang; a working pool answers promptly.
    assert client.stats().startswith(b"{") or b"dram" in client.stats()
