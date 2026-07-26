"""Client ↔ real daemon round-trips: every verb against a live kvblockd."""

from __future__ import annotations

import hashlib
import itertools
import threading

import pytest

from kvblockd import client as client_mod
from kvblockd import protocol as p
from kvblockd.client import Client, _Conn
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


def test_rcvbuf_granted_exposed(client, daemon):
    # Default: SO_RCVBUF is NOT set (autotuning preserved) but the effective
    # buffer the kernel reports is still exposed — truth for the JSONL.
    assert isinstance(client.granted_rcvbuf, int)
    assert client.granted_rcvbuf > 0
    # Opt-in: an explicit so_rcvbuf asks before connect and exposes what the
    # kernel GRANTED (containers clamp silently; Linux doubles the ask).
    c = Client(daemon["addr"], namespace=daemon["namespace"], token=daemon["token"],
               streams=1, so_rcvbuf=1 << 20)
    try:
        assert isinstance(c.granted_rcvbuf, int)
        assert c.granted_rcvbuf > 0
    finally:
        c.close()


def test_shard_bounds_contiguous_cover():
    for n, k in ((7, 4), (8, 4), (600, 4), (3, 4), (1, 1), (5, 2)):
        bounds = Client._shard_bounds(n, min(k, n))
        assert bounds[0][0] == 0 and bounds[-1][1] == n
        for (_, e0), (s1, _) in itertools.pairwise(bounds):
            assert e0 == s1  # contiguous, no gap/overlap
        assert all(e > s for s, e in bounds)


def test_scatter_fanout_stitches_key_order_and_global_idx(daemon):
    # 4-way fan-out: statuses stitched back in key order across shards, and
    # alloc sees a GLOBALLY unique index (the slab-slot safety contract).
    c = Client(daemon["addr"], namespace=daemon["namespace"], token=daemon["token"], streams=4)
    try:
        n = 64
        ks = [_key(f"fan{i}") for i in range(n)]
        present = {i: bytes([i]) * (256 + i) for i in range(0, n, 3)}
        for i, blob in present.items():
            c.put(ks[i], blob)

        seen_idx = []
        lock = threading.Lock()
        bodies = {}

        def alloc(idx, prefix, body_len):
            with lock:
                seen_idx.append(idx)
            buf = bytearray(body_len)
            bodies[idx] = buf
            return memoryview(buf)

        sts = c.batch_get_scatter(ks, prefix_len=0, alloc=alloc)
        assert len(sts) == n
        for i in range(n):
            if i in present:
                assert sts[i] == p.Status.OK, f"index {i} missing"
                assert bytes(bodies[i]) == present[i], f"index {i}: wrong bytes"
            else:
                assert sts[i] == p.Status.NOT_FOUND
        assert len(seen_idx) == len(set(seen_idx)), "alloc saw a duplicate global idx"
        assert set(seen_idx) == set(present), "alloc invoked for a non-OK descriptor"
    finally:
        c.close()


def test_shard_connection_failure_yields_misses_for_that_shard_only(daemon, monkeypatch):
    # Kill exactly ONE shard's drain: its keys become misses; every other
    # shard's blocks stay valid and correctly stitched.
    c = Client(daemon["addr"], namespace=daemon["namespace"], token=daemon["token"], streams=4)
    try:
        n = 8
        ks = [_key(f"die{i}") for i in range(n)]
        blobs = {i: bytes([0xA0 + i]) * 512 for i in range(n)}
        for i, blob in blobs.items():
            c.put(ks[i], blob)

        bounds = Client._shard_bounds(n, 4)
        dead_base = bounds[1][0]  # second shard
        orig = _Conn.batch_get_scatter

        def flaky(self, keys, prefix_len, alloc, idx_base=0, deadline=None):
            if idx_base == dead_base:
                raise ConnectionLost("injected shard death")
            return orig(self, keys, prefix_len, alloc, idx_base, deadline)

        monkeypatch.setattr(client_mod._Conn, "batch_get_scatter", flaky)
        bodies = {}

        def alloc(idx, prefix, body_len):
            buf = bytearray(body_len)
            bodies[idx] = buf
            return memoryview(buf)

        sts = c.batch_get_scatter(ks, prefix_len=0, alloc=alloc)
        dead = set(range(*bounds[1]))
        for i in range(n):
            if i in dead:
                assert sts[i] == p.Status.NOT_FOUND, f"dead-shard key {i} not a miss"
            else:
                assert sts[i] == p.Status.OK, f"live-shard key {i} lost"
                assert bytes(bodies[i]) == blobs[i], f"live-shard key {i}: wrong bytes"
    finally:
        c.close()


@pytest.mark.parametrize("verify", [True, False], ids=["verify-on", "verify-off"])
def test_streams1_mixed_sizes_scatter_then_stream_stays_in_sync(daemon, verify):
    """streams=1: ONE connection drains a single batch_get_scatter whose OK
    blocks span every size class around the 32B prefix boundary — dlen < 32
    (prefix-only, empty body), dlen == 31/32/33 (the exact boundary), mid
    (1000), large (100000) and a ZERO-length blob — interleaved with misses.
    Statuses, prefix bytes and body bytes must all be exact, misses must never
    invoke alloc, and a follow-up verb on the SAME connection must work (the
    drain left no unread payload on the stream). Run under verify on AND off:
    the xxh3 digest path consumes the same bytes a digest-less drain must.

    Multi-frame (F_MORE) NOTE: the python client hard-codes its HELLO
    proposal to (0, 0) — it cannot negotiate a smaller max_frame_len from the
    client side, and the protocol floor is 16 MiB anyway, so this batch
    (~101KB) always arrives in one frame; multi-frame reassembly is covered
    by the Go wire tests, not forcible here."""
    c = Client(daemon["addr"], namespace=daemon["namespace"], token=daemon["token"],
               streams=1, verify=verify)
    try:
        sizes = [5, 31, 32, 33, 1000, 100000, 0]
        keys, blobs = [], {}
        for i, sz in enumerate(sizes):
            k = _key(f"mix-{verify}-{i}")
            blob = bytes((i * 37 + j) % 256 for j in range(sz))
            assert c.put(k, blob) in (p.Status.OK, p.Status.OK_EXISTS)
            blobs[len(keys)] = blob
            keys.append(k)
            keys.append(_key(f"mix-miss-{verify}-{i}"))  # interleaved miss

        prefixes, bodies = {}, {}

        def alloc(idx, pfx, body_len):
            prefixes[idx] = pfx
            buf = bytearray(body_len)
            bodies[idx] = buf
            return memoryview(buf)

        sts = c.batch_get_scatter(keys, prefix_len=32, alloc=alloc)
        assert len(sts) == len(keys)
        for idx in range(len(keys)):
            if idx in blobs:
                blob = blobs[idx]
                assert sts[idx] == p.Status.OK, f"idx {idx} (len {len(blob)}) missing"
                assert prefixes[idx] == blob[:32], f"idx {idx}: prefix bytes wrong"
                assert bytes(bodies[idx]) == blob[32:], f"idx {idx}: body bytes wrong"
            else:
                assert sts[idx] == p.Status.NOT_FOUND
                assert idx not in prefixes, f"alloc invoked for miss idx {idx}"
        # Stream sync: the same (single) conn must serve the next verb cleanly.
        n, _ = c.batch_exists(keys[:2])
        assert n == 1  # keys[0] present, keys[1] is the interleaved miss
        vals, sts2 = c.batch_get_bytes([keys[0]])
        assert sts2[0] == p.Status.OK and vals[0] == blobs[0]
    finally:
        c.close()


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
