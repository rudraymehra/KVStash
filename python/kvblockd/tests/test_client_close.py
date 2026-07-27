"""Persistent scatter executor: the drain workers live in Client.__init__
(not one throwaway pool per batch_get_scatter call), are reused across
calls, and are shut down BEFORE the pool on close — post-close ops degrade
fast (eviction-class errors), never hang, and no worker thread leaks."""

from __future__ import annotations

import threading
import time

import pytest

from kvblockd import protocol as p
from kvblockd.client import Client
from kvblockd.errors import ConnectionLost


def _mk(daemon, streams=2):
    return Client(daemon["addr"], namespace=daemon["namespace"],
                  token=daemon["token"], streams=streams)


def _keys(n=8):
    return [bytes([i]) * 32 for i in range(n)]


def _kvb_get_threads():
    return {t.ident for t in threading.enumerate()
            if t.name.startswith("kvb-get") and t.is_alive()}


def _miss_alloc(idx, prefix, body_len):
    return None  # every block drained as a miss — the wire path still runs


def test_executor_is_persistent_and_workers_reused(daemon):
    c = _mk(daemon)
    try:
        ex = c._executor
        before = _kvb_get_threads()
        c.batch_get_scatter(_keys(), 0, _miss_alloc)
        first = _kvb_get_threads() - before  # workers born on the first call
        assert first
        c.batch_get_scatter(_keys(), 0, _miss_alloc)
        assert c._executor is ex                        # never rebuilt per call
        assert (_kvb_get_threads() - before) == first   # threads REUSED, not respawned
    finally:
        c.close()


def test_close_shuts_executor_before_pool_and_leaks_no_threads(daemon):
    c = _mk(daemon)
    before = _kvb_get_threads()
    c.batch_get_scatter(_keys(), 0, _miss_alloc)
    order = []
    real_shutdown, real_close = c._executor.shutdown, c._pool.close
    c._executor.shutdown = lambda **kw: (order.append("executor"), real_shutdown(**kw))[1]
    c._pool.close = lambda: (order.append("pool"), real_close())[1]
    c.close()
    # Executor first: a shard mid-drain keeps its checked-out conn (bounded
    # recvs) instead of racing a pool that could hand it a fresh dial.
    assert order == ["executor", "pool"]
    deadline = time.monotonic() + 5.0
    while (_kvb_get_threads() - before) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert (_kvb_get_threads() - before) == set()  # no leaked drain workers


def test_close_between_shard_submits_never_leaves_live_writers(daemon):
    """A concurrent close() landing BETWEEN shard submits (the connector's
    kvb-store drain / kvb-lookup threads do exactly this via _drop_client)
    must not let batch_get_scatter settle while an already-submitted shard
    is still writing caller-owned memoryviews — the caller re-maps those
    bytes for its next load the moment the call returns."""
    c = _mk(daemon, streams=2)
    keys = _keys(8)
    body = b"\xabPAYLOAD" * 32
    for k in keys:
        c.put(k, body)
    bufs = {}
    entered = threading.Event()

    def slow_alloc(idx, prefix, blen):
        entered.set()          # shard 0 provably mid-drain (past checkout)
        time.sleep(0.25)       # its writes land well after a bare raise would
        buf = bytearray(blen)
        bufs[idx] = buf
        return memoryview(buf)

    real_submit = c._executor.submit
    calls = {"n": 0}

    def submit(fn, *a, **kw):
        calls["n"] += 1
        if calls["n"] == 2:
            assert entered.wait(5.0)
            c.close()          # the drain-thread race, made deterministic
        return real_submit(fn, *a, **kw)

    c._executor.submit = submit
    with pytest.raises(ConnectionLost):
        c.batch_get_scatter(keys, 0, slow_alloc)
    assert bufs  # shard 0 really was mid-drain when close landed
    # The contract under test: once the call settles, NO thread may still
    # write caller-owned memory.
    snap = {i: bytes(b) for i, b in bufs.items()}
    time.sleep(0.6)
    assert {i: bytes(b) for i, b in bufs.items()} == snap


def test_post_close_ops_degrade_fast_never_hang(daemon):
    c = _mk(daemon)
    c.batch_get_scatter(_keys(), 0, _miss_alloc)
    c.close()
    t0 = time.monotonic()
    with pytest.raises(ConnectionLost):
        c.batch_get_scatter(_keys(), 0, _miss_alloc)      # fanned-out path
    # Single-shard path: the shard sees the closed pool's ConnectionLost and
    # degrades its keys to misses — even softer than a raise, still instant.
    assert c.batch_get_scatter(_keys(1), 0, _miss_alloc) == [p.Status.NOT_FOUND]
    with pytest.raises(ConnectionLost):
        c.put(_keys(1)[0], b"x")
    assert time.monotonic() - t0 < 2.0  # eviction-class degrades, not timeouts
