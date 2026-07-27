"""Quality-wave suite (client package): error taxonomy under one root,
FrameError at the parse boundary, pool-checkout deadlines, misses-by-cause
counters, and the canonical-hashseed disclosure. Live daemon, real wire —
fakes appear only where a byte-level server bug must be simulated."""

from __future__ import annotations

import logging
import socket
import threading
import time

import pytest

from kvblockd import protocol as p
from kvblockd.client import Client, _Conn
from kvblockd.errors import (
    ConnectionLost,
    FatalProtocol,
    FrameError,
    KvblockdError,
    StatusError,
)


@pytest.fixture
def client(daemon):
    c = Client(daemon["addr"], namespace=daemon["namespace"], token=daemon["token"],
               streams=2)
    yield c
    c.close()


# ---------------------------------------------------------------------------
# commit 28: one error root; FrameError wraps struct/Index at the boundary
# ---------------------------------------------------------------------------
def test_error_taxonomy_single_root():
    """`except KvblockdError` must catch EVERY client-raised error — including
    the two that used to live in protocol.py on plain Exception."""
    for exc in (ConnectionLost, FatalProtocol, FrameError, StatusError):
        assert issubclass(exc, KvblockdError), exc
    # protocol re-exports the SAME classes (compat import path).
    assert p.StatusError is StatusError
    assert p.FrameError is FrameError
    # The pretty status name still renders (deferred protocol import).
    assert "NOT_FOUND" in str(StatusError(p.Op.BATCH_GET, p.Status.NOT_FOUND))


def test_short_exists_body_raises_frame_error_not_struct_error(client, monkeypatch):
    """A body shorter than its declared layout must surface as FrameError
    (a KvblockdError, evicting the conn) — never a raw struct.error."""
    monkeypatch.setattr(_Conn, "_read_body",
                        lambda self, n: p.pack_preamble(p.Status.OK, 1))
    with pytest.raises(FrameError):
        client.batch_exists([b"\x01" * 32])


# ---------------------------------------------------------------------------
# commit 12: checkout deadline + batch_exists deadline
# ---------------------------------------------------------------------------
def test_pool_checkout_timeout_raises_connection_lost(client):
    conn = client._pool.checkout()  # hold ONE of the two slots
    conn2 = client._pool.checkout()  # ...and the other: the pool is dry
    try:
        t0 = time.monotonic()
        with pytest.raises(ConnectionLost, match="checkout timed out"):
            client._pool.checkout(timeout=0.05)
        assert time.monotonic() - t0 < 1.0
    finally:
        client._pool.checkin(conn)
        client._pool.checkin(conn2)


def test_get_deadline_covers_pool_checkout(daemon):
    """With every conn checked out, a deadlined batch_get_scatter must come
    back as misses within its budget — not block on the semaphore past it
    (the front-door gap the in-band deadline machinery had)."""
    c = Client(daemon["addr"], namespace=daemon["namespace"], token=daemon["token"],
               streams=1)
    try:
        held = c._pool.checkout()  # the ONLY conn: checkout must time out
        t0 = time.monotonic()
        statuses = c.batch_get_scatter([b"\x02" * 32], 0,
                                       lambda i, pre, n: None,
                                       deadline=time.monotonic() + 0.1)
        elapsed = time.monotonic() - t0
        c._pool.checkin(held)
        assert statuses == [p.Status.NOT_FOUND]
        assert elapsed < 2.0, f"blocked {elapsed:.2f}s past the deadline on checkout"
        snap = c.counters.snapshot()
        assert snap["degraded_keys"] + snap["deadline_misses"] >= 1
    finally:
        c.close()


def test_hung_accepting_daemon_costs_the_dial_budget():
    """A daemon that ACCEPTS but never answers HELLO must cost
    connect_timeout (the dial budget), never op_timeout — dialing runs on
    caller threads (the connector's scheduler path among them), and a 10s
    HELLO stall re-paid at every breaker expiry is the audited permanent
    ~2/3 scheduling stall."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)  # accepts (backlog) but nobody ever answers HELLO
    try:
        t0 = time.monotonic()
        with pytest.raises((ConnectionLost, OSError)):
            Client(srv.getsockname(), namespace="t", token="sekret",
                   streams=1, connect_timeout=0.3, op_timeout=30.0)
        elapsed = time.monotonic() - t0
        assert elapsed < 3.0, \
            f"dial stalled {elapsed:.1f}s — op_timeout leaked into the HELLO recv"
    finally:
        srv.close()


def test_socket_moves_to_op_timeout_after_hello(daemon):
    """The dial budget ends when HELLO parses: the steady-state per-recv
    timeout (and the _Conn._op_timeout snapshot the whole deadline machinery
    restores to) must be op_timeout. Leaving the socket on connect_timeout
    silently tightened every later recv to the dial budget."""
    c = Client(daemon["addr"], namespace=daemon["namespace"], token=daemon["token"],
               streams=1, connect_timeout=2.0, op_timeout=7.5)
    try:
        conn = c._pool.checkout()
        try:
            assert conn._op_timeout == 7.5, \
                f"_op_timeout stuck at the dial budget ({conn._op_timeout})"
            assert conn._sock.gettimeout() == 7.5
        finally:
            c._pool.checkin(conn)
    finally:
        c.close()


def test_batch_exists_deadline(client):
    import os

    key = os.urandom(32)  # unique: the session daemon is shared across tests
    # A healthy daemon answers well inside the budget.
    n, _ = client.batch_exists([key], deadline=time.monotonic() + 5.0)
    assert n == 0
    # An already-expired deadline fails fast and loudly.
    with pytest.raises(ConnectionLost):
        client.batch_exists([key], deadline=time.monotonic() - 0.001)


# ---------------------------------------------------------------------------
# commit 15: misses-by-cause counters + eviction disclosure
# ---------------------------------------------------------------------------
def test_eviction_counter_and_warning(client, caplog):
    caplog.set_level(logging.WARNING, logger="kvblockd.pool")

    def boom(conn):
        raise RuntimeError("injected verb failure")

    with pytest.raises(RuntimeError):
        client._pool.run(boom)
    assert client.counters.snapshot()["evictions"] == 1
    assert client._pool.evictions == 1
    assert any("connection evicted" in r.getMessage() for r in caplog.records)


def test_degrade_to_miss_warns_and_counts(daemon, caplog):
    """A shard-level connection death must leave a rate-limited WARNING and a
    degraded_keys count — the silent path was an on-call trap."""
    caplog.set_level(logging.WARNING, logger="kvblockd.client")
    c = Client(daemon["addr"], namespace=daemon["namespace"], token=daemon["token"],
               streams=1)
    try:
        monkey_evt = threading.Event()

        def dead_scatter(self, keys, prefix_len, alloc, idx_base=0, deadline=None):
            monkey_evt.set()
            raise ConnectionLost("injected mid-drain death")

        real = _Conn.batch_get_scatter
        _Conn.batch_get_scatter = dead_scatter
        try:
            statuses = c.batch_get_scatter([b"\x04" * 32, b"\x05" * 32], 0,
                                           lambda i, pre, n: None)
        finally:
            _Conn.batch_get_scatter = real
        assert monkey_evt.is_set()
        assert statuses == [p.Status.NOT_FOUND] * 2
        assert c.counters.snapshot()["degraded_keys"] == 2
        assert any("degraded to misses" in r.getMessage() for r in caplog.records)
    finally:
        c.close()


# ---------------------------------------------------------------------------
# commit 28: canonical-hashseed disclosure
# ---------------------------------------------------------------------------
def test_noncanonical_hashseed_warns(monkeypatch, caplog):
    from kvblockd.hashing import startup_determinism_check

    caplog.set_level(logging.WARNING, logger="kvblockd.hashing")
    monkeypatch.setenv("PYTHONHASHSEED", "1")
    startup_determinism_check()  # deterministic, so it passes...
    assert any("NON-CANONICAL" in r.getMessage() for r in caplog.records), \
        "a non-canonical pinned seed must be loudly disclosed (fleet-split trap)"

    caplog.clear()
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    startup_determinism_check()
    assert not any("NON-CANONICAL" in r.getMessage() for r in caplog.records)
