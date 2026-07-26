"""Async-lookup suite (kvblockd_async_lookup, default OFF): None means "still
resolving, ask again"; the async tuple element stays False (no async loads
this wave); aborts prune; the pending map is bounded and deadline-pruned."""

from __future__ import annotations

import time

import pytest

torch = pytest.importorskip("torch")

from test_connector import StubRequest, StubVllmConfig, fill_block, fresh_kv, run_step

from vllm_kvblockd.connector import _LOOKUP_PENDING_CAP, KvblockdConnector, _LookupResolver


def make_conn(daemon, **extra):
    cfg = StubVllmConfig(daemon["port"])
    cfg.kv_transfer_config.kv_connector_extra_config.update(extra)
    return KvblockdConnector(cfg, role="scheduler", kv_cache_config=None)


def poll_lookup(conn, req, num_computed=0, timeout=5.0):
    """Call get_num_new_matched_tokens the way the scheduler would: retry
    while it answers None. Returns the final (hit, async) tuple; asserts the
    async element is False on EVERY answer."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        n, is_async = conn.get_num_new_matched_tokens(req, num_computed)
        assert is_async is False  # no async loads this wave, ever
        if n is not None:
            return n
        time.sleep(0.01)
    pytest.fail("async lookup never resolved")


class FakeResolver:
    """Resolver stand-in that NEVER answers — makes deadline/cap behavior
    deterministic (the real one answers a local daemon in microseconds)."""

    def __init__(self):
        self.posted = []

    def alive(self):
        return True

    def pop(self, rid):
        return None

    def discard(self, rid, inflight=False):
        return None

    def post(self, *item):
        self.posted.append(item)

    def stop(self, timeout=1.0):
        return None


def test_async_lookup_retry(daemon):
    """First answer is (None, False); a later call returns the real hit with
    the local prefix-cache count subtracted — and the sync path stays the
    default (flag off = no resolver, no None)."""
    toks = list(range(500, 509))  # 2 blocks
    salt = "al-retry"
    seeder = make_conn(daemon)
    kv = fresh_kv()
    fill_block(kv, 2, seed=301)
    fill_block(kv, 5, seed=302)
    run_step(seeder, StubRequest("alr0", toks, salt), [2, 5], kv)
    seeder.shutdown()
    assert seeder._resolver is None  # flag off: the async machinery never woke

    conn = make_conn(daemon, kvblockd_async_lookup=True)
    req = StubRequest("alr1", toks, salt)
    first = conn.get_num_new_matched_tokens(req, 0)
    assert first == (None, False), "a fresh async lookup must answer None, ask-again"
    assert poll_lookup(conn, req) == 8
    assert "alr1" not in conn._lookup_pending  # resolved -> pruned

    # A resolved hit subtracts the local prefix-cache count, like sync does.
    req2 = StubRequest("alr2", toks, salt)
    assert conn.get_num_new_matched_tokens(req2, 4) == (None, False)
    assert poll_lookup(conn, req2, num_computed=4) == 4
    conn.shutdown()


def test_lookup_abort_cleanup(daemon):
    """request_finished must prune BOTH maps — a queued/aborted request that
    leaves a pending deadline or an unclaimed result behind is the leak class
    async lookups keep reintroducing upstream (vLLM #42372)."""
    toks = list(range(510, 519))
    conn = make_conn(daemon, kvblockd_async_lookup=True)
    reqs = [StubRequest(f"ab{i}", toks, "al-abort") for i in range(4)]
    for r in reqs:
        assert conn.get_num_new_matched_tokens(r, 0) == (None, False)
    assert len(conn._lookup_pending) == 4
    # Let every result land in the resolver's map, then abort ALL of them
    # before anyone claims a result.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        with conn._resolver._lock:
            if len(conn._resolver._results) == 4:
                break
        time.sleep(0.01)
    for r in reqs:
        conn.request_finished(r, [])
    assert conn._lookup_pending == {}
    with conn._resolver._lock:
        assert conn._resolver._results == {}
    conn.shutdown()


def test_lookup_bounded_map_and_deadline(daemon):
    """(a) An expired pending lookup answers (0, False) and is pruned; (b) at
    the pending-map cap a NEW lookup answers (0, False) — NEVER None, which
    with no queued work would park the request forever."""
    toks = list(range(520, 529))
    conn = make_conn(daemon, kvblockd_async_lookup=True,
                     kvblockd_lookup_timeout_s=0.05)
    fake = FakeResolver()  # never answers: deadline behavior is deterministic
    conn._resolver = fake

    req = StubRequest("bd1", toks, "al-deadline")
    assert conn.get_num_new_matched_tokens(req, 0) == (None, False)
    assert "bd1" in conn._lookup_pending
    assert conn.get_num_new_matched_tokens(req, 0) == (None, False)  # not expired yet
    time.sleep(0.08)
    assert conn.get_num_new_matched_tokens(req, 0) == (0, False)  # expired -> miss
    assert "bd1" not in conn._lookup_pending  # pruned

    now = time.monotonic()
    conn._lookup_pending = {f"fill{i}": now + 100.0 for i in range(_LOOKUP_PENDING_CAP)}
    posted_before = len(fake.posted)
    req2 = StubRequest("bd2", toks, "al-deadline")
    assert conn.get_num_new_matched_tokens(req2, 0) == (0, False)  # never None
    assert len(fake.posted) == posted_before  # nothing was queued at the cap
    assert "bd2" not in conn._lookup_pending
    conn._lookup_pending.clear()
    conn.shutdown()


def test_discard_before_post_leaves_no_orphan(daemon):
    """The orphan flavor test_lookup_abort_cleanup can't see: the lookup
    EXPIRES (pruned, answered miss) and the request FINISHES while the
    resolver is still ON THE WIRE — the late result used to squat in
    _results forever (nobody left to pop it). The discard now tombstones the
    in-flight rid and the resolver swallows the late result, leaving BOTH
    maps empty once it unblocks."""
    import threading

    toks = list(range(540, 549))
    conn = make_conn(daemon, kvblockd_async_lookup=True,
                     kvblockd_lookup_timeout_s=0.05)
    gate = threading.Event()

    class BlockedClient:
        """batch_exists parks on the gate — a daemon that answers late."""

        def batch_exists(self, keys):
            gate.wait(10.0)
            return len(keys), None

    conn._ensure = lambda: BlockedClient()
    req = StubRequest("orph1", toks, "al-orphan")
    assert conn.get_num_new_matched_tokens(req, 0) == (None, False)  # posted; on the wire
    time.sleep(0.08)
    assert conn.get_num_new_matched_tokens(req, 0) == (0, False)  # expired -> miss, pruned
    conn.request_finished(req, [])  # finished/aborted while still in flight
    gate.set()

    # Fence: the queue is FIFO, so once a lookup posted AFTER the orphan
    # resolves, the orphan's work has completed and reconciled.
    fence = StubRequest("orph-fence", toks, "al-orphan")
    assert conn.get_num_new_matched_tokens(fence, 0) == (None, False)
    assert poll_lookup(conn, fence) == 8
    with conn._resolver._lock:
        assert conn._resolver._results == {}, "late result orphaned after discard"
        assert conn._resolver._tombstones == set()  # consumed, never accumulated
    conn.shutdown()


def test_lookup_resolver_dead_falls_back_sync(daemon):
    """A dead resolver thread must not wedge scheduling: the lookup falls
    back to the inline synchronous answer (a real hit, not None/miss)."""
    toks = list(range(530, 539))
    salt = "al-dead"
    seeder = make_conn(daemon)
    kv = fresh_kv()
    fill_block(kv, 1, seed=303)
    fill_block(kv, 3, seed=304)
    run_step(seeder, StubRequest("ald0", toks, salt), [1, 3], kv)
    seeder.shutdown()

    conn = make_conn(daemon, kvblockd_async_lookup=True)
    conn._resolver = _LookupResolver(conn)
    conn._resolver.stop(2.0)  # thread exits via the sentinel
    assert not conn._resolver.alive()
    req = StubRequest("ald1", toks, salt)
    assert conn.get_num_new_matched_tokens(req, 0) == (8, False)  # inline sync
    conn.shutdown()
