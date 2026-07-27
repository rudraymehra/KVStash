"""Quality-wave suite (vllm package): scheduler-path deadlines, dial hygiene,
lookup-cache guards, hot-path reordering + chain-key memoization, config
validation, telemetry, and the write-behind drain fan-out (per-request worker
affinity with byte-exact accounting). Same conventions as the sibling suites:
stub vLLM surface, real daemon where bytes matter, fakes only to control
timing/failures."""

from __future__ import annotations

import logging
import threading
import time
import zlib
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from kvblockd import protocol as kp
from kvblockd.errors import ConnectionLost
from test_connector import BLOCK, NBLOCKS, StubRequest, StubVllmConfig
from test_tier_manager import BS, fake_spec, fake_view, job, make_manager, wait_jobs

from vllm_kvblockd import connector as conn_mod
from vllm_kvblockd import tier_manager as tm_mod
from vllm_kvblockd.config import AdapterConfig, block_chain_keys
from vllm_kvblockd.connector import (
    KvblockdConnector,
    KvblockdConnectorMetadata,
)
from vllm_kvblockd.tier_manager import KvblockdTierManager, LookupResult, _DualQueuePool


def make_conn(port=1, **extra):
    cfg = StubVllmConfig(port)
    cfg.kv_transfer_config.kv_connector_extra_config.update(extra)
    return KvblockdConnector(cfg, role="scheduler", kv_cache_config=None)


# ---------------------------------------------------------------------------
# commit 28: config validation + unknown-key disclosure
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("key", "value"), [
    ("kvblockd_op_timeout_s", 0),
    ("kvblockd_op_timeout_s", -1),
    ("kvblockd_connect_timeout_s", 0),
    ("kvblockd_streams", 0),
    ("kvblockd_store_queue_bytes", -1),
    ("kvblockd_store_drain_workers", 0),
])
def test_config_range_validation_names_the_key(key, value):
    cfg = StubVllmConfig(9440)
    cfg.kv_transfer_config.kv_connector_extra_config[key] = value
    with pytest.raises(ValueError, match=key):
        AdapterConfig.from_vllm_config(cfg)


def test_unknown_kvblockd_key_warns(caplog):
    caplog.set_level(logging.WARNING, logger="vllm_kvblockd")
    cfg = StubVllmConfig(9440)
    cfg.kv_transfer_config.kv_connector_extra_config["kvblockd_endpont"] = "typo:1"
    AdapterConfig.from_vllm_config(cfg)
    assert any("kvblockd_endpont" in r.getMessage() and "IGNORED" in r.getMessage()
               for r in caplog.records)
    # ...and only kvblockd_* keys are judged (other connectors share the dict).
    caplog.clear()
    cfg2 = StubVllmConfig(9440)
    cfg2.kv_transfer_config.kv_connector_extra_config["other_connector_knob"] = 1
    AdapterConfig.from_vllm_config(cfg2)
    assert not caplog.records


# ---------------------------------------------------------------------------
# commit 12: scheduler-side EXISTS budget
# ---------------------------------------------------------------------------
class _ExistsCapture:
    def __init__(self):
        self.deadlines: list[float | None] = []

    def batch_exists(self, keys, deadline=None):
        self.deadlines.append(deadline)
        return 0, None

    def close(self):
        pass


def test_sync_lookup_arms_the_exists_deadline():
    conn = make_conn()  # default kvblockd_exists_timeout_s = 0.25
    cap = _ExistsCapture()
    conn._client = cap
    conn.get_num_new_matched_tokens(StubRequest("ed1", list(range(9)), "qs-ed"), 0)
    assert cap.deadlines, "sync lookup never reached batch_exists"
    left = cap.deadlines[0] - time.monotonic()
    assert 0 < left <= 0.3, f"deadline not ~exists_timeout ahead ({left:.3f}s)"
    conn.shutdown()

    conn0 = make_conn(kvblockd_exists_timeout_s=0)  # <=0 disables (old behavior)
    cap0 = _ExistsCapture()
    conn0._client = cap0
    conn0.get_num_new_matched_tokens(StubRequest("ed2", list(range(9)), "qs-ed"), 0)
    assert cap0.deadlines == [None]
    conn0.shutdown()


def test_slow_dial_survives_the_exists_budget(monkeypatch):
    """The exists budget covers the EXCHANGE, not the dial: a healthy daemon
    whose dial+HELLO outlives exists_timeout_s must still answer a HIT. The
    t0-armed deadline raised 'deadline exceeded before checkout' after every
    SUCCESSFUL dial, dropped the freshly-primed client, and armed the 5s
    breaker — a self-sustaining permanent-0%-hit-rate loop against a healthy
    daemon (the dial has its own bounded budget: connect_timeout + breaker)."""
    et = 0.2

    class SlowDialHit:
        def __init__(self, *a, **kw):
            time.sleep(2 * et)  # one lost SYN / WAN link: dial >> exists budget

        def batch_exists(self, keys, deadline=None):
            # Mirror the real client's pre-checkout contract (client.py): an
            # already-expired deadline raises without touching the wire.
            if deadline is not None and deadline - time.monotonic() <= 0:
                raise ConnectionLost("EXISTS deadline exceeded before checkout")
            return len(keys), None

        def close(self):
            pass

    monkeypatch.setattr(conn_mod, "Client", SlowDialHit)
    conn = make_conn(kvblockd_exists_timeout_s=et)
    try:
        got, _ = conn.get_num_new_matched_tokens(
            StubRequest("sd1", list(range(9)), "qs-sd"), 0)
        assert got == 8, f"slow dial turned a full hit into {got} tokens"
        assert isinstance(conn._client, SlowDialHit), "healthy fresh client was dropped"
        assert time.monotonic() >= conn._next_dial, \
            "breaker armed against a healthy daemon"
    finally:
        conn.shutdown()


# ---------------------------------------------------------------------------
# commit 16: dial OUTSIDE the connector lock; tier breaker
# ---------------------------------------------------------------------------
def test_concurrent_ensure_degrades_instead_of_queueing(monkeypatch):
    """While one caller dials, every other _ensure must fail fast (miss) —
    never queue behind the connect_timeout on _client_lock."""
    dialing = threading.Event()
    release = threading.Event()

    class SlowDial:
        def __init__(self, *a, **kw):
            dialing.set()
            release.wait(5.0)
            raise ConnectionLost("dial failed after the slow window")

    monkeypatch.setattr(conn_mod, "Client", SlowDial)
    conn = make_conn()
    errs: list[BaseException] = []

    def dial_thread():
        try:
            conn._ensure()
        except Exception as e:  # noqa: BLE001 — collected for assertions
            errs.append(e)

    t = threading.Thread(target=dial_thread, daemon=True)
    t.start()
    assert dialing.wait(2.0)
    t0 = time.monotonic()
    with pytest.raises(ConnectionError, match="dial in progress"):
        conn._ensure()
    assert time.monotonic() - t0 < 0.5, "second caller queued behind the dial"
    release.set()
    t.join(5.0)
    assert len(errs) == 1
    # The failed dial armed the breaker: instant suppression, no re-dial.
    with pytest.raises(ConnectionError, match="suppressed"):
        conn._ensure()
    conn.shutdown()


def test_tier_dial_breaker_fails_fast(monkeypatch):
    calls = []

    class BoomClient:
        def __init__(self, *a, **kw):
            calls.append(1)
            raise ConnectionLost("injected dial failure")

    monkeypatch.setattr(tm_mod, "Client", BoomClient)
    view, _ = fake_view()
    mgr = KvblockdTierManager(fake_spec(), view, "kvblockd",
                              endpoint="kvblockd://127.0.0.1:1",
                              n_read_threads=1, n_write_threads=1, streams=1)
    try:
        with pytest.raises(ConnectionLost):
            mgr._ensure()
        assert len(calls) == 1
        with pytest.raises(ConnectionError, match="suppressed"):
            mgr._ensure()
        assert len(calls) == 1, "breaker window must suppress the re-dial"
        assert mgr.stats_snapshot()["dial_failures"] == 1
    finally:
        mgr.shutdown()


# ---------------------------------------------------------------------------
# commit 13: coherent tier sizing
# ---------------------------------------------------------------------------
def test_tier_streams_default_ties_to_worker_count():
    view, _ = fake_view()
    mgr = KvblockdTierManager(fake_spec(), view, "kvblockd",
                              endpoint="kvblockd://127.0.0.1:1",
                              n_read_threads=2, n_write_threads=3)
    try:
        assert mgr._streams == 5  # n_read + n_write: one source of truth
    finally:
        mgr.shutdown()
    view2, _ = fake_view()
    mgr2 = KvblockdTierManager(fake_spec(), view2, "kvblockd",
                               endpoint="kvblockd://127.0.0.1:1",
                               n_read_threads=2, n_write_threads=3, streams=2)
    try:
        assert mgr2._streams == 2  # explicit value honored unchanged
    finally:
        mgr2.shutdown()


def test_tier_client_takes_the_inline_get_path(daemon):
    """Pool workers already provide the parallelism: per-tile loads must take
    the nshards<=1 inline path (get_fanout=1), not fan 8-key tiles out."""
    view, _ = fake_view()
    mgr = make_manager(daemon, view)
    try:
        assert mgr._ensure()._get_fanout == 1
    finally:
        mgr.shutdown()


# ---------------------------------------------------------------------------
# commit 12 (tier): per-load deadline + bounded wait_idle
# ---------------------------------------------------------------------------
class _TierCapture:
    def __init__(self, block_size):
        self._bs = block_size
        self.deadlines: list[float | None] = []

    def batch_get_scatter(self, wire, prefix_len, alloc, deadline=None):
        self.deadlines.append(deadline)
        out = []
        for i in range(len(wire)):
            mv = alloc(i, b"", self._bs)
            if mv is None:
                out.append(kp.Status.NOT_FOUND)
                continue
            mv[:] = bytes(self._bs)
            out.append(kp.Status.OK)
        return out

    def close(self):
        pass


def test_submit_load_passes_a_deadline():
    view, _ = fake_view()
    mgr = KvblockdTierManager(fake_spec(), view, "kvblockd",
                              endpoint="kvblockd://127.0.0.1:1",
                              n_read_threads=1, n_write_threads=1, streams=1)
    try:
        cap = _TierCapture(BS)
        mgr._client = cap
        mgr.submit_load(job(41, [b"\x07" * 32 + b"\x00" * 4], [3]))
        assert wait_jobs(mgr, {41}) == {41: True}
        assert cap.deadlines and cap.deadlines[0] is not None
        left = cap.deadlines[0] - time.monotonic()
        assert 20 < left <= 31, f"deadline not ~load_deadline_s ahead ({left:.1f}s)"
    finally:
        mgr.shutdown()

    view2, _ = fake_view()
    mgr2 = KvblockdTierManager(fake_spec(), view2, "kvblockd",
                               endpoint="kvblockd://127.0.0.1:1",
                               n_read_threads=1, n_write_threads=1, streams=1,
                               load_deadline_s=0)  # <=0 disables: old behavior
    try:
        cap2 = _TierCapture(BS)
        mgr2._client = cap2
        mgr2.submit_load(job(42, [b"\x08" * 32 + b"\x00" * 4], [3]))
        assert wait_jobs(mgr2, {42}) == {42: True}
        assert cap2.deadlines == [None]
    finally:
        mgr2.shutdown()


def test_wait_idle_survives_cross_job_queueing():
    """2 jobs, 1 worker, every task inside its own from-start budget: bounds
    stamped at ENQUEUE time ignore cross-job queueing on the shared pool, so
    the old wait_idle returned while job B's healthy task was still copying
    into the primary memoryview — the framework then unpins/reuses those
    slots under a live recv_into (torn KV bytes, the corruption class the
    drain_jobs contract forbids). wait_idle must wait for BOTH."""
    pool = _DualQueuePool(1, 0, name="t_xjob")
    done = []

    def work():
        time.sleep(0.4)
        done.append(1)

    pool.enqueue_load(53, 1, [work], bound_s=0.5)
    pool.enqueue_load(54, 1, [work], bound_s=0.5)
    t0 = time.monotonic()
    pool.wait_idle()
    elapsed = time.monotonic() - t0
    assert len(done) == 2, \
        f"wait_idle returned at {elapsed:.2f}s with a healthy in-budget task running"
    assert dict(pool.get_finished()) == {53: True, 54: True}
    pool.shutdown()


def test_wait_idle_watchdog_returns_on_a_wedged_task():
    """A RUNNING task past its own START-relative budget is the client bug
    the armor targets (every started task is wire-deadline-bounded, so zero
    progress for a whole budget is genuine): return loudly instead of
    wedging the scheduler process."""
    pool = _DualQueuePool(1, 0, name="t_wedge")
    gate = threading.Event()
    started = threading.Event()

    def wedged():
        started.set()
        gate.wait(10.0)

    pool.enqueue_load(51, 1, [wedged], bound_s=0.2)
    assert started.wait(2.0)
    t0 = time.monotonic()
    pool.wait_idle()  # must RETURN once the RUNNING task outlives its budget
    assert time.monotonic() - t0 < 2.0, "wait_idle ignored the progress watchdog"
    gate.set()
    pool.shutdown()


def test_wait_idle_sheds_only_queued_tasks_past_the_job_bound():
    """A job whose wall-clock bound expires while its tasks are still QUEUED
    fails those tasks (never started -> no partial copy -> safe to fail; the
    framework recomputes) — but a RUNNING in-budget task is never abandoned."""
    pool = _DualQueuePool(1, 0, name="t_shed")
    ran = threading.Event()

    def slow():
        time.sleep(0.5)

    pool.enqueue_load(55, 1, [slow], bound_s=5.0)
    pool.enqueue_load(56, 1, [ran.set], bound_s=0.15)  # queued behind 55
    t0 = time.monotonic()
    pool.wait_idle()
    elapsed = time.monotonic() - t0
    got = dict(pool.get_finished())
    assert got == {55: True, 56: False}, got
    assert not ran.is_set(), "a shed task must never have started"
    assert elapsed >= 0.45, \
        f"returned at {elapsed:.2f}s, before the RUNNING task finished"
    pool.shutdown()


def test_wait_idle_unbounded_job_keeps_the_unbounded_wait():
    # Unbounded jobs keep today's unbounded wait (stores).
    pool2 = _DualQueuePool(1, 0, name="t_unbound")
    gate2 = threading.Event()
    started2 = threading.Event()

    def wedged2():
        started2.set()
        gate2.wait(10.0)

    pool2.enqueue_load(52, 1, [wedged2])  # no bound
    assert started2.wait(2.0)
    waiter = threading.Thread(target=pool2.wait_idle, daemon=True)
    waiter.start()
    waiter.join(0.3)
    assert waiter.is_alive(), "unbounded job must keep the unbounded wait"
    gate2.set()
    waiter.join(5.0)
    assert not waiter.is_alive()
    pool2.shutdown()


# ---------------------------------------------------------------------------
# commit 17: pairing guards + unknown-sentinel lookup cache
# ---------------------------------------------------------------------------
def test_length_mismatch_fails_the_job_loudly(caplog):
    caplog.set_level(logging.ERROR, logger="vllm_kvblockd")
    view, _ = fake_view()
    mgr = KvblockdTierManager(fake_spec(), view, "kvblockd",
                              endpoint="kvblockd://127.0.0.1:1",
                              n_read_threads=1, n_write_threads=1, streams=1)
    try:
        k = b"\x09" * 32 + b"\x00" * 4
        mgr.submit_store(job(61, [k, k], [0]))       # 2 keys vs 1 block_id
        mgr.submit_load(job(62, [k], [0, 1]))        # 1 key vs 2 block_ids
        got = wait_jobs(mgr, {61, 62})
        assert got == {61: False, 62: False}
        snap = mgr.stats_snapshot()
        assert snap["stores_failed"] >= 1 and snap["loads_failed"] >= 1
        assert any("never truncate" in r.getMessage() for r in caplog.records)
    finally:
        mgr.shutdown()


def test_out_of_range_block_id_fails_before_wire(daemon):
    view, backing = fake_view()
    mgr = make_manager(daemon, view)
    try:
        k = b"\x0a" * 32 + b"\x00" * 4
        mgr.submit_store(job(63, [k], [NBLOCKS + 700]))  # far past the view
        assert wait_jobs(mgr, {63}) == {63: False}
        # Nothing was stored under the key (no short PUT reached the wire).
        ctx = SimpleNamespace(req_id="oob-req")
        from test_tier_manager import poll_lookup

        assert poll_lookup(mgr, k, ctx) == LookupResult.MISS
        mgr.submit_load(job(64, [k], [NBLOCKS + 700]))
        assert wait_jobs(mgr, {64}) == {64: False}
        assert bytes(backing) == bytes(len(backing))  # view untouched
    finally:
        mgr.shutdown()


def test_lookup_blip_is_unknown_not_poisoned_miss(daemon, monkeypatch):
    """A transient EXISTS failure must answer MISS (fail-open) but re-query —
    the old definitive all-False poisoned the key to MISS for the request's
    lifetime even after the daemon recovered."""
    # Injectable backoff: the blip's _drop_client arms the tier breaker for
    # _REDIAL_BACKOFF_S (5.0), which raced the HIT-recovery loop's 5.0s
    # deadline with only tens of ms of margin — nondeterministic red.
    monkeypatch.setattr(tm_mod, "_REDIAL_BACKOFF_S", 0.2)
    view, backing = fake_view()
    mgr = make_manager(daemon, view)
    try:
        k = b"\x0b" * 32 + b"\x00" * 4
        backing[0:BS] = bytes([5]) * BS
        mgr.submit_store(job(71, [k], [0]))
        wait_jobs(mgr, {71})

        real_ensure = mgr._ensure
        state = {"failed": 0}

        def flaky():
            if state["failed"] == 0:
                state["failed"] = 1
                raise ConnectionLost("injected blip")
            return real_ensure()

        mgr._ensure = flaky
        ctx = SimpleNamespace(req_id="blip-req")
        assert mgr.lookup(k, ctx) == LookupResult.RETRY  # first sight
        # Drive until the FAILED round-trip resolves: fail-open MISS.
        deadline = time.time() + 5.0
        r = LookupResult.RETRY
        while time.time() < deadline and r == LookupResult.RETRY:
            mgr.on_schedule_end(None)
            time.sleep(0.02)
            r = mgr.lookup(k, ctx)
        assert r == LookupResult.MISS  # fail-open, never a parked RETRY
        assert state["failed"] == 1  # the blip actually happened
        # The SAME key must now be re-queried and upgrade to HIT.
        deadline = time.time() + 5.0
        while time.time() < deadline and r != LookupResult.HIT:
            mgr.on_schedule_end(None)
            time.sleep(0.02)
            r = mgr.lookup(k, ctx)
        assert r == LookupResult.HIT, "blip permanently poisoned the key to MISS"
    finally:
        mgr.shutdown()


# ---------------------------------------------------------------------------
# commit 18: decode-step bail before the O(context) copy; chain-key memo
# ---------------------------------------------------------------------------
class _CountingRequest:
    def __init__(self, rid, n_prompt):
        self.request_id = rid
        self.num_prompt_tokens = n_prompt
        self.cache_salt = None
        self.mm_features = []
        self.lora_request = None
        self.accesses = 0

    @property
    def all_token_ids(self):
        self.accesses += 1
        return list(range(self.num_prompt_tokens))


def test_decode_step_never_materializes_all_token_ids():
    conn = make_conn()
    req = _CountingRequest("dm1", 9)  # aligned = 8
    conn._inflight["dm1"] = req
    conn._blocks["dm1"] = [0, 1]
    meta = KvblockdConnectorMetadata()
    # Pure decode step: computed(8) >= aligned(8) -> bail BEFORE the copy.
    conn._build_cached_req_meta(meta, "dm1", 0, set(), [8], [None], {})
    assert meta.requests == []
    assert req.accesses == 0, "decode step copied all_token_ids for nothing"
    # A store-emitting chunk still pays exactly one materialization.
    meta2 = KvblockdConnectorMetadata()
    conn._build_cached_req_meta(meta2, "dm1", 0, set(), [4], [None], {"dm1": 4})
    assert len(meta2.requests) == 1
    assert req.accesses == 1
    conn.shutdown()


def test_chain_key_cache_extends_exactly(monkeypatch):
    conn = make_conn()
    seed = conn._seed("qs-chain", [], "")
    toks = list(range(1000, 1024))  # 24 tokens = 6 blocks of 4
    expect = block_chain_keys(seed, toks, BLOCK)

    hashed_tokens = []
    real = block_chain_keys

    def counting(seed_, token_ids, block_size):
        hashed_tokens.append(len(token_ids))
        return real(seed_, token_ids, block_size)

    monkeypatch.setattr(conn_mod, "block_chain_keys", counting)
    first = conn._chain_keys("cc1", seed, toks[:16])   # chunk 1: 4 blocks
    second = conn._chain_keys("cc1", seed, toks)       # chunk 2: +2 blocks
    assert second == expect, "extension diverged from the full recompute"
    assert first == expect[:4]
    assert hashed_tokens == [16, 8], f"re-hashed the whole prompt: {hashed_tokens}"
    # A different seed for the same rid must fall back to a full recompute.
    other_seed = conn._seed("qs-chain-B", [], "")
    third = conn._chain_keys("cc1", other_seed, toks)
    assert third == block_chain_keys(other_seed, toks, BLOCK)
    conn.shutdown()


# ---------------------------------------------------------------------------
# item #9: write-behind drain fan-out (per-request affinity, exact accounting)
# ---------------------------------------------------------------------------
class _RecordingClient:
    def __init__(self):
        self._lock = threading.Lock()
        self.puts: list[tuple[str, bytes]] = []  # (thread name, key)

    def put(self, key, bufs, ttl_ms=0):
        with self._lock:
            self.puts.append((threading.current_thread().name, bytes(key)))
        return 0

    def close(self):
        pass


def test_drain_worker_count_defaults_and_clamps():
    conn = make_conn()
    assert conn._store_workers == 1 and len(conn._sqs) == 1  # today's behavior
    conn.shutdown()
    conn2 = make_conn(kvblockd_store_drain_workers=8, kvblockd_streams=2)
    assert conn2._store_workers == 2, "workers must clamp to streams"
    conn2.shutdown()


def test_fanout_preserves_per_request_order_and_affinity():
    conn = make_conn(kvblockd_store_drain_workers=3, kvblockd_streams=4)
    assert conn._store_workers == 3
    conn._store_thread_start = lambda: None  # stage only (pause the drain)
    # rids chosen to hash to three DISTINCT workers (crc32 mod 3: 1, 0, 2).
    keys = {"req-A": [bytes([1, i]) + b"\x00" * 30 for i in range(6)],
            "r1": [bytes([2, i]) + b"\x00" * 30 for i in range(6)],
            "r2": [bytes([3, i]) + b"\x00" * 30 for i in range(6)]}
    for i in range(6):  # interleave the three requests
        for rid in keys:
            assert conn._sq_enqueue(keys[rid][i], bytearray(b"x" * 64), rid=rid)
    # AFFINITY: each request's blocks all landed in ONE worker's deque, in
    # exact enqueue order (the property that keeps a partial delivery a
    # usable consecutive prefix).
    for rid, ks in keys.items():
        wi = zlib.crc32(rid.encode()) % conn._store_workers
        staged = [bytes(k) for k, _b, _s in conn._sqs[wi] if bytes(k) in set(ks)]
        assert staged == ks, f"{rid}: order/affinity broken in worker {wi}"
    # Drain with 3 REAL worker threads against a recording client: every blob
    # delivered, each request served by exactly one thread, in block order.
    fake = _RecordingClient()
    conn._client = fake
    del conn._store_thread_start
    conn._store_thread_start()
    assert conn._store_flush(10.0) == 0
    assert conn.dropped_puts == 0 and conn.failed_puts == 0
    for rid, ks in keys.items():
        rows = [(t, k) for t, k in fake.puts if k in set(ks)]
        assert [k for _t, k in rows] == ks, f"{rid}: delivery order broken"
        assert len({t for t, _k in rows}) == 1, f"{rid}: served by multiple workers"
    assert len({t for t, _k in fake.puts}) == 3, "3 workers should all have served"
    conn.shutdown()


def test_dial_pending_is_a_park_not_a_second_strike(monkeypatch):
    """A requeued store block that wakes into another caller's IN-FLIGHT dial
    must PARK (requeue, no strike): charging the dial-pending raise as the
    second put failure counted failed_puts += 1 and freed the slot without a
    second wire attempt — a block permanently lost and a failed_puts line
    inflated with a non-attempt (disclosure exactness, roadmap item #9)."""
    monkeypatch.setattr(conn_mod, "_REDIAL_BACKOFF_S", 0.15)
    conn = make_conn()
    conn._store_thread_start = lambda: None  # stage first, drain later

    class BlipsAlways:
        def __init__(self):
            self.calls = 0

        def put(self, key, bufs, ttl_ms=0):
            self.calls += 1
            raise ConnectionLost("injected blip")

        def close(self):
            pass

    bad = BlipsAlways()
    good = _RecordingClient()
    conn._client = bad
    assert conn._sq_enqueue(b"\x0c" * 32, bytearray(b"y" * 64), rid="park-req")
    with conn._client_lock:
        conn._dialing = True  # a concurrent caller owns the dial at wake time
    del conn._store_thread_start
    conn._store_thread_start()
    # Blip -> strike 1 -> requeue -> breaker park (0.15s); at wake _ensure
    # sees the in-flight dial. Give the drain time to hit BOTH edges.
    time.sleep(0.6)
    assert bad.calls == 1, "the first (real) attempt should have happened exactly once"
    assert conn.failed_puts == 0, "dial-pending was charged as a second strike"
    with conn._client_lock:  # ...the dial lands on a healthy daemon
        conn._dialing = False
        conn._client = good
    assert conn._store_flush(5.0) == 0
    assert conn.failed_puts == 0 and conn.dropped_puts == 0
    assert [k for _t, k in good.puts] == [b"\x0c" * 32], \
        "the parked block must be delivered after the dial lands"
    conn.shutdown()


def test_fanout_shutdown_disclosure_is_aggregate_exact(caplog):
    caplog.set_level(logging.WARNING, logger="vllm_kvblockd")
    conn = make_conn(kvblockd_store_drain_workers=2, kvblockd_streams=2,
                     kvblockd_store_flush_timeout_s=0.2)
    conn._store_thread_start = lambda: None  # nothing ever drains
    total_bytes = 0
    for i, rid in enumerate(("sd-A", "sd-B", "sd-C", "sd-D")):
        buf = bytearray(bytes([i]) * 96)
        total_bytes += len(buf)
        assert conn._sq_enqueue(bytes([9, i]) + b"\x00" * 30, buf, rid=rid)
    assert sum(len(q) for q in conn._sqs) == 4
    conn.shutdown()
    assert conn.dropped_puts == 4
    assert conn.dropped_put_bytes == total_bytes
    lines = [r for r in caplog.records if "kvblockd store queue:" in r.getMessage()]
    assert len(lines) == 1 and "dropped=4" in lines[0].getMessage()


# ---------------------------------------------------------------------------
# commit 15: periodic stats summary + snapshot
# ---------------------------------------------------------------------------
def test_stats_summary_line_emits(caplog):
    caplog.set_level(logging.WARNING, logger="vllm_kvblockd")
    conn = make_conn()
    conn._client = _ExistsCapture()
    conn._stats_last_emit = time.monotonic() - 3600  # force the interval
    conn.get_num_new_matched_tokens(StubRequest("st1", list(range(9)), "qs-st"), 0)
    assert any("kvblockd stats:" in r.getMessage() for r in caplog.records)
    snap = conn.stats.snapshot()
    assert snap["misses"] >= 1 and snap["lookup_count"] >= 1
    conn.shutdown()
