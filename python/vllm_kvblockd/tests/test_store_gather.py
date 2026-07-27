"""Gathered-store fast path (CUDA stores staged via batched index_select +
async D2H into pinned slots, one device sync before any enqueue).

CI runs CPU-only torch, so — exactly like test_slab_scatter — the CUDA-only
dispatch is forced through its seams (`_slab_path_ok`, `_alloc_pinned`,
`_store_sync`) while the chunk math, the slot pool, the deferred enqueue, the
tail-skip accounting, and the degrade ladders are all the REAL code. The
oracle everywhere is the original bytearray path: same paged tensors, same
keys, byte-identical blobs, byte-identical drop trace."""

from __future__ import annotations

import logging
import random
import threading
import time

import pytest

torch = pytest.importorskip("torch")

from test_async_store import TOTAL, FakeClient, pause_drain, resume_drain, wait_until
from test_connector import BLOCK, HID, LAYERS, StubVllmConfig, fill_block

from vllm_kvblockd import connector as conn_mod
from vllm_kvblockd.config import AdapterConfig
from vllm_kvblockd.connector import (
    KvblockdConnector,
    KvblockdConnectorMetadata,
    KvbReqMeta,
)


def make_conn(**extra) -> KvblockdConnector:
    """A connector that never dials (port 1) — staging never touches the wire."""
    cfg = StubVllmConfig(1)
    cfg.kv_transfer_config.kv_connector_extra_config.update(extra)
    return KvblockdConnector(cfg, role="scheduler", kv_cache_config=None)


def plain_alloc(n):
    # CPU stand-in for the pinned allocation (CI has no CUDA allocator).
    return torch.empty(n, dtype=torch.uint8)


def gather_conn(**extra) -> KvblockdConnector:
    conn = make_conn(**extra)
    conn._slab_path_ok = lambda dev: True
    conn._alloc_pinned = plain_alloc
    return conn


def big_kv(num_paged_blocks: int):
    return {n: torch.zeros(num_paged_blocks, 2, BLOCK, HID, dtype=torch.bfloat16)
            for n in LAYERS}


def smeta(rid, toks, bids, start=0, end=None, salt=None) -> KvbReqMeta:
    if end is None:
        end = len(bids)
    return KvbReqMeta(req_id=rid, token_ids=list(toks), cache_salt=salt, mm_ids=[],
                      lora_name="", block_ids=list(bids), load_start_block=0,
                      num_load_blocks=0, store_start_block=start, store_end_block=end)


def stage(conn, metas, kv) -> None:
    """One worker-side store step: capture layers, bind meta, wait_for_save."""
    for name, t in kv.items():
        conn.save_kv_layer(name, t, None)
    conn.bind_connector_metadata(KvblockdConnectorMetadata(requests=list(metas)))
    conn.wait_for_save()
    conn.clear_connector_metadata()


def queue_snapshot(conn):
    with conn._sq_cond:
        return [(bytes(k), bytes(b)) for k, b, _s in conn._sq]


def slots_free(conn) -> int:
    with conn._sq_cond:
        return len(conn._store_slot_free)


def assert_slot_invariant(conn) -> None:
    """free + queued == total (call only when nothing is staging/in flight)."""
    with conn._sq_cond:
        queued = sum(1 for _k, _b, s in conn._sq if s is not None)
        assert len(conn._store_slot_free) + queued == conn._store_slots_total


# ------------------------------------------------------------------- config

def test_store_staging_bytes_knob_parses():
    """Unset -> None (auto-size at first alloc); 0 -> explicitly off;
    an explicit value overrides the auto-size."""
    cfg = StubVllmConfig(1)
    assert AdapterConfig.from_vllm_config(cfg).store_staging_bytes is None
    cfg.kv_transfer_config.kv_connector_extra_config["kvblockd_store_staging_bytes"] = 0
    assert AdapterConfig.from_vllm_config(cfg).store_staging_bytes == 0
    cfg.kv_transfer_config.kv_connector_extra_config["kvblockd_store_staging_bytes"] = 12345
    assert AdapterConfig.from_vllm_config(cfg).store_staging_bytes == 12345


# ---------------------------------------------------------- byte identity

def test_gather_store_byte_identical_and_never_walks_blocks(caplog):
    """RED-PROOF: 70 blocks (2 gather chunks + tail) through the fast path
    enqueue (key, bytes) pairs byte-identical to the bytearray path, WITHOUT
    a single _block_bytes call (the old code calls it layers x blocks), and
    stamp the store path once."""
    nblk, npaged = 70, 80
    toks = list(range(1000, 1000 + nblk * BLOCK))
    bids = random.Random(7).sample(range(npaged), nblk)
    kv = big_kv(npaged)
    for b in bids:
        fill_block(kv, b, seed=9000 + b)
    budget = npaged * TOTAL  # no drops in this test

    old = make_conn(kvblockd_store_queue_bytes=budget)
    pause_drain(old)
    stage(old, [smeta("bd", toks, bids, salt="sg-bd")], kv)
    old_items = queue_snapshot(old)
    assert len(old_items) == nblk

    new = gather_conn(kvblockd_store_queue_bytes=budget)
    pause_drain(new)
    walks = []
    orig_bb = new._block_bytes
    new._block_bytes = lambda t: (walks.append(1), orig_bb(t))[1]
    caplog.clear()  # the oracle conn stamped its own (bytearray) path above
    with caplog.at_level(logging.WARNING, logger="vllm_kvblockd"):
        stage(new, [smeta("bd", toks, bids, salt="sg-bd")], kv)
    assert walks == [], "fast path must never walk blocks through _block_bytes"
    assert queue_snapshot(new) == old_items
    assert_slot_invariant(new)
    assert ["kvblockd store path: gathered-slots"] == [
        r.getMessage() for r in caplog.records
        if r.getMessage().startswith("kvblockd store path:")]
    old.shutdown()
    new.shutdown()


def test_enqueue_deferred_until_after_device_sync():
    """RED-PROOF (correctness invariant, not tuning): on the fast path every
    enqueue happens only AFTER the one device sync — a torn D2H must never be
    published under a content-chained key."""
    conn = gather_conn(kvblockd_store_queue_bytes=8 * TOTAL)
    pause_drain(conn)
    events = []
    real_sync = conn._store_sync
    conn._store_sync = lambda dev: (events.append("sync"), real_sync(dev))[1]
    real_enq = conn._sq_enqueue

    def spy_enq(key, buf, slot_id=None):
        events.append("enqueue")
        return real_enq(key, buf, slot_id)

    conn._sq_enqueue = spy_enq
    kv = big_kv(8)
    fill_block(kv, 0, seed=100)
    fill_block(kv, 1, seed=101)
    stage(conn, [smeta("sy", list(range(500, 508)), [0, 1], salt="sg-sync")], kv)
    assert events == ["sync", "enqueue", "enqueue"]
    conn.shutdown()


# --------------------------------------------------------- slot lifecycle

def test_shutdown_frees_every_queued_slot():
    conn = gather_conn(kvblockd_store_queue_bytes=8 * TOTAL)  # auto: 8+2 slots
    pause_drain(conn)
    kv = big_kv(8)
    for b in (0, 1, 2):
        fill_block(kv, b, seed=110 + b)
    stage(conn, [smeta("sd", list(range(600, 612)), [0, 1, 2], salt="sg-shut")], kv)
    assert conn._store_slots_total == 10
    assert slots_free(conn) == 7
    assert_slot_invariant(conn)
    conn.shutdown()  # drain never started: remainder disclosed dropped
    assert conn.dropped_puts == 3
    assert slots_free(conn) == 10  # every lease came home before clear()


def test_delivered_and_requeued_slots_return_at_reconcile(monkeypatch):
    """Delivery frees the slot at the drain's reconcile block; a requeue KEEPS
    the lease and the post-backoff retry both delivers and frees it."""
    monkeypatch.setattr(conn_mod, "_REDIAL_BACKOFF_S", 0.05)
    conn = gather_conn(kvblockd_store_queue_bytes=8 * TOTAL)
    fake = FakeClient(fail_first=1)
    conn._ensure = lambda: fake
    kv = big_kv(8)
    fill_block(kv, 3, seed=120)
    stage(conn, [smeta("rc", list(range(700, 704)), [3], salt="sg-rec")], kv)
    assert conn._store_flush(10.0) == 0
    assert conn.failed_puts == 0 and len(fake.puts) == 1
    assert slots_free(conn) == conn._store_slots_total
    assert_slot_invariant(conn)
    conn.shutdown()


def test_slot_exhaustion_falls_back_per_block_no_drop_no_latch():
    """Pool exhaustion is congestion, not failure: blocks past the pool take
    the bytearray path IN ORDER — nothing dropped, nothing latched — and the
    blob bytes stay identical to the pure bytearray path."""
    kv = big_kv(8)
    for b in (0, 1, 2):
        fill_block(kv, b, seed=130 + b)
    toks = list(range(800, 812))

    old = make_conn(kvblockd_store_queue_bytes=8 * TOTAL)
    pause_drain(old)
    stage(old, [smeta("ex", toks, [0, 1, 2], salt="sg-exh")], kv)

    conn = gather_conn(kvblockd_store_queue_bytes=8 * TOTAL,
                       kvblockd_store_staging_bytes=TOTAL)  # exactly ONE slot
    pause_drain(conn)
    stage(conn, [smeta("ex", toks, [0, 1, 2], salt="sg-exh")], kv)
    with conn._sq_cond:
        slot_ids = [s for _k, _b, s in conn._sq]
    assert slot_ids[0] is not None and slot_ids[1:] == [None, None]
    assert queue_snapshot(conn) == queue_snapshot(old)
    assert conn.dropped_puts == 0 and conn.dropped_put_bytes == 0
    assert conn._store_gather_fails == 0 and not conn._store_gather_disabled
    old.shutdown()
    conn.shutdown()
    assert slots_free(conn) == 1


def test_pool_disabled_by_zero_and_alloc_failure_latches_once():
    kv = big_kv(8)
    fill_block(kv, 0, seed=140)
    toks = list(range(900, 904))

    off = gather_conn(kvblockd_store_queue_bytes=8 * TOTAL,
                      kvblockd_store_staging_bytes=0)
    pause_drain(off)
    stage(off, [smeta("z", toks, [0], salt="sg-off")], kv)
    assert len(off._sq) == 1 and off._store_slab is None
    assert not off._store_slab_disabled  # explicitly off is not a failure
    off.shutdown()

    boom_calls = []

    def boom(n):
        boom_calls.append(n)
        raise RuntimeError("cudaHostAlloc failed")

    latch = gather_conn(kvblockd_store_queue_bytes=8 * TOTAL)
    latch._alloc_pinned = boom
    pause_drain(latch)
    stage(latch, [smeta("z", toks, [0], salt="sg-latch")], kv)
    stage(latch, [smeta("z2", toks, [0], salt="sg-latch2")], kv)
    assert latch._store_slab_disabled and len(boom_calls) == 1  # no retry storm
    assert len(latch._sq) == 2  # bytearray staging kept serving
    latch.shutdown()


# ------------------------------------------------- drop-trace equivalence

def drop_schedule_trace(conn):
    """Budget fits 2 of 4 blocks: enqueue b0,b1; b2 refused; b3 tail-skipped."""
    kv = big_kv(8)
    for b in range(4):
        fill_block(kv, b, seed=150 + b)
    pause_drain(conn)
    stage(conn, [smeta("dt", list(range(1100, 1116)), [0, 1, 2, 3], salt="sg-drop")], kv)
    snap = queue_snapshot(conn)
    return (conn.dropped_puts, conn.dropped_put_bytes, conn.failed_puts,
            dict(conn._store_holes), snap)


def test_drop_trace_byte_identical_under_budget_pressure():
    """Pool >= queue budget (auto-size) => the fast path's refusals, tail
    skips, hole records, and queued bytes are byte-identical to the old path."""
    budget = 2 * TOTAL + TOTAL // 2
    old = make_conn(kvblockd_store_queue_bytes=budget)
    new = gather_conn(kvblockd_store_queue_bytes=budget)
    old_trace = drop_schedule_trace(old)
    new_trace = drop_schedule_trace(new)
    assert new_trace == old_trace
    assert old_trace[0] == 2 and old_trace[3] == {"dt": 2}  # the schedule bit
    assert_slot_invariant(new)
    old.shutdown()
    new.shutdown()
    assert slots_free(new) == new._store_slots_total


def requeue_schedule_trace(conn):
    """OPEN-1's worst schedule: queue at byte budget, one item requeued at
    the head (its slot lease KEPT), a fresh stage arriving mid-backoff."""
    fake = FakeClient(fail_first=1)
    conn._ensure = lambda: fake
    pause_drain(conn)
    kv = big_kv(8)
    for b in (0, 1, 2):
        fill_block(kv, b, seed=160 + b)
    stage(conn, [smeta("rq-a", list(range(1200, 1208)), [0, 1], salt="sg-rq-a")], kv)
    with conn._sq_cond:
        assert len(conn._sq) == 2  # queue exactly at budget
    resume_drain(conn)
    # Requeued state: first put failed, the item is back at the head with its
    # lease, the drain is sitting out the redial backoff.
    assert wait_until(lambda: fake.fail_first == 0 and len(conn._sq) == 2
                      and conn._sq_inflight == 0)
    stage(conn, [smeta("rq-b", list(range(1300, 1304)), [2], salt="sg-rq-b")], kv)
    assert conn._store_flush(10.0) == 0
    return (conn.dropped_puts, conn.dropped_put_bytes, conn.failed_puts,
            [k for k, _ in fake.puts])


def test_requeue_at_full_queue_keeps_trace_identical_without_fallback(monkeypatch):
    """OPEN-1 settled by test: with auto-size (budget + 2 slots) the requeue-
    during-full-queue schedule leases every slot it needs (no bytearray
    fallback) and the drop trace matches the old path byte-for-byte."""
    monkeypatch.setattr(conn_mod, "_REDIAL_BACKOFF_S", 1.0)
    old = make_conn(kvblockd_store_queue_bytes=2 * TOTAL)
    old_trace = requeue_schedule_trace(old)
    old.shutdown()

    new = gather_conn(kvblockd_store_queue_bytes=2 * TOTAL)  # auto: 2+2 slots
    lease_misses = []
    real_lease = new._store_slot_lease

    def spy_lease():
        s = real_lease()
        if s is None:
            lease_misses.append(1)
        return s

    new._store_slot_lease = spy_lease
    new_trace = requeue_schedule_trace(new)
    assert new_trace == old_trace
    assert old_trace[0] == 1 and old_trace[2] == 0  # b dropped; a0 retried fine
    assert lease_misses == [], "auto-size headroom must cover the worst schedule"
    assert slots_free(new) == new._store_slots_total == 4
    new.shutdown()


# --------------------------------------------------------- degrade ladders

def test_three_gather_failures_latch_off_with_identical_output():
    """3 consecutive gather-setup failures latch the fast path OFF for the
    connector's lifetime; every step's blobs still land via the bytearray
    path, byte-identical, with zero drops and zero slot leaks."""
    kv = big_kv(8)
    for b in (0, 1):
        fill_block(kv, b, seed=170 + b)
    toks = list(range(1400, 1408))

    old = make_conn(kvblockd_store_queue_bytes=64 * TOTAL)
    pause_drain(old)
    for i in range(4):
        stage(old, [smeta(f"lt{i}", toks, [0, 1], salt=f"sg-latch3-{i}")], kv)
    old_items = queue_snapshot(old)

    conn = gather_conn(kvblockd_store_queue_bytes=64 * TOTAL)
    pause_drain(conn)
    boom_calls = []

    def boom(dev, n_layers, bytes_per_layer):
        boom_calls.append(1)
        raise RuntimeError("cudaMalloc failed")

    conn._scratch_ring = boom
    for i in range(4):
        stage(conn, [smeta(f"lt{i}", toks, [0, 1], salt=f"sg-latch3-{i}")], kv)
    assert conn._store_gather_disabled and len(boom_calls) == 3  # 4th never set up
    assert queue_snapshot(conn) == old_items
    assert (conn.dropped_puts, conn.failed_puts) == (old.dropped_puts, old.failed_puts) == (0, 0)
    assert slots_free(conn) == conn._store_slots_total  # setup fails lease nothing
    old.shutdown()
    conn.shutdown()


def test_finish_failure_counts_losses_and_isolates_plans():
    """RED-PROOF (accounting exactness): a raise out of the sync-failed
    rebuild must count every staged-but-never-enqueued block into
    dropped_puts/dropped_put_bytes, record the prefix hole, bring the leases
    home, and must NOT discard the other request's plan. The old outer
    except freed the leases and counted NOTHING — 4 blocks vanished with
    the shutdown line reading dropped=0."""
    kv = big_kv(8)
    for b in range(4):
        fill_block(kv, b, seed=200 + b)
    toks_b = list(range(2100, 2108))

    old = make_conn(kvblockd_store_queue_bytes=8 * TOTAL)
    pause_drain(old)
    stage(old, [smeta("gb", toks_b, [2, 3], salt="sg-gap-b")], kv)

    conn = gather_conn(kvblockd_store_queue_bytes=8 * TOTAL)
    pause_drain(conn)
    conn._store_sync = lambda dev: False  # every gathered blob is torn -> rebuild
    real_bb = conn._block_bytes
    state = {"raised": False}

    def sick_once(t):
        # The rebuild's FIRST host copy (plan "ga") dies like a sick CUDA
        # device; every later copy (plan "gb") works — the isolation probe.
        if not state["raised"]:
            state["raised"] = True
            raise RuntimeError("CUDA error: device-side assert (injected)")
        return real_bb(t)

    conn._block_bytes = sick_once
    stage(conn, [smeta("ga", list(range(2000, 2008)), [0, 1], salt="sg-gap-a"),
                 smeta("gb", toks_b, [2, 3], salt="sg-gap-b")], kv)
    # Plan "ga": both blocks lost mid-rebuild -> counted, hole recorded.
    assert conn.dropped_puts == 2
    assert conn.dropped_put_bytes == 2 * TOTAL
    assert conn._store_holes == {"ga": 0}
    # Plan "gb": rebuilt and enqueued, byte-identical to the bytearray oracle.
    assert queue_snapshot(conn) == queue_snapshot(old)
    assert slots_free(conn) == conn._store_slots_total
    assert_slot_invariant(conn)
    old.shutdown()
    conn.shutdown()


def test_finish_raise_mid_plan_counts_only_unsent_tail():
    """RED-PROOF: a raise after some items were already handed to
    _sq_enqueue counts ONLY the unsent tail (a delivered block must never be
    double-counted), records the hole at the first unsent block, and frees
    exactly the unsent leases."""
    conn = gather_conn(kvblockd_store_queue_bytes=8 * TOTAL)
    pause_drain(conn)
    kv = big_kv(8)
    for b in range(3):
        fill_block(kv, b, seed=210 + b)
    real_enq = conn._sq_enqueue
    calls = []

    def boom_on_second(key, buf, slot_id=None):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("injected enqueue-plumbing fault")
        return real_enq(key, buf, slot_id)

    conn._sq_enqueue = boom_on_second
    stage(conn, [smeta("mp", list(range(2200, 2212)), [0, 1, 2], salt="sg-mid")], kv)
    with conn._sq_cond:
        assert len(conn._sq) == 1  # block 0 reached the queue and stays
    assert conn.dropped_puts == 2 and conn.dropped_put_bytes == 2 * TOTAL
    assert conn._store_holes == {"mp": 1}
    assert slots_free(conn) == conn._store_slots_total - 1  # queue owns block 0's
    assert_slot_invariant(conn)
    conn.shutdown()


def test_sync_failure_rebuilds_bytearrays_and_frees_slots():
    """A failed device sync means every gathered blob is possibly torn: the
    blobs are REBUILT from the still-valid paged memory through the bytearray
    path (identical bytes, identical accounting), the slots come home, and
    the failure counts toward the latch."""
    kv = big_kv(8)
    for b in (0, 1):
        fill_block(kv, b, seed=180 + b)
    toks = list(range(1500, 1508))

    old = make_conn(kvblockd_store_queue_bytes=8 * TOTAL)
    pause_drain(old)
    stage(old, [smeta("sf", toks, [0, 1], salt="sg-syncfail")], kv)

    conn = gather_conn(kvblockd_store_queue_bytes=8 * TOTAL)
    pause_drain(conn)
    conn._store_sync = lambda dev: False
    stage(conn, [smeta("sf", toks, [0, 1], salt="sg-syncfail")], kv)
    assert queue_snapshot(conn) == queue_snapshot(old)
    with conn._sq_cond:
        assert all(s is None for _k, _b, s in conn._sq)  # rebuilt, not slot-backed
    assert slots_free(conn) == conn._store_slots_total
    assert conn._store_gather_fails == 1 and not conn._store_gather_disabled
    assert conn.dropped_puts == 0
    old.shutdown()
    conn.shutdown()


# ------------------------------------------------- load-priority park-once

def test_drain_parks_once_per_gate_episode_not_once_per_pop(monkeypatch):
    """RED-PROOF (MED-1): N queued blobs under one wedged load gate must cost
    ONE ceiling total, not N — and the drain stays live the whole time."""
    monkeypatch.setattr(conn_mod, "_DRAIN_GATE_CEILING_S", 0.3)
    conn = make_conn()
    fake = FakeClient()
    conn._ensure = lambda: fake
    pause_drain(conn)
    with conn._sq_cond:
        conn._loads_inflight += 1  # wedged gate: never clears on its own
    for i in range(4):
        assert conn._sq_enqueue(bytes([i]) * 32, bytearray(b"x" * 64))
    t0 = time.monotonic()
    resume_drain(conn)
    assert wait_until(lambda: len(fake.puts) == 4, timeout=5.0)
    elapsed = time.monotonic() - t0
    assert elapsed < 0.8, f"gate cost {elapsed:.2f}s across 4 pops — must be one ceiling"
    # Releasing the gate ends the episode: the arm resets for the next one.
    with conn._sq_cond:
        conn._loads_inflight -= 1
        conn._sq_cond.notify_all()
    assert conn._sq_enqueue(b"\x77" * 32, bytearray(b"y" * 64))
    assert wait_until(lambda: len(fake.puts) == 5)
    with conn._sq_cond:
        assert conn._drain_park_until is None
    conn.shutdown()


def test_new_load_episode_during_put_gets_fresh_park(monkeypatch):
    """RED-PROOF (stale park-arm): a NEW load episode that begins while the
    drain is inside put() must park its own fresh ceiling. The old code
    reset the arm only when the DRAIN sampled loads_inflight == 0 — a
    gate-down/gate-up flip inside one put() left the previous episode's
    expired arm in place, and every such overlapped episode got ZERO park."""
    monkeypatch.setattr(conn_mod, "_DRAIN_GATE_CEILING_S", 0.3)
    conn = make_conn()

    class GatedClient(FakeClient):
        def __init__(self):
            super().__init__()
            self.put_times = []
            self.in_put = threading.Event()
            self.release = threading.Event()

        def put(self, key, bufs, ttl_ms=0):
            self.put_times.append(time.monotonic())
            self.in_put.set()
            assert self.release.wait(5.0)
            self.release.clear()
            return super().put(key, bufs, ttl_ms)

    fake = GatedClient()
    conn._ensure = lambda: fake
    # The load itself is a seam: it parks on an event so each episode's gate
    # is raised and cleared through the REAL start_load_kv edges.
    load_gate = threading.Event()
    load_up = threading.Event()

    def parked_load(req):
        load_up.set()
        assert load_gate.wait(5.0)

    conn._load_one = parked_load
    lmeta = KvbReqMeta(req_id="ld", token_ids=list(range(BLOCK)), cache_salt=None,
                       mm_ids=[], lora_name="", block_ids=[0], load_start_block=0,
                       num_load_blocks=1, store_start_block=1, store_end_block=1)

    def episode():
        conn.bind_connector_metadata(KvblockdConnectorMetadata(requests=[lmeta]))
        conn.start_load_kv(None)
        conn.clear_connector_metadata()

    pause_drain(conn)
    for i in range(2):
        assert conn._sq_enqueue(bytes([i]) * 32, bytearray(b"x" * 64))
    t1 = threading.Thread(target=episode, daemon=True)
    t1.start()
    assert load_up.wait(5.0)
    load_up.clear()
    t0 = time.monotonic()
    resume_drain(conn)
    assert fake.in_put.wait(5.0)
    fake.in_put.clear()
    assert fake.put_times[0] - t0 >= 0.25  # episode 1 paid its ceiling
    # WHILE the drain is inside put(): episode 1 ends, episode 2 begins.
    load_gate.set()
    t1.join(5.0)
    assert not t1.is_alive()
    load_gate.clear()
    t2 = threading.Thread(target=episode, daemon=True)
    t2.start()
    assert load_up.wait(5.0)
    load_up.clear()
    t0 = time.monotonic()
    fake.release.set()  # put #1 returns; the drain re-checks the gate
    assert fake.in_put.wait(5.0)
    fake.in_put.clear()
    park2 = fake.put_times[1] - t0
    assert park2 >= 0.25, f"episode 2 inherited a stale arm (parked {park2:.3f}s)"
    load_gate.set()
    t2.join(5.0)
    assert not t2.is_alive()
    fake.release.set()  # let put #2 finish
    assert wait_until(lambda: len(fake.puts) == 2)
    conn.shutdown()
