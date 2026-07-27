"""Write-behind store suite (kvblockd_async_store): wait_for_save must stage
OWNED copies and return immediately; a single FIFO drain thread delivers them.
Same conventions as test_connector.py: stub vLLM surface, REAL daemon on the
wire, per-test cache_salt keyspaces. Fake clients appear ONLY to control put
timing/failures — the byte-identity assertions always ride the real wire."""

from __future__ import annotations

import logging
import threading
import time

import pytest

torch = pytest.importorskip("torch")

from kvblockd.client import Client
from kvblockd.errors import ConnectionLost
from test_connector import (
    BLOCK,
    HID,
    LAYERS,
    StubForwardContext,
    StubNewReq,
    StubRequest,
    StubSchedulerOutput,
    StubVllmConfig,
    fill_block,
    fresh_kv,
    run_step,
)

from vllm_kvblockd.config import block_chain_keys
from vllm_kvblockd.connector import BLOB_PREFIX_LEN, KvblockdConnector

# One test-engine blob: 32B prefix + per-layer (2, BLOCK, HID) bfloat16 bytes.
TOTAL = BLOB_PREFIX_LEN + len(LAYERS) * (2 * BLOCK * HID * 2)


def make_conn(daemon, **extra):
    cfg = StubVllmConfig(daemon["port"])
    cfg.kv_transfer_config.kv_connector_extra_config.update(extra)
    return KvblockdConnector(cfg, role="scheduler", kv_cache_config=None)


def store_step(conn, request, block_ids, kv):
    """One scheduler+worker step that ONLY stores — no flush, so tests can
    assert on the staged (undrained) queue."""
    n, _ = conn.get_num_new_matched_tokens(request, 0)
    conn.update_state_after_alloc(request, None, n)
    out = StubSchedulerOutput(
        [StubNewReq(request, block_ids)],
        {request.request_id: len(request.prompt_token_ids)},
    )
    conn.bind_connector_metadata(conn.build_connector_meta(out))
    conn.start_load_kv(StubForwardContext(kv))
    for name, t in kv.items():
        conn.save_kv_layer(name, t, None)
    conn.wait_for_save()
    conn.clear_connector_metadata()


def pause_drain(conn):
    """Shadow the lazy thread start so enqueued blobs stay inspectable."""
    conn._store_thread_start = lambda: None


def resume_drain(conn):
    del conn._store_thread_start  # back to the class method
    conn._store_thread_start()


def wait_until(fn, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if fn():
            return True
        time.sleep(0.01)
    return False


class FakeClient:
    """Drop-in for the pooled client where the test must control put timing
    or inject failures. batch_exists reports a cold cache."""

    def __init__(self, put_delay=0.0, fail_first=0):
        self.puts: list[tuple[bytes, bytes]] = []
        self.put_delay = put_delay
        self.fail_first = fail_first
        self.closed = False
        self.put_started = threading.Event()

    def batch_exists(self, keys):
        return 0, None

    def put(self, key, bufs, ttl_ms=0):
        self.put_started.set()
        if self.put_delay:
            time.sleep(self.put_delay)
        if self.fail_first > 0:
            self.fail_first -= 1
            raise ConnectionLost("injected put failure")
        self.puts.append((bytes(key), b"".join(bytes(b) for b in bufs)))
        return 0

    def close(self):
        self.closed = True


def test_store_bytes_are_owned(daemon):
    """Mutating the paged tensors AFTER wait_for_save returned (and before the
    drain ran) must not change the delivered bytes: the staged copies are
    owned, never aliasing views of paged memory."""
    salt = "as-owned"
    toks = list(range(300, 309))  # 9 tokens -> 2 blocks
    conn = make_conn(daemon)
    pause_drain(conn)
    kv = fresh_kv()
    fill_block(kv, 2, seed=201)
    fill_block(kv, 5, seed=202)
    expected = {n: (kv[n][2].clone(), kv[n][5].clone()) for n in LAYERS}
    store_step(conn, StubRequest("ao1", toks, salt), [2, 5], kv)
    assert len(conn._sq) == 2  # staged, not yet drained

    # The engine reuses the paged blocks for another request:
    for t in kv.values():
        t.zero_()
    resume_drain(conn)
    assert conn._store_flush(10.0) == 0
    conn.shutdown()

    conn2 = make_conn(daemon)
    req = StubRequest("ao2", toks, salt)
    n, _ = conn2.get_num_new_matched_tokens(req, 0)
    assert n == 8
    kv2 = fresh_kv()
    run_step(conn2, req, [1, 3], kv2)
    assert conn2.get_block_ids_with_load_errors() == set()
    for name in LAYERS:
        assert torch.equal(kv2[name][1], expected[name][0])
        assert torch.equal(kv2[name][3], expected[name][1])
    conn2.shutdown()


def test_engine_never_blocks(daemon):
    """A put that takes 2s must not hold wait_for_save hostage: staging is
    memcpy-fast, the TCP time lands on the drain thread."""
    conn = make_conn(daemon)
    fake = FakeClient(put_delay=2.0)
    conn._client = fake
    kv = fresh_kv()
    fill_block(kv, 1, seed=203)
    toks = list(range(310, 315))  # 1 block
    t0 = time.monotonic()
    store_step(conn, StubRequest("nb1", toks, "as-noblock"), [1], kv)
    elapsed = time.monotonic() - t0
    assert elapsed < 1.0, f"wait_for_save blocked {elapsed:.2f}s on the put"
    assert conn._store_flush(10.0) == 0
    assert len(fake.puts) == 1
    conn.shutdown()


def test_drop_under_budget_pressure(daemon):
    """Past kvblockd_store_queue_bytes the enqueue drops (never blocks, never
    raises) and counts the loss in the public counters."""
    conn = make_conn(daemon, kvblockd_store_queue_bytes=TOTAL + TOTAL // 2)  # fits ONE
    pause_drain(conn)
    kv = fresh_kv()
    fill_block(kv, 0, seed=204)
    fill_block(kv, 1, seed=205)
    store_step(conn, StubRequest("bp1", list(range(320, 329)), "as-budget"), [0, 1], kv)
    assert len(conn._sq) == 1
    assert conn._sq_bytes <= TOTAL + TOTAL // 2
    assert conn.dropped_puts == 1
    assert conn.dropped_put_bytes == TOTAL
    conn.shutdown()


def test_tail_skip_after_drop(daemon):
    """Once one block of a request drops, the request's REMAINING blocks must
    be dropped in the same call WITHOUT being copied or offered to the queue:
    prefix-chain keys make post-hole blocks unreachable by the consecutive-
    prefix lookup, so queueing them would spend budget on dead bytes."""
    conn = make_conn(daemon, kvblockd_store_queue_bytes=TOTAL)  # exactly one fits
    pause_drain(conn)
    offered = []
    real_enqueue = conn._sq_enqueue

    def spy(key, buf):
        offered.append(bytes(key))
        return real_enqueue(key, buf)

    conn._sq_enqueue = spy
    kv = fresh_kv()
    for bid, seed in ((0, 206), (1, 207), (2, 208), (3, 209)):
        fill_block(kv, bid, seed)
    toks = list(range(330, 347))  # 17 tokens -> 4 blocks
    store_step(conn, StubRequest("ts1", toks, "as-tail"), [0, 1, 2, 3], kv)
    assert len(offered) == 2, "blocks after the first drop must not reach the queue"
    assert len(conn._sq) == 1
    assert conn.dropped_puts == 3  # block 1 (budget) + blocks 2,3 (tail-skip)
    assert conn.dropped_put_bytes == 3 * TOTAL
    conn.shutdown()


def test_tail_skip_persists_across_chunked_prefill_steps(daemon):
    """Once block j of a request drops, LATER _stage_one calls for the SAME
    request (chunked-prefill continuation stores) must not copy or queue rows
    past the hole: prefix-chain keys make them unreachable by the
    consecutive-prefix lookup, so they are dead bytes even when the budget
    has freed up again — skip them AND count them dropped. The recorded hole
    is pruned when the request finishes."""
    from vllm_kvblockd.connector import KvbReqMeta

    conn = make_conn(daemon, kvblockd_store_queue_bytes=3 * TOTAL + TOTAL // 2)
    pause_drain(conn)
    kv = fresh_kv()
    for bid in range(6):
        fill_block(kv, bid, seed=240 + bid)
    for name, t in kv.items():
        conn.save_kv_layer(name, t, None)
    toks = list(range(450, 475))  # 25 tokens -> 6 full blocks

    def meta(start, end):
        return KvbReqMeta(req_id="tsp1", token_ids=toks[:24], cache_salt="as-persist",
                          mm_ids=[], lora_name="", block_ids=[0, 1, 2, 3, 4, 5],
                          load_start_block=0, num_load_blocks=0,
                          store_start_block=start, store_end_block=end)

    conn._stage_one(meta(0, 4))  # step 1: three fit, block 3 drops -> the hole
    assert len(conn._sq) == 3 and conn.dropped_puts == 1
    resume_drain(conn)
    assert conn._store_flush(10.0) == 0  # the budget is fully free again
    pause_drain(conn)

    offered = []
    real_enqueue = conn._sq_enqueue

    def spy(key, buf):
        offered.append(bytes(key))
        return real_enqueue(key, buf)

    conn._sq_enqueue = spy
    conn._stage_one(meta(4, 6))  # step 2 (later chunk): rows 4,5 are past the hole
    assert offered == [], "post-hole rows must never be copied or offered to the queue"
    assert len(conn._sq) == 0
    assert conn.dropped_puts == 3  # block 3 (budget) + blocks 4,5 (persisted skip)
    assert conn.dropped_put_bytes == 3 * TOTAL
    conn.request_finished(StubRequest("tsp1", toks), [])
    assert conn._store_holes == {}  # pruned with the request
    conn.shutdown()


def test_delivery_order_and_completeness(daemon):
    """The drain is FIFO and complete: delivery order == enqueue order,
    exactly — per-request block order is what makes a partial delivery a
    usable consecutive prefix."""
    conn = make_conn(daemon)
    fake = FakeClient()
    conn._client = fake
    pause_drain(conn)
    kv = fresh_kv()
    for bid, seed in ((0, 210), (1, 211), (2, 212), (3, 213)):
        fill_block(kv, bid, seed)
    toks_a = list(range(340, 349))
    toks_b = list(range(350, 359))
    store_step(conn, StubRequest("fo-a", toks_a, "as-fifo-a"), [0, 1], kv)
    store_step(conn, StubRequest("fo-b", toks_b, "as-fifo-b"), [2, 3], kv)
    expected = (
        block_chain_keys(conn._seed("as-fifo-a", [], ""), toks_a[:8], BLOCK)
        + block_chain_keys(conn._seed("as-fifo-b", [], ""), toks_b[:8], BLOCK)
    )
    assert len(conn._sq) == 4
    resume_drain(conn)
    assert conn._store_flush(10.0) == 0
    assert [k for k, _ in fake.puts] == expected
    conn.shutdown()


def test_shutdown_flushes(daemon, caplog):
    """shutdown() drains what is queued (bounded by the flush timeout); a
    WEDGED client cannot hold shutdown hostage — the remainder is counted
    dropped and disclosed."""
    caplog.set_level(logging.WARNING, logger="vllm_kvblockd")
    conn = make_conn(daemon)
    fake = FakeClient(put_delay=0.05)
    conn._client = fake
    kv = fresh_kv()
    fill_block(kv, 4, seed=214)
    fill_block(kv, 5, seed=215)
    store_step(conn, StubRequest("sf1", list(range(360, 369)), "as-flush"), [4, 5], kv)
    conn.shutdown()
    assert len(fake.puts) == 2  # everything queued was delivered
    assert conn.dropped_puts == 0
    assert any("kvblockd store queue: dropped=0 failed=0 dropped_bytes=0"
               in r.getMessage() for r in caplog.records)

    # Wedged-client path: put blocks far past the flush timeout.
    caplog.clear()
    conn2 = make_conn(daemon, kvblockd_store_flush_timeout_s=0.4)
    wedged = FakeClient(put_delay=60.0)
    conn2._client = wedged
    store_step(conn2, StubRequest("sf2", list(range(370, 379)), "as-wedge"), [6, 7], kv)
    assert wedged.put_started.wait(5.0)
    t0 = time.monotonic()
    conn2.shutdown()
    assert time.monotonic() - t0 < 3.0, "a wedged put held shutdown hostage"
    assert conn2.dropped_puts == 2  # 1 in flight + 1 still queued
    assert any("dropped=2" in r.getMessage() and "kvblockd store queue:" in r.getMessage()
               for r in caplog.records)


def test_drain_survives_a_daemon_blip_without_burning_the_backlog(chaos_daemon, monkeypatch):
    """A connection-class put failure must NOT burn the staged backlog at one
    doomed put per item under the dial breaker: the drain re-queues the failed
    item ONCE at the head and waits out the redial backoff before retrying —
    a daemon blip costs one backoff window, after which EVERYTHING staged is
    delivered in order and failed_puts stays 0."""
    from vllm_kvblockd import connector as conn_mod

    monkeypatch.setattr(conn_mod, "_REDIAL_BACKOFF_S", 0.5)
    conn = make_conn(chaos_daemon, kvblockd_op_timeout_s=2.0,
                     kvblockd_connect_timeout_s=2.0)
    pause_drain(conn)
    kv = fresh_kv()
    fill_block(kv, 0, seed=230)
    fill_block(kv, 1, seed=231)
    toks = list(range(430, 439))  # 2 blocks
    store_step(conn, StubRequest("bl1", toks, "as-blip"), [0, 1], kv)
    assert len(conn._sq) == 2

    # The blip: the daemon dies mid-run; a fresh one takes over the SAME
    # endpoint (kubelet restart / systemd Restart= — the redial target).
    chaos_daemon["proc"].kill()
    chaos_daemon["proc"].wait(timeout=5)
    chaos_daemon["respawn"](chaos_daemon["port"])

    resume_drain(conn)
    assert conn._store_flush(15.0) == 0
    assert conn.failed_puts == 0, "a blip must cost a backoff window, not failed puts"
    assert conn.dropped_puts == 0
    conn.shutdown()

    conn2 = make_conn(chaos_daemon)  # fresh view of the recovered daemon
    n, _ = conn2.get_num_new_matched_tokens(StubRequest("bl2", toks, "as-blip"), 0)
    assert n == 8, "backlog blocks never reached the recovered daemon"
    conn2.shutdown()


def test_drain_gives_up_on_an_item_after_its_retry_fails(daemon, monkeypatch):
    """The requeue is ONCE per item: a second consecutive failure of the SAME
    item counts it failed and moves on — a dead endpoint drains the backlog at
    one retry per item, never an infinite requeue loop."""
    from vllm_kvblockd import connector as conn_mod

    monkeypatch.setattr(conn_mod, "_REDIAL_BACKOFF_S", 0.05)
    conn = make_conn(daemon)
    always_dead = FakeClient(fail_first=10**9)
    conn._ensure = lambda: always_dead  # every put attempt fails, forever
    pause_drain(conn)
    kv = fresh_kv()
    fill_block(kv, 2, seed=232)
    fill_block(kv, 3, seed=233)
    store_step(conn, StubRequest("gu1", list(range(440, 449)), "as-giveup"), [2, 3], kv)
    assert len(conn._sq) == 2
    resume_drain(conn)
    assert wait_until(lambda: conn.failed_puts == 2)
    assert wait_until(lambda: len(conn._sq) == 0 and conn._sq_inflight == 0)
    assert conn._store_thread.is_alive()  # the thread survives to serve a recovery
    conn.shutdown()


def test_failed_put_survives(daemon, monkeypatch):
    """A ConnectionLost out of a put drops the client (breaker discipline)
    and leaves the drain thread ALIVE; the item is re-queued once and the
    post-backoff retry redials the real daemon and delivers it — the failure
    costs a backoff window, not the block (failed_puts stays 0)."""
    from vllm_kvblockd import connector as conn_mod

    monkeypatch.setattr(conn_mod, "_REDIAL_BACKOFF_S", 0.2)
    conn = make_conn(daemon)
    fake = FakeClient(fail_first=1)
    conn._client = fake
    pause_drain(conn)
    kv = fresh_kv()
    fill_block(kv, 4, seed=216)
    toks = list(range(380, 385))
    store_step(conn, StubRequest("fp1", toks, "as-fail-1"), [4], kv)
    resume_drain(conn)
    # _drop_client runs right after the failed put; the fake dies with it.
    assert wait_until(lambda: fake.closed), "client not dropped after put failure"
    assert conn._store_flush(10.0) == 0
    assert conn.failed_puts == 0, "one failure + healthy retry must not count failed"
    assert conn._store_thread.is_alive()

    fill_block(kv, 5, seed=217)
    toks2 = list(range(390, 395))
    store_step(conn, StubRequest("fp2", toks2, "as-fail-2"), [5], kv)
    assert conn._store_flush(10.0) == 0
    assert conn._store_thread.is_alive()

    conn2 = make_conn(daemon)  # fresh view of the real daemon
    n, _ = conn2.get_num_new_matched_tokens(StubRequest("fp3", toks, "as-fail-1"), 0)
    assert n == 4, "the re-queued put never reached the daemon"
    n, _ = conn2.get_num_new_matched_tokens(StubRequest("fp4", toks2, "as-fail-2"), 0)
    assert n == 4, "the post-redial put never reached the daemon"
    conn.shutdown()
    conn2.shutdown()


def test_dropped_puts_disclosure_line(daemon, caplog):
    """The shutdown summary is ALWAYS emitted at WARNING — zero included (the
    bench greps for the line's presence; INFO is dropped in engine-core)."""
    caplog.set_level(logging.WARNING, logger="vllm_kvblockd")
    conn = make_conn(daemon)
    conn.shutdown()  # nothing ever stored
    lines = [r for r in caplog.records if "kvblockd store queue:" in r.getMessage()]
    assert len(lines) == 1
    assert lines[0].levelno == logging.WARNING
    assert "dropped=0 failed=0 dropped_bytes=0" in lines[0].getMessage()

    caplog.clear()
    conn2 = make_conn(daemon, kvblockd_store_queue_bytes=1)  # everything drops
    pause_drain(conn2)
    kv = fresh_kv()
    fill_block(kv, 0, seed=218)
    fill_block(kv, 1, seed=219)
    store_step(conn2, StubRequest("dd1", list(range(400, 409)), "as-disc"), [0, 1], kv)
    conn2.shutdown()
    lines = [r for r in caplog.records if "kvblockd store queue:" in r.getMessage()]
    assert lines and f"dropped=2 failed=0 dropped_bytes={2 * TOTAL}" in lines[-1].getMessage()


def test_preemption_reuse_safe(daemon):
    """handle_preemptions is a documented no-op and preemption-driven block
    reuse cannot corrupt staged stores: the queue holds owned copies."""
    salt = "as-preempt"
    toks = list(range(410, 419))
    conn = make_conn(daemon)
    pause_drain(conn)
    kv = fresh_kv()
    fill_block(kv, 6, seed=220)
    fill_block(kv, 7, seed=221)
    expected = {n: (kv[n][6].clone(), kv[n][7].clone()) for n in LAYERS}
    store_step(conn, StubRequest("pr1", toks, salt), [6, 7], kv)
    qlen = len(conn._sq)
    conn.handle_preemptions(["pr1"])  # must not raise, must not touch the queue
    assert len(conn._sq) == qlen == 2
    # Preemption frees the blocks; another request immediately reuses them:
    for t in kv.values():
        t[6] = torch.ones_like(t[6])
        t[7].zero_()
    resume_drain(conn)
    assert conn._store_flush(10.0) == 0
    conn.shutdown()

    conn2 = make_conn(daemon)
    req = StubRequest("pr2", toks, salt)
    n, _ = conn2.get_num_new_matched_tokens(req, 0)
    assert n == 8
    kv2 = fresh_kv()
    run_step(conn2, req, [1, 3], kv2)
    for name in LAYERS:
        assert torch.equal(kv2[name][1], expected[name][0])
        assert torch.equal(kv2[name][3], expected[name][1])
    conn2.shutdown()


def test_sync_flag_bit_identical(daemon):
    """kvblockd_async_store=False keeps the original synchronous path: no
    staging, no thread, no queue — and the blob BYTES on the wire are
    identical to what the async path stores for the same KV content."""
    toks = list(range(420, 429))
    kv = fresh_kv()
    fill_block(kv, 2, seed=222)
    fill_block(kv, 5, seed=223)

    sync_conn = make_conn(daemon, kvblockd_async_store=False)
    staged = []
    sync_conn._stage_one = lambda req: staged.append(req)  # must never run
    run_step(sync_conn, StubRequest("sy1", toks, "as-sync"), [2, 5], kv)
    assert staged == []
    assert sync_conn._store_thread is None
    assert len(sync_conn._sq) == 0 and sync_conn.dropped_puts == 0

    async_conn = make_conn(daemon)  # default: async on
    run_step(async_conn, StubRequest("sy2", toks, "as-async"), [2, 5], kv)

    # Same KV bytes, two salts, two paths -> byte-identical blobs on the wire.
    client = Client((daemon["host"], daemon["port"]), namespace=daemon["namespace"],
                    token=daemon["token"], streams=1)
    try:
        keys_sync = block_chain_keys(sync_conn._seed("as-sync", [], ""), toks[:8], BLOCK)
        keys_async = block_chain_keys(async_conn._seed("as-async", [], ""), toks[:8], BLOCK)
        vals_sync, _ = client.batch_get_bytes(keys_sync)
        vals_async, _ = client.batch_get_bytes(keys_async)
    finally:
        client.close()
    assert all(v is not None for v in vals_sync + vals_async)
    assert vals_sync == vals_async
    sync_conn.shutdown()
    async_conn.shutdown()


def test_drain_parks_while_a_load_is_inflight(daemon):
    """Load-priority gate: with a load in flight the kvb-store drain parks
    (bounded by the ceiling) instead of contending for the wire; releasing
    the gate drains promptly and the flush contract is untouched."""
    conn = make_conn(daemon)
    with conn._sq_cond:
        conn._loads_inflight += 1
    assert conn._sq_enqueue(b"\x11" * 32, bytearray(b"x" * 64))
    time.sleep(0.1)  # well inside _DRAIN_GATE_CEILING_S
    with conn._sq_cond:
        assert len(conn._sq) == 1  # parked, not popped
    with conn._sq_cond:
        conn._loads_inflight -= 1
        conn._sq_cond.notify_all()
    assert conn._store_flush(5.0) == 0
    conn.shutdown()
