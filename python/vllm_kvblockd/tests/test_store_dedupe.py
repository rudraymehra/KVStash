"""Acked-key store dedupe suite (kvblockd_store_dedupe_keys): a request
re-served by vLLM's LOCAL prefix cache (n_load=0) must stop re-copying and
re-PUTting blocks the daemon already acked — the LRU is populated ONLY by the
drain thread on OK/OK_EXISTS verdicts, consulted by _stage_one BEFORE any
copy, advisory and TTL-bounded (the daemon evicts; a wrongly-skipped re-store
is a future miss inside the window, never a wrong byte). The knob ships
INERT (default 0 — today's self-healing re-put), so every test here enables
it explicitly, exactly like the multi-turn arm will. Same conventions as
test_async_store.py: FakeClient only to observe put traffic; byte-identity
assertions ride the real wire."""

from __future__ import annotations

import logging
import time

import pytest

torch = pytest.importorskip("torch")

from kvblockd.errors import ConnectionLost
from test_async_store import FakeClient, make_conn, wait_until
from test_connector import (
    BLOCK,
    LAYERS,
    StubRequest,
    fill_block,
    fresh_kv,
    run_step,
)

from vllm_kvblockd.config import block_chain_keys
from vllm_kvblockd.connector import KvbReqMeta, _AckedKeyLRU


def make_dedupe_conn(daemon, **extra):
    """The knob is INERT at the defaults — arm it the way the arm will."""
    extra.setdefault("kvblockd_store_dedupe_keys", 4096)
    return make_conn(daemon, **extra)


def local_hit_meta(rid, toks, salt, block_ids, start, end):
    """The spec's target shape: a request whose prefix was served by vLLM's
    LOCAL cache — n_load=0, so store_start derives to the whole range."""
    return KvbReqMeta(req_id=rid, token_ids=toks, cache_salt=salt, mm_ids=[],
                      lora_name="", block_ids=block_ids,
                      load_start_block=0, num_load_blocks=0,
                      store_start_block=start, store_end_block=end)


def prime(conn, kv, n_blocks=2, base_seed=801):
    for bid in range(n_blocks):
        fill_block(kv, bid, seed=base_seed + bid)
    for name, t in kv.items():
        conn.save_kv_layer(name, t, None)


def wait_acked(conn, salt, toks):
    """Wait until the drain thread's acks landed in the LRU: the flush only
    proves the puts RETURNED — the add happens a hair later on the drain
    thread (settle armor for the tests, not a serving-path contract)."""
    keys = block_chain_keys(conn._seed(salt, [], ""), toks, BLOCK)
    assert wait_until(lambda: all(conn._acked_keys.hit(k) for k in keys))


def test_acked_prefix_is_never_restaged(daemon, caplog):
    """Turn 2 of the same local-prefix request must not copy, queue, or put
    the already-acked blocks: zero new puts, zero blob builds, the skip
    counted in deduped_puts and disclosed in the shutdown summary."""
    caplog.set_level(logging.WARNING, logger="vllm_kvblockd")
    conn = make_dedupe_conn(daemon)
    fake = FakeClient()
    conn._client = fake
    kv = fresh_kv()
    prime(conn, kv)
    toks = list(range(800, 808))  # 2 full blocks
    meta = local_hit_meta("dd1", toks, "sd-skip", [0, 1], 0, 2)

    conn._stage_one(meta)
    assert conn._store_flush(10.0) == 0
    assert len(fake.puts) == 2  # turn 1 delivered (and acked) both blocks
    wait_acked(conn, "sd-skip", toks)

    built = []
    real_build = conn._build_block_blob

    def spy(bid, names, bpl, prefix, total):
        built.append(bid)
        return real_build(bid, names, bpl, prefix, total)

    conn._build_block_blob = spy
    meta2 = local_hit_meta("dd2", toks, "sd-skip", [0, 1], 0, 2)  # turn 2
    assert conn._stage_one(meta2) is None
    assert built == [], "an acked block must be skipped BEFORE any copy"
    assert len(conn._sq) == 0
    assert conn.deduped_puts == 2
    assert conn._store_flush(5.0) == 0
    assert len(fake.puts) == 2, "turn 2 re-put blocks the daemon already acked"
    conn.shutdown()
    assert any("deduped=2" in r.getMessage() and "kvblockd store queue:" in r.getMessage()
               for r in caplog.records)


def test_dedupe_window_expires(daemon):
    """Past the TTL the ack proves nothing (the daemon may have evicted):
    the same range re-stages exactly like today."""
    conn = make_dedupe_conn(daemon, kvblockd_store_dedupe_ttl_s=0.05)
    fake = FakeClient()
    conn._client = fake
    kv = fresh_kv()
    prime(conn, kv)
    toks = list(range(810, 818))
    conn._stage_one(local_hit_meta("de1", toks, "sd-ttl", [0, 1], 0, 2))
    assert conn._store_flush(10.0) == 0
    assert len(fake.puts) == 2
    time.sleep(0.15)  # the ack window closes
    conn._stage_one(local_hit_meta("de2", toks, "sd-ttl", [0, 1], 0, 2))
    assert conn._store_flush(10.0) == 0
    assert len(fake.puts) == 4 and conn.deduped_puts == 0
    conn.shutdown()


def test_dedupe_disabled_by_knob_keeps_todays_behavior(daemon):
    """UNTOUCHED defaults ship the feature OFF: no LRU, every turn re-puts —
    today's self-healing behavior (an acked-then-evicted block is re-stored
    by the very next turn, not missing until the TTL). Also the A/B arm."""
    conn = make_conn(daemon)  # no knob: the default itself must be inert
    assert conn._acked_keys is None
    fake = FakeClient()
    conn._client = fake
    kv = fresh_kv()
    prime(conn, kv)
    toks = list(range(820, 828))
    for rid in ("dk1", "dk2"):
        conn._stage_one(local_hit_meta(rid, toks, "sd-off", [0, 1], 0, 2))
        assert conn._store_flush(10.0) == 0
    assert len(fake.puts) == 4 and conn.deduped_puts == 0
    conn.shutdown()


def test_only_server_acks_populate(daemon, monkeypatch):
    """A block that FAILED delivery (or never left the queue) must never be
    skipped later: only an OK/OK_EXISTS verdict from the drain thread's put
    lands in the LRU — false positives are structurally impossible."""
    from vllm_kvblockd import connector as conn_mod

    monkeypatch.setattr(conn_mod, "_REDIAL_BACKOFF_S", 0.05)
    conn = make_dedupe_conn(daemon)
    fake = FakeClient(fail_first=4)  # both blocks fail their put AND the retry
    conn._client = fake
    conn._ensure = lambda: fake  # keep failing puts on the fake, no redial
    kv = fresh_kv()
    prime(conn, kv)
    toks = list(range(830, 838))
    conn._stage_one(local_hit_meta("fp1", toks, "sd-fail", [0, 1], 0, 2))
    assert wait_until(lambda: conn.failed_puts == 2)
    assert wait_until(lambda: len(conn._sq) == 0 and conn._sq_inflight == 0)

    conn._stage_one(local_hit_meta("fp2", toks, "sd-fail", [0, 1], 0, 2))
    assert conn.deduped_puts == 0, "a failed put must never count as an ack"
    assert conn._store_flush(10.0) == 0
    assert len(fake.puts) == 2, "the retry turn must actually deliver both blocks"
    conn.shutdown()


def test_only_the_leading_run_is_skipped(daemon):
    """The consult is a leading-run walk: an acked key BEHIND an unacked one
    neither skips (correct: _finish_stage's tail-skip arithmetic counts by
    block index) nor blocks the unacked head from staging."""
    conn = make_dedupe_conn(daemon)
    fake = FakeClient()
    conn._client = fake
    kv = fresh_kv()
    prime(conn, kv, n_blocks=3)
    toks = list(range(840, 852))  # 3 full blocks
    keys = block_chain_keys(conn._seed("sd-lead", [], ""), toks, BLOCK)
    # Hand-plant a NON-leading ack (block 1 only): nothing may be skipped.
    conn._acked_keys.add(keys[1])
    conn._stage_one(local_hit_meta("lr1", toks, "sd-lead", [0, 1, 2], 0, 3))
    assert conn._store_flush(10.0) == 0
    assert len(fake.puts) == 3 and conn.deduped_puts == 0
    wait_acked(conn, "sd-lead", toks)
    # Now the LEADING run (blocks 0 and 1 re-acked by the drain above, plus
    # block 2): a fresh turn skips everything.
    conn._stage_one(local_hit_meta("lr2", toks, "sd-lead", [0, 1, 2], 0, 3))
    assert len(conn._sq) == 0 and conn.deduped_puts == 3
    conn.shutdown()


def test_dedupe_end_to_end_bytes_still_served(daemon):
    """The skipped re-store is safe BECAUSE the daemon still holds the bytes:
    turn 2 stages nothing, yet a fresh engine loads the turn-1 bytes exactly."""
    salt = "sd-e2e"
    toks = list(range(860, 869))  # 9 tokens -> 2 aligned blocks
    conn = make_dedupe_conn(daemon)
    kv = fresh_kv()
    fill_block(kv, 2, seed=821)
    fill_block(kv, 5, seed=822)
    run_step(conn, StubRequest("ee1", toks, salt), [2, 5], kv)  # turn 1: real wire
    wait_acked(conn, salt, toks[:8])

    conn._stage_one(local_hit_meta("ee2", toks[:8], salt, [2, 5], 0, 2))  # turn 2
    assert len(conn._sq) == 0 and conn.deduped_puts == 2
    conn.shutdown()

    conn2 = make_conn(daemon)
    req = StubRequest("ee3", toks, salt)
    n, _ = conn2.get_num_new_matched_tokens(req, 0)
    assert n == 8, "the deduped blocks must still be on the daemon"
    kv2 = fresh_kv()
    run_step(conn2, req, [1, 3], kv2)
    assert conn2.get_block_ids_with_load_errors() == set()
    for name in LAYERS:
        assert torch.equal(kv2[name][1], kv[name][2])
        assert torch.equal(kv2[name][3], kv[name][5])
    conn2.shutdown()


def test_connection_loss_clears_acked_keys(daemon):
    """A connection-class failure proves every ack stale: the daemon behind
    the redial may have restarted with an EMPTY store, so pre-outage acks
    must not suppress the self-healing re-store until the TTL runs out —
    _drop_client wipes the LRU the moment it drops the client."""
    conn = make_dedupe_conn(daemon)
    fake = FakeClient()
    conn._client = fake
    kv = fresh_kv()
    prime(conn, kv)
    toks = list(range(870, 878))
    conn._stage_one(local_hit_meta("cl1", toks, "sd-drop", [0, 1], 0, 2))
    assert conn._store_flush(10.0) == 0
    assert len(fake.puts) == 2
    wait_acked(conn, "sd-drop", toks)

    # The outage: a batch op died ConnectionLost, the client is dropped.
    conn._drop_client(ConnectionLost("daemon restarted"))
    fake2 = FakeClient()
    conn._client = fake2  # the redial, onto a possibly-empty store

    conn._stage_one(local_hit_meta("cl2", toks, "sd-drop", [0, 1], 0, 2))
    assert conn.deduped_puts == 0, "a pre-outage ack must not survive the drop"
    assert conn._store_flush(10.0) == 0
    assert len(fake2.puts) == 2, "the post-outage turn must re-store both blocks"
    conn.shutdown()


def test_lru_is_bounded_and_ttl_limited():
    """The set is armor, not a cache: capacity evicts the OLDEST ack, the TTL
    expires trust without a fresh ack, and a re-ack refreshes both."""
    lru = _AckedKeyLRU(2, ttl_s=30.0)
    lru.add(b"k1")
    lru.add(b"k2")
    lru.add(b"k3")  # over cap: k1 (oldest ack) evicted
    assert not lru.hit(b"k1")
    assert lru.hit(b"k2") and lru.hit(b"k3")
    lru.add(b"k2")  # re-ack: k2 is now the newest
    lru.add(b"k4")  # evicts k3, not k2
    assert lru.hit(b"k2") and not lru.hit(b"k3")

    fast = _AckedKeyLRU(8, ttl_s=0.02)
    fast.add(b"k5")
    assert fast.hit(b"k5")
    time.sleep(0.05)
    assert not fast.hit(b"k5"), "an expired ack proves nothing"
