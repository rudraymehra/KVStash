"""Chaos suite for the fail-open guarantee (connector docstring): a dead,
hung, or slow daemon costs the engine at most one bounded delay per breaker
window — the engine never fails and never waits unboundedly. Every test runs
against ITS OWN real daemon (chaos_daemon), because it kills or pauses it.

Timing bounds are deliberately loose multiples of the configured timeouts —
they must catch "waits forever / waits a whole default timeout stack", not
measure the scheduler."""

from __future__ import annotations

import os
import signal
import time

import pytest

torch = pytest.importorskip("torch")

from test_connector import (
    StubForwardContext,
    StubNewReq,
    StubRequest,
    StubSchedulerOutput,
    StubVllmConfig,
    fill_block,
    fresh_kv,
    run_step,
)

from vllm_kvblockd.connector import KvblockdConnector

# One knob set for the whole suite: every wire wait is sub-second so the
# "bounded" assertions can be tight without flaking.
FAST = {
    "kvblockd_op_timeout_s": 0.5,
    "kvblockd_connect_timeout_s": 0.5,
    "kvblockd_load_deadline_s": 2.0,
}
# Generous ceiling for one degraded load: deadline + one op_timeout overshoot
# (a recv already in flight when the deadline expires) + scheduling slack.
BOUND_S = 5.0


def make_conn(info, **extra):
    cfg = StubVllmConfig(info["port"])
    cfg.kv_transfer_config.kv_connector_extra_config.update(dict(FAST, **extra))
    return KvblockdConnector(cfg, role="scheduler", kv_cache_config=None)


def promised_load_step(conn, req, block_ids, kv, n_promised_blocks):
    """Bind a step that PROMISES a load of the first n_promised_blocks and
    run start_load_kv; returns elapsed seconds."""
    conn.update_state_after_alloc(req, None, n_promised_blocks * 4)
    meta = conn.build_connector_meta(
        StubSchedulerOutput([StubNewReq(req, block_ids)],
                            {req.request_id: len(req.prompt_token_ids)})
    )
    conn.bind_connector_metadata(meta)
    t0 = time.monotonic()
    conn.start_load_kv(StubForwardContext(kv))  # must NEVER raise
    elapsed = time.monotonic() - t0
    conn.clear_connector_metadata()
    return elapsed


def test_kill9_mid_run_degrades_then_recovers(chaos_daemon):
    """kill -9 with primed (established) connections: the next load hits dead
    sockets, degrades to flagged misses within the bound, and — once a daemon
    is back on the SAME endpoint and the breaker window has passed — the
    connector redials and serves loads again. The engine never sees an
    exception at any point."""
    salt = "chaos-kill9"
    toks = list(range(600, 609))  # 2 blocks
    conn = make_conn(chaos_daemon)
    kv = fresh_kv()
    fill_block(kv, 2, seed=401)
    fill_block(kv, 5, seed=402)
    run_step(conn, StubRequest("k9-store", toks, salt), [2, 5], kv)
    n, _ = conn.get_num_new_matched_tokens(StubRequest("k9-probe", toks, salt), 0)
    assert n == 8  # stored AND the pooled connections are primed

    chaos_daemon["proc"].kill()  # SIGKILL: sockets die with the process
    chaos_daemon["proc"].wait(timeout=5)

    req = StubRequest("k9-load", toks, salt)
    elapsed = promised_load_step(conn, req, [4, 6], fresh_kv(), n_promised_blocks=2)
    assert elapsed < BOUND_S, f"degraded load took {elapsed:.2f}s (unbounded?)"
    assert conn.get_block_ids_with_load_errors() == {4, 6}  # flagged, not silent

    # Recovery: fresh daemon, SAME endpoint. The 5s dial breaker is the prod
    # backoff; collapsing it here IS "after the breaker window" at test speed.
    chaos_daemon["respawn"](chaos_daemon["port"])
    conn._next_dial = 0.0
    kv2 = fresh_kv()
    fill_block(kv2, 1, seed=403)
    fill_block(kv2, 3, seed=404)
    salt2 = "chaos-kill9-b"
    run_step(conn, StubRequest("k9-restore", toks, salt2), [1, 3], kv2)
    req2 = StubRequest("k9-reload", toks, salt2)
    n, _ = conn.get_num_new_matched_tokens(req2, 0)
    assert n == 8
    kv3 = fresh_kv()
    run_step(conn, req2, [6, 7], kv3)
    assert conn.get_block_ids_with_load_errors() == set()
    for name in kv3:
        assert torch.equal(kv3[name][6], kv2[name][1])
        assert torch.equal(kv3[name][7], kv2[name][3])
    conn.shutdown()


def test_sigstop_mid_run_bounded_then_recovers(chaos_daemon):
    """SIGSTOP freezes the daemon with connections ALIVE — the pathological
    hang: nothing errors, nothing answers. Loads must degrade within the
    timeout bound (flagged misses), and after SIGCONT the SAME stored blocks
    hit again (nothing was lost, only delayed)."""
    salt = "chaos-stop"
    toks = list(range(620, 629))
    conn = make_conn(chaos_daemon)
    kv = fresh_kv()
    fill_block(kv, 2, seed=405)
    fill_block(kv, 5, seed=406)
    run_step(conn, StubRequest("st-store", toks, salt), [2, 5], kv)
    n, _ = conn.get_num_new_matched_tokens(StubRequest("st-probe", toks, salt), 0)
    assert n == 8

    os.kill(chaos_daemon["proc"].pid, signal.SIGSTOP)
    try:
        req = StubRequest("st-load", toks, salt)
        elapsed = promised_load_step(conn, req, [4, 6], fresh_kv(), n_promised_blocks=2)
        assert elapsed < BOUND_S, f"paused-daemon load took {elapsed:.2f}s (hung?)"
        assert conn.get_block_ids_with_load_errors() == {4, 6}
    finally:
        os.kill(chaos_daemon["proc"].pid, signal.SIGCONT)

    conn._next_dial = 0.0  # breaker window over (prod: 5s)
    req2 = StubRequest("st-reload", toks, salt)
    n, _ = conn.get_num_new_matched_tokens(req2, 0)
    assert n == 8  # the blocks survived the pause
    kv2 = fresh_kv()
    run_step(conn, req2, [1, 3], kv2)
    assert conn.get_block_ids_with_load_errors() == set()
    for name in kv2:
        assert torch.equal(kv2[name][1], kv[name][2])
        assert torch.equal(kv2[name][3], kv[name][5])
    conn.shutdown()


def test_load_deadline_abandons_remaining_slab_passes():
    """A load bigger than the staging cap drains in passes; once the per-load
    deadline expires the remaining passes must be ABANDONED (their bids
    flagged), not trickled through one bounded op at a time."""
    import torch as _torch
    from kvblockd import protocol as kp

    cfg = StubVllmConfig(1)
    cfg.kv_transfer_config.kv_connector_extra_config.update(
        dict(FAST, kvblockd_staging_bytes=256,  # one 256B body per pass
             kvblockd_load_deadline_s=0.3))
    conn = KvblockdConnector(cfg, role="scheduler", kv_cache_config=None)
    conn._slab_path_ok = lambda dev: True  # force the pass-based slab lane on CPU
    conn._alloc_pinned = lambda n: _torch.empty(n, dtype=_torch.uint8)

    class SlowClient:
        calls = 0

        def batch_get_scatter(self, keys, prefix_len, alloc, deadline=None):
            SlowClient.calls += 1
            time.sleep(0.4)  # pass 1 alone blows the 0.3s load deadline
            return [kp.Status.NOT_FOUND] * len(keys)

        def close(self):
            return None

    conn._client = SlowClient()
    req = StubRequest("dl-load", list(range(660, 669)), "chaos-deadline")
    elapsed = promised_load_step(conn, req, [4, 6], fresh_kv(), n_promised_blocks=2)
    assert SlowClient.calls == 1, "pass 2 ran after the load deadline expired"
    assert elapsed < 1.0  # one pass, not len(keys) passes
    assert conn.get_block_ids_with_load_errors() == {4, 6}  # both flagged
    conn.shutdown()


def test_daemon_down_at_boot_serves_pure_recompute():
    """No daemon was EVER reachable: the connector must construct, answer the
    first lookup within ~connect_timeout, answer subsequent calls instantly
    (dial breaker), and run promised-load/store steps as flagged recompute —
    never an exception, never an unbounded wait."""
    cfg = StubVllmConfig(1)  # nothing listens on port 1
    cfg.kv_transfer_config.kv_connector_extra_config.update(FAST)
    conn = KvblockdConnector(cfg, role="scheduler", kv_cache_config=None)
    req = StubRequest("boot-1", list(range(640, 649)), "chaos-boot")

    t0 = time.monotonic()
    assert conn.get_num_new_matched_tokens(req, 0) == (0, False)
    first = time.monotonic() - t0
    assert first < FAST["kvblockd_connect_timeout_s"] + 1.0, f"first miss took {first:.2f}s"

    t0 = time.monotonic()
    assert conn.get_num_new_matched_tokens(req, 0) == (0, False)
    assert time.monotonic() - t0 < 0.1, "breaker-suppressed miss was not instant"

    kv = fresh_kv()
    elapsed = promised_load_step(conn, req, [0, 1], kv, n_promised_blocks=2)
    assert elapsed < BOUND_S
    assert conn.get_block_ids_with_load_errors() == {0, 1}
    for name, t in kv.items():
        conn.save_kv_layer(name, t, None)
    conn.wait_for_save()  # store path: enqueue + background failure, no raise
    conn.shutdown()
