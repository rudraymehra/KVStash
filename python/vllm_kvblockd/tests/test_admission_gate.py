"""Never-lose-to-recompute admission gate (kvblockd_min_hit_tokens) and the
token-scaled load deadline, both shipped INERT: the suite proves BOTH
directions — the knobs change behavior when set, and the defaults are
byte-for-byte today's behavior. The cost-crossover estimator knob
(kvblockd_recompute_ms_per_token) is REFUSED at boot until its EMA is
plumbed worker->scheduler — proven here too. A gated hit is a whole
refusal, never a truncation (causal attention makes the tail the expensive
end). Same conventions as test_connector.py: stub vLLM surface, REAL daemon,
per-test cache_salt keyspaces."""

from __future__ import annotations

import time

import pytest

torch = pytest.importorskip("torch")

from kvblockd import protocol as kp
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

from vllm_kvblockd.connector import BLOB_PREFIX_LEN, KvblockdConnector

TOTAL = BLOB_PREFIX_LEN + len(LAYERS) * (2 * BLOCK * HID * 2)


def make_conn(daemon, **extra):
    cfg = StubVllmConfig(daemon["port"])
    cfg.kv_transfer_config.kv_connector_extra_config.update(extra)
    return KvblockdConnector(cfg, role="scheduler", kv_cache_config=None)


def seed_two_blocks(daemon, salt, toks):
    """Store toks' two aligned blocks under salt; returns the stored kv."""
    conn = make_conn(daemon)
    kv = fresh_kv()
    fill_block(kv, 2, seed=601)
    fill_block(kv, 5, seed=602)
    run_step(conn, StubRequest("ag-seed-" + salt, toks, salt), [2, 5], kv)
    conn.shutdown()
    return kv


def test_min_hit_tokens_refuses_short_hits_whole(daemon):
    """A hit below the threshold is answered as a FULL miss — and a hit at
    the threshold is admitted WHOLE (no truncation in either direction)."""
    salt = "ag-min"
    toks = list(range(700, 709))  # 9 tokens -> 2 aligned blocks (8 tokens)
    seed_two_blocks(daemon, salt, toks)

    gated = make_conn(daemon, kvblockd_min_hit_tokens=9)
    assert gated.get_num_new_matched_tokens(StubRequest("ag1", toks, salt), 0) == (0, False)
    gated.shutdown()

    at_threshold = make_conn(daemon, kvblockd_min_hit_tokens=8)
    n, _ = at_threshold.get_num_new_matched_tokens(StubRequest("ag2", toks, salt), 0)
    assert n == 8, "an at-threshold hit must be admitted whole, never truncated"
    at_threshold.shutdown()

    inert = make_conn(daemon)  # default 0: today's behavior exactly
    assert inert.get_num_new_matched_tokens(StubRequest("ag3", toks, salt), 0)[0] == 8
    inert.shutdown()


def test_min_hit_tokens_gates_the_external_delta(daemon):
    """The threshold applies to what would actually be LOADED (the external
    tokens beyond the local prefix-cache hit) — that is the cost being
    weighed against recompute, not the absolute prefix length."""
    salt = "ag-delta"
    toks = list(range(710, 719))
    seed_two_blocks(daemon, salt, toks)
    conn = make_conn(daemon, kvblockd_min_hit_tokens=5)
    # 8 stored - 4 local = 4 external < 5: refused.
    assert conn.get_num_new_matched_tokens(StubRequest("ag4", toks, salt), BLOCK) == (0, False)
    # 8 external >= 5: admitted whole.
    assert conn.get_num_new_matched_tokens(StubRequest("ag5", toks, salt), 0)[0] == 8
    conn.shutdown()


def test_min_hit_tokens_gates_async_lookup(daemon):
    """The async lookup path answers through the same gate."""
    salt = "ag-async"
    toks = list(range(720, 729))
    seed_two_blocks(daemon, salt, toks)
    conn = make_conn(daemon, kvblockd_async_lookup=True, kvblockd_min_hit_tokens=9)
    req = StubRequest("ag6", toks, salt)
    assert conn.get_num_new_matched_tokens(req, 0) == (None, False)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        n, is_async = conn.get_num_new_matched_tokens(req, 0)
        assert is_async is False
        if n is not None:
            assert n == 0, "the resolved async hit must pass the gate"
            break
        time.sleep(0.01)
    else:
        pytest.fail("async lookup never resolved")
    conn.shutdown()


def test_estimator_refuses_when_loading_cannot_pay(daemon):
    """kvblockd_recompute_ms_per_token is REFUSED at boot, not honored: the
    throughput EMA lives in the worker-role connector and the gate runs in
    the scheduler-role connector (separate objects/processes in vLLM), so a
    nonzero value would silently admit every hit forever — the estimator can
    only refuse-when-loading-cannot-pay once the EMA is plumbed cross-role.
    A fused single-instance harness must not fake that state; this proves
    the loud refusal instead. Explicit 0 still constructs (inert)."""
    with pytest.raises(ValueError, match="not yet plumbed cross-role"):
        make_conn(daemon, kvblockd_recompute_ms_per_token=1.0)
    conn = make_conn(daemon, kvblockd_recompute_ms_per_token=0)
    toks = list(range(730, 739))
    seed_two_blocks(daemon, "ag-est", toks)
    assert conn.get_num_new_matched_tokens(
        StubRequest("ag7", toks, "ag-est"), 0)[0] == 8  # 0 = inert: admits
    conn.shutdown()


def test_loads_feed_the_estimator_ema(daemon):
    """A real completed load must observe throughput into the EMA (the
    estimator's only evidence source)."""
    salt = "ag-ema"
    toks = list(range(740, 749))
    seed_two_blocks(daemon, salt, toks)
    conn = make_conn(daemon)
    assert conn._load_bps_ema is None
    req = StubRequest("ag10", toks, salt)
    n, _ = conn.get_num_new_matched_tokens(req, 0)
    assert n == 8
    run_step(conn, req, [1, 3], fresh_kv())
    assert conn.get_block_ids_with_load_errors() == set()
    assert conn._load_bps_ema is not None and conn._load_bps_ema > 0
    assert conn._load_bytes_per_token == TOTAL / BLOCK
    conn.shutdown()


class DeadlineSpy:
    """Client stand-in that records the armed GET deadline and misses."""

    def __init__(self):
        self.deadlines: list[float | None] = []

    def batch_exists(self, keys, deadline=None):
        return 0, None

    def batch_get_scatter(self, keys, prefix_len, alloc, deadline=None):
        self.deadlines.append(deadline)
        return [kp.Status.NOT_FOUND] * len(keys)

    def put(self, key, bufs, ttl_ms=0):
        return 0

    def close(self):
        pass


def drive_load(conn, req, block_ids, n_promised):
    conn.update_state_after_alloc(req, None, n_promised)
    meta = conn.build_connector_meta(StubSchedulerOutput(
        [StubNewReq(req, block_ids)], {req.request_id: len(req.prompt_token_ids)}))
    conn.bind_connector_metadata(meta)
    conn.start_load_kv(StubForwardContext(fresh_kv()))
    conn.get_block_ids_with_load_errors()  # drain the flags (misses expected)
    conn.clear_connector_metadata()


def test_load_deadline_scales_per_block_and_caps(daemon):
    """deadline budget = min(cap, base + per_block * n_load_blocks); the
    flat-compat defaults (per_block=0, cap unset) arm exactly base — proven
    against the deadline the client is actually handed."""
    toks = list(range(750, 759))  # 2 aligned blocks

    # Scaled: base 1.0 + 0.5 * 2 blocks = 2.0s (flat code would arm 1.0).
    conn = make_conn(daemon, kvblockd_load_deadline_s=1.0,
                     kvblockd_load_deadline_per_block_s=0.5)
    spy = DeadlineSpy()
    conn._client = spy
    t0 = time.monotonic()
    drive_load(conn, StubRequest("dl1", toks, "ag-dl1"), [0, 1], 2 * BLOCK)
    assert spy.deadlines and spy.deadlines[0] is not None
    budget = spy.deadlines[0] - t0
    assert 1.8 < budget <= 2.05, f"expected ~2.0s token-scaled budget, armed {budget:.2f}s"
    conn.shutdown()

    # Capped: base 1.0 + 100 * 2 blocks, cap 3.0 -> 3.0s.
    conn = make_conn(daemon, kvblockd_load_deadline_s=1.0,
                     kvblockd_load_deadline_per_block_s=100.0,
                     kvblockd_load_deadline_cap_s=3.0)
    spy = DeadlineSpy()
    conn._client = spy
    t0 = time.monotonic()
    drive_load(conn, StubRequest("dl2", toks, "ag-dl2"), [0, 1], 2 * BLOCK)
    budget = spy.deadlines[0] - t0
    assert 2.8 < budget <= 3.05, f"expected the 3.0s cap, armed {budget:.2f}s"
    conn.shutdown()

    # Flat-compat defaults: exactly base (today's behavior).
    conn = make_conn(daemon, kvblockd_load_deadline_s=1.0)
    spy = DeadlineSpy()
    conn._client = spy
    t0 = time.monotonic()
    drive_load(conn, StubRequest("dl3", toks, "ag-dl3"), [0, 1], 2 * BLOCK)
    budget = spy.deadlines[0] - t0
    assert 0.8 < budget <= 1.05, f"expected the flat 1.0s base, armed {budget:.2f}s"
    conn.shutdown()
