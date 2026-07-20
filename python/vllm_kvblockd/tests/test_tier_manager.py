"""Tier-manager suite: the vLLM/GPU surface is MOCKED (synthetic primary
memoryview standing in for the /dev/shm mmap, hand-built JobMetadata with
numpy block_ids, SimpleNamespace OffloadingSpec) while the kvblockd wire is
REAL (daemon fixture). GPU e2e under OffloadingConnector is deferred —
python/vllm_kvblockd/DEFER.md has the revisit trigger."""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest
from blake3 import blake3

from vllm_kvblockd.tier_manager import KvblockdTierManager, LookupResult, _DualQueuePool

BS = 4096      # bytes per primary block
NBLOCKS = 16


def fake_spec():
    """The OffloadingSpec fields tier_fingerprint_fields() reads (mirrors
    vllm.v1.kv_offload.base.OffloadingSpec's attributes)."""
    return SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(model="facebook/opt-125m"),
            cache_config=SimpleNamespace(cache_dtype="bfloat16", block_size=16),
        ),
        hash_block_size=16,
        block_size_factor=1,
    )


def fake_view():
    """(NBLOCKS, BS) memoryview — same shape contract as the framework's
    primary KV view (leading stride == bytes per block)."""
    backing = bytearray(NBLOCKS * BS)
    return memoryview(backing).cast("B", (NBLOCKS, BS)), backing


def okey(tag: str, i: int, group: int = 0) -> bytes:
    """OffloadKey = 32B chain-hash stand-in + 4B big-endian group index."""
    return blake3(f"{tag}-{i}".encode()).digest() + group.to_bytes(4, "big")


def job(job_id, keys, block_ids):
    return SimpleNamespace(
        job_id=job_id,
        keys=list(keys),
        block_ids=np.array(block_ids, dtype=np.int64),
        is_promotion=False,
        req_context=SimpleNamespace(req_id=f"req-{job_id}"),
    )


def make_manager(daemon, view, **kw):
    kw.setdefault("n_read_threads", 2)
    kw.setdefault("n_write_threads", 2)
    kw.setdefault("streams", 2)
    return KvblockdTierManager(
        fake_spec(), view, "kvblockd",
        endpoint=f"kvblockd://{daemon['host']}:{daemon['port']}",
        namespace=daemon["namespace"], token=daemon["token"], **kw,
    )


def wait_jobs(mgr, want_ids, timeout=10.0):
    """Poll get_finished_jobs until every id in want_ids reported."""
    got = {}
    deadline = time.time() + timeout
    while time.time() < deadline and set(want_ids) - set(got):
        for r in mgr.get_finished_jobs():
            got[r.job_id] = r.success
        time.sleep(0.02)
    assert set(want_ids) <= set(got), f"jobs never finished: got {got}"
    return got


def poll_lookup(mgr, key, ctx, timeout=5.0):
    """Drive the RETRY protocol the scheduler would: lookup, flush at step
    end, retry — until the batcher resolves."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = mgr.lookup(key, ctx)
        if r != LookupResult.RETRY:
            return r
        mgr.on_schedule_end(None)
        time.sleep(0.02)
    raise AssertionError("lookup never resolved past RETRY")


def test_store_load_byte_exact_roundtrip(daemon):
    view, backing = fake_view()
    mgr = make_manager(daemon, view)
    try:
        keys = [okey("rt", i) for i in range(3)]
        patterns = [bytes([(i * 37 + j) % 256 for j in range(BS)]) for i in range(3)]
        for i, p in enumerate(patterns):
            backing[i * BS : (i + 1) * BS] = p

        mgr.submit_store(job(1, keys, [0, 1, 2]))
        assert wait_jobs(mgr, {1}) == {1: True}

        # Wipe the source region, load into DIFFERENT slots (promotion path).
        backing[:] = bytes(len(backing))
        mgr.submit_load(job(2, keys, [8, 9, 10]))
        assert wait_jobs(mgr, {2}) == {2: True}
        for i, p in enumerate(patterns):
            assert backing[(8 + i) * BS : (9 + i) * BS] == p, f"block {i} corrupt"
        # Untouched slots stayed zero (loads write ONLY their slices).
        assert backing[11 * BS : 12 * BS] == bytes(BS)
    finally:
        mgr.shutdown()


def test_lookup_retry_then_hit_then_miss(daemon):
    view, backing = fake_view()
    mgr = make_manager(daemon, view)
    try:
        stored, absent = okey("lk", 1), okey("lk", 2)
        backing[0:BS] = bytes([7]) * BS
        mgr.submit_store(job(3, [stored], [0]))
        wait_jobs(mgr, {3})

        ctx = SimpleNamespace(req_id="lk-req")
        # First sight is RETRY by construction (nothing resolved yet).
        assert mgr.lookup(stored, ctx) == LookupResult.RETRY
        assert poll_lookup(mgr, stored, ctx) == LookupResult.HIT
        assert poll_lookup(mgr, absent, ctx) == LookupResult.MISS

        # cleanup drops the cached state; a re-ask goes through RETRY again.
        mgr.on_request_finished(ctx)
        assert mgr.lookup(stored, ctx) == LookupResult.RETRY
    finally:
        mgr.shutdown()


def test_group_index_partitions_keys(daemon):
    """Same chain hash, different KV-cache-group index -> different blob."""
    view, backing = fake_view()
    mgr = make_manager(daemon, view)
    try:
        k_g0, k_g1 = okey("grp", 1, group=0), okey("grp", 1, group=1)
        backing[0:BS] = bytes([1]) * BS
        mgr.submit_store(job(4, [k_g0], [0]))
        wait_jobs(mgr, {4})
        ctx = SimpleNamespace(req_id="grp-req")
        assert poll_lookup(mgr, k_g0, ctx) == LookupResult.HIT
        assert poll_lookup(mgr, k_g1, ctx) == LookupResult.MISS
    finally:
        mgr.shutdown()


def test_load_of_missing_key_fails_job_not_process(daemon):
    view, backing = fake_view()
    mgr = make_manager(daemon, view)
    try:
        mgr.submit_load(job(5, [okey("gone", 9)], [4]))
        assert wait_jobs(mgr, {5}) == {5: False}
        assert backing[4 * BS : 5 * BS] == bytes(BS)  # slot untouched
    finally:
        mgr.shutdown()


def test_drain_jobs_waits_never_aborts(daemon):
    view, backing = fake_view()
    mgr = make_manager(daemon, view, tile_keys=1)
    try:
        keys = [okey("drain", i) for i in range(8)]
        for i in range(8):
            backing[i * BS : (i + 1) * BS] = bytes([i + 1]) * BS
        mgr.submit_store(job(6, keys, list(range(8))))
        mgr.drain_jobs()  # must BLOCK until all 8 tiles copied
        # After drain, the result is available without further waiting...
        results = {r.job_id: r.success for r in mgr.get_finished_jobs()}
        assert results.get(6) is True
        # ...and every block really did land (no aborted mid-flight copy).
        wiped = bytearray(len(backing))
        backing[:] = wiped
        mgr.submit_load(job(7, keys, list(range(8))))
        mgr.drain_jobs()
        assert {r.job_id: r.success for r in mgr.get_finished_jobs()} == {7: True}
        for i in range(8):
            assert backing[i * BS : (i + 1) * BS] == bytes([i + 1]) * BS
    finally:
        mgr.shutdown()


def test_zero_key_job_still_reports(daemon):
    view, _ = fake_view()
    mgr = make_manager(daemon, view)
    try:
        mgr.submit_store(job(8, [], []))
        assert wait_jobs(mgr, {8}) == {8: True}
    finally:
        mgr.shutdown()


def test_touch_is_fire_and_forget(daemon):
    view, backing = fake_view()
    mgr = make_manager(daemon, view)
    try:
        k = okey("touch", 1)
        backing[0:BS] = bytes([3]) * BS
        mgr.submit_store(job(9, [k], [0]))
        wait_jobs(mgr, {9})
        mgr.touch([k], SimpleNamespace(req_id="t-req"))
        mgr.drain_jobs()  # advisory work must not deadlock drain
        assert mgr.get_finished_jobs() == []  # ...and never reports a job
    finally:
        mgr.shutdown()


def test_never_raise_daemon_dead():
    """Dead endpoint: lookups resolve MISS, jobs report failure, shutdown is
    clean — the scheduler process must never see an exception from us."""
    view, backing = fake_view()
    mgr = KvblockdTierManager(
        fake_spec(), view, "kvblockd",
        endpoint="kvblockd://127.0.0.1:1", namespace="vllm", token="tok",
        n_read_threads=1, n_write_threads=1, streams=1, op_timeout=0.5,
    )
    try:
        ctx = SimpleNamespace(req_id="dead-req")
        assert poll_lookup(mgr, okey("dead", 1), ctx, timeout=15.0) == LookupResult.MISS
        mgr.submit_store(job(10, [okey("dead", 2)], [0]))
        got = wait_jobs(mgr, {10}, timeout=15.0)
        assert got == {10: False}
    finally:
        mgr.shutdown()


def test_pool_shutdown_reports_queued_jobs_failed():
    """Shutdown must resolve EVERY job exactly once: running tasks finish
    (never aborted mid-copy), queued-but-unstarted ones are cancelled and
    their jobs reported FAILED — and the accounting must stay consistent so
    has_pending/wait_idle cannot hang or go negative afterwards."""
    import threading

    pool = _DualQueuePool(1, 0, name="t_shutdown")  # single worker
    gate = threading.Event()
    started = threading.Event()

    def slow():
        started.set()
        assert gate.wait(5)

    pool.enqueue_load(1, 1, [slow])
    assert started.wait(2)
    pool.enqueue_load(2, 1, [lambda: None])  # stuck behind the running task

    t = threading.Thread(target=pool.shutdown)
    t.start()
    time.sleep(0.05)  # let shutdown drain the queue while job 1 runs
    gate.set()
    t.join(5)
    assert not t.is_alive()

    got = dict(pool.get_finished())
    assert got == {1: True, 2: False}
    assert pool.has_pending() is False
    pool.wait_idle()  # must return immediately, never hang post-shutdown

    # Work enqueued AFTER shutdown fails immediately (still one result/job).
    pool.enqueue_load(3, 1, [lambda: None])
    assert dict(pool.get_finished()) == {3: False}
    pool.enqueue_fire_and_forget(lambda: None)  # advisory: dropped, no result
    assert pool.get_finished() == []


def test_manager_shutdown_never_strands_jobs():
    """Manager-level: a store job racing shutdown (dead daemon) surfaces
    exactly one failed JobResult; post-shutdown submits fail immediately."""
    view, _ = fake_view()
    mgr = KvblockdTierManager(
        fake_spec(), view, "kvblockd",
        endpoint="kvblockd://127.0.0.1:1", namespace="vllm", token="tok",
        n_read_threads=1, n_write_threads=1, streams=1, op_timeout=0.5,
    )
    mgr.submit_store(job(20, [okey("sd", 1)], [0]))
    mgr.shutdown()
    assert {r.job_id: r.success for r in mgr.get_finished_jobs()} == {20: False}
    mgr.submit_store(job(21, [okey("sd", 2)], [1]))
    assert {r.job_id: r.success for r in mgr.get_finished_jobs()} == {21: False}
    mgr.drain_jobs()  # gated on _stop: returns, never hangs
    assert mgr.has_pending_work() is False


def test_rfc_shaped_config_kwargs_accepted(daemon):
    """SPEC-5-§2.4-shaped tier dicts carry module_path/class_name; the merged
    factory forwards them as kwargs — they must not crash the constructor."""
    view, _ = fake_view()
    mgr = make_manager(
        daemon, view, module_path="vllm_kvblockd.tier_manager", class_name="KvblockdTierManager"
    )
    mgr.shutdown()


def test_block_size_inference_and_override():
    view, _ = fake_view()
    spec = fake_spec()
    m = KvblockdTierManager(spec, view, "kvblockd", endpoint="kvblockd://127.0.0.1:1",
                            n_read_threads=1, n_write_threads=1)
    try:
        assert m._block_size == BS  # inferred from strides[0], fs-manager style
    finally:
        m.shutdown()
    # A flat 1D view carries no block geometry: refuse unless told.
    flat = memoryview(bytearray(4 * BS))
    with pytest.raises(ValueError):
        KvblockdTierManager(spec, flat, "kvblockd", endpoint="kvblockd://127.0.0.1:1",
                            n_read_threads=1, n_write_threads=1)
    m2 = KvblockdTierManager(spec, flat, "kvblockd", endpoint="kvblockd://127.0.0.1:1",
                             block_bytes=BS, n_read_threads=1, n_write_threads=1)
    try:
        assert m2._block_size == BS
    finally:
        m2.shutdown()


def test_fingerprint_partitions_daemons_keyspace(daemon):
    """Two managers with different model configs never see each other's
    blobs, even for identical OffloadKeys (config identity lives in the key)."""
    view_a, backing_a = fake_view()
    mgr_a = make_manager(daemon, view_a)
    spec_b = fake_spec()
    spec_b.vllm_config.model_config.model = "meta-llama/Llama-3.1-8B"
    view_b, _ = fake_view()
    mgr_b = KvblockdTierManager(
        spec_b, view_b, "kvblockd",
        endpoint=f"kvblockd://{daemon['host']}:{daemon['port']}",
        namespace=daemon["namespace"], token=daemon["token"],
        n_read_threads=2, n_write_threads=2, streams=2,
    )
    try:
        k = okey("fp", 1)
        backing_a[0:BS] = bytes([9]) * BS
        mgr_a.submit_store(job(11, [k], [0]))
        wait_jobs(mgr_a, {11})
        ctx = SimpleNamespace(req_id="fp-req")
        assert poll_lookup(mgr_a, k, ctx) == LookupResult.HIT
        assert poll_lookup(mgr_b, k, ctx) == LookupResult.MISS
    finally:
        mgr_a.shutdown()
        mgr_b.shutdown()
