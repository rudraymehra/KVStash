"""Pinned pre-warm suite: one eager allocation pass (load slab AND store
pool) at the FIRST CUDA layout capture (not the first load/store), each
disclosed with its measured duration; a failed pre-warm leaves the existing
lazy paths untouched. CUDA is simulated by overriding
_slab_path_ok/_alloc_pinned — the CI box has no GPU."""

from __future__ import annotations

import logging

import pytest

torch = pytest.importorskip("torch")

from test_async_store import TOTAL
from test_connector import StubForwardContext, StubVllmConfig, fresh_kv

from vllm_kvblockd.connector import KvblockdConnector


def make_conn(**extra):
    cfg = StubVllmConfig(1)  # nothing listens on port 1; prewarm never dials
    cfg.kv_transfer_config.kv_connector_extra_config.update(extra)
    return KvblockdConnector(cfg, role="worker", kv_cache_config=None)


def test_prewarm_at_first_cuda_capture(caplog):
    caplog.set_level(logging.WARNING, logger="vllm_kvblockd")
    # Store pool explicitly OFF (0 = no pool, no latch): this test pins down
    # the LOAD slab's prewarm alone; the pool has its own tests below.
    conn = make_conn(kvblockd_staging_bytes=1 << 20, kvblockd_prewarm_bytes=4 << 20,
                     kvblockd_store_staging_bytes=0)
    conn._slab_path_ok = lambda dev: True  # pretend the paged tensors are CUDA
    allocs = []

    def fake_pin(n):
        allocs.append(n)
        return torch.empty(n, dtype=torch.uint8)

    conn._alloc_pinned = fake_pin
    conn._capture_layers(StubForwardContext(fresh_kv()))  # capture — NOT a load
    assert allocs == [1 << 20], "prewarm size must be min(staging cap, prewarm bytes)"
    assert conn._slab is not None and conn._slab.numel() == 1 << 20
    lines = [r for r in caplog.records if "kvblockd pinned prewarm:" in r.getMessage()]
    assert len(lines) == 1 and lines[0].levelno == logging.WARNING
    assert f"{1 << 20} bytes" in lines[0].getMessage()  # the stall is published

    conn._capture_layers(StubForwardContext(fresh_kv()))  # second capture: no-op
    assert allocs == [1 << 20]
    assert len([r for r in caplog.records
                if "kvblockd pinned prewarm:" in r.getMessage()]) == 1
    conn.shutdown()


def test_prewarm_pins_pipeline_reserve_not_staging_cap():
    """RED-PROOF (R4): under the DEFAULT 2GiB staging cap the load path only
    ever reserves two 256MiB-capped slab halves — prewarming the whole cap
    pinned ~1.5GiB that no load would touch, on top of the ~1GiB store pool.
    The prewarm must pin exactly the pipeline reserve (2*half_blocks*body)."""
    conn = make_conn(kvblockd_store_staging_bytes=0)  # default staging: 2GiB
    conn._slab_path_ok = lambda dev: True
    allocs = []

    def fake_pin(n):
        allocs.append(n)
        return torch.empty(8, dtype=torch.uint8)  # size probe only

    conn._alloc_pinned = fake_pin
    conn._capture_layers(StubForwardContext(fresh_kv()))
    body = 256  # fresh_kv: 2 layers x 128B per block
    half_blocks = (256 << 20) // body  # min(cap/2=1GiB, 256MiB) // body
    assert allocs == [2 * half_blocks * body]  # 512MiB — the pipeline reserve
    assert allocs[0] < conn._staging_bytes     # never the whole 2GiB cap
    conn.shutdown()


def test_prewarm_bytes_still_bounds_the_pin():
    """kvblockd_prewarm_bytes stays the explicit override: it can shrink the
    pin below the pipeline reserve (it never grows past it)."""
    conn = make_conn(kvblockd_prewarm_bytes=1 << 20, kvblockd_store_staging_bytes=0)
    conn._slab_path_ok = lambda dev: True
    allocs = []

    def fake_pin(n):
        allocs.append(n)
        return torch.empty(8, dtype=torch.uint8)

    conn._alloc_pinned = fake_pin
    conn._capture_layers(StubForwardContext(fresh_kv()))
    assert allocs == [1 << 20]
    conn.shutdown()


def test_prewarm_failure_keeps_lazy_path():
    """cudaHostAlloc failing at prewarm must NOT latch the slab off — the
    lazy _slab_reserve path (with its own failure policy) keeps serving."""
    conn = make_conn(kvblockd_staging_bytes=1 << 20)
    conn._slab_path_ok = lambda dev: True

    def broken_pin(n):
        raise RuntimeError("cudaHostAlloc OOM (injected)")

    conn._alloc_pinned = broken_pin
    conn._capture_layers(StubForwardContext(fresh_kv()))  # must not raise
    assert conn._slab is None
    assert conn._slab_disabled is False  # the lazy path's latch is untouched
    conn._alloc_pinned = lambda n: torch.empty(n, dtype=torch.uint8)
    assert conn._slab_reserve(4096) is True  # lazy allocation still works
    assert conn._slab is not None
    conn.shutdown()


def test_prewarm_skipped_on_cpu():
    """The CPU backend never pins: capture with CPU paged tensors must not
    touch the allocator."""
    conn = make_conn()
    called = []
    conn._alloc_pinned = lambda n: called.append(n)
    conn._capture_layers(StubForwardContext(fresh_kv()))
    assert called == [] and conn._slab is None and conn._store_slab is None
    conn.shutdown()


def test_store_pool_prewarm_at_first_cuda_capture(caplog):
    """RED-PROOF: the pinned store pool is allocated at the first CUDA layout
    capture — not lazily inside the first measured wait_for_save — sized
    exactly as _store_pool_ready's lazy path would, with the measured
    duration published at WARNING like the load prewarm."""
    caplog.set_level(logging.WARNING, logger="vllm_kvblockd")
    conn = make_conn(kvblockd_staging_bytes=1 << 20,
                     kvblockd_store_staging_bytes=4 * TOTAL)
    conn._slab_path_ok = lambda dev: True
    allocs = []

    def fake_pin(n):
        allocs.append(n)
        return torch.empty(n, dtype=torch.uint8)

    conn._alloc_pinned = fake_pin
    conn._capture_layers(StubForwardContext(fresh_kv()))  # capture — NOT a store
    assert allocs == [1 << 20, 4 * TOTAL]  # load slab first, then the pool
    assert conn._store_slab is not None and conn._store_slots_total == 4
    assert conn._store_slot_stride == TOTAL
    lines = [r for r in caplog.records
             if "kvblockd pinned store-pool prewarm:" in r.getMessage()]
    assert len(lines) == 1 and lines[0].levelno == logging.WARNING
    assert f"{4 * TOTAL} bytes" in lines[0].getMessage()

    conn._capture_layers(StubForwardContext(fresh_kv()))  # second capture: no-op
    assert allocs == [1 << 20, 4 * TOTAL]
    conn.shutdown()


def test_store_pool_prewarm_respects_async_store_off():
    """With write-behind stores off the pool can never be used — prewarming
    it would pin RAM for nothing."""
    conn = make_conn(kvblockd_staging_bytes=1 << 20, kvblockd_async_store=False)
    conn._slab_path_ok = lambda dev: True
    conn._alloc_pinned = lambda n: torch.empty(n, dtype=torch.uint8)
    conn._capture_layers(StubForwardContext(fresh_kv()))
    assert conn._slab is not None and conn._store_slab is None
    assert not conn._store_slab_disabled
    conn.shutdown()


def test_load_slab_prewarm_failure_still_prewarms_store_pool():
    """The two prewarms fail independently: a load-slab pin failure must not
    starve the store pool of its eager allocation (and vice versa the pool's
    own latch stays with _store_pool_ready)."""
    conn = make_conn(kvblockd_staging_bytes=1 << 20,
                     kvblockd_store_staging_bytes=4 * TOTAL)
    conn._slab_path_ok = lambda dev: True
    calls = []

    def pin(n):
        calls.append(n)
        if len(calls) == 1:
            raise RuntimeError("cudaHostAlloc OOM (injected)")
        return torch.empty(n, dtype=torch.uint8)

    conn._alloc_pinned = pin
    conn._capture_layers(StubForwardContext(fresh_kv()))
    assert conn._slab is None and not conn._slab_disabled  # lazy path intact
    assert conn._store_slab is not None and conn._store_slots_total == 4
    conn.shutdown()
