"""Pinned pre-warm suite: one eager slab allocation at the FIRST CUDA layout
capture (not the first load), disclosed with its measured duration; a failed
pre-warm leaves the existing lazy slab path untouched. CUDA is simulated by
overriding _slab_path_ok/_alloc_pinned — the CI box has no GPU."""

from __future__ import annotations

import logging

import pytest

torch = pytest.importorskip("torch")

from test_connector import StubForwardContext, StubVllmConfig, fresh_kv

from vllm_kvblockd.connector import KvblockdConnector


def make_conn(**extra):
    cfg = StubVllmConfig(1)  # nothing listens on port 1; prewarm never dials
    cfg.kv_transfer_config.kv_connector_extra_config.update(extra)
    return KvblockdConnector(cfg, role="worker", kv_cache_config=None)


def test_prewarm_at_first_cuda_capture(caplog):
    caplog.set_level(logging.WARNING, logger="vllm_kvblockd")
    conn = make_conn(kvblockd_staging_bytes=1 << 20, kvblockd_prewarm_bytes=4 << 20)
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
    assert called == [] and conn._slab is None
    conn.shutdown()
