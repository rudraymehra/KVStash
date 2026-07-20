"""Guarded SGLang imports.

The backend must IMPORT (and instantiate) with no sglang installed — the
A6-style CI tripwire and the pip-install-from-clean-venv check both rely on
it. Only the ABC is taken from sglang when present; everything else is
duck-typed. When sglang is absent we substitute a minimal base class with
the same non-abstract helpers the real ABC provides.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("sglang_kvblockd")

try:  # broad on purpose: a broken sglang install must not break OUR import;
    # the CI tripwire's --require-sglang leg is what distinguishes the cases.
    from sglang.srt.mem_cache.hicache_storage import (  # type: ignore
        STORAGE_BATCH_SIZE,
        HiCacheStorage,
        HiCacheStorageConfig,
        HiCacheStorageExtraInfo,
    )

    HAS_SGLANG = True
except Exception as _e:  # noqa: BLE001
    HAS_SGLANG = False
    _IMPORT_ERROR = _e
    STORAGE_BATCH_SIZE = 128  # upstream constant (hicache_storage.py)
    HiCacheStorageConfig = None  # type: ignore[assignment]
    HiCacheStorageExtraInfo = None  # type: ignore[assignment]

    class HiCacheStorage:  # type: ignore[no-redef]
        """Shim base: the two non-abstract hooks our subclass leans on. The
        real ABC's abstract methods are all implemented by the subclass, so
        swapping bases changes nothing but the isinstance chain."""

        def register_mem_pool_host(self, mem_pool_host):
            self.mem_pool_host = mem_pool_host

        def register_mem_host_pool_v2(self, host_pool, host_pool_name):
            if not hasattr(self, "registered_pools"):
                self.registered_pools = {}
            self.registered_pools[host_pool_name] = host_pool


def storage_metrics_cls():
    """Resolve SGLang's StorageMetrics dataclass across its module moves
    (sglang.srt.observability.metrics_collector @ 0.5.15; earlier releases
    kept it in sglang.srt.metrics.collector). None ⇒ caller falls back to a
    plain dict."""
    try:
        from sglang.srt.observability.metrics_collector import StorageMetrics

        return StorageMetrics
    except Exception:  # noqa: BLE001
        pass
    try:
        from sglang.srt.metrics.collector import StorageMetrics  # type: ignore

        return StorageMetrics
    except Exception:  # noqa: BLE001
        return None
