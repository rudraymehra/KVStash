#!/usr/bin/env python3
"""A6-tripwire check for the NATIVE vLLM adapter (vllm_kvblockd): both
classes must IMPORT and INSTANTIATE across every pinned vLLM release —
interface drift trips CI, not production. Mirrors import_check.py (the
LMCache-adapter tripwire). No daemon needed (clients dial lazily).

With vllm installed, the connector must be a real KVConnectorBase_V1 subclass
and instantiate through the 3-arg constructor; the tier manager must subclass
SecondaryTierManager and land in SecondaryTierFactory's registry. Where a
release lacks the tiering module, that HALF is a documented SKIP (printed,
exit 0) — recorded in python/README.md's version-support policy.

Exit 1 on any failure."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

# Import tripwire, not a determinism test: the hash-seed self-check has its
# own pytest coverage and would otherwise demand a pinned env here.
os.environ.setdefault("KVBLOCKD_SKIP_HASHSEED_CHECK", "1")


def _stub_vllm_config():
    ktc = SimpleNamespace(
        kv_connector_extra_config={
            "kvblockd_endpoint": "kvblockd://127.0.0.1:9440",
            "kvblockd_namespace": "vllm",
            "kvblockd_token": "t",
        }
    )
    return SimpleNamespace(
        kv_transfer_config=ktc,
        cache_config=SimpleNamespace(block_size=16, cache_dtype="auto"),
        model_config=SimpleNamespace(model="facebook/opt-125m", dtype="torch.bfloat16"),
        parallel_config=SimpleNamespace(world_size=1),
    )


def check_connector() -> int:
    from vllm_kvblockd.connector import KvblockdConnector

    role = "scheduler"
    have_vllm = False
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorBase_V1,
            KVConnectorRole,
        )

        have_vllm = True
        role = KVConnectorRole.SCHEDULER
        if not issubclass(KvblockdConnector, KVConnectorBase_V1):
            print("FAIL: KvblockdConnector is not a KVConnectorBase_V1 subclass", file=sys.stderr)
            return 1
    except ImportError:
        print("note: vllm absent — connector checked against fallback base")

    conn = KvblockdConnector(_stub_vllm_config(), role, kv_cache_config=None)
    for m in ("get_num_new_matched_tokens", "update_state_after_alloc",
              "build_connector_meta", "request_finished", "start_load_kv",
              "wait_for_layer_load", "save_kv_layer", "wait_for_save",
              "get_finished", "get_block_ids_with_load_errors"):
        if not callable(getattr(conn, m, None)):
            print(f"FAIL: connector missing {m}", file=sys.stderr)
            return 1
    conn.shutdown()
    print(f"import_check_vllm_native: connector OK (vllm={'yes' if have_vllm else 'no'})")
    return 0


def check_tier_manager() -> int:
    import numpy as np  # noqa: F401 - JobMetadata.block_ids contract

    from vllm_kvblockd.tier_manager import KvblockdTierManager

    tiering = False
    try:
        from vllm.v1.kv_offload.tiering.base import SecondaryTierManager
        from vllm.v1.kv_offload.tiering.factory import SecondaryTierFactory

        tiering = True
        if not issubclass(KvblockdTierManager, SecondaryTierManager):
            print("FAIL: KvblockdTierManager is not a SecondaryTierManager subclass",
                  file=sys.stderr)
            return 1
        if "kvblockd" not in SecondaryTierFactory._registry:
            print("FAIL: 'kvblockd' tier type not registered on import", file=sys.stderr)
            return 1
        import vllm_kvblockd.tier_manager as tm

        if not hasattr(tm, "KvblockdTieringSpec"):
            print("FAIL: KvblockdTieringSpec (spec_module_path vehicle) missing",
                  file=sys.stderr)
            return 1
        # Presence is not enough: an upstream abstractmethod addition breaks
        # CONSTRUCTION, not import. __new__ runs Python's real abstract-class
        # instantiation gate without needing a full VllmConfig/KVCacheConfig.
        try:
            tm.KvblockdTieringSpec.__new__(tm.KvblockdTieringSpec)
        except TypeError as e:
            print(f"FAIL: KvblockdTieringSpec not instantiable: {e}", file=sys.stderr)
            return 1
    except ImportError as e:
        # Documented skip: a release predating/moving the tiering surface still
        # gets the connector altitude; see python/README.md version policy.
        print(f"SKIP: vllm tiering module unavailable ({e}) — tier half not exercised")

    view = memoryview(bytearray(4 * 4096)).cast("B", (4, 4096))
    spec = SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(model="facebook/opt-125m"),
            cache_config=SimpleNamespace(cache_dtype="bfloat16", block_size=16),
        ),
        hash_block_size=16,
        block_size_factor=1,
    )
    mgr = KvblockdTierManager(
        spec, view, "kvblockd", endpoint="kvblockd://127.0.0.1:9440",
        namespace="vllm", token="t", n_read_threads=1, n_write_threads=1,
        module_path="vllm_kvblockd.tier_manager", class_name="KvblockdTierManager",
    )
    for m in ("lookup", "submit_store", "submit_load", "get_finished_jobs",
              "on_new_request", "drain_jobs", "shutdown"):
        if not callable(getattr(mgr, m, None)):
            print(f"FAIL: tier manager missing {m}", file=sys.stderr)
            return 1
    mgr.shutdown()
    print(f"import_check_vllm_native: tier manager OK (tiering={'yes' if tiering else 'SKIP'})")
    return 0


def main() -> int:
    try:
        rc = check_connector()
        rc |= check_tier_manager()
    except Exception as e:  # noqa: BLE001 — any surprise IS the tripwire firing
        print(f"import_check_vllm_native FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
