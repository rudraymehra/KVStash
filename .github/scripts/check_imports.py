#!/usr/bin/env python3
"""A6 interface tripwire: assert the vLLM/LMCache connector methods kvblockd's
adapters depend on still exist. Catches the day an upstream release renames them
(the incumbents' #1 documented pain).

Exit codes:
  0  all present, OR the target package isn't importable (SKIP — an install
     failure is visible in the pip step; don't mask real drift as it).
  1  package imported but a required symbol/method is MISSING (real drift).

Usage: check_imports.py --target lmcache|vllm|both
"""
import argparse
import inspect
import sys


def _missing(cls, names):
    return [n for n in names if not hasattr(cls, n)]


def check_lmcache() -> int:
    # Two-stage: absence of the top package = SKIP (install issue, visible in the
    # pip step). But if the package IS installed and the symbol/module path is
    # gone, that's exactly the drift we exist to catch → DRIFT, not SKIP.
    import importlib
    try:
        importlib.import_module("lmcache")
    except ImportError as e:
        print(f"SKIP: lmcache not installed ({e})")
        return 0
    try:
        from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
    except Exception as e:  # noqa: BLE001 — installed but path/symbol gone = drift
        print(f"DRIFT lmcache: import path/symbol changed ({type(e).__name__}: {e})")
        return 1

    fails = []
    # Breakage signal = a required method DISAPPEARED (rename/delete). We gate on
    # existence, not on whether it stays @abstractmethod — upstream giving one a
    # concrete default is normal evolution (method still there, adapter still
    # overrides it), so that's a WARN, not a false-RED on a scheduled run.
    required = ["exists", "exists_sync", "get", "put", "list", "close"]
    miss_required = _missing(RemoteConnector, required)
    if miss_required:
        fails.append(f"RemoteConnector missing required methods {miss_required}")
    have_abstract = set(getattr(RemoteConnector, "__abstractmethods__", frozenset()))
    no_longer_abstract = [m for m in required if m not in have_abstract]
    if no_longer_abstract:
        print(f"WARN: lmcache methods no longer abstract (concrete default added): {no_longer_abstract}")

    present = ["batched_contains", "batched_async_contains", "batched_get_non_blocking",
              "support_batched_async_contains", "support_batched_get_non_blocking"]
    miss = _missing(RemoteConnector, present)
    if miss:
        fails.append(f"RemoteConnector missing methods {miss}")

    for coro in ("batched_async_contains", "batched_get_non_blocking"):
        fn = getattr(RemoteConnector, coro, None)
        if fn is not None and not inspect.iscoroutinefunction(fn):
            fails.append(f"RemoteConnector.{coro} is no longer a coroutine (async overlap path changed)")

    if fails:
        print("DRIFT lmcache:\n  - " + "\n  - ".join(fails))
        return 1
    print("OK: lmcache RemoteConnector interface intact")
    return 0


def check_vllm() -> int:
    import importlib
    try:
        importlib.import_module("vllm")
    except ImportError as e:
        print(f"SKIP: vllm not installed ({e})")
        return 0
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
    except Exception as e:  # noqa: BLE001 — installed but path/symbol gone = drift
        print(f"DRIFT vllm: import path/symbol changed ({type(e).__name__}: {e})")
        return 1

    fails = []
    # Stable core — hard assert (this is what LMCache implements, rarely renamed).
    core = ["start_load_kv", "wait_for_layer_load", "save_kv_layer", "wait_for_save",
            "get_finished", "get_num_new_matched_tokens", "update_state_after_alloc",
            "build_connector_meta", "request_finished"]
    miss = _missing(KVConnectorBase_V1, core)
    if miss:
        fails.append(f"KVConnectorBase_V1 missing {miss}")

    # Tiering — soft/warn only (newer, churn-prone; classes move between releases).
    tiering = [
        ("vllm.v1.kv_offload.tiering.base", "SecondaryTierManager"),
        ("vllm.v1.kv_offload.tiering.async_lookup", "AsyncLookupManager"),
        ("vllm.v1.kv_offload.tiering.spec", "TieringOffloadingSpec"),
    ]
    for mod, name in tiering:
        try:
            m = __import__(mod, fromlist=[name])
            if not hasattr(m, name):
                print(f"WARN: {mod}.{name} not found (tiering churn — not a gate failure)")
        except Exception as e:  # noqa: BLE001
            print(f"WARN: cannot import {mod} ({type(e).__name__}) — tiering churn, not a gate failure")

    if fails:
        print("DRIFT vllm:\n  - " + "\n  - ".join(fails))
        return 1
    print("OK: vllm KVConnectorBase_V1 core interface intact")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["lmcache", "vllm", "both"], default="both")
    args = ap.parse_args()
    rc = 0
    if args.target in ("lmcache", "both"):
        rc |= check_lmcache()
    if args.target in ("vllm", "both"):
        rc |= check_vllm()
    return rc


if __name__ == "__main__":
    sys.exit(main())
