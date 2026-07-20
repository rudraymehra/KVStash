#!/usr/bin/env python3
"""A6-style tripwire check for the SGLang HiCache backend (socket #4).

Two legs:
  (default)         backend must IMPORT and INSTANTIATE with NO sglang
                    installed (shim base) and no daemon running.
  --require-sglang  sglang must be importable, the backend must subclass the
                    REAL HiCacheStorage ABC (SGLang's dynamic factory
                    enforces issubclass), instantiate via the factory's
                    calling convention `cls(storage_config, kwargs)`, and
                    the v2 stubs must raise NotImplementedError naming the
                    upstream issue.

Exit 1 on any failure. No daemon needed (the backend dials lazily).
"""

from __future__ import annotations

import sys


def main() -> int:
    require_sglang = "--require-sglang" in sys.argv[1:]

    try:
        from sglang_kvblockd import KvblockdHiCacheStorage
        from sglang_kvblockd._compat import HAS_SGLANG
    except Exception as e:  # noqa: BLE001
        print(f"import failed: {e}", file=sys.stderr)
        return 1

    if require_sglang:
        if not HAS_SGLANG:
            print("sglang required but the ABC import fell back to the shim",
                  file=sys.stderr)
            return 1
        from sglang.srt.mem_cache.hicache_storage import (
            HiCacheStorage,
            HiCacheStorageConfig,
        )

        if not issubclass(KvblockdHiCacheStorage, HiCacheStorage):
            print("backend is not a subclass of the real HiCacheStorage ABC "
                  "(StorageBackendFactory would reject it)", file=sys.stderr)
            return 1
        storage_config = HiCacheStorageConfig(
            tp_rank=0, tp_size=1, pp_rank=0, pp_size=1,
            attn_cp_rank=0, attn_cp_size=1,
            is_mla_model=False, enable_storage_metrics=False,
            is_page_first_layout=True, model_name="tripwire/model",
            extra_config={"endpoint": "kvblockd://127.0.0.1:9440",
                          "namespace": "tripwire", "token": "t",
                          "interface_v1": 1},
        )
    else:
        if HAS_SGLANG:
            print("note: sglang IS importable in this env; shim leg is moot")
        from types import SimpleNamespace

        storage_config = SimpleNamespace(
            tp_rank=0, tp_size=1, pp_rank=0, pp_size=1, is_mla_model=False,
            is_page_first_layout=True, model_name="tripwire/model",
            extra_config={"endpoint": "kvblockd://127.0.0.1:9440",
                          "namespace": "tripwire", "token": "t",
                          "interface_v1": 1},
        )

    # The dynamic factory's exact calling convention (verified at
    # v0.5.15.post1 backend_factory.py): backend_class(storage_config, kwargs).
    backend = KvblockdHiCacheStorage(storage_config, {})

    # The required interface surface must exist and be callable-shaped.
    for m in ("get", "batch_get", "set", "batch_set", "exists", "batch_exists",
              "batch_get_v1", "batch_set_v1", "register_mem_pool_host",
              "clear", "get_stats", "close"):
        if not callable(getattr(backend, m, None)):
            print(f"missing interface method: {m}", file=sys.stderr)
            backend.close()
            return 1

    # v2 stubs raise with the upstream issue reference (pre-registered ruling).
    for m in ("batch_exists_v2", "batch_get_v2", "batch_set_v2"):
        try:
            getattr(backend, m)([])
        except NotImplementedError as e:
            if "18239" not in str(e):
                print(f"{m} raised without the upstream issue reference",
                      file=sys.stderr)
                backend.close()
                return 1
        else:
            print(f"{m} did not raise NotImplementedError", file=sys.stderr)
            backend.close()
            return 1

    # Never-raise on a dead daemon: constructor never dials; a probe misses.
    backend.close()
    dead = KvblockdHiCacheStorage(storage_config, {
        "endpoint": "kvblockd://127.0.0.1:1",
        "op_timeout": 0.5, "connect_timeout": 0.3,
    })
    if dead.batch_exists(["deadbeef"]) != 0 or dead.exists("deadbeef") is not False:
        print("dead-daemon probe did not degrade to a miss", file=sys.stderr)
        dead.close()
        return 1
    dead.close()

    print(f"import_check: OK ({'sglang ABC' if require_sglang else 'shim'} leg)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
