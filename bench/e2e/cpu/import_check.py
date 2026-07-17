#!/usr/bin/env python3
"""A6-tripwire check: the adapter must IMPORT and INSTANTIATE against a stub
context across every pinned vLLM/LMCache release — proving interface drift
trips CI, not production. No daemon needed (the connector dials lazily).
Exit 1 on any failure."""

from __future__ import annotations

import sys


def main() -> int:
    try:
        from lmcache_kvblockd.adapter import KvblockdConnectorAdapter
        from lmcache_kvblockd.connector import KvblockdRemoteConnector, make_connector
    except Exception as e:  # noqa: BLE001
        print(f"import failed: {e}", file=sys.stderr)
        return 1

    class StubConfig:
        extra_config = {"kvblockd_token": "t"}

    class StubContext:
        url = "kvblockd://127.0.0.1:9440?namespace=lmcache&streams=2"
        local_cpu_backend = None
        config = StubConfig()
        metadata = None
        plugin_name = "kvblockd"

    adapter = KvblockdConnectorAdapter()
    if not adapter.can_parse(StubContext.url):
        print("adapter.can_parse rejected the kvblockd:// URL", file=sys.stderr)
        return 1
    conn = make_connector(StubContext())
    if not isinstance(conn, KvblockdRemoteConnector):
        print("make_connector did not build a KvblockdRemoteConnector", file=sys.stderr)
        return 1
    # Fast-path toggles must all report True (the whole point of the backend).
    for m in ("support_ping", "support_batched_get", "support_batched_put",
              "support_batched_contains", "support_batched_async_contains",
              "support_batched_get_non_blocking"):
        if getattr(conn, m)() is not True:
            print(f"{m}() is not True", file=sys.stderr)
            return 1
    print("import_check: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
