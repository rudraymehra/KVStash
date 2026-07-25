#!/usr/bin/env python3
"""Interface-drift tripwire: the adapter must IMPORT and INSTANTIATE against a stub
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
        remote_url = None
        extra_config = {
            "kvblockd_token": "t",
            "remote_storage_plugin.kvblockd.url":
                "kvblockd://127.0.0.1:9440?namespace=lmcache&streams=2",
        }

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
    # The plugin backend dials a VIRTUAL plugin:// url (lmcache 0.5.x
    # remote_backend.py); rejecting it is the silent zero-bytes failure.
    if not adapter.can_parse("plugin://kvblockd"):
        print("adapter.can_parse rejected plugin://kvblockd", file=sys.stderr)
        return 1
    if adapter.can_parse("plugin://redis"):
        print("adapter.can_parse claimed someone else's plugin url", file=sys.stderr)
        return 1
    conn = make_connector(StubContext())
    if not isinstance(conn, KvblockdRemoteConnector):
        print("make_connector did not build a KvblockdRemoteConnector", file=sys.stderr)
        return 1

    class StubPluginContext(StubContext):
        url = "plugin://kvblockd"

    conn2 = adapter.create_connector(StubPluginContext())
    if not isinstance(conn2, KvblockdRemoteConnector):
        print("plugin:// context did not build a KvblockdRemoteConnector", file=sys.stderr)
        return 1
    if conn2._addr != ("127.0.0.1", 9440):
        print(f"plugin:// endpoint resolution wrong: {conn2._addr}", file=sys.stderr)
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
