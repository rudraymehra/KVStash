"""KvblockdConnectorAdapter — registers the kvblockd:// scheme with LMCache's
ConnectorManager (plugin path: remote_storage_plugins + extra_config
module_path/class_name; see docs/notes/lmcache-head-verify.md).

TWO URL shapes must parse, because LMCache 0.5.x dials remote backends two
different ways (lmcache/v1/storage_backend/remote_backend.py, init_connection):

  1. legacy `remote_url` backend  → url = "kvblockd://host:port?..."
  2. plugin backend (the `remote_storage_plugins` path) → url =
     "plugin://kvblockd" — a VIRTUAL url carrying no endpoint at all.

An adapter that only matches kvblockd:// makes the plugin backend fail
"No adapter found for URL: plugin://kvblockd"; RemoteBackend swallows that
(connection stays None) and every op silently degrades to a miss — the
GPU-box zero-bytes failure. For plugin:// the real endpoint comes from
extra_config["remote_storage_plugin.kvblockd.url"] (falling back to
config.remote_url when it is a kvblockd:// url). This mirrors how builtin
adapters (fs_adapter.py) treat plugin:// URLs in lmcache 0.5.1.

The LMCache ConnectorAdapter base is resolved at import time; if LMCache is
absent (CI import checks, or unit tests without lmcache) we fall back to
`object` so the module still imports and the class still instantiates."""

from __future__ import annotations

from .connector import make_connector

try:
    from lmcache.v1.storage_backend.connector import ConnectorAdapter as _Base
except Exception:  # noqa: BLE001 — availability fallback: LMCache not installed — keep the module importable
    _Base = object

SCHEME = "kvblockd://"
PLUGIN_SCHEME = "plugin://"
PLUGIN_TYPE = "kvblockd"


def _plugin_type(name: str) -> str:
    """Mirror lmcache's extract_plugin_type: '{type}' or '{type}.{instance}'."""
    return name.split(".", 1)[0]


class KvblockdConnectorAdapter(_Base):
    """Matches kvblockd:// and plugin://kvblockd[.instance] URLs and builds a
    KvblockdRemoteConnector."""

    def __init__(self, schema: str = SCHEME):
        try:
            super().__init__(schema)  # ConnectorAdapter(schema) when present
        except TypeError:
            self.schema = schema  # object() base: set it ourselves

    def can_parse(self, url: str) -> bool:
        if url.startswith(SCHEME):
            return True
        if url.startswith(PLUGIN_SCHEME):
            return _plugin_type(url[len(PLUGIN_SCHEME):]) == PLUGIN_TYPE
        return False

    def create_connector(self, context):
        return make_connector(context, url=self._resolve_url(context))

    @staticmethod
    def _resolve_url(context) -> str:
        """The real kvblockd:// endpoint for this context.

        plugin:// URLs are virtual — the endpoint lives in config. Raising here
        is deliberate and SAFE: RemoteBackend.init_connection catches it and
        logs a warning, and a loud misconfiguration beats a silent no-op."""
        url = getattr(context, "url", "") or ""
        if url.startswith(SCHEME):
            return url
        cfg = getattr(context, "config", None)
        extra = getattr(cfg, "extra_config", None) or {}
        pname = getattr(context, "plugin_name", None) or url[len(PLUGIN_SCHEME):] or PLUGIN_TYPE
        # Full instance name first ("kvblockd.primary"), then the bare type.
        for name in dict.fromkeys((pname, _plugin_type(pname))):
            candidate = extra.get(f"remote_storage_plugin.{name}.url")
            if candidate:
                return candidate
        remote_url = getattr(cfg, "remote_url", None)
        if remote_url and remote_url.startswith(SCHEME):
            return remote_url
        raise ValueError(
            f"kvblockd plugin selected (url={url!r}) but no endpoint configured: set "
            f"extra_config['remote_storage_plugin.{_plugin_type(pname)}.url'] = "
            f"'kvblockd://HOST:PORT?namespace=...&streams=...'"
        )
