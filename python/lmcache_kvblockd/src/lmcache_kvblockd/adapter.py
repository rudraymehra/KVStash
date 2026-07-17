"""KvblockdConnectorAdapter — registers the kvblockd:// scheme with LMCache's
ConnectorManager (plugin path: remote_storage_plugins + extra_config
module_path/class_name; see docs/notes/lmcache-head-verify.md).

The LMCache ConnectorAdapter base is resolved at import time; if LMCache is
absent (the A6 import tripwire, or unit tests without lmcache) we fall back to
`object` so the module still imports and the class still instantiates."""

from __future__ import annotations

from .connector import make_connector

try:
    from lmcache.v1.storage_backend.connector import ConnectorAdapter as _Base
except Exception:  # LMCache not installed — keep the module importable
    _Base = object

SCHEME = "kvblockd://"


class KvblockdConnectorAdapter(_Base):
    """Matches kvblockd:// URLs and builds a KvblockdRemoteConnector."""

    def __init__(self, schema: str = SCHEME):
        try:
            super().__init__(schema)  # ConnectorAdapter(schema) when present
        except TypeError:
            self.schema = schema  # object() base: set it ourselves

    def can_parse(self, url: str) -> bool:
        return url.startswith(SCHEME)

    def create_connector(self, context):
        return make_connector(context)
