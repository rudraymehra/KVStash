"""Registration proof at the LMCache API level: LMCache 0.5.x actually SELECTS
our connector from the exact YAML the e2e uses. No GPU, no vLLM, no daemon —
the connector dials lazily, so CreateConnector succeeds with nothing listening.

Why this exists: on the first real GPU run, kvblockd received ZERO bytes while
vLLM served happily from LMCache's local tier. Root cause (lmcache 0.5.1,
lmcache/v1/storage_backend/remote_backend.py init_connection): a backend created
via `remote_storage_plugins` dials the VIRTUAL url "plugin://kvblockd", never
`remote_url` — and our adapter only matched "kvblockd://". CreateConnector then
raised "No adapter found", RemoteBackend caught it, logged one warning, kept
`connection = None`, and every put/contains silently became a miss. These tests
drive LMCache's real connector factory down BOTH paths so that failure mode can
never regress silently again.

Skips (does not fail) when lmcache is not installed — the light CI connector
job installs our packages with --no-deps; the registration jobs install real
lmcache and run this for real.
"""

from __future__ import annotations

import asyncio
import pathlib

import pytest

pytest.importorskip("lmcache", reason="registration tests need real lmcache")

from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.storage_backend.connector import CreateConnector

from lmcache_kvblockd.connector import KvblockdRemoteConnector

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
E2E_YAML = REPO_ROOT / "bench" / "e2e" / "cpu" / "lmcache_kvblockd.yaml"
ENDPOINT_KEY = "remote_storage_plugin.kvblockd.url"


@pytest.fixture()
def loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture()
def cfg():
    return LMCacheEngineConfig.from_file(str(E2E_YAML))


def _unwrap(conn):
    """CreateConnector returns InstrumentedRemoteConnector(_connector=inner)."""
    return getattr(conn, "_connector", conn)


def _close(loop, conn):
    loop.run_until_complete(_unwrap(conn).close())


def test_e2e_yaml_carries_the_plugin_keys(cfg):
    """LMCacheEngineConfig must parse every key our registration relies on."""
    assert cfg.remote_storage_plugins == ["kvblockd"]
    extra = cfg.extra_config
    assert extra["remote_storage_plugin.kvblockd.module_path"] == "lmcache_kvblockd.adapter"
    assert extra["remote_storage_plugin.kvblockd.class_name"] == "KvblockdConnectorAdapter"
    assert extra[ENDPOINT_KEY].startswith("kvblockd://")
    # remote_url must stay ABSENT: it would spawn a second (deprecated)
    # RemoteBackend next to the plugin one and double every put.
    assert cfg.remote_url is None


def test_plugin_url_selects_our_connector(cfg, loop):
    """THE path RemoteBackend takes for remote_storage_plugins: the virtual
    plugin://kvblockd url (remote_backend.py init_connection), not remote_url.
    This is exactly what silently failed on the GPU box."""
    conn = CreateConnector(
        "plugin://kvblockd", loop, None, cfg, None, plugin_name="kvblockd"
    )
    inner = _unwrap(conn)
    assert isinstance(inner, KvblockdRemoteConnector), type(inner)
    assert inner._addr == ("127.0.0.1", 9440)
    assert inner._namespace == "lmcache"
    assert inner._streams == 4
    assert inner._token == "e2e-token"
    _close(loop, conn)


def test_plugin_instance_name_selects_our_connector(cfg, loop):
    """plugin names may be '{type}.{instance}'; the type must still match."""
    conn = CreateConnector(
        "plugin://kvblockd.primary", loop, None, cfg, None,
        plugin_name="kvblockd.primary",
    )
    inner = _unwrap(conn)
    assert isinstance(inner, KvblockdRemoteConnector), type(inner)
    assert inner._addr == ("127.0.0.1", 9440)
    _close(loop, conn)


def test_legacy_kvblockd_url_still_selects_our_connector(cfg, loop):
    """The deprecated remote_url path (a literal kvblockd:// url) keeps working
    for configs in the wild that still use it."""
    url = cfg.extra_config[ENDPOINT_KEY]
    conn = CreateConnector(url, loop, None, cfg, None)
    inner = _unwrap(conn)
    assert isinstance(inner, KvblockdRemoteConnector), type(inner)
    assert inner._addr == ("127.0.0.1", 9440)
    assert inner._namespace == "lmcache"
    _close(loop, conn)


def test_wire_key_works_on_a_real_cache_engine_key():
    """_wire must accept lmcache's REAL CacheEngineKey. The 0.5.x key has NO
    fmt field; the old _wire read key.fmt, raised AttributeError on every real
    key, and the never-raise wrappers silently turned that into a permanent
    miss — invisible to any test built on fakes. Also: dtype must be part of
    the wire identity (same tokens, different dtype => different keys)."""
    torch = pytest.importorskip("torch")
    from lmcache.utils import CacheEngineKey

    k16 = CacheEngineKey("facebook/opt-125m", 1, 0, 0xDEADBEEF, torch.bfloat16)
    k32 = CacheEngineKey("facebook/opt-125m", 1, 0, 0xDEADBEEF, torch.float32)
    w16 = KvblockdRemoteConnector._wire(k16)
    assert isinstance(w16, bytes) and len(w16) == 32
    assert KvblockdRemoteConnector._wire(k16) == w16  # deterministic
    assert KvblockdRemoteConnector._wire(k32) != w16  # dtype-distinct


def test_plugin_without_endpoint_fails_loud_not_silent(cfg, loop):
    """A plugin backend with no endpoint anywhere must raise an ACTIONABLE
    error (RemoteBackend logs it), never hand back a connector aimed at a
    default address."""
    cfg_dict = cfg.to_dict()
    cfg_dict["extra_config"] = {
        k: v for k, v in cfg.extra_config.items() if k != ENDPOINT_KEY
    }
    cfg_dict["remote_url"] = None
    broken = LMCacheEngineConfig.from_dict(cfg_dict)
    with pytest.raises(ValueError, match=ENDPOINT_KEY.replace(".", r"\.")):
        CreateConnector(
            "plugin://kvblockd", loop, None, broken, None, plugin_name="kvblockd"
        )
