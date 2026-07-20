"""KvblockdHiCacheStorage — SGLang HiCacheStorage (v1 interface) backed by
kvblockd. Interface pinned against sglang v0.5.15.post1 (tag code-read
2026-07-20); CPU-validated only — the GPU e2e verdict is DEFER, see
docs/design/sglang-hicache-v1.1.md.

Three contracts dominate the design (they are the W5 LMCache connector's,
transplanted to a synchronous surface):

  NEVER RAISE on the data path. SGLang's HiCacheController threads treat a
  raised exception as fatal; a False/0/None is just a cache miss. Every
  public v1/generic method is wrapped so any failure degrades to a miss,
  logged at most once per interval. (The v2 stubs are the deliberate
  exception — they raise NotImplementedError by ruling, see below.)

  ZERO-COPY GET into the pinned L2 host pool. register_mem_pool_host maps
  the whole pool tensor once (ctypes over data_ptr, Mooncake-style whole-
  pool registration); at batch_get_v1 time host_indices resolve — via the
  pool's own get_page_buffer_meta — to (ptr,size) pairs we translate into
  memoryview slices of that mapping, and the wire client recv_into's tensor
  bytes straight into them. No intermediate copy on the read path. (PUT
  bodies are copied once into the outgoing frame by the wire client —
  send-side MSG_ZEROCOPY is the daemon's job, not Python's.)

  RANK/MODEL ISOLATION by construction. keymap folds a model/tp/pp
  fingerprint into every 32-byte key, so mismatched deployments sharing a
  namespace miss instead of silently corrupting each other (see keymap.py).

v2 (PoolTransfer / CacheControllerV2) is stubbed: the upstream surface is
still churning (sgl-project/sglang#18239) and the week-12 ruling is to not
build against it until it stabilizes in a tagged release.
"""

from __future__ import annotations

import ctypes
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional
from urllib.parse import urlparse

from kvblockd import protocol as kp
from kvblockd.client import Client

from sglang_kvblockd import keymap
from sglang_kvblockd._compat import HiCacheStorage, storage_metrics_cls

logger = logging.getLogger("sglang_kvblockd")

_V2_MSG = (
    "kvblockd does not implement the HiCache v2 controller interface "
    "(PoolTransfer/CacheControllerV2): upstream is still churning it — see "
    "https://github.com/sgl-project/sglang/issues/18239. Pre-registered "
    "week-12 ruling: do not build against v2 until it stabilizes in a "
    "tagged SGLang release; run this backend with extra-config "
    '{"interface_v1": 1}.'
)


def _torch():  # lazy so the package imports (and instantiates) without torch
    import torch

    return torch


class _RateLimitedLog:
    """One full traceback per key on first sight, then one terse line per
    interval. Per-instance state (mirrors the W5 connector)."""

    def __init__(self, interval: float = 10.0):
        self._interval = interval
        self._last: dict[str, float] = {}
        self._full: set[str] = set()

    def maybe(self, key: str, msg: str, exc: BaseException | None = None):
        now = time.monotonic()
        if now - self._last.get(key, 0.0) < self._interval:
            return
        self._last[key] = now
        if exc is None:
            logger.warning("%s", msg)
        elif key not in self._full:
            self._full.add(key)
            logger.warning("%s: %s", msg, exc, exc_info=True)
        else:
            logger.warning("%s: %s", msg, exc)


def _parse_endpoint(endpoint: str) -> tuple[str, int]:
    """Accept 'kvblockd://host:port' or bare 'host:port'."""
    if "//" in endpoint:
        u = urlparse(endpoint)
        return u.hostname or "127.0.0.1", u.port or 9440
    host, _, port = endpoint.partition(":")
    return host or "127.0.0.1", int(port or 9440)


def _tensor_view(t) -> memoryview:
    """Writable byte view over a CPU-contiguous torch tensor's storage.
    ctypes.from_address does NOT pin the tensor — every caller must hold a
    reference to `t` for the view's whole lifetime."""
    if t.device.type != "cpu":
        raise ValueError("kvblockd backend needs CPU (host) tensors")
    if not t.is_contiguous():
        raise ValueError("kvblockd backend needs contiguous tensors")
    n = t.numel() * t.element_size()
    return memoryview((ctypes.c_char * n).from_address(t.data_ptr())).cast("B")


def _ptr_view(ptr: int, size: int) -> memoryview:
    """Writable byte view over a raw host pointer (mooncake/hf3fs-style
    target_location=int calls). Caller guarantees the region stays alive."""
    return memoryview((ctypes.c_char * size).from_address(ptr)).cast("B")


class KvblockdHiCacheStorage(HiCacheStorage):
    """SGLang HiCacheStorage v1 backend for kvblockd.

    Constructed by SGLang's dynamic StorageBackendFactory as
    ``backend_class(storage_config, kwargs)`` — note the SECOND POSITIONAL
    ARG is a plain dict (verified at v0.5.15.post1, backend_factory.py).
    Connection settings ride in ``storage_config.extra_config`` (the
    --hicache-storage-backend-extra-config JSON):

        endpoint   "kvblockd://127.0.0.1:9440"      (required in practice)
        namespace  "sglang-e2e"                      (default "default")
        token      "..."            (or env KVBLOCKD_TOKEN via the daemon op)
        streams    4                 wire connections / put fan-out width
        verify     true              client-side xxh3 verify on GET
        op_timeout / connect_timeout seconds
        interface_v1  1              REQUIRED for the zero-copy path: the
                                     controller only routes a dynamic
                                     backend through batch_get_v1/set_v1
                                     when extra_config["interface_v1"] is
                                     truthy (cache_controller.py).

    Never dials at construction: the import-and-instantiate tripwire (and
    SGLang's factory) must succeed with no daemon running.
    """

    def __init__(self, storage_config=None, kwargs: dict | None = None, **extra):
        cfg_extra = keymap._cfg_get(storage_config, "extra_config", None) or {}
        merged: dict[str, Any] = {}
        merged.update(cfg_extra)
        merged.update(kwargs or {})
        merged.update(extra)

        host, port = _parse_endpoint(str(merged.get("endpoint", "127.0.0.1:9440")))
        self._addr = (host, port)
        self._namespace = str(merged.get("namespace", "default"))
        self._token = str(merged.get("token", ""))
        self._streams = int(merged.get("streams", 4))
        self._verify = bool(merged.get("verify", True))
        self._op_timeout = float(merged.get("op_timeout", 10.0))
        self._connect_timeout = float(merged.get("connect_timeout", 5.0))

        self._scheme = keymap.scheme_from_config(storage_config)
        self._enable_metrics = bool(
            keymap._cfg_get(storage_config, "enable_storage_metrics", True)
        )

        self._client: Client | None = None
        self._lock = threading.Lock()
        self._closed = False
        self._log = _RateLimitedLog()
        self._exec = ThreadPoolExecutor(
            max_workers=self._streams, thread_name_prefix="kvblockd-sgl"
        )

        # Pinned-pool registration state (filled by register_mem_pool_host).
        self._pool_ready = False
        self._kv_tensor = None  # ref keeps the ctypes mapping's memory alive
        self._base_ptr = 0
        self._base_len = 0
        self._base_view: memoryview | None = None
        self._page_size = 0
        self.registered_pools: dict[Any, Any] = {}

        # StorageMetrics accumulators (mooncake shape: drained by get_stats).
        self._stats_lock = threading.Lock()
        self._prefetch_pgs: list[int] = []
        self._backup_pgs: list[int] = []
        self._prefetch_bw: list[float] = []  # GB/s per batch
        self._backup_bw: list[float] = []

        if not merged.get("interface_v1", 0):
            logger.warning(
                "extra-config lacks interface_v1=1 — SGLang's controller will "
                "drive the slow generic page path, not batch_get_v1/set_v1."
            )

    # ------------------------------------------------------------------ util
    def _ensure(self) -> Client:
        with self._lock:
            if self._closed:
                raise RuntimeError("backend closed")
            if self._client is None:
                self._client = Client(
                    self._addr,
                    namespace=self._namespace,
                    token=self._token,
                    streams=self._streams,
                    connect_timeout=self._connect_timeout,
                    op_timeout=self._op_timeout,
                    verify=self._verify,
                )
            return self._client

    def close(self):
        with self._lock:
            self._closed = True
            client, self._client = self._client, None
        if client is not None:
            client.close()
        self._exec.shutdown(wait=False)

    def _guard(self, op: str, default, fn):
        """NEVER RAISE: any Exception degrades to `default` (a miss).
        BaseException (KeyboardInterrupt/SystemExit) propagates."""
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            self._log.maybe(op, f"kvblockd {op} failed (treated as miss)", e)
            return default() if callable(default) else default

    def _note(self, prefetch: bool, pages: int, nbytes: int, elapsed: float):
        if not self._enable_metrics or pages <= 0:
            return
        gbps = (nbytes / (1 << 30)) / elapsed if elapsed > 0 else 0.0
        with self._stats_lock:
            if prefetch:
                self._prefetch_pgs.append(pages)
                self._prefetch_bw.append(gbps)
            else:
                self._backup_pgs.append(pages)
                self._backup_bw.append(gbps)

    # -------------------------------------------------- pinned-pool plumbing
    def register_mem_pool_host(self, mem_pool_host):
        """Pre-register the WHOLE L2 host pool once (Mooncake-style): build a
        single writable byte view over kv_buffer; batch_get_v1 later slices
        it per page. Never raises — an unusable pool disables the v1 fast
        paths (they return misses) while the generic path keeps working."""
        super().register_mem_pool_host(mem_pool_host)
        self._pool_ready = False
        kv = getattr(mem_pool_host, "kv_buffer", None)
        if kv is None:
            logger.error("host pool has no kv_buffer (logical anchor pool?) — "
                         "zero-copy v1 paths disabled")
            return
        layout = getattr(mem_pool_host, "layout", None)
        if layout is not None and layout not in ("page_first", "page_first_direct",
                                                 "page_head"):
            logger.error("host pool layout %r unsupported (need the page-first "
                         "family) — zero-copy v1 paths disabled", layout)
            return
        if not hasattr(mem_pool_host, "get_page_buffer_meta"):
            logger.error("host pool lacks get_page_buffer_meta — zero-copy v1 "
                         "paths disabled")
            return
        try:
            page_size = int(mem_pool_host.page_size)
            if getattr(kv, "device", None) is not None and kv.device.type != "cpu":
                logger.error("kv_buffer is on %s, not CPU — v1 paths disabled",
                             kv.device)
                return
            if hasattr(kv, "is_contiguous") and not kv.is_contiguous():
                logger.error("kv_buffer is not contiguous — v1 paths disabled")
                return
            self._kv_tensor = kv  # keep the mapping's memory alive
            self._base_ptr = kv.data_ptr()
            self._base_len = kv.numel() * kv.element_size()
            self._base_view = memoryview(
                (ctypes.c_char * self._base_len).from_address(self._base_ptr)
            ).cast("B")
            self._page_size = page_size
            self._pool_ready = True
            logger.info("registered host pool: %d bytes, page_size=%d, layout=%s",
                        self._base_len, page_size, layout)
        except Exception as e:  # noqa: BLE001
            self._log.maybe("register_pool", "host pool registration failed "
                            "(v1 paths disabled)", e)

    def _resolve_regions(self, host_indices, n_keys: int):
        """host_indices (CPU int64 tensor; token-level, len == n_keys *
        page_size — page-start form len == n_keys also accepted) → aligned
        per-physical-object (memoryview, size) lists via the pool's own
        get_page_buffer_meta (upstream owns the layout math; we own the
        ptr→offset translation and bounds checks). None ⇒ unresolvable.
        A persistent drift re-fails on EVERY batch, so failures log through
        the rate limiter, not raw logger.error."""
        pool = self.mem_pool_host
        idx = host_indices
        if hasattr(idx, "numel"):
            if idx.numel() == n_keys and self._page_size > 1:
                idx = idx.repeat_interleave(self._page_size)  # page-start form
            if idx.numel() != n_keys * self._page_size:
                self._log.maybe("resolve_regions",
                                f"host_indices numel {idx.numel()} != n_keys "
                                f"{n_keys} * page_size {self._page_size}")
                return None
            if idx.device.type != "cpu":
                idx = idx.cpu()
        ptrs, sizes = pool.get_page_buffer_meta(idx)
        if len(ptrs) == 0 or len(ptrs) % n_keys != 0:
            self._log.maybe("resolve_regions",
                            f"get_page_buffer_meta returned {len(ptrs)} "
                            f"regions for {n_keys} keys")
            return None
        if any(not isinstance(p, int) for p in ptrs):
            self._log.maybe("resolve_regions",
                            "multi-buffer page meta (layer_first MLA?) unsupported")
            return None
        rpp = len(ptrs) // n_keys
        want = keymap.multiplier(self._scheme)
        if rpp != want:
            self._log.maybe("resolve_regions",
                            f"regions/page {rpp} != expected {want} (layout/"
                            "key-scheme mismatch — is the pool page-first?)")
            return None
        views: list[memoryview] = []
        base = self._base_view
        assert base is not None
        for p, s in zip(ptrs, sizes):
            off = p - self._base_ptr
            if off < 0 or off + s > self._base_len:
                self._log.maybe("resolve_regions",
                                f"page region [{off},+{s}) outside registered pool")
                return None
            views.append(base[off:off + s])
        return views, list(sizes)

    # ------------------------------------------------------- existence probes
    def exists(self, key: str) -> bool:
        return self.batch_exists([key]) >= 1

    def batch_exists(self, keys: List[str], extra_info=None) -> int:
        """Consecutive-from-index-0 logical hit count, in ONE BATCH_EXISTS
        round trip over the interleaved physical keys. A logical page counts
        only if ALL its physical objects exist, so the flat consecutive count
        floor-divides by the per-page multiplier. Mid-prefetch calls with
        keys the controller hasn't backed up yet simply shorten the prefix —
        partial count, never an error."""
        if not keys:
            return 0

        def _do():
            wire = keymap.wire_keys(self._scheme, keys)
            n_consec, _ = self._ensure().batch_exists(wire)
            return n_consec // keymap.multiplier(self._scheme)

        return self._guard("batch_exists", 0, _do)

    # ------------------------------------------------------ zero-copy v1 path
    def batch_get_v1(self, keys: List[str], host_indices,
                     extra_info=None) -> List[bool]:
        """Fetch pages straight into the pinned host pool: resolve
        host_indices → pool byte regions, then batch_get_scatter recv_into's
        each hit. Per page: True iff every physical object landed byte-exact
        (a size-mismatched or missing object is a miss for that page)."""
        n = len(keys)
        if n == 0:
            return []

        def _do():
            if not self._pool_ready:
                return [False] * n
            resolved = self._resolve_regions(host_indices, n)
            if resolved is None:
                return [False] * n
            views, sizes = resolved
            wire = keymap.wire_keys(self._scheme, keys)

            def alloc(i, _prefix, body_len):
                # Stored object must exactly fill its pool region; anything
                # else is a layout/config drift → treat as a miss (drained).
                return views[i] if body_len == sizes[i] else None

            t0 = time.perf_counter()
            statuses = self._ensure().batch_get_scatter(wire, 0, alloc)
            elapsed = time.perf_counter() - t0
            m = keymap.multiplier(self._scheme)
            out = [
                all(statuses[i * m + j] == kp.Status.OK for j in range(m))
                for i in range(n)
            ]
            got_bytes = sum(
                sizes[i * m + j] for i in range(n) for j in range(m) if out[i]
            )
            self._note(True, sum(out), got_bytes, elapsed)
            return out

        return self._guard("batch_get_v1", lambda: [False] * n, _do)

    def batch_set_v1(self, keys: List[str], host_indices,
                     extra_info=None) -> List[bool]:
        """Back pages up from the pinned pool. One BATCH_EXISTS pre-pass
        skips already-stored objects (kvblockd blocks are write-once and
        content-addressed — a re-PUT would be answered OK_EXISTS anyway, the
        pre-pass just saves the payload bytes); the remaining PUTs fan out
        across the connection pool."""
        n = len(keys)
        if n == 0:
            return []

        def _do():
            if not self._pool_ready:
                return [False] * n
            resolved = self._resolve_regions(host_indices, n)
            if resolved is None:
                return [False] * n
            views, sizes = resolved
            wire = keymap.wire_keys(self._scheme, keys)
            client = self._ensure()
            n_consec, per = client.batch_exists(wire)
            if per is None:  # bitmap feature off: trust only the prefix
                per = [i < n_consec for i in range(len(wire))]
            ok = [bool(per[i]) for i in range(len(wire))]
            todo = [i for i in range(len(wire)) if not ok[i]]

            def _put_one(i: int) -> bool:
                try:
                    return kp.status_ok(client.put(wire[i], views[i]))
                except Exception as e:  # noqa: BLE001
                    self._log.maybe("put", "kvblockd put failed (page dropped)", e)
                    return False

            t0 = time.perf_counter()
            put_bytes = 0
            if todo:
                for i, good in zip(todo, self._exec.map(_put_one, todo)):
                    ok[i] = good
                    if good:
                        put_bytes += sizes[i]
            elapsed = time.perf_counter() - t0
            m = keymap.multiplier(self._scheme)
            out = [all(ok[i * m + j] for j in range(m)) for i in range(n)]
            self._note(False, sum(out), put_bytes, elapsed)
            return out

        return self._guard("batch_set_v1", lambda: [False] * n, _do)

    # -------------------------------------------- generic (non-zero-copy) path
    # One logical page = ONE whole-flat-page object under the `_pg` suffix
    # (the controller's generic path moves whole flat pages via
    # get_dummy_flat_data_page/set_from_flat_data_page). Cold path; upstream
    # marks it "todo: deprecate" — correctness over cleverness here.
    def _page_wire_key(self, key: str) -> bytes:
        return keymap.wire_key32(self._scheme, key + keymap.page_suffix(self._scheme))

    def _out_buffer(self, value):
        """value/target_location → (memoryview, owner_ref) for a PUT."""
        if value is None:
            return None
        if isinstance(value, (bytes, bytearray, memoryview)):
            return memoryview(value), value
        if hasattr(value, "data_ptr"):  # torch tensor (duck-typed)
            t = value if value.is_contiguous() else value.contiguous()
            if t.device.type != "cpu":
                raise ValueError("kvblockd set needs CPU tensors")
            return _tensor_view(t), t  # owner ref keeps ctypes view alive
        return None

    def _in_buffer(self, target_location, target_sizes):
        """target_location (+sizes) → (memoryview, owner, expected_len)."""
        if hasattr(target_location, "data_ptr"):  # torch tensor
            n = target_location.numel() * target_location.element_size()
            return _tensor_view(target_location), target_location, n
        if isinstance(target_location, int):  # raw host pointer + explicit size
            size = target_sizes if isinstance(target_sizes, int) else int(sum(target_sizes))
            return _ptr_view(target_location, size), None, size
        if isinstance(target_location, (bytearray, memoryview)):
            mv = memoryview(target_location)
            return mv, target_location, len(mv)
        raise ValueError(f"unsupported target_location {type(target_location)!r}")

    def get(self, key: str, target_location: Optional[Any] = None,
            target_sizes: Optional[Any] = None):
        """Fetch one whole flat page. With a target, bytes land in it
        zero-copy and the target is returned; without one, a fresh flat
        uint8 tensor is returned (raw bytes if torch is absent). None ⇒ miss."""

        def _do():
            wire = [self._page_wire_key(key)]
            client = self._ensure()
            if target_location is None:
                vals, statuses = client.batch_get_bytes(wire)
                if statuses[0] != kp.Status.OK or vals[0] is None:
                    return None
                try:
                    torch = _torch()
                    return torch.frombuffer(bytearray(vals[0]), dtype=torch.uint8)
                except ImportError:
                    return vals[0]
            view, _owner, expected = self._in_buffer(target_location, target_sizes)

            def alloc(_i, _prefix, body_len):
                return view if body_len == expected else None

            statuses = client.batch_get_scatter(wire, 0, alloc)
            return target_location if statuses[0] == kp.Status.OK else None

        return self._guard("get", None, _do)

    def batch_get(self, keys: List[str], target_locations: Optional[Any] = None,
                  target_sizes: Optional[Any] = None):
        # Probe the containers with `is None`, never truthiness: a stacked
        # tensor hits Tensor.__bool__ ("Boolean value of Tensor ... is
        # ambiguous") and the raise would escape the public method.
        if target_locations is None:
            target_locations = [None] * len(keys)
        if target_sizes is None:
            target_sizes = [None] * len(keys)
        return [
            self.get(k, loc, sz)
            for k, loc, sz in zip(keys, target_locations, target_sizes)
        ]

    def set(self, key: str, value: Optional[Any] = None,
            target_location: Optional[Any] = None,
            target_sizes: Optional[Any] = None) -> bool:
        def _do():
            src = value if value is not None else target_location
            if isinstance(src, int):  # raw pointer + sizes form
                size = target_sizes if isinstance(target_sizes, int) else int(sum(target_sizes))
                buf, owner = _ptr_view(src, size), None
            else:
                out = self._out_buffer(src)
                if out is None:
                    return False
                buf, owner = out
            st = self._ensure().put(self._page_wire_key(key), buf)
            del owner  # explicit: owner (and its buffer) lived through put()
            return kp.status_ok(st)

        return self._guard("set", False, _do)

    def batch_set(self, keys: List[str], values: Optional[Any] = None,
                  target_locations: Optional[Any] = None,
                  target_sizes: Optional[Any] = None) -> bool:
        # `is None` probes for the same reason as batch_get above.
        if values is None:
            values = [None] * len(keys)
        if target_locations is None:
            target_locations = [None] * len(keys)
        if target_sizes is None:
            target_sizes = [None] * len(keys)
        return all(
            self.set(k, v, loc, sz)
            for k, v, loc, sz in zip(keys, values, target_locations, target_sizes)
        )

    # ------------------------------------------------------------ maintenance
    def clear(self) -> None:
        """kvblockd v1 has no namespace-flush/enumeration verb, and blocks
        are write-once with TTL/eviction owning reclamation — clear() is a
        deliberate no-op (logged once). Stale pages age out server-side."""
        self._log.maybe("clear", "clear() is a no-op: kvblockd has no flush "
                        "verb; TTL/eviction reclaims", None)

    def get_stats(self):
        """Local transfer counters → SGLang StorageMetrics (drained on read,
        the mooncake contract). Falls back to a plain dict when sglang (or
        its metrics module) is absent so the CPU suite can assert on it."""
        with self._stats_lock:
            snap = (self._prefetch_pgs, self._backup_pgs,
                    self._prefetch_bw, self._backup_bw)
            self._prefetch_pgs, self._backup_pgs = [], []
            self._prefetch_bw, self._backup_bw = [], []
        cls = storage_metrics_cls()
        if cls is None:
            return {"prefetch_pgs": snap[0], "backup_pgs": snap[1],
                    "prefetch_bandwidth": snap[2], "backup_bandwidth": snap[3]}
        m = cls()
        m.prefetch_pgs.extend(snap[0])
        m.backup_pgs.extend(snap[1])
        m.prefetch_bandwidth.extend(snap[2])
        m.backup_bandwidth.extend(snap[3])
        return m

    def debug_server_stats(self) -> dict:
        """Parsed kvblockd STATS JSON (blocks/bytes/arena/evictions…) for
        operators and tests; not part of the HiCacheStorage surface."""

        def _do():
            import json

            return json.loads(self._ensure().stats().decode("utf-8"))

        return self._guard("stats", dict, _do)

    # -------------------------------------------------------------- v2 stubs
    # register_mem_host_pool_v2 (a registration hook, not a data path) keeps
    # the inherited store-only behavior so speculative-decode init doesn't
    # explode; the three v2 DATA methods raise by ruling.
    def batch_exists_v2(self, keys, pool_transfers=None, extra_info=None):
        raise NotImplementedError(_V2_MSG)

    def batch_get_v2(self, transfers, extra_info=None):
        raise NotImplementedError(_V2_MSG)

    def batch_set_v2(self, transfers, extra_info=None):
        raise NotImplementedError(_V2_MSG)
