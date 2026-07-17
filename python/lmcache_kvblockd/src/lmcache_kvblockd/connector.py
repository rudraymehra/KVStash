"""KvblockdRemoteConnector — the LMCache RemoteConnector backed by kvblockd.

Two contracts dominate the design:

  NEVER RAISE. LMCache's serving engine treats an exception (or a hang) as
  fatal; a returned None/empty/0 is just a cache miss. Every public method is
  wrapped so any failure degrades to a miss, logged at most once per interval.

  ZERO-COPY GET. Blocks are stored as [32B metadata prefix || tensor bytes].
  On get, the client reads the prefix first, we allocate a MemoryObj of the
  right (dtype, shape) from LMCache's pinned LocalCPUBackend, and the client
  recv_into's the tensor bytes straight into it — no intermediate copy.

The wire client is synchronous; the RemoteConnector API is async. One
ThreadPoolExecutor(streams) bridges them with ONE hop per batch.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

from kvblockd import protocol as kp
from kvblockd.client import Client
from kvblockd.hashing import startup_determinism_check, wire_key

from . import meta

logger = logging.getLogger("lmcache_kvblockd")


def _torch():  # lazy so import_check can load us without torch/lmcache
    import torch

    return torch


def _memory_format(fmt_int: int):
    """Reconstruct LMCache's MemoryFormat from the stored int, or None if
    LMCache/the enum value is unavailable (→ the backend's default)."""
    try:
        from lmcache.v1.memory_management import MemoryFormat

        return MemoryFormat(fmt_int)
    except Exception:
        return None


class ConnectionLostShim(RuntimeError):
    """Raised by _ensure on a closed connector; caught by the never-raise
    wrappers so a teardown-time op degrades to a miss."""


class _RateLimitedLog:
    """One full traceback per key on first sight, then one terse line per
    interval. Per-INSTANCE state (a connector's failures don't silence
    another's)."""

    def __init__(self, interval=10.0):
        self._interval = interval
        self._last: dict[str, float] = {}
        self._full: set[str] = set()

    def maybe(self, key: str, msg: str, exc: BaseException | None = None):
        now = time.monotonic()
        if now - self._last.get(key, 0.0) < self._interval:
            return
        self._last[key] = now
        if exc is not None and key not in self._full:
            self._full.add(key)
            logger.warning("%s: %s", msg, exc, exc_info=True)
        else:
            logger.warning("%s: %s", msg, exc)


class KvblockdRemoteConnector:
    """Implements LMCache's RemoteConnector surface. Constructed lazily —
    the socket is not dialed until first use, so instantiation succeeds with
    no daemon present (the import/instantiate tripwire relies on this)."""

    def __init__(self, host: str, port: int, namespace: str, token: str,
                 local_cpu_backend=None, streams: int = 4, op_timeout: float = 10.0):
        # op_timeout is deliberately BELOW LMCache's blocking_timeout_secs
        # (~30s) so a stalled daemon becomes a miss here rather than an engine
        # stall — and so our executor thread never outlives LMCache's cancel.
        self._addr = (host, port)
        self._namespace = namespace
        self._token = token
        self._backend = local_cpu_backend
        self._streams = streams
        self._op_timeout = op_timeout
        self._client: Client | None = None
        self._exec = ThreadPoolExecutor(max_workers=streams, thread_name_prefix="kvblockd")
        self._log = _RateLimitedLog()
        self._lock = __import__("threading").Lock()
        self._closed = False

    # --- lifecycle ---
    def post_init(self):
        try:
            startup_determinism_check()
        except Exception as e:  # loud, but non-fatal per never-raise
            logger.error("kvblockd determinism check failed: %s", e)

    def _ensure(self) -> Client:
        # Locked so concurrent executor threads don't each dial a Client
        # (leaking the loser's sockets); refuses to resurrect a closed
        # connector.
        with self._lock:
            if self._closed:
                raise ConnectionLostShim("connector closed")
            if self._client is None:
                self._client = Client(self._addr, namespace=self._namespace, token=self._token,
                                      streams=self._streams, op_timeout=self._op_timeout)
            return self._client

    async def close(self):
        with self._lock:
            self._closed = True
            client, self._client = self._client, None
        if client is not None:
            client.close()
        self._exec.shutdown(wait=False)

    async def _run(self, fn, default, op="op"):
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(self._exec, fn)
        except Exception as e:  # NEVER RAISE — degrade to a miss (BaseException/
            # CancelledError intentionally propagates: the caller cancelled).
            self._log.maybe(op, f"kvblockd {op} failed (treated as miss)", e)
            return default() if callable(default) else default

    # --- key mapping ---
    @staticmethod
    def _wire(key) -> bytes:
        # CacheEngineKey field order: fmt, model_name, world_size, worker_id, chunk_hash.
        return wire_key([str(key.fmt), str(key.model_name), str(key.world_size),
                         str(key.worker_id), str(key.chunk_hash)])

    # --- MemoryObj (de)serialization ---
    def _dtype_name(self, dtype) -> str:
        return str(dtype).removeprefix("torch.")

    def _serialize(self, memory_obj) -> bytes | None:
        """MemoryObj → [prefix || tensor bytes]. None ⇒ skip the put (miss).
        The stored fmt is the MemoryObj's MemoryFormat enum value — carried so
        the retrieval path can reconstruct the exact layout (KV_2LTD etc.)."""
        try:
            t = memory_obj.tensor
            dtype_name = self._dtype_name(t.dtype)
            fmt = int(getattr(memory_obj.metadata, "fmt", 0)) if memory_obj.metadata else 0
            raw = t.contiguous().view(_torch().uint8).numpy().tobytes()
            prefix = meta.encode(fmt, dtype_name, tuple(t.shape), len(raw))
            return prefix + raw
        except meta.MetaError:
            return None  # unsupported dtype/shape → treat as uncacheable

    def _allocator(self, results: dict):
        """alloc(idx, prefix, body_len)->memoryview|None for the client's
        zero-copy scatter path: parse the prefix, allocate a pinned MemoryObj
        of that (fmt, dtype, shape) so vLLM reads back the SAME layout, stash
        it, return a flat writable uint8 view ALIASING its storage."""
        torch = _torch()

        def alloc(idx, prefix, body_len):
            try:
                fmt_int, dtype_name, shape, meta_body_len = meta.decode(prefix)
            except meta.MetaError:
                return None  # unrecognized prefix → miss
            try:
                if meta_body_len != body_len:
                    return None
                dtype = getattr(torch, dtype_name)
                fmt = _memory_format(fmt_int)
                if self._backend is None:
                    return None
                obj = (self._backend.allocate(shape, dtype, fmt) if fmt is not None
                       else self._backend.allocate(shape, dtype))
                if obj is None:
                    return None
                base = obj.tensor.view(torch.uint8)
                if not base.is_contiguous():
                    # .contiguous() would COPY → recv_into fills a throwaway and
                    # the returned obj stays garbage while xxh3 still passes.
                    # Refuse rather than silently corrupt (pinned buffers are
                    # contiguous in practice; this is the tripwire).
                    return None
                flat = base.numpy().reshape(-1)
                results[idx] = obj
                return memoryview(flat)
            except Exception:
                return None

        return alloc

    # --- the RemoteConnector surface ---
    async def exists(self, key) -> bool:
        # batched_async_contains is the async path; batched_contains is sync
        # (base-class shape) and must not be awaited.
        return (await self.batched_async_contains("", [key])) >= 1

    async def exists_sync(self, key) -> bool:
        # Route through the executor: the base declares this async, but the
        # work is blocking socket I/O — running it inline would stall the
        # event loop (heartbeats, other connectors) for a full round trip.
        return (await self._run(lambda: self._ensure().batch_exists([self._wire(key)])[0],
                                0, op="exists_sync")) >= 1

    async def get(self, key):
        objs = await self.batched_get_non_blocking("", [key])
        return objs[0] if objs else None

    async def put(self, key, memory_obj):
        # ref_count_down happens INSIDE _do (the executor thread), after all
        # tensor reads — so an asyncio cancel of the await can never fire it
        # while the thread is still serializing (the use-after-release that
        # would store torn bytes under a VALID checksum). shield guarantees
        # _do runs to completion (the ref is consumed) even under cancel.
        def _do():
            try:
                blob = self._serialize(memory_obj)
                if blob is None:
                    return False
                self._ensure().put(self._wire(key), blob)
                return True
            finally:
                try:
                    memory_obj.ref_count_down()
                except Exception:
                    pass
        loop = asyncio.get_running_loop()
        fut = loop.run_in_executor(self._exec, _do)
        try:
            return await asyncio.shield(fut)
        except asyncio.CancelledError:
            raise  # _do still completes under shield → ref consumed, no race
        except Exception as e:
            self._log.maybe("put", "kvblockd put failed (treated as miss)", e)
            return False

    async def list(self):
        return []  # kvblockd has no enumeration verb (documented)

    def remove_sync(self, key) -> bool:
        try:
            sts = self._ensure().delete([self._wire(key)], force=True)
            return bool(sts) and kp.status_ok(sts[0])
        except BaseException as e:  # sync path: nothing above catches — never raise
            self._log.maybe("remove", "remove_sync failed", e)
            return False

    # --- fast-path toggles (all supported) ---
    def support_ping(self) -> bool:
        return True

    def support_batched_get(self) -> bool:
        return True

    def support_batched_put(self) -> bool:
        return True

    def support_batched_contains(self) -> bool:
        return True

    def support_batched_async_contains(self) -> bool:
        return True

    def support_batched_get_non_blocking(self) -> bool:
        return True

    async def ping(self) -> int:
        return await self._run(lambda: (self._ensure().stats(), 1)[1], 0, op="ping")

    def batched_contains_sync(self, keys) -> int:
        try:
            n, _ = self._ensure().batch_exists([self._wire(k) for k in keys])
            return n
        except BaseException as e:  # sync path: never raise
            self._log.maybe("contains", "batched_contains failed", e)
            return 0

    def batched_contains(self, keys) -> int:
        return self.batched_contains_sync(keys)

    async def batched_async_contains(self, lookup_id: str, keys, pin: bool = False) -> int:
        def _do():
            wire = [self._wire(k) for k in keys]
            n, _ = self._ensure().batch_exists(wire)
            if pin and n > 0:
                # Session-pin the consecutive prefix via LEASE (§3.5).
                self._ensure().touch_lease(wire[:n], kp.LEASE_GRANT)
            return n
        return await self._run(_do, 0, op="async_contains")

    async def batched_get_non_blocking(self, lookup_id: str, keys):
        """Return only the consecutive prefix of retrieved MemoryObjs;
        ref_count_down any allocated object we don't return (LMCache's leak
        rule). On asyncio cancellation the returned prefix would be dropped by
        the caller, so a done-callback ref-downs the whole result if the await
        is cancelled — no pinned buffer leaks on a cancelled get."""
        def _do():
            wire = [self._wire(k) for k in keys]
            results: dict[int, object] = {}
            statuses = self._ensure().batch_get_scatter(wire, meta.PREFIX_LEN, self._allocator(results))
            # Consecutive prefix of OK results.
            prefix = []
            for i in range(len(keys)):
                if statuses[i] == kp.Status.OK and i in results:
                    prefix.append(results[i])
                else:
                    break
            # Drop anything past the prefix (or a miss with a stray alloc).
            for i, obj in results.items():
                if i >= len(prefix):
                    try:
                        obj.ref_count_down()
                    except Exception:
                        pass
            return prefix

        loop = asyncio.get_running_loop()
        fut = loop.run_in_executor(self._exec, _do)
        try:
            return await fut
        except asyncio.CancelledError:
            def _drain(f):
                try:
                    for obj in (f.result() or []):
                        obj.ref_count_down()
                except Exception:
                    pass
            fut.add_done_callback(_drain)
            raise
        except Exception as e:
            self._log.maybe("get", "kvblockd get failed (treated as miss)", e)
            return []
        return await self._run(_do, list)

    async def batched_get(self, keys):
        return await self.batched_get_non_blocking("", keys)

    async def batched_put(self, keys, memory_objs):
        oks = 0
        for k, mo in zip(keys, memory_objs):
            if await self.put(k, mo):
                oks += 1
        return oks


def make_connector(context) -> "KvblockdRemoteConnector":
    """Build a connector from a ConnectorContext (url kvblockd://host:port?
    namespace=X&streams=N; token from extra_config or KVBLOCKD_TOKEN)."""
    from urllib.parse import parse_qs, urlparse

    u = urlparse(context.url)
    host, port = u.hostname or "127.0.0.1", u.port or 9440
    q = parse_qs(u.query)
    namespace = q.get("namespace", ["default"])[0]
    streams = int(q.get("streams", ["4"])[0])
    token = os.environ.get("KVBLOCKD_TOKEN", "")
    cfg = getattr(context, "config", None)
    extra = getattr(cfg, "extra_config", None) if cfg else None
    if extra and extra.get("kvblockd_token"):
        token = extra["kvblockd_token"]
    return KvblockdRemoteConnector(
        host, port, namespace, token,
        local_cpu_backend=getattr(context, "local_cpu_backend", None),
        streams=streams,
    )
