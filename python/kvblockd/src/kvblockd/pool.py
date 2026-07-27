"""A tiny thread-safe connection pool: a bounded set of live connections,
checked out one-at-a-time per verb call. An errored connection is dropped
(closed, not returned) and lazily replaced on the next checkout — the pool
self-heals the way pkg/client redials on release, but lazily.

v0.1 does no cross-socket tiling; `run(fn)` is the single seam a future
tiled scheduler would replace. `streams` bounds concurrency (one in-flight
request per connection), which is what lets the LMCache connector's
ThreadPoolExecutor fan batches across sockets safely.
"""

from __future__ import annotations

import logging
import threading
import time

from kvblockd.errors import ConnectionLost, StatusError

logger = logging.getLogger("kvblockd.pool")

_EVICT_WARN_INTERVAL_S = 10.0


class Pool:
    def __init__(self, factory, streams: int, on_evict=None):
        streams = max(streams, 1)
        self._factory = factory
        self._sem = threading.Semaphore(streams)
        self._lock = threading.Lock()
        self._idle: list = []
        self._closed = False
        # Telemetry: evicted-connection count (the degrade-to-miss machinery's
        # loudest silent path). on_evict is an optional zero-arg callback the
        # owning Client uses to fold this into its own counters.
        self.evictions = 0
        self._on_evict = on_evict
        self._evict_warn_last = 0.0

    def checkout(self, timeout: float | None = None):
        """Acquire a pool slot (and a live connection). timeout bounds the
        SEMAPHORE wait — under contention the elaborate in-band deadline
        machinery is otherwise defeated at the front door: a caller with a
        hard budget would block here indefinitely before its first recv was
        ever clamped. An expired wait raises ConnectionLost so the GET path's
        degrade-to-miss catch handles it like any other dead-conn event
        (nothing was acquired, so there is nothing to evict or release)."""
        if timeout is None:
            self._sem.acquire()
        elif not self._sem.acquire(timeout=max(0.0, timeout)):
            raise ConnectionLost(f"pool checkout timed out after {max(0.0, timeout):.3f}s")
        try:
            with self._lock:
                if self._closed:
                    raise ConnectionLost("pool closed")
                if self._idle:
                    return self._idle.pop()
            return self._factory()  # dial outside the lock
        except BaseException:
            self._sem.release()
            raise

    def checkin(self, conn):
        with self._lock:
            if self._closed:
                conn.close()
            else:
                self._idle.append(conn)
        self._sem.release()

    def discard(self, conn):
        conn.close()
        self.evictions += 1  # benign data race: telemetry, not accounting
        cb = self._on_evict
        if cb is not None:
            try:
                cb()
            except Exception:  # noqa: BLE001, S110 — a telemetry callback must never poison the release path
                pass
        self._sem.release()

    def run(self, fn, checkout_timeout: float | None = None):
        """Check out a connection, run fn(conn), return its result. A
        StatusError leaves the stream in sync → the conn is re-pooled;
        ANYTHING ELSE evicts it. The catch-all is load-bearing: a FrameError,
        struct.error, IndexError, MemoryError, or a raising user callback all
        mean the stream state is unknown, so the connection must not be reused
        — and, critically, the semaphore MUST be released on every path or
        the pool starves permanently after `streams` such errors (a hang the
        never-raise wrapper cannot catch)."""
        conn = self.checkout(checkout_timeout)
        try:
            result = fn(conn)
        except StatusError:
            self.checkin(conn)  # in sync — reuse
            raise
        except BaseException as e:
            self.discard(conn)  # unknown stream state or dead — evict, release the slot
            # Rate-limited disclosure: an eviction is a degrade-to-miss event
            # a hit-rate cliff traces back to; silence here was an on-call trap.
            now = time.monotonic()
            if now - self._evict_warn_last >= _EVICT_WARN_INTERVAL_S:
                self._evict_warn_last = now
                logger.warning("kvblockd pool: connection evicted (%d total): %s",
                               self.evictions, e)
            raise
        else:
            self.checkin(conn)
            return result

    def close(self):
        with self._lock:
            self._closed = True
            idle, self._idle = self._idle, []
        for c in idle:
            c.close()
