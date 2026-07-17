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

import threading

from kvblockd.errors import ConnectionLost
from kvblockd.protocol import StatusError


class Pool:
    def __init__(self, factory, streams: int):
        if streams < 1:
            streams = 1
        self._factory = factory
        self._sem = threading.Semaphore(streams)
        self._lock = threading.Lock()
        self._idle: list = []
        self._closed = False

    def checkout(self):
        self._sem.acquire()
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
        self._sem.release()

    def run(self, fn):
        """Check out a connection, run fn(conn), return its result. A
        StatusError leaves the stream in sync → the conn is re-pooled;
        ANYTHING ELSE evicts it. The catch-all is load-bearing: a FrameError,
        struct.error, IndexError, MemoryError, or a raising user callback all
        mean the stream state is unknown, so the connection must not be reused
        — and, critically, the semaphore MUST be released on every path or
        the pool starves permanently after `streams` such errors (a hang the
        never-raise wrapper cannot catch)."""
        conn = self.checkout()
        try:
            result = fn(conn)
        except StatusError:
            self.checkin(conn)  # in sync — reuse
            raise
        except BaseException:
            self.discard(conn)  # unknown stream state or dead — evict, release the slot
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
