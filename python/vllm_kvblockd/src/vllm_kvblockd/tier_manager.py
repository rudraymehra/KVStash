"""KvblockdTierManager — vLLM SecondaryTierManager backed by kvblockd.

CHURN-WATCH: the tiering surface is RFC #38260 lineage; the MERGED code is
the contract, not the RFC. Written against vLLM v0.25.0 (see UPSTREAM.lock);
same class shape verified present at v0.22.1/v0.23.0. Re-verify
vllm/v1/kv_offload/tiering/base.py before every vLLM bump.

This is the GPU-serving altitude (OffloadingConnector). It NEVER touches the
GPU: the framework hands us a zero-copy memoryview of the CPU primary tier
(/dev/shm/vllm_offload_{instance_id}.mmap) and pins the referenced slots for
the duration of each job; we read/write byte ranges addressed by block_ids.
Store = PUT_STREAM->COMMIT from the slice; load = BATCH_GET recv'd DIRECTLY
into the slice (kvblockd's scatter path — no intermediate copy).

Every method here runs in the SCHEDULER process and must be non-blocking:
  - lookup() is answered from an async EXISTS batcher (modeled on vLLM's
    AsyncLookupManager, vendored as a pattern so upstream moving that module
    cannot break us): keys accumulate per step, one BATCH_EXISTS per step on
    a background thread, RETRY until resolved.
  - submit_store()/submit_load() enqueue tiled tasks on a dual-queue thread
    pool (modeled on tiering/fs DualQueueThreadPool: read-priority and
    write-priority threads that can each drain the other queue).
  - drain_jobs() WAITS for in-flight copies, never aborts them — the base
    contract is explicit that a partial copy corrupts the primary memoryview
    or the backing store.

Key identity: vLLM's OffloadKey = raw chain-hash bytes + 4-byte big-endian
KV-cache-group index. vLLM's chain hash already folds the request's
cache_salt (first-block extra keys) — C-14 at this altitude is satisfied
upstream; we bind the OffloadKey to OUR config identity with
BLAKE3(fingerprint || offload_key) (config.tier_wire_key), where the
fingerprint mirrors FileMapper's config.json fields (parallel-agnostic).
Cross-instance sharing therefore requires PYTHONHASHSEED pinned identically
everywhere — enforced loudly at construction.

GPU end-to-end validation is DEFERRED (no GPU budget this week) — see
python/vllm_kvblockd/DEFER.md for the exact revisit trigger. The unit suite
(tests/test_tier_manager.py) drives this class against a synthetic
memoryview + hand-built JobMetadata, byte-for-byte.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from collections import deque
from collections.abc import Iterable

from kvblockd import protocol as kp
from kvblockd.client import Client

from .config import (
    fingerprint,
    parse_endpoint,
    require_pinned_hashseed,
    tier_fingerprint_fields,
    tier_wire_key,
)

logger = logging.getLogger("vllm_kvblockd")

try:  # vLLM tiering present: subclass the real ABC + reuse its result types.
    from vllm.v1.kv_offload.base import LookupResult, RequestOffloadingContext
    from vllm.v1.kv_offload.tiering.base import JobResult
    from vllm.v1.kv_offload.tiering.base import SecondaryTierManager as _TierBase

    _HAS_VLLM_TIERING = True
except Exception:  # pragma: no cover - exercised by the no-vllm test env
    _HAS_VLLM_TIERING = False
    import enum
    from dataclasses import dataclass

    class LookupResult(enum.Enum):  # type: ignore[no-redef]
        MISS = enum.auto()
        HIT = enum.auto()
        HIT_PENDING = enum.auto()
        RETRY = enum.auto()

    @dataclass
    class RequestOffloadingContext:  # type: ignore[no-redef]
        policy: object | None = None

    @dataclass
    class JobResult:  # type: ignore[no-redef]
        job_id: int
        success: bool

    class _TierBase:  # type: ignore[no-redef]
        def __init__(self, offloading_spec, primary_kv_view: memoryview, tier_type: str):
            self._offloading_spec = offloading_spec
            self._primary_kv_view = primary_kv_view
            self.tier_type = tier_type


class _JobState:
    """Thread-safe completion tracker for one job's tiled tasks (the
    tiering/fs JobState, plus a report flag for fire-and-forget work)."""

    __slots__ = ("job_id", "_n_tasks", "_completed", "_success", "_lock", "report")

    def __init__(self, job_id: int, n_tasks: int, report: bool = True):
        self.job_id = job_id
        self._n_tasks = n_tasks
        self._completed = 0
        self._success = True
        self._lock = threading.Lock()
        self.report = report

    def task_done(self, success: bool) -> tuple[bool, bool]:
        with self._lock:
            self._completed += 1
            if not success:
                self._success = False
            return self._completed == self._n_tasks, self._success


class _DualQueuePool:
    """Two queues, two thread groups: load-priority threads drain loads first
    then stores; store-priority threads the reverse. Neither starves. Modeled
    on vLLM's tiering/fs thread pool; vendored so tiering-module churn cannot
    take our worker machinery with it."""

    def __init__(self, n_read: int, n_write: int, name: str = "kvblockd_tier"):
        self._load_q: deque = deque()
        self._store_q: deque = deque()
        self._cond = threading.Condition(threading.Lock())
        self._stop = False
        self._finished: deque[tuple[int, bool]] = deque()
        self._inflight = 0  # guarded by _cond
        self._threads = [
            threading.Thread(target=self._worker, args=(True,), name=f"{name}_l{i}", daemon=True)
            for i in range(n_read)
        ] + [
            threading.Thread(target=self._worker, args=(False,), name=f"{name}_s{i}", daemon=True)
            for i in range(n_write)
        ]
        for t in self._threads:
            t.start()

    def _enqueue(self, q: deque, state: _JobState | None, tasks, n_tasks: int) -> None:
        with self._cond:
            if self._stop:
                # Work enqueued after shutdown can never run; a job still owes
                # its one JobResult — report it failed immediately.
                if state is not None and state.report:
                    self._finished.append((state.job_id, False))
                    self._cond.notify_all()
                return
            if state is not None:
                if n_tasks == 0:
                    # A zero-task job still owes exactly one JobResult.
                    self._finished.append((state.job_id, True))
                    self._cond.notify_all()
                    return
                self._inflight += 1
            for fn in tasks:
                q.append((fn, state))
            self._cond.notify(n_tasks)

    def enqueue_load(self, job_id: int, n_tasks: int, tasks) -> None:
        self._enqueue(self._load_q, _JobState(job_id, n_tasks), tasks, n_tasks)

    def enqueue_store(self, job_id: int, n_tasks: int, tasks) -> None:
        self._enqueue(self._store_q, _JobState(job_id, n_tasks), tasks, n_tasks)

    def enqueue_fire_and_forget(self, fn) -> None:
        """Jobless best-effort task (TOUCH recency): no result, no inflight
        accounting — drain_jobs must not wait on advisory work."""
        with self._cond:
            if self._stop:
                return  # advisory work owes nothing; drop it
            self._store_q.append((fn, None))
            self._cond.notify(1)

    def get_finished(self) -> list[tuple[int, bool]]:
        out = []
        while self._finished:
            out.append(self._finished.popleft())
        return out

    def has_pending(self) -> bool:
        with self._cond:
            if self._stop:
                # Post-shutdown, no queued/in-flight work can make progress;
                # only unreported results remain actionable.
                return bool(self._finished)
            return self._inflight > 0 or bool(self._finished)

    def wait_idle(self) -> None:
        """Block until no job task is in flight. NEVER cancels queued or
        running tasks — the SecondaryTierManager.drain_jobs contract forbids
        aborting mid-flight copies (a partial copy corrupts the primary
        memoryview or the backing store). Results stay queued for
        get_finished_jobs(). Gated on _stop so a racing shutdown can never
        strand this wait."""
        with self._cond:
            self._cond.wait_for(lambda: self._stop or self._inflight == 0)

    def shutdown(self, wait: bool = True) -> None:
        with self._cond:
            self._stop = True
            # Cancel queued-but-unstarted tasks by accounting them as FAILED:
            # each job still surfaces exactly one JobResult, and _inflight is
            # never zeroed — a task currently RUNNING resolves through the
            # normal worker path (its job reports failed too, poisoned here).
            for dq in (self._load_q, self._store_q):
                while dq:
                    _, state = dq.popleft()
                    if state is None:
                        continue  # fire-and-forget: no result owed
                    finished, _ = state.task_done(False)
                    if finished:
                        if state.report:
                            self._finished.append((state.job_id, False))
                        self._inflight -= 1
            self._cond.notify_all()
        if wait:
            for t in self._threads:
                t.join()

    def _worker(self, load_priority: bool) -> None:
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._stop or self._load_q or self._store_q)
                if self._stop:
                    return
                primary = self._load_q if load_priority else self._store_q
                secondary = self._store_q if load_priority else self._load_q
                task, state = primary.popleft() if primary else secondary.popleft()
            try:
                task()
                ok = True
            except Exception as exc:
                logger.warning("kvblockd tier task failed (job %s): %s",
                               getattr(state, "job_id", "-"), exc)
                ok = False
            if state is None:
                continue
            finished, success = state.task_done(ok)
            if finished:
                with self._cond:
                    if state.report:
                        self._finished.append((state.job_id, success))
                    self._inflight -= 1
                    self._cond.notify_all()


class _AsyncExistsBatcher:
    """Non-blocking EXISTS for the scheduler thread (the AsyncLookupManager
    pattern): lookup() accumulates unseen keys and returns the cached
    tri-state; flush() posts the step's batch to a background thread (one
    BATCH_EXISTS per step); results drain lazily on the next lookup().

    Ownership model (no locks, mirrors upstream): _state/_batch belong to the
    scheduler thread; the two SimpleQueues are the only cross-thread edges."""

    def __init__(self, exists_fn, tier_type: str):
        self._exists_fn = exists_fn  # list[wire_key] -> list[bool]; never raises
        self._state: dict[bytes, object] = {}  # key -> True|False|None
        self._req_keys: dict[str, set[bytes]] = {}
        self._key_reqs: dict[bytes, set[str]] = {}
        self._batch: list[bytes] = []
        self._in_q: queue.SimpleQueue = queue.SimpleQueue()
        self._out_q: queue.SimpleQueue = queue.SimpleQueue()
        self._need_drain = False
        self._thread = threading.Thread(
            target=self._worker, name=f"kvblockd_lookup_{tier_type}", daemon=True
        )
        self._thread.start()

    def lookup(self, key: bytes, req_id: str):
        if self._need_drain:
            self._drain()
            self._need_drain = False
        if key not in self._state:
            self._state[key] = None
            self._batch.append(key)
        self._key_reqs.setdefault(key, set()).add(req_id)
        self._req_keys.setdefault(req_id, set()).add(key)
        return self._state[key]

    def flush(self) -> None:
        self._need_drain = True
        if self._batch:
            self._in_q.put(self._batch)
            self._batch = []

    def _drain(self) -> None:
        while True:
            try:
                results = self._out_q.get_nowait()
            except queue.Empty:
                return
            for key, present in results:
                if key in self._state:
                    self._state[key] = present

    def cleanup(self, req_id: str) -> None:
        for key in self._req_keys.pop(req_id, ()):
            reqs = self._key_reqs.get(key)
            if reqs is not None:
                reqs.discard(req_id)
                if not reqs:
                    self._key_reqs.pop(key, None)
                    self._state.pop(key, None)

    def shutdown(self) -> None:
        self._in_q.put(None)
        self._thread.join()

    def _worker(self) -> None:
        while True:
            batch = self._in_q.get()
            if batch is None:
                return
            present = self._exists_fn(batch)
            self._out_q.put(list(zip(batch, present)))


class KvblockdTierManager(_TierBase):
    """SecondaryTierManager storing whole primary-tier blocks as opaque
    kvblockd blobs (one CPU block = one blob, raw bytes, no framing — the
    block length is fixed by config, and config identity lives in the key)."""

    def __init__(
        self,
        offloading_spec,
        primary_kv_view: memoryview,
        tier_type: str,
        endpoint: str = "kvblockd://127.0.0.1:9440",
        namespace: str = "vllm",
        token: str | None = None,
        streams: int = 8,
        n_read_threads: int = 16,
        n_write_threads: int = 16,
        tile_keys: int = 8,
        verify: bool = True,
        op_timeout: float = 10.0,
        block_bytes: int | None = None,
        # RFC-#38260-shaped tier configs carry these INSIDE the tier dict; the
        # merged factory forwards every non-"type" key to us, so accept and
        # ignore them rather than crash on **config.
        module_path: str | None = None,  # noqa: ARG002 - config-shape compat
        class_name: str | None = None,  # noqa: ARG002 - config-shape compat
    ):
        super().__init__(offloading_spec, primary_kv_view, tier_type)
        require_pinned_hashseed()  # OffloadKeys inherit vLLM's NONE_HASH seeding

        if block_bytes is not None:
            self._block_size = int(block_bytes)
        else:
            # Mirror tiering/fs: the view is (num_blocks, bytes_per_block),
            # so the leading stride IS the block size.
            assert primary_kv_view.strides is not None, "primary_kv_view.strides is None"
            self._block_size = int(primary_kv_view.strides[0])
        if self._block_size <= 1:
            raise ValueError(
                f"cannot infer block size from view strides {primary_kv_view.strides}; "
                "pass block_bytes explicitly"
            )
        # One flat byte view; every task slices it by byte range (fs/io.py's cast).
        self._bytes = primary_kv_view.cast("B")

        host, port = parse_endpoint(endpoint)
        self._addr = (host, port)
        self._namespace = namespace
        self._token = token if token is not None else os.environ.get("KVBLOCKD_TOKEN", "")
        self._streams = streams
        self._verify = verify
        self._op_timeout = op_timeout
        self._tile = max(1, int(tile_keys))
        self._client: Client | None = None
        self._client_lock = threading.Lock()
        self._closed = False

        self._fp = fingerprint(tier_fingerprint_fields(offloading_spec))
        self._pool = _DualQueuePool(n_read_threads, n_write_threads)
        self._lookup = _AsyncExistsBatcher(self._batch_exists, tier_type)
        self._last_err: tuple[float, str] | None = None

    # ------------------------------------------------------------------
    # client + key plumbing
    # ------------------------------------------------------------------
    def _ensure(self) -> Client:
        with self._client_lock:
            if self._closed:
                raise ConnectionError("tier manager shut down")
            if self._client is None:
                self._client = Client(
                    self._addr, namespace=self._namespace, token=self._token,
                    streams=self._streams, op_timeout=self._op_timeout, verify=self._verify,
                )
            return self._client

    def _key(self, offload_key) -> bytes:
        return tier_wire_key(self._fp, bytes(offload_key))

    def _warn(self, what: str, exc: BaseException) -> None:
        now = time.monotonic()
        if self._last_err is None or now - self._last_err[0] > 10.0 or self._last_err[1] != what:
            self._last_err = (now, what)
            logger.warning("kvblockd tier %s failed: %s", what, exc)

    def _batch_exists(self, wire_keys: list[bytes]) -> list[bool]:
        """Background-thread EXISTS; never raises (a dead daemon = all-miss).
        Uses the per-key bitmap when the daemon granted FEAT_EXISTS_BITMAP,
        else falls back to the consecutive-prefix count."""
        try:
            n_consec, per_key = self._ensure().batch_exists(wire_keys)
        except Exception as exc:
            self._warn("BATCH_EXISTS", exc)
            return [False] * len(wire_keys)
        if per_key is not None and len(per_key) == len(wire_keys):
            return [bool(b) for b in per_key]
        return [i < n_consec for i in range(len(wire_keys))]

    # ------------------------------------------------------------------
    # SecondaryTierManager surface (ALL scheduler-process, ALL non-blocking)
    # ------------------------------------------------------------------
    def lookup(self, key, req_context) -> LookupResult:
        result = self._lookup.lookup(self._key(key), getattr(req_context, "req_id", ""))
        if result is None:
            return LookupResult.RETRY
        return LookupResult.HIT if result else LookupResult.MISS

    def on_new_request(self, req_context) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    def on_request_finished(self, req_context) -> None:
        self._lookup.cleanup(getattr(req_context, "req_id", ""))

    def on_schedule_end(self, context) -> None:
        self._lookup.flush()

    def touch(self, keys, req_context) -> None:
        wire = [self._key(k) for k in keys]
        if not wire:
            return

        def _touch():
            try:
                self._ensure().touch_lease(wire, kp.TOUCH_RECENCY)
            except Exception as exc:
                self._warn("TOUCH", exc)

        self._pool.enqueue_fire_and_forget(_touch)

    def submit_store(self, job_metadata) -> None:
        """Primary -> kvblockd. Enqueue only; copies happen on pool threads
        while the framework keeps the block_ids slots pinned. Already-present
        blocks dedup server-side (write-once: OK_EXISTS)."""
        pairs = list(zip(job_metadata.keys, job_metadata.block_ids))
        tiles = [pairs[i : i + self._tile] for i in range(0, len(pairs), self._tile)]

        def make_task(tile):
            def _store():
                client = self._ensure()
                for key, bid in tile:
                    off = int(bid) * self._block_size
                    client.put(self._key(key), self._bytes[off : off + self._block_size])

            return _store

        self._pool.enqueue_store(
            int(job_metadata.job_id), len(tiles), (make_task(t) for t in tiles)
        )

    def submit_load(self, job_metadata) -> None:
        """kvblockd -> primary. BATCH_GET per tile, received DIRECTLY into the
        primary memoryview slices (zero-copy scatter). Any miss/short/corrupt
        block fails the whole job — the framework must never treat a
        partially-filled slot as loaded."""
        pairs = list(zip(job_metadata.keys, job_metadata.block_ids))
        tiles = [pairs[i : i + self._tile] for i in range(0, len(pairs), self._tile)]

        def make_task(tile):
            def _load():
                client = self._ensure()
                wire = [self._key(key) for key, _ in tile]
                slots = [int(bid) * self._block_size for _, bid in tile]

                def alloc(idx, prefix, body_len):
                    if body_len != self._block_size:
                        return None  # wrong-sized blob: refuse, count as miss
                    off = slots[idx]
                    return self._bytes[off : off + self._block_size]

                statuses = client.batch_get_scatter(wire, 0, alloc)
                misses = sum(1 for s in statuses if s != kp.Status.OK)
                if misses:
                    raise LookupError(f"{misses}/{len(tile)} blocks missing on load")

            return _load

        self._pool.enqueue_load(
            int(job_metadata.job_id), len(tiles), (make_task(t) for t in tiles)
        )

    def get_finished_jobs(self) -> Iterable[JobResult]:
        return [JobResult(job_id=jid, success=ok) for jid, ok in self._pool.get_finished()]

    def has_pending_work(self) -> bool:
        return self._pool.has_pending()

    def drain_jobs(self) -> None:
        # WAITS for every in-flight copy; never aborts one mid-flight (base
        # contract: a partial copy corrupts the primary memoryview or the
        # backing store). Completed results remain for get_finished_jobs().
        self._pool.wait_idle()

    def shutdown(self) -> None:
        self._lookup.shutdown()
        self._pool.shutdown(wait=True)
        with self._client_lock:
            self._closed = True
            client, self._client = self._client, None
        if client is not None:
            client.close()

    def get_stats(self):
        return None  # OffloadingConnectorStats wiring lands with the GPU e2e


if _HAS_VLLM_TIERING:
    try:
        from vllm.v1.kv_offload.tiering.factory import SecondaryTierFactory
        from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec

        class KvblockdTieringSpec(TieringOffloadingSpec):
            """spec_module_path vehicle: OffloadingSpecFactory resolves
            spec_name="KvblockdTieringSpec" from THIS module (the registry
            shadows the stock "TieringOffloadingSpec" name, so an out-of-tree
            name is what makes vLLM import us — and importing us is what
            registers the "kvblockd" tier below). Behavior is unchanged."""

        try:
            SecondaryTierFactory.register_tier(
                "kvblockd", "vllm_kvblockd.tier_manager", "KvblockdTierManager"
            )
        except ValueError:
            pass  # already registered (double import) — fine
    except Exception as exc:  # pragma: no cover - tiering moved/renamed
        logger.warning("kvblockd tier registration unavailable: %s", exc)
