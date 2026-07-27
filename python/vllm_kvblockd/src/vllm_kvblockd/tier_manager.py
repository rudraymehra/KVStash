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
cache_salt (first-block extra keys) — salt isolation at this altitude is
satisfied upstream; we bind the OffloadKey to OUR config identity with
BLAKE3(fingerprint || offload_key) (config.tier_wire_key), where the
fingerprint mirrors FileMapper's config.json fields (parallel-agnostic).
Cross-instance sharing therefore requires PYTHONHASHSEED pinned identically
everywhere — enforced loudly at construction.

GPU end-to-end validation is DEFERRED — see python/vllm_kvblockd/DEFER.md
for the exact revisit trigger. The unit suite
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
from kvblockd.errors import ConnectionLost

from .config import (
    fingerprint,
    parse_endpoint,
    require_pinned_hashseed,
    tier_fingerprint_fields,
    tier_wire_key,
)

logger = logging.getLogger("vllm_kvblockd")

# After a failed dial, further dial attempts short-circuit for this long —
# tasks fail fast (jobs report failed immediately) instead of each paying a
# fresh connect_timeout serially (the connector's proven breaker, ported).
_REDIAL_BACKOFF_S = 5.0
# One task-failure line per kind per window, with a suppressed count — 32
# workers against a dead daemon otherwise emit thousands of warnings/sec.
_TASK_WARN_INTERVAL_S = 10.0

# _AsyncExistsBatcher tri-state extras. A definitive True/False comes only
# from the daemon; a FAILED round-trip yields UNKNOWN_MISS — answered as a
# miss NOW (fail-open: a dead daemon must resolve lookups, never park them)
# but re-queried on the key's next lookup, so a transient blip cannot poison
# a key to MISS for the lifetime of a long-running request. REFRESHING marks
# a re-query already queued (still answered as a miss until it lands).
_UNKNOWN_MISS = object()
_REFRESHING = object()
_ABSENT = object()  # _AsyncExistsBatcher.lookup's "never seen" sentinel


class _ErrorReporter:
    """Keyed, rate-limited failure reporter with suppressed counts: at most
    one line per key per interval, each line carrying how many identical
    reports the window swallowed. Thread-safe (pool workers share it)."""

    def __init__(self, interval: float = _TASK_WARN_INTERVAL_S):
        self._interval = interval
        self._lock = threading.Lock()
        self._state: dict[str, tuple[float, int]] = {}  # key -> (last emit, suppressed)

    def report(self, key: str, msg: str, exc: BaseException | None = None) -> None:
        now = time.monotonic()
        with self._lock:
            last, suppressed = self._state.get(key, (0.0, 0))
            if now - last < self._interval:
                self._state[key] = (last, suppressed + 1)
                return
            self._state[key] = (now, 0)
        if suppressed:
            logger.warning("%s (x%d suppressed in last %.0fs): %s",
                           msg, suppressed, self._interval, exc)
        else:
            logger.warning("%s: %s", msg, exc)


class _TierStats:
    """Counters for the tier altitude's failure paths — the metrics an
    on-call alerts on where before there was only a log storm. Scraped via
    KvblockdTierManager.stats_snapshot()."""

    __slots__ = ("_lock", "dial_failures", "loads_failed", "stores_failed",
                 "touches_failed")

    def __init__(self):
        self._lock = threading.Lock()
        self.stores_failed = 0
        self.loads_failed = 0
        self.touches_failed = 0
        self.dial_failures = 0

    def bump(self, name: str, n: int = 1) -> None:
        with self._lock:
            setattr(self, name, getattr(self, name) + n)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "stores_failed": self.stores_failed,
                "loads_failed": self.loads_failed,
                "touches_failed": self.touches_failed,
                "dial_failures": self.dial_failures,
            }

try:  # vLLM tiering present: subclass the real ABC + reuse its result types.
    from vllm.v1.kv_offload.base import LookupResult, RequestOffloadingContext
    from vllm.v1.kv_offload.tiering.base import JobResult
    from vllm.v1.kv_offload.tiering.base import SecondaryTierManager as _TierBase

    _HAS_VLLM_TIERING = True
except Exception:  # noqa: BLE001 — availability fallback: tiering ABC absent/moved  # pragma: no cover - exercised by the no-vllm test env
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
    tiering/fs JobState, plus a report flag for fire-and-forget work).

    task_bound_s: per-task wall-clock budget, stamped when a worker STARTS
    the task (start-relative — an enqueue-relative bound would charge
    cross-job queueing on the shared pool against the task). Feeds
    wait_idle's progress watchdog; None = unbounded."""

    __slots__ = ("_completed", "_lock", "_n_tasks", "_success", "job_id", "kind",
                 "report", "task_bound_s")

    def __init__(self, job_id: int, n_tasks: int, report: bool = True, kind: str = "job",
                 task_bound_s: float | None = None):
        self.job_id = job_id
        self.task_bound_s = task_bound_s
        self._n_tasks = n_tasks
        self._completed = 0
        self._success = True
        self._lock = threading.Lock()
        self.report = report
        self.kind = kind  # "load" | "store" — failure telemetry + reporter key

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
    take our worker machinery with it.

    Every enqueued job may carry two wall-clock bounds: bound_s (the job's
    ENQUEUE-relative serial worst case — past it, the job's still-QUEUED
    tasks are shed as failed: never started, so no partial copy exists and
    the framework recovers by recompute) and task_bound_s (each task's
    START-relative budget, stamped when a worker picks it up). wait_idle
    waits for every RUNNING task up to its own start-relative deadline —
    enqueue-relative bounds alone ignored cross-job queueing on the shared
    pool and returned while a healthy in-budget task was still copying into
    the primary memoryview — and returns early ONLY when every remaining
    task has outlived its own from-start budget (a client bug: started
    tasks are wire-deadline-bounded), so drain_jobs can neither wedge the
    scheduler process nor unpin slots under a live copy."""

    def __init__(self, n_read: int, n_write: int, name: str = "kvblockd_tier",
                 reporter: _ErrorReporter | None = None):
        self._load_q: deque = deque()
        self._store_q: deque = deque()
        self._cond = threading.Condition(threading.Lock())
        self._stop = False
        self._finished: deque[tuple[int, bool]] = deque()
        self._inflight = 0  # guarded by _cond
        # job_id -> wall-clock deadline (monotonic) or None (unbounded);
        # one entry per INFLIGHT job, guarded by _cond. Past it wait_idle
        # sheds the job's still-QUEUED tasks (started ones are never shed).
        self._job_deadlines: dict[int, float | None] = {}
        # RUNNING-task watchdog: token -> monotonic deadline, stamped when a
        # worker STARTS a bounded task and popped when it settles; guarded
        # by _cond. wait_idle waits on these, never on enqueue-time bounds.
        self._running_deadlines: dict[int, float] = {}
        self._task_seq = 0  # watchdog token source; guarded by _cond
        self._reporter = reporter if reporter is not None else _ErrorReporter()
        self._threads = [
            threading.Thread(target=self._worker, args=(True,), name=f"{name}_l{i}", daemon=True)
            for i in range(n_read)
        ] + [
            threading.Thread(target=self._worker, args=(False,), name=f"{name}_s{i}", daemon=True)
            for i in range(n_write)
        ]
        for t in self._threads:
            t.start()

    def _enqueue(self, q: deque, state: _JobState | None, tasks, n_tasks: int,
                 bound_s: float | None = None) -> None:
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
                self._job_deadlines[state.job_id] = (
                    None if bound_s is None else time.monotonic() + bound_s)
            for fn in tasks:
                q.append((fn, state))
            self._cond.notify(n_tasks)

    def enqueue_load(self, job_id: int, n_tasks: int, tasks,
                     bound_s: float | None = None,
                     task_bound_s: float | None = None) -> None:
        # task_bound_s defaults to bound_s: a job's whole bound is a valid
        # (conservative) per-task ceiling, so a bounded job can never regress
        # to the unsafe enqueue-relative-only wait.
        self._enqueue(self._load_q,
                      _JobState(job_id, n_tasks, kind="load",
                                task_bound_s=bound_s if task_bound_s is None
                                else task_bound_s),
                      tasks, n_tasks, bound_s)

    def enqueue_store(self, job_id: int, n_tasks: int, tasks,
                      bound_s: float | None = None,
                      task_bound_s: float | None = None) -> None:
        self._enqueue(self._store_q,
                      _JobState(job_id, n_tasks, kind="store",
                                task_bound_s=bound_s if task_bound_s is None
                                else task_bound_s),
                      tasks, n_tasks, bound_s)

    def enqueue_fire_and_forget(self, fn) -> None:
        """Jobless best-effort task (TOUCH recency): no result, no inflight
        accounting — drain_jobs must not wait on advisory work."""
        with self._cond:
            if self._stop:
                return  # advisory work owes nothing; drop it
            self._store_q.append((fn, None))
            self._cond.notify(1)

    def fail_job(self, job_id: int) -> None:
        """Report a job FAILED without running anything — the submit-side
        boundary check path (e.g. a keys/block_ids length mismatch must fail
        the job loudly, never truncate silently, and never raise into the
        scheduler)."""
        with self._cond:
            self._finished.append((job_id, False))
            self._cond.notify_all()

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

    def _shed_expired_queued(self, now: float) -> bool:
        """Called with _cond held. Fail (task_done(False)) the still-QUEUED
        tasks of every inflight job whose enqueue-relative bound has expired.
        Shedding un-started work is safe — no partial copy exists, the job
        reports failed and the framework recovers by recompute — where the
        old early-return abandoned a RUNNING copy to slot reuse (torn KV).
        Tasks a worker already picked up are NEVER touched here."""
        expired = {jid for jid, b in self._job_deadlines.items()
                   if b is not None and b <= now}
        if not expired:
            return False
        shed = 0
        for dq in (self._load_q, self._store_q):
            for _ in range(len(dq)):  # full rotation preserves queue order
                fn, state = dq.popleft()
                if state is None or state.job_id not in expired:
                    dq.append((fn, state))
                    continue
                shed += 1
                finished, _ = state.task_done(False)
                if finished:
                    if state.report:
                        self._finished.append((state.job_id, False))
                    self._inflight -= 1
                    self._job_deadlines.pop(state.job_id, None)
        if shed:
            logger.warning(
                "kvblockd tier drain_jobs: shed %d queued (never-started) task(s) "
                "of job(s) past their wall-clock bound — the job(s) report failed "
                "and the framework recomputes; running tasks are never shed", shed)
            self._cond.notify_all()
        return shed > 0

    def wait_idle(self) -> None:
        """Block until no job task is in flight. NEVER abandons a RUNNING
        task — the SecondaryTierManager.drain_jobs contract forbids aborting
        mid-flight copies (a partial copy corrupts the primary memoryview or
        the backing store), and returning while one runs lets the framework
        unpin/reuse primary slots under a live recv_into. Results stay
        queued for get_finished_jobs(). Gated on _stop so a racing shutdown
        can never strand this wait.

        Fail-open armor, two altitudes (only when EVERY inflight job is
        bounded; an unbounded job — stores, or load_deadline_s<=0 — keeps
        today's unbounded wait):
          - a job past its enqueue-relative bound has its QUEUED tasks shed
            (safe: never started), so cross-job queueing on the shared pool
            is load-shedding, never corruption;
          - every RUNNING task is waited on up to its own START-relative
            deadline (stamped by the worker that picked it up). Only when
            every remaining task has outlived its own from-start budget —
            a genuine client bug, since each started task's wire deadline
            bounds it — does this return early, LOUDLY."""
        with self._cond:
            while not (self._stop or self._inflight == 0):
                bounds = list(self._job_deadlines.values())
                if not bounds or any(b is None for b in bounds):
                    self._cond.wait()
                    continue
                now = time.monotonic()
                if self._shed_expired_queued(now):
                    continue  # accounting changed: re-evaluate everything
                # Next actionable instant: an unexpired job bound (queued
                # work to shed) or a running task's start-relative deadline.
                events = [b for b in self._job_deadlines.values()
                          if b is not None and b > now]
                events += [d for d in self._running_deadlines.values() if d > now]
                if events:
                    self._cond.wait(min(events) - now)
                    continue
                logger.warning(
                    "kvblockd tier drain_jobs: %d job(s) still in flight past every "
                    "job bound AND every running task's own start-relative deadline "
                    "— returning to keep the scheduler alive (started tasks are "
                    "wire-deadline-bounded, so this is a client bug; results still "
                    "surface via get_finished_jobs)",
                    self._inflight)
                return

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
                        self._job_deadlines.pop(state.job_id, None)
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
                # Task STARTS here: stamp its start-relative watchdog
                # deadline (wait_idle waits on these — an enqueue-relative
                # bound would charge cross-job queueing against the task).
                token = 0
                if state is not None and state.task_bound_s is not None:
                    self._task_seq += 1
                    token = self._task_seq
                    self._running_deadlines[token] = (
                        time.monotonic() + state.task_bound_s)
            try:
                task()
                ok = True
            except Exception as exc:  # noqa: BLE001 — worker thread must survive any task failure; the job is reported failed below
                kind = getattr(state, "kind", "touch")
                self._reporter.report(
                    kind, f"kvblockd tier {kind} task failed "
                          f"(job {getattr(state, 'job_id', '-')})", exc)
                ok = False
            if state is None:
                continue  # fire-and-forget never registers a watchdog token
            finished, success = state.task_done(ok)
            if not token and not finished:
                continue
            with self._cond:
                if token:
                    self._running_deadlines.pop(token, None)
                if finished:
                    if state.report:
                        self._finished.append((state.job_id, success))
                    self._inflight -= 1
                    self._job_deadlines.pop(state.job_id, None)
                # A settled task changes what wait_idle is waiting on even
                # when its job is not finished (the watchdog set shrank).
                self._cond.notify_all()


class _AsyncExistsBatcher:
    """Non-blocking EXISTS for the scheduler thread (the AsyncLookupManager
    pattern): lookup() accumulates unseen keys and returns the cached
    tri-state; flush() posts the step's batch to a background thread (one
    BATCH_EXISTS per step); results drain lazily on the next lookup().

    Ownership model (no locks, mirrors upstream): _state/_batch belong to the
    scheduler thread; the two SimpleQueues are the only cross-thread edges.

    Failure semantics (tri-state over boolean for cached remote facts): a
    FAILED round-trip resolves its keys to _UNKNOWN_MISS — answered MISS
    immediately (fail-open: a dead daemon must never park a request on RETRY
    forever) but RE-QUERIED on the key's next lookup, so a transient blip
    upgrades back to HIT after recovery instead of poisoning the key to MISS
    for the request's lifetime. The worker also COALESCES its backlog: every
    batch queued during an outage collapses into one round-trip, so recovery
    costs one exchange, not one per stalled step."""

    def __init__(self, exists_fn, tier_type: str):
        self._exists_fn = exists_fn  # list[wire_key] -> list[bool] | None (failure)
        self._state: dict[bytes, object] = {}  # key -> True|False|None|sentinels
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
        """Returns True (HIT), False (MISS — definitive or fail-open), or
        None (RETRY: a round-trip is in flight and no answer is cached)."""
        if self._need_drain:
            self._drain()
            self._need_drain = False
        cur = self._state.get(key, _ABSENT)
        if cur is _ABSENT:
            self._state[key] = None
            self._batch.append(key)
            cur = None
        elif cur is _UNKNOWN_MISS:
            # Fail-open answer stays MISS, but the fact is unknown — queue a
            # re-query so a recovered daemon can upgrade it to HIT.
            self._state[key] = _REFRESHING
            self._batch.append(key)
            cur = _REFRESHING
        self._key_reqs.setdefault(key, set()).add(req_id)
        self._req_keys.setdefault(req_id, set()).add(key)
        if cur is None:
            return None
        return cur is True  # False | _REFRESHING | _UNKNOWN_MISS -> miss

    def pending_batches(self) -> int:
        """Telemetry: batches queued for the worker (backlog gauge)."""
        return self._in_q.qsize()

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
            stop_after = False
            while True:  # coalesce the backlog: one round-trip per drain
                try:
                    more = self._in_q.get_nowait()
                except queue.Empty:
                    break
                if more is None:
                    stop_after = True  # serve what we have, then exit
                    break
                batch.extend(more)
            present = self._exists_fn(batch)
            if present is None:  # round-trip failed: unknown, not all-False
                self._out_q.put([(k, _UNKNOWN_MISS) for k in batch])
            else:
                self._out_q.put(list(zip(batch, present)))
            if stop_after:
                return


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
        # None (the default) ties the connection count to the worker count
        # (n_read_threads + n_write_threads): a pool of W workers against a
        # streams-limited semaphore smaller than W just queues the surplus on
        # Pool._sem — one source of truth for concurrency. An explicit value
        # is honored unchanged.
        streams: int | None = None,
        n_read_threads: int = 16,
        n_write_threads: int = 16,
        tile_keys: int = 8,
        verify: bool = True,
        op_timeout: float = 10.0,
        # Wall-clock ceiling for ONE load task's batch_get_scatter (mirrors
        # the connector's kvblockd_load_deadline_s): op_timeout bounds each
        # recv, but a daemon that trickles bytes passes every per-recv check
        # forever and would wedge drain_jobs — which runs in the SCHEDULER
        # process. Past the deadline the task fails (the framework recovers
        # by recompute — a failed load job is recoverable BY DESIGN in the
        # tiering contract and is counted in stats). <=0 disables (the
        # pre-wave behavior).
        load_deadline_s: float = 30.0,
        block_bytes: int | None = None,
        # RFC-#38260-shaped tier configs carry these INSIDE the tier dict; the
        # merged factory forwards every non-"type" key to us, so accept and
        # ignore them rather than crash on **config.
        module_path: str | None = None,
        class_name: str | None = None,
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
        self._streams = max(1, (n_read_threads + n_write_threads) if streams is None
                            else int(streams))
        self._verify = verify
        self._op_timeout = op_timeout
        self._load_deadline_s = float(load_deadline_s)
        self._tile = max(1, int(tile_keys))
        self._client: Client | None = None
        self._client_lock = threading.Lock()
        self._next_dial = 0.0  # monotonic gate arming the dial breaker
        self._closed = False

        self._fp = fingerprint(tier_fingerprint_fields(offloading_spec))
        self._stats = _TierStats()
        self._reporter = _ErrorReporter()
        self._pool = _DualQueuePool(n_read_threads, n_write_threads,
                                    reporter=self._reporter)
        self._lookup = _AsyncExistsBatcher(self._batch_exists, tier_type)
        self._last_err: tuple[float, str] | None = None
        # job_id -> "load"|"store", pruned in get_finished_jobs: failed-job
        # counters need the kind after the pool has forgotten the state.
        self._job_kinds: dict[int, str] = {}

    # ------------------------------------------------------------------
    # client + key plumbing
    # ------------------------------------------------------------------
    def _ensure(self) -> Client:
        with self._client_lock:
            if self._closed:
                raise ConnectionError("tier manager shut down")
            if self._client is None:
                # Dial breaker (the connector's, ported): without it every
                # pool task independently pays a fresh connect_timeout dial
                # during an outage — 32 workers × 5s serialized on this lock.
                now = time.monotonic()
                if now < self._next_dial:
                    raise ConnectionError("kvblockd dial suppressed after recent failure")
                try:
                    self._client = Client(
                        self._addr, namespace=self._namespace, token=self._token,
                        streams=self._streams, op_timeout=self._op_timeout,
                        verify=self._verify,
                        # Fan-out hint: every pool worker already provides
                        # parallelism, so per-tile batch_get_scatter takes the
                        # nshards<=1 inline path — no cross-shard fanning of
                        # 8-key tiles from 32 concurrent workers.
                        get_fanout=1,
                    )
                except Exception:
                    self._next_dial = time.monotonic() + _REDIAL_BACKOFF_S
                    self._stats.bump("dial_failures")
                    raise
            return self._client

    def _drop_client(self, exc: BaseException) -> None:
        """Connection-class failure out of a task: the pooled conns are dead
        (daemon gone/blackholed) — drop the whole client and arm the breaker
        so the outage costs one connect_timeout per backoff window, not one
        per task (the connector's discipline, mirrored)."""
        if not isinstance(exc, (ConnectionLost, OSError)):
            return  # e.g. StatusError: the stream is in sync, keep the pool
        with self._client_lock:
            client, self._client = self._client, None
            if client is None:
                return  # dial failure/suppression: _ensure already armed it
            self._next_dial = time.monotonic() + _REDIAL_BACKOFF_S
        client.close()

    def _key(self, offload_key) -> bytes:
        return tier_wire_key(self._fp, bytes(offload_key))

    def _warn(self, what: str, exc: BaseException) -> None:
        now = time.monotonic()
        if self._last_err is None or now - self._last_err[0] > 10.0 or self._last_err[1] != what:
            self._last_err = (now, what)
            logger.warning("kvblockd tier %s failed: %s", what, exc)

    def _batch_exists(self, wire_keys: list[bytes]) -> list[bool] | None:
        """Background-thread EXISTS; never raises. A FAILED round-trip returns
        None — 'unknown', which the batcher answers as a fail-open miss and
        re-queries later — never a definitive all-False (a transient blip must
        not poison keys to MISS for a long-running request's lifetime). Uses
        the per-key bitmap when the daemon granted FEAT_EXISTS_BITMAP, else
        falls back to the consecutive-prefix count."""
        try:
            n_consec, per_key = self._ensure().batch_exists(wire_keys)
        except Exception as exc:  # noqa: BLE001 — never raises (documented): failure = unknown, fail-open miss
            self._warn("BATCH_EXISTS", exc)
            self._drop_client(exc)
            return None
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
            except Exception as exc:  # noqa: BLE001 — fire-and-forget background touch; never raises into the pool thread
                self._warn("TOUCH", exc)
                self._stats.bump("touches_failed")
                self._drop_client(exc)

        self._pool.enqueue_fire_and_forget(_touch)

    def _job_pairs(self, job_metadata, kind: str) -> list | None:
        """(key, block_id) pairs, or None after failing the job LOUDLY on a
        length mismatch: plain zip would silently truncate, and on the load
        path truncation means the framework treats unfilled primary slots as
        loaded — the exact corruption class the module docstring forbids.
        Never raises into the scheduler (never-raise boundary)."""
        keys = list(job_metadata.keys)
        bids = list(job_metadata.block_ids)
        if len(keys) != len(bids):
            jid = int(job_metadata.job_id)
            logger.error(
                "kvblockd tier %s job %d: %d keys vs %d block_ids — failing the "
                "job (a truncated pairing would silently corrupt; never truncate)",
                kind, jid, len(keys), len(bids))
            self._stats.bump(f"{kind}s_failed")
            self._pool.fail_job(jid)
            return None
        return list(zip(keys, bids, strict=True))

    def _check_offsets(self, tile) -> None:
        """Bounds-check every block offset BEFORE any wire traffic: a bogus
        block_id past the view end yields a SHORT Python slice — a silent
        short PUT (permanent miss at load time) or an alloc buffer shorter
        than the promised body. The raise fails the task -> the job reports
        failed loudly (pool worker catches; never the scheduler)."""
        limit = len(self._bytes)
        for _key, bid in tile:
            off = int(bid) * self._block_size
            if off < 0 or off + self._block_size > limit:
                raise ValueError(
                    f"block_id {int(bid)} out of range: byte range "
                    f"[{off}, {off + self._block_size}) exceeds the primary view "
                    f"({limit} bytes)")

    def submit_store(self, job_metadata) -> None:
        """Primary -> kvblockd. Enqueue only; copies happen on pool threads
        while the framework keeps the block_ids slots pinned. Already-present
        blocks dedup server-side (write-once: OK_EXISTS)."""
        jid = int(job_metadata.job_id)
        pairs = self._job_pairs(job_metadata, "store")
        if pairs is None:
            return
        tiles = [pairs[i : i + self._tile] for i in range(0, len(pairs), self._tile)]

        def make_task(tile):
            def _store():
                self._check_offsets(tile)  # before any wire traffic
                client = self._ensure()
                try:
                    for key, bid in tile:
                        off = int(bid) * self._block_size
                        client.put(self._key(key), self._bytes[off : off + self._block_size])
                except (ConnectionLost, OSError) as exc:
                    self._drop_client(exc)  # breaker discipline; the job fails
                    raise

            return _store

        self._job_kinds[jid] = "store"
        self._pool.enqueue_store(jid, len(tiles), (make_task(t) for t in tiles))

    def submit_load(self, job_metadata) -> None:
        """kvblockd -> primary. BATCH_GET per tile, received DIRECTLY into the
        primary memoryview slices (zero-copy scatter). Any miss/short/corrupt
        block fails the whole job — the framework must never treat a
        partially-filled slot as loaded. A FAILED load job is recoverable BY
        DESIGN: the framework recomputes the blocks (the vLLM tiering
        contract), so eviction between lookup-HIT and load costs latency,
        never a wrong byte — counted in stats (loads_failed) so an operator
        can see eviction races without log archaeology."""
        jid = int(job_metadata.job_id)
        pairs = self._job_pairs(job_metadata, "load")
        if pairs is None:
            return
        tiles = [pairs[i : i + self._tile] for i in range(0, len(pairs), self._tile)]

        def make_task(tile):
            def _load():
                self._check_offsets(tile)  # before any wire traffic
                client = self._ensure()
                wire = [self._key(key) for key, _ in tile]
                slots = [int(bid) * self._block_size for _, bid in tile]

                def alloc(idx, prefix, body_len):
                    if body_len != self._block_size:
                        return None  # wrong-sized blob: refuse, count as miss
                    off = slots[idx]
                    return self._bytes[off : off + self._block_size]

                # Per-task wall-clock deadline: op_timeout bounds each recv,
                # but only this stops a daemon that trickles bytes forever —
                # the trickle-armor batch_get_scatter was built for.
                deadline = (time.monotonic() + self._load_deadline_s
                            if self._load_deadline_s > 0 else None)
                try:
                    statuses = client.batch_get_scatter(wire, 0, alloc, deadline=deadline)
                except (ConnectionLost, OSError) as exc:
                    self._drop_client(exc)  # breaker discipline; the job fails
                    raise
                misses = sum(1 for s in statuses if s != kp.Status.OK)
                if misses:
                    raise LookupError(f"{misses}/{len(tile)} blocks missing on load")

            return _load

        # Two bounds for wait_idle: the JOB bound (enqueue-relative serial
        # worst case: n_tasks armed windows + op_timeout slack) sheds only
        # still-QUEUED tasks when it expires; each STARTED task gets its own
        # start-relative watchdog (one armed window + the same slack) — the
        # wire deadline inside the task binds first, so the watchdog firing
        # means the client itself wedged.
        bound = (self._load_deadline_s * len(tiles) + self._op_timeout
                 if self._load_deadline_s > 0 else None)
        task_bound = (self._load_deadline_s + self._op_timeout
                      if self._load_deadline_s > 0 else None)
        self._job_kinds[jid] = "load"
        self._pool.enqueue_load(jid, len(tiles), (make_task(t) for t in tiles), bound,
                                task_bound)

    def get_finished_jobs(self) -> Iterable[JobResult]:
        out = []
        for jid, ok in self._pool.get_finished():
            kind = self._job_kinds.pop(jid, None)
            if not ok and kind is not None:
                self._stats.bump(f"{kind}s_failed")
            out.append(JobResult(job_id=jid, success=ok))
        return out

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

    def stats_snapshot(self) -> dict[str, int]:
        """Real counters for the failure paths (misses-by-cause is the
        product's primary signal): failed store/load jobs, dropped touches,
        dial failures, plus the lookup batcher's backlog gauge. Rigs and
        operators scrape this; the vLLM-typed get_stats() hook below stays
        conservative until the OffloadingConnectorStats wiring lands."""
        snap = self._stats.snapshot()
        snap["lookup_batches_pending"] = self._lookup.pending_batches()
        return snap

    def get_stats(self):
        if _HAS_VLLM_TIERING:
            # The upstream contract types this OffloadingConnectorStats |
            # None and feeds it to vLLM's metrics aggregation; handing that
            # machinery an untyped dict risks crashing the engine (never-
            # raise posture). Typed wiring lands with the GPU e2e —
            # stats_snapshot() carries the counters until then.
            return None
        return self.stats_snapshot()


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
    except Exception as exc:  # noqa: BLE001 — registration is best-effort: tiering moved/renamed  # pragma: no cover
        logger.warning("kvblockd tier registration unavailable: %s", exc)
