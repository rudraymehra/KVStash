"""StoreQueue — the write-behind store machinery of KvblockdConnector
(kvblockd_async_store), extracted verbatim from connector.py.

Unit ownership: the per-worker deques, the byte gauges and public
disclosure counters, the drain worker threads and their load-priority
gate, the acked-key dedupe LRU, the pinned store-slot pool, and the
gathered-store staging live HERE. The composing connector aliases this
state back onto itself (its test/bench seams are unchanged) and delegates
the methods.

Cross-boundary calls route back through the composing connector
(``self._c``) ON PURPOSE, not as style: tests and operators monkeypatch
those seams as CONNECTOR instance attributes (e.g. ``conn._sq_enqueue``,
``conn._store_sync``, ``conn._scratch_ring``), and a bound-method call
inside this class would silently bypass the patch. Behavior is
byte-for-byte the pre-extraction connector's.
"""

from __future__ import annotations

import logging
import threading
import time
import zlib
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from kvblockd.errors import ConnectionLost

# Late-bound module reads (_cm._REDIAL_BACKOFF_S, _cm._DRAIN_GATE_CEILING_S):
# tests patch those knobs on the CONNECTOR module, and an imported copy would
# freeze the unpatched value.
from . import connector as _cm
from .connector import (
    _DIAL_PENDING_PARK_S,
    BLOB_PREFIX_LEN,
    _DialPending,
    _torch,
    encode_blob_prefix,
)
from .slab_loader import _SCATTER_CHUNK, _SCRATCH_MAX_FAILS

if TYPE_CHECKING:  # import cycle by design: connector.py imports this module
    from .connector import KvblockdConnector, KvbReqMeta

logger = logging.getLogger("vllm_kvblockd")


class _AckedKeyLRU:
    """Bounded TTL'd set of keys the daemon recently ACKED (OK/OK_EXISTS).
    Populated ONLY by the kvb-store drain thread on put verdicts — an ack is
    the daemon saying "present", so a false positive is impossible; a false
    negative (capacity/TTL eviction here) just costs today's re-put. ADVISORY
    by construction: the daemon evicts (S3-FIFO/TTL), so an entry's ack-time
    truth decays — the TTL bounds how long it is trusted, and it is NEVER
    refreshed on a hit (a hit adds no new evidence of daemon-side presence;
    only a fresh ack does). Write-once keys make content staleness
    impossible: the worst wrong answer is a skipped re-store after a daemon
    eviction — a future miss bounded by the window, never a wrong byte.
    Thread-safe: hit() runs on the engine thread, add() on the drain thread."""

    def __init__(self, cap: int, ttl_s: float):
        self._cap = cap
        self._ttl = ttl_s
        self._lock = threading.Lock()
        self._entries: dict[bytes, float] = {}  # key -> ack time (insertion-ordered)

    def add(self, key: bytes) -> None:
        with self._lock:
            self._entries.pop(key, None)  # re-ack refreshes both TTL and recency
            self._entries[key] = time.monotonic()
            while len(self._entries) > self._cap:
                self._entries.pop(next(iter(self._entries)))  # oldest ack out

    def hit(self, key: bytes) -> bool:
        with self._lock:
            t = self._entries.get(key)
            if t is None:
                return False
            if time.monotonic() - t > self._ttl:
                self._entries.pop(key, None)  # expired: past acks prove nothing now
                return False
            return True

    def clear(self) -> None:
        """Forget every ack at once — for the moment a connection-class
        failure proves them ALL stale (a daemon restart empties the store,
        so pre-outage acks would suppress the self-healing re-put)."""
        with self._lock:
            self._entries.clear()


@dataclass
class _StagePlan:
    """One request's gathered-store staging, enqueue DEFERRED until after the
    device sync (wait_for_save). items rows are mutable [j, key, buf, slot_id]
    — slot_id is nulled as ownership moves (queue / free list / rebuild), so
    cleanup at any exit frees each slot exactly once. names/bytes_per_layer/
    block_ids/prefix ride along so a failed sync can rebuild the blobs from
    the still-valid paged memory."""

    req_id: str
    total: int
    end: int
    items: list[list]  # [j, key, bytearray | memoryview, slot_id | None]
    dev: object
    names: list[str]
    bytes_per_layer: int
    block_ids: list[int]
    prefix: bytes
    # items[:sent] have SETTLED accounting (enqueued, or refused-and-counted
    # by the tail-skip); _abandon_plan counts exactly items[sent:] — the
    # blocks a mid-plan raise would otherwise lose with dropped_puts=0.
    sent: int = 0


class StoreQueue:
    """The connector's write-behind store lifecycle, unit-owned.

    Locking topology (unchanged by the extraction): ONE Condition,
    _sq_cond, guards ALL queue state — the per-worker deques, the byte
    gauges, the public disclosure counters, the slot free list,
    _loads_inflight and the drain-gate arm. Slot-pool invariant under
    _sq_cond: free + staging + queued + inflight == total slots,
    single-free at every exit (the ownership walk is documented on the
    state below and on _finish_stage/_store_drain, moved verbatim)."""

    def __init__(self, connector: KvblockdConnector):
        # The composing connector: every cross-boundary call routes through
        # it so connector-level monkeypatch seams always intercept.
        self._c = connector
        # Write-behind store queue (kvblockd_async_store, default on):
        # wait_for_save stages OWNED byte copies here and returns; N daemon
        # workers ("kvb-store[-i]", lazily started on first enqueue) drain
        # them — one deque per worker, requests HASHED to a worker (stable
        # crc32 of req_id), so per-request block order is preserved exactly
        # as the single FIFO preserved it: a partial delivery stays a usable
        # consecutive prefix under the prefix-chain keys. Effective count is
        # min(kvblockd_store_drain_workers, streams) — each worker owns one
        # pooled conn's worth of put throughput; default 1 = the original
        # single-thread drain, byte-identical. All queue state (deques, byte
        # gauges, slot free list, public counters) is guarded by _sq_cond.
        self._store_workers = max(1, min(self._c._cfg.store_drain_workers,
                                         self._c._cfg.streams))
        self._sqs: list[deque[tuple[bytes, bytearray | memoryview, int | None]]] = [
            deque() for _ in range(self._store_workers)]
        self._sq_cond = threading.Condition()
        self._sq_bytes = 0            # bytes currently queued
        self._sq_inflight = 0         # blocks popped, put() not finished
        self._sq_inflight_bytes = 0
        # In-flight blocks the SHUTDOWN disclosure already counted dropped
        # (join timed out mid-put). The drain thread reconciles when the put
        # finally returns: delivered -> un-count the pessimistic drop; failed
        # -> already disclosed dropped, must not ALSO count failed.
        self._sq_inflight_counted = 0
        # req_id -> FIRST store block that was dropped for that request. The
        # keys are a prefix chain, so every later block of the request is
        # unreachable by the consecutive-prefix lookup — later _stage_one
        # calls (chunked-prefill continuations) skip past the hole instead of
        # queueing dead bytes. Pruned in request_finished.
        self._store_holes: dict[str, int] = {}
        self._store_threads: list[threading.Thread | None] = [None] * self._store_workers
        self._store_stop = False      # shutdown: drain the remainder, then exit
        self._store_abort = False     # wedged flush: exit without draining
        # Public disclosure counters (the bench reads/greps these).
        self.dropped_puts = 0
        self.dropped_put_bytes = 0
        self.failed_puts = 0
        self.deduped_puts = 0  # blocks skipped by the acked-key store dedupe
        # Acked-key store dedupe (kvblockd_store_dedupe_keys): drain-thread
        # acks in, _stage_one leading-run skips out. None = disabled (knob 0
        # or TTL<=0) — every path then behaves exactly as before the LRU.
        self._acked_keys: _AckedKeyLRU | None = None
        if self._c._cfg.store_dedupe_keys > 0 and self._c._cfg.store_dedupe_ttl_s > 0:
            self._acked_keys = _AckedKeyLRU(self._c._cfg.store_dedupe_keys,
                                            self._c._cfg.store_dedupe_ttl_s)
        # Loads currently pulling KV (guarded by _sq_cond like all queue
        # state); the drain's load-priority gate parks on it.
        self._loads_inflight = 0
        # Episode arm for the drain's load-priority gate (under _sq_cond —
        # armed by the drain, cleared on episode edges): the gate parks ONCE
        # per raised episode, not once per pop — N queued blobs under one
        # raised gate cost one ceiling total, never N ceilings. Episodes are
        # EDGE-TRIGGERED: start_load_kv clears the arm on the 0->1 transition
        # of _loads_inflight, because the drain only samples the counter
        # between puts — an episode that begins while the drain is inside
        # put() must not inherit the previous episode's expired arm.
        self._drain_park_until: float | None = None

        # Pinned store-slot pool (CUDA gathered-store fast path): one pinned
        # tensor cut into `total`-stride slots (32B prefix + body), leased at
        # stage time, released when the slot's blob leaves the queue for good.
        # Separate from the LOAD slab on purpose: a slot's lifetime crosses
        # into the drain thread, while the load slab's contract ends at its
        # trailing stream synchronize. Slot ownership walk: ALLOCATED (stager
        # frees on a mid-fill failure) -> QUEUED (stager frees on enqueue
        # refusal; shutdown walks _sq and frees before clear()) -> INFLIGHT
        # (ONLY the drain's reconcile block frees — the memoryview sits in
        # _send_frame's iovec, and reusing it mid-sendmsg would publish
        # garbage under a content-chained key; a requeue keeps the lease).
        # Invariant under _sq_cond: free + staging + queued + inflight ==
        # total slots, single-free at every exit.
        self._store_slab = None           # 1-D pinned uint8 torch tensor
        self._store_slab_np = None        # numpy view (memoryview source)
        self._store_slot_stride = 0       # bytes per slot (== blob total)
        self._store_slot_free: list[int] = []   # free slot ids, under _sq_cond
        self._store_slots_total = 0
        self._store_slab_disabled = False  # first-alloc failure -> permanent
        self._store_gather_fails = 0      # consecutive gather-path failures
        self._store_gather_disabled = False  # latched after _SCRATCH_MAX_FAILS
        # Store-path attribution, one-shot (+ one switch line) — deliberately
        # NOT _note_path: the bench greps "kvblockd load path:" and a store
        # stamp would consume that one-shot.
        self._reported_store_path = None
        self._store_path_switch_logged = False

    @property
    def _sq(self):
        """Worker 0's deque — exact for the default single-worker drain (the
        bench/test seam that inspects the staged queue); multi-worker callers
        must iterate _sqs."""
        return self._sqs[0]

    @property
    def _store_thread(self):
        """First live drain thread or None (single-worker back-compat seam)."""
        return next((t for t in self._store_threads if t is not None), None)

    # ------------------------------------------------------------------
    # write-behind store queue (kvblockd_async_store)
    # ------------------------------------------------------------------
    def _stage_one(self, req: KvbReqMeta) -> _StagePlan | None:
        """Copy every store-range block into ONE owned buffer per block
        (32B layout prefix + all layers, exactly the blob _store_one streams)
        and enqueue it. The copies MUST happen inside wait_for_save: after it
        returns the scheduler may reuse the paged blocks, and _block_bytes is
        zero-copy on CPU — draining the view later would stream whatever the
        engine wrote over it (silent corruption armor, not an optimization).

        CUDA paged tensors with a live slot pool stage through batched
        gathers + async D2H instead of layers x blocks synchronous copies;
        that path DEFERS the enqueues behind wait_for_save's single device
        sync and returns the plan. Everything else (and every gather-path
        failure — paged memory is still valid here) runs the original inline
        bytearray loop and returns None."""
        names, dtype_name, bytes_per_layer = self._c._layout()
        if not names:
            return None
        total = BLOB_PREFIX_LEN + bytes_per_layer * len(names)
        prefix = encode_blob_prefix(dtype_name, len(names), self._c._block_size,
                                    bytes_per_layer, total)
        seed = self._c._seed(req.cache_salt, req.mm_ids, req.lora_name)
        keys = self._c._chain_keys(req.req_id, seed, req.token_ids)
        end = min(req.store_end_block, len(keys), len(req.block_ids))
        hole = self._store_holes.get(req.req_id)
        if hole is not None and end > hole:
            # TAIL-SKIP, PERSISTED: an earlier step of this request already
            # dropped block `hole`, so every row at/past it is unreachable by
            # the consecutive-prefix lookup no matter how much budget freed up
            # since — count those rows dropped (no copies built) and cap the
            # loop below the hole.
            skipped = end - max(req.store_start_block, hole)
            if skipped > 0:
                with self._sq_cond:
                    self.dropped_puts += skipped
                    self.dropped_put_bytes += skipped * total
            end = hole
        if end <= req.store_start_block:
            return None
        start = req.store_start_block
        if self._acked_keys is not None:
            # Acked-key dedupe, BEFORE any copy/D2H: skip the LEADING run of
            # keys the daemon already acked. Leading-run only, on purpose —
            # acks land in per-request block order (FIFO drain), so the
            # target workload (a re-served local-prefix request) re-offers an
            # already-acked PREFIX; skipping a mid-range key would also break
            # _finish_stage's tail-skip arithmetic (it counts plan rows by
            # block index). A non-leading acked key just re-puts and collects
            # OK_EXISTS — today's cost, not a correctness event.
            while start < end and self._acked_keys.hit(keys[start]):
                start += 1
            if start > req.store_start_block:
                with self._sq_cond:
                    self.deduped_puts += start - req.store_start_block
            if start >= end:
                return None  # the whole range is recently-acked: nothing to stage
        dev = self._c._layer_kv[names[0]].device
        if (self._c._slab_path_ok(dev) and not self._store_gather_disabled
                and self._c._store_pool_ready(total)):
            plan = self._c._stage_gather(req, names, bytes_per_layer, total,
                                      prefix, keys, start, end)
            if plan is not None:
                self._c._note_store_path("gathered-slots")
                return plan
        self._c._note_store_path("bytearray")
        for j in range(start, end):
            buf = self._c._build_block_blob(req.block_ids[j], names,
                                         bytes_per_layer, prefix, total)
            if not self._c._sq_enqueue(keys[j], buf, rid=req.req_id):
                # TAIL-SKIP: block keys are a prefix chain, so once block j is
                # missing every later block of THIS request is unreachable by
                # BATCH_EXISTS's consecutive-prefix count — copying/queueing
                # them would spend budget on bytes no lookup can ever count.
                # Count them dropped (without building the copies), record the
                # hole so LATER steps of this request skip past it too, stop.
                self._store_holes[req.req_id] = j  # j < any prior hole (end is capped)
                skipped = end - j - 1
                if skipped > 0:
                    with self._sq_cond:
                        self.dropped_puts += skipped
                        self.dropped_put_bytes += skipped * total
                return None
        return None

    def _build_block_blob(self, bid: int, names, bytes_per_layer: int,
                          prefix: bytes, total: int) -> bytearray:
        """One owned blob for physical block bid — the original per-block
        copy loop, byte-for-byte (the gather path's fallback oracle)."""
        buf = bytearray(total)
        buf[:BLOB_PREFIX_LEN] = prefix
        dst = np.frombuffer(buf, dtype=np.uint8)  # writable view of buf
        for li, name in enumerate(names):
            src = self._c._block_bytes(self._c._layer_kv[name][bid])
            dst[BLOB_PREFIX_LEN + li * bytes_per_layer:
                BLOB_PREFIX_LEN + (li + 1) * bytes_per_layer] = src  # copies
        return buf

    # ------------------------------------------------------------------
    # gathered-store fast path (CUDA paged tensors, pinned slot pool)
    # ------------------------------------------------------------------
    def _store_pool_ready(self, total: int) -> bool:
        """Whether store slots of exactly this stride can be leased; allocates
        the pool at the first CUDA layout capture (_maybe_prewarm) or, if
        that never ran, lazily on the first CUDA store. Stride is fixed at
        first allocation — a layout change mid-run simply stops matching and
        the bytearray path serves (never realloc: queued slots reference the
        old tensor). Auto-size (config unset) = queue byte budget + 2 slots
        (1 in flight + 1 staging while the queue sits at budget). Enqueues
        are DEFERRED behind the device sync, so a lease can be denied while
        the queue still has room; denial is congestion, not failure — the
        block degrades to an IDENTICAL bytearray blob (same bytes, same
        accounting), never a drop. Never raises."""
        if self._store_slab_disabled or total <= 0:
            return False
        if self._store_slab is not None:
            return total == self._store_slot_stride
        cfg_bytes = self._c._cfg.store_staging_bytes
        if cfg_bytes is not None and cfg_bytes <= 0:
            return False  # explicitly off: not a failure, no latch
        if cfg_bytes is None:
            n_slots = self._c._cfg.store_queue_bytes // total + 2
        else:
            n_slots = cfg_bytes // total
        if n_slots <= 0:
            self._store_slab_disabled = True  # can never fit one blob
            self._c._log.maybe("store-slab",
                            f"kvblockd_store_staging_bytes={cfg_bytes} holds no "
                            f"{total}-byte slot — bytearray staging keeps serving")
            return False
        try:
            slab = self._c._alloc_pinned(n_slots * total)
            slab_np = slab.numpy()
        except Exception as e:  # noqa: BLE001 — never-raise boundary: pin failure degrades to bytearray staging
            self._store_slab_disabled = True
            self._c._log.maybe("store-slab",
                            "pinned store-slot pool allocation failed — "
                            "bytearray staging keeps serving", e)
            return False
        self._store_slab, self._store_slab_np = slab, slab_np
        self._store_slot_stride = total
        with self._sq_cond:
            self._store_slots_total = n_slots
            self._store_slot_free = list(range(n_slots))
        return True

    def _store_slot_lease(self) -> int | None:
        with self._sq_cond:
            if self._store_slot_free:
                return self._store_slot_free.pop()
        return None

    def _store_gather_fail(self, msg: str, exc: BaseException | None) -> None:
        """Consecutive gather-path failure accounting (mirrors the load side's
        _scratch_fails / _SCRATCH_MAX_FAILS latch). Slot-pool EXHAUSTION never
        lands here — congestion falls back per block without counting."""
        self._store_gather_fails += 1
        if self._store_gather_fails >= _SCRATCH_MAX_FAILS:
            self._store_gather_disabled = True
            self._c._log.maybe(
                "store-gather",
                f"{msg} {self._store_gather_fails}x in a row — latched OFF for "
                "this connector's lifetime (bytearray staging)", exc)
        else:
            self._c._log.maybe("store-gather", f"{msg} — bytearray fallback", exc)

    def _note_store_path(self, path: str) -> None:
        """Store-side twin of _note_path (same WARNING rationale), on its own
        state so the load stamp's one-shot is never consumed by a store."""
        if self._reported_store_path is None:
            self._reported_store_path = path
            logger.warning("kvblockd store path: %s", path)
        elif path != self._reported_store_path and not self._store_path_switch_logged:
            self._store_path_switch_logged = True
            logger.warning("kvblockd store path: %s (switched from %s mid-run)",
                           path, self._reported_store_path)

    def _stage_gather(self, req: KvbReqMeta, names, bytes_per_layer: int,
                      total: int, prefix: bytes, keys, start: int,
                      end: int) -> _StagePlan | None:
        """Batched gather staging: per chunk one index_select per layer into
        the (shared) GPU scratch — the exact inverse of the load's
        index_copy_ — then one async D2H per slot into its pinned body
        (prefix gaps break slab contiguity, so per-slot it is). The 32B
        prefix is CPU-stamped at lease time. Same-stream ordering makes each
        chunk's D2H precede the next chunk's gather overwrite for free; the
        blobs are only provably complete after wait_for_save's device sync,
        which is why every enqueue is deferred into the returned plan.
        A slot-pool dry spell mid-request is congestion, not failure: that
        block takes an inline bytearray (slot None) and keeps its place in
        the request's enqueue order. start is the caller's dedupe-adjusted
        first block (== req.store_start_block with the acked-key LRU off).
        Returns None on failure (counted toward the latch) with every leased
        slot freed — paged memory is still valid, so the caller's bytearray
        loop rebuilds everything."""
        torch = _torch()
        n_layers = len(names)
        dev = self._c._layer_kv[names[0]].device
        try:
            paged_u8 = {}
            for name in names:
                t = self._c._layer_kv[name]
                # view, NEVER reshape: reshape would silently COPY a
                # non-contiguous paged tensor and the gather would read a
                # temporary (the load path's refuter-verified BLOCKER class);
                # view aliases or raises, and the raise lands here.
                paged_u8[name] = t.view(torch.uint8).view(t.shape[0], -1)
            scratch = self._c._scratch_ring(dev, n_layers, bytes_per_layer)[0]
        except Exception as e:  # noqa: BLE001 — never-raise boundary: setup failure degrades to bytearray staging
            self._c._store_gather_fail("gathered-store setup failed", e)
            return None
        items: list[list] = []                 # [j, key, buf, slot_id]
        gathered: list[tuple[int, int]] = []   # (items index, slot id)
        try:
            for j in range(start, end):
                slot = self._c._store_slot_lease()
                if slot is None:
                    items.append([j, keys[j],
                                  self._c._build_block_blob(req.block_ids[j], names,
                                                         bytes_per_layer, prefix,
                                                         total), None])
                    continue
                base = slot * total
                mv = memoryview(self._store_slab_np[base:base + total])
                mv[:BLOB_PREFIX_LEN] = prefix
                items.append([j, keys[j], mv, slot])
                gathered.append((len(items) - 1, slot))
            for c0 in range(0, len(gathered), _SCATTER_CHUNK):
                chunk = gathered[c0:c0 + _SCATTER_CHUNK]
                nblk = len(chunk)
                idx = torch.tensor([req.block_ids[items[i][0]] for i, _ in chunk],
                                   dtype=torch.long, device=dev)
                for li, name in enumerate(names):
                    # No index_select(out=): strided-view out= is
                    # version-fragile; the assignment form is not.
                    scratch[:nblk, li] = paged_u8[name].index_select(0, idx)
                for ci, (_i, slot) in enumerate(chunk):
                    body = self._store_slab[slot * total + BLOB_PREFIX_LEN:
                                            (slot + 1) * total]
                    body.view(n_layers, bytes_per_layer).copy_(
                        scratch[ci], non_blocking=True)
        except Exception as e:  # noqa: BLE001 — never-raise boundary: mid-fill failure frees the leases and degrades
            try:
                # A D2H may still be in flight into these slots: sync before
                # returning them, or a re-lease could race the tail of it.
                self._c._store_sync(dev)
            except Exception:  # noqa: BLE001, S110 — best effort; the slots are being abandoned either way
                pass
            with self._sq_cond:
                for _i, slot in gathered:
                    self._store_slot_free.append(slot)
            self._c._store_gather_fail("gathered-store staging failed", e)
            return None
        return _StagePlan(req_id=req.req_id, total=total, end=end, items=items,
                          dev=dev, names=names, bytes_per_layer=bytes_per_layer,
                          block_ids=list(req.block_ids), prefix=prefix)

    def _store_sync(self, dev) -> bool:
        """ONE event recorded after the last issued D2H, synchronized — the
        gathered blobs are complete-or-rebuilt before any enqueue and before
        wait_for_save returns (the paged buffer is only stable until then).
        No-op off-CUDA (the CPU test seam). Returns False on failure — the
        caller must treat every gathered blob as torn."""
        if getattr(dev, "type", "") != "cuda":
            return True
        torch = _torch()
        try:
            ev = torch.cuda.Event()
            ev.record(torch.cuda.current_stream(dev))
            ev.synchronize()
            return True
        except Exception as e:  # noqa: BLE001 — never-raise boundary: an unfinished stream means nothing is provably staged
            self._c._log.maybe("store-gather", "gathered-store device sync failed", e)
            return False

    def _rebuild_plan(self, plan: _StagePlan) -> None:
        """Device sync failed: every slot-backed blob in the plan is possibly
        torn. Paged memory is still valid (wait_for_save has not returned),
        so rebuild each through the bytearray path and free its slot —
        identical bytes, identical accounting, just no overlap won."""
        for it in plan.items:
            j, _key, _buf, slot = it
            if slot is None:
                continue
            it[2] = self._c._build_block_blob(plan.block_ids[j], plan.names,
                                           plan.bytes_per_layer, plan.prefix,
                                           plan.total)
            it[3] = None
            with self._sq_cond:
                self._store_slot_free.append(slot)

    def _finish_stage(self, plan: _StagePlan) -> None:
        """Deferred enqueue of one synced plan, in block order — the same
        tail-skip contract as the inline loop (the refused block itself is
        counted by _sq_enqueue; the unreachable tail is counted here). Slot
        ownership moves to the queue tuple on enqueue (it[3] nulled), so
        every slot has exactly one owner at every exit. plan.sent advances
        only once _sq_enqueue RETURNS (it never raises — its own contract —
        and settles the item's accounting whether it accepts or refuses), so
        a raise anywhere leaves _abandon_plan counting exactly the items
        never handed over, with their leases still marked in it[3]."""
        for i, it in enumerate(plan.items):
            j, key, buf, slot = it
            ok = self._c._sq_enqueue(key, buf, slot, plan.req_id)
            # Settled: on True the queue tuple owns the lease; on False the
            # refusal branch below frees it. Either way the item has left
            # the abandonable window.
            it[3] = None
            plan.sent = i + 1
            if ok:
                continue
            with self._sq_cond:
                if slot is not None:  # refused: the lease never reached the queue
                    self._store_slot_free.append(slot)
                for tail in plan.items[i + 1:]:
                    if tail[3] is not None:
                        self._store_slot_free.append(tail[3])
                        tail[3] = None
            self._store_holes[plan.req_id] = j  # j < any prior hole (end is capped)
            skipped = plan.end - j - 1
            if skipped > 0:
                with self._sq_cond:
                    self.dropped_puts += skipped
                    self.dropped_put_bytes += skipped * plan.total
            plan.sent = len(plan.items)  # tail fully counted here — not abandonable
            return

    def _abandon_plan(self, plan: _StagePlan) -> None:
        """A raise abandoned this plan before every item reached _sq_enqueue:
        the unsent items are lost HERE, so they are counted HERE — a vanished
        block with dropped_puts=0 breaks the disclosure contract the bench
        rigs grep — the prefix hole is recorded so later steps of the request
        tail-skip past it, and the leases come home only AFTER a best-effort
        device sync (a D2H may still be in flight into these slots; same
        rationale as _stage_gather's own mid-fill failure path). Idempotent:
        a second call finds sent == len(items) and counts nothing."""
        lost = plan.items[plan.sent:]
        plan.sent = len(plan.items)
        if not lost:
            return
        try:
            self._c._store_sync(plan.dev)
        except Exception:  # noqa: BLE001, S110 — best effort; the slots are being abandoned either way
            pass
        with self._sq_cond:
            for it in lost:
                if it[3] is not None:
                    self._store_slot_free.append(it[3])
                    it[3] = None
            self.dropped_puts += len(lost)
            self.dropped_put_bytes += len(lost) * plan.total
        self._store_holes[plan.req_id] = lost[0][0]  # j < any prior hole (end is capped)

    def _sq_enqueue(self, key: bytes, buf: bytearray | memoryview,
                    slot_id: int | None = None, rid: str = "") -> bool:
        """Enqueue one owned blob (slot_id set when buf is a pinned store
        slot: the queue tuple then owns the lease). rid routes the blob to a
        drain worker (stable hash), so ONE request's blocks always share one
        worker's FIFO — per-request delivery order is preserved under any
        worker count. NEVER blocks and NEVER raises: past the byte budget
        (shared across all workers' queues, or during shutdown) the block is
        dropped and counted — the CALLER frees a refused slot — a lost store
        is a future miss, an engine stall is an incident."""
        n = len(buf)
        wi = zlib.crc32(rid.encode()) % self._store_workers if rid else 0
        with self._sq_cond:
            if self._store_stop or self._sq_bytes + n > self._c._cfg.store_queue_bytes:
                self.dropped_puts += 1
                self.dropped_put_bytes += n
                dropped, failed, dbytes = (self.dropped_puts, self.failed_puts,
                                           self.dropped_put_bytes)
            else:
                self._sqs[wi].append((key, buf, slot_id))
                self._sq_bytes += n
                self._c.stats.note_hwm(self._sq_bytes)
                self._sq_cond.notify_all()
                dropped = None
        if dropped is not None:
            # Rate-limited in-run disclosure (the shutdown summary line is
            # unconditional); the bench populate phase greps `dropped=`.
            # OUTSIDE the lock: _log.maybe formats and may hit a logging
            # handler — never hold _sq_cond across foreign code.
            self._c._log.maybe(
                "store-drop",
                f"kvblockd store queue overflow: dropped={dropped} "
                f"failed={failed} dropped_bytes={dbytes}",
            )
            return False
        try:
            self._c._store_thread_start()
        except Exception as e:  # noqa: BLE001 — the never-raises contract must survive thread exhaustion: the blob IS queued, a later enqueue restarts the drain and shutdown counts any remainder
            self._c._log.maybe("store", "kvb-store drain thread start failed", e)
        return True

    def _store_thread_start(self) -> None:
        """Lazily start the "kvb-store" drain worker(s). Only ever called
        from the engine's serving thread (wait_for_save), so the check-then-
        start needs no extra lock. Worker 0 keeps the bare "kvb-store" name
        (tooling greps it); a dead worker (never expected) is restarted."""
        for i in range(self._store_workers):
            t = self._store_threads[i]
            if t is not None and t.is_alive():
                continue
            t = threading.Thread(target=self._store_drain, args=(i,),
                                 name="kvb-store" if i == 0 else f"kvb-store-{i}",
                                 daemon=True)
            self._store_threads[i] = t
            t.start()

    def _store_drain(self, wi: int = 0) -> None:
        """FIFO drain loop over THIS worker's deque (requests are hashed to a
        worker at enqueue, so per-request block order is exactly the single-
        FIFO order): pop one staged blob, put it. A CONNECTION-class
        failure (daemon gone, breaker window) re-queues the item ONCE at the
        head of the SAME deque and waits out the redial backoff before
        retrying — a blip costs one backoff window, never the whole backlog
        burned at one doomed put per item. A SECOND consecutive failure of
        the same item counts it failed (failed_puts) and moves on (no
        infinite loop) — but ONLY real attempts count: _DialPending (another
        caller owns the in-flight dial) PARKS the item (requeue, no strike,
        no attempt telemetry) until the dial resolves, so failed_puts is
        never inflated with a non-attempt. Non-connection failures count
        immediately. Either way
        the client is dropped with the same breaker discipline as loads and
        the thread NEVER dies to an op error, so delivery resumes after a
        redial. OK_EXISTS = dedup, fine. All counters/gauges stay under
        _sq_cond, shared across workers — the disclosure arithmetic is
        aggregate and worker-count-independent."""
        q = self._sqs[wi]
        retry_of: bytearray | memoryview | None = None  # the buf that already failed once
        while True:
            with self._sq_cond:
                while not q and not self._store_stop:
                    self._sq_cond.wait()
                # Load-priority gate: a load is actively pulling KV, so park
                # (bounded by the ceiling) before spending wire/GIL on a
                # store. The ceiling is armed ONCE per raised episode — N
                # queued blobs under one wedged gate cost one ceiling total,
                # never N — and episodes are edge-triggered: start_load_kv
                # clears the arm on the 0->1 load transition (this thread
                # only samples the counter between puts, so it can miss an
                # entire gate-down/gate-up flip inside one put). Shutdown
                # flags cut the park short (they are in the predicate and
                # notify_all'd) — a flush is never held hostage to the gate.
                while (self._loads_inflight > 0 and not self._store_stop
                       and not self._store_abort):
                    if self._drain_park_until is None:
                        # Armed INSIDE the loop: an episode edge can clear
                        # the arm mid-wait, and that new episode is owed its
                        # own fresh ceiling.
                        self._drain_park_until = time.monotonic() + _cm._DRAIN_GATE_CEILING_S
                    left = self._drain_park_until - time.monotonic()
                    if left <= 0:
                        break
                    self._sq_cond.wait(left)
                if self._loads_inflight == 0:
                    self._drain_park_until = None  # episode over cleanly
                if self._store_abort or (not q and self._store_stop):
                    return
                key, buf, slot_id = q.popleft()
                n = len(buf)
                self._sq_bytes -= n
                self._sq_inflight += 1
                self._sq_inflight_bytes += n
            err: BaseException | None = None
            pt0 = time.monotonic()
            try:
                self._c._ensure().put(key, [buf])
            except Exception as e:  # noqa: BLE001 — never let the drain thread die: a lost store is a future miss
                err = e
            # A dial owned by another caller is NOT an attempt: no strike, no
            # attempt telemetry — PARK (requeue unchanged) until it resolves.
            park = isinstance(err, _DialPending)
            if not park:
                # Latency telemetry counts ATTEMPTS (failures included): a
                # store path that is slow because it is failing must read slow.
                self._c.stats.bump("store_count")
                self._c.stats.bump("store_time_s", time.monotonic() - pt0)
            requeue = (err is not None
                       and isinstance(err, (ConnectionLost, OSError))
                       and (park or buf is not retry_of))
            with self._sq_cond:
                self._sq_inflight -= 1
                self._sq_inflight_bytes -= n
                # Reconcile with a shutdown that already disclosed this
                # in-flight block as dropped (join timed out mid-put).
                counted_dropped = self._sq_inflight_counted > 0
                if counted_dropped:
                    self._sq_inflight_counted -= 1
                requeued = False
                if err is None:
                    retry_of = None
                    if counted_dropped:  # delivered after all: un-count it
                        self.dropped_puts -= 1
                        self.dropped_put_bytes -= n
                elif requeue and not self._store_abort:
                    q.appendleft((key, buf, slot_id))  # keeps the lease AND the order
                    self._sq_bytes += n
                    if not park:  # a park is not a strike: leave retry_of as-is
                        retry_of = buf
                    requeued = True
                elif not counted_dropped:  # dropped-at-shutdown is not ALSO failed
                    retry_of = None
                    self.failed_puts += 1
                # THE inflight free point (delivered or terminal): the put has
                # returned, so the memoryview is out of _send_frame's iovec.
                if slot_id is not None and not requeued:
                    self._store_slot_free.append(slot_id)
                self._sq_cond.notify_all()
            if err is None:
                # Ack-populated dedupe: put() returned OK or OK_EXISTS (any
                # other status raises), so the daemon holds this key NOW —
                # the only evidence the LRU ever accepts. Outside _sq_cond
                # (the LRU has its own lock).
                if self._acked_keys is not None:
                    self._acked_keys.add(key)
                continue
            if park:
                if not requeued:
                    continue  # shutdown raced the park: already disclosed
                # Hold until the in-flight dial resolves: sit out the breaker
                # window if one is armed, else short slices — each re-check
                # is lock-only (_ensure raises _DialPending again while the
                # dial is still owned), and the dial itself is bounded by
                # connect_timeout, so the park always terminates.
                wake = max(self._c._next_dial, time.monotonic() + _DIAL_PENDING_PARK_S)
                with self._sq_cond:
                    while not self._store_stop and not self._store_abort:
                        left = wake - time.monotonic()
                        if left <= 0:
                            break
                        self._sq_cond.wait(left)
                continue
            self._c._log.maybe("store", "kvblockd async store failed", err)
            self._c._drop_client(err)
            if requeue:
                # Sit out exactly the dial breaker's window (armed by the
                # _drop_client above, or by another thread earlier) instead of
                # a full extra backoff on top of it — the post-blip backlog
                # hold is the breaker window, no longer. The deadline loop
                # ignores enqueue wakeups; shutdown's stop/abort flags cut it
                # short so a flush is never held hostage.
                wake = self._c._next_dial
                if wake <= time.monotonic():
                    wake = time.monotonic() + _cm._REDIAL_BACKOFF_S
                with self._sq_cond:
                    while not self._store_stop and not self._store_abort:
                        left = wake - time.monotonic()
                        if left <= 0:
                            break
                        self._sq_cond.wait(left)

    def _store_flush(self, timeout: float) -> int:
        """Wait (up to timeout) until every worker's queue is empty and
        nothing is in flight. Returns the UNDELIVERED block count at timeout."""
        deadline = time.monotonic() + timeout
        with self._sq_cond:
            while any(self._sqs) or self._sq_inflight:
                left = deadline - time.monotonic()
                if left <= 0:
                    return sum(len(q) for q in self._sqs) + self._sq_inflight
                self._sq_cond.wait(left)
        return 0

    def _store_shutdown(self) -> None:
        """Flag + notify + bounded join; whatever the drain thread could not
        deliver inside kvblockd_store_flush_timeout_s is counted dropped. The
        summary line is ALWAYS emitted, zero included, at WARNING — this
        logger is unconfigured in vLLM's engine-core process, where the root
        default drops INFO (proven on the bench rig) — so the bench can grep
        for the line's presence, not just its value."""
        with self._sq_cond:
            self._store_stop = True
            self._sq_cond.notify_all()
        # ONE shared flush budget across every worker's join — N workers must
        # not multiply the shutdown ceiling.
        join_by = time.monotonic() + self._c._cfg.store_flush_timeout_s
        for t in self._store_threads:
            if t is not None and t.is_alive():
                t.join(max(0.0, join_by - time.monotonic()))
        with self._sq_cond:
            self._store_abort = True  # a wedged put must not keep delivering
            remainder = sum(len(q) for q in self._sqs) + self._sq_inflight
            if remainder:
                self.dropped_puts += remainder
                self.dropped_put_bytes += self._sq_bytes + self._sq_inflight_bytes
                # An in-flight put is UNDELIVERED at disclosure time, so it
                # counts dropped — but the put may still return: remember how
                # many were counted so the drain thread can reconcile
                # (delivered -> un-count; failed -> not ALSO failed_puts).
                self._sq_inflight_counted = self._sq_inflight
            # QUEUED slots come home before the clear (their puts never
            # started, so no iovec can reference them); an INFLIGHT slot is
            # NEVER freed here — only the drain's reconcile block may, once
            # the put has returned (else a wedged sendmsg still reads it).
            for q in self._sqs:
                for _k, _b, sid in q:
                    if sid is not None:
                        self._store_slot_free.append(sid)
                q.clear()
            self._sq_bytes = 0
            self._sq_cond.notify_all()
            dropped, failed, dbytes, deduped = (self.dropped_puts, self.failed_puts,
                                                self.dropped_put_bytes,
                                                self.deduped_puts)
        # deduped rides at the END so the bench's existing dropped=/failed=
        # greps keep matching the unchanged head of the line.
        logger.warning("kvblockd store queue: dropped=%d failed=%d dropped_bytes=%d"
                       " deduped=%d", dropped, failed, dbytes, deduped)

    def _store_one(self, req: KvbReqMeta) -> None:
        names, dtype_name, bytes_per_layer = self._c._layout()
        if not names:
            return
        total = BLOB_PREFIX_LEN + bytes_per_layer * len(names)
        prefix = encode_blob_prefix(dtype_name, len(names), self._c._block_size,
                                    bytes_per_layer, total)
        seed = self._c._seed(req.cache_salt, req.mm_ids, req.lora_name)
        keys = self._c._chain_keys(req.req_id, seed, req.token_ids)
        client = self._c._ensure()
        end = min(req.store_end_block, len(keys), len(req.block_ids))
        for j in range(req.store_start_block, end):
            bid = req.block_ids[j]
            bufs = [prefix]
            bufs.extend(self._c._block_bytes(self._c._layer_kv[name][bid]) for name in names)
            client.put(keys[j], bufs)  # OK_EXISTS = idempotent dedup (write-once)
