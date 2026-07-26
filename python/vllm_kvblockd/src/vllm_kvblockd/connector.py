"""KvblockdConnector — vLLM-native KVConnectorBase_V1 backed by kvblockd.

CHURN-WATCH: KVConnectorBase_V1 is explicitly unstable (vLLM RFC #38260
tracking); pinned per-minor, CI matrix vs the last 4 releases.
Re-verify the base.py diff before every vLLM bump.

Version assumptions (verified against vendored sources, see UPSTREAM.lock):
  - Written against vLLM v0.25.0 (tag 702f4814fe54fabff350d43cb753ae3e47c0c276),
    modeled on its ExampleConnector (ex-SharedStorageConnector).
  - Constructor is the 3-arg form (vllm_config, role, kv_cache_config) —
    identical across the tested window v0.22.1..v0.25.x, and MANDATORY at v0.25
    (the factory rejects 2-arg external connectors).
  - v0.25's factory refuses connectors without SupportsHMA unless
    --disable-hybrid-kv-cache-manager is set. v0.1 does not implement
    SupportsHMA -> serve with that flag (bench/e2e/vllm-native-cpu.sh does).
  - Target backend: vLLM CPU backend (the no-GPU e2e gate). The load/save
    paths stage through CPU tensors, so a CUDA engine works in principle via
    an extra host copy, but GPU serving is validated by tier_manager.py's
    OffloadingConnector altitude, not this file (see DEFER.md).

Design (the locked mapping): ONE contiguous vLLM block (block_size tokens x
all layers) = ONE kvblockd blob. Keys are a BLAKE3 chain over raw token ids,
seeded by (config fingerprint, cache_salt, LoRA adapter name, mm identifiers)
— see config.py. The chain gives the prefix property BATCH_EXISTS's
consecutive-prefix count was built for, and folding cache_salt into the seed
makes per-request isolation structural.

NEVER RAISE on the serving path: every failure degrades to a cache miss
(LMCache #2204 posture). The only boot-time exception is DeterminismError —
refusing to start beats a fleet that silently never shares cache.

FAIL-OPEN GUARANTEE: a dead, hung, or slow daemon costs the engine at most
one bounded delay per breaker window — connect_timeout on a dial, op_timeout
per recv, kvblockd_load_deadline_s across a whole load (a trickling daemon
passes every per-recv check forever; the deadline abandons the remaining
shards, flags the unfilled block ids, and degrades to recompute) — and then
the dial breaker answers everything instantly for _REDIAL_BACKOFF_S. The
engine never fails and never waits unboundedly; chaos-tested in
tests/test_chaos.py (kill -9 / SIGSTOP mid-run, daemon down at boot).

KNOWN COST, by design: a slow-but-ACCEPTING daemon never errors, so it never
arms the dial breaker — EVERY load against it pays up to the full
kvblockd_load_deadline_s before degrading to recompute, load after load,
until an operator intervenes. The guarantee is bounded-per-load, not
bounded-per-outage: distinguishing "slow store, still worth asking" from
"slow store, stop asking" needs a latency-based breaker policy this wave
deliberately does not have.
"""

from __future__ import annotations

import logging
import os
import queue
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from kvblockd import protocol as kp
from kvblockd.client import Client
from kvblockd.errors import ConnectionLost

from .config import (
    BLOB_VERSION,
    AdapterConfig,
    block_chain_keys,
    chain_seed,
    require_pinned_hashseed,
)

logger = logging.getLogger("vllm_kvblockd")

try:  # vLLM absent (unit tests, CI cells without vllm) -> importable fallback.
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # type: ignore
        KVConnectorBase_V1 as _Base,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # type: ignore
        KVConnectorMetadata as _MetaBase,
    )

    _HAS_VLLM = True
except Exception:  # noqa: BLE001 — availability fallback: a broken vllm install must not break import  # pragma: no cover - exercised by the no-vllm test env
    _HAS_VLLM = False

    class _MetaBase:  # type: ignore[no-redef]
        pass

    class _Base:  # type: ignore[no-redef]
        """Shape-compatible stand-in: metadata plumbing the worker-side tests
        drive, with none of vLLM's config validation."""

        def __init__(self, *args, **kwargs):
            self._connector_metadata = None

        def bind_connector_metadata(self, connector_metadata) -> None:
            self._connector_metadata = connector_metadata

        def clear_connector_metadata(self) -> None:
            self._connector_metadata = None

        def _get_connector_metadata(self):
            assert self._connector_metadata is not None
            return self._connector_metadata

        def get_finished(self, finished_req_ids):
            return None, None


def _torch():  # lazy so the CI import check can load us without torch
    import torch

    return torch


# --- 32B per-blob layout prefix (mirrors lmcache_kvblockd.meta's style) ---
# The prefix is drift armor: a blob whose declared layout does not match the
# LIVE engine's layout is treated as a miss, never scattered into the paged
# buffer. Config changes already diverge the fingerprint (hence the key); this
# catches the residue (e.g. an attention-backend layout flip within a config).
BLOB_MAGIC = b"KVN1"
# BLOB_VERSION lives in config.py (folded into the fingerprint there) and is
# re-exported here for the codec's callers/tests.
BLOB_PREFIX_LEN = 32
_BLOB = struct.Struct("<4sBBHHII14x")  # magic ver dtype n_layers tokens bytes/layer total
assert _BLOB.size == BLOB_PREFIX_LEN

# Pinned dtype codes (same table as lmcache_kvblockd.meta — kept in sync by
# tests/test_connector.py::test_dtype_codes_match_w5).
DTYPE_CODES = {
    "float16": 0, "bfloat16": 1, "float32": 2, "float64": 3,
    "uint8": 4, "int8": 5, "int32": 6, "int64": 7,
    "float8_e4m3fn": 8, "float8_e5m2": 9,
}
CODE_DTYPES = {v: k for k, v in DTYPE_CODES.items()}

# After a failed dial, further dial attempts short-circuit for this long —
# callers degrade to a miss instantly instead of each eating a connect timeout.
_REDIAL_BACKOFF_S = 5.0

# Chunked H2D scatter: blocks per chunk and the GPU scratch depth. One chunk
# = one non_blocking H2D of a contiguous pinned-slab region + one index_copy_
# per layer, replacing chunk×layers synchronous pageable copies. Depth 1 is
# deliberate (reviewer-verified): every chunk runs on the SAME stream, so the
# copy_ into the scratch buffer cannot start until the previous chunk's
# index_copy_ reads finished — a second buffer bought no overlap, only VRAM.
_SCATTER_CHUNK = 64
_SCRATCH_RING = 1

# Consecutive chunked-scatter setup failures (scratch alloc / paged view)
# before the chunked path latches OFF for the connector's lifetime — the
# per-block-from-slab copies keep serving loads either way.
_SCRATCH_MAX_FAILS = 3

# Async-lookup pending-map ceiling: at the cap a NEW lookup answers (0, False)
# — a miss — instead of None, because a None with no queued work would park
# the request on a result that never comes (never-None-under-pressure rule).
_LOOKUP_PENDING_CAP = 1024


class BlobError(ValueError):
    """Unrecognized/incompatible blob prefix — the caller treats it as a miss."""


def encode_blob_prefix(dtype_name: str, n_layers: int, tokens_per_block: int,
                       bytes_per_layer: int, total_len: int) -> bytes:
    if dtype_name not in DTYPE_CODES:
        raise BlobError(f"unsupported dtype {dtype_name!r}")
    return _BLOB.pack(BLOB_MAGIC, BLOB_VERSION, DTYPE_CODES[dtype_name],
                      n_layers, tokens_per_block, bytes_per_layer, total_len)


def decode_blob_prefix(prefix: bytes) -> tuple[str, int, int, int, int]:
    if len(prefix) < BLOB_PREFIX_LEN:
        raise BlobError("prefix too short")
    magic, ver, dcode, n_layers, tpb, bpl, total = _BLOB.unpack(prefix[:BLOB_PREFIX_LEN])
    if magic != BLOB_MAGIC:
        raise BlobError(f"bad magic {magic!r}")
    if ver != BLOB_VERSION:
        raise BlobError(f"unknown blob version {ver}")
    if dcode not in CODE_DTYPES:
        raise BlobError(f"unknown dtype code {dcode}")
    return CODE_DTYPES[dcode], n_layers, tpb, bpl, total


class _RateLimitedLog:
    """One full traceback per key on first sight, then one terse line per
    interval (per-instance — mirrors the LMCache connector's discipline)."""

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
            logger.warning("%s: %s", msg, exc, exc_info=exc)
        else:
            logger.warning("%s: %s", msg, exc)


def align_to_block_size(num_tokens: int, block_size: int) -> int:
    """Largest reusable token count: for an n-token prompt vLLM must compute
    at least the last token itself, so only ((n-1)//B)*B tokens are loadable
    (ExampleConnector's rule, kept bit-identical)."""
    return (num_tokens - 1) // block_size * block_size


@dataclass
class KvbReqMeta:
    """Everything the worker needs to (re)derive keys and move bytes for one
    request. Plain lists/ints/strs only — this crosses the scheduler->worker
    boundary by serialization."""

    req_id: str
    token_ids: list[int]        # aligned prefix only (multiples of block_size)
    cache_salt: str | None      # folded into the chain seed (salt isolation)
    mm_ids: list[str]
    lora_name: str              # "" = base model; folded into the chain seed
    block_ids: list[int]        # physical block ids, KV-cache group 0
    load_start_block: int       # load blocks [load_start_block,
    num_load_blocks: int        #              load_start_block + num_load_blocks)
    store_start_block: int      # store blocks [store_start_block, store_end_block)
    store_end_block: int


@dataclass
class KvblockdConnectorMetadata(_MetaBase):
    requests: list[KvbReqMeta] = field(default_factory=list)


class _LookupResolver:
    """Off-thread BATCH_EXISTS for the flag-gated async lookup. The scheduler
    thread posts PLAIN DATA ONLY — (rid, aligned token ids, cache_salt, mm
    ids, lora name) — NEVER the Request object: a queued Request outlives its
    abort and keeps blocks/tensors pinned (the vLLM #42372 leak class). One
    daemon thread derives the chain keys and asks the daemon; results are
    plain hit-token ints. An op failure posts 0 (a failed lookup is a miss)
    and drops the client with the usual breaker discipline — the thread
    itself never dies to an op error."""

    def __init__(self, connector: KvblockdConnector):
        self._conn = connector
        self._q: queue.SimpleQueue = queue.SimpleQueue()
        self._results: dict[str, int] = {}
        # rids discarded while their work was still queued/on the wire: the
        # late result is swallowed instead of orphaned in _results (nobody is
        # left to pop it — the vLLM #42372 leak class). Consumed by _run on
        # completion; bounded as armor against a dead thread never consuming.
        self._tombstones: set[str] = set()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="kvb-lookup", daemon=True)
        self._thread.start()

    def alive(self) -> bool:
        return self._thread.is_alive()

    def post(self, rid: str, token_ids: list[int], cache_salt: str | None,
             mm_ids: list[str], lora_name: str) -> None:
        self._q.put((rid, token_ids, cache_salt, mm_ids, lora_name))

    def pop(self, rid: str) -> int | None:
        with self._lock:
            return self._results.pop(rid, None)

    def discard(self, rid: str, inflight: bool = False) -> None:
        """Drop rid's result. inflight=True says the caller KNOWS work for
        rid is still queued/on the wire (a pending entry existed with no
        result): tombstone it so the late post is swallowed. A result that
        already landed is simply popped — nothing is in flight then."""
        with self._lock:
            if self._results.pop(rid, None) is not None:
                return
            if inflight:
                if len(self._tombstones) >= _LOOKUP_PENDING_CAP:
                    self._tombstones.pop()  # bounded armor: shed an arbitrary stale entry
                self._tombstones.add(rid)

    def stop(self, timeout: float = 1.0) -> None:
        self._q.put(None)  # sentinel
        self._thread.join(timeout)

    def _run(self) -> None:
        conn = self._conn
        while True:
            item = self._q.get()
            if item is None:
                return
            rid, token_ids, cache_salt, mm_ids, lora_name = item
            hit = 0
            try:
                seed = conn._seed(cache_salt, mm_ids, lora_name)
                keys = block_chain_keys(seed, token_ids, conn._block_size)
                n_consec, _ = conn._ensure().batch_exists(keys)
                hit = min(n_consec * conn._block_size, len(token_ids))
            except Exception as e:  # noqa: BLE001 — a failed lookup is a miss; the resolver lives on
                conn._log.maybe("lookup", "kvblockd async BATCH_EXISTS failed (miss)", e)
                conn._drop_client(e)
            with self._lock:
                if rid in self._tombstones:  # discarded while we were on the wire
                    self._tombstones.discard(rid)
                else:
                    self._results[rid] = hit


class KvblockdConnector(_Base):
    """The KVConnectorBase_V1 socket. Scheduler side answers
    get_num_new_matched_tokens with kvblockd's BATCH_EXISTS (<1ms p99 verb,
    purpose-built for this call — it blocks scheduling); worker side moves
    whole blocks with BATCH_GET / PUT_STREAM->COMMIT."""

    def __init__(self, vllm_config, role, kv_cache_config=None):
        try:
            super().__init__(vllm_config, role, kv_cache_config)
        except TypeError:
            super().__init__()  # object-shaped fallback base
        require_pinned_hashseed()  # DeterminismError names the fix; boot-only raise
        self._cfg = AdapterConfig.from_vllm_config(vllm_config)
        self._block_size = self._cfg.block_size
        self._client: Client | None = None
        self._client_lock = threading.Lock()
        self._next_dial = 0.0  # monotonic gate arming the dial breaker
        self._log = _RateLimitedLog()
        self._closed = False

        # Scheduler-side state.
        self._inflight: dict[str, object] = {}        # req_id -> Request
        # req_id -> LOCAL prefix-cache hit tokens, recorded from the argument
        # of get_num_new_matched_tokens. At v0.25 request.num_computed_tokens
        # is still 0 when update_state_after_alloc runs (the scheduler assigns
        # it afterwards), so this stash is the only source of the local count.
        self._local_hit_tokens: dict[str, int] = {}
        self._need_load_blocks: dict[str, tuple[int, int]] = {}  # req_id -> (start, n)
        self._seeds: dict[str, bytes] = {}            # req_id -> chain seed
        # Async lookup (kvblockd_async_lookup, default OFF): resolver thread
        # created lazily on first use; pending maps req_id -> expiry deadline.
        self._resolver: _LookupResolver | None = None
        self._lookup_pending: dict[str, float] = {}
        # req_id -> full group-0 block list, accumulated across steps: the
        # SchedulerOutput only carries NEW block ids for cached requests, and
        # the Request object exposes none — chunked-prefill continuation
        # stores need the request's whole list.
        self._blocks: dict[str, list[int]] = {}

        # Worker-side state.
        self._layer_kv: dict[str, object] = {}        # layer_name -> paged KV tensor
        self._load_errors: set[int] = set()

        # Pinned staging slab (CUDA loads only): lazily allocated on the first
        # CUDA-device load, grown geometrically UP TO the configured cap
        # (kvblockd_staging_bytes), REUSED across loads — never freed
        # per-load. Loads bigger than the cap drain through the slab in
        # cap-sized passes. Slots are disjoint by pass-local block index,
        # which is what makes the client's concurrent drain threads safe.
        self._staging_bytes = self._cfg.staging_bytes
        self._slab = None                 # 1-D pinned uint8 torch tensor
        self._slab_np = None              # numpy view of the slab (memoryview source)
        self._slab_disabled = False       # first-pin failure -> permanent per-block fallback
        self._prewarm_done = False        # one eager-pin attempt at first CUDA capture
        # GPU scratch for the chunked scatter: _SCRATCH_RING × [chunk,
        # n_layers, bytes_per_layer] uint8, cached per (device, layout).
        self._gpu_scratch = None
        self._gpu_scratch_key = None
        self._scratch_fails = 0           # consecutive chunked-setup failures
        self._chunked_disabled = False    # latched after _SCRATCH_MAX_FAILS in a row
        # Machine-readable path attribution: one INFO line on the first
        # completed load, one more if the path ever switches mid-run.
        self._reported_path = None
        self._path_switch_logged = False
        self._debug_scatter_checked = False  # KVBLOCKD_DEBUG_SCATTER_CHECK=1, once

        # Write-behind store queue (kvblockd_async_store, default on):
        # wait_for_save stages OWNED byte copies here and returns; ONE daemon
        # thread ("kvb-store", lazily started on first enqueue) drains FIFO.
        # FIFO preserves per-request block order, so a partial delivery is a
        # usable consecutive prefix under the prefix-chain keys. All queue
        # state (deque, byte gauges, public counters) is guarded by _sq_cond.
        self._sq: deque[tuple[bytes, bytearray]] = deque()
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
        self._store_thread: threading.Thread | None = None
        self._store_stop = False      # shutdown: drain the remainder, then exit
        self._store_abort = False     # wedged flush: exit without draining
        # Public disclosure counters (the bench reads/greps these).
        self.dropped_puts = 0
        self.dropped_put_bytes = 0
        self.failed_puts = 0

    # ------------------------------------------------------------------
    # client plumbing (lazy: import/instantiate must succeed with no daemon)
    # ------------------------------------------------------------------
    def _ensure(self) -> Client:
        with self._client_lock:
            if self._closed:
                raise ConnectionError("connector closed")
            if self._client is None:
                # Dial breaker: without it, every caller of a dead endpoint
                # eats a full connect timeout (one stalled scheduler step per
                # waiting request under a blackholed daemon).
                now = time.monotonic()
                if now < self._next_dial:
                    raise ConnectionError("kvblockd dial suppressed after recent failure")
                try:
                    self._client = Client(
                        (self._cfg.host, self._cfg.port),
                        namespace=self._cfg.namespace,
                        token=self._cfg.token,
                        streams=self._cfg.streams,
                        connect_timeout=self._cfg.connect_timeout,
                        op_timeout=self._cfg.op_timeout,
                        verify=self._cfg.verify,
                        so_rcvbuf=self._cfg.so_rcvbuf,
                    )
                except Exception:
                    self._next_dial = time.monotonic() + _REDIAL_BACKOFF_S
                    raise
            return self._client

    def _drop_client(self, exc: BaseException) -> None:
        """ConnectionLost/OSError out of a batch op means the pooled
        connections are dead (daemon gone or blackholed): drop the whole
        client and re-arm the dial breaker, so the outage costs ONE
        connect_timeout per backoff window — not one per load, with every
        pooled conn re-dialing under it. A dial-suppressed ConnectionError
        (client already None) re-arms nothing: extending the window on every
        suppressed call would starve the retry forever under constant load."""
        if not isinstance(exc, (ConnectionLost, OSError)):
            return  # e.g. StatusError: the conn is in sync, keep the pool
        with self._client_lock:
            client, self._client = self._client, None
            if client is None:
                return  # dial failure/suppression: _ensure already armed it
            self._next_dial = time.monotonic() + _REDIAL_BACKOFF_S
        client.close()

    def shutdown(self):
        if self._resolver is not None:
            self._resolver.stop(1.0)  # sentinel + bounded join
        # Flush the write-behind queue FIRST — the drain thread needs the
        # client to deliver what is still staged.
        self._store_shutdown()
        with self._client_lock:
            self._closed = True
            client, self._client = self._client, None
        if client is not None:
            client.close()

    # ------------------------------------------------------------------
    # key derivation (shared by both sides — MUST agree byte-for-byte)
    # ------------------------------------------------------------------
    def _seed(self, cache_salt: str | None, mm_ids: list[str], lora_name: str) -> bytes:
        return chain_seed(self._cfg.fingerprint, cache_salt, mm_ids, lora_name)

    def _seed_for_request(self, request) -> bytes:
        rid = getattr(request, "request_id", None)
        if rid is not None and rid in self._seeds:
            return self._seeds[rid]
        seed = self._seed(getattr(request, "cache_salt", None), self._mm_ids(request),
                          self._lora_name(request))
        if rid is not None:
            self._seeds[rid] = seed
        return seed

    @staticmethod
    def _mm_ids(request) -> list[str]:
        try:
            return [f.identifier for f in (getattr(request, "mm_features", None) or [])]
        except Exception:  # noqa: BLE001 — never-raise boundary: an unreadable request shape means "no mm features", not an engine exception
            return []

    @staticmethod
    def _lora_name(request) -> str:
        """KV computed under a LoRA adapter is only valid under that adapter:
        the name is part of the key identity ("" = base model)."""
        return str(getattr(getattr(request, "lora_request", None), "lora_name", "") or "")

    # ------------------------------------------------------------------
    # Scheduler side
    # ------------------------------------------------------------------
    def get_num_new_matched_tokens(self, request, num_computed_tokens: int):
        """Consecutive-prefix hit count x block_size beyond what is computed.
        Idempotent (vLLM may call it repeatedly; the only state written is the
        req_id-keyed local-hit stash, overwritten in place). The second tuple
        element is ALWAYS False — no async LOADS this wave (the upstream
        async-load hang class needs its own validation rig); with
        kvblockd_async_lookup the first element may be None ("still resolving,
        ask again"), never under pressure (see _lookup_async)."""
        rid = getattr(request, "request_id", None)
        if rid is not None:
            # num_computed_tokens here IS the local prefix-cache hit; the
            # Request object still reads 0 at update_state_after_alloc time.
            self._local_hit_tokens[rid] = int(num_computed_tokens or 0)
        if self._cfg.async_lookup and rid is not None:
            return self._lookup_async(rid, request, num_computed_tokens)
        return self._lookup_sync(request, num_computed_tokens)

    def _lookup_sync(self, request, num_computed_tokens: int):
        """The original synchronous lookup (flag off / fallback), unchanged."""
        try:
            token_ids = list(getattr(request, "prompt_token_ids", None) or [])
            aligned = align_to_block_size(len(token_ids), self._block_size)
            if aligned <= num_computed_tokens:
                return 0, False
            seed = self._seed(getattr(request, "cache_salt", None), self._mm_ids(request),
                              self._lora_name(request))
            keys = block_chain_keys(seed, token_ids[:aligned], self._block_size)
            n_consec, _ = self._ensure().batch_exists(keys)
            hit_tokens = min(n_consec * self._block_size, aligned)
            return max(0, hit_tokens - num_computed_tokens), False
        except Exception as e:  # noqa: BLE001 — never raise: a failed lookup is a miss
            self._log.maybe("lookup", "kvblockd BATCH_EXISTS failed (treated as miss)", e)
            self._drop_client(e)
            return 0, False

    def _lookup_async(self, rid: str, request, num_computed_tokens: int):
        """Flag-on lookup: post once, answer (None, False) while the resolver
        works, then (hit, False). Order of checks: result → pending+deadline →
        new. Guard rails: an expired pending entry becomes a miss and is
        pruned (a wedged resolver must not park the request forever); at the
        pending-map cap a NEW request is answered (0, False) — NEVER None,
        which with no queued work would never be followed by a result; a dead
        resolver thread falls back to the inline synchronous lookup."""
        try:
            token_ids = list(getattr(request, "prompt_token_ids", None) or [])
            aligned = align_to_block_size(len(token_ids), self._block_size)
            if aligned <= num_computed_tokens:
                if (self._lookup_pending.pop(rid, None) is not None
                        and self._resolver is not None):
                    # posted work is still in flight: tombstone the late result
                    self._resolver.discard(rid, inflight=True)
                return 0, False
            if self._resolver is None:
                self._resolver = _LookupResolver(self)
            resolver = self._resolver
            hit = resolver.pop(rid)  # a result posted before a thread death still counts
            if hit is not None:
                self._lookup_pending.pop(rid, None)
                return max(0, min(hit, aligned) - num_computed_tokens), False
            if not resolver.alive():
                # Dead resolver (never expected; armor): serve inline, sync.
                self._log.maybe("lookup-thread",
                                "async lookup resolver thread is dead — inline sync fallback")
                self._lookup_pending.pop(rid, None)
                return self._lookup_sync(request, num_computed_tokens)
            deadline = self._lookup_pending.get(rid)
            if deadline is not None:
                if time.monotonic() > deadline:
                    self._lookup_pending.pop(rid, None)
                    # pop() above answered None, so the work is still queued/
                    # on the wire — tombstone it or the late result orphans.
                    resolver.discard(rid, inflight=True)
                    self._log.maybe("lookup-deadline",
                                    f"async lookup timed out — treated as miss req={rid}")
                    return 0, False
                return None, False
            if len(self._lookup_pending) >= _LOOKUP_PENDING_CAP:
                self._log.maybe("lookup-cap",
                                "async lookup pending map at capacity — treated as miss")
                return 0, False
            resolver.post(rid, token_ids[:aligned], getattr(request, "cache_salt", None),
                          self._mm_ids(request), self._lora_name(request))
            self._lookup_pending[rid] = time.monotonic() + self._cfg.lookup_timeout_s
            return None, False
        except Exception as e:  # noqa: BLE001 — never raise: a failed lookup is a miss
            self._log.maybe("lookup", "kvblockd async lookup failed (treated as miss)", e)
            if (self._lookup_pending.pop(rid, None) is not None
                    and self._resolver is not None):
                self._resolver.discard(rid, inflight=True)  # same orphan armor
            return 0, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens: int):
        rid = getattr(request, "request_id", None)
        if rid is None:
            return
        self._inflight[rid] = request
        self._seeds[rid] = self._seed_for_request(request)
        local = self._local_hit_tokens.pop(rid, None)
        if local is None:
            local = int(getattr(request, "num_computed_tokens", 0) or 0)
        if num_external_tokens > 0:
            # A local hit L + an external hit E means logical blocks
            # [L/B, (L+E)/B) must be fetched — NOT [0, E/B): the daemon's
            # consecutive prefix covers [0, (L+E)/B), the local cache already
            # holds [0, L/B), and mapping the external count onto the head
            # would leave the tail counted-computed but unfilled.
            start = local // self._block_size
            end = (local + num_external_tokens) // self._block_size
            self._need_load_blocks[rid] = (start, end - start)

    def build_connector_meta(self, scheduler_output):
        meta = KvblockdConnectorMetadata()
        try:
            self._build_meta_into(meta, scheduler_output)
        except Exception as e:  # noqa: BLE001 — never raise: an empty meta = a no-op step
            self._log.maybe("meta", "build_connector_meta failed (no-op step)", e)
        return meta

    def _build_meta_into(self, meta: KvblockdConnectorMetadata, scheduler_output) -> None:
        num_scheduled = getattr(scheduler_output, "num_scheduled_tokens", {}) or {}

        for new_req in getattr(scheduler_output, "scheduled_new_reqs", []) or []:
            rid = new_req.req_id
            token_ids = list(new_req.prompt_token_ids or [])
            aligned = align_to_block_size(len(token_ids), self._block_size)
            if aligned <= 0:
                self._need_load_blocks.pop(rid, None)
                continue
            load_start, n_load = self._need_load_blocks.pop(rid, (0, 0))
            # Store only blocks FULLY computed after this step — under chunked
            # prefill later chunks arrive via scheduled_cached_reqs below.
            computed = int(getattr(new_req, "num_computed_tokens", 0) or 0)
            done = min(aligned, computed + int(num_scheduled.get(rid, 0)))
            store_end = done // self._block_size
            request = self._inflight.get(rid)
            salt = getattr(request, "cache_salt", None) if request is not None else None
            # Strict .identifier: a repr()-derived id differs per process, so
            # it can never round-trip as a key — failing the step (caught by
            # build_connector_meta -> no-op) beats minting a nondeterministic
            # keyspace. v0.25 mm features always carry .identifier.
            mm_ids = [f.identifier for f in (new_req.mm_features or [])]
            block_ids = list(new_req.block_ids[0])
            self._blocks[rid] = list(block_ids)
            # Blocks below load_start are the local prefix-cache hit and
            # blocks below load_start+n_load were just fetched — when a load
            # happened, the daemon's consecutive prefix proved all of them
            # present, so storing starts after the loaded range.
            store_start = load_start + n_load
            meta.requests.append(
                KvbReqMeta(
                    req_id=rid,
                    token_ids=token_ids[:aligned],
                    cache_salt=salt,
                    mm_ids=mm_ids,
                    lora_name=self._lora_name(request) if request is not None else "",
                    block_ids=block_ids,
                    load_start_block=load_start,
                    num_load_blocks=n_load,
                    store_start_block=store_start,
                    store_end_block=max(store_end, store_start),
                )
            )

        cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
        if cached is not None:
            self._build_cached_meta(meta, cached, num_scheduled)

        # A tracked load that never surfaced in this step's scheduler output
        # (preempted before running) stays queued for the resumed path; if the
        # request is gone entirely, request_finished prunes it.

    def _build_cached_meta(self, meta, cached, num_scheduled) -> None:
        req_ids = list(getattr(cached, "req_ids", []) or [])
        resumed = getattr(cached, "resumed_req_ids", set()) or set()
        computed_list = getattr(cached, "num_computed_tokens", []) or []
        new_block_ids = getattr(cached, "new_block_ids", []) or []
        for i, rid in enumerate(req_ids):
            request = self._inflight.get(rid)
            if request is None:
                continue
            computed = int(computed_list[i]) if i < len(computed_list) else 0
            scheduled = int(num_scheduled.get(rid, 0))
            all_tokens = list(getattr(request, "all_token_ids", None) or [])
            n_prompt = int(getattr(request, "num_prompt_tokens", len(all_tokens)) or 0)
            aligned = align_to_block_size(n_prompt, self._block_size)
            if aligned <= 0:
                continue
            # Keep the accumulated block list current: for a resumed request
            # new_block_ids IS the full list (the preempted blocks were
            # freed); otherwise it appends. Accumulate only from a tracked
            # base — extending an unseen request would misindex every block.
            blocks_i = new_block_ids[i] if i < len(new_block_ids) else None
            if rid in resumed:
                if blocks_i is not None:
                    self._blocks[rid] = list(blocks_i[0])
                else:  # pre-preemption list is stale (those blocks were freed)
                    self._blocks.pop(rid, None)
            elif blocks_i is not None and rid in self._blocks:
                self._blocks[rid].extend(blocks_i[0])
            block_ids = self._blocks.get(rid)
            done = min(aligned, computed + scheduled)
            if rid in resumed and rid in self._need_load_blocks:
                load_start, n_load = self._need_load_blocks.pop(rid)
                if block_ids is None:
                    continue
                # Blocks computed during the resume step itself still need a
                # store; everything at/below the loaded range is present.
                store_start = max(load_start + n_load, computed // self._block_size)
                meta.requests.append(
                    KvbReqMeta(
                        req_id=rid,
                        token_ids=all_tokens[:aligned],
                        cache_salt=getattr(request, "cache_salt", None),
                        mm_ids=self._mm_ids(request),
                        lora_name=self._lora_name(request),
                        block_ids=list(block_ids),
                        load_start_block=load_start,
                        num_load_blocks=n_load,
                        store_start_block=store_start,
                        store_end_block=max(done // self._block_size, store_start),
                    )
                )
                continue
            # Chunked-prefill continuation: store the blocks this step completes.
            if computed >= aligned:
                continue  # prompt fully covered (decode steps store nothing)
            store_start = computed // self._block_size
            store_end = done // self._block_size
            if store_end <= store_start:
                continue
            # A store row needs physical ids up to store_end; a shorter list
            # means tracking gapped (e.g. connector restarted mid-request) —
            # skip rather than guess (a smaller cache, never a wrong byte).
            if not block_ids or len(block_ids) < store_end:
                continue
            meta.requests.append(
                KvbReqMeta(
                    req_id=rid,
                    token_ids=all_tokens[:aligned],
                    cache_salt=getattr(request, "cache_salt", None),
                    mm_ids=self._mm_ids(request),
                    lora_name=self._lora_name(request),
                    block_ids=list(block_ids),
                    load_start_block=0,
                    num_load_blocks=0,
                    store_start_block=store_start,
                    store_end_block=store_end,
                )
            )

    def request_finished(self, request, block_ids):
        rid = getattr(request, "request_id", None)
        if rid is not None:
            self._inflight.pop(rid, None)
            self._need_load_blocks.pop(rid, None)
            self._local_hit_tokens.pop(rid, None)
            self._seeds.pop(rid, None)
            self._blocks.pop(rid, None)
            self._store_holes.pop(rid, None)
            # Async-lookup abort cleanup (the #42372 leak class): an aborted
            # request must not leave a pending deadline, an unclaimed result,
            # OR a result still on the wire — a pending entry with no result
            # means the work is in flight, so the discard tombstones it.
            pending = self._lookup_pending.pop(rid, None) is not None
            if self._resolver is not None:
                self._resolver.discard(rid, inflight=pending)
        return False, None

    # ------------------------------------------------------------------
    # Worker side
    # ------------------------------------------------------------------
    def _capture_layers(self, forward_context) -> None:
        """Refresh layer_name -> paged KV tensor from the forward context
        (ExampleConnector's access pattern; kv_cache may be a per-virtual-
        engine list on some releases)."""
        try:
            layers = getattr(forward_context, "no_compile_layers", None) or {}
            ve = int(getattr(forward_context, "virtual_engine", 0) or 0)
            for name, layer in layers.items():
                kv = getattr(layer, "kv_cache", None)
                if kv is None:
                    continue
                if isinstance(kv, (list, tuple)):
                    kv = kv[ve]
                self._layer_kv[name] = kv
        except Exception as e:  # noqa: BLE001 — never-raise boundary: failed capture degrades to load/store no-ops, never into the engine
            self._log.maybe("layers", "capturing paged KV tensors failed", e)
        self._maybe_prewarm()

    def _maybe_prewarm(self) -> None:
        """One eager pinned-slab allocation at the FIRST CUDA layout capture
        (not the first load): cudaHostAlloc of a multi-GiB slab takes hundreds
        of ms, and paying it lazily buries the stall inside the first measured
        load. Sized min(staging cap, kvblockd_prewarm_bytes). Failure falls
        back to the EXISTING lazy path untouched (_slab_disabled stays False —
        _slab_reserve owns that latch), never fatal. The measured duration is
        published at WARNING (INFO is dropped in engine-core) so the bench can
        attribute the one-time cost."""
        if self._prewarm_done or self._slab is not None or self._slab_disabled:
            return
        try:
            names = sorted(self._layer_kv)
            if not names:
                return
            dev = self._layer_kv[names[0]].device
            if not self._slab_path_ok(dev):
                # CPU backend: the slab never pays here — and the paged
                # tensors' device never changes mid-run, so LATCH instead of
                # re-walking the layer map on every single capture.
                self._prewarm_done = True
                return
            self._prewarm_done = True  # one attempt per connector lifetime
            want = min(self._staging_bytes, self._cfg.prewarm_bytes)
            if want <= 0:
                return
            t0 = time.monotonic()
            slab = self._alloc_pinned(want)
            slab_np = slab.numpy()
            self._slab, self._slab_np = slab, slab_np
            logger.warning("kvblockd pinned prewarm: %d bytes in %.0f ms",
                           want, (time.monotonic() - t0) * 1e3)
        except Exception as e:  # noqa: BLE001 — never-raise boundary: a failed prewarm keeps the lazy slab path
            self._log.maybe("slab", "pinned prewarm failed — lazy slab path keeps serving", e)

    def _layout(self):
        """(sorted layer names, dtype_name, bytes_per_layer_block) of the LIVE
        engine — the oracle every stored/loaded blob must match. All layers
        must agree: with heterogeneous per-layer pages (hybrid/sliding-window
        models) one layer's size would lie for all, producing self-mismatched
        blobs that can never load — refuse the layout instead, so the degrade
        is a visible no-op, not a permanent silent miss."""
        names = sorted(self._layer_kv)
        if not names:
            return [], "", 0
        t0 = self._layer_kv[names[0]]
        dtype_name = str(t0.dtype).removeprefix("torch.")
        block0 = t0[0]
        bytes_per_layer = block0.numel() * block0.element_size()
        for name in names[1:]:
            t = self._layer_kv[name]
            b = t[0]
            if b.numel() * b.element_size() != bytes_per_layer or t.dtype != t0.dtype:
                self._log.maybe(
                    "layout",
                    f"per-layer KV pages are not uniform ({name} vs {names[0]}) — "
                    "connector cannot map blocks to blobs; disabled for this step",
                )
                return [], "", 0
        return names, dtype_name, bytes_per_layer

    def _block_bytes(self, kv_block):
        """Contiguous uint8 numpy view of one paged block (zero-copy on CPU;
        staged host copy off-CPU — the CPU backend is this connector's lane)."""
        torch = _torch()
        t = kv_block
        if t.device.type != "cpu":
            t = t.to("cpu")
        if not t.is_contiguous():
            t = t.contiguous()
        return t.view(torch.uint8).numpy().reshape(-1)

    @staticmethod
    def _load_range_ids(req: KvbReqMeta) -> list[int]:
        """Physical ids of the PROMISED load range — the exact set that must
        be flagged when the load cannot happen (the scheduler counted these
        blocks computed; anything unfilled and unflagged is silent garbage)."""
        return req.block_ids[req.load_start_block : req.load_start_block + req.num_load_blocks]

    def start_load_kv(self, forward_context, **kwargs) -> None:
        self._capture_layers(forward_context)
        try:
            metadata = self._get_connector_metadata()
        except Exception:  # noqa: BLE001 — never-raise boundary: missing metadata = nothing to load this step
            return
        requests = getattr(metadata, "requests", None) or []
        for req in requests:
            if req.num_load_blocks > 0:
                try:
                    self._load_one(req)
                except Exception as e:  # noqa: BLE001 — never raise; blocks flagged as errors
                    self._log.maybe("load", f"kvblockd load failed req={req.req_id}", e)
                    self._drop_client(e)
                    self._load_errors.update(self._load_range_ids(req))

    def _load_one(self, req: KvbReqMeta) -> None:
        names, dtype_name, bytes_per_layer = self._layout()
        if not names:
            self._load_errors.update(self._load_range_ids(req))
            return
        total = BLOB_PREFIX_LEN + bytes_per_layer * len(names)
        seed = self._seed(req.cache_salt, req.mm_ids, req.lora_name)
        start = req.load_start_block
        end = start + req.num_load_blocks
        keys = block_chain_keys(seed, req.token_ids, self._block_size)[start:end]
        # A promised block with no derivable key (token list shorter than the
        # promise) can never be filled — flag it now, don't drop it silently.
        for blk in range(start + len(keys), end):
            if blk < len(req.block_ids):
                self._load_errors.add(req.block_ids[blk])

        body_len = total - BLOB_PREFIX_LEN
        dev = self._layer_kv[names[0]].device
        # Overall per-load deadline (kvblockd_load_deadline_s): op_timeout
        # bounds each recv, this bounds the WHOLE load — a daemon trickling
        # bytes passes every per-recv check forever, and the engine counted
        # these blocks computed, so the only safe degrade is: abandon the
        # remaining shards, flag the unfilled bids, recompute.
        deadline = (time.monotonic() + self._cfg.load_deadline_s
                    if self._cfg.load_deadline_s > 0 else None)
        used_ring = False
        took_slab = False
        if keys and self._slab_path_ok(dev):
            # Cap-sized passes: the slab never grows past the configured cap;
            # a load bigger than the cap drains through it pass by pass.
            cap_blocks = self._staging_bytes // body_len if body_len > 0 else 0
            pass_blocks = min(len(keys), cap_blocks)
            if pass_blocks > 0 and self._slab_reserve(pass_blocks * body_len):
                took_slab = True
                used_ring = self._load_slab(req, names, dtype_name, bytes_per_layer,
                                            total, keys, pass_blocks, deadline)
        if keys and not took_slab:
            self._load_perblock(req, names, dtype_name, bytes_per_layer, total, keys,
                                deadline)
        if keys:
            self._note_path("chunked-slab" if used_ring else "per-block")

    def _note_path(self, path: str) -> None:
        """One machine-readable line on the first completed load — the bench
        rig greps it to attribute measured numbers to the path that produced
        them — plus one more line if the path ever switches mid-run.

        WARNING level on purpose: this logger lives outside vLLM's logging
        config, and in the engine-core process an unconfigured logger drops
        INFO under the root default — the certification run recorded 'path
        unattributed' exactly that way. WARNING passes the default filter;
        one line per process lifetime is not noise."""
        if self._reported_path is None:
            self._reported_path = path
            logger.warning("kvblockd load path: %s", path)
        elif path != self._reported_path and not self._path_switch_logged:
            self._path_switch_logged = True
            logger.warning("kvblockd load path: %s (switched from %s mid-run)",
                           path, self._reported_path)

    def _load_perblock(self, req: KvbReqMeta, names, dtype_name, bytes_per_layer,
                       total, keys, deadline: float | None = None) -> None:
        """The original per-block load: pageable torch.empty staging + one
        synchronous dst.copy_ per (block, layer). This is THE path for CPU
        paged tensors (the CI backend / bench/e2e/cpu rig depend on it staying
        byte-identical in behavior) and the degrade when pinning fails."""
        torch = _torch()
        start = req.load_start_block
        staged: dict[int, object] = {}

        def alloc(idx, prefix, body_len):
            try:
                d, n_layers, tpb, bpl, tot = decode_blob_prefix(prefix)
            except BlobError:
                return None
            if (d != dtype_name or n_layers != len(names) or tpb != self._block_size
                    or bpl != bytes_per_layer or tot != total
                    or body_len != total - BLOB_PREFIX_LEN):
                return None  # layout drift -> miss, never a corrupt scatter
            buf = torch.empty(body_len, dtype=torch.uint8)
            staged[idx] = buf
            return memoryview(buf.numpy())

        statuses = self._ensure().batch_get_scatter(keys, BLOB_PREFIX_LEN, alloc,
                                                    deadline=deadline)
        for j, st in enumerate(statuses):
            blk = start + j
            bid = req.block_ids[blk] if blk < len(req.block_ids) else None
            if st != kp.Status.OK or j not in staged:
                if bid is not None:
                    self._load_errors.add(bid)
                continue
            if bid is None:
                continue
            buf = staged[j]
            for li, name in enumerate(names):
                dst = self._layer_kv[name][bid]
                src = buf[li * bytes_per_layer : (li + 1) * bytes_per_layer]
                try:
                    dst.copy_(src.view(dst.dtype).reshape(dst.shape))
                except Exception as e:  # noqa: BLE001 — never-raise boundary: a failed scatter marks the block errored, not the engine
                    self._log.maybe("scatter", f"scatter into {name} failed", e)
                    self._load_errors.add(bid)
                    break

    # ------------------------------------------------------------------
    # pinned-slab load path (CUDA paged tensors)
    # ------------------------------------------------------------------
    def _slab_path_ok(self, device) -> bool:
        """Slab staging + chunked H2D only pays (and only pins) for CUDA paged
        tensors; the CPU backend keeps the original per-block path unchanged."""
        return getattr(device, "type", "") == "cuda"

    def _alloc_pinned(self, nbytes: int):
        """cudaHostAlloc via torch — a seam the tests mock (pin failure / CPU CI)."""
        torch = _torch()
        return torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)

    def _slab_reserve(self, nbytes: int) -> bool:
        """Ensure the connector-owned pinned uint8 slab holds nbytes: lazily
        allocated on first use, grown geometrically (>= 2x) up to the
        configured cap, REUSED across loads — never freed per-load. A pin
        failure with NO working slab disables the slab for the connector's
        lifetime; a GROWTH failure keeps the old slab (this load falls back
        per-block, smaller loads keep the slab lane). Never raises."""
        if self._slab_disabled:
            return False
        if nbytes > self._staging_bytes:
            return False  # over the cap: callers split loads into cap-sized passes
        if self._slab is not None and self._slab.numel() >= nbytes:
            return True
        want = nbytes if self._slab is None else max(nbytes, 2 * self._slab.numel())
        want = min(want, self._staging_bytes)  # still >= nbytes (checked above)
        try:
            slab = self._alloc_pinned(want)
            slab_np = slab.numpy()
        except Exception as e:  # noqa: BLE001 — never-raise boundary: cudaHostAlloc failure degrades to the per-block path
            if self._slab is None:
                self._log.maybe("slab", "pinned slab allocation failed — per-block fallback", e)
                self._slab_disabled = True
            else:
                self._log.maybe("slab", "pinned slab GROWTH failed — keeping the "
                                        "existing slab; per-block fallback for this load", e)
            return False
        self._slab, self._slab_np = slab, slab_np
        return True

    def _load_slab(self, req: KvbReqMeta, names, dtype_name, bytes_per_layer,
                   total, keys, pass_blocks: int, deadline: float | None = None) -> bool:
        """Slab-staged load: the client drains block bodies straight into
        disjoint pinned-slab slots (the layout gate runs in alloc BEFORE any
        body byte is accepted, exactly like the per-block path), then
        _scatter_slab moves them to the GPU in chunked batches. Loads bigger
        than pass_blocks drain through the slab in pass_blocks-sized passes —
        _scatter_slab's trailing stream synchronize makes reusing the slots
        for the next pass safe, and key_offset keeps every pass's statuses
        mapped to the right GLOBAL block ids. Returns whether any pass used
        the chunked fast path (path attribution)."""
        body_len = total - BLOB_PREFIX_LEN
        slab_np = self._slab_np
        used_ring = False
        for p0 in range(0, len(keys), pass_blocks):
            if deadline is not None and time.monotonic() > deadline:
                # Load deadline blown between passes: flag every remaining
                # promised bid and stop — the scheduler counted them computed,
                # so an unflagged unfilled block is silent garbage.
                for blk in range(req.load_start_block + p0,
                                 req.load_start_block + len(keys)):
                    if blk < len(req.block_ids):
                        self._load_errors.add(req.block_ids[blk])
                self._log.maybe("load-deadline",
                                f"load deadline exceeded — abandoning {len(keys) - p0} "
                                f"remaining blocks (recompute) req={req.req_id}")
                break
            sub = keys[p0:p0 + pass_blocks]

            def alloc(idx, prefix, blen):
                # Runs on the client's concurrent drain threads: slots are
                # disjoint by (pass-local) idx and nothing else is mutated
                # here — thread-safe by construction. idx is 0-based within
                # this batch_get_scatter call, so slots never exceed the pass.
                try:
                    d, n_layers, tpb, bpl, tot = decode_blob_prefix(prefix)
                except BlobError:
                    return None
                if (d != dtype_name or n_layers != len(names) or tpb != self._block_size
                        or bpl != bytes_per_layer or tot != total or blen != body_len):
                    return None  # layout drift -> miss, never a corrupt scatter
                off = idx * body_len
                return memoryview(slab_np[off:off + blen])

            statuses = self._ensure().batch_get_scatter(sub, BLOB_PREFIX_LEN, alloc,
                                                        deadline=deadline)
            used_ring |= self._scatter_slab(req, names, bytes_per_layer, statuses,
                                            key_offset=p0)
        return used_ring

    def _scratch_ring(self, dev, n_layers, bytes_per_layer):
        """The GPU scratch ring (2 × [chunk, n_layers, bytes_per_layer] uint8),
        cached per (device, layout) and reused across loads."""
        key = (str(dev), n_layers, bytes_per_layer)
        if self._gpu_scratch is not None and self._gpu_scratch_key == key:
            return self._gpu_scratch
        torch = _torch()
        ring = [
            torch.empty((_SCATTER_CHUNK, n_layers, bytes_per_layer),
                        dtype=torch.uint8, device=dev)
            for _ in range(_SCRATCH_RING)
        ]
        self._gpu_scratch, self._gpu_scratch_key = ring, key
        return ring

    def _scatter_slab(self, req: KvbReqMeta, names, bytes_per_layer, statuses,
                      key_offset: int = 0) -> bool:
        """Chunked batched H2D scatter from the slab. A chunk whose statuses
        are ALL OK takes the fast path: ONE non_blocking H2D of the contiguous
        slab region into the scratch buffer, then per layer one index_copy_
        into the paged buffer viewed as uint8 rows (bid index built only from
        status-OK blocks — in the fast path that is the whole chunk). Any
        chunk containing a non-OK/missing block falls back to per-block copies
        for THAT chunk only. Never raises: any failure flags the affected
        block ids (chunk-superset flagging allowed) and degrades. One stream
        synchronize at the end — the load is synchronous by contract, and the
        sync is what makes reusing the slab for a next pass safe. key_offset
        maps this pass's statuses onto the request's global block range.
        Returns whether the chunked fast path was available (path attribution)."""
        torch = _torch()
        n_layers = len(names)
        body_len = n_layers * bytes_per_layer
        start = req.load_start_block + key_offset
        n = len(statuses)
        # Physical bid per local index; None = skip (non-OK — flagged — or no id).
        bids: list[int | None] = [None] * n
        for j, st in enumerate(statuses):
            blk = start + j
            bid = req.block_ids[blk] if blk < len(req.block_ids) else None
            if st != kp.Status.OK:
                if bid is not None:
                    self._load_errors.add(bid)
                continue
            bids[j] = bid
        dev = self._layer_kv[names[0]].device
        paged_u8 = None
        ring = None
        if not self._chunked_disabled:
            try:
                paged_u8 = {}
                for name in names:
                    t = self._layer_kv[name]
                    # view, NEVER reshape: view aliases or raises, and the
                    # raise lands here -> per-block fallback. reshape silently
                    # COPIES a non-contiguous paged tensor, so index_copy_
                    # would write into a temporary and every byte would be
                    # silently lost (the refuter-verified BLOCKER).
                    paged_u8[name] = t.view(torch.uint8).view(t.shape[0], -1)
                ring = self._scratch_ring(dev, n_layers, bytes_per_layer)
                self._scratch_fails = 0
            except Exception as e:  # noqa: BLE001 — never-raise boundary: non-viewable layout / scratch OOM degrades to per-block copies
                ring = None
                self._scratch_fails += 1
                if self._scratch_fails >= _SCRATCH_MAX_FAILS:
                    self._chunked_disabled = True
                    self._log.maybe(
                        "scatter",
                        f"chunked-scatter setup failed {self._scratch_fails}x in a row "
                        "— latched OFF for this connector's lifetime (per-block copies)", e)
                else:
                    self._log.maybe("scatter", "chunked-scatter setup failed — per-block fallback", e)
        first_fast: tuple[int, int] | None = None  # (slab slot j, bid) for the debug check
        for c0 in range(0, n, _SCATTER_CHUNK):
            c1 = min(c0 + _SCATTER_CHUNK, n)
            chunk_bids = bids[c0:c1]
            if ring is not None and all(b is not None for b in chunk_bids):
                try:
                    nblk = c1 - c0
                    src = self._slab[c0 * body_len : c1 * body_len].view(
                        nblk, n_layers, bytes_per_layer)
                    scratch = ring[0]
                    scratch[:nblk].copy_(src, non_blocking=True)
                    idx = torch.tensor(chunk_bids, dtype=torch.long, device=dev)
                    for li, name in enumerate(names):
                        paged_u8[name].index_copy_(0, idx, scratch[:nblk, li])
                    if first_fast is None:
                        first_fast = (c0, chunk_bids[0])
                except Exception as e:  # noqa: BLE001 — never-raise boundary: a failed chunk flags its blocks, not the engine
                    self._log.maybe("scatter", "chunked H2D scatter failed", e)
                    self._load_errors.update(b for b in chunk_bids if b is not None)
                continue
            # Mixed / degraded chunk: per-block copies for this chunk only.
            for j in range(c0, c1):
                bid = bids[j]
                if bid is None:
                    continue
                self._scatter_block_from_slab(j, bid, names, bytes_per_layer)
        if getattr(dev, "type", "") == "cuda":
            try:
                torch.cuda.current_stream(dev).synchronize()
            except Exception as e:  # noqa: BLE001 — never-raise boundary: an unfinished stream means nothing is provably loaded
                self._log.maybe("scatter", "stream synchronize failed", e)
                self._load_errors.update(b for b in bids if b is not None)
        if (first_fast is not None and not self._debug_scatter_checked
                and os.environ.get("KVBLOCKD_DEBUG_SCATTER_CHECK") == "1"):
            self._debug_check_scatter(first_fast, names, bytes_per_layer)
        return ring is not None

    def _debug_check_scatter(self, first_fast: tuple[int, int], names,
                             bytes_per_layer) -> None:
        """KVBLOCKD_DEBUG_SCATTER_CHECK=1: after the first chunked-scatter
        load, compare ONE scattered block's bytes on the paged tensor against
        its slab source (uint8 equality) and log PASS/FAIL once. Runs after
        the stream synchronize; off by default; never raises."""
        self._debug_scatter_checked = True
        torch = _torch()
        j, bid = first_fast
        try:
            body_len = len(names) * bytes_per_layer
            slot = self._slab[j * body_len : (j + 1) * body_len]
            ok = True
            for li, name in enumerate(names):
                got = (self._layer_kv[name][bid].contiguous()
                       .view(torch.uint8).reshape(-1).cpu())
                want = slot[li * bytes_per_layer : (li + 1) * bytes_per_layer]
                if not torch.equal(got, want):
                    ok = False
                    break
            logger.info("kvblockd debug scatter check: %s (slab slot %d -> paged block %d)",
                        "PASS" if ok else "FAIL", j, bid)
        except Exception as e:  # noqa: BLE001 — a broken debug probe must not break the load
            logger.info("kvblockd debug scatter check: FAIL (comparison errored: %s)", e)

    def _scatter_block_from_slab(self, j: int, bid: int, names, bytes_per_layer) -> None:
        """Per-block scatter of slab slot j into physical block bid — the same
        per-layer copy_ as the original path, sourced from the slab."""
        body_len = len(names) * bytes_per_layer
        buf = self._slab[j * body_len : (j + 1) * body_len]
        for li, name in enumerate(names):
            dst = self._layer_kv[name][bid]
            src = buf[li * bytes_per_layer : (li + 1) * bytes_per_layer]
            try:
                dst.copy_(src.view(dst.dtype).reshape(dst.shape))
            except Exception as e:  # noqa: BLE001 — never-raise boundary: a failed scatter marks the block errored, not the engine
                self._log.maybe("scatter", f"scatter into {name} failed", e)
                self._load_errors.add(bid)
                break

    def wait_for_layer_load(self, layer_name: str) -> None:
        # No-op by design: blob granularity is a whole block across ALL layers,
        # and start_load_kv loads synchronously before the forward pass — there
        # is nothing per-layer left to wait for.
        return

    def save_kv_layer(self, layer_name: str, kv_layer, attn_metadata, **kwargs) -> None:
        # Accumulate only: keep the paged-buffer reference; all extraction
        # happens once in wait_for_save when every layer's KV is final.
        self._layer_kv[layer_name] = kv_layer

    def wait_for_save(self):
        """Stage-sync + drain-async (kvblockd_async_store, default on): every
        block's bytes are copied into an OWNED buffer HERE, before returning —
        the vLLM base contract only protects the paged buffer until this call
        returns, and on CPU _block_bytes hands back an ALIASING view of paged
        memory — then a background thread drains the copies over TCP. With the
        flag off this is the original synchronous put loop, byte-identical."""
        try:
            metadata = self._get_connector_metadata()
        except Exception:  # noqa: BLE001 — never-raise boundary: missing metadata = nothing to store this step
            return
        requests = getattr(metadata, "requests", None) or []
        for req in requests:
            if req.store_end_block > req.store_start_block:
                try:
                    if self._cfg.async_store:
                        self._stage_one(req)
                    else:
                        self._store_one(req)
                except Exception as e:  # noqa: BLE001 — never raise: a lost store is a future miss
                    self._log.maybe("store", f"kvblockd store failed req={req.req_id}", e)
                    self._drop_client(e)

    # ------------------------------------------------------------------
    # write-behind store queue (kvblockd_async_store)
    # ------------------------------------------------------------------
    def _stage_one(self, req: KvbReqMeta) -> None:
        """Copy every store-range block into ONE owned bytearray per block
        (32B layout prefix + all layers, exactly the blob _store_one streams)
        and enqueue it. The copies MUST happen inside wait_for_save: after it
        returns the scheduler may reuse the paged blocks, and _block_bytes is
        zero-copy on CPU — draining the view later would stream whatever the
        engine wrote over it (silent corruption armor, not an optimization)."""
        names, dtype_name, bytes_per_layer = self._layout()
        if not names:
            return
        total = BLOB_PREFIX_LEN + bytes_per_layer * len(names)
        prefix = encode_blob_prefix(dtype_name, len(names), self._block_size,
                                    bytes_per_layer, total)
        seed = self._seed(req.cache_salt, req.mm_ids, req.lora_name)
        keys = block_chain_keys(seed, req.token_ids, self._block_size)
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
        for j in range(req.store_start_block, end):
            bid = req.block_ids[j]
            buf = bytearray(total)
            buf[:BLOB_PREFIX_LEN] = prefix
            dst = np.frombuffer(buf, dtype=np.uint8)  # writable view of buf
            for li, name in enumerate(names):
                src = self._block_bytes(self._layer_kv[name][bid])
                dst[BLOB_PREFIX_LEN + li * bytes_per_layer:
                    BLOB_PREFIX_LEN + (li + 1) * bytes_per_layer] = src  # copies
            if not self._sq_enqueue(keys[j], buf):
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
                return

    def _sq_enqueue(self, key: bytes, buf: bytearray) -> bool:
        """Enqueue one owned blob. NEVER blocks and NEVER raises: past the
        byte budget (or during shutdown) the block is dropped and counted —
        a lost store is a future miss, an engine stall is an incident."""
        n = len(buf)
        with self._sq_cond:
            if self._store_stop or self._sq_bytes + n > self._cfg.store_queue_bytes:
                self.dropped_puts += 1
                self.dropped_put_bytes += n
                dropped, failed, dbytes = (self.dropped_puts, self.failed_puts,
                                           self.dropped_put_bytes)
            else:
                self._sq.append((key, buf))
                self._sq_bytes += n
                self._sq_cond.notify_all()
                dropped = None
        if dropped is not None:
            # Rate-limited in-run disclosure (the shutdown summary line is
            # unconditional); the bench populate phase greps `dropped=`.
            # OUTSIDE the lock: _log.maybe formats and may hit a logging
            # handler — never hold _sq_cond across foreign code.
            self._log.maybe(
                "store-drop",
                f"kvblockd store queue overflow: dropped={dropped} "
                f"failed={failed} dropped_bytes={dbytes}",
            )
            return False
        self._store_thread_start()
        return True

    def _store_thread_start(self) -> None:
        """Lazily start the single "kvb-store" drain thread. Only ever called
        from the engine's serving thread (wait_for_save), so the check-then-
        start needs no extra lock."""
        t = self._store_thread
        if t is not None and t.is_alive():
            return
        t = threading.Thread(target=self._store_drain, name="kvb-store", daemon=True)
        self._store_thread = t
        t.start()

    def _store_drain(self) -> None:
        """FIFO drain loop: pop one staged blob, put it. A CONNECTION-class
        failure (daemon gone, breaker window) re-queues the item ONCE at the
        head and waits out the redial backoff before retrying — a blip costs
        one backoff window, never the whole backlog burned at one doomed put
        per item. A SECOND consecutive failure of the same item counts it
        failed (failed_puts) and moves on (no infinite loop); non-connection
        failures count immediately. Either way the client is dropped with the
        same breaker discipline as loads and the thread NEVER dies to an op
        error, so delivery resumes after a redial. OK_EXISTS = dedup, fine."""
        retry_of: bytearray | None = None  # the buf that already failed once
        while True:
            with self._sq_cond:
                while not self._sq and not self._store_stop:
                    self._sq_cond.wait()
                if self._store_abort or (not self._sq and self._store_stop):
                    return
                key, buf = self._sq.popleft()
                n = len(buf)
                self._sq_bytes -= n
                self._sq_inflight += 1
                self._sq_inflight_bytes += n
            err: BaseException | None = None
            try:
                self._ensure().put(key, [buf])
            except Exception as e:  # noqa: BLE001 — never let the drain thread die: a lost store is a future miss
                err = e
            requeue = (err is not None
                       and isinstance(err, (ConnectionLost, OSError))
                       and buf is not retry_of)
            with self._sq_cond:
                self._sq_inflight -= 1
                self._sq_inflight_bytes -= n
                # Reconcile with a shutdown that already disclosed this
                # in-flight block as dropped (join timed out mid-put).
                counted_dropped = self._sq_inflight_counted > 0
                if counted_dropped:
                    self._sq_inflight_counted -= 1
                if err is None:
                    retry_of = None
                    if counted_dropped:  # delivered after all: un-count it
                        self.dropped_puts -= 1
                        self.dropped_put_bytes -= n
                elif requeue and not self._store_abort:
                    self._sq.appendleft((key, buf))
                    self._sq_bytes += n
                    retry_of = buf
                elif not counted_dropped:  # dropped-at-shutdown is not ALSO failed
                    retry_of = None
                    self.failed_puts += 1
                self._sq_cond.notify_all()
            if err is None:
                continue
            self._log.maybe("store", "kvblockd async store failed", err)
            self._drop_client(err)
            if requeue:
                # Sit out the redial backoff (and the dial breaker _drop_client
                # just armed) instead of instantly re-failing the retry. The
                # deadline loop ignores enqueue wakeups; shutdown's stop/abort
                # flags cut it short so a flush is never held hostage.
                wake = max(time.monotonic() + _REDIAL_BACKOFF_S, self._next_dial)
                with self._sq_cond:
                    while not self._store_stop and not self._store_abort:
                        left = wake - time.monotonic()
                        if left <= 0:
                            break
                        self._sq_cond.wait(left)

    def _store_flush(self, timeout: float) -> int:
        """Wait (up to timeout) until the queue is empty and nothing is in
        flight. Returns the number of UNDELIVERED blocks left at timeout."""
        deadline = time.monotonic() + timeout
        with self._sq_cond:
            while self._sq or self._sq_inflight:
                left = deadline - time.monotonic()
                if left <= 0:
                    return len(self._sq) + self._sq_inflight
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
        t = self._store_thread
        if t is not None and t.is_alive():
            t.join(self._cfg.store_flush_timeout_s)
        with self._sq_cond:
            self._store_abort = True  # a wedged put must not keep delivering
            remainder = len(self._sq) + self._sq_inflight
            if remainder:
                self.dropped_puts += remainder
                self.dropped_put_bytes += self._sq_bytes + self._sq_inflight_bytes
                # An in-flight put is UNDELIVERED at disclosure time, so it
                # counts dropped — but the put may still return: remember how
                # many were counted so the drain thread can reconcile
                # (delivered -> un-count; failed -> not ALSO failed_puts).
                self._sq_inflight_counted = self._sq_inflight
            self._sq.clear()
            self._sq_bytes = 0
            self._sq_cond.notify_all()
            dropped, failed, dbytes = (self.dropped_puts, self.failed_puts,
                                       self.dropped_put_bytes)
        logger.warning("kvblockd store queue: dropped=%d failed=%d dropped_bytes=%d",
                       dropped, failed, dbytes)

    def handle_preemptions(self, *args, **kwargs) -> None:
        """Explicit NO-OP, on purpose: the write-behind queue holds OWNED
        copies staged inside wait_for_save — nothing in it references paged
        memory, so a preempted request's blocks can be freed and reused
        immediately without corrupting a queued store."""
        return

    def _store_one(self, req: KvbReqMeta) -> None:
        names, dtype_name, bytes_per_layer = self._layout()
        if not names:
            return
        total = BLOB_PREFIX_LEN + bytes_per_layer * len(names)
        prefix = encode_blob_prefix(dtype_name, len(names), self._block_size,
                                    bytes_per_layer, total)
        seed = self._seed(req.cache_salt, req.mm_ids, req.lora_name)
        keys = block_chain_keys(seed, req.token_ids, self._block_size)
        client = self._ensure()
        end = min(req.store_end_block, len(keys), len(req.block_ids))
        for j in range(req.store_start_block, end):
            bid = req.block_ids[j]
            bufs = [prefix]
            bufs.extend(self._block_bytes(self._layer_kv[name][bid]) for name in names)
            client.put(keys[j], bufs)  # OK_EXISTS = idempotent dedup (write-once)

    def get_finished(self, finished_req_ids):
        return None, None  # all loads/saves are synchronous within the step

    def get_block_ids_with_load_errors(self) -> set[int]:
        errs, self._load_errors = self._load_errors, set()
        return errs
