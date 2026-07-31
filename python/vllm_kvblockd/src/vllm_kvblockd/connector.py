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
import queue
import struct
import threading
import time
from dataclasses import dataclass, field

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
# v2 layout: one former pad byte is codec_id (13 pad bytes remain). The
# version bump makes every pre-codec reader degrade v2 blobs to a clean miss
# (decode rejects unknown versions), never a misread field — and the
# fingerprint folds BLOB_VERSION, so the keyspaces fork too.
_BLOB = struct.Struct("<4sBBHHIIB13x")  # magic ver dtype n_layers tokens bytes/layer total codec
assert _BLOB.size == BLOB_PREFIX_LEN

# Pinned dtype codes (same table as lmcache_kvblockd.meta — kept in sync by
# tests/test_connector.py::test_dtype_codes_match_w5).
DTYPE_CODES = {
    "float16": 0, "bfloat16": 1, "float32": 2, "float64": 3,
    "uint8": 4, "int8": 5, "int32": 6, "int64": 7,
    "float8_e4m3fn": 8, "float8_e5m2": 9,
}
CODE_DTYPES = {v: k for k, v in DTYPE_CODES.items()}

# Blob body codecs (item 6 keystone — the FIELD ships before any serde).
# The prefix always describes the DECODED layout; for codec != raw the wire's
# per-blob body_len is the COMPRESSED size (client.py hands it to alloc), and
# slab slot sizing switches from body_len to the codec's max_body_len so
# fixed-ratio codecs keep the fixed-stride math — that switch lands WITH the
# first serde, not before. Codec identity never enters key derivation
# (namespaces are the tool for split fleets). Every alloc gate refuses a
# codec this build is not configured to decode — a clean per-block miss,
# never a corrupt scatter.
CODEC_RAW = 0        # body = the paged block bytes, verbatim
CODEC_FP8_CAST = 1   # RESERVED: on-GPU fp8-cast serde — ships only behind
#                      the pre-registered lossy quality gate; until then any
#                      fp8-cast blob is refused on load (per-block miss)
CODEC_NAMES = {CODEC_RAW: "raw", CODEC_FP8_CAST: "fp8-cast"}

# After a failed dial, further dial attempts short-circuit for this long —
# callers degrade to a miss instantly instead of each eating a connect timeout.
_REDIAL_BACKOFF_S = 5.0
# How long the store drain parks per check while another caller's dial is in
# flight (see _DialPending) — bounded overall by connect_timeout, since the
# dial itself is.
_DIAL_PENDING_PARK_S = 0.05


class _DialPending(ConnectionError):
    """_ensure()'s 'another caller owns the in-flight dial' signal. NOT a
    wire failure and NOT a delivery attempt: lookups/loads degrade to a miss
    on it exactly as before (it is still a ConnectionError), but the store
    drain PARKS on it — requeue with NO strike — because charging it as a
    requeued block's second put failure counted failed_puts += 1 and freed
    the slot without a second wire attempt ever happening (a permanently
    lost block plus a failed_puts count inflated with a non-attempt,
    breaking byte-for-byte disclosure exactness)."""
# Load-priority drain gate: while the engine is actively pulling KV, the
# kvb-store drain parks (bounded) so store traffic never contends with a
# latency-path load for the wire or the GIL. The ceiling guarantees a
# wedged counter can only ever DELAY the drain, never stop it.
_DRAIN_GATE_CEILING_S = 0.25

# Async-lookup pending-map ceiling: at the cap a NEW lookup answers (0, False)
# — a miss — instead of None, because a None with no queued work would park
# the request on a result that never comes (never-None-under-pressure rule).
_LOOKUP_PENDING_CAP = 1024

# Worker-side chain-key memo ceiling (entries = requests). Chunked prefill
# re-derives the FULL BLAKE3 chain over the whole prompt in _stage_one and
# _load_one each step — O(prompt^2) over a long prefill; the chain is
# append-only per request, so caching (seed, keys) and extending from the
# last key makes each step O(new blocks). Correctness never depends on the
# cache (evicted entries recompute), so a plain FIFO cap is enough armor.
_CHAIN_CACHE_CAP = 2048

# Periodic one-line stats summary interval (WARNING on purpose — this logger
# is unconfigured in vLLM's engine-core process, where the root default drops
# INFO; the grep-based rigs read the same line).
_STATS_SUMMARY_S = 60.0


class _ConnStats:
    """Connector telemetry: hits/misses, per-verb latency sums, store-queue
    high-water, breaker trips, deadline aborts. Public via
    KvblockdConnector.stats (attributes + snapshot()) and a periodic one-line
    WARNING summary — the answers to 'is the cache helping', 'is the queue
    about to overflow', and 'is the breaker flapping' without log
    archaeology. Thread-safe: engine, resolver, and drain threads all bump."""

    __slots__ = ("_lock", "breaker_trips", "hits", "load_count",
                 "load_deadline_aborts", "load_time_s", "lookup_count",
                 "lookup_time_s", "misses", "sq_bytes_hwm", "store_count",
                 "store_time_s")

    def __init__(self):
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.lookup_count = 0
        self.lookup_time_s = 0.0
        self.load_count = 0
        self.load_time_s = 0.0
        self.load_deadline_aborts = 0
        self.store_count = 0
        self.store_time_s = 0.0
        self.sq_bytes_hwm = 0
        self.breaker_trips = 0

    def bump(self, name: str, n=1) -> None:
        with self._lock:
            setattr(self, name, getattr(self, name) + n)

    def note_lookup(self, hit: bool, elapsed: float | None = None) -> None:
        with self._lock:
            if hit:
                self.hits += 1
            else:
                self.misses += 1
            if elapsed is not None:
                self.lookup_count += 1
                self.lookup_time_s += elapsed

    def note_hwm(self, sq_bytes: int) -> None:
        with self._lock:
            self.sq_bytes_hwm = max(self.sq_bytes_hwm, sq_bytes)

    def snapshot(self) -> dict:
        with self._lock:
            return {s: getattr(self, s) for s in self.__slots__ if s != "_lock"}


class BlobError(ValueError):
    """Unrecognized/incompatible blob prefix — the caller treats it as a miss."""


def encode_blob_prefix(dtype_name: str, n_layers: int, tokens_per_block: int,
                       bytes_per_layer: int, total_len: int,
                       codec: int = CODEC_RAW) -> bytes:
    if dtype_name not in DTYPE_CODES:
        raise BlobError(f"unsupported dtype {dtype_name!r}")
    if codec not in CODEC_NAMES:
        # Refused at ENCODE time too: a blob tagged with a codec nothing
        # defines would be undecodable everywhere, forever.
        raise BlobError(f"unknown codec {codec}")
    return _BLOB.pack(BLOB_MAGIC, BLOB_VERSION, DTYPE_CODES[dtype_name],
                      n_layers, tokens_per_block, bytes_per_layer, total_len,
                      codec)


def decode_blob_prefix(prefix: bytes) -> tuple[str, int, int, int, int, int]:
    """(dtype, n_layers, tokens/block, bytes/layer, total, codec) — the
    layout is always the DECODED one; a non-raw codec means the wire body is
    the codec's compressed form of exactly that layout."""
    if len(prefix) < BLOB_PREFIX_LEN:
        raise BlobError("prefix too short")
    magic, ver, dcode, n_layers, tpb, bpl, total, codec = _BLOB.unpack(
        prefix[:BLOB_PREFIX_LEN])
    if magic != BLOB_MAGIC:
        raise BlobError(f"bad magic {magic!r}")
    if ver != BLOB_VERSION:
        raise BlobError(f"unknown blob version {ver}")
    if dcode not in CODE_DTYPES:
        raise BlobError(f"unknown dtype code {dcode}")
    if codec not in CODEC_NAMES:
        # An id from a future release: this build cannot know the body's
        # encoding, so the block is a clean miss — never a guessed decode.
        raise BlobError(f"unknown codec {codec}")
    return CODE_DTYPES[dcode], n_layers, tpb, bpl, total, codec


# Composed collaborators — the two unit-owned extractions of this file's
# biggest lifecycles: the slab/scatter/pipelined LOAD machinery and the
# write-behind STORE queue machinery. Imported HERE, not at the top of the
# file, because both modules import blob-codec / dial-plumbing names defined
# above (cycle-safe by ordering). _AckedKeyLRU is re-exported unchanged for
# its existing importers (tests).
from .slab_loader import SlabLoader
from .store_queue import StoreQueue, _AckedKeyLRU, _StagePlan  # noqa: F401


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
        # True while ONE caller dials outside _client_lock; every concurrent
        # caller fails fast (degrades to miss) instead of queueing behind a
        # connect_timeout — the documented 'one bounded delay' contract.
        self._dialing = False
        self._log = _RateLimitedLog()
        self._closed = False
        # Telemetry (commit: degrade-to-miss visibility). Public on purpose.
        self.stats = _ConnStats()
        self._stats_last_emit = time.monotonic()

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
        # req_id -> (seed, chain keys derived so far). The chain is append-
        # only per request (prompt tokens are immutable, the seed pins
        # salt/lora/mm identity), so chunked-prefill steps extend from the
        # last cached key instead of re-hashing the whole prompt — O(new
        # blocks) per step instead of O(prompt) (O(prompt^2) per prefill).
        # Worker-side + engine-thread-only; FIFO-capped, never authoritative.
        self._chain_cache: dict[str, tuple[bytes, list[bytes]]] = {}
        # Cost-crossover estimator inputs (kvblockd_recompute_ms_per_token):
        # EMA of observed load throughput (bytes/s over whole _load_one
        # calls) and the live layout's blob bytes per token. Written by the
        # WORKER-role instance's load path, read by the SCHEDULER-role
        # instance's gate — and vLLM always constructs those as separate
        # objects/processes, so the gate's EMA is None in every real
        # deployment. The config parse therefore REFUSES the knob when
        # nonzero (see config.py) until a worker->scheduler channel carries
        # the EMA across; the machinery stays because it is that future
        # plumbing's evidence source.
        self._load_bps_ema: float | None = None
        self._load_bytes_per_token = 0.0

        self._prewarm_done = False        # one eager-pin attempt at first CUDA capture
        # Composed collaborators (see slab_loader.py / store_queue.py): the
        # LOAD machinery's and STORE machinery's state lives on them and is
        # aliased back onto the connector below (_alias_state), so every
        # existing seam — tests, the bench counters, cross-lifecycle
        # callers — reads and patches the connector exactly as before.
        self._loader = SlabLoader(self)
        self._store_q = StoreQueue(self)

    # ------------------------------------------------------------------
    # client plumbing (lazy: import/instantiate must succeed with no daemon)
    # ------------------------------------------------------------------
    def _ensure(self) -> Client:
        """Return the live client, dialing OUTSIDE _client_lock: a dial
        blocks up to connect_timeout and primes a connection, and holding
        the lock across it stalled every other caller (engine, resolver,
        drain, shutdown) behind one redial — quietly exceeding the documented
        'one bounded delay' contract. Under the lock: breaker check + a
        dialing flag; concurrent callers seeing the flag raise the dial-
        suppressed ConnectionError immediately (degrade to miss)."""
        with self._client_lock:
            if self._closed:
                raise ConnectionError("connector closed")
            if self._client is not None:
                return self._client
            # Dial breaker: without it, every caller of a dead endpoint
            # eats a full connect timeout (one stalled scheduler step per
            # waiting request under a blackholed daemon).
            now = time.monotonic()
            if now < self._next_dial:
                raise ConnectionError("kvblockd dial suppressed after recent failure")
            if self._dialing:
                raise _DialPending("kvblockd dial in progress — degraded to miss")
            self._dialing = True
        try:
            client = Client(
                (self._cfg.host, self._cfg.port),
                namespace=self._cfg.namespace,
                token=self._cfg.token,
                streams=self._cfg.streams,
                get_fanout=self._cfg.get_fanout,
                connect_timeout=self._cfg.connect_timeout,
                op_timeout=self._cfg.op_timeout,
                verify=self._cfg.verify,
                so_rcvbuf=self._cfg.so_rcvbuf,
            )
        except Exception:
            with self._client_lock:
                self._dialing = False
                self._next_dial = time.monotonic() + _REDIAL_BACKOFF_S
            self.stats.bump("breaker_trips")
            raise
        with self._client_lock:
            self._dialing = False
            if self._closed:
                pass  # shutdown raced the dial: don't install (closed below)
            elif self._client is None:
                self._client = client
                return client
            else:
                # Another caller installed first (cannot happen under the
                # dialing flag, but stay single-owner defensively).
                pass
            stale = client
        stale.close()
        raise ConnectionError("connector closed" if self._closed
                              else "kvblockd dial superseded")

    def _drop_client(self, exc: BaseException) -> None:
        """ConnectionLost/OSError out of a batch op means the pooled
        connections are dead (daemon gone or blackholed): drop the whole
        client and re-arm the dial breaker, so the outage costs ONE
        connect_timeout per backoff window — not one per load, with every
        pooled conn re-dialing under it. A dial-suppressed ConnectionError
        (client already None) re-arms nothing: extending the window on every
        suppressed call would starve the retry forever under constant load."""
        if isinstance(exc, _DialPending):
            # Not a wire failure — another caller owns the in-flight dial,
            # and racing it here could drop the client that dial is about
            # to install (then arm the breaker against a healthy daemon).
            return
        if not isinstance(exc, (ConnectionLost, OSError)):
            return  # e.g. StatusError: the conn is in sync, keep the pool
        with self._client_lock:
            client, self._client = self._client, None
            if client is None:
                return  # dial failure/suppression: _ensure already armed it
            self._next_dial = time.monotonic() + _REDIAL_BACKOFF_S
        self.stats.bump("breaker_trips")
        # Every entry in the ack LRU was proven against the connection that
        # just died; the daemon behind the redial may have restarted with an
        # EMPTY store, and a trusted stale ack suppresses the self-healing
        # re-put for up to the TTL. Connection loss marks them ALL stale.
        if self._acked_keys is not None:
            self._acked_keys.clear()
        client.close()

    def shutdown(self):
        if self._resolver is not None:
            self._resolver.stop(1.0)  # sentinel + bounded join
        if self._prefetch_ex is not None:
            # Loads are engine-thread-synchronous, so no drain can be in
            # flight here; join the worker so no load thread outlives us.
            self._prefetch_ex.shutdown(wait=True)
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

    def _chain_keys(self, rid: str, seed: bytes, token_ids: list[int]) -> list[bytes]:
        """Chain keys for token_ids, memoized per request (worker side). The
        chain's prefix property makes extension exact: key_i folds key_{i-1},
        so block_chain_keys(prev=last cached key, tail tokens) IS the tail of
        block_chain_keys(seed, all tokens) — proven byte-for-byte by
        tests/test_connector.py::test_chain_key_cache_extends_exactly. A seed
        mismatch (different salt/lora/mm identity, or a flag-only row) or a
        SHORTER token list than the cache falls back to a full recompute —
        the cache can only ever save work, never change bytes."""
        n_blocks = len(token_ids) // self._block_size
        ent = self._chain_cache.pop(rid, None)  # pop: reinsert refreshes FIFO order
        if ent is not None and ent[0] == seed and len(ent[1]) <= n_blocks:
            keys = ent[1]
            if len(keys) < n_blocks:
                prev = keys[-1] if keys else seed
                keys = keys + block_chain_keys(
                    prev, token_ids[len(keys) * self._block_size:], self._block_size)
        else:
            keys = block_chain_keys(seed, token_ids, self._block_size)
        self._chain_cache[rid] = (seed, keys)
        while len(self._chain_cache) > _CHAIN_CACHE_CAP:
            self._chain_cache.pop(next(iter(self._chain_cache)))
        return keys

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
        self._maybe_stats_summary()
        if self._cfg.async_lookup and rid is not None:
            return self._lookup_async(rid, request, num_computed_tokens)
        return self._lookup_sync(request, num_computed_tokens)

    def _maybe_stats_summary(self) -> None:
        """One periodic WARNING line per role instance (scheduler counters
        and worker counters live in different processes) — the grep-based
        rigs' view of hit rate, queue pressure, and breaker health. Never
        raises; steady-state cost is one clock read."""
        now = time.monotonic()
        if now - self._stats_last_emit < _STATS_SUMMARY_S:
            return
        self._stats_last_emit = now
        try:
            s = self.stats.snapshot()
            client = self._client
            cc = getattr(client, "counters", None)
            extra = ""
            if cc is not None:
                c = cc.snapshot()
                extra = (f" client_evictions={c['evictions']}"
                         f" client_deadline_misses={c['deadline_misses']}"
                         f" client_corrupt_blocks={c['corrupt_blocks']}"
                         f" client_degraded_keys={c['degraded_keys']}")
            with self._sq_cond:
                sq_bytes = self._sq_bytes
            logger.warning(
                "kvblockd stats: hits=%d misses=%d lookup_s=%.3f/%d load_s=%.3f/%d "
                "store_s=%.3f/%d sq_bytes=%d sq_hwm=%d breaker_trips=%d "
                "deadline_aborts=%d dropped=%d failed=%d deduped=%d%s",
                s["hits"], s["misses"], s["lookup_time_s"], s["lookup_count"],
                s["load_time_s"], s["load_count"], s["store_time_s"], s["store_count"],
                sq_bytes, s["sq_bytes_hwm"], s["breaker_trips"],
                s["load_deadline_aborts"], self.dropped_puts, self.failed_puts,
                self.deduped_puts, extra)
        except Exception:  # noqa: BLE001, S110 — telemetry must never break serving
            pass

    def _load_ms_per_token(self) -> float | None:
        """Measured load cost in ms/token from the throughput EMA, or None
        while nothing has been measured (the gate then admits)."""
        ema = self._load_bps_ema
        bpt = self._load_bytes_per_token
        if not ema or bpt <= 0:
            return None
        return bpt / ema * 1000.0

    def _observe_load(self, ok_blocks: int, total: int, elapsed: float) -> None:
        """Feed one completed _load_one into the throughput EMA. ok_blocks
        approximates delivered blocks (promised minus newly-flagged); misses
        counted as delivered only OVERSTATE throughput, which biases the gate
        toward admitting — the safe direction (today's behavior).

        DEBT for the cross-role plumbing wave (the parse refuses the knob
        until then): the timer starts before the ONE-TIME pinned-slab
        allocation, and the EMA has no decay/probe — a slow first sample
        would refuse every hit for the life of the process. Both must be
        fixed in the same change that carries the EMA worker->scheduler."""
        if ok_blocks <= 0 or elapsed <= 0 or total <= 0:
            return
        bps = ok_blocks * total / elapsed
        ema = self._load_bps_ema
        self._load_bps_ema = bps if ema is None else 0.2 * bps + 0.8 * ema
        self._load_bytes_per_token = total / self._block_size

    def _gate_hit_tokens(self, ext_tokens: int) -> int:
        """Never-lose-to-recompute admission gate — INERT at the defaults
        (min_hit_tokens=0, recompute_ms_per_token=0). Refuses the WHOLE hit
        or admits the whole hit, NEVER truncates it to the threshold: causal
        attention makes the TAIL of the prefix the expensive end, so a
        truncated head-hit would keep exactly the costly part as recompute
        while still paying the load. A refused hit is a disclosed miss —
        vLLM recomputes, never a wrong byte."""
        if ext_tokens <= 0:
            return 0
        mht = self._cfg.min_hit_tokens
        if mht > 0 and ext_tokens < mht:
            return 0
        # rc is 0.0 in every real deployment — the config parse refuses a
        # nonzero value until the EMA is plumbed worker->scheduler (the
        # branch is kept as the landing point for that plumbing).
        rc = self._cfg.recompute_ms_per_token
        if rc > 0:
            load_ms = self._load_ms_per_token()
            if load_ms is not None and load_ms >= rc:
                return 0  # loading cannot pay at ANY length under a linear model
        return ext_tokens

    def _lookup_sync(self, request, num_computed_tokens: int):
        """The original synchronous lookup (flag off / fallback), now routed
        through the admission gate (identity at the gate's defaults) and
        budgeted by kvblockd_exists_timeout_s: this call blocks EVERY
        scheduling step, so a hung-but-accepting daemon must cost a bounded
        blip (then the breaker answers instantly for its window), never the
        global 10s op_timeout per recv — the verb is <1ms p99 by design."""
        t0 = time.monotonic()
        try:
            token_ids = list(getattr(request, "prompt_token_ids", None) or [])
            aligned = align_to_block_size(len(token_ids), self._block_size)
            if aligned <= num_computed_tokens:
                return 0, False
            seed = self._seed(getattr(request, "cache_salt", None), self._mm_ids(request),
                              self._lora_name(request))
            keys = block_chain_keys(seed, token_ids[:aligned], self._block_size)
            # The budget covers the EXCHANGE, never the dial or the O(prompt)
            # chain hashing above: _ensure() may legitimately spend up to
            # connect_timeout priming a connection (its own bounded,
            # breaker-guarded budget — the documented 'one bounded delay').
            # Armed at t0, any dial+HELLO slower than exists_timeout_s raised
            # 'deadline exceeded before checkout' against a HEALTHY daemon,
            # dropped the freshly-primed client, and armed the breaker — a
            # self-sustaining permanent-0%-hit-rate loop.
            client = self._ensure()
            et = self._cfg.exists_timeout_s
            deadline = time.monotonic() + et if et > 0 else None
            n_consec, _ = client.batch_exists(keys, deadline=deadline)
            hit_tokens = min(n_consec * self._block_size, aligned)
            granted = self._gate_hit_tokens(max(0, hit_tokens - num_computed_tokens))
            self.stats.note_lookup(granted > 0, time.monotonic() - t0)
            return granted, False
        except Exception as e:  # noqa: BLE001 — never raise: a failed lookup is a miss
            self._log.maybe("lookup", "kvblockd BATCH_EXISTS failed (treated as miss)", e)
            self._drop_client(e)
            self.stats.note_lookup(False, time.monotonic() - t0)
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
                granted = self._gate_hit_tokens(
                    max(0, min(hit, aligned) - num_computed_tokens))
                self.stats.note_lookup(granted > 0)
                return granted, False
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
                    self.stats.note_lookup(False)
                    return 0, False
                return None, False
            if len(self._lookup_pending) >= _LOOKUP_PENDING_CAP:
                self._log.maybe("lookup-cap",
                                "async lookup pending map at capacity — treated as miss")
                self.stats.note_lookup(False)
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
            self.stats.note_lookup(False)
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
        except Exception as e:  # noqa: BLE001 — never raise: last-resort boundary; per-request failures are flagged inside
            self._log.maybe("meta", "build_connector_meta failed (no-op step)", e)
        return meta

    def _build_meta_into(self, meta: KvblockdConnectorMetadata, scheduler_output) -> None:
        num_scheduled = getattr(scheduler_output, "num_scheduled_tokens", {}) or {}

        for new_req in getattr(scheduler_output, "scheduled_new_reqs", []) or []:
            try:
                self._build_new_req_meta(meta, new_req, num_scheduled)
            except Exception as e:  # noqa: BLE001 — one bad request must not kill the step
                self._flag_failed_req(meta, getattr(new_req, "req_id", None), new_req, e)

        cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
        if cached is not None:
            self._build_cached_meta(meta, cached, num_scheduled)

        # A tracked load that never surfaced in this step's scheduler output
        # (preempted before running) stays queued for the resumed path; if the
        # request is gone entirely, request_finished prunes it.

    def _build_new_req_meta(self, meta, new_req, num_scheduled) -> None:
        rid = new_req.req_id
        token_ids = list(new_req.prompt_token_ids or [])
        aligned = align_to_block_size(len(token_ids), self._block_size)
        # PEEK the load promise; it is consumed only after its row (or its
        # flag-only stand-in) is in the meta — any exception in between must
        # leave it recoverable by _flag_failed_req.
        load_start, n_load = self._need_load_blocks.get(rid, (0, 0))
        if aligned <= 0:
            if n_load > 0:  # a promise with no storable prefix: flag, never drop
                if self._flag_only_row(meta, rid, load_start, n_load,
                                       self._known_block_ids(rid, new_req)):
                    self._need_load_blocks.pop(rid, None)
            else:
                self._need_load_blocks.pop(rid, None)
            return
        # Store only blocks FULLY computed after this step — under chunked
        # prefill later chunks arrive via scheduled_cached_reqs below.
        computed = int(getattr(new_req, "num_computed_tokens", 0) or 0)
        done = min(aligned, computed + int(num_scheduled.get(rid, 0)))
        store_end = done // self._block_size
        request = self._inflight.get(rid)
        salt = getattr(request, "cache_salt", None) if request is not None else None
        # Strict .identifier: a repr()-derived id differs per process, so
        # it can never round-trip as a key — failing THIS request (caught by
        # the per-request armor, which flags the promised range) beats
        # minting a nondeterministic keyspace. v0.25 mm features always
        # carry .identifier.
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
        self._need_load_blocks.pop(rid, None)  # promise delivered — consume it

    def _flag_failed_req(self, meta, rid, new_req, err) -> None:
        """Per-request meta build failed. The other requests still get their
        rows, and a recorded load promise MUST surface: the scheduler already
        counted those blocks computed, so an unfilled, unflagged block is
        silent garbage attention (this file's own worker-side invariant).
        The promise is PEEKED here and consumed only if a flag row actually
        landed — with no known block ids the promise stays queued so a later
        step that carries ids can still emit the load (or flag) row."""
        if rid is None:
            self._log.maybe("meta-req", "kvblockd meta build failed (no req_id)", err)
            return
        load_start, n_load = self._need_load_blocks.get(rid, (0, 0))
        self._log.maybe(
            "meta-req",
            f"kvblockd meta build failed req={rid}"
            + (" — flagging the promised load range" if n_load > 0 else ""),
            err,
        )
        if n_load > 0 and self._flag_only_row(meta, rid, load_start, n_load,
                                              self._known_block_ids(rid, new_req)):
            self._need_load_blocks.pop(rid, None)  # consumed WITH the row

    def _known_block_ids(self, rid, new_req=None) -> list[int] | None:
        bids = self._blocks.get(rid)
        if bids:
            return bids
        if new_req is not None:
            try:
                return list(new_req.block_ids[0])
            except Exception:  # noqa: BLE001 — shape churn here may be exactly what failed the build
                return None
        return None

    def _flag_only_row(self, meta, rid, load_start, n_load, block_ids) -> bool:
        """Emit a row whose ONLY job is to surface a promised-but-unbuildable
        load range: empty token_ids derive zero keys, so the worker's
        no-derivable-key armor flags every promised bid for recompute, and
        the empty store range keeps wait_for_save away from it. Returns True
        when the row landed (the caller may consume the promise); False when
        no physical ids are known — the caller must KEEP the promise."""
        if n_load <= 0:
            return True
        if not block_ids:
            # Nothing to hand to get_block_ids_with_load_errors — disclose
            # and keep the promise; vanishing silently is the one forbidden
            # outcome.
            self._log.maybe("meta-req",
                            f"promised load range has no known block ids yet req={rid}")
            return False
        meta.requests.append(
            KvbReqMeta(
                req_id=rid,
                token_ids=[],
                cache_salt=None,
                mm_ids=[],
                lora_name="",
                block_ids=list(block_ids),
                load_start_block=load_start,
                num_load_blocks=n_load,
                store_start_block=load_start + n_load,
                store_end_block=load_start + n_load,
            )
        )
        return True

    def _build_cached_meta(self, meta, cached, num_scheduled) -> None:
        req_ids = list(getattr(cached, "req_ids", []) or [])
        resumed = getattr(cached, "resumed_req_ids", set()) or set()
        computed_list = getattr(cached, "num_computed_tokens", []) or []
        new_block_ids = getattr(cached, "new_block_ids", []) or []
        for i, rid in enumerate(req_ids):
            try:
                self._build_cached_req_meta(meta, rid, i, resumed, computed_list,
                                            new_block_ids, num_scheduled)
            except Exception as e:  # noqa: BLE001 — one bad request must not kill the step
                self._flag_failed_req(meta, rid, None, e)

    def _build_cached_req_meta(self, meta, rid, i, resumed, computed_list,
                               new_block_ids, num_scheduled) -> None:
        request = self._inflight.get(rid)
        if request is None:
            return
        computed = int(computed_list[i]) if i < len(computed_list) else 0
        scheduled = int(num_scheduled.get(rid, 0))
        # Cheapest predicate first: `aligned` needs only num_prompt_tokens,
        # and the pure-decode bail below fires for EVERY running request on
        # EVERY decode step — materializing list(request.all_token_ids) first
        # was an O(context) copy per request per step in the scheduling loop
        # (a 128k-token request copied its whole history to then do nothing).
        # all_token_ids is materialized ONLY on the paths that emit a row.
        all_tokens: list | None = None
        n_prompt = getattr(request, "num_prompt_tokens", None)
        if n_prompt is None:  # shape-churn fallback: same bytes as before
            all_tokens = list(getattr(request, "all_token_ids", None) or [])
            n_prompt = len(all_tokens)
        n_prompt = int(n_prompt or 0)
        aligned = align_to_block_size(n_prompt, self._block_size)
        if aligned <= 0:
            return
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
            if block_ids is None:
                # No physical ids this step: KEEP the promise queued (the
                # request must surface with blocks before it can run) and
                # disclose — popping here would drop it silently.
                self._log.maybe("meta-req",
                                f"resumed load promise kept — no block ids yet req={rid}")
                return
            load_start, n_load = self._need_load_blocks[rid]  # peek; consumed below
            # Blocks computed during the resume step itself still need a
            # store; everything at/below the loaded range is present.
            store_start = max(load_start + n_load, computed // self._block_size)
            if all_tokens is None:  # a row is being emitted: NOW pay the copy
                all_tokens = list(getattr(request, "all_token_ids", None) or [])
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
            self._need_load_blocks.pop(rid, None)  # promise delivered — consume it
            return
        # Chunked-prefill continuation: store the blocks this step completes.
        if rid in self._need_load_blocks:
            # An unconsumed load promise means some counted-computed blocks
            # were never fetched OR flagged — storing them would publish the
            # garbage cache-wide under content-chained keys. Skip (a smaller
            # cache, never a wrong byte); the promise stays queued.
            self._log.maybe("meta-req",
                            f"store skipped under a pending load promise req={rid}")
            return
        if computed >= aligned:
            return  # prompt fully covered (decode steps store nothing)
        store_start = computed // self._block_size
        store_end = done // self._block_size
        if store_end <= store_start:
            return
        # A store row needs physical ids up to store_end; a shorter list
        # means tracking gapped (e.g. connector restarted mid-request) —
        # skip rather than guess (a smaller cache, never a wrong byte).
        if not block_ids or len(block_ids) < store_end:
            return
        if all_tokens is None:  # a row is being emitted: NOW pay the copy
            all_tokens = list(getattr(request, "all_token_ids", None) or [])
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
            self._chain_cache.pop(rid, None)  # worker-role instances rely on the FIFO cap
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
        """One eager pinned-allocation pass at the FIRST CUDA layout capture
        (not the first load/store): cudaHostAlloc of a multi-GiB region takes
        hundreds of ms, and paying it lazily buries the stall inside the
        first measured load — or, for the store pool, inside the first
        measured wait_for_save. Load slab sized min(staging cap,
        kvblockd_prewarm_bytes, the pipelined path's two-half reserve);
        store pool sized by _store_pool_ready
        exactly as its lazy path would (the blob stride is computable from
        the captured layout). Each part fails back to its EXISTING lazy path
        (_slab_reserve owns the load-slab latch, _store_pool_ready owns the
        pool latch), never fatal, and never blocks the other part. Measured
        durations are published at WARNING (INFO is dropped in engine-core)
        so the bench can attribute the one-time cost."""
        if self._prewarm_done:
            return
        names = sorted(self._layer_kv)
        if not names:
            return
        try:
            if not self._slab_path_ok(self._layer_kv[names[0]].device):
                # CPU backend: neither pool ever pays here — and the paged
                # tensors' device never changes mid-run, so LATCH instead of
                # re-walking the layer map on every single capture.
                self._prewarm_done = True
                return
        except Exception as e:  # noqa: BLE001 — never-raise boundary: an unreadable device means no prewarm; lazy paths keep serving
            self._log.maybe("slab", "prewarm device probe failed — lazy paths keep serving", e)
            return
        self._prewarm_done = True  # one attempt per connector lifetime
        if self._slab is None and not self._slab_disabled:
            try:
                want = min(self._staging_bytes, self._cfg.prewarm_bytes)
                # Pin what the load path will actually RESERVE, not the whole
                # staging cap: the pipelined path reserves two half-cap
                # passes (2×min(cap/2, kvblockd_pipeline_half_bytes),
                # ~512MiB at the defaults), and pinning the 2GiB cap on top
                # of the ~1GiB store pool left 1.5GiB dead.
                # kvblockd_prewarm_bytes still bounds it (explicit override);
                # layoutless/oversized-body engines keep the old cap-sized
                # behavior (the serial path may reserve up to the cap there).
                names_, _d, bpl_ = self._layout()
                body_ = bpl_ * len(names_)
                if names_ and body_ > 0:
                    hb = min(self._staging_bytes // 2,
                             self._cfg.pipeline_half_bytes) // body_
                    if hb > 0:
                        want = min(want, 2 * hb * body_)
                if want > 0:
                    t0 = time.monotonic()
                    slab = self._alloc_pinned(want)
                    slab_np = slab.numpy()
                    self._slab, self._slab_np = slab, slab_np
                    logger.warning("kvblockd pinned prewarm: %d bytes in %.0f ms",
                                   want, (time.monotonic() - t0) * 1e3)
            except Exception as e:  # noqa: BLE001 — never-raise boundary: a failed prewarm keeps the lazy slab path
                self._log.maybe("slab", "pinned prewarm failed — lazy slab path keeps serving", e)
        # Store pool: the same stall class (~1 GiB pinned at auto-size
        # defaults, ON TOP of the load slab above — budget both on a
        # RAM-tight rig), paid eagerly here so the first CUDA wait_for_save
        # is not the one measured step that eats it. _store_pool_ready owns
        # sizing, the explicitly-off case, and the failure latch; only the
        # duration line is added here.
        if not self._cfg.async_store or self._store_slab is not None:
            return
        try:
            lnames, _dtype, bytes_per_layer = self._layout()
            if not lnames:
                return
            total = BLOB_PREFIX_LEN + bytes_per_layer * len(lnames)
            t0 = time.monotonic()
            if self._store_pool_ready(total):
                logger.warning("kvblockd pinned store-pool prewarm: %d bytes in %.0f ms",
                               int(self._store_slab.numel()),
                               (time.monotonic() - t0) * 1e3)
        except Exception as e:  # noqa: BLE001 — never-raise boundary: a failed pool prewarm keeps the lazy pool path
            self._log.maybe("store-slab",
                            "store-pool prewarm failed — lazy pool path keeps serving", e)

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
        loads = [req for req in requests if req.num_load_blocks > 0]
        if not loads:
            return
        # Raise the load-priority gate for the whole pull: the kvb-store
        # drain parks (bounded) instead of contending for the wire/GIL.
        with self._sq_cond:
            self._loads_inflight += 1
            if self._loads_inflight == 1:
                # Episode edge: clear the arm so the drain parks a fresh
                # ceiling for THIS episode. The drain cannot own this reset —
                # it only samples the counter between puts, so a gate that
                # drops and re-raises inside one put() would otherwise keep
                # the previous episode's expired arm and park zero.
                self._drain_park_until = None
        try:
            for req in loads:
                try:
                    self._load_one(req)
                except Exception as e:  # noqa: BLE001 — never raise; blocks flagged as errors
                    self._log.maybe("load", f"kvblockd load failed req={req.req_id}", e)
                    self._drop_client(e)
                    self._load_errors.update(self._load_range_ids(req))
        finally:
            with self._sq_cond:
                self._loads_inflight -= 1
                self._sq_cond.notify_all()

    # Load-side delegation: the slab/scatter/pipelined load machinery lives
    # in slab_loader.SlabLoader (unit-owned). These delegates keep every
    # connector-level seam: tests monkeypatch them as instance attributes,
    # and ALL internal cross-boundary calls route through the connector so
    # those patches always intercept.
    def _load_one(self, req: KvbReqMeta) -> None:
        self._loader._load_one(req)

    def _note_path(self, path: str) -> None:
        self._loader._note_path(path)

    def _load_perblock(self, req: KvbReqMeta, names, dtype_name, bytes_per_layer,
                       total, keys, deadline: float | None = None) -> None:
        self._loader._load_perblock(req, names, dtype_name, bytes_per_layer,
                                    total, keys, deadline)

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
        return self._loader._slab_reserve(nbytes)

    def _load_slab(self, req: KvbReqMeta, names, dtype_name, bytes_per_layer,
                   total, keys, pass_blocks: int, deadline: float | None = None,
                   key_base: int = 0) -> bool:
        return self._loader._load_slab(req, names, dtype_name, bytes_per_layer,
                                       total, keys, pass_blocks, deadline, key_base)

    def _copy_stream(self, dev):
        return self._loader._copy_stream(dev)

    def _make_event(self):
        return self._loader._make_event()

    def _current_stream(self, dev):
        return self._loader._current_stream(dev)

    def _entry_fence(self, stream, dev) -> None:
        self._loader._entry_fence(stream, dev)

    def _stream_scope(self, stream):
        return SlabLoader._stream_scope(stream)

    def _prefetch_submit(self, fn):
        return self._loader._prefetch_submit(fn)

    def _pipeline_fail(self, msg: str, exc: BaseException | None) -> None:
        self._loader._pipeline_fail(msg, exc)

    def _load_pipelined(self, req: KvbReqMeta, names, dtype_name, bytes_per_layer,
                        total, keys, half_blocks: int,
                        deadline: float | None = None) -> str | None:
        return self._loader._load_pipelined(req, names, dtype_name, bytes_per_layer,
                                            total, keys, half_blocks, deadline)

    def _scratch_ring(self, dev, n_layers, bytes_per_layer):
        return self._loader._scratch_ring(dev, n_layers, bytes_per_layer)

    def _idx_staging(self, n: int):
        return self._loader._idx_staging(n)

    def _scatter_slab(self, req: KvbReqMeta, names, bytes_per_layer, statuses,
                      key_offset: int = 0, sync: bool = True,
                      slab_base: int = 0) -> bool:
        return self._loader._scatter_slab(req, names, bytes_per_layer, statuses,
                                          key_offset, sync, slab_base)

    def _debug_check_scatter(self, first_fast: tuple[int, int], names,
                             bytes_per_layer) -> None:
        self._loader._debug_check_scatter(first_fast, names, bytes_per_layer)

    def _scatter_block_from_slab(self, j: int, bid: int, names, bytes_per_layer,
                                 slab_base: int = 0) -> None:
        self._loader._scatter_block_from_slab(j, bid, names, bytes_per_layer,
                                              slab_base)

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
        flag off this is the original synchronous put loop, byte-identical.

        CUDA paged tensors take the gathered-store fast path when available:
        _stage_one issues batched gathers + async D2H into pinned slots and
        returns a deferred-enqueue plan; ONE device sync covers every plan
        BEFORE any enqueue and before this call returns. The sync is a
        correctness invariant, not tuning — a torn D2H is a wrong byte
        published cache-wide under a content-chained key, never a miss."""
        self._maybe_stats_summary()
        try:
            metadata = self._get_connector_metadata()
        except Exception:  # noqa: BLE001 — never-raise boundary: missing metadata = nothing to store this step
            return
        requests = getattr(metadata, "requests", None) or []
        plans: list[_StagePlan] = []
        for req in requests:
            if req.store_end_block > req.store_start_block:
                try:
                    if self._cfg.async_store:
                        plan = self._stage_one(req)
                        if plan is not None:
                            plans.append(plan)
                    else:
                        self._store_one(req)
                except Exception as e:  # noqa: BLE001 — never raise: a lost store is a future miss
                    self._log.maybe("store", f"kvblockd store failed req={req.req_id}", e)
                    self._drop_client(e)
        if not plans:
            return
        try:
            if self._store_sync(plans[0].dev):
                self._store_gather_fails = 0  # consecutive-failure counter
                torn = False
            else:
                # Nothing D2H'd is provably complete. Paged memory is still
                # valid (we have not returned), so rebuild every slot-backed
                # blob through the bytearray path and release the slots.
                self._store_gather_fail("gathered-store device sync failed", None)
                torn = True
            for plan in plans:
                # Per-plan armor: one plan's raise (e.g. a sick device inside
                # the rebuild's host copies) must not discard the OTHER
                # plans' staged blobs, and whatever THIS plan never handed to
                # _sq_enqueue is counted where it is lost — the disclosure
                # counters are the accounting of record.
                try:
                    if torn:
                        self._rebuild_plan(plan)
                    self._finish_stage(plan)
                except Exception as e:  # noqa: BLE001 — never raise: the plan degrades to counted drops
                    self._log.maybe("store",
                                    f"gathered-store finish failed req={plan.req_id}", e)
                    self._abandon_plan(plan)
        except Exception as e:  # noqa: BLE001 — never raise: last-resort boundary (per-plan failures were handled above)
            self._log.maybe("store", "gathered-store finish failed", e)
            for plan in plans:
                self._abandon_plan(plan)

    # ------------------------------------------------------------------
    # write-behind store queue (kvblockd_async_store)
    # ------------------------------------------------------------------
    # Store-side delegation: the write-behind queue, drain workers, slot
    # pool, and gathered-store staging live in store_queue.StoreQueue
    # (unit-owned); same seam-preserving convention as the load side.
    def _stage_one(self, req: KvbReqMeta) -> _StagePlan | None:
        return self._store_q._stage_one(req)

    def _build_block_blob(self, bid: int, names, bytes_per_layer: int,
                          prefix: bytes, total: int) -> bytearray:
        return self._store_q._build_block_blob(bid, names, bytes_per_layer,
                                               prefix, total)

    def _store_pool_ready(self, total: int) -> bool:
        return self._store_q._store_pool_ready(total)

    def _store_slot_lease(self) -> int | None:
        return self._store_q._store_slot_lease()

    def _store_gather_fail(self, msg: str, exc: BaseException | None) -> None:
        self._store_q._store_gather_fail(msg, exc)

    def _note_store_path(self, path: str) -> None:
        self._store_q._note_store_path(path)

    def _stage_gather(self, req: KvbReqMeta, names, bytes_per_layer: int,
                      total: int, prefix: bytes, keys, start: int,
                      end: int) -> _StagePlan | None:
        return self._store_q._stage_gather(req, names, bytes_per_layer, total,
                                           prefix, keys, start, end)

    def _store_sync(self, dev) -> bool:
        return self._store_q._store_sync(dev)

    def _rebuild_plan(self, plan: _StagePlan) -> None:
        self._store_q._rebuild_plan(plan)

    def _finish_stage(self, plan: _StagePlan) -> None:
        self._store_q._finish_stage(plan)

    def _abandon_plan(self, plan: _StagePlan) -> None:
        self._store_q._abandon_plan(plan)

    def _sq_enqueue(self, key: bytes, buf: bytearray | memoryview,
                    slot_id: int | None = None, rid: str = "") -> bool:
        return self._store_q._sq_enqueue(key, buf, slot_id, rid)

    def _store_thread_start(self) -> None:
        self._store_q._store_thread_start()

    def _store_drain(self, wi: int = 0) -> None:
        self._store_q._store_drain(wi)

    def _store_flush(self, timeout: float) -> int:
        return self._store_q._store_flush(timeout)

    def _store_shutdown(self) -> None:
        self._store_q._store_shutdown()

    def _store_one(self, req: KvbReqMeta) -> None:
        self._store_q._store_one(req)

    def handle_preemptions(self, *args, **kwargs) -> None:
        """Explicit NO-OP, on purpose: the write-behind queue holds OWNED
        copies staged inside wait_for_save — nothing in it references paged
        memory, so a preempted request's blocks can be freed and reused
        immediately without corrupting a queued store."""
        return

    def get_finished(self, finished_req_ids):
        return None, None  # all loads/saves are synchronous within the step

    def get_block_ids_with_load_errors(self) -> set[int]:
        errs, self._load_errors = self._load_errors, set()
        return errs


def _alias_state(owner: str, names: tuple[str, ...]) -> None:
    """Alias collaborator-owned state onto the connector, one property
    (getter + setter) per name: tests, the bench, and connector-resident
    code keep reading AND monkeypatching these as connector attributes,
    while exactly ONE storage location exists — the collaborator's. A
    read-only collaborator property (_sq, _store_thread) stays read-only
    through the alias (the setattr raises exactly as it did before)."""
    for name in names:
        def _get(self, _o=owner, _n=name):
            return getattr(getattr(self, _o), _n)

        def _set(self, value, _o=owner, _n=name):
            setattr(getattr(self, _o), _n, value)

        setattr(KvblockdConnector, name, property(_get, _set))


_alias_state("_loader", (
    "_staging_bytes", "_slab", "_slab_np", "_slab_disabled",
    "_gpu_scratch", "_gpu_scratch_key", "_scratch_torn", "_scratch_fails",
    "_chunked_disabled", "_idx_pin", "_load_stream", "_prefetch_ex",
    "_pipeline_fails", "_pipeline_disabled", "_reported_path",
    "_debug_scatter_checked",
))
_alias_state("_store_q", (
    "_store_workers", "_sqs", "_sq", "_sq_cond", "_sq_bytes", "_sq_inflight",
    "_sq_inflight_bytes", "_sq_inflight_counted", "_store_holes",
    "_store_threads", "_store_thread", "_store_stop", "_store_abort",
    "dropped_puts", "dropped_put_bytes", "failed_puts", "deduped_puts",
    "_acked_keys", "_loads_inflight", "_drain_park_until",
    "_store_slab", "_store_slab_np", "_store_slot_stride", "_store_slot_free",
    "_store_slots_total", "_store_slab_disabled", "_store_gather_fails",
    "_store_gather_disabled", "_reported_store_path",
    "_store_path_switch_logged",
))
