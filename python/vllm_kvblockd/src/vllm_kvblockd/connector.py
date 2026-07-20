"""KvblockdConnector — vLLM-native KVConnectorBase_V1 backed by kvblockd.

CHURN-WATCH: KVConnectorBase_V1 is explicitly unstable (vLLM RFC #38260
tracking); pinned per-minor, CI matrix vs last 4 releases (gate A6).
Re-verify the base.py diff before every vLLM bump.

Version assumptions (verified against vendored sources, see UPSTREAM.lock):
  - Written against vLLM v0.25.0 (tag 702f4814fe54fabff350d43cb753ae3e47c0c276),
    modeled on its ExampleConnector (ex-SharedStorageConnector).
  - Constructor is the 3-arg form (vllm_config, role, kv_cache_config) —
    identical across the A6 window v0.22.1..v0.25.x, and MANDATORY at v0.25
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
is correction C-14.

NEVER RAISE on the serving path: every failure degrades to a cache miss
(LMCache #2204 posture). The only boot-time exception is DeterminismError —
refusing to start beats a fleet that silently never shares cache.
"""

from __future__ import annotations

import logging
import struct
import threading
import time
from dataclasses import dataclass, field

from kvblockd import protocol as kp
from kvblockd.client import Client

from .config import AdapterConfig, block_chain_keys, chain_seed, require_pinned_hashseed

logger = logging.getLogger("vllm_kvblockd")

try:  # vLLM absent (unit tests, A6 lmcache cells) -> importable fallback.
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # type: ignore
        KVConnectorBase_V1 as _Base,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # type: ignore
        KVConnectorMetadata as _MetaBase,
    )

    _HAS_VLLM = True
except Exception:  # pragma: no cover - exercised by the no-vllm test env
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


def _torch():  # lazy so the A6 import check can load us without torch
    import torch

    return torch


# --- 32B per-blob layout prefix (mirrors lmcache_kvblockd.meta's style) ---
# The prefix is drift armor: a blob whose declared layout does not match the
# LIVE engine's layout is treated as a miss, never scattered into the paged
# buffer. Config changes already diverge the fingerprint (hence the key); this
# catches the residue (e.g. an attention-backend layout flip within a config).
BLOB_MAGIC = b"KVN1"
BLOB_VERSION = 1
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
    interval (per-instance — mirrors the W5 connector's discipline)."""

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
    cache_salt: str | None      # C-14: folded into the chain seed
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
        # req_id -> full group-0 block list, accumulated across steps: the
        # SchedulerOutput only carries NEW block ids for cached requests, and
        # the Request object exposes none — chunked-prefill continuation
        # stores need the request's whole list.
        self._blocks: dict[str, list[int]] = {}

        # Worker-side state.
        self._layer_kv: dict[str, object] = {}        # layer_name -> paged KV tensor
        self._load_errors: set[int] = set()

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
                    )
                except Exception:
                    self._next_dial = time.monotonic() + _REDIAL_BACKOFF_S
                    raise
            return self._client

    def shutdown(self):
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
        except Exception:
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
        req_id-keyed local-hit stash, overwritten in place); sync (async=False).
        """
        rid = getattr(request, "request_id", None)
        if rid is not None:
            # num_computed_tokens here IS the local prefix-cache hit; the
            # Request object still reads 0 at update_state_after_alloc time.
            self._local_hit_tokens[rid] = int(num_computed_tokens or 0)
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
        except Exception as e:  # never raise: a failed lookup is a miss
            self._log.maybe("lookup", "kvblockd BATCH_EXISTS failed (treated as miss)", e)
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
        except Exception as e:  # never raise: an empty meta = a no-op step
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
        except Exception as e:
            self._log.maybe("layers", "capturing paged KV tensors failed", e)

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
        except Exception:
            return
        requests = getattr(metadata, "requests", None) or []
        for req in requests:
            if req.num_load_blocks > 0:
                try:
                    self._load_one(req)
                except Exception as e:  # never raise; blocks flagged as errors
                    self._log.maybe("load", f"kvblockd load failed req={req.req_id}", e)
                    self._load_errors.update(self._load_range_ids(req))

    def _load_one(self, req: KvbReqMeta) -> None:
        torch = _torch()
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

        statuses = self._ensure().batch_get_scatter(keys, BLOB_PREFIX_LEN, alloc)
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
                except Exception as e:
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
        try:
            metadata = self._get_connector_metadata()
        except Exception:
            return
        requests = getattr(metadata, "requests", None) or []
        for req in requests:
            if req.store_end_block > req.store_start_block:
                try:
                    self._store_one(req)
                except Exception as e:  # never raise: a lost store is a future miss
                    self._log.maybe("store", f"kvblockd store failed req={req.req_id}", e)

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
