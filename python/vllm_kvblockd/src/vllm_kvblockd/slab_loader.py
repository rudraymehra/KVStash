"""SlabLoader — the pinned-slab / scatter / pipelined LOAD machinery of
KvblockdConnector (CUDA paged tensors), extracted verbatim from connector.py.

Unit ownership: the pinned staging slab, the GPU scratch ring, the pinned
index staging, the dedicated copy stream + single-worker prefetch thread,
and every latch/failure counter of the load fast paths live HERE. The
composing connector aliases this state back onto itself (its test/bench
seams are unchanged) and delegates the methods.

Cross-boundary calls route back through the composing connector
(``self._c``) ON PURPOSE, not as style: tests and operators monkeypatch
those seams as CONNECTOR instance attributes (e.g. ``conn._alloc_pinned``,
``conn._scatter_slab``), and a bound-method call inside this class would
silently bypass the patch. Behavior is byte-for-byte the pre-extraction
connector's.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import logging
import os
import time
from typing import TYPE_CHECKING

from kvblockd import protocol as kp

from .connector import (
    BLOB_PREFIX_LEN,
    CODEC_RAW,
    BlobError,
    _torch,
    decode_blob_prefix,
)

if TYPE_CHECKING:  # import cycle by design: connector.py imports this module
    from .connector import KvblockdConnector, KvbReqMeta

logger = logging.getLogger("vllm_kvblockd")

# Chunked H2D scatter: blocks per chunk and the GPU scratch depth. One chunk
# = one non_blocking H2D of a contiguous pinned-slab region + one index_copy_
# per layer, replacing chunk×layers synchronous pageable copies. Depth 1 is
# deliberate (reviewer-verified): every chunk runs on the SAME stream, so the
# copy_ into the scratch buffer cannot start until the previous chunk's
# index_copy_ reads finished — a second buffer bought no overlap, only VRAM.
# That same-stream argument now carries THREE users of the one ring buffer:
# serial-slab chunks (engine's current stream), pipelined-slab chunks (the
# dedicated copy stream — BOTH slab halves' chunks run on it, so half
# alternation never overlaps ring reuse), and the gathered-store path
# (current stream). Cross-path safety is an EXIT-SYNC invariant, not a
# stream one: each path synchronizes the stream it issued ring work on
# before returning to the engine (_scatter_slab's trailing sync, the
# pipelined load's final copy-stream event, _store_sync in wait_for_save),
# so no two paths ever have ring work in flight at once.
_SCATTER_CHUNK = 64
_SCRATCH_RING = 1

# Consecutive chunked-scatter setup failures (scratch alloc / paged view)
# before the chunked path latches OFF for the connector's lifetime — the
# per-block-from-slab copies keep serving loads either way.
_SCRATCH_MAX_FAILS = 3

# Pipelined-load half sizing lives in config (kvblockd_pipeline_half_bytes,
# default 256MiB): passes must be small enough that pass p+1's wire drain
# genuinely overlaps pass p's H2D+scatter (one 2GiB pass has nothing to
# overlap with), yet big enough to amortize per-pass costs. The load path
# reserves TWO halves, so the pinned footprint of the pipeline is
# 2×min(staging/2, the knob) — what _maybe_prewarm pins.


class SlabLoader:
    """The connector's slab/scatter load lifecycle, unit-owned. The scratch
    ring it owns is SHARED with the store side (the gathered-store staging
    borrows it through the connector's _scratch_ring seam); the poison
    protocol (_scratch_torn) and the exit-sync invariant — each path
    synchronizes the stream it issued ring work on before returning — are
    documented on the methods below, moved verbatim with their code."""

    def __init__(self, connector: KvblockdConnector):
        # The composing connector: every cross-boundary call routes through
        # it so connector-level monkeypatch seams always intercept.
        self._c = connector
        # Pinned staging slab (CUDA loads only): lazily allocated on the first
        # CUDA-device load, grown geometrically UP TO the configured cap
        # (kvblockd_staging_bytes), REUSED across loads — never freed
        # per-load. Loads bigger than the cap drain through the slab in
        # cap-sized passes. Slots are disjoint by pass-local block index,
        # which is what makes the client's concurrent drain threads safe.
        self._staging_bytes = self._c._cfg.staging_bytes
        self._slab = None                 # 1-D pinned uint8 torch tensor
        self._slab_np = None              # numpy view of the slab (memoryview source)
        self._slab_disabled = False       # first-pin failure -> permanent per-block fallback
        # GPU scratch for the chunked scatter: _SCRATCH_RING × [chunk,
        # n_layers, bytes_per_layer] uint8, cached per (device, layout).
        self._gpu_scratch = None
        self._gpu_scratch_key = None
        # A failed pipelined exit sync means ring work may STILL be in
        # flight on the copy stream; until a drain proves otherwise, no
        # path may hand out ring memory (see _scratch_ring).
        self._scratch_torn = False
        self._scratch_fails = 0           # consecutive chunked-setup failures
        self._chunked_disabled = False    # latched after _SCRATCH_MAX_FAILS in a row
        # Pinned int64 staging for chunk-index uploads (beside _gpu_scratch):
        # torch.tensor(list, device=cuda) is a BLOCKING pageable H2D that
        # would puncture the pipeline from inside every chunk.
        self._idx_pin = None
        # Pipelined double-buffered load (CUDA): dedicated copy stream +
        # single-worker prefetch thread, both lazy; setup failures (never
        # drain/network failures — those degrade one load without latching)
        # count toward the same 3-in-a-row latch as the scratch ring.
        self._load_stream = None
        self._prefetch_ex: concurrent.futures.ThreadPoolExecutor | None = None
        self._pipeline_fails = 0          # consecutive pipelined-SETUP failures
        self._pipeline_disabled = False   # latched after _SCRATCH_MAX_FAILS in a row
        # Machine-readable path attribution: one INFO line on the first
        # completed load, one more per mid-run switch.
        self._reported_path = None
        self._debug_scatter_checked = False  # KVBLOCKD_DEBUG_SCATTER_CHECK=1, once

    def _load_one(self, req: KvbReqMeta) -> None:
        names, dtype_name, bytes_per_layer = self._c._layout()
        if not names:
            self._c._load_errors.update(self._c._load_range_ids(req))
            return
        total = BLOB_PREFIX_LEN + bytes_per_layer * len(names)
        seed = self._c._seed(req.cache_salt, req.mm_ids, req.lora_name)
        start = req.load_start_block
        end = start + req.num_load_blocks
        keys = self._c._chain_keys(req.req_id, seed, req.token_ids)[start:end]
        # A promised block with no derivable key (token list shorter than the
        # promise) can never be filled — flag it now, don't drop it silently.
        for blk in range(start + len(keys), end):
            if blk < len(req.block_ids):
                self._c._load_errors.add(req.block_ids[blk])

        body_len = total - BLOB_PREFIX_LEN
        dev = self._c._layer_kv[names[0]].device
        # Overall per-load deadline (kvblockd_load_deadline_s): op_timeout
        # bounds each recv, this bounds the WHOLE load — a daemon trickling
        # bytes passes every per-recv check forever, and the engine counted
        # these blocks computed, so the only safe degrade is: abandon the
        # remaining shards, flag the unfilled bids, recompute. The budget is
        # token-scaled: min(cap, base + per_block_s * n_load_blocks) — at the
        # flat-compat defaults (per_block=0, cap unset) it is exactly base.
        deadline = None
        if self._c._cfg.load_deadline_s > 0:
            budget = (self._c._cfg.load_deadline_s
                      + self._c._cfg.load_deadline_per_block_s * req.num_load_blocks)
            if self._c._cfg.load_deadline_cap_s > 0:
                budget = min(budget, self._c._cfg.load_deadline_cap_s)
            deadline = time.monotonic() + budget
        t0 = time.monotonic()
        errs_before = len(self._c._load_errors)
        path = None
        took_slab = False
        if keys and self._c._slab_path_ok(dev):
            if not self._pipeline_disabled and body_len > 0:
                # Pipelined double-buffered halves: passes small enough that
                # the next pass's wire drain overlaps this pass's H2D+scatter.
                hb = min(min(self._staging_bytes // 2,
                             self._c._cfg.pipeline_half_bytes) // body_len,
                         len(keys))
                if hb == 0 and body_len > self._c._cfg.pipeline_half_bytes:
                    # A knob below one blob body silently serializes every
                    # load — the boot-time >0 check can't see body_len, so
                    # the disclosure lands here (rate-limited, not a latch).
                    self._c._log.maybe(
                        "pipeline-half",
                        f"kvblockd_pipeline_half_bytes="
                        f"{self._c._cfg.pipeline_half_bytes} < one blob body "
                        f"({body_len}B) — pipelined path disabled, serial "
                        "slab passes serve loads")
                need = (2 * hb if len(keys) > hb else hb) * body_len
                if hb > 0 and self._c._slab_reserve(need):
                    path = self._c._load_pipelined(req, names, dtype_name, bytes_per_layer,
                                                total, keys, hb, deadline)
                    took_slab = path is not None
            if not took_slab:
                # Serial cap-sized passes (pipeline latched off, setup failed,
                # or one body outgrows a half): the slab never grows past the
                # configured cap; bigger loads drain through it pass by pass.
                cap_blocks = self._staging_bytes // body_len if body_len > 0 else 0
                pass_blocks = min(len(keys), cap_blocks)
                if pass_blocks > 0 and self._c._slab_reserve(pass_blocks * body_len):
                    took_slab = True
                    used_ring = self._c._load_slab(req, names, dtype_name, bytes_per_layer,
                                                total, keys, pass_blocks, deadline)
                    path = "chunked-slab" if used_ring else "per-block"
        if keys and not took_slab:
            self._c._load_perblock(req, names, dtype_name, bytes_per_layer, total, keys,
                                deadline)
            path = "per-block"
        if keys:
            self._c._note_path(path)
            elapsed = time.monotonic() - t0
            self._c.stats.bump("load_count")
            self._c.stats.bump("load_time_s", elapsed)
            # Estimator feed: promised minus newly-flagged approximates the
            # delivered blocks (a bid flagged twice undercounts errors, i.e.
            # overstates throughput — the admit-biased direction).
            self._c._observe_load(len(keys) - (len(self._c._load_errors) - errs_before),
                               total, elapsed)

    def _note_path(self, path: str) -> None:
        """One machine-readable line on the first completed load — the bench
        rig takes the LAST match to attribute measured numbers to the path
        that served the run's tail — plus one line per DISTINCT switch (three
        paths exist now, so a single one-shot switch line could leave the
        last match naming a path that stopped serving). Steady state still
        logs nothing.

        WARNING level on purpose: this logger lives outside vLLM's logging
        config, and in the engine-core process an unconfigured logger drops
        INFO under the root default — the certification run recorded 'path
        unattributed' exactly that way. A line per switch is not noise."""
        if self._reported_path is None:
            self._reported_path = path
            logger.warning("kvblockd load path: %s", path)
        elif path != self._reported_path:
            logger.warning("kvblockd load path: %s (switched from %s mid-run)",
                           path, self._reported_path)
            self._reported_path = path

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
                d, n_layers, tpb, bpl, tot, codec = decode_blob_prefix(prefix)
            except BlobError:
                return None
            if (codec != CODEC_RAW  # this build decodes only raw bodies —
                    # a codec blob is a clean miss until its serde (and the
                    # pre-registered quality gate) land
                    or d != dtype_name or n_layers != len(names) or tpb != self._c._block_size
                    or bpl != bytes_per_layer or tot != total
                    or body_len != total - BLOB_PREFIX_LEN):
                return None  # codec/layout drift -> miss, never a corrupt scatter
            buf = torch.empty(body_len, dtype=torch.uint8)
            staged[idx] = buf
            return memoryview(buf.numpy())

        statuses = self._c._ensure().batch_get_scatter(keys, BLOB_PREFIX_LEN, alloc,
                                                    deadline=deadline)
        for j, st in enumerate(statuses):
            blk = start + j
            bid = req.block_ids[blk] if blk < len(req.block_ids) else None
            if st != kp.Status.OK or j not in staged:
                if bid is not None:
                    self._c._load_errors.add(bid)
                continue
            if bid is None:
                continue
            buf = staged[j]
            for li, name in enumerate(names):
                dst = self._c._layer_kv[name][bid]
                src = buf[li * bytes_per_layer : (li + 1) * bytes_per_layer]
                try:
                    dst.copy_(src.view(dst.dtype).reshape(dst.shape))
                except Exception as e:  # noqa: BLE001 — never-raise boundary: a failed scatter marks the block errored, not the engine
                    self._c._log.maybe("scatter", f"scatter into {name} failed", e)
                    self._c._load_errors.add(bid)
                    break

    # ------------------------------------------------------------------
    # pinned-slab load path (CUDA paged tensors)
    # ------------------------------------------------------------------
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
            slab = self._c._alloc_pinned(want)
            slab_np = slab.numpy()
        except Exception as e:  # noqa: BLE001 — never-raise boundary: cudaHostAlloc failure degrades to the per-block path
            if self._slab is None:
                self._c._log.maybe("slab", "pinned slab allocation failed — per-block fallback", e)
                self._slab_disabled = True
            else:
                self._c._log.maybe("slab", "pinned slab GROWTH failed — keeping the "
                                        "existing slab; per-block fallback for this load", e)
            return False
        self._slab, self._slab_np = slab, slab_np
        return True

    def _load_slab(self, req: KvbReqMeta, names, dtype_name, bytes_per_layer,
                   total, keys, pass_blocks: int, deadline: float | None = None,
                   key_base: int = 0) -> bool:
        """Slab-staged load: the client drains block bodies straight into
        disjoint pinned-slab slots (the layout gate runs in alloc BEFORE any
        body byte is accepted, exactly like the per-block path), then
        _scatter_slab moves them to the GPU in chunked batches. Loads bigger
        than pass_blocks drain through the slab in pass_blocks-sized passes —
        _scatter_slab's trailing stream synchronize makes reusing the slots
        for the next pass safe, and key_offset keeps every pass's statuses
        mapped to the right GLOBAL block ids. key_base offsets that mapping
        when `keys` is a TAIL of the load's key list (the pipelined path's
        mid-load serial fallback). Returns whether any pass used the chunked
        fast path (path attribution)."""
        body_len = total - BLOB_PREFIX_LEN
        slab_np = self._slab_np
        used_ring = False
        for p0 in range(0, len(keys), pass_blocks):
            if deadline is not None and time.monotonic() > deadline:
                # Load deadline blown between passes: flag every remaining
                # promised bid and stop — the scheduler counted them computed,
                # so an unflagged unfilled block is silent garbage.
                for blk in range(req.load_start_block + key_base + p0,
                                 req.load_start_block + key_base + len(keys)):
                    if blk < len(req.block_ids):
                        self._c._load_errors.add(req.block_ids[blk])
                self._c._log.maybe("load-deadline",
                                f"load deadline exceeded — abandoning {len(keys) - p0} "
                                f"remaining blocks (recompute) req={req.req_id}")
                self._c.stats.bump("load_deadline_aborts")
                break
            sub = keys[p0:p0 + pass_blocks]

            def alloc(idx, prefix, blen):
                # Runs on the client's concurrent drain threads: slots are
                # disjoint by (pass-local) idx and nothing else is mutated
                # here — thread-safe by construction. idx is 0-based within
                # this batch_get_scatter call, so slots never exceed the pass.
                try:
                    d, n_layers, tpb, bpl, tot, codec = decode_blob_prefix(prefix)
                except BlobError:
                    return None
                # codec gate first: slot sizing below assumes the raw fixed
                # stride (per-codec max_body_len sizing lands WITH a serde).
                if (codec != CODEC_RAW
                        or d != dtype_name or n_layers != len(names) or tpb != self._c._block_size
                        or bpl != bytes_per_layer or tot != total or blen != body_len):
                    return None  # codec/layout drift -> miss, never a corrupt scatter
                off = idx * body_len
                return memoryview(slab_np[off:off + blen])

            statuses = self._c._ensure().batch_get_scatter(sub, BLOB_PREFIX_LEN, alloc,
                                                        deadline=deadline)
            used_ring |= self._c._scatter_slab(req, names, bytes_per_layer, statuses,
                                            key_offset=key_base + p0)
        return used_ring

    # ------------------------------------------------------------------
    # pipelined load path (double-buffered slab halves, CUDA)
    # ------------------------------------------------------------------
    def _copy_stream(self, dev):
        """The dedicated load stream (cached; loads are engine-thread-serial,
        so one is enough). None off-CUDA: CPU copies complete synchronously,
        so no stream or fence exists to wait on. Test seam."""
        if getattr(dev, "type", "") != "cuda":
            return None
        if self._load_stream is None:
            torch = _torch()
            self._load_stream = torch.cuda.Stream(device=dev)
        return self._load_stream

    def _make_event(self):
        """One fence event (slab-half reuse + the final exit sync). None when
        no copy stream exists — there is then nothing asynchronous to fence.
        Test seam: the CPU suites substitute recording stand-ins."""
        if self._load_stream is None:
            return None
        torch = _torch()
        return torch.cuda.Event()

    def _current_stream(self, dev):
        """The engine's compute stream. Test seam."""
        return _torch().cuda.current_stream(dev)

    def _entry_fence(self, stream, dev) -> None:
        """Order the copy stream after the engine's compute stream before
        any paged write. INVARIANT: a paged block freed by a finishing
        request and reallocated to this load may still be READ by an
        in-flight prior-step kernel on the compute stream; an unordered
        index_copy_ over it is a silent wrong byte no flag ever covers.
        The serial path gets this for free by issuing on the compute
        stream itself. Device-side wait (~µs) — the host never stalls.
        No stream, nothing asynchronous to order."""
        if stream is None:
            return
        ev = self._c._make_event()
        if ev is None:
            return
        ev.record(self._c._current_stream(dev))
        stream.wait_event(ev)

    @staticmethod
    def _stream_scope(stream):
        if stream is None:
            return contextlib.nullcontext()
        return _torch().cuda.stream(stream)

    def _prefetch_submit(self, fn):
        """Submit one drain to the persistent single-worker prefetch thread
        (lazily started, joined in shutdown). ONE worker on purpose: exactly
        one drain in flight at a time is what keeps the client's 4-conn pool
        math and the store drain's load-priority gate unchanged."""
        if self._prefetch_ex is None:
            self._prefetch_ex = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="kvb-load-prefetch")
        return self._prefetch_ex.submit(fn)

    def _pipeline_fail(self, msg: str, exc: BaseException | None) -> None:
        """Consecutive pipelined-SETUP failure accounting (mirrors
        _scratch_fails). Drain/network failures NEVER land here: latching on
        a transient server blip would disable the pipeline for the process
        lifetime, and a drain raise already degrades that one load serially
        without spending the latch."""
        self._pipeline_fails += 1
        if self._pipeline_fails >= _SCRATCH_MAX_FAILS:
            self._pipeline_disabled = True
            # Own rate-limit key: the per-load fallback line below fires
            # first and would otherwise suppress this one-shot disclosure.
            self._c._log.maybe(
                "pipeline-latch",
                f"{msg} {self._pipeline_fails}x in a row — latched OFF for this "
                "connector's lifetime (serial slab passes)", exc)
        else:
            self._c._log.maybe("pipeline", f"{msg} — serial slab fallback for this load", exc)

    def _load_pipelined(self, req: KvbReqMeta, names, dtype_name, bytes_per_layer,
                        total, keys, half_blocks: int,
                        deadline: float | None = None) -> str | None:
        """Double-buffered slab load: a single-worker prefetch thread drains
        pass p+1 into slab half (p+1)%2 over the wire WHILE the engine thread
        scatters half p on the dedicated copy stream — the wire and PCIe
        legs, strictly additive in the serial path, now overlap. The caller
        reserved BOTH halves; the client's alloc contract already writes
        disjoint slots thread-safely, so no client change is involved.

        Fencing: at ENTRY the copy stream waits on an event recorded on
        the engine's compute stream (_entry_fence) — a paged block freed
        by a finishing request and reallocated to this load may still be
        read by an in-flight prior-step kernel, and scattering into it
        from an unordered stream would be a silent wrong byte
        (write-after-read; the serial path is ordered for free by issuing
        on the compute stream). Every LATER event is a copy-stream event:
        half h's slots are
        re-drained only after free_ev[h] — recorded after the half's last
        chunk — synchronized; the same fence covers the pinned idx staging
        the next pass's scatter refills, which is why it runs before EVERY
        scatter, last pass included. Both halves' chunks share the ONE copy
        stream, so the depth-1 scratch ring stays safe by same-stream
        ordering (the _SCRATCH_RING comment's argument, now on this stream).
        One final copy-stream event before returning keeps the synchronous
        vLLM contract, makes the paged writes visible engine-wide, and is
        this path's exit-sync for the ring handoff to _stage_gather.

        Degrades, never raises: non-OK statuses flag per block (in
        _scatter_slab); an expired deadline scatters the half already
        drained-and-verified and flags ONLY the undrained remainder; a drain
        raise flags that pass's promised sub-range and hands the undrained
        tail to the serial slab path WITHOUT touching the latch; a failed
        final sync flags every promised bid (nothing is provably loaded).
        Returns the path stamp — attributed to whichever lane moved the
        majority of the load's blocks — or None when SETUP failed (counted
        toward the pipeline latch; the caller runs the serial path)."""
        body_len = total - BLOB_PREFIX_LEN
        n = len(keys)
        slab_np = self._slab_np
        half_off = half_blocks * body_len
        start = req.load_start_block

        def drain(p0: int, sub, half: int):
            def alloc(idx, prefix, blen):
                # Client drain threads: slots disjoint by (half, pass-local
                # idx), nothing else mutated — thread-safe by construction.
                try:
                    d, n_layers, tpb, bpl, tot, codec = decode_blob_prefix(prefix)
                except BlobError:
                    return None
                # codec gate first: half/slot strides assume the raw fixed
                # body_len (per-codec max_body_len sizing lands WITH a serde).
                if (codec != CODEC_RAW
                        or d != dtype_name or n_layers != len(names) or tpb != self._c._block_size
                        or bpl != bytes_per_layer or tot != total or blen != body_len):
                    return None  # codec/layout drift -> miss, never a corrupt scatter
                off = half * half_off + idx * body_len
                return memoryview(slab_np[off:off + blen])

            return self._c._ensure().batch_get_scatter(sub, BLOB_PREFIX_LEN, alloc,
                                                    deadline=deadline)

        def submit(i: int):
            p0 = i * half_blocks
            sub = keys[p0:p0 + half_blocks]
            return self._c._prefetch_submit(lambda: drain(p0, sub, i % 2))

        def flag_range(a: int, b: int) -> None:
            for blk in range(start + a, start + min(b, n)):
                if blk < len(req.block_ids):
                    self._c._load_errors.add(req.block_ids[blk])

        try:
            dev = self._c._layer_kv[names[0]].device
            stream = self._c._copy_stream(dev)
            # Unprovable ordering == setup failure: fall back to the serial
            # path, which is compute-stream-ordered by construction.
            self._c._entry_fence(stream, dev)
            fut = submit(0)
        except Exception as e:  # noqa: BLE001 — never-raise boundary: setup failure degrades to the serial slab path
            self._c._pipeline_fail("pipelined-load setup failed", e)
            return None
        self._pipeline_fails = 0
        n_passes = -(-n // half_blocks)
        free_ev = [None, None]
        pipelined_blocks = 0
        serial_from: int | None = None
        try:
            for i in range(n_passes):
                p0 = i * half_blocks
                p1 = min(p0 + half_blocks, n)
                try:
                    statuses = fut.result()
                except Exception as e:  # noqa: BLE001 — drain failure: flag this pass, serve the tail serially, no latch (transient network is not a setup fault)
                    self._c._log.maybe("load", f"pipelined drain failed req={req.req_id} "
                                            "— serial fallback for the remainder", e)
                    self._c._drop_client(e)
                    flag_range(p0, p1)
                    serial_from = p1
                    fut = None
                    break
                fut = None
                # The wall-clock deadline bounds WIRE time (the client threads
                # it into every recv); a half that drained in budget was also
                # xxh3-verified, so it is scattered even at the deadline —
                # only the UNDRAINED remainder is abandoned and flagged.
                expired = (deadline is not None and i + 1 < n_passes
                           and time.monotonic() > deadline)
                # Fence before EVERY scatter: pass i-1's event guards both
                # the half pass i+1 will overwrite and the pinned idx slices
                # this pass's scatter refills. It runs HERE, on the engine
                # thread, on purpose: it is ~free in steady state (the event
                # was recorded one full wire drain ago), and moving it into
                # the drain job was measured-refuted as a perf lever — the
                # submit below already precedes the scatter, so the wire's
                # restart never waited on the GPU in the first place.
                ev = free_ev[(i + 1) % 2]
                if ev is not None:
                    ev.synchronize()
                    free_ev[(i + 1) % 2] = None
                if i + 1 < n_passes and not expired:
                    fut = submit(i + 1)
                with self._c._stream_scope(stream):
                    self._c._scatter_slab(req, names, bytes_per_layer, statuses,
                                       key_offset=p0, sync=False,
                                       slab_base=(i % 2) * half_off)
                pipelined_blocks += p1 - p0
                ev = self._c._make_event()
                if ev is not None:
                    ev.record(stream)
                    free_ev[i % 2] = ev
                if expired:
                    flag_range(p1, n)
                    self._c._log.maybe("load-deadline",
                                    f"load deadline exceeded — abandoning {n - p1} "
                                    f"remaining blocks (recompute) req={req.req_id}")
                    self._c.stats.bump("load_deadline_aborts")
                    break
        except Exception as e:  # noqa: BLE001 — never-raise boundary: engine-side raise (event/submit) flags the whole promise (licensed superset)
            self._c._log.maybe("load", f"pipelined load failed mid-run req={req.req_id}", e)
            self._c._drop_client(e)
            self._c._load_errors.update(self._c._load_range_ids(req))
            serial_from = None
            if fut is not None:
                # An unconsumed drain keeps writing slab slots a FUTURE load
                # would reuse; it is deadline/op_timeout-bounded, so waiting
                # it out here is the bounded, safe option.
                with contextlib.suppress(Exception):
                    fut.result()
        # Exit sync — the copy stream's OWN event, not the current stream's:
        # slot-reuse for the next load, paged-write visibility for the
        # forward pass, and the scratch-ring handoff to the store path all
        # hang off this one fence.
        try:
            ev = self._c._make_event()
            if ev is not None:
                ev.record(stream)
                ev.synchronize()
        except Exception as e:  # noqa: BLE001 — never-raise boundary: without the fence nothing is provably loaded
            self._c._log.maybe("load", "pipelined final sync failed — flagging every "
                                    "promised block", e)
            self._c._load_errors.update(self._c._load_range_ids(req))
            # Ring work may STILL be in flight on the copy stream, and the
            # failed fence was the only thing that could prove otherwise.
            # Latch the pipeline and poison the shared scratch ring: every
            # consumer (serial chunks, _stage_gather) degrades until
            # _scratch_ring re-creates it behind a PROVEN copy-stream drain
            # — otherwise the store path could publish torn bytes
            # cache-wide under a key whose xxh3 was computed over the tear.
            self._pipeline_disabled = True
            self._gpu_scratch, self._gpu_scratch_key = None, None
            self._scratch_torn = True
            return "pipelined-slab"
        serial_blocks = 0
        serial_used_ring = False
        if serial_from is not None and serial_from < n:
            rest = keys[serial_from:]
            cap = self._slab.numel() // body_len if body_len > 0 else 0
            pass_blocks = min(len(rest), cap)
            try:
                if pass_blocks > 0:
                    serial_used_ring = self._c._load_slab(req, names, dtype_name,
                                                       bytes_per_layer, total, rest,
                                                       pass_blocks, deadline,
                                                       key_base=serial_from)
                    serial_blocks = len(rest)
                else:  # unreachable (the slab holds two halves) — flag, don't drop
                    flag_range(serial_from, n)
            except Exception as e:  # noqa: BLE001 — never-raise boundary: a connection-class drain raise armed the breaker, so the remainder's redial is suppressed — flagged misses, not a raise
                self._c._log.maybe("load", f"serial remainder failed req={req.req_id}", e)
                self._c._drop_client(e)
                serial_blocks = 0
                flag_range(serial_from, n)  # superset of what landed — licensed
        if serial_blocks > pipelined_blocks:
            return "chunked-slab" if serial_used_ring else "per-block"
        return "pipelined-slab"

    def _scratch_ring(self, dev, n_layers, bytes_per_layer):
        """The GPU scratch ring (2 × [chunk, n_layers, bytes_per_layer] uint8),
        cached per (device, layout) and reused across loads."""
        if self._scratch_torn:
            # A failed pipelined exit sync dropped the old ring with work
            # possibly in flight; re-allocating before the copy stream
            # provably drained could hand the allocator-recycled bytes to a
            # new ring mid-write. A raise here lands in the callers'
            # setup-failure ladders (per-block / bytearray staging).
            ev = self._c._make_event()
            if ev is not None:
                ev.record(self._load_stream)
                ev.synchronize()
            self._scratch_torn = False
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

    def _idx_staging(self, n: int):
        """Pinned int64 staging for the chunk-index uploads (>= n entries,
        cached, grown geometrically like the slab). Chunks slice it at their
        pass-local offset, so slices are disjoint within a pass; the caller's
        pass fence (trailing sync / per-half event) covers reuse across
        passes exactly as it covers the slab slots the indices scatter."""
        if self._idx_pin is None or self._idx_pin.numel() < n:
            torch = _torch()
            want = max(n, _SCATTER_CHUNK)
            if self._idx_pin is not None:
                want = max(want, 2 * self._idx_pin.numel())
            self._idx_pin = self._c._alloc_pinned(want * 8).view(torch.int64)
        return self._idx_pin

    def _scatter_slab(self, req: KvbReqMeta, names, bytes_per_layer, statuses,
                      key_offset: int = 0, sync: bool = True,
                      slab_base: int = 0) -> bool:
        """Chunked batched H2D scatter from the slab. A chunk whose statuses
        are ALL OK takes the fast path: ONE non_blocking H2D of the contiguous
        slab region into the scratch buffer, then per layer one index_copy_
        into the paged buffer viewed as uint8 rows (bid index built only from
        status-OK blocks — in the fast path that is the whole chunk). Any
        chunk containing a non-OK/missing block falls back to per-block copies
        for THAT chunk only. Never raises: any failure flags the affected
        block ids (chunk-superset flagging allowed) and degrades. One stream
        synchronize at the end — the load is synchronous by contract, and the
        sync is what makes reusing the slab for a next pass safe. sync=False
        is the PIPELINED caller only: it runs this on its copy stream and
        owns both fences itself (per-half events for slot reuse, one final
        copy-stream event before returning). key_offset maps this pass's
        statuses onto the request's global block range; slab_base is the byte
        offset of this pass's staging region (0 for the serial path; a
        slab-half base for the pipelined one — pass-local index j lives at
        slab_base + j*body_len). Returns whether the chunked fast path was
        available (path attribution)."""
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
                    self._c._load_errors.add(bid)
                continue
            bids[j] = bid
        dev = self._c._layer_kv[names[0]].device
        paged_u8 = None
        ring = None
        idx_pin = None
        idx_prefilled = False
        if not self._chunked_disabled:
            try:
                paged_u8 = {}
                for name in names:
                    t = self._c._layer_kv[name]
                    # view, NEVER reshape: view aliases or raises, and the
                    # raise lands here -> per-block fallback. reshape silently
                    # COPIES a non-contiguous paged tensor, so index_copy_
                    # would write into a temporary and every byte would be
                    # silently lost (the refuter-verified BLOCKER).
                    paged_u8[name] = t.view(torch.uint8).view(t.shape[0], -1)
                ring = self._c._scratch_ring(dev, n_layers, bytes_per_layer)
                idx_pin = self._c._idx_staging(n)
                if all(b is not None for b in bids):
                    # All-OK pass: ONE list→tensor conversion + pinned write
                    # for the whole pass instead of one per 64-block chunk —
                    # the chunk loop below slices what this prefilled. Safe
                    # exactly when the per-chunk writes were: the caller's
                    # pre-scatter fence (serial trailing sync / the pipelined
                    # per-half event) proves the previous pass's uploads from
                    # these slices are done before this host write.
                    idx_pin[:n].copy_(torch.tensor(bids, dtype=torch.long))
                    idx_prefilled = True
                self._scratch_fails = 0
            except Exception as e:  # noqa: BLE001 — never-raise boundary: non-viewable layout / scratch OOM degrades to per-block copies
                ring = None
                self._scratch_fails += 1
                if self._scratch_fails >= _SCRATCH_MAX_FAILS:
                    self._chunked_disabled = True
                    self._c._log.maybe(
                        "scatter",
                        f"chunked-scatter setup failed {self._scratch_fails}x in a row "
                        "— latched OFF for this connector's lifetime (per-block copies)", e)
                else:
                    self._c._log.maybe("scatter", "chunked-scatter setup failed — per-block fallback", e)
        first_fast: tuple[int, int] | None = None  # (slab slot j, bid) for the debug check
        for c0 in range(0, n, _SCATTER_CHUNK):
            c1 = min(c0 + _SCATTER_CHUNK, n)
            chunk_bids = bids[c0:c1]
            if ring is not None and all(b is not None for b in chunk_bids):
                try:
                    nblk = c1 - c0
                    src = self._slab[slab_base + c0 * body_len :
                                     slab_base + c1 * body_len].view(
                        nblk, n_layers, bytes_per_layer)
                    scratch = ring[0]
                    scratch[:nblk].copy_(src, non_blocking=True)
                    # Pinned staging slice [c0:c1) — disjoint per chunk within
                    # a pass, so an earlier chunk's still-in-flight upload is
                    # never overwritten; cross-PASS reuse is fenced by the
                    # trailing sync (serial) / per-half events (pipelined).
                    seg = idx_pin[c0:c1]
                    if not idx_prefilled:
                        seg.copy_(torch.tensor(chunk_bids, dtype=torch.long))
                    idx = seg.to(dev, non_blocking=True)
                    for li, name in enumerate(names):
                        paged_u8[name].index_copy_(0, idx, scratch[:nblk, li])
                    if first_fast is None:  # (slab BYTE offset, bid)
                        first_fast = (slab_base + c0 * body_len, chunk_bids[0])
                except Exception as e:  # noqa: BLE001 — never-raise boundary: a failed chunk flags its blocks, not the engine
                    self._c._log.maybe("scatter", "chunked H2D scatter failed", e)
                    self._c._load_errors.update(b for b in chunk_bids if b is not None)
                continue
            # Mixed / degraded chunk: per-block copies for this chunk only.
            for j in range(c0, c1):
                bid = bids[j]
                if bid is None:
                    continue
                self._c._scatter_block_from_slab(j, bid, names, bytes_per_layer,
                                              slab_base=slab_base)
        if sync and getattr(dev, "type", "") == "cuda":
            try:
                torch.cuda.current_stream(dev).synchronize()
            except Exception as e:  # noqa: BLE001 — never-raise boundary: an unfinished stream means nothing is provably loaded
                self._c._log.maybe("scatter", "stream synchronize failed", e)
                self._c._load_errors.update(b for b in bids if b is not None)
        if (first_fast is not None and not self._debug_scatter_checked
                and os.environ.get("KVBLOCKD_DEBUG_SCATTER_CHECK") == "1"):
            self._c._debug_check_scatter(first_fast, names, bytes_per_layer)
        return ring is not None

    def _debug_check_scatter(self, first_fast: tuple[int, int], names,
                             bytes_per_layer) -> None:
        """KVBLOCKD_DEBUG_SCATTER_CHECK=1: after the first chunked-scatter
        load, compare ONE scattered block's bytes on the paged tensor against
        its slab source (uint8 equality) and log PASS/FAIL once. Runs after
        the stream synchronize; off by default; never raises."""
        self._debug_scatter_checked = True
        torch = _torch()
        off, bid = first_fast  # slab BYTE offset (half-aware), physical bid
        try:
            body_len = len(names) * bytes_per_layer
            slot = self._slab[off : off + body_len]
            ok = True
            for li, name in enumerate(names):
                got = (self._c._layer_kv[name][bid].contiguous()
                       .view(torch.uint8).reshape(-1).cpu())
                want = slot[li * bytes_per_layer : (li + 1) * bytes_per_layer]
                if not torch.equal(got, want):
                    ok = False
                    break
            logger.info("kvblockd debug scatter check: %s (slab offset %d -> paged block %d)",
                        "PASS" if ok else "FAIL", off, bid)
        except Exception as e:  # noqa: BLE001 — a broken debug probe must not break the load
            logger.info("kvblockd debug scatter check: FAIL (comparison errored: %s)", e)

    def _scatter_block_from_slab(self, j: int, bid: int, names, bytes_per_layer,
                                 slab_base: int = 0) -> None:
        """Per-block scatter of slab slot j (at byte base slab_base) into
        physical block bid — the same per-layer copy_ as the original path,
        sourced from the slab."""
        body_len = len(names) * bytes_per_layer
        buf = self._slab[slab_base + j * body_len : slab_base + (j + 1) * body_len]
        for li, name in enumerate(names):
            dst = self._c._layer_kv[name][bid]
            src = buf[li * bytes_per_layer : (li + 1) * bytes_per_layer]
            try:
                dst.copy_(src.view(dst.dtype).reshape(dst.shape))
            except Exception as e:  # noqa: BLE001 — never-raise boundary: a failed scatter marks the block errored, not the engine
                self._c._log.maybe("scatter", f"scatter into {name} failed", e)
                self._c._load_errors.add(bid)
                break
