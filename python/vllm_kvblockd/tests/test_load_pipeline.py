"""Pipelined load path (double-buffered slab halves + copy-stream events).

CPU-only CI: the CUDA-only dispatch is forced (`_slab_path_ok`), pinned
allocations stand in as plain CPU tensors (`_alloc_pinned`), the copy stream
is absent (None — CPU copies complete synchronously) and the events are
recording FAKES (`_make_event` is the seam) — the pass choreography, the
half offsets, the error/deadline ladders, the path stamps, and the
fetch-once contract are all the REAL code. The wire is a scripted in-process
client: byte-exact blobs, injectable failures/delays, and a full call log
(the exact-count hit gate's probe)."""

from __future__ import annotations

import contextlib
import logging
import random
import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from kvblockd import protocol as kp
from kvblockd.errors import ConnectionLost
from test_connector import BLOCK
from test_slab_scatter import (
    bare_connector,
    big_kv,
    force_slab,
    same_bytes,
    scatter_reference,
    slab_load_fixture,
)

from vllm_kvblockd.config import block_chain_keys
from vllm_kvblockd.connector import (
    BLOB_PREFIX_LEN,
    KvbReqMeta,
    encode_blob_prefix,
)

NBLOCKS = 8       # default load size: 4 passes of half_blocks=2
NUM_PAGED = 64


class FakeEvent:
    """Recording stand-in for torch.cuda.Event: every record/synchronize
    lands in the shared log with the event's creation ordinal."""

    def __init__(self, log, eid):
        self._log, self.eid = log, eid

    def record(self, stream=None):
        self._log.append(("record", self.eid))

    def synchronize(self):
        self._log.append(("sync", self.eid))


def fake_events(conn, log):
    counter = {"n": 0}

    def make():
        ev = FakeEvent(log, counter["n"])
        counter["n"] += 1
        return ev

    conn._make_event = make


class FakePipeClient:
    """Scripted wire: serves byte-exact blobs through the REAL alloc contract
    (prefix first, body into the returned view), logs every call's keys, and
    injects failures/delays by call index."""

    def __init__(self, blobs, fail_calls=(), delays=None, log=None,
                 fail_exc=RuntimeError):
        self.blobs = blobs
        self.calls: list[list[bytes]] = []
        self.fail_calls = set(fail_calls)
        self.delays = dict(delays or {})
        self.log = log
        self.fail_exc = fail_exc

    def batch_get_scatter(self, keys, prefix_len, alloc, deadline=None):
        i = len(self.calls)
        self.calls.append(list(keys))
        if self.log is not None:
            self.log.append(("drain", i))
        time.sleep(self.delays.get(i, 0))
        if i in self.fail_calls:
            raise self.fail_exc("injected drain failure")
        statuses = []
        for j, k in enumerate(keys):
            blob = self.blobs.get(k)
            if blob is None:
                statuses.append(kp.Status.NOT_FOUND)
                continue
            view = alloc(j, blob[:prefix_len], len(blob) - prefix_len)
            if view is None:
                statuses.append(kp.Status.NOT_FOUND)
                continue
            memoryview(view)[:] = blob[prefix_len:]
            statuses.append(kp.Status.OK)
        return statuses

    def close(self):
        return None


def make_world(nblocks=NBLOCKS, half_blocks=2, salt="pipe", missing=(),
               fail_calls=(), delays=None, log=None, blobs=None,
               fail_exc=RuntimeError):
    """(conn, req, world) with a forced-slab connector whose staging cap
    yields exactly `half_blocks`-sized pipeline passes, permuted physical
    bids, and a scripted client serving deterministic random bodies."""
    conn = bare_connector()
    force_slab(conn)
    conn._layer_kv = big_kv(NUM_PAGED)
    names, dtype_name, bpl = conn._layout()
    body = len(names) * bpl
    total = BLOB_PREFIX_LEN + body
    conn._staging_bytes = 2 * half_blocks * body
    toks = list(range(500, 500 + nblocks * BLOCK))
    seed = conn._seed(salt, [], "")
    keys = block_chain_keys(seed, toks, BLOCK)
    prefix = encode_blob_prefix(dtype_name, len(names), BLOCK, bpl, total)
    rng = np.random.default_rng(7)
    bodies = {}
    if blobs is None:
        blobs = {}
        for i, k in enumerate(keys):
            body_bytes = rng.integers(0, 256, body, dtype=np.uint8).tobytes()
            bodies[i] = body_bytes
            if i not in missing:
                blobs[k] = prefix + body_bytes
    bids = random.Random(3).sample(range(NUM_PAGED), nblocks)
    req = KvbReqMeta(req_id="pipe", token_ids=toks, cache_salt=salt, mm_ids=[],
                     lora_name="", block_ids=bids, load_start_block=0,
                     num_load_blocks=nblocks, store_start_block=nblocks,
                     store_end_block=nblocks)
    fake = FakePipeClient(blobs, fail_calls=fail_calls, delays=delays, log=log,
                          fail_exc=fail_exc)
    conn._client = fake
    world = {"names": names, "bpl": bpl, "body": body, "total": total,
             "keys": keys, "bodies": bodies, "blobs": blobs, "bids": bids,
             "fake": fake, "prefix": prefix}
    return conn, req, world


def assert_loaded(conn, w, idx):
    """Loaded block i's paged bytes == the blob body, per layer, byte-exact."""
    for i in idx:
        for li, name in enumerate(w["names"]):
            got = (conn._layer_kv[name][w["bids"][i]].contiguous()
                   .view(torch.uint8).reshape(-1))
            want = torch.frombuffer(
                bytearray(w["bodies"][i][li * w["bpl"]:(li + 1) * w["bpl"]]),
                dtype=torch.uint8)
            assert torch.equal(got, want), f"block {i} layer {name} diverged"


def assert_untouched(conn, w, idx):
    for i in idx:
        for name in w["names"]:
            assert torch.count_nonzero(conn._layer_kv[name][w["bids"][i]]) == 0


# ------------------------------------------------------- B1: byte identity

def test_pipelined_matches_serial_byte_exact():
    """The same load through the OLD serial slab path and the NEW pipelined
    path lands byte-identical pages (4 passes, permuted bids)."""
    conn_p, req_p, w = make_world()
    conn_p._load_one(req_p)
    assert conn_p._load_errors == set()
    assert conn_p._reported_path == "pipelined-slab"
    assert_loaded(conn_p, w, range(NBLOCKS))

    conn_s, req_s, ws = make_world(blobs=w["blobs"])
    ws["bodies"] = w["bodies"]
    conn_s._pipeline_disabled = True
    conn_s._load_one(req_s)
    assert conn_s._load_errors == set()
    assert conn_s._reported_path == "chunked-slab"
    for name in w["names"]:
        assert same_bytes(conn_p._layer_kv[name], conn_s._layer_kv[name]), \
            f"{name}: pipelined diverged from the serial oracle"


# ------------------------------------------ B2: per-half miss flag surgery

def test_missing_half_flags_exactly_that_half():
    """Pass 1's blobs missing: exactly those bids flagged; every other half's
    bytes land intact (no superset creep on the healthy path)."""
    conn, req, w = make_world(missing={2, 3})
    conn._load_one(req)
    assert conn._load_errors == {w["bids"][2], w["bids"][3]}
    assert_loaded(conn, w, [0, 1, 4, 5, 6, 7])
    assert_untouched(conn, w, [2, 3])


# ----------------------- B3 + R3: drain raise -> serial remainder, no latch

def test_drain_raise_flags_pass_serves_remainder_serially_without_latch():
    """A non-connection drain raise (client kept): the raised pass is
    flagged, the undrained tail is served by the serial slab path."""
    conn, req, w = make_world(fail_calls={1})
    conn._load_one(req)  # must not raise
    assert conn._load_errors == {w["bids"][2], w["bids"][3]}
    assert_loaded(conn, w, [0, 1, 4, 5, 6, 7])
    # R3: a drain fault is NOT a setup failure — the latch is untouched.
    assert conn._pipeline_fails == 0 and not conn._pipeline_disabled
    # Exact-count hit gate: the remainder was served WITHOUT re-fetching any
    # key (the raised pass's keys were already spent on the wire).
    fetched = [k for call in w["fake"].calls for k in call]
    assert fetched == w["keys"]


def test_connection_class_drain_raise_flags_remainder_via_breaker():
    """A ConnectionLost drain drops the client and arms the dial breaker, so
    the serial remainder degrades to flagged misses instead of hammering a
    dead endpoint — bounded, disclosed, never a raise."""
    conn, req, w = make_world(fail_calls={1}, fail_exc=ConnectionLost)
    conn._load_one(req)  # must not raise
    assert conn._load_errors == {w["bids"][i] for i in range(2, NBLOCKS)}
    assert_loaded(conn, w, [0, 1])  # the healthy pass still landed
    assert conn._client is None     # breaker discipline: client dropped
    assert conn._pipeline_fails == 0 and not conn._pipeline_disabled  # R3


# ------------------------------------- B5: never-raise through start_load_kv

def test_drain_raise_never_escapes_start_load_kv():
    from test_connector import StubForwardContext, StubNewReq, StubRequest, StubSchedulerOutput

    conn, _req, w = make_world(fail_calls={0})
    toks = list(range(500, 500 + NBLOCKS * BLOCK + 1))  # +1: aligned == 8 blocks
    sreq = StubRequest("pipe-slk", toks, "pipe")
    conn.update_state_after_alloc(sreq, None, NBLOCKS * BLOCK)
    meta = conn.build_connector_meta(StubSchedulerOutput(
        [StubNewReq(sreq, w["bids"])], {"pipe-slk": len(toks)}))
    conn.bind_connector_metadata(meta)
    conn.start_load_kv(StubForwardContext(conn._layer_kv))  # must NOT raise
    errs = conn.get_block_ids_with_load_errors()
    assert {w["bids"][0], w["bids"][1]} == errs  # pass 0 flagged, disclosed
    assert_loaded(conn, w, [2, 3, 4, 5, 6, 7])   # serial remainder landed
    conn.shutdown()


# --------------------------------------------- B4 + R2: deadline after drain

def test_deadline_after_drain_scatters_half_and_flags_only_remainder():
    """The deadline bounds WIRE time: a half that drained (and verified)
    just past the wall is still scattered; ONLY the undrained remainder is
    flagged — flagged set == promised − filled, exactly."""
    conn, req, w = make_world(delays={0: 0.45})
    conn._cfg.load_deadline_s = 0.3
    t0 = time.monotonic()
    conn._load_one(req)
    assert time.monotonic() - t0 < 2.0
    assert len(w["fake"].calls) == 1  # no pass submitted past the wall
    assert_loaded(conn, w, [0, 1])    # the drained half landed (R2)
    assert conn._load_errors == {w["bids"][i] for i in range(2, NBLOCKS)}
    assert_untouched(conn, w, range(2, NBLOCKS))


# --------------------------------------------------- B6: setup-failure latch

def test_three_setup_failures_latch_pipeline_off_serial_keeps_serving(caplog):
    conn, req, w = make_world()
    calls = []

    def boom(dev):
        calls.append(1)
        raise RuntimeError("no copy stream (injected)")

    conn._copy_stream = boom
    with caplog.at_level(logging.WARNING, logger="vllm_kvblockd"):
        for _ in range(3):
            conn._load_one(req)
            assert conn._load_errors == set()
            assert_loaded(conn, w, range(NBLOCKS))  # serial kept serving
    assert conn._pipeline_disabled and len(calls) == 3
    assert any("latched OFF" in r.getMessage() for r in caplog.records)
    conn._load_one(req)
    assert len(calls) == 3  # latched: setup never re-attempted
    assert conn._reported_path == "chunked-slab"


# ------------------------------------------------ B7/OPEN-1: path stamping

def test_midload_fallback_stamp_names_the_majority_mover(caplog):
    """OPEN-1 settled: the stamp attributes the load to the lane that moved
    the MAJORITY of its blocks — the bench takes the LAST 'kvblockd load
    path:' match, and the tail of a mostly-serial load was served serial."""
    with caplog.at_level(logging.WARNING, logger="vllm_kvblockd"):
        # Raise at pass 1 of 4: pipelined moved 2 blocks, serial moved 4.
        conn, req, _w = make_world(fail_calls={1})
        conn._load_one(req)
        assert conn._reported_path == "chunked-slab"
        # Raise at the LAST pass: pipelined moved 6, serial 0.
        conn2, req2, w2 = make_world(fail_calls={3})
        conn2._load_one(req2)
        assert conn2._reported_path == "pipelined-slab"
        assert conn2._load_errors == {w2["bids"][6], w2["bids"][7]}
        assert_loaded(conn2, w2, range(6))
    lines = [r.getMessage() for r in caplog.records
             if r.getMessage().startswith("kvblockd load path:")]
    assert lines == ["kvblockd load path: chunked-slab",
                     "kvblockd load path: pipelined-slab"]


# ------------------------------------------------- B8: exact-count hit gate

def test_every_key_fetched_exactly_once():
    conn, req, w = make_world()
    conn._load_one(req)
    fetched = [k for call in w["fake"].calls for k in call]
    assert fetched == w["keys"]                 # in order, nothing re-fetched
    assert len(set(fetched)) == len(w["keys"])  # and nothing duplicated


# ------------------------------------- B9: event fencing of slab-half reuse

def test_half_never_redrained_before_its_event_synced():
    """free_ev[h] (recorded after half h's last chunk) must be SYNCED before
    the pass that overwrites half h is submitted: sync(pass p's event)
    strictly precedes drain(pass p+2), for both halves."""
    log = []
    conn, req, w = make_world(log=log)
    fake_events(conn, log)
    conn._load_one(req)
    assert conn._load_errors == set()
    assert_loaded(conn, w, range(NBLOCKS))
    # Events 0..3 are the per-pass fences (creation order == pass order).
    def pos(entry):
        assert entry in log, f"{entry} never happened: {log}"
        return log.index(entry)
    assert pos(("sync", 0)) < pos(("drain", 2))  # half A reuse fenced
    assert pos(("sync", 1)) < pos(("drain", 3))  # half B reuse fenced


# ---------------------------------------------- B10: flags settle by return

def test_all_flag_mutations_complete_before_return():
    conn, req, w = make_world(fail_calls={3}, delays={3: 0.05})
    conn._load_one(req)
    errs = conn.get_block_ids_with_load_errors()
    assert errs == {w["bids"][6], w["bids"][7]}
    time.sleep(0.1)  # nothing trickles in from any background thread
    assert conn._load_errors == set()


# ------------------------- B13: exit sync + ring handoff + thread lifecycle

def test_exit_sync_precedes_store_gather_ring_use_and_no_thread_survives():
    log = []
    conn, req, w = make_world(log=log)
    fake_events(conn, log)
    conn._load_one(req)
    # The final copy-stream event (last one created) is recorded AND synced
    # before _load_one returns — the log ends with it.
    last_eid = max(e for op, e in log if op == "record")
    assert log[-2:] == [("record", last_eid), ("sync", last_eid)]
    # Ring handoff: a store-side gather grabbing the shared scratch ring
    # happens strictly after that exit sync.
    real_ring = conn._scratch_ring

    def logging_ring(dev, n_layers, bytes_per_layer):
        log.append(("ring-store", 0))
        return real_ring(dev, n_layers, bytes_per_layer)

    conn._scratch_ring = logging_ring
    req.store_start_block, req.store_end_block = 0, 2
    plan = conn._stage_gather(req, w["names"], w["bpl"], w["total"],
                              w["prefix"], w["keys"], 0, 2)
    assert plan is not None
    assert log.index(("ring-store", 0)) > log.index(("sync", last_eid))
    # Shutdown joins the prefetch worker: no load thread of THIS connector
    # survives (scoped to its executor — other tests' connectors may leak
    # idle workers into threading.enumerate()).
    workers = list(conn._prefetch_ex._threads)
    assert workers and any(t.is_alive() for t in workers)
    conn.shutdown()
    assert not any(t.is_alive() for t in workers)


# -------------------------- B14: entry fence (compute->copy stream ordering)

def test_entry_fence_orders_copy_stream_after_compute_before_first_scatter():
    """Write-after-read armor: a paged block freed by a finishing request
    and reallocated to this load may still be READ by an in-flight
    prior-step kernel on the compute stream. Before the FIRST scatter the
    copy stream must wait on an event recorded on the COMPUTE stream —
    without it the pipelined index_copy_ is a silent wrong byte."""
    log = []
    conn, req, w = make_world(log=log)
    compute = object()  # the engine's current stream, by identity

    class FakeStream:
        def wait_event(self, ev):
            log.append(("wait", ev.eid))

    class TaggedEvent(FakeEvent):
        def record(self, stream=None):
            tag = "compute" if stream is compute else "copy"
            self._log.append(("record", self.eid, tag))

    counter = {"n": 0}

    def make():
        ev = TaggedEvent(log, counter["n"])
        counter["n"] += 1
        return ev

    conn._make_event = make
    conn._copy_stream = lambda dev: FakeStream()
    conn._current_stream = lambda dev: compute
    conn._stream_scope = lambda s: contextlib.nullcontext()
    real_scatter = conn._scatter_slab

    def scatter(*a, **kw):
        log.append(("scatter", kw.get("key_offset", 0)))
        return real_scatter(*a, **kw)

    conn._scatter_slab = scatter
    conn._load_one(req)
    assert conn._load_errors == set()
    assert conn._reported_path == "pipelined-slab"
    assert_loaded(conn, w, range(NBLOCKS))
    # The fence event (eid 0, created at entry) is recorded on the COMPUTE
    # stream and waited on the copy stream BEFORE the first scatter.
    i_rec = log.index(("record", 0, "compute"))
    i_wait = log.index(("wait", 0))
    i_scat = log.index(("scatter", 0))
    assert i_rec < i_wait < i_scat
    # It is the ONLY compute-stream record and the only cross-stream wait —
    # every later fence stays copy-stream-internal.
    assert [e for e in log if e[0] == "record" and e[2] == "compute"] \
        == [("record", 0, "compute")]
    assert [e for e in log if e[0] == "wait"] == [("wait", 0)]


# ------------------- B15: failed exit sync latches + poisons the shared ring

def test_final_sync_failure_latches_pipeline_and_poisons_ring_until_drain():
    """A failed exit sync leaves copy-stream ring work possibly in flight:
    every promised bid is flagged, the pipeline latches OFF, the shared
    scratch ring is dropped, and the store path must refuse the ring
    (bytearray staging) until a copy-stream drain PROVES re-creation safe
    — otherwise _stage_gather could publish torn bytes cache-wide under a
    key whose xxh3 was computed over the tear."""
    log = []
    conn, req, w = make_world(log=log)
    sick = {"on": True}
    counter = {"n": 0}

    class SickableEvent(FakeEvent):
        def synchronize(self):
            if sick["on"]:
                raise RuntimeError("device sick (injected)")
            super().synchronize()

    def make():
        ev = SickableEvent(log, counter["n"])
        counter["n"] += 1
        return ev

    conn._make_event = make
    conn._load_one(req)  # must not raise
    # Nothing provably loaded: the whole promise flagged, path stamped.
    assert conn._load_errors == set(w["bids"])
    assert conn._reported_path == "pipelined-slab"
    assert conn._pipeline_disabled
    assert conn._gpu_scratch is None and conn._scratch_torn
    # Store side, device still sick: _stage_one must refuse the torn ring
    # and degrade to bytearray staging (no gathered plan).
    conn._cfg.store_staging_bytes = 64 * w["total"]
    req.store_start_block, req.store_end_block = 0, 2
    assert conn._stage_one(req) is None
    assert conn._reported_store_path == "bytearray"
    assert conn._scratch_torn  # still unproven — stays poisoned
    assert not conn._store_gather_disabled  # degrade, not a store latch
    # A successful copy-stream drain re-arms the ring: gathered staging
    # resumes and the poison clears.
    sick["on"] = False
    plan = conn._stage_one(req)
    assert plan is not None
    assert not conn._scratch_torn and conn._gpu_scratch is not None


# ------------------------------------------------- B11: pinned idx staging

def test_idx_staging_allocated_once_and_mixed_chunks_stay_byte_exact():
    nblocks = 130  # 2 full chunks + tail
    conn, req, names, bpl = slab_load_fixture(nblocks)
    statuses = [kp.Status.OK] * nblocks
    statuses[5] = kp.Status.NOT_FOUND  # chunk 0 goes mixed -> per-block

    ref_kv = {n: torch.zeros_like(t) for n, t in conn._layer_kv.items()}
    scatter_reference(conn, req, names, bpl, statuses, ref_kv)

    conn._scatter_slab(req, names, bpl, statuses)
    pin = conn._idx_pin
    assert pin is not None and pin.numel() >= nblocks
    conn._scatter_slab(req, names, bpl, statuses)
    assert conn._idx_pin is pin  # reused across scatters, never rebuilt
    assert conn._load_errors == {req.block_ids[5]}
    for name in names:
        assert same_bytes(conn._layer_kv[name], ref_kv[name]), f"{name} diverged"


# ------------------------------------- B16: the pipeline-half knob's shapes

def test_pipeline_half_knob_reshapes_passes():
    """kvblockd_pipeline_half_bytes caps a half below staging/2: a 1-body
    half turns the 4-pass default world into 8 single-block passes — same
    bytes, same order, every key fetched exactly once."""
    conn, req, w = make_world()
    conn._staging_bytes = 16 * w["body"]  # staging no longer the binding cap
    conn._cfg.pipeline_half_bytes = w["body"]  # 1 block per half
    conn._load_one(req)
    assert conn._load_errors == set()
    assert conn._reported_path == "pipelined-slab"
    assert_loaded(conn, w, range(NBLOCKS))
    assert [len(c) for c in w["fake"].calls] == [1] * NBLOCKS
    fetched = [k for call in w["fake"].calls for k in call]
    assert fetched == w["keys"]


def test_pipeline_half_knob_below_body_serializes_with_disclosure(caplog):
    """A knob smaller than one blob body cannot be refused at boot (blob
    size is engine-dependent), so it must DISCLOSE that it serialized the
    load path — a silent serialization is the exact failure the boot-time
    >0 refusal exists to prevent. The load itself still lands byte-exact
    through the serial slab lane, and the pipeline latch is untouched."""
    conn, req, w = make_world()
    conn._cfg.pipeline_half_bytes = w["body"] - 1
    with caplog.at_level(logging.WARNING, logger="vllm_kvblockd"):
        conn._load_one(req)
    assert conn._load_errors == set()
    assert conn._reported_path == "chunked-slab"  # serial lane served it
    assert_loaded(conn, w, range(NBLOCKS))
    assert not conn._pipeline_disabled and conn._pipeline_fails == 0
    assert any("pipelined path disabled" in r.getMessage() for r in caplog.records)
