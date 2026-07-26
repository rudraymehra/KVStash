"""Pinned-slab staging + chunked batched scatter (the warm-load fast path).

CI runs CPU-only torch, so the CUDA-only dispatch is exercised by forcing the
slab path (`_slab_path_ok`) and standing in a plain CPU tensor for the pinned
allocation (`_alloc_pinned` is the seam) — the chunk math, the layout gate,
the disjoint-slot slab, and the degrade ladders are all the REAL code. The
production CPU lane (per-block, unchanged) is asserted to never touch the
slab. Wire-touching tests use the same real-daemon fixture as test_connector.
"""

from __future__ import annotations

import logging
import random

import pytest

torch = pytest.importorskip("torch")

from kvblockd import protocol as kp
from test_connector import (
    BLOCK,
    LAYERS,
    StubForwardContext,
    StubNewReq,
    StubRequest,
    StubSchedulerOutput,
    StubVllmConfig,
    fill_block,
    fresh_kv,
    make_connector,
    run_step,
)

from vllm_kvblockd.connector import KvblockdConnector, KvbReqMeta

HID = 8


def bare_connector() -> KvblockdConnector:
    """A connector that never dials (port 1) — for slab/scatter unit tests."""
    return KvblockdConnector(StubVllmConfig(1), role="scheduler", kv_cache_config=None)


def plain_alloc(n):
    # CPU stand-in for the pinned allocation (CI has no CUDA allocator).
    return torch.empty(n, dtype=torch.uint8)


def force_slab(conn) -> None:
    conn._slab_path_ok = lambda dev: True
    conn._alloc_pinned = plain_alloc


def big_kv(num_paged_blocks: int):
    return {n: torch.zeros(num_paged_blocks, 2, BLOCK, HID, dtype=torch.bfloat16)
            for n in LAYERS}


def same_bytes(a, b) -> bool:
    """Byte-level equality — the property under test. (torch.equal on bf16
    would compare VALUES, and random slab bytes decode to NaNs, which compare
    unequal to themselves.)"""
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def scatter_reference(conn, req, names, bytes_per_layer, statuses, layer_kv):
    """The OLD per-(block,layer) copy_ loop, re-stated as the oracle: what the
    chunked path must be byte-identical to for every OK block."""
    body_len = len(names) * bytes_per_layer
    for j, st in enumerate(statuses):
        if st != kp.Status.OK:
            continue
        bid = req.block_ids[req.load_start_block + j]
        slot = conn._slab[j * body_len:(j + 1) * body_len]
        for li, name in enumerate(names):
            dst = layer_kv[name][bid]
            src = slot[li * bytes_per_layer:(li + 1) * bytes_per_layer]
            dst.copy_(src.view(dst.dtype).reshape(dst.shape))


def slab_load_fixture(nblocks: int, num_paged: int = 200):
    """(connector, req, names, bpl) with a filled slab of nblocks random slots
    mapped onto PERMUTED physical block ids — chunk order != bid order."""
    conn = bare_connector()
    force_slab(conn)
    conn._layer_kv = big_kv(num_paged)
    names, _, bpl = conn._layout()
    body_len = len(names) * bpl
    assert conn._slab_reserve(nblocks * body_len)
    g = torch.Generator().manual_seed(1234)
    conn._slab[: nblocks * body_len] = torch.randint(
        0, 256, (nblocks * body_len,), dtype=torch.uint8, generator=g)
    rng = random.Random(99)
    bids = rng.sample(range(num_paged), nblocks)
    req = KvbReqMeta(req_id="scatter", token_ids=[], cache_salt=None, mm_ids=[],
                     lora_name="", block_ids=bids, load_start_block=0,
                     num_load_blocks=nblocks, store_start_block=0, store_end_block=0)
    return conn, req, names, bpl


# --------------------------------------------------------------------- slab

def test_slab_reuse_growth_and_disjoint_slots():
    conn = bare_connector()
    conn._alloc_pinned = plain_alloc
    assert conn._slab_reserve(1000)
    slab1 = conn._slab
    assert slab1.numel() >= 1000
    # Reuse: a smaller (or equal) reservation NEVER reallocates or frees.
    assert conn._slab_reserve(500)
    assert conn._slab is slab1
    # Geometric growth: >= 2x the old capacity, not just the new need.
    assert conn._slab_reserve(1500)
    assert conn._slab.numel() >= 2000
    # Disjoint slots by construction: writes through slot views never bleed.
    body = 128
    mv0 = memoryview(conn._slab_np[0:body])
    mv2 = memoryview(conn._slab_np[2 * body:3 * body])
    mv0[:] = b"\x11" * body
    mv2[:] = b"\x22" * body
    assert bytes(conn._slab_np[0:body]) == b"\x11" * body
    assert bytes(conn._slab_np[body:2 * body]) != b"\x11" * body  # untouched slot
    assert bytes(conn._slab_np[2 * body:3 * body]) == b"\x22" * body


def test_slab_pin_failure_trips_breaker_and_never_raises():
    conn = bare_connector()
    calls = []

    def boom(n):
        calls.append(n)
        raise RuntimeError("cudaHostAlloc failed")

    conn._alloc_pinned = boom
    assert conn._slab_reserve(64) is False
    assert conn._slab_disabled and conn._slab is None
    assert conn._slab_reserve(64) is False  # permanent: no retry storm
    assert len(calls) == 1


def test_pin_failure_load_degrades_to_perblock_byte_identical(daemon):
    """Slab dispatch active but pinning fails -> the CURRENT per-block path
    serves the load, byte-identical, no engine-visible error."""
    toks = list(range(300, 309))
    salt = "t-slab-pinfail"
    conn1 = make_connector(daemon)
    kv1 = fresh_kv()
    fill_block(kv1, 2, seed=311)
    fill_block(kv1, 5, seed=312)
    run_step(conn1, StubRequest("pf-store", toks, salt), [2, 5], kv1)
    conn1.shutdown()

    conn2 = make_connector(daemon)
    conn2._slab_path_ok = lambda dev: True

    def boom(n):
        raise RuntimeError("no pinned memory here")

    conn2._alloc_pinned = boom
    req = StubRequest("pf-load", toks, salt)
    kv2 = fresh_kv()
    assert conn2.get_num_new_matched_tokens(req, 0)[0] == 8
    run_step(conn2, req, [1, 3], kv2)
    for name in LAYERS:
        assert torch.equal(kv2[name][1], kv1[name][2])
        assert torch.equal(kv2[name][3], kv1[name][5])
    assert conn2.get_block_ids_with_load_errors() == set()
    assert conn2._slab is None and conn2._slab_disabled
    conn2.shutdown()


def test_cpu_path_never_allocates_slab(daemon):
    """Production CPU backend: the dispatch must keep the original per-block
    lane and never pin/allocate a slab (the vllm-native-cpu CI contract)."""
    toks = list(range(320, 329))
    salt = "t-slab-cpu"
    conn1 = make_connector(daemon)
    kv1 = fresh_kv()
    fill_block(kv1, 0, seed=321)
    fill_block(kv1, 1, seed=322)
    run_step(conn1, StubRequest("cpu-store", toks, salt), [0, 1], kv1)
    conn1.shutdown()

    conn2 = make_connector(daemon)
    req = StubRequest("cpu-load", toks, salt)
    kv2 = fresh_kv()
    run_step(conn2, req, [4, 6], kv2)
    for name in LAYERS:
        assert torch.equal(kv2[name][4], kv1[name][0])
        assert torch.equal(kv2[name][6], kv1[name][1])
    assert conn2._slab is None  # CPU tensors -> per-block path, no pinning
    assert conn2.get_block_ids_with_load_errors() == set()
    conn2.shutdown()


def test_forced_slab_roundtrip_byte_identity_and_reuse(daemon):
    """The full slab lane over the REAL wire (layout gate in alloc, slab
    slots, chunked scatter): byte-identical load, slab reused across loads."""
    toks = list(range(340, 349))
    salt = "t-slab-wire"
    conn1 = make_connector(daemon)
    kv1 = fresh_kv()
    fill_block(kv1, 2, seed=341)
    fill_block(kv1, 6, seed=342)
    run_step(conn1, StubRequest("sw-store", toks, salt), [2, 6], kv1)
    conn1.shutdown()

    conn2 = make_connector(daemon)
    force_slab(conn2)
    kv2 = fresh_kv()
    run_step(conn2, StubRequest("sw-load", toks, salt), [1, 3], kv2)
    for name in LAYERS:
        assert torch.equal(kv2[name][1], kv1[name][2])
        assert torch.equal(kv2[name][3], kv1[name][6])
    assert conn2.get_block_ids_with_load_errors() == set()
    assert conn2._slab is not None  # the slab lane actually ran
    slab_before = conn2._slab

    kv3 = fresh_kv()
    run_step(conn2, StubRequest("sw-load-2", toks, salt), [5, 7], kv3)
    for name in LAYERS:
        assert torch.equal(kv3[name][5], kv1[name][2])
        assert torch.equal(kv3[name][7], kv1[name][6])
    assert conn2._slab is slab_before  # reused, never freed per-load
    conn2.shutdown()


def test_forced_slab_layout_drift_is_miss_not_scatter(daemon):
    """The slab alloc runs the SAME layout gate before accepting any body
    byte: drifted blobs are misses + flags, never a corrupt scatter."""
    toks = list(range(360, 369))
    salt = "t-slab-drift"
    conn1 = make_connector(daemon)
    kv1 = fresh_kv()
    fill_block(kv1, 2, seed=361)
    fill_block(kv1, 3, seed=362)
    run_step(conn1, StubRequest("dr-store", toks, salt), [2, 3], kv1)
    conn1.shutdown()

    conn2 = make_connector(daemon)
    force_slab(conn2)
    req = StubRequest("dr-load", toks, salt)
    conn2.update_state_after_alloc(req, None, 8)
    meta = conn2.build_connector_meta(
        StubSchedulerOutput([StubNewReq(req, [4, 5])], {"dr-load": 9}))
    conn2.bind_connector_metadata(meta)
    small_kv = {n: torch.zeros(8, 2, BLOCK, HID // 2, dtype=torch.bfloat16) for n in LAYERS}
    conn2.start_load_kv(StubForwardContext(small_kv))
    assert conn2.get_block_ids_with_load_errors() == {4, 5}
    for t in small_kv.values():
        assert torch.count_nonzero(t) == 0  # nothing was written
    conn2.shutdown()


# ---------------------------------------------------------- chunked scatter

def test_chunked_scatter_all_ok_byte_exact_vs_perblock():
    """130 blocks = 2 full chunks + a tail, permuted bids: the chunked fast
    path must land byte-identically to the old per-(block,layer) loop."""
    nblocks = 130
    conn, req, names, bpl = slab_load_fixture(nblocks)
    statuses = [kp.Status.OK] * nblocks

    ref_kv = {n: torch.zeros_like(t) for n, t in conn._layer_kv.items()}
    scatter_reference(conn, req, names, bpl, statuses, ref_kv)

    conn._scatter_slab(req, names, bpl, statuses)
    assert conn._load_errors == set()
    for name in names:
        assert same_bytes(conn._layer_kv[name], ref_kv[name]), f"{name} diverged"


def test_chunked_scatter_mixed_chunk_falls_back_per_block():
    """A non-OK block inside chunk 0 pushes THAT chunk onto the per-block
    path (OK neighbours still land, byte-exact); chunk 1 keeps the fast path.
    Non-OK blocks are flagged and their paged rows never written."""
    nblocks = 100  # chunk 0 = [0,64) mixed, chunk 1 = [64,100) all-OK
    conn, req, names, bpl = slab_load_fixture(nblocks)
    statuses = [kp.Status.OK] * nblocks
    statuses[5] = kp.Status.NOT_FOUND
    statuses[63] = kp.Status.NOT_FOUND

    ref_kv = {n: torch.zeros_like(t) for n, t in conn._layer_kv.items()}
    scatter_reference(conn, req, names, bpl, statuses, ref_kv)

    conn._scatter_slab(req, names, bpl, statuses)
    assert conn._load_errors == {req.block_ids[5], req.block_ids[63]}
    for name in names:
        assert same_bytes(conn._layer_kv[name], ref_kv[name]), f"{name} diverged"
        # the missing blocks' rows stayed zero (ref never wrote them either)
        assert torch.count_nonzero(conn._layer_kv[name][req.block_ids[5]]) == 0


def test_chunk_failure_flags_chunk_bids_and_degrades():
    """A CUDA/copy failure inside one chunk flags that chunk's bids (superset
    flagging is allowed) and never raises; other chunks still land."""
    nblocks = 74  # chunk 0 = 64 blocks (will fail), chunk 1 = 10 (fits scratch)
    conn, req, names, bpl = slab_load_fixture(nblocks)
    statuses = [kp.Status.OK] * nblocks

    # Scratch too small for a full chunk: chunk 0's batched copy_ raises,
    # chunk 1 (10 blocks) fits and takes the fast path.
    def tiny_ring(dev, n_layers, bytes_per_layer):
        return [torch.empty((32, n_layers, bytes_per_layer), dtype=torch.uint8, device=dev)
                for _ in range(2)]

    conn._scratch_ring = tiny_ring

    ref_kv = {n: torch.zeros_like(t) for n, t in conn._layer_kv.items()}
    scatter_reference(conn, req, names, bpl, [kp.Status.OK] * nblocks, ref_kv)

    conn._scatter_slab(req, names, bpl, statuses)  # must not raise
    assert set(req.block_ids[:64]).issubset(conn._load_errors)
    assert not conn._load_errors.intersection(req.block_ids[64:])
    for name in names:
        for j in range(64, nblocks):  # the healthy chunk landed byte-exact
            bid = req.block_ids[j]
            assert same_bytes(conn._layer_kv[name][bid], ref_kv[name][bid])


def test_noncontiguous_paged_tensor_lands_byte_exact_never_silent_loss():
    """BLOCKER regression: a PERMUTED (non-contiguous, stride(-1)==1) paged
    tensor — [2, N, B, H] permuted to [N, 2, B, H] — must either land the
    bytes byte-exact or flag load_errors, NEVER lose them silently. The old
    reshape() silently COPIED such a tensor, so index_copy_ wrote into a
    temporary and every loaded byte vanished with no flag; view() aliases or
    raises, and the raise routes the load to the byte-exact per-block
    fallback. With the fix: bytes land AND nothing is flagged."""
    nblocks = 20
    num_paged = 64
    conn = bare_connector()
    force_slab(conn)
    base = {n: torch.zeros(2, num_paged, BLOCK, HID, dtype=torch.bfloat16) for n in LAYERS}
    conn._layer_kv = {n: t.permute(1, 0, 2, 3) for n, t in base.items()}
    for t in conn._layer_kv.values():
        assert not t.is_contiguous() and t.stride(-1) == 1

    names, _, bpl = conn._layout()
    body_len = len(names) * bpl
    assert conn._slab_reserve(nblocks * body_len)
    g = torch.Generator().manual_seed(4242)
    conn._slab[: nblocks * body_len] = torch.randint(
        0, 256, (nblocks * body_len,), dtype=torch.uint8, generator=g)
    bids = random.Random(5).sample(range(num_paged), nblocks)
    req = KvbReqMeta(req_id="noncontig", token_ids=[], cache_salt=None, mm_ids=[],
                     lora_name="", block_ids=bids, load_start_block=0,
                     num_load_blocks=nblocks, store_start_block=0, store_end_block=0)
    statuses = [kp.Status.OK] * nblocks

    # Oracle: the per-block loop into a CONTIGUOUS tensor of the same logical shape.
    ref_kv = {n: torch.zeros(num_paged, 2, BLOCK, HID, dtype=torch.bfloat16) for n in LAYERS}
    scatter_reference(conn, req, names, bpl, statuses, ref_kv)

    conn._scatter_slab(req, names, bpl, statuses)  # must not raise
    assert conn._load_errors == set()  # nothing flagged...
    for name in names:
        for bid in bids:               # ...because every byte actually landed
            assert same_bytes(conn._layer_kv[name][bid], ref_kv[name][bid]), \
                f"{name} block {bid}: silent loss on a non-contiguous paged tensor"


# ------------------------------------------------------- slab cap + breakers

def test_slab_reserve_respects_cap_and_growth_failure_keeps_old_slab():
    """The staging cap bounds the slab (over-cap reservations are refused
    WITHOUT allocating or tripping the breaker; geometric growth clamps AT
    the cap), and a GROWTH failure keeps the working slab — the permanent
    breaker only trips when there is no slab at all."""
    conn = bare_connector()
    calls = []

    def alloc(n):
        calls.append(n)
        return plain_alloc(n)

    conn._alloc_pinned = alloc
    conn._staging_bytes = 1024
    assert conn._slab_reserve(2048) is False       # over the cap: refused
    assert calls == [] and not conn._slab_disabled  # ...without alloc or breaker
    assert conn._slab_reserve(600)
    assert conn._slab.numel() == 600
    assert conn._slab_reserve(700)                  # want=max(700,1200) -> capped
    assert conn._slab.numel() == 1024
    slab1 = conn._slab

    def boom(n):
        raise RuntimeError("pinned pool exhausted")

    conn._alloc_pinned = boom
    conn._staging_bytes = 4096
    assert conn._slab_reserve(2048) is False       # growth failed for THIS load
    assert conn._slab is slab1                     # old slab intact
    assert not conn._slab_disabled                 # breaker NOT tripped
    assert conn._slab_reserve(512) is True         # smaller loads keep the lane
    assert conn._slab is slab1


def test_slab_cap_drains_oversized_load_in_passes_byte_exact(daemon):
    """A load bigger than the staging cap drains through the slab in
    cap-sized passes over the REAL wire: byte-identical result, correct
    global block mapping across passes, slab never grown past the cap."""
    toks = list(range(400, 425))  # 25 tokens -> aligned 24 -> 6 blocks
    salt = "t-slab-cap"
    conn1 = make_connector(daemon)
    kv1 = fresh_kv()
    for bid, seed in ((0, 401), (1, 402), (2, 403), (3, 404), (4, 405), (5, 406)):
        fill_block(kv1, bid, seed)
    run_step(conn1, StubRequest("cap-store", toks, salt), [0, 1, 2, 3, 4, 5], kv1)
    conn1.shutdown()

    conn2 = make_connector(daemon)
    force_slab(conn2)
    # body_len = n_layers * bytes_per_layer = 2 * 128 = 256 for the test engine
    conn2._staging_bytes = 2 * 256    # 2-block passes -> the 6-block load takes 3
    req = StubRequest("cap-load", toks, salt)
    kv2 = fresh_kv()
    assert conn2.get_num_new_matched_tokens(req, 0)[0] == 24
    run_step(conn2, req, [2, 3, 4, 5, 6, 7], kv2)
    for j, src_bid in enumerate((0, 1, 2, 3, 4, 5)):
        for name in LAYERS:
            assert torch.equal(kv2[name][2 + j], kv1[name][src_bid]), \
                f"{name}: pass-local index leaked into the global block mapping (block {j})"
    assert conn2.get_block_ids_with_load_errors() == set()
    assert conn2._slab is not None and conn2._slab.numel() <= 512  # never past the cap
    conn2.shutdown()


def test_scratch_setup_failures_latch_chunked_off_after_three():
    """3 CONSECUTIVE chunked-setup failures latch the chunked path off for
    the connector's lifetime (no more alloc attempts); the per-block-from-slab
    copies keep landing bytes the whole time."""
    nblocks = 10
    conn, req, names, bpl = slab_load_fixture(nblocks)
    calls = []

    def boom(dev, n_layers, bytes_per_layer):
        calls.append(1)
        raise RuntimeError("cudaMalloc failed")

    conn._scratch_ring = boom
    statuses = [kp.Status.OK] * nblocks
    for _ in range(3):
        assert conn._scatter_slab(req, names, bpl, statuses) is False
    assert conn._chunked_disabled and len(calls) == 3
    conn._scatter_slab(req, names, bpl, statuses)
    assert len(calls) == 3  # latched: setup never re-attempted
    ref_kv = {n: torch.zeros_like(t) for n, t in conn._layer_kv.items()}
    scatter_reference(conn, req, names, bpl, statuses, ref_kv)
    for name in names:
        assert same_bytes(conn._layer_kv[name], ref_kv[name])  # bytes always landed
    assert conn._load_errors == set()


def test_scratch_failure_counter_resets_on_success():
    """A success between failures resets the consecutive counter: 2 failures,
    1 success, 2 failures never reaches the 3-in-a-row latch."""
    nblocks = 8
    conn, req, names, bpl = slab_load_fixture(nblocks)
    real = KvblockdConnector._scratch_ring
    fail = {"on": True}

    def flaky(dev, n_layers, bytes_per_layer):
        if fail["on"]:
            raise RuntimeError("transient scratch OOM")
        return real(conn, dev, n_layers, bytes_per_layer)

    conn._scratch_ring = flaky
    statuses = [kp.Status.OK] * nblocks
    conn._scatter_slab(req, names, bpl, statuses)
    conn._scatter_slab(req, names, bpl, statuses)          # 2 consecutive failures
    fail["on"] = False
    assert conn._scatter_slab(req, names, bpl, statuses) is True  # success resets
    fail["on"] = True
    conn._scatter_slab(req, names, bpl, statuses)
    conn._scatter_slab(req, names, bpl, statuses)          # 2 more — not 3 in a row
    assert not conn._chunked_disabled
    assert conn._scratch_fails == 2


# ------------------------------------------------------------ path indicator

def test_load_path_logged_once_and_switch_logged_once(daemon, caplog):
    """Exactly one machine-readable 'kvblockd load path:' INFO line on the
    first completed load; one more (once) if the path degrades mid-run."""
    toks = list(range(430, 439))
    salt = "t-path"
    conn1 = make_connector(daemon)
    kv1 = fresh_kv()
    fill_block(kv1, 0, seed=431)
    fill_block(kv1, 1, seed=432)
    run_step(conn1, StubRequest("pth-store", toks, salt), [0, 1], kv1)
    conn1.shutdown()

    conn2 = make_connector(daemon)
    force_slab(conn2)

    def path_lines():
        return [r.getMessage() for r in caplog.records
                if r.getMessage().startswith("kvblockd load path:")]

    with caplog.at_level(logging.INFO, logger="vllm_kvblockd"):
        run_step(conn2, StubRequest("pth-load-1", toks, salt), [2, 3], fresh_kv())
        assert path_lines() == ["kvblockd load path: chunked-slab"]
        run_step(conn2, StubRequest("pth-load-2", toks, salt), [4, 5], fresh_kv())
        assert len(path_lines()) == 1  # steady state: no repeat
        conn2._chunked_disabled = True  # e.g. the 3-failure latch tripped
        run_step(conn2, StubRequest("pth-load-3", toks, salt), [6, 7], fresh_kv())
        run_step(conn2, StubRequest("pth-load-4", toks, salt), [0, 1], fresh_kv())
        assert path_lines() == [
            "kvblockd load path: chunked-slab",
            "kvblockd load path: per-block (switched from chunked-slab mid-run)",
        ]
    conn2.shutdown()


def test_cpu_load_reports_per_block_path(daemon, caplog):
    """The production CPU lane attributes itself as per-block (job.sh WARNs
    when it sees that on a CUDA run)."""
    toks = list(range(440, 449))
    salt = "t-path-cpu"
    conn1 = make_connector(daemon)
    kv1 = fresh_kv()
    fill_block(kv1, 0, seed=441)
    fill_block(kv1, 1, seed=442)
    run_step(conn1, StubRequest("pc-store", toks, salt), [0, 1], kv1)
    conn1.shutdown()

    conn2 = make_connector(daemon)
    with caplog.at_level(logging.INFO, logger="vllm_kvblockd"):
        run_step(conn2, StubRequest("pc-load", toks, salt), [2, 3], fresh_kv())
    assert [r.getMessage() for r in caplog.records
            if r.getMessage().startswith("kvblockd load path:")] == \
        ["kvblockd load path: per-block"]
    conn2.shutdown()


# --------------------------------------------------------- debug byte check

def test_debug_scatter_check_logs_pass_once(monkeypatch, caplog):
    """KVBLOCKD_DEBUG_SCATTER_CHECK=1: after the FIRST chunked-scatter load,
    one PASS/FAIL line comparing a scattered block against its slab source —
    then silence (once per connector). Off by default."""
    nblocks = 10
    conn, req, names, bpl = slab_load_fixture(nblocks)
    statuses = [kp.Status.OK] * nblocks

    def check_lines():
        return [r.getMessage() for r in caplog.records
                if "debug scatter check" in r.getMessage()]

    # Off by default: no probe, no line.
    monkeypatch.delenv("KVBLOCKD_DEBUG_SCATTER_CHECK", raising=False)
    with caplog.at_level(logging.INFO, logger="vllm_kvblockd"):
        conn._scatter_slab(req, names, bpl, statuses)
        assert check_lines() == []
        assert not conn._debug_scatter_checked

        monkeypatch.setenv("KVBLOCKD_DEBUG_SCATTER_CHECK", "1")
        conn._scatter_slab(req, names, bpl, statuses)
        lines = check_lines()
        assert len(lines) == 1 and "PASS" in lines[0]
        conn._scatter_slab(req, names, bpl, statuses)
        assert len(check_lines()) == 1  # once, ever
