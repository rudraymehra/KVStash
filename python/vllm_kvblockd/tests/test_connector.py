"""Connector suite: stub vLLM surface (Request / SchedulerOutput /
ForwardContext shaped like the vendored v0.25.0 sources) + a REAL kvblockd
daemon — the wire, the chain keys, and the blob layout are never mocked.
torch is required (paged-KV tensors); vllm is not.

Each test uses a distinct cache_salt: kvblockd is write-once and keys are
content-chained, so salts give tests disjoint keyspaces on the shared daemon
— exercising the C-14 mechanism as test isolation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from vllm_kvblockd import connector as conn_mod  # noqa: E402
from vllm_kvblockd.connector import (  # noqa: E402
    BLOB_PREFIX_LEN,
    BlobError,
    KvblockdConnector,
    align_to_block_size,
    decode_blob_prefix,
    encode_blob_prefix,
)

BLOCK = 4          # tokens per block (test-sized engine)
NBLOCKS = 8        # paged blocks per layer
HID = 8            # per-token KV width
LAYERS = ("model.layers.0.attn", "model.layers.1.attn")


# --- vLLM stand-ins (field names mirror upstream_refs/*.py) ---
class StubKTC:
    def __init__(self, port):
        self.kv_connector_extra_config = {
            "kvblockd_endpoint": f"kvblockd://127.0.0.1:{port}",
            "kvblockd_namespace": "vllm",
            "kvblockd_token": "tok",
            "kvblockd_streams": 2,
            "kvblockd_op_timeout_s": 5.0,
        }

    def get_from_extra_config(self, key, default):
        return self.kv_connector_extra_config.get(key, default)


class StubVllmConfig:
    def __init__(self, port):
        self.kv_transfer_config = StubKTC(port)
        self.cache_config = type("C", (), {"block_size": BLOCK, "cache_dtype": "auto"})()
        self.model_config = type(
            "M", (), {"model": "facebook/opt-125m", "dtype": "torch.bfloat16"}
        )()
        self.parallel_config = type("P", (), {"world_size": 1})()


class StubRequest:
    def __init__(self, rid, token_ids, cache_salt=None):
        self.request_id = rid
        self.prompt_token_ids = list(token_ids)
        self.all_token_ids = list(token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_computed_tokens = 0
        self.cache_salt = cache_salt
        self.mm_features = []


class StubNewReq:
    def __init__(self, req: StubRequest, block_ids, num_computed_tokens=0):
        self.req_id = req.request_id
        self.prompt_token_ids = req.prompt_token_ids
        self.mm_features = []
        self.block_ids = (list(block_ids),)  # tuple-of-lists: one KV group
        # In SchedulerOutput this is local + external (assigned post-alloc).
        self.num_computed_tokens = num_computed_tokens


class StubCachedReqs:
    """CachedRequestData shape (v0.25): new tokens only, NEW block ids only;
    for a resumed request new_block_ids[i] is the request's full list."""

    def __init__(self, req_ids, num_computed_tokens, new_block_ids, resumed=()):
        self.req_ids = list(req_ids)
        self.resumed_req_ids = set(resumed)
        self.num_computed_tokens = list(num_computed_tokens)
        self.new_block_ids = list(new_block_ids)


class StubSchedulerOutput:
    def __init__(self, new_reqs, num_scheduled, cached=None):
        self.scheduled_new_reqs = new_reqs
        self.scheduled_cached_reqs = cached
        self.num_scheduled_tokens = num_scheduled


class StubLayer:
    def __init__(self, kv):
        self.kv_cache = kv  # tensor directly (v0.25 ExampleConnector shape)


class StubForwardContext:
    def __init__(self, layer_kv):
        self.no_compile_layers = {n: StubLayer(t) for n, t in layer_kv.items()}
        self.virtual_engine = 0


def fresh_kv():
    """(layer -> paged KV tensor) in FlashAttention-ish layout
    (num_blocks, 2, block_size, hid)."""
    return {n: torch.zeros(NBLOCKS, 2, BLOCK, HID, dtype=torch.bfloat16) for n in LAYERS}


def fill_block(kv, bid, seed):
    g = torch.Generator().manual_seed(seed)
    for t in kv.values():
        t[bid] = torch.rand(2, BLOCK, HID, generator=g).to(torch.bfloat16)


def make_connector(daemon):
    return KvblockdConnector(StubVllmConfig(daemon["port"]), role="scheduler", kv_cache_config=None)


def run_step(conn, request, block_ids, kv, num_computed=0):
    """One scheduler step + one worker step for a single request; returns the
    number of externally-matched tokens the scheduler saw."""
    n, is_async = conn.get_num_new_matched_tokens(request, num_computed)
    assert is_async is False
    conn.update_state_after_alloc(request, None, n)
    out = StubSchedulerOutput(
        [StubNewReq(request, block_ids)],
        {request.request_id: len(request.prompt_token_ids) - num_computed},
    )
    meta = conn.build_connector_meta(out)
    conn.bind_connector_metadata(meta)
    fwd = StubForwardContext(kv)
    conn.start_load_kv(fwd)
    for name, t in kv.items():
        conn.wait_for_layer_load(name)
        conn.save_kv_layer(name, t, None)
    conn.wait_for_save()
    conn.clear_connector_metadata()
    return n


def test_save_then_load_byte_identity(daemon):
    """Round 1 stores; a FRESH connector (fresh-engine sim — this is the
    restart-persistence property at unit scale) hits, loads into DIFFERENT
    physical blocks, bytes identical."""
    toks = list(range(10, 19))  # 9 tokens -> aligned 8 -> 2 blocks
    salt = "t-roundtrip"

    conn1 = make_connector(daemon)
    kv1 = fresh_kv()
    fill_block(kv1, 2, seed=21)
    fill_block(kv1, 5, seed=22)
    n = run_step(conn1, StubRequest("r1", toks, salt), [2, 5], kv1)
    assert n == 0  # cold cache
    conn1.shutdown()

    conn2 = make_connector(daemon)
    req = StubRequest("r2", toks, salt)
    n, _ = conn2.get_num_new_matched_tokens(req, 0)
    assert n == 8  # both blocks hit, full aligned prefix
    kv2 = fresh_kv()
    run_step(conn2, req, [1, 3], kv2)
    for name in LAYERS:
        assert torch.equal(kv2[name][1], kv1[name][2])
        assert torch.equal(kv2[name][3], kv1[name][5])
    assert conn2.get_block_ids_with_load_errors() == set()
    conn2.shutdown()


def test_partial_prefix_hit_count(daemon):
    """Store a 2-block prompt; a longer prompt sharing block 0 only matches
    1 block (consecutive-prefix semantics end at the divergence)."""
    salt = "t-prefix"
    base = [40, 41, 42, 43, 50, 51, 52, 53, 99]  # blocks [40..43], [50..53]
    conn = make_connector(daemon)
    kv = fresh_kv()
    fill_block(kv, 0, seed=31)
    fill_block(kv, 1, seed=32)
    run_step(conn, StubRequest("p1", base, salt), [0, 1], kv)

    forked = [40, 41, 42, 43, 77, 78, 79, 80, 99]  # block 1 diverges
    n, _ = conn.get_num_new_matched_tokens(StubRequest("p2", forked, salt), 0)
    assert n == BLOCK  # exactly one block
    # num_computed_tokens is subtracted (scheduler already has block 0)
    n, _ = conn.get_num_new_matched_tokens(StubRequest("p3", forked, salt), BLOCK)
    assert n == 0
    conn.shutdown()


def test_cache_salt_isolates(daemon):
    """C-14 end-to-end: same tokens under salt A stored, salt B NEVER hits."""
    toks = list(range(60, 69))
    conn = make_connector(daemon)
    kv = fresh_kv()
    fill_block(kv, 6, seed=41)
    fill_block(kv, 7, seed=42)
    run_step(conn, StubRequest("s1", toks, "tenant-a"), [6, 7], kv)

    assert conn.get_num_new_matched_tokens(StubRequest("s2", toks, "tenant-a"), 0)[0] == 8
    assert conn.get_num_new_matched_tokens(StubRequest("s3", toks, "tenant-b"), 0)[0] == 0
    assert conn.get_num_new_matched_tokens(StubRequest("s4", toks, None), 0)[0] == 0
    conn.shutdown()


def test_never_raise_daemon_absent():
    """Dead endpoint: every serving-path method degrades to a miss/no-op."""
    cfg = StubVllmConfig(1)  # nothing listens on port 1
    cfg.kv_transfer_config.kv_connector_extra_config["kvblockd_op_timeout_s"] = 0.5
    conn = KvblockdConnector(cfg, role="scheduler", kv_cache_config=None)
    req = StubRequest("d1", list(range(9)), "t-dead")
    assert conn.get_num_new_matched_tokens(req, 0) == (0, False)
    conn.update_state_after_alloc(req, None, 4)  # pretend a hit was promised
    meta = conn.build_connector_meta(
        StubSchedulerOutput([StubNewReq(req, [0, 1])], {"d1": 9})
    )
    conn.bind_connector_metadata(meta)
    kv = fresh_kv()
    conn.start_load_kv(StubForwardContext(kv))  # must not raise
    # the promised-but-unloadable block is surfaced, not hidden
    assert 0 in conn.get_block_ids_with_load_errors()
    conn.wait_for_save()  # must not raise
    conn.shutdown()


def test_missing_blocks_reported_as_load_errors(daemon):
    """A promised load whose blobs are absent flags exactly those physical
    block ids via get_block_ids_with_load_errors (worker-side honesty)."""
    conn = make_connector(daemon)
    req = StubRequest("m1", list(range(70, 79)), "t-miss")
    conn.update_state_after_alloc(req, None, 8)  # promise 2 blocks, store nothing
    meta = conn.build_connector_meta(
        StubSchedulerOutput([StubNewReq(req, [4, 6])], {"m1": 9})
    )
    conn.bind_connector_metadata(meta)
    conn.start_load_kv(StubForwardContext(fresh_kv()))
    assert conn.get_block_ids_with_load_errors() == {4, 6}
    assert conn.get_block_ids_with_load_errors() == set()  # take-once semantics
    conn.shutdown()


def test_mixed_local_and_remote_hit_loads_tail_range(daemon):
    """BLOCKER regression, pinned to the v0.25 scheduler ordering
    (UPSTREAM.lock): get_num_new_matched_tokens receives the LOCAL prefix-
    cache hit L as its argument while request.num_computed_tokens is still 0,
    and update_state_after_alloc also runs BEFORE num_computed_tokens is
    assigned. With L local + E remote tokens the worker must load logical
    blocks [L/B, (L+E)/B) into their OWN physical slots — loading [0, E/B)
    leaves the tail counted-computed but uninitialized (silent KV corruption)."""
    toks = list(range(200, 213))  # 13 tokens -> aligned 12 -> 3 blocks
    salt = "t-mixed"

    conn1 = make_connector(daemon)
    kv1 = fresh_kv()
    for bid, seed in ((0, 71), (1, 72), (2, 73)):
        fill_block(kv1, bid, seed)
    run_step(conn1, StubRequest("mx-store", toks, salt), [0, 1, 2], kv1)
    conn1.shutdown()

    conn = make_connector(daemon)
    req = StubRequest("mx1", toks, salt)
    assert req.num_computed_tokens == 0  # v0.25: not assigned yet
    n, _ = conn.get_num_new_matched_tokens(req, BLOCK)  # L = 1 local block
    assert n == 2 * BLOCK  # E = blocks 1..2 come from the daemon
    conn.update_state_after_alloc(req, None, n)  # num_computed_tokens STILL 0

    out = StubSchedulerOutput(
        [StubNewReq(req, [4, 5, 6], num_computed_tokens=3 * BLOCK)],
        {"mx1": len(toks) - 3 * BLOCK},
    )
    meta = conn.build_connector_meta(out)
    conn.bind_connector_metadata(meta)
    kv2 = fresh_kv()
    conn.start_load_kv(StubForwardContext(kv2))
    assert conn.get_block_ids_with_load_errors() == set()
    for name in LAYERS:
        # Logical block 0 is the LOCAL hit: vLLM's paged cache owns physical
        # block 4 — the connector must not write it.
        assert torch.count_nonzero(kv2[name][4]) == 0
        assert torch.equal(kv2[name][5], kv1[name][1])
        assert torch.equal(kv2[name][6], kv1[name][2])
    conn.shutdown()


def test_mixed_hit_miss_flags_promised_tail_blocks(daemon):
    """A promised mixed-hit load that cannot be satisfied must flag the
    physical blocks of the PROMISED range [L/B, (L+E)/B) — flagging the head
    instead lets the scheduler trust an unfilled tail."""
    conn = make_connector(daemon)
    toks = list(range(220, 233))  # 3 aligned blocks, nothing stored
    req = StubRequest("mx2", toks, "t-mixed-miss")
    n, _ = conn.get_num_new_matched_tokens(req, BLOCK)  # L = 1 local block
    assert n == 0  # nothing stored under this salt
    conn.update_state_after_alloc(req, None, 2 * BLOCK)  # promise E = 2 blocks
    meta = conn.build_connector_meta(
        StubSchedulerOutput([StubNewReq(req, [4, 5, 6])], {"mx2": len(toks)})
    )
    conn.bind_connector_metadata(meta)
    conn.start_load_kv(StubForwardContext(fresh_kv()))
    assert conn.get_block_ids_with_load_errors() == {5, 6}
    conn.shutdown()


def test_store_bounded_by_scheduled_tokens(daemon):
    """Chunked-prefill guard: only blocks FULLY computed this step are stored
    (never garbage bytes from an uncomputed page)."""
    toks = list(range(80, 89))  # aligned 8 = 2 blocks
    conn = make_connector(daemon)
    req = StubRequest("c1", toks, "t-chunk")
    conn.update_state_after_alloc(req, None, 0)
    # only 5 of 9 tokens scheduled -> only block 0 is complete
    meta = conn.build_connector_meta(StubSchedulerOutput([StubNewReq(req, [0, 1])], {"c1": 5}))
    kv = fresh_kv()
    fill_block(kv, 0, seed=51)
    conn.bind_connector_metadata(meta)
    conn.start_load_kv(StubForwardContext(kv))
    conn.wait_for_save()
    conn.clear_connector_metadata()
    n, _ = conn.get_num_new_matched_tokens(StubRequest("c2", toks, "t-chunk"), 0)
    assert n == BLOCK  # one block stored, not two
    conn.shutdown()


def test_lora_identity_isolates(daemon):
    """KV stored under a LoRA adapter must only hit for that adapter — never
    another adapter, never the base model (the name folds into the seed)."""
    toks = list(range(140, 149))
    conn = make_connector(daemon)
    kv = fresh_kv()
    fill_block(kv, 0, seed=81)
    fill_block(kv, 1, seed=82)
    stored = StubRequest("l1", toks, "t-lora")
    stored.lora_request = SimpleNamespace(lora_name="ad-a")
    run_step(conn, stored, [0, 1], kv)

    same = StubRequest("l2", toks, "t-lora")
    same.lora_request = SimpleNamespace(lora_name="ad-a")
    other = StubRequest("l3", toks, "t-lora")
    other.lora_request = SimpleNamespace(lora_name="ad-b")
    base = StubRequest("l4", toks, "t-lora")  # no lora_request at all
    assert conn.get_num_new_matched_tokens(same, 0)[0] == 8
    assert conn.get_num_new_matched_tokens(other, 0)[0] == 0
    assert conn.get_num_new_matched_tokens(base, 0)[0] == 0
    conn.shutdown()


def test_chunked_prefill_continuation_stores_later_chunks(daemon):
    """Chunk 2+ arrives via scheduled_cached_reqs carrying only the NEW block
    ids; the connector must accumulate the full list across steps and store
    the blocks each step completes — otherwise everything past chunk 1 is
    never cached."""
    toks = list(range(150, 167))  # 17 tokens -> aligned 16 -> 4 blocks
    salt = "t-chunk2"
    conn = make_connector(daemon)
    rid = "ck1"
    req = StubRequest(rid, toks, salt)

    # Step 1: new request, 8 of 17 tokens scheduled, physical blocks [2, 3].
    assert conn.get_num_new_matched_tokens(req, 0)[0] == 0
    conn.update_state_after_alloc(req, None, 0)
    kv = fresh_kv()
    fill_block(kv, 2, seed=91)
    fill_block(kv, 3, seed=92)
    meta = conn.build_connector_meta(
        StubSchedulerOutput([StubNewReq(req, [2, 3])], {rid: 8})
    )
    conn.bind_connector_metadata(meta)
    conn.start_load_kv(StubForwardContext(kv))
    conn.wait_for_save()
    conn.clear_connector_metadata()

    # Step 2: continuation — 9 more tokens, NEW physical blocks [6, 7] only.
    req.num_computed_tokens = 8
    fill_block(kv, 6, seed=93)
    fill_block(kv, 7, seed=94)
    meta = conn.build_connector_meta(
        StubSchedulerOutput(
            [], {rid: 9},
            cached=StubCachedReqs([rid], [8], [([6, 7],)]),
        )
    )
    conn.bind_connector_metadata(meta)
    conn.start_load_kv(StubForwardContext(kv))
    conn.wait_for_save()
    conn.clear_connector_metadata()

    # All FOUR blocks must now hit; a fresh engine loads them byte-identical.
    conn2 = make_connector(daemon)
    req2 = StubRequest("ck2", toks, salt)
    n, _ = conn2.get_num_new_matched_tokens(req2, 0)
    assert n == 16  # chunk-1-only behavior would stop at 8
    conn2.update_state_after_alloc(req2, None, n)
    kv2 = fresh_kv()
    meta = conn2.build_connector_meta(
        StubSchedulerOutput([StubNewReq(req2, [0, 1, 4, 5], num_computed_tokens=16)],
                            {"ck2": 1})
    )
    conn2.bind_connector_metadata(meta)
    conn2.start_load_kv(StubForwardContext(kv2))
    assert conn2.get_block_ids_with_load_errors() == set()
    for name in LAYERS:
        assert torch.equal(kv2[name][0], kv[name][2])
        assert torch.equal(kv2[name][1], kv[name][3])
        assert torch.equal(kv2[name][4], kv[name][6])
        assert torch.equal(kv2[name][5], kv[name][7])
    conn.shutdown()
    conn2.shutdown()


def test_blob_prefix_codec():
    p = encode_blob_prefix("bfloat16", 12, 16, 4096, BLOB_PREFIX_LEN + 12 * 4096)
    assert len(p) == BLOB_PREFIX_LEN
    assert decode_blob_prefix(p) == ("bfloat16", 12, 16, 4096, BLOB_PREFIX_LEN + 12 * 4096)
    with pytest.raises(BlobError):
        decode_blob_prefix(b"XXXX" + p[4:])
    with pytest.raises(BlobError):
        encode_blob_prefix("complex64", 1, 1, 8, 40)
    assert align_to_block_size(9, 4) == 8
    assert align_to_block_size(8, 4) == 4  # last token always recomputed


def test_layout_drift_is_a_miss_not_a_scatter(daemon):
    """A stored blob whose layout doesn't match the live engine (e.g. a config
    change that somehow kept the key) must be refused, not scattered."""
    toks = list(range(90, 99))
    salt = "t-drift"
    conn = make_connector(daemon)
    kv = fresh_kv()
    fill_block(kv, 2, seed=61)
    fill_block(kv, 3, seed=62)
    run_step(conn, StubRequest("g1", toks, salt), [2, 3], kv)

    # Same keys, but the live engine now has HALF the KV width.
    req = StubRequest("g2", toks, salt)
    conn.update_state_after_alloc(req, None, 8)
    meta = conn.build_connector_meta(StubSchedulerOutput([StubNewReq(req, [4, 5])], {"g2": 9}))
    conn.bind_connector_metadata(meta)
    small_kv = {n: torch.zeros(NBLOCKS, 2, BLOCK, HID // 2, dtype=torch.bfloat16) for n in LAYERS}
    conn.start_load_kv(StubForwardContext(small_kv))
    assert conn.get_block_ids_with_load_errors() == {4, 5}
    for t in small_kv.values():
        assert torch.count_nonzero(t) == 0  # nothing was written
    conn.shutdown()


def test_dtype_codes_match_w5():
    """The blob dtype table must stay in lockstep with the W5 adapter's
    (python/lmcache_kvblockd/src/lmcache_kvblockd/meta.py)."""
    lm_meta = pytest.importorskip("lmcache_kvblockd.meta")
    assert conn_mod.DTYPE_CODES == lm_meta.DTYPE_CODES
