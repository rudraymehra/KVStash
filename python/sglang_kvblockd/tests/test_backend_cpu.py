"""KvblockdHiCacheStorage CPU unit suite — fake host pools (upstream
get_page_buffer_meta contract, ported verbatim for page_first) + a real
kvblockd daemon subprocess. NO sglang, NO GPU: torch CPU tensors stand in
for the pinned L2 pool, exactly the W5 fake-MemoryObj pattern.

Covers the four week-12 contract points:
  (a) batch_exists returns the CONSECUTIVE-from-index-0 logical count
  (b) batch_get_v1 round-trips byte-exact into host_indices-addressed slots
  (c) 128-key batch_exists (STORAGE_BATCH_SIZE) answers <1ms p99 on loopback
  (d) never-raise: dead daemon ⇒ False/None/0 returns, no exceptions
plus MLA single-object mapping, fingerprint isolation, size-mismatch → miss,
layout rejection, the generic get/set path, stats, and the v2 stubs.
"""

from __future__ import annotations

import os
import time
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")

from kvblockd.client import Client  # noqa: E402
from sglang_kvblockd import keymap  # noqa: E402
from sglang_kvblockd._compat import STORAGE_BATCH_SIZE  # noqa: E402
from sglang_kvblockd.backend import KvblockdHiCacheStorage  # noqa: E402


# --- fake host pools (mirror memory_pool_host.get_page_buffer_meta) ---------
class FakeMHAPool:
    """MHA page_first host pool: kv_buffer (2, tokens, layer, head, dim);
    per page get_page_buffer_meta yields [k_ptr, v_ptr] — the upstream
    page_first branch ported verbatim."""

    layout = "page_first"

    def __init__(self, n_tokens=64, page_size=4, layer_num=2, head_num=2,
                 head_dim=8, dtype=torch.bfloat16):
        self.page_size = page_size
        self.size = n_tokens
        self.layer_num, self.head_num, self.head_dim = layer_num, head_num, head_dim
        self.dtype = dtype
        self.kv_buffer = torch.zeros((2, n_tokens, layer_num, head_num, head_dim),
                                     dtype=dtype)

    def _per_token(self):
        return self.layer_num * self.head_num * self.head_dim * self.dtype.itemsize

    def get_page_buffer_meta(self, indices):
        assert len(indices) % self.page_size == 0
        ptrs = []
        base = self.kv_buffer.data_ptr()
        idx = indices.tolist()
        v_offset = self.size * self._per_token()
        for i in range(0, len(idx), self.page_size):
            k_ptr = base + idx[i] * self._per_token()
            ptrs.extend([k_ptr, k_ptr + v_offset])
        elem = self.page_size * self._per_token()
        return ptrs, [elem] * len(ptrs)

    # test helpers ------------------------------------------------------
    def page_bytes(self, slot: int):
        """(k_bytes, v_bytes) of the page occupying pool slot `slot`."""
        flat = self.kv_buffer.view(torch.uint8).reshape(2, -1)
        n = self.page_size * self._per_token()
        return (bytes(flat[0, slot * n:(slot + 1) * n].tolist()),
                bytes(flat[1, slot * n:(slot + 1) * n].tolist()))

    def fill_page(self, slot: int, seed: int):
        flat = self.kv_buffer.view(torch.uint8).reshape(2, -1)
        n = self.page_size * self._per_token()
        pat = (torch.arange(n, dtype=torch.int64) * (seed * 2 + 1)) % 251
        flat[0, slot * n:(slot + 1) * n] = pat.to(torch.uint8)
        flat[1, slot * n:(slot + 1) * n] = ((pat + 97) % 251).to(torch.uint8)


class FakeMLAPool:
    layout = "page_first"

    def __init__(self, n_tokens=64, page_size=4, layer_num=2, dim=16,
                 dtype=torch.bfloat16):
        self.page_size = page_size
        self.size = n_tokens
        self.layer_num, self.dim, self.dtype = layer_num, dim, dtype
        self.kv_buffer = torch.zeros((n_tokens, layer_num, 1, dim), dtype=dtype)

    def _per_token(self):
        return self.layer_num * self.dim * self.dtype.itemsize

    def get_page_buffer_meta(self, indices):
        assert len(indices) % self.page_size == 0
        base = self.kv_buffer.data_ptr()
        idx = indices.tolist()
        ptrs = [base + idx[i] * self._per_token()
                for i in range(0, len(idx), self.page_size)]
        elem = self.page_size * self._per_token()
        return ptrs, [elem] * len(ptrs)


def host_indices(slots, page_size):
    """Token-level indices for pool page slots (upstream contract:
    len == n_pages * page_size)."""
    return torch.cat([torch.arange(s * page_size, (s + 1) * page_size,
                                   dtype=torch.int64) for s in slots])


def mha_config(**over):
    base = dict(model_name="test/model-8b", tp_rank=0, tp_size=1, pp_rank=0,
                pp_size=1, is_mla_model=False, is_page_first_layout=True,
                enable_storage_metrics=True, extra_config=None)
    base.update(over)
    return NS(**base)


def make_backend(daemon, cfg=None, pool=None, **extra):
    b = KvblockdHiCacheStorage(cfg or mha_config(), {
        "endpoint": daemon["endpoint"], "namespace": daemon["namespace"],
        "token": daemon["token"], "interface_v1": 1, "op_timeout": 5.0,
        **extra,
    })
    if pool is not None:
        b.register_mem_pool_host(pool)
    return b


def keyset(prefix, n):
    """Chained-SHA-256-shaped logical page keys (64 hex chars)."""
    return [f"{prefix}{i:04d}".ljust(64, "a") for i in range(n)]


def stats_field(stats, name):
    """get_stats() returns a dict without sglang, StorageMetrics with it —
    the suite runs in both environments (the CI leg re-runs post-install)."""
    return stats[name] if isinstance(stats, dict) else getattr(stats, name)


# --- (b) byte-exact zero-copy round-trip ------------------------------------
def test_batch_set_get_v1_roundtrip_byte_exact(daemon):
    src = FakeMHAPool()
    keys = keyset("rt", 4)
    src_slots = [0, 1, 2, 3]
    for s in src_slots:
        src.fill_page(s, seed=s + 1)
    b1 = make_backend(daemon, pool=src)
    assert b1.batch_set_v1(keys, host_indices(src_slots, src.page_size)) == [True] * 4

    # Fresh pool, different (reversed) destination slots: bytes must follow
    # the KEYS, not the slots.
    dst = FakeMHAPool()
    b2 = make_backend(daemon, pool=dst)
    dst_slots = [7, 5, 3, 1]
    assert b2.batch_get_v1(keys, host_indices(dst_slots, dst.page_size)) == [True] * 4
    for i in range(4):
        assert dst.page_bytes(dst_slots[i]) == src.page_bytes(src_slots[i]), f"page {i}"
    b1.close()
    b2.close()


def test_batch_get_v1_partial_miss_prefix(daemon):
    src = FakeMHAPool()
    keys = keyset("pm", 4)
    for s in (0, 1):
        src.fill_page(s, seed=s + 10)
    b = make_backend(daemon, pool=src)
    assert b.batch_set_v1(keys[:2], host_indices([0, 1], src.page_size)) == [True, True]

    dst = FakeMHAPool()
    b2 = make_backend(daemon, pool=dst)
    got = b2.batch_get_v1(keys, host_indices([0, 1, 2, 3], dst.page_size))
    assert got == [True, True, False, False]
    # Missed pages' slots stay untouched (zeros).
    assert dst.page_bytes(2) == (b"\x00" * len(dst.page_bytes(2)[0]),) * 2
    b.close()
    b2.close()


# --- (a) consecutive-prefix batch_exists ------------------------------------
def test_batch_exists_consecutive_from_zero(daemon):
    pool = FakeMHAPool()
    keys = keyset("ex", 4)
    for s in (0, 1, 3):  # hole at page 2: hit,hit,miss,hit ⇒ 2
        pool.fill_page(s, seed=s + 20)
    b = make_backend(daemon, pool=pool)
    assert b.batch_set_v1([keys[0], keys[1], keys[3]],
                          host_indices([0, 1, 3], pool.page_size)) == [True] * 3
    assert b.batch_exists(keys) == 2
    assert b.batch_exists([keys[3]]) == 1  # present, just not prefix-reachable
    assert b.exists(keys[2]) is False
    assert b.exists(keys[0]) is True
    assert b.batch_exists([]) == 0
    b.close()


# --- (c) 128-key probe latency ----------------------------------------------
def test_batch_exists_128_keys_p99_loopback(daemon):
    pool = FakeMHAPool(n_tokens=STORAGE_BATCH_SIZE * 2, page_size=2,
                       layer_num=1, head_num=1, head_dim=4)
    keys = keyset("lat", STORAGE_BATCH_SIZE)
    b = make_backend(daemon, pool=pool)
    slots = list(range(STORAGE_BATCH_SIZE))
    assert all(b.batch_set_v1(keys, host_indices(slots, pool.page_size)))

    for _ in range(10):  # warm-up: dial + server touch
        assert b.batch_exists(keys) == STORAGE_BATCH_SIZE
    lat = []
    for _ in range(200):
        t0 = time.perf_counter()
        n = b.batch_exists(keys)
        lat.append(time.perf_counter() - t0)
        assert n == STORAGE_BATCH_SIZE
    lat.sort()
    p99 = lat[int(len(lat) * 0.99) - 1] * 1000
    budget = float(os.environ.get("KVB_SGL_EXISTS_BUDGET_MS", "1.0"))
    assert p99 < budget, f"128-key batch_exists p99 {p99:.3f}ms >= {budget}ms"
    b.close()


# --- (d) never-raise --------------------------------------------------------
def test_never_raise_daemon_absent():
    pool = FakeMHAPool()
    b = KvblockdHiCacheStorage(mha_config(), {
        "endpoint": "kvblockd://127.0.0.1:1", "namespace": "sgl", "token": "t",
        "interface_v1": 1, "op_timeout": 0.5, "connect_timeout": 0.3,
    })
    b.register_mem_pool_host(pool)
    keys = keyset("nr", 3)
    hi = host_indices([0, 1, 2], pool.page_size)
    assert b.batch_exists(keys) == 0
    assert b.exists(keys[0]) is False
    assert b.batch_get_v1(keys, hi) == [False] * 3
    assert b.batch_set_v1(keys, hi) == [False] * 3
    assert b.get(keys[0]) is None
    assert b.get(keys[0], torch.zeros(8, dtype=torch.uint8)) is None
    assert b.set(keys[0], torch.ones(8, dtype=torch.uint8)) is False
    assert b.batch_get(keys) == [None] * 3
    assert b.batch_set(keys, [torch.ones(4, dtype=torch.uint8)] * 3) is False
    # Tensor CONTAINERS (not just tensor elements) must degrade too — the
    # `or`-coercion regression raised out of these before any wire attempt.
    assert b.batch_get(keys, torch.zeros(3, 4, dtype=torch.uint8)) == [None] * 3
    assert b.batch_set(keys, torch.ones(3, 4, dtype=torch.uint8)) is False
    b.clear()  # no-op, must not raise
    assert b.debug_server_stats() == {}
    stats = b.get_stats()
    assert stats_field(stats, "prefetch_pgs") == []
    assert stats_field(stats, "backup_pgs") == []
    b.close()


def test_methods_before_pool_registration_degrade(daemon):
    b = make_backend(daemon)  # no pool registered
    keys = keyset("np", 2)
    hi = host_indices([0, 1], 4)
    assert b.batch_get_v1(keys, hi) == [False, False]
    assert b.batch_set_v1(keys, hi) == [False, False]
    assert b.batch_exists(keys) == 0  # nothing stored; wire path still fine
    b.close()


# --- MLA mapping -------------------------------------------------------------
def test_mla_single_object_roundtrip(daemon):
    cfg = mha_config(model_name="test/mla-model", is_mla_model=True,
                     tp_rank=2, tp_size=4)
    src = FakeMLAPool()
    keys = keyset("ml", 3)
    flat = src.kv_buffer.view(torch.uint8).reshape(-1)
    flat[:] = (torch.arange(flat.numel(), dtype=torch.int64) % 251).to(torch.uint8)
    b = make_backend(daemon, cfg=cfg, pool=src)
    before = b.debug_server_stats().get("blocks", 0)
    assert b.batch_set_v1(keys, host_indices([0, 1, 2], src.page_size)) == [True] * 3
    assert b.debug_server_stats().get("blocks", 0) - before == 3  # ONE object/page
    assert b.batch_exists(keys) == 3

    dst = FakeMLAPool()
    # A different tp geometry must still hit (MLA KV is TP-invariant).
    cfg2 = mha_config(model_name="test/mla-model", is_mla_model=True,
                      tp_rank=0, tp_size=8)
    b2 = make_backend(daemon, cfg=cfg2, pool=dst)
    assert b2.batch_get_v1(keys, host_indices([2, 1, 0], dst.page_size)) == [True] * 3
    n = src.page_size * src._per_token()
    sflat = src.kv_buffer.view(torch.uint8).reshape(-1)
    dflat = dst.kv_buffer.view(torch.uint8).reshape(-1)
    assert torch.equal(dflat[2 * n:3 * n], sflat[0 * n:1 * n])
    assert torch.equal(dflat[0 * n:1 * n], sflat[2 * n:3 * n])
    b.close()
    b2.close()


# --- isolation + corruption tripwires ----------------------------------------
def test_fingerprint_isolation_no_cross_hit(daemon):
    keys = keyset("iso", 2)
    p1 = FakeMHAPool()
    p1.fill_page(0, seed=3)
    p1.fill_page(1, seed=4)
    b_tp1 = make_backend(daemon, cfg=mha_config(tp_size=1), pool=p1)
    assert all(b_tp1.batch_set_v1(keys, host_indices([0, 1], p1.page_size)))

    # Same namespace, same logical keys, different tp_size ⇒ different bytes
    # would be fetched — the fingerprint must make these MISS instead.
    b_tp2 = make_backend(daemon, cfg=mha_config(tp_size=2), pool=FakeMHAPool())
    assert b_tp2.batch_exists(keys) == 0
    assert b_tp2.batch_get_v1(keys, host_indices([0, 1], 4)) == [False, False]
    b_tp1.close()
    b_tp2.close()


def test_size_mismatch_is_a_miss_not_corruption(daemon):
    pool = FakeMHAPool()
    keys = keyset("sz", 1)
    scheme = keymap.scheme_from_config(mha_config())
    wire = keymap.wire_keys(scheme, keys)
    # Adversarial setup: store WRONG-LENGTH bytes under this page's k AND v
    # keys via the raw client (simulates layout drift between deployments
    # that somehow share a fingerprint).
    ep = daemon["endpoint"].removeprefix("kvblockd://")
    c = Client(ep, namespace=daemon["namespace"], token=daemon["token"], streams=1)
    for w in wire:
        c.put(w, b"short-and-wrong")
    c.close()

    b = make_backend(daemon, pool=pool)
    assert b.batch_exists(keys) == 1  # objects exist...
    got = b.batch_get_v1(keys, host_indices([0], pool.page_size))
    assert got == [False]  # ...but wrong size ⇒ miss, pool untouched
    assert pool.page_bytes(0) == (b"\x00" * len(pool.page_bytes(0)[0]),) * 2
    b.close()


def test_unsupported_layout_disables_v1_paths(daemon):
    pool = FakeMHAPool()
    pool.layout = "layer_first"
    b = make_backend(daemon, pool=pool)
    keys = keyset("lf", 2)
    hi = host_indices([0, 1], pool.page_size)
    assert b.batch_get_v1(keys, hi) == [False, False]
    assert b.batch_set_v1(keys, hi) == [False, False]
    # Generic path is layout-independent and must still work.
    t = torch.arange(64, dtype=torch.uint8)
    assert b.set(keys[0], t) is True
    out = b.get(keys[0], torch.zeros(64, dtype=torch.uint8))
    assert out is not None and torch.equal(out, t)
    b.close()


# --- generic (non-zero-copy) path --------------------------------------------
def test_generic_get_set_roundtrip(daemon):
    b = make_backend(daemon)
    key = keyset("gn", 1)[0]
    t = (torch.arange(256, dtype=torch.int64) % 251).to(torch.uint8).reshape(2, 128)
    assert b.set(key, t) is True
    assert b.set(key, t) is True  # idempotent re-set (OK_EXISTS) is success

    target = torch.zeros(2, 128, dtype=torch.uint8)
    out = b.get(key, target)
    assert out is target and torch.equal(target, t)

    fresh = b.get(key)  # no target: flat uint8 tensor
    assert fresh is not None and torch.equal(fresh.reshape(2, 128), t)

    assert b.get("missing".ljust(64, "f")) is None
    wrong = torch.zeros(16, dtype=torch.uint8)  # size mismatch ⇒ miss
    assert b.get(key, wrong) is None

    assert b.batch_set(keyset("gb", 2), [t, t + 0]) is True
    got = b.batch_get(keyset("gb", 2), [torch.zeros_like(t), torch.zeros_like(t)])
    assert all(g is not None for g in got)
    b.close()


def test_generic_batch_tensor_containers_never_raise(daemon):
    """Regression: batch_get/batch_set probed their CONTAINER arguments with
    `x or [None]*n` — a stacked-tensor container hit Tensor.__bool__
    ("Boolean value of Tensor ... is ambiguous") and the raise escaped the
    public method, which never-raise forbids. Containers must be probed with
    `is None`; a 2-D tensor whose rows are the per-key values/targets (and a
    tensor of sizes) must round-trip."""
    b = make_backend(daemon)
    keys = keyset("tc", 2)
    vals = (torch.arange(2 * 96, dtype=torch.int64) % 251).to(torch.uint8).reshape(2, 96)
    sizes = torch.full((2,), 96, dtype=torch.int64)
    with pytest.raises(RuntimeError):
        bool(vals)  # the container IS truthiness-ambiguous — the old failure mode
    assert b.batch_set(keys, vals, target_sizes=sizes) is True
    targets = torch.zeros(2, 96, dtype=torch.uint8)
    got = b.batch_get(keys, targets, target_sizes=sizes)
    assert all(g is not None for g in got)
    assert torch.equal(targets, vals)
    b.close()


# --- stats + stubs ------------------------------------------------------------
def test_get_stats_accumulates_and_drains(daemon):
    pool = FakeMHAPool()
    keys = keyset("st", 2)
    pool.fill_page(0, seed=5)
    pool.fill_page(1, seed=6)
    b = make_backend(daemon, pool=pool)
    assert all(b.batch_set_v1(keys, host_indices([0, 1], pool.page_size)))
    assert all(b.batch_get_v1(keys, host_indices([2, 3], pool.page_size)))
    s = b.get_stats()  # dict without sglang, StorageMetrics with it
    assert stats_field(s, "backup_pgs") == [2]
    assert stats_field(s, "prefetch_pgs") == [2]
    assert len(stats_field(s, "backup_bandwidth")) == 1
    assert stats_field(s, "backup_bandwidth")[0] > 0
    assert len(stats_field(s, "prefetch_bandwidth")) == 1
    assert stats_field(s, "prefetch_bandwidth")[0] > 0
    s2 = b.get_stats()  # drained on read (mooncake contract)
    assert stats_field(s2, "backup_pgs") == []
    assert stats_field(s2, "prefetch_pgs") == []

    # Re-backing-up existing pages PUTs nothing (write-once dedup) but still
    # reports success.
    assert all(b.batch_set_v1(keys, host_indices([0, 1], pool.page_size)))
    b.close()


def test_v2_stubs_raise_with_upstream_reference(daemon):
    b = make_backend(daemon)
    for fn in (lambda: b.batch_exists_v2(["k"]),
               lambda: b.batch_get_v2([]),
               lambda: b.batch_set_v2([])):
        with pytest.raises(NotImplementedError) as e:
            fn()
        assert "18239" in str(e.value)
        assert "interface_v1" in str(e.value)
    # Registration HOOK must not raise (speculative-decode init calls it).
    b.register_mem_host_pool_v2(FakeMHAPool(), "draft")
    assert "draft" in b.registered_pools
    b.close()
