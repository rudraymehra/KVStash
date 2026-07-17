"""Connector unit suite: fake MemoryObjs + a real kvblockd daemon subprocess,
NO vLLM. Proves the never-raise contract, ref_count_down discipline, and the
metadata round-trip. torch is required (MemoryObj tensors); lmcache is not
(we fake the MemoryObj surface)."""

from __future__ import annotations

import asyncio
import http.client
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from lmcache_kvblockd.connector import KvblockdRemoteConnector  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]


# --- fakes modeled on LMCache's test doubles ---
class FakeKey:
    def __init__(self, i):
        self.fmt = "vllm"
        self.model_name = "facebook/opt-125m"
        self.world_size = 1
        self.worker_id = 0
        self.chunk_hash = 1000 + i


class FakeMeta:
    fmt = 2  # a MemoryFormat value


class FakeMemoryObj:
    def __init__(self, tensor):
        self.tensor = tensor
        self.metadata = FakeMeta()
        self.ref_downs = 0

    def ref_count_down(self):
        self.ref_downs += 1


class FakeBackend:
    """Stands in for LMCache's LocalCPUBackend pinned allocator. Accepts the
    optional fmt arg so the mem_fmt round-trip path is exercisable."""

    def __init__(self):
        self.allocated = []
        self.fmts = []

    def allocate(self, shape, dtype, fmt=None):
        self.fmts.append(fmt)
        obj = FakeMemoryObj(torch.zeros(shape, dtype=dtype))
        self.allocated.append(obj)
        return obj


# --- daemon fixture (duplicated small helper; mirrors kvblockd/tests/conftest) ---
def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_healthz(port, timeout=15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
            c.request("GET", "/healthz")
            if c.getresponse().status == 200:
                return
        except OSError:
            pass
        time.sleep(0.1)
    raise RuntimeError("daemon never healthy")


@pytest.fixture(scope="module")
def daemon():
    if shutil.which("go") is None:
        pytest.skip("go toolchain not available")
    tmp = Path(tempfile.mkdtemp(prefix="kvb-conn-"))
    binp = tmp / "kvblockd"
    subprocess.run(["go", "build", "-o", str(binp), "./cmd/kvblockd"], cwd=_REPO, check=True)
    dp, mp = _free_port(), _free_port()
    (tmp / "ns.yaml").write_text("namespaces:\n  - { name: lm, id: 1, token: tok }\n")
    (tmp / "cfg.yaml").write_text(
        f'listen_addr: "127.0.0.1:{dp}"\nmetrics_addr: "127.0.0.1:{mp}"\n'
        f'dram_arena_bytes: 67108864\npinned_bytes_cap: 16777216\n'
        f'namespaces_path: "{tmp / "ns.yaml"}"\n'
    )
    proc = subprocess.Popen([str(binp), "-config", str(tmp / "cfg.yaml")],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        _wait_healthz(mp)
        yield {"host": "127.0.0.1", "port": dp}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(tmp, ignore_errors=True)


def _connector(daemon, backend=None):
    return KvblockdRemoteConnector(daemon["host"], daemon["port"], "lm", "tok",
                                   local_cpu_backend=backend, streams=2)


def run(coro):
    return asyncio.run(coro)


def test_put_contains_get_roundtrip(daemon):
    backend = FakeBackend()
    conn = _connector(daemon, backend)
    keys = [FakeKey(i) for i in range(3)]
    tensors = [torch.arange(16, dtype=torch.bfloat16) + i for i in range(3)]
    mos = [FakeMemoryObj(t) for t in tensors]

    async def go():
        oks = await conn.batched_put(keys, mos)
        assert oks == 3
        n = await conn.batched_async_contains("x", keys)
        assert n == 3
        got = await conn.batched_get_non_blocking("x", keys)
        assert len(got) == 3
        for i, obj in enumerate(got):
            assert torch.equal(obj.tensor, tensors[i])
        await conn.close()

    run(go())
    # ref_count_down fired exactly once per put.
    assert all(mo.ref_downs == 1 for mo in mos)


def test_metadata_survives_roundtrip(daemon):
    backend = FakeBackend()
    conn = _connector(daemon, backend)
    key = FakeKey(42)
    # DETERMINISTIC content: keys are content-addressed in production (same
    # key ⇒ same bytes), and the daemon is write-once — random content under
    # a fixed key correctly draws ERR_IMMUTABLE_CONFLICT on repeat runs
    # (pytest-repeat found exactly that).
    t = (torch.arange(8 * 64, dtype=torch.float32) * 0.5).reshape(8, 64)
    mo = FakeMemoryObj(t)

    async def go():
        assert await conn.put(key, mo)
        got = await conn.batched_get_non_blocking("x", [key])
        assert len(got) == 1
        assert got[0].tensor.shape == t.shape
        assert got[0].tensor.dtype == t.dtype
        assert torch.equal(got[0].tensor, t)
        await conn.close()

    run(go())


def test_never_raise_daemon_absent():
    # No daemon at this port → every method degrades to a miss, no exception.
    conn = KvblockdRemoteConnector("127.0.0.1", 1, "lm", "tok",
                                   local_cpu_backend=FakeBackend(), streams=1, op_timeout=1.0)

    async def go():
        assert conn.batched_contains([FakeKey(0)]) == 0  # sync method
        assert await conn.batched_async_contains("x", [FakeKey(0)]) == 0
        assert await conn.batched_get_non_blocking("x", [FakeKey(0)]) == []
        assert await conn.exists(FakeKey(0)) is False
        mo = FakeMemoryObj(torch.zeros(4, dtype=torch.float16))
        assert await conn.put(FakeKey(0), mo) is False
        assert mo.ref_downs == 1  # still ref-downed on the failure path
        assert await conn.list() == []
        await conn.close()

    run(go())


def test_never_raise_daemon_killed_midway(daemon):
    conn = _connector(daemon, FakeBackend())
    key = FakeKey(7)
    mo = FakeMemoryObj(torch.ones(32, dtype=torch.bfloat16))
    run(conn.put(key, mo))
    # (Full SIGKILL-mid-batch is covered by the e2e; here we assert a closed
    # client re-dials cleanly rather than raising.)
    run(conn.close())
    conn2 = _connector(daemon, FakeBackend())
    assert conn2.batched_contains([key]) == 1
    run(conn2.close())


def test_unsupported_dtype_is_miss(daemon):
    conn = _connector(daemon, FakeBackend())
    key = FakeKey(99)
    mo = FakeMemoryObj(torch.zeros(4, dtype=torch.complex64))  # not in DTYPE_CODES

    async def go():
        assert await conn.put(key, mo) is False  # skipped, not raised
        assert mo.ref_downs == 1
        await conn.close()

    run(go())


def test_mem_fmt_threaded_to_allocate(daemon, monkeypatch):
    # The retrieval path must pass the stored MemoryFormat to allocate (Opus
    # HIGH: dropping it round-trips a wrong layout). LMCache's enum is absent
    # in this venv, so stub _memory_format to a sentinel and assert allocate
    # receives it.
    import lmcache_kvblockd.connector as conn_mod
    monkeypatch.setattr(conn_mod, "_memory_format", lambda i: ("FMT", i))
    backend = FakeBackend()
    conn = _connector(daemon, backend)
    key = FakeKey(1234)
    t = (torch.arange(64, dtype=torch.bfloat16))
    mo = FakeMemoryObj(t)
    mo.metadata.fmt = 3  # a non-default MemoryFormat value

    async def go():
        assert await conn.put(key, mo)
        got = await conn.batched_get_non_blocking("x", [key])
        assert len(got) == 1
        await conn.close()

    run(go())
    # allocate saw the reconstructed fmt sentinel carrying the stored value 3.
    assert backend.fmts and backend.fmts[-1] == ("FMT", 3)
