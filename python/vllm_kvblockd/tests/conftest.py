"""Shared fixtures: a real kvblockd daemon subprocess (the wire is never
mocked — mirrors python/kvblockd/tests/conftest.py's fixture pattern), and a
fast-instantiation env (the PYTHONHASHSEED probe spawns subprocesses; one
dedicated test exercises it, everything else skips it)."""

from __future__ import annotations

import http.client
import os
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def _skip_hashseed_probe(monkeypatch):
    # The determinism probe is subprocess-based (~200ms); tests that assert on
    # it re-enable it explicitly (see test_config.py).
    monkeypatch.setenv("KVBLOCKD_SKIP_HASHSEED_CHECK", "1")


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_healthz(port: int, timeout: float = 15.0) -> None:
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


def _daemon_bin(tmp: Path) -> Path:
    prebuilt = os.environ.get("KVBLOCKD_TEST_BIN")
    if prebuilt:
        return Path(prebuilt)  # pre-built by the harness: keeps the fixture fast
    if shutil.which("go") is None:
        pytest.skip("go toolchain not available")
    binp = tmp / "kvblockd"
    if not binp.exists():
        subprocess.run(["go", "build", "-o", str(binp), "./cmd/kvblockd"], cwd=_REPO, check=True)
    return binp


def _spawn_daemon(tmp: Path, data_port: int | None = None) -> dict:
    """Start one DRAM-only daemon; returns connection facts + the Popen handle.
    data_port pins the listen port (the chaos tests restart a daemon on the
    SAME endpoint to prove the client redials its way back to health)."""
    binp = _daemon_bin(tmp)
    dp = data_port if data_port is not None else _free_port()
    mp = _free_port()
    (tmp / "ns.yaml").write_text("namespaces:\n  - { name: vllm, id: 1, token: tok }\n")
    (tmp / f"cfg-{dp}-{mp}.yaml").write_text(
        f'listen_addr: "127.0.0.1:{dp}"\nmetrics_addr: "127.0.0.1:{mp}"\n'
        f'admin_addr: ""\n'  # disabled: default 9441 is FIXED and collides with the
        # kvblockd suite's session daemon when both suites run in one pytest invocation
        f"dram_arena_bytes: 134217728\npinned_bytes_cap: 33554432\n"
        f'namespaces_path: "{tmp / "ns.yaml"}"\n'
    )
    proc = subprocess.Popen(
        [str(binp), "-config", str(tmp / f"cfg-{dp}-{mp}.yaml")],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _wait_healthz(mp)
    return {"host": "127.0.0.1", "port": dp, "namespace": "vllm", "token": "tok",
            "proc": proc}


def _reap(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


@pytest.fixture(scope="session")
def daemon():
    tmp = Path(tempfile.mkdtemp(prefix="kvb-vllm-"))
    info = _spawn_daemon(tmp)
    try:
        yield {k: info[k] for k in ("host", "port", "namespace", "token")}
    finally:
        _reap(info["proc"])
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def chaos_daemon():
    """Function-scoped daemon the test may kill/pause — NEVER the shared
    session daemon. info["respawn"](port) starts a fresh daemon (same or a
    pinned port) and registers it for teardown."""
    tmp = Path(tempfile.mkdtemp(prefix="kvb-chaos-"))
    procs: list[subprocess.Popen] = []
    info = _spawn_daemon(tmp)
    procs.append(info["proc"])

    def respawn(port: int | None = None) -> dict:
        fresh = _spawn_daemon(tmp, data_port=port)
        procs.append(fresh["proc"])
        return fresh

    info["respawn"] = respawn
    try:
        yield info
    finally:
        for proc in procs:
            try:
                _reap(proc)
            except OSError:
                pass
        shutil.rmtree(tmp, ignore_errors=True)
