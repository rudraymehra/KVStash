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


@pytest.fixture(scope="session")
def daemon():
    tmp = Path(tempfile.mkdtemp(prefix="kvb-vllm-"))
    prebuilt = os.environ.get("KVBLOCKD_TEST_BIN")
    if prebuilt:
        binp = Path(prebuilt)  # pre-built by the harness: keeps the fixture fast
    else:
        if shutil.which("go") is None:
            pytest.skip("go toolchain not available")
        binp = tmp / "kvblockd"
        subprocess.run(["go", "build", "-o", str(binp), "./cmd/kvblockd"], cwd=_REPO, check=True)
    dp, mp = _free_port(), _free_port()
    (tmp / "ns.yaml").write_text("namespaces:\n  - { name: vllm, id: 1, token: tok }\n")
    (tmp / "cfg.yaml").write_text(
        f'listen_addr: "127.0.0.1:{dp}"\nmetrics_addr: "127.0.0.1:{mp}"\n'
        f"dram_arena_bytes: 134217728\npinned_bytes_cap: 33554432\n"
        f'namespaces_path: "{tmp / "ns.yaml"}"\n'
    )
    proc = subprocess.Popen(
        [str(binp), "-config", str(tmp / "cfg.yaml")],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_healthz(mp)
        yield {"host": "127.0.0.1", "port": dp, "namespace": "vllm", "token": "tok"}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(tmp, ignore_errors=True)
