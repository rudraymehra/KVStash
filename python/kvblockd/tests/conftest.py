"""Session fixture: build the real kvblockd daemon and run a DRAM-only
instance the client tests exchange live frames with. Skips cleanly if the Go
toolchain is absent (wheel-only CI legs)."""

from __future__ import annotations

import http.client
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_healthz(host: str, port: int, timeout: float = 15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            c = http.client.HTTPConnection(host, port, timeout=1)
            c.request("GET", "/healthz")
            if c.getresponse().status == 200:
                return
        except OSError:
            pass
        time.sleep(0.1)
    raise RuntimeError("daemon /healthz never went 200")


@pytest.fixture(scope="session")
def daemon():
    if shutil.which("go") is None:
        pytest.skip("go toolchain not available")
    tmp = Path(tempfile.mkdtemp(prefix="kvb-daemon-"))
    binpath = tmp / "kvblockd"
    subprocess.run(
        ["go", "build", "-o", str(binpath), "./cmd/kvblockd"],
        cwd=_REPO, check=True,
    )
    data_port, metrics_port = _free_port(), _free_port()
    (tmp / "ns.yaml").write_text(
        "namespaces:\n  - { name: t, id: 7, token: sekret }\n"
    )
    (tmp / "cfg.yaml").write_text(
        f'listen_addr: "127.0.0.1:{data_port}"\n'
        f'metrics_addr: "127.0.0.1:{metrics_port}"\n'
        f"dram_arena_bytes: 67108864\n"
        f"pinned_bytes_cap: 16777216\n"  # must be <= arena (default 128 MiB > 64 MiB fails)
        f'namespaces_path: "{tmp / "ns.yaml"}"\n'
    )
    proc = subprocess.Popen([str(binpath), "-config", str(tmp / "cfg.yaml")],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        _wait_healthz("127.0.0.1", metrics_port)
        yield {"addr": ("127.0.0.1", data_port), "metrics": metrics_port,
               "namespace": "t", "token": "sekret", "proc": proc}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(tmp, ignore_errors=True)
