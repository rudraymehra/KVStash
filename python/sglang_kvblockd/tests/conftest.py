"""Module fixture: build the real kvblockd daemon and run a DRAM-only
instance the backend tests exchange live frames with (the W5 test pattern).
Skips cleanly when the Go toolchain is absent (wheel-only CI legs)."""

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


def _wait_healthz(port: int, timeout: float = 15.0):
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
    raise RuntimeError("daemon /healthz never went 200")


@pytest.fixture(scope="session")
def daemon():
    if shutil.which("go") is None:
        pytest.skip("go toolchain not available")
    tmp = Path(tempfile.mkdtemp(prefix="kvb-sgl-"))
    binp = tmp / "kvblockd"
    subprocess.run(["go", "build", "-o", str(binp), "./cmd/kvblockd"],
                   cwd=_REPO, check=True)
    dp, mp = _free_port(), _free_port()
    (tmp / "ns.yaml").write_text("namespaces:\n  - { name: sgl, id: 3, token: tok }\n")
    (tmp / "cfg.yaml").write_text(
        f'listen_addr: "127.0.0.1:{dp}"\nmetrics_addr: "127.0.0.1:{mp}"\n'
        f"dram_arena_bytes: 67108864\npinned_bytes_cap: 16777216\n"
        f'namespaces_path: "{tmp / "ns.yaml"}"\n'
    )
    proc = subprocess.Popen([str(binp), "-config", str(tmp / "cfg.yaml")],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        _wait_healthz(mp)
        yield {"endpoint": f"kvblockd://127.0.0.1:{dp}", "namespace": "sgl",
               "token": "tok", "metrics": mp}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(tmp, ignore_errors=True)
