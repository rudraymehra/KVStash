#!/usr/bin/env python3
"""Gate A — is the client reload path BYTE-bound or BLOCK-bound?

The whole fp8 story hinges on this. fp8 halves the KV bytes per token but
keeps the block COUNT identical (blocks are 16 tokens either way), so:

  * byte-bound  -> halving bytes ~halves reload time -> fp8 roughly doubles
                   every wire/host-bound multiple.
  * block-bound -> per-block cost (syscalls, launches, bookkeeping) dominates
                   and fp8 buys almost nothing on a fast link.

This isolates it WITHOUT a GPU and WITHOUT a paid rig: one local daemon on
loopback, the real wire client, real xxh3 verification, and the same block
COUNT at two block SIZES (bf16-like vs fp8-like). Any per-block overhead is
held constant by construction; only the payload changes.

    python3 bench/e2e/gate_a_bytes_vs_blocks.py            # default 2048 blocks
    BLOCKS=4096 python3 bench/e2e/gate_a_bytes_vs_blocks.py

Reported: effective GB/s and the byte-scaling exponent. Verdict thresholds
are pre-registered below so the answer cannot drift to whatever we hope for.
"""
from __future__ import annotations

import hashlib
import http.client
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python" / "kvblockd" / "src"))

from kvblockd.client import Client

# Pre-registered verdict thresholds (frozen before the run):
#   ratio = t(big) / t(small) for a 2x byte ratio at equal block count.
#   >= 1.70 -> byte-bound (fp8 promoted)
#   <= 1.30 -> block-bound (fp8 on fast links killed)
#   between -> mixed (fp8 partial)
BYTE_BOUND_MIN = 1.70
BLOCK_BOUND_MAX = 1.30

BLOCKS = int(os.environ.get("BLOCKS", "2048"))
BIG = 1_048_576   # 1 MiB  ~ a bf16 block of a 56-KiB/token model at block 16
SMALL = 524_288   # 512 KiB ~ the same block under fp8 (half the bytes)
REPS = int(os.environ.get("REPS", "3"))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_healthz(port: int, timeout: float = 30.0) -> None:
    end = time.time() + timeout
    while time.time() < end:
        try:
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
            c.request("GET", "/healthz")
            if c.getresponse().status == 200:
                return
        except OSError:
            pass
        time.sleep(0.1)
    raise RuntimeError("daemon never became healthy")


def start_daemon(arena_bytes: int):
    tmp = Path(tempfile.mkdtemp(prefix="gateA-"))
    binpath = os.environ.get("KVBLOCKD_TEST_BIN")
    if not binpath:
        binpath = str(tmp / "kvblockd")
        subprocess.run(["go", "build", "-o", binpath, "./cmd/kvblockd"],
                       cwd=REPO, check=True)
    dport, mport = _free_port(), _free_port()
    (tmp / "ns.yaml").write_text("namespaces:\n  - { name: t, id: 7, token: sekret }\n")
    (tmp / "cfg.yaml").write_text(
        f'listen_addr: "127.0.0.1:{dport}"\n'
        f'metrics_addr: "127.0.0.1:{mport}"\n'
        f'admin_addr: ""\n'
        f"dram_arena_bytes: {arena_bytes}\n"
        f"pinned_bytes_cap: {arena_bytes // 4}\n"
        f'namespaces_path: "{tmp / "ns.yaml"}"\n'
        f"max_blob_len: 33554432\n"
    )
    proc = subprocess.Popen([binpath, "-config", str(tmp / "cfg.yaml")],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        _wait_healthz(mport)
    except Exception as e:
        proc.kill()
        err = proc.stderr.read().decode()[-2000:] if proc.stderr else ""
        raise RuntimeError(f"daemon boot failed:\n{err}") from e
    return proc, tmp, dport


def measure(dport: int, blocks: int, block_bytes: int) -> tuple[float, float]:
    """Populate `blocks` of `block_bytes`, then time a full verified reload.

    The reload streams in bounded key batches (default 512 = the server's
    DefaultMaxBatchKeys) and DISCARDS each chunk's payloads after counting
    hits: holding every value, as the first version did, put ~17 GB of
    client RSS beside the daemon's prefaulted arena at BLOCKS=16384 and the
    kernel OOM killer took the daemon mid-read (dmesg banked next to the
    artifacts). Chunking is identical in both arms — equal block count,
    equal batch shape — so the ratio semantics are unchanged, and bounded
    batches are what the real connector sends anyway.
    """
    payload = os.urandom(block_bytes)          # incompressible, per methodology
    # Keys are 32-byte prefix hashes by contract; derive distinct ones per
    # (size, index) so the two arms never share a key.
    keys = [hashlib.sha256(f"gateA-{block_bytes}-{i}".encode()).digest()
            for i in range(blocks)]
    chunk = int(os.environ.get("CHUNK_KEYS", "512"))
    c = Client(("127.0.0.1", dport), namespace="t", token="sekret")
    try:
        for k in keys:
            c.put(k, [payload])
        best = None
        for _ in range(REPS):
            t0 = time.perf_counter()
            hits = 0
            for off in range(0, blocks, chunk):
                vals, _statuses = c.batch_get_bytes(keys[off:off + chunk])
                hits += sum(1 for v in vals if v is not None)
            dt = time.perf_counter() - t0
            if hits != blocks:
                # Explicit raise, not assert: survives python -O, and a miss
                # here means eviction/arena pressure, which would otherwise be
                # published as a bandwidth result.
                raise RuntimeError(
                    f"short read: {hits}/{blocks} hits at {block_bytes}B/block — "
                    "arena too small or eviction active; the timing is invalid")
            best = dt if best is None else min(best, dt)
    finally:
        c.close()
    total = blocks * block_bytes
    return best, total / best / 1e9


def main() -> int:
    # The arena must hold BOTH arms at once (one daemon serves them in
    # sequence): sizing for only the big arm silently evicts and the second
    # arm reads misses — the first version of this script did exactly that and
    # reported the eviction as a result.
    total_bytes = BLOCKS * (BIG + SMALL)
    arena = int(total_bytes * 1.4) + (128 << 20)
    print(f"Gate A: {BLOCKS} blocks x {{{BIG//1024} KiB, {SMALL//1024} KiB}}, "
          f"best of {REPS}, loopback, xxh3 verify ON")
    print("  (equal block COUNT by construction — only payload bytes differ)")
    proc, tmp, dport = start_daemon(arena)
    try:
        t_big, bps_big = measure(dport, BLOCKS, BIG)
        t_small, bps_small = measure(dport, BLOCKS, SMALL)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(tmp, ignore_errors=True)

    ratio = t_big / t_small
    print(f"\n  {BIG//1024:>5} KiB/block: {t_big*1000:8.1f} ms  "
          f"({BLOCKS*BIG/1e9:.2f} GB, {bps_big:.2f} GB/s)")
    print(f"  {SMALL//1024:>5} KiB/block: {t_small*1000:8.1f} ms  "
          f"({BLOCKS*SMALL/1e9:.2f} GB, {bps_small:.2f} GB/s)")
    print(f"\n  time ratio (2x bytes, same block count) = {ratio:.2f}x")
    if ratio >= BYTE_BOUND_MIN:
        verdict = ("BYTE-BOUND on the CLIENT path (loopback: recv + xxh3 + "
                   "memcpy) -> halving KV bytes ~halves this stage. Scope: says "
                   "nothing about the GPU scatter stage, which needs a GPU rig; "
                   "fp8 promoted for the client/wire half")
    elif ratio <= BLOCK_BOUND_MAX:
        verdict = ("BLOCK-BOUND on the CLIENT path -> per-block cost dominates: "
                   "fp8 on fast links is a WASH for this stage (publish bf16, keep "
                   "only the wire-bound fp8 cell)")
    else:
        verdict = ("MIXED -> fp8 helps partially; expect a fractional, not 2x, "
                   "improvement on fast links")
    print(f"  VERDICT: {verdict}")
    print(f"  (pre-registered thresholds: >= {BYTE_BOUND_MIN} byte-bound, "
          f"<= {BLOCK_BOUND_MAX} block-bound)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
