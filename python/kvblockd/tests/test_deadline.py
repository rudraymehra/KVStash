"""GET deadline: the op timeout bounds each recv, but a server that keeps
TRICKLING bytes passes every per-recv check forever — the per-call deadline
is what turns that into a bounded miss. Proven against a hand-rolled trickle
server over a socketpair (the real daemon cannot be told to stall)."""

from __future__ import annotations

import socket
import struct
import threading
import time
from types import SimpleNamespace

import pytest

from kvblockd import protocol as p
from kvblockd.client import Client, _Conn
from kvblockd.errors import ConnectionLost

BLOB_LEN = 64
N_KEYS = 6
TRICKLE_S = 0.4  # per-block delivery gap: each recv succeeds well inside any
                 # sane op timeout, so ONLY a whole-call deadline can stop it


def _recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed")
        buf += chunk
    return buf


def _trickle_server(sock, n_keys, blob_len):
    """Answer one BATCH_GET with a valid single-frame response whose payload
    arrives one blob every TRICKLE_S — a 'healthy', maddeningly slow server."""
    try:
        hdr = p.Header.parse(_recv_exact(sock, p.HEADER_SIZE))
        _recv_exact(sock, hdr.payload_len)  # the keylist; content irrelevant
        head = p.pack_preamble(p.Status.OK, n_keys) + struct.pack("<II", 0, n_keys)
        descs = b"".join(p._DESC.pack(p.Status.OK, blob_len, 0) for _ in range(n_keys))
        rh = p.Header(p.Op.BATCH_GET, flags=p.F_RESP, ns=1, request_id=1)
        rh.payload_len = len(head) + len(descs) + n_keys * blob_len
        sock.sendall(rh.pack() + head + descs)
        for _ in range(n_keys):
            time.sleep(TRICKLE_S)
            sock.sendall(bytes(blob_len))
    except OSError:
        return  # client gave up (the point of the test)


def test_batch_get_deadline_bounds_a_trickling_server():
    a, b = socket.socketpair()
    a.settimeout(5.0)  # per-recv op timeout: NEVER hit here — that's the trap
    limits = SimpleNamespace(max_batch_keys=64, max_frame_len=1 << 20,
                             max_blob_len=1 << 20, initial_credit=0, features=0)
    conn = _Conn(a, limits, namespace_id=1, verify=False)
    t = threading.Thread(target=_trickle_server, args=(b, N_KEYS, BLOB_LEN), daemon=True)
    t.start()
    keys = [bytes([i]) * 32 for i in range(N_KEYS)]
    landed = {}

    def alloc(idx, prefix, body_len):
        buf = bytearray(body_len)
        landed[idx] = buf
        return memoryview(buf)

    deadline_s = 0.6
    t0 = time.monotonic()
    with pytest.raises(ConnectionLost):  # mid-stream abandon MUST evict the conn
        conn.batch_get_scatter(keys, 0, alloc, deadline=time.monotonic() + deadline_s)
    elapsed = time.monotonic() - t0
    # Bound: the deadline plus at most one trickle grain already in flight —
    # nowhere near the N_KEYS * TRICKLE_S = 2.4s a deadline-less drain takes.
    assert elapsed < deadline_s + TRICKLE_S + 0.5, f"drain ran {elapsed:.2f}s"
    assert len(landed) < N_KEYS  # it really did abandon mid-batch
    a.close()
    b.close()


def _midbody_trickle_server(sock, blob_len, grain_s):
    """Answer one BATCH_GET whose header/preamble/descriptor arrive instantly,
    then trickle the SINGLE blob's body one byte per grain_s — the pathological
    server the per-frame/per-descriptor deadline checks are blind to, because
    the whole stall happens inside one block's recv loop."""
    try:
        hdr = p.Header.parse(_recv_exact(sock, p.HEADER_SIZE))
        _recv_exact(sock, hdr.payload_len)  # the keylist; content irrelevant
        head = p.pack_preamble(p.Status.OK, 1) + struct.pack("<II", 0, 1)
        desc = p._DESC.pack(p.Status.OK, blob_len, 0)
        rh = p.Header(p.Op.BATCH_GET, flags=p.F_RESP, ns=1, request_id=1)
        rh.payload_len = len(head) + len(desc) + blob_len
        sock.sendall(rh.pack() + head + desc)
        for _ in range(blob_len):
            time.sleep(grain_s)
            sock.sendall(b"\x00")
    except OSError:
        return  # client gave up (the point of the test)


def test_deadline_bounds_a_mid_body_trickle():
    """One byte per 0.05s INSIDE one blob's body: each recv succeeds well
    inside any sane op timeout, and after the descriptors no between-frame or
    between-descriptor deadline check ever runs again — only a deadline
    threaded into the recv loop itself can stop it. Full delivery would take
    ~3s; the 0.4s deadline must turn it into a bounded ConnectionLost, never a
    COMPLETE at 3s."""
    grain_s = 0.05
    blob_len = 60  # 60 bytes x 0.05s = 3s full delivery
    deadline_s = 0.4
    a, b = socket.socketpair()
    a.settimeout(5.0)  # per-recv op timeout: NEVER hit here — that's the trap
    limits = SimpleNamespace(max_batch_keys=64, max_frame_len=1 << 20,
                             max_blob_len=1 << 20, initial_credit=0, features=0)
    conn = _Conn(a, limits, namespace_id=1, verify=False)
    t = threading.Thread(target=_midbody_trickle_server, args=(b, blob_len, grain_s),
                         daemon=True)
    t.start()

    def alloc(idx, prefix, body_len):
        return memoryview(bytearray(body_len))

    t0 = time.monotonic()
    with pytest.raises(ConnectionLost):  # mid-body abandon MUST evict the conn
        conn.batch_get_scatter([bytes(32)], 0, alloc,
                               deadline=time.monotonic() + deadline_s)
    elapsed = time.monotonic() - t0
    # Bound: the deadline plus one trickle grain and slack — nowhere near the
    # 3s a deadline-blind body drain takes.
    assert elapsed < deadline_s + 0.6, f"mid-body drain ran {elapsed:.2f}s"
    a.close()
    b.close()


def test_client_expired_deadline_is_all_misses(daemon):
    """Client level: a deadline already in the past yields NOT_FOUND for every
    key WITHOUT starting a tile — no request goes out, and (the observable
    part) no pooled connection gets evicted. Skipping the shard check would
    still miss via _Conn's own guard, but each such call would send a doomed
    request and burn one healthy connection per tile."""
    cl = Client(daemon["addr"], namespace=daemon["namespace"], token=daemon["token"],
                streams=2)
    try:
        key = bytes(range(32))
        assert cl.put(key, b"x" * 128) in (p.Status.OK, p.Status.OK_EXISTS)
        calls = []

        def alloc(idx, prefix, body_len):
            calls.append(idx)
            return memoryview(bytearray(body_len))

        idle_before = list(cl._pool._idle)
        assert idle_before  # the put primed at least one pooled conn
        statuses = cl.batch_get_scatter([key], 0, alloc,
                                        deadline=time.monotonic() - 1.0)
        assert statuses == [p.Status.NOT_FOUND]
        assert calls == []  # nothing was fetched
        assert list(cl._pool._idle) == idle_before  # no conn touched/evicted
        # and WITHOUT a deadline the same key still serves (nothing broke):
        assert cl.batch_get_scatter([key], 0, alloc) == [p.Status.OK]
    finally:
        cl.close()
