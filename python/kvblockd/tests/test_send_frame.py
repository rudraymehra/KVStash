"""_send_frame vectored path: iovec advance across partial sends, and
byte-identical framing vs the join fallback (the wire must not know which
path built it)."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from kvblockd import client as client_mod
from kvblockd import protocol as p
from kvblockd.client import _Conn
from kvblockd.errors import ConnectionLost


class _SendmsgSock:
    """Accepts at most `cap` bytes per sendmsg — forces the resume loop."""

    def __init__(self, cap):
        self.cap = cap
        self.data = bytearray()
        self.calls = 0

    def gettimeout(self):
        return None

    def sendmsg(self, bufs):
        self.calls += 1
        flat = b"".join(bytes(b) for b in bufs)
        take = min(self.cap, len(flat))
        self.data += flat[:take]
        return take


class _SendallSock:
    def __init__(self):
        self.data = bytearray()

    def gettimeout(self):
        return None

    def sendall(self, b):
        self.data += bytes(b)


def _limits():
    return SimpleNamespace(max_batch_keys=64, max_frame_len=1 << 20,
                           max_blob_len=1 << 20, initial_credit=0, features=0)


def _frame(sock):
    conn = _Conn(sock, _limits(), namespace_id=1, verify=False)
    hdr = p.Header(p.Op.PUT_STREAM, ns=1, request_id=1)
    bufs = [b"abc", memoryview(bytearray(b"defgh")), b"", bytearray(b"ij")]
    conn._send_frame(hdr, bufs)
    return bytes(sock.data)


def test_vectored_partial_sends_reassemble_exactly():
    sock = _SendmsgSock(cap=7)  # smaller than the header: every boundary hit
    wire = _frame(sock)
    h = p.Header.parse(wire[: p.HEADER_SIZE])
    assert h.payload_len == 10
    assert wire[p.HEADER_SIZE:] == b"abcdefghij"
    assert sock.calls > 1  # the resume loop actually ran


def test_vectored_matches_join_fallback(monkeypatch):
    vec = _frame(_SendmsgSock(cap=1 << 20))
    monkeypatch.setattr(client_mod, "_HAS_SENDMSG", False)
    join = _frame(_SendallSock())
    assert vec == join


class _SlowDrainSock:
    """Accepts a trickle of bytes per sendmsg, forever — defeats any PER-CALL
    timeout, so only the whole-frame send deadline can stop it (the ladder's
    live-reproduced unbounded-send scenario)."""

    def __init__(self):
        self.timeouts = []

    def gettimeout(self):
        return 0.2  # finite op timeout: arms the whole-frame deadline

    def settimeout(self, t):
        self.timeouts.append(t)

    def sendmsg(self, bufs):
        time.sleep(0.05)  # always well inside any per-call window
        return 8


def test_send_frame_bounded_by_whole_frame_deadline():
    """A peer draining a few bytes per call must hit ConnectionLost near the
    op timeout — never extend one frame without bound (sendall bounded the
    WHOLE operation; the resume loop must too)."""
    sock = _SlowDrainSock()
    conn = _Conn(sock, _limits(), namespace_id=1, verify=False)
    hdr = p.Header(p.Op.PUT_STREAM, ns=1, request_id=1)
    t0 = time.monotonic()
    with pytest.raises(ConnectionLost):
        conn._send_frame(hdr, [bytes(1 << 20)])
    assert time.monotonic() - t0 < 2.0  # bounded near 0.2s, not unbounded
    # The finally restored the steady-state op timeout after the clamps.
    assert sock.timeouts[-1] == 0.2
