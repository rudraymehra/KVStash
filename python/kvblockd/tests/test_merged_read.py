"""Merged next-prefix reads (scatter-gather recv): unit tests over scripted
sockets. The real-daemon suite (test_client_daemon's mixed-sizes case) covers
the same path end-to-end; these pin the syscall count — the whole point of
the merge — and the spill arithmetic under partial reads, which a healthy
loopback daemon never exercises."""

from __future__ import annotations

import threading

import pytest
import xxhash

import kvblockd.client as client_mod
from kvblockd import protocol as p
from kvblockd.client import _Conn


class ScriptedSock:
    """A byte-stream player: the recv side serves `reply` in chunks of at
    most `chunk` bytes per call (partial fills included, exactly like a
    kernel under load); the send side swallows requests. Counts every
    recv-family syscall — the merged-read assertions hang off that."""

    def __init__(self, reply: bytes, chunk: int = 1 << 20, op_timeout=5.0):
        self.reply = memoryview(bytes(reply))
        self.off = 0
        self.chunk = chunk
        self.recv_calls = 0
        self._t = op_timeout

    def gettimeout(self):
        return self._t

    def settimeout(self, t):
        self._t = t

    def sendmsg(self, bufs):
        return sum(len(b) for b in bufs)

    def sendall(self, b):
        return None

    def _serve(self, views) -> int:
        self.recv_calls += 1
        budget = min(self.chunk, len(self.reply) - self.off)
        served = 0
        for v in views:
            if budget <= 0:
                break
            take = min(len(v), budget)
            v[:take] = self.reply[self.off:self.off + take]
            self.off += take
            budget -= take
            served += take
        return served

    def recv_into(self, view, n):
        return self._serve([view[:n]])

    def recvmsg_into(self, buffers, ancbufsize=0, flags=0):
        return self._serve([memoryview(b) for b in buffers]), [], 0, None


def build_get_frame(entries, first_index=0, total=None, more=False) -> bytes:
    """One BATCH_GET response frame. entries: [(status, payload bytes)];
    non-OK entries are payload-free on the wire (descriptor only), exactly
    like the server. Descriptor xxh3 covers the WHOLE payload."""
    descs = bytearray()
    payloads = bytearray()
    for st, pl in entries:
        if p.status_ok(st):
            descs += p._DESC.pack(st, len(pl), xxhash.xxh3_64(pl).intdigest())
            payloads += pl
        else:
            descs += p._DESC.pack(st, 0, 0)
    body = (p.pack_preamble(p.Status.OK, len(entries))
            + p._GET_IDX.pack(first_index, total if total is not None else len(entries))
            + descs + payloads)
    flags = p.F_RESP | (p.F_MORE if more else 0)
    hdr = p.Header(p.Op.BATCH_GET, flags=flags, payload_len=len(body))
    return hdr.pack() + bytes(body)


def build_get_reply(entries) -> bytes:
    return build_get_frame(entries)


def build_get_frames(entries, split: int) -> bytes:
    """The same batch windowed across TWO frames at `split` — the server's
    F_MORE shape (first_index advances, total_keys constant)."""
    n = len(entries)
    return (build_get_frame(entries[:split], first_index=0, total=n, more=True)
            + build_get_frame(entries[split:], first_index=split, total=n))


def scripted_conn(reply: bytes, chunk: int = 1 << 20, verify: bool = True):
    sock = ScriptedSock(reply, chunk=chunk)
    return _Conn(sock, None, namespace_id=1, verify=verify), sock


def payload(seed: int, body_len: int, prefix_len: int = 32) -> bytes:
    pfx = bytes((seed + j) % 256 for j in range(prefix_len))
    return pfx + bytes((seed * 31 + j) % 256 for j in range(body_len))


def run_scatter(conn, n_keys, prefix_len=32):
    prefixes, bodies = {}, {}

    def alloc(idx, pfx, body_len):
        prefixes[idx] = pfx
        buf = bytearray(body_len)
        bodies[idx] = buf
        return memoryview(buf)

    sts = conn.batch_get_scatter([bytes([i]) * 32 for i in range(n_keys)],
                                 prefix_len, alloc)
    return sts, prefixes, bodies


def assert_exact(entries, sts, prefixes, bodies, prefix_len=32):
    for i, (st, pl) in enumerate(entries):
        if p.status_ok(st):
            assert sts[i] == p.Status.OK, f"key {i} lost"
            pfx = min(prefix_len, len(pl))
            assert bytes(prefixes[i]) == pl[:pfx], f"key {i}: wrong prefix"
            assert bytes(bodies[i]) == pl[pfx:], f"key {i}: wrong body"
        else:
            assert sts[i] == p.Status.NOT_FOUND


def test_merged_read_deletes_the_per_block_prefix_recv():
    """3 OK blocks, whole reply available: header + preamble + region reads
    are 3 syscalls, then ONE body read per block — the two non-first blocks'
    prefixes ride the previous body's scatter read. 6 total, not 8."""
    entries = [(p.Status.OK, payload(i, 1024)) for i in range(3)]
    conn, sock = scripted_conn(build_get_reply(entries))
    sts, prefixes, bodies = run_scatter(conn, 3)
    assert_exact(entries, sts, prefixes, bodies)
    assert sock.recv_calls == 6, f"expected 6 recv syscalls, got {sock.recv_calls}"


def test_fallback_without_recvmsg_into_stays_exact(monkeypatch):
    """Platforms without recvmsg_into keep the plain path: same bytes, two
    extra prefix recvs."""
    monkeypatch.setattr(client_mod, "_HAS_RECVMSG_INTO", False)
    entries = [(p.Status.OK, payload(i, 1024)) for i in range(3)]
    conn, sock = scripted_conn(build_get_reply(entries))
    sts, prefixes, bodies = run_scatter(conn, 3)
    assert_exact(entries, sts, prefixes, bodies)
    assert sock.recv_calls == 8


@pytest.mark.parametrize("chunk", [7, 31, 32, 33, 1000])
def test_spill_arithmetic_under_partial_reads(chunk):
    """Dribbling kernel: every recv serves at most `chunk` bytes, so the
    final body read lands anywhere relative to the spill boundary — bytes
    must stay exact through every lead handoff."""
    entries = [(p.Status.OK, payload(i, 257 + 13 * i)) for i in range(4)]
    conn, _sock = scripted_conn(build_get_reply(entries), chunk=chunk)
    sts, prefixes, bodies = run_scatter(conn, 4)
    assert_exact(entries, sts, prefixes, bodies)


def test_tiny_blob_bounds_the_overread():
    """The block AFTER a merge target can be smaller than prefix_len: the
    merge must read min(prefix_len, its WHOLE payload) — one byte more would
    swallow the frame that follows. The tiny block itself has no body read,
    so the block after IT pays a plain prefix recv (no merge chain)."""
    entries = [(p.Status.OK, payload(1, 1000)),
               (p.Status.OK, payload(2, 0, prefix_len=20)),  # dlen 20 < 32
               (p.Status.OK, payload(3, 500))]
    conn, _sock = scripted_conn(build_get_reply(entries))
    sts, prefixes, bodies = run_scatter(conn, 3)
    for i in (0, 2):
        assert sts[i] == p.Status.OK
        assert bytes(prefixes[i]) == entries[i][1][:32]
        assert bytes(bodies[i]) == entries[i][1][32:]
    assert sts[1] == p.Status.OK
    assert bytes(prefixes[1]) == entries[1][1]  # whole 20B payload is prefix
    assert bytes(bodies[1]) == b""


def test_merge_skips_payload_free_descriptors():
    """Non-OK descriptors between OK blocks carry no payload: the merge
    target is the NEXT OK block's prefix, straight across the gap."""
    entries = [(p.Status.OK, payload(1, 800)),
               (p.Status.NOT_FOUND, b""),
               (p.Status.EVICTED, b""),
               (p.Status.OK, payload(4, 900))]
    conn, sock = scripted_conn(build_get_reply(entries))
    sts, prefixes, bodies = run_scatter(conn, 4)
    assert_exact(entries, sts, prefixes, bodies)
    # header + preamble + region(+lead) + body0(+block3 prefix) + body3 = 5
    assert sock.recv_calls == 5


def test_merge_stops_at_the_frame_boundary_f_more():
    """THE invariant the merge reasons hardest about: the LAST OK block of a
    frame must stop exactly at its body's end — the next wire bytes are the
    continuation frame's HEADER. Two-frame reply (F_MORE), 2+2 blocks: the
    syscall count proves block 1 (frame 1's last) did NOT merge (its
    successor block 2's prefix is read separately, off frame 2's region
    read), while blocks 0 and 2 did."""
    entries = [(p.Status.OK, payload(i, 1024)) for i in range(4)]
    conn, sock = scripted_conn(build_get_frames(entries, split=2))
    sts, prefixes, bodies = run_scatter(conn, 4)
    assert_exact(entries, sts, prefixes, bodies)
    # Frame 1: header + preamble + region(+lead=blk0 prefix) + body0(+blk1
    # prefix merged) + body1 (NO merge — frame ends) = 5.
    # Frame 2: header + preamble + region(+lead=blk2 prefix) + body2(+blk3
    # prefix merged) + body3 = 5.
    assert sock.recv_calls == 10, f"expected 10 recv syscalls, got {sock.recv_calls}"


def test_lying_descriptors_are_refused_before_any_body_read():
    """Hostile-server armor: a frame whose OK descriptor lengths do not
    account for its declared payload_len EXACTLY is refused at the region
    read — before any payload byte lands in a caller buffer — because every
    later read (merged over-reads included) would be consuming bytes that
    are not this frame's payload."""
    from kvblockd.errors import ConnectionLost

    entries = [(p.Status.OK, payload(1, 500)), (p.Status.OK, payload(2, 300))]
    frame = bytearray(build_get_reply(entries))
    # Inflate desc 0's length field by 64 without adding payload bytes.
    desc0_len_off = p.HEADER_SIZE + p.PREAMBLE_SIZE + p._GET_IDX.size + 4
    cur = int.from_bytes(frame[desc0_len_off:desc0_len_off + 4], "little")
    frame[desc0_len_off:desc0_len_off + 4] = (cur + 64).to_bytes(4, "little")
    conn, _sock = scripted_conn(bytes(frame))
    calls = []

    def alloc(idx, pfx, body_len):
        calls.append(idx)
        return memoryview(bytearray(body_len))

    with pytest.raises(ConnectionLost, match="does not match its"):
        conn.batch_get_scatter([bytes([i]) * 32 for i in range(2)], 32, alloc)
    assert calls == []  # refused before any alloc/body read


def test_merged_read_with_deadline_armed_stays_exact():
    """The deadline clamp threads through the recvmsg_into loop exactly as
    through recv_into (the re-arm-only-when-binding logic is shared) — a far
    deadline must not change a single byte."""
    import time

    entries = [(p.Status.OK, payload(i, 700)) for i in range(3)]
    conn, _sock = scripted_conn(build_get_reply(entries), chunk=113)
    prefixes, bodies = {}, {}

    def alloc(idx, pfx, body_len):
        prefixes[idx] = pfx
        buf = bytearray(body_len)
        bodies[idx] = buf
        return memoryview(buf)

    sts = conn.batch_get_scatter([bytes([i]) * 32 for i in range(3)], 32, alloc,
                                 deadline=time.monotonic() + 3600.0)
    assert_exact(entries, sts, prefixes, bodies)


def test_reused_hasher_stays_correct_across_calls():
    """The per-conn xxh3 hasher is reset per block: two back-to-back drains
    on one conn must verify every digest (a stale hasher state would trip
    _check_digest and evict)."""
    entries1 = [(p.Status.OK, payload(i, 300)) for i in range(2)]
    entries2 = [(p.Status.OK, payload(i + 9, 700)) for i in range(3)]
    conn, _sock = scripted_conn(build_get_reply(entries1) + build_get_reply(entries2),
                                verify=True)
    sts1, pfx1, bod1 = run_scatter(conn, 2)
    assert_exact(entries1, sts1, pfx1, bod1)
    sts2, pfx2, bod2 = run_scatter(conn, 3)
    assert_exact(entries2, sts2, pfx2, bod2)


def test_corrupt_body_still_detected_through_merged_reads():
    """The merge must not weaken verify: flip one body byte and the drain
    evicts with the corruption error, exactly like the plain path."""
    good = payload(1, 1000)
    bad = payload(2, 1000)
    frame = build_get_reply([(p.Status.OK, good), (p.Status.OK, bad)])
    # Flip a byte inside the SECOND body (after both descs + first payload).
    flip = len(frame) - 10
    frame = frame[:flip] + bytes([frame[flip] ^ 0xFF]) + frame[flip + 1:]
    conn, _sock = scripted_conn(frame, verify=True)
    from kvblockd.errors import ConnectionLost
    with pytest.raises(ConnectionLost, match="xxh3 mismatch"):
        run_scatter(conn, 2)


def test_conn_hasher_is_private_per_connection():
    """Two conns never share a hasher (the Pool's single-owner contract is
    what makes reuse safe — this pins that the reuse stayed per-conn)."""
    c1, _ = scripted_conn(b"", verify=True)
    c2, _ = scripted_conn(b"", verify=True)
    assert c1._xxh is not c2._xxh
    c3, _ = scripted_conn(b"", verify=False)
    assert c3._xxh is None


def test_threaded_conns_do_not_cross_hashers():
    """Fan-out shape: two conns draining concurrently, each verifying — a
    shared hasher would interleave updates and fail both digests."""
    entries_a = [(p.Status.OK, payload(1, 4096)) for _ in range(4)]
    entries_b = [(p.Status.OK, payload(7, 4096)) for _ in range(4)]
    conn_a, _ = scripted_conn(build_get_reply(entries_a), chunk=97)
    conn_b, _ = scripted_conn(build_get_reply(entries_b), chunk=89)
    results = {}

    def drain(name, conn, n):
        results[name] = run_scatter(conn, n)

    ta = threading.Thread(target=drain, args=("a", conn_a, 4))
    tb = threading.Thread(target=drain, args=("b", conn_b, 4))
    ta.start(); tb.start(); ta.join(); tb.join()
    assert_exact(entries_a, *results["a"])
    assert_exact(entries_b, *results["b"])
