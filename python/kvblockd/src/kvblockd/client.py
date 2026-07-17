"""Synchronous KVB1 client — one in-flight request per connection, mirroring
pkg/client's semantics exactly (HELLO negotiation, NOP-skip, release-
cleanliness: a StatusError keeps the conn in sync, anything else evicts it).

The zero-copy read seam is batch_get_scatter: it hands each block's leading
`prefix_len` bytes to an allocator callback, then recv_into's the remaining
tensor bytes straight into the buffer the callback returns — the path the
LMCache connector uses to land bytes in a pinned MemoryObj with no extra copy.
"""

from __future__ import annotations

import socket
import struct
import xxhash

from kvblockd import protocol as p
from kvblockd.errors import ConnectionLost, FatalProtocol
from kvblockd.pool import Pool

_MAX_SCRATCH = 1 << 20  # reused metadata-read buffer cap (mirrors Go maxReadReuse)


class Limits:
    __slots__ = ("max_batch_keys", "max_frame_len", "max_blob_len", "initial_credit", "features")

    def __init__(self, r: p.HelloResp):
        self.max_batch_keys = r.max_batch_keys
        self.max_frame_len = r.max_frame_len
        self.max_blob_len = r.max_blob_len
        self.initial_credit = r.initial_credit
        self.features = r.features


class _Conn:
    """One socket. Not thread-safe: the Pool guarantees single ownership."""

    def __init__(self, sock: socket.socket, limits: Limits, namespace_id: int, verify: bool):
        self._sock = sock
        self.limits = limits
        self.namespace_id = namespace_id
        self._verify = verify
        self._hdr = bytearray(p.HEADER_SIZE)
        self._scratch = bytearray(_MAX_SCRATCH)

    def close(self):
        try:
            self._sock.close()
        except OSError:
            pass

    # --- framing ---
    def _send_frame(self, hdr: p.Header, bufs=()):
        payload = b"".join(bytes(b) for b in bufs)
        hdr.payload_len = len(payload)
        try:
            self._sock.sendall(hdr.pack())
            if payload:
                self._sock.sendall(payload)
        except OSError as e:
            raise ConnectionLost(f"send: {e}") from e

    def _recv_into(self, view: memoryview):
        """Fill view completely or raise ConnectionLost."""
        got = 0
        n = len(view)
        while got < n:
            try:
                r = self._sock.recv_into(view[got:], n - got)
            except OSError as e:
                raise ConnectionLost(f"recv: {e}") from e
            if r == 0:
                raise ConnectionLost("peer closed mid-frame")
            got += r

    def _next_header(self) -> p.Header:
        """Read one response header, skipping unsolicited NOPs (credit-only
        control) unless F_FATAL is set."""
        while True:
            self._recv_into(memoryview(self._hdr))
            h = p.Header.parse(bytes(self._hdr))
            if h.opcode == p.Op.NOP and not (h.flags & p.F_FATAL):
                if h.payload_len:  # a NOP never carries a body, but be safe
                    self._drain(h.payload_len)
                continue
            if h.flags & p.F_FATAL:
                body = self._read_body(h.payload_len)
                st, _ = p.parse_preamble(body) if len(body) >= p.PREAMBLE_SIZE else (p.Status.FATAL_PROTOCOL, 0)
                raise FatalProtocol(f"server fatal: status {st:#x}")
            return h

    def _read_body(self, n: int) -> bytes:
        if n == 0:
            return b""
        buf = self._scratch if n <= _MAX_SCRATCH else bytearray(n)
        view = memoryview(buf)[:n]
        self._recv_into(view)
        return bytes(view)

    def _drain(self, n: int):
        remaining = n
        while remaining:
            chunk = min(remaining, _MAX_SCRATCH)
            self._recv_into(memoryview(self._scratch)[:chunk])
            remaining -= chunk

    # --- verbs ---
    def batch_exists(self, keys: list[bytes]):
        want_bitmap = bool(self.limits.features & p.FEAT_EXISTS_BITMAP)
        self._send_frame(p.Header(p.Op.BATCH_EXISTS, ns=self.namespace_id, request_id=1),
                         [p.pack_keylist(keys)])
        h = self._next_header()
        body = self._read_body(h.payload_len)
        status, count = p.parse_preamble(body)
        if not p.status_ok(status):
            raise p.StatusError(p.Op.BATCH_EXISTS, status)
        n_consec, _ = struct.unpack_from("<II", body, p.PREAMBLE_SIZE)
        per_key = None
        if want_bitmap:
            # Not a packed bitmap: one status byte per key (padded to 8), the
            # AppendExistsResp layout — OK/OK_EXISTS ⇒ present.
            off = p.PREAMBLE_SIZE + 8
            per_key = [p.status_ok(body[off + i]) for i in range(count)]
        return n_consec, per_key

    def batch_get_scatter(self, keys: list[bytes], prefix_len: int, alloc, idx_base: int = 0):
        """For each OK block (key order, F_MORE frames reassembled): read the
        first prefix_len bytes, call alloc(GLOBAL index, prefix, body_len) →
        memoryview or None. A returned view receives the remaining bytes via
        recv_into; None drains the block and marks the key NOT_FOUND. Only OK
        descriptors carry a payload — every non-OK status (NOT_FOUND, EVICTED,
        …) is descriptor-only and maps to a local NOT_FOUND, mirroring Go's
        readGetInto. idx_base offsets the alloc index into the caller's global
        keyspace (so tiled batches never collide). Returns tile-local statuses.
        """
        n = len(keys)
        statuses = [p.Status.NOT_FOUND] * n
        if n == 0:
            return statuses
        self._send_frame(p.Header(p.Op.BATCH_GET, ns=self.namespace_id, request_id=1),
                         [p.pack_keylist(keys)])
        seen = 0
        while True:
            h = self._next_header()
            status, first, total, descs = self._read_region(h, n)
            if not p.status_ok(status):
                raise p.StatusError(p.Op.BATCH_GET, status)
            # Validate the frame window against the request (a desynced or
            # buggy server must not drive an IndexError or a silent short read).
            if total != n:
                raise ConnectionLost(f"GET total_keys {total} != requested {n}")
            if first != seen or first + len(descs) > n:
                raise ConnectionLost(f"GET frame window [{first},{first+len(descs)}) invalid at seen={seen}, n={n}")
            for j, (dstatus, dlen, dxxh) in enumerate(descs):
                local = first + j
                if not p.status_ok(dstatus):  # NOT_FOUND / EVICTED / any non-OK: no payload
                    statuses[local] = p.Status.NOT_FOUND
                    continue
                self._scatter_one(idx_base + local, local, dlen, dxxh, prefix_len, alloc, statuses)
            seen = first + len(descs)
            if not (h.flags & p.F_MORE):
                break
        if seen != n:
            raise ConnectionLost(f"GET returned {seen} of {n} descriptors")
        return statuses

    def _read_region(self, h: p.Header, n: int):
        """Read + parse a GET response region incrementally (the payloads are
        consumed later by _scatter_one). Reads the preamble, then — for an OK
        batch — the index + descriptor array, bounding the descriptor count by
        the requested key count so a bogus u32 can't drive a giant allocation.
        Returns (status, first_index, total_keys, descs)."""
        pre = bytearray(p.PREAMBLE_SIZE)
        self._recv_into(memoryview(pre))
        status, count = p.parse_preamble(bytes(pre))
        if not p.status_ok(status):
            return status, 0, 0, []  # non-OK batch status is preamble-only
        if count > n:
            raise ConnectionLost(f"GET descriptor count {count} exceeds requested {n}")
        rest_len = p._GET_IDX.size + p.DESC_SIZE * count
        rest = bytearray(rest_len)
        self._recv_into(memoryview(rest))
        return p.parse_get_region(bytes(pre) + bytes(rest))

    def _scatter_one(self, gidx, local, dlen, dxxh, prefix_len, alloc, statuses):
        pfx = prefix_len if dlen >= prefix_len else dlen
        prefix = bytearray(pfx)
        if pfx:
            self._recv_into(memoryview(prefix))
        body_len = dlen - pfx
        digest = xxhash.xxh3_64() if self._verify else None
        if digest is not None and pfx:
            digest.update(prefix)
        view = alloc(gidx, bytes(prefix), body_len)
        if view is None:
            if body_len:
                # Drain but still verify if asked (a corrupt miss should evict).
                self._drain_verify(body_len, digest)
            self._check_digest(digest, dxxh, gidx)
            statuses[local] = p.Status.NOT_FOUND
            return
        mv = memoryview(view)
        if len(mv) < body_len:
            raise ConnectionLost(f"alloc view too small: {len(mv)} < {body_len}")
        self._recv_into(mv[:body_len])
        if digest is not None:
            digest.update(mv[:body_len])
        self._check_digest(digest, dxxh, gidx)
        statuses[local] = p.Status.OK

    def _drain_verify(self, n, digest):
        remaining = n
        while remaining:
            chunk = min(remaining, _MAX_SCRATCH)
            self._recv_into(memoryview(self._scratch)[:chunk])
            if digest is not None:
                digest.update(memoryview(self._scratch)[:chunk])
            remaining -= chunk

    def _check_digest(self, digest, want, idx):
        if digest is not None and digest.intdigest() != want:
            raise ConnectionLost(f"block {idx}: xxh3 mismatch (corrupt payload)")

    def batch_get_bytes(self, keys: list[bytes]):
        """Simple whole-payload GET (tests / non-zero-copy callers). Returns
        (values, statuses); a miss is None."""
        out: list[bytes | None] = [None] * len(keys)
        bodies: dict[int, bytearray] = {}

        def alloc(idx, prefix, body_len):  # prefix_len=0 → prefix is always b""
            buf = bytearray(body_len)
            bodies[idx] = buf
            return memoryview(buf)

        statuses = self.batch_get_scatter(keys, prefix_len=0, alloc=alloc)
        for i, s in enumerate(statuses):
            if s == p.Status.OK:
                out[i] = bytes(bodies[i])
        return out, statuses

    def put(self, key: bytes, bufs, ttl_ms: int = 0):
        digest = xxhash.xxh3_64()
        total = 0
        for b in bufs:
            mv = memoryview(b)
            digest.update(mv)
            total += len(mv)
        xxh = digest.intdigest()
        # BEGIN
        self._send_frame(
            p.Header(p.Op.PUT_STREAM, flags=p.with_subop(0, p.PUT_BEGIN), ns=self.namespace_id,
                     request_id=1, key=key),
            [p.pack_put_begin(total, ttl_ms, xxh)],
        )
        h = self._next_header()
        st, _ = p.parse_preamble(self._read_body(h.payload_len))
        if st == p.Status.OK_EXISTS:
            return p.Status.OK_EXISTS  # idempotent hit; no body sent
        if st != p.Status.OK:
            raise p.StatusError(p.Op.PUT_STREAM, st)
        # CHUNKs (bounded by negotiated max_frame_len), then COMMIT.
        cap = self.limits.max_frame_len or (16 << 20)
        for b in bufs:
            mv = memoryview(b)
            for off in range(0, len(mv), cap):
                chunk = mv[off:off + cap]
                self._send_frame(
                    p.Header(p.Op.PUT_STREAM, flags=p.with_subop(0, p.PUT_CHUNK), ns=self.namespace_id,
                             request_id=1, key=key),
                    [chunk],
                )
        self._send_frame(
            p.Header(p.Op.PUT_STREAM, flags=p.with_subop(0, p.PUT_COMMIT), ns=self.namespace_id,
                     request_id=1, key=key),
            [p.pack_put_commit(xxh)],
        )
        h = self._next_header()
        st, _ = p.parse_preamble(self._read_body(h.payload_len))
        if not p.status_ok(st):
            raise p.StatusError(p.Op.PUT_STREAM, st)
        return st

    def _key_status_verb(self, op, keys, flags=0, aux=0):
        self._send_frame(p.Header(op, flags=flags, ns=self.namespace_id, request_id=1),
                         [p.pack_keylist(keys, aux)])
        h = self._next_header()
        body = self._read_body(h.payload_len)
        st, count = p.parse_preamble(body)
        if not p.status_ok(st):
            raise p.StatusError(op, st)
        # to_status (not Status()) — a forward-compat server may emit a per-key
        # code this client predates; decode tolerantly, never crash the stream.
        return [p.to_status(body[p.PREAMBLE_SIZE + i]) for i in range(count)]

    def delete(self, keys, force=False):
        return self._key_status_verb(p.Op.DELETE, keys, flags=(p.F_FORCE if force else 0))

    def touch_lease(self, keys, sub, ttl_ms=0):
        return self._key_status_verb(p.Op.TOUCH_LEASE, keys, flags=p.with_subop(0, sub), aux=ttl_ms)

    def pin(self, keys, sub):
        return self._key_status_verb(p.Op.PIN, keys, flags=p.with_subop(0, sub))

    def stats(self) -> bytes:
        self._send_frame(p.Header(p.Op.STATS, ns=self.namespace_id, request_id=1),
                         [struct.pack("<II", 0, 0)])
        h = self._next_header()
        body = self._read_body(h.payload_len)
        st, count = p.parse_preamble(body)
        if not p.status_ok(st):
            raise p.StatusError(p.Op.STATS, st)
        return body[p.PREAMBLE_SIZE:p.PREAMBLE_SIZE + count]


def _dial_one(addr, namespace, token, features, connect_timeout, op_timeout, verify) -> _Conn:
    host, port = addr
    sock = socket.create_connection((host, port), timeout=connect_timeout)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.settimeout(op_timeout)
    hdr = p.Header(p.Op.HELLO, request_id=1)
    body = p.pack_hello_req(features, 0, 0, token.encode(), namespace.encode(), b"kvblockd-py")
    hdr.payload_len = len(body)
    try:
        sock.sendall(hdr.pack() + body)
    except OSError as e:
        sock.close()
        raise ConnectionLost(f"hello send: {e}") from e
    conn = _Conn(sock, None, 0, verify)  # limits filled below
    try:
        h = conn._next_header()
        resp = p.HelloResp.parse(conn._read_body(h.payload_len))
    except BaseException:
        conn.close()  # HELLO rejection (StatusError/Fatal/Frame) must not leak the fd
        raise
    conn.limits = Limits(resp)
    conn.namespace_id = resp.namespace_id
    return conn


class Client:
    """A pool of connections to one kvblockd namespace."""

    def __init__(self, addr, *, namespace, token, streams=4,
                 connect_timeout=5.0, op_timeout=10.0, verify=True):
        if isinstance(addr, str):
            host, _, port = addr.partition(":")
            addr = (host, int(port))
        feats = p.FEAT_EXISTS_BITMAP  # in-order (no OOO); bitmap for per-key EXISTS

        def factory():
            return _dial_one(addr, namespace, token, feats, connect_timeout, op_timeout, verify)

        self._pool = Pool(factory, streams)
        # Prime one conn so limits are known and auth failures surface at construct.
        c = self._pool.checkout()
        self.limits = c.limits
        self._pool.checkin(c)

    def close(self):
        self._pool.close()

    def _run(self, fn):
        return self._pool.run(fn)

    def _split(self, keys):
        cap = self.limits.max_batch_keys or len(keys)
        for i in range(0, len(keys), cap):
            yield keys[i:i + cap]

    def batch_exists(self, keys):
        if not keys:
            return 0, []
        # Split above the negotiated cap; consecutive-prefix stops at the first
        # miss, so a broken prefix in tile k makes later tiles irrelevant.
        total_consec, per = 0, []
        broken = False
        for tile in self._split(keys):
            nc, pk = self._run(lambda c: c.batch_exists(tile))
            if pk is not None:
                per.extend(pk)
            if not broken:
                total_consec += nc
                if nc < len(tile):
                    broken = True
        return total_consec, (per or None)

    def batch_get_scatter(self, keys, prefix_len, alloc):
        if not keys:
            return []
        out = []
        base = 0
        for tile in self._split(keys):
            # idx_base makes alloc see the GLOBAL index — without it, tiled
            # batches collide in the caller's results keyed by index (the
            # reproduced cross-key corruption).
            b = base
            out.extend(self._run(lambda c, t=tile, bb=b: c.batch_get_scatter(t, prefix_len, alloc, bb)))
            base += len(tile)
        return out

    def batch_get_bytes(self, keys):
        if not keys:
            return [], []
        vals, sts = [], []
        for tile in self._split(keys):
            v, s = self._run(lambda c: c.batch_get_bytes(tile))
            vals.extend(v)
            sts.extend(s)
        return vals, sts

    def put(self, key, data, ttl_ms=0):
        bufs = [data] if isinstance(data, (bytes, bytearray, memoryview)) else list(data)
        return self._run(lambda c: c.put(key, bufs, ttl_ms))

    def delete(self, keys, force=False):
        out = []
        for tile in self._split(keys):
            out.extend(self._run(lambda c: c.delete(tile, force)))
        return out

    def touch_lease(self, keys, sub, ttl_ms=0):
        out = []
        for tile in self._split(keys):
            out.extend(self._run(lambda c: c.touch_lease(tile, sub, ttl_ms)))
        return out

    def pin(self, keys, sub):
        out = []
        for tile in self._split(keys):
            out.extend(self._run(lambda c: c.pin(tile, sub)))
        return out

    def stats(self):
        return self._run(lambda c: c.stats())
