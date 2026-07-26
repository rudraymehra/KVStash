"""Synchronous KVB1 client — one in-flight request per connection, mirroring
pkg/client's semantics exactly (HELLO negotiation, NOP-skip, release-
cleanliness: a StatusError keeps the conn in sync, anything else evicts it).

The zero-copy read seam is batch_get_scatter: it hands each block's leading
`prefix_len` bytes to an allocator callback, then recv_into's the remaining
tensor bytes straight into the buffer the callback returns — the path the
LMCache connector uses to land bytes in a pinned MemoryObj with no extra copy.
"""

from __future__ import annotations

import concurrent.futures
import logging
import socket
import struct
import time

import xxhash

from kvblockd import protocol as p
from kvblockd.errors import ConnectionLost, FatalProtocol
from kvblockd.pool import Pool

logger = logging.getLogger("kvblockd.client")

_MAX_SCRATCH = 1 << 20  # reused metadata-read buffer cap (mirrors Go maxReadReuse)
_MAX_GET_FANOUT = 4     # batch_get_scatter shard ceiling (one pooled conn each)


class Limits:
    __slots__ = ("features", "initial_credit", "max_batch_keys", "max_blob_len", "max_frame_len")

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
        self._pfxbuf = bytearray(64)  # reused per-block prefix staging (grown on demand)
        self.granted_rcvbuf = 0  # stamped by _dial_one (the kernel-GRANTED SO_RCVBUF)
        # The steady-state per-recv timeout (what the dialer configured on the
        # socket); restored after any deadline-clamped call. None = blocking.
        self._op_timeout = sock.gettimeout()
        # Per-CALL hard deadline (monotonic), armed by batch_get_scatter and
        # consulted by EVERY _recv_into — header, preamble, descriptors,
        # prefix, body, and drain reads are all bounded in one place. A
        # trickling server passes every per-recv op-timeout check forever;
        # only clamping each recv to the remaining budget stops it.
        self._deadline: float | None = None

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
        """Fill view completely or raise ConnectionLost. When a per-call
        deadline is armed (see __init__/_arm_deadline), each recv is clamped
        to min(op_timeout, remaining budget) and an exhausted budget raises —
        abandoning mid-read leaves the stream state unknown, so the caller
        must evict the connection (Pool.run does, on any non-StatusError)."""
        got = 0
        n = len(view)
        deadline = self._deadline
        while got < n:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ConnectionLost("recv deadline exceeded (conn evicted)")
                try:
                    self._sock.settimeout(remaining if self._op_timeout is None
                                          else min(self._op_timeout, remaining))
                except OSError as e:
                    raise ConnectionLost(f"settimeout: {e}") from e
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

    def batch_get_scatter(self, keys: list[bytes], prefix_len: int, alloc, idx_base: int = 0,
                          deadline: float | None = None):
        """For each OK block (key order, F_MORE frames reassembled): read the
        first prefix_len bytes, call alloc(GLOBAL index, prefix, body_len) →
        memoryview or None. A returned view receives the remaining bytes via
        recv_into; None drains the block and marks the key NOT_FOUND. Only OK
        descriptors carry a payload — every non-OK status (NOT_FOUND, EVICTED,
        …) is descriptor-only and maps to a local NOT_FOUND, mirroring Go's
        readGetInto. idx_base offsets the alloc index into the caller's global
        keyspace (so tiled batches never collide). Returns tile-local statuses.

        deadline (optional, time.monotonic()-based): a hard wall-clock ceiling
        for the WHOLE drain, threaded into _recv_into itself — every read
        (header, preamble, descriptors, prefix, body, drain) is clamped to
        the remaining budget, so a server trickling bytes INSIDE one blob's
        body is cut off just like one stalling between frames. Past the
        deadline this raises ConnectionLost — abandoning mid-stream leaves the
        conn's read state unknown, so evicting it is the only safe degrade
        (the caller's shard turns its remaining keys into misses)."""
        n = len(keys)
        statuses = [p.Status.NOT_FOUND] * n
        if n == 0:
            return statuses
        self._deadline = deadline  # armed for every _recv_into below
        try:
            self._send_frame(p.Header(p.Op.BATCH_GET, ns=self.namespace_id, request_id=1),
                             [p.pack_keylist(keys)])
            seen = 0
            while True:
                if deadline is not None and time.monotonic() > deadline:
                    raise ConnectionLost("GET deadline exceeded between frames (conn evicted)")
                h = self._next_header()
                status, first, total, descs, lead = self._read_region(h, n, prefix_len)
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
                    if deadline is not None and time.monotonic() > deadline:
                        raise ConnectionLost("GET deadline exceeded mid-frame (conn evicted)")
                    lead = self._scatter_one(idx_base + local, local, dlen, dxxh,
                                             prefix_len, alloc, statuses, lead)
                if lead:  # every pre-read byte belongs to some OK payload above
                    raise ConnectionLost(f"GET frame left {len(lead)} unclaimed payload bytes")
                seen = first + len(descs)
                if not (h.flags & p.F_MORE):
                    break
            if seen != n:
                raise ConnectionLost(f"GET returned {seen} of {n} descriptors")
            return statuses
        finally:
            # Disarm and restore the steady-state op timeout: on success the
            # conn is re-pooled and must not carry a stale clamp; on a
            # deadline raise it is evicted anyway, restoring is just hygiene.
            self._deadline = None
            if deadline is not None:
                try:
                    self._sock.settimeout(self._op_timeout)
                except OSError:
                    pass  # dead socket: the conn is being evicted regardless

    def _read_region(self, h: p.Header, n: int, prefix_len: int = 0):
        """Read + parse a GET response region incrementally (the payloads are
        consumed later by _scatter_one). Reads the preamble, then — for an OK
        batch — the index + descriptor array, bounding the descriptor count by
        the requested key count so a bogus u32 can't drive a giant allocation.

        Drain micro-optimization: the frame header's payload_len tells us how
        many payload bytes follow the descriptor region WITHIN THIS FRAME, so
        the first payload's leading min(prefix_len, available) bytes are read
        as part of the same buffered read — the first block of every frame
        skips its separate 32B prefix recv. The over-read is bounded by the
        frame's own length, so it can never swallow a future frame or NOP.
        Returns (status, first_index, total_keys, descs, lead_bytes)."""
        pre = bytearray(p.PREAMBLE_SIZE)
        self._recv_into(memoryview(pre))
        status, count = p.parse_preamble(bytes(pre))
        if not p.status_ok(status):
            return status, 0, 0, [], b""  # non-OK batch status is preamble-only
        if count > n:
            raise ConnectionLost(f"GET descriptor count {count} exceeds requested {n}")
        rest_len = p._GET_IDX.size + p.DESC_SIZE * count
        payload_in_frame = h.payload_len - p.PREAMBLE_SIZE - rest_len
        if payload_in_frame < 0:
            raise ConnectionLost(f"GET frame payload_len {h.payload_len} shorter than its header region")
        extra = min(prefix_len, payload_in_frame) if prefix_len > 0 else 0
        rest = bytearray(rest_len + extra)
        self._recv_into(memoryview(rest))
        region = p.parse_get_region(bytes(pre) + bytes(rest[:rest_len]))
        return (*region, bytes(rest[rest_len:]))

    def _scatter_one(self, gidx, local, dlen, dxxh, prefix_len, alloc, statuses, lead=b""):
        """Consume one OK block's payload. `lead` is payload bytes already
        pulled off the socket by _read_region's merged read; the unconsumed
        remainder is returned for the next block. Because `lead` is at most
        prefix_len bytes, it can only ever hold prefix-region bytes (a block
        with dlen < prefix_len has an empty body), never body bytes — so body
        reads always come straight off the socket into the alloc view."""
        pfx = min(prefix_len, dlen)
        take = min(pfx, len(lead))
        if take == pfx:
            prefix = lead[:pfx]  # fully pre-read: no recv, no staging copy
        else:
            if len(self._pfxbuf) < pfx:
                self._pfxbuf = bytearray(pfx)
            mv = memoryview(self._pfxbuf)[:pfx]
            if take:
                mv[:take] = lead[:take]
            self._recv_into(mv[take:])
            prefix = bytes(mv)
        lead = lead[pfx:] if len(lead) > pfx else b""
        body_len = dlen - pfx
        digest = xxhash.xxh3_64() if self._verify else None
        if digest is not None and pfx:
            digest.update(prefix)
        view = alloc(gidx, prefix, body_len)
        if view is None:
            if body_len:
                # Drain but still verify if asked (a corrupt miss should evict).
                self._drain_verify(body_len, digest)
            self._check_digest(digest, dxxh, gidx)
            statuses[local] = p.Status.NOT_FOUND
            return lead
        mv = memoryview(view)
        if len(mv) < body_len:
            raise ConnectionLost(f"alloc view too small: {len(mv)} < {body_len}")
        self._recv_into(mv[:body_len])
        if digest is not None:
            digest.update(mv[:body_len])
        self._check_digest(digest, dxxh, gidx)
        statuses[local] = p.Status.OK
        return lead

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


def _connect(host, port, connect_timeout, so_rcvbuf=None) -> tuple[socket.socket, int]:
    """create_connection with an OPT-IN receive-buffer override. so_rcvbuf is
    None by default and then setsockopt is never called: per tcp(7), setting
    SO_RCVBUF DISABLES receive-window autotuning and clamps the buffer at
    net.core.rmem_max — which is read-only inside network namespaces (the
    sysctl is not netns-writable), so an unprivileged container silently
    clamps a 4MiB ask to ~416KiB granted, while autotuning via the per-netns
    net.ipv4.tcp_rmem could have grown far beyond it. An unconditional set is
    therefore a pessimization on real-NIC rigs; only ask when the caller
    explicitly knows better. When set, it happens BEFORE connect (so the
    window scales from the SYN). Either way, the EFFECTIVE buffer the kernel
    reports is returned alongside the socket — stamp truth, not assumptions."""
    err = None
    for af, kind, proto, _, sa in socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM):
        sock = None
        try:
            sock = socket.socket(af, kind, proto)
            if so_rcvbuf is not None:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, int(so_rcvbuf))
            sock.settimeout(connect_timeout)
            sock.connect(sa)
            granted = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
            return sock, granted
        except OSError as e:
            err = e
            if sock is not None:
                sock.close()
        except BaseException:  # KeyboardInterrupt/anything: the fd must not leak
            if sock is not None:
                sock.close()
            raise
    raise err if err is not None else OSError(f"getaddrinfo returned nothing for {host}:{port}")


def _dial_one(addr, namespace, token, features, connect_timeout, op_timeout, verify,
              so_rcvbuf=None) -> _Conn:
    host, port = addr
    sock, granted_rcvbuf = _connect(host, port, connect_timeout, so_rcvbuf)
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
    conn.granted_rcvbuf = granted_rcvbuf
    logger.debug("kvblockd conn to %s:%s — SO_RCVBUF asked %s, kernel reports %d",
                 host, port, "autotune (not set)" if so_rcvbuf is None else so_rcvbuf,
                 granted_rcvbuf)
    return conn


class Client:
    """A pool of connections to one kvblockd namespace."""

    def __init__(self, addr, *, namespace, token, streams=4,
                 connect_timeout=5.0, op_timeout=10.0, verify=True, so_rcvbuf=None):
        """so_rcvbuf: OPT-IN SO_RCVBUF override in bytes. Leave None (the
        default) to keep kernel receive-window autotuning — setting it
        disables autotuning and clamps at net.core.rmem_max (see _connect)."""
        if isinstance(addr, str):
            host, _, port = addr.partition(":")
            addr = (host, int(port))
        feats = p.FEAT_EXISTS_BITMAP  # in-order (no OOO); bitmap for per-key EXISTS

        def factory():
            return _dial_one(addr, namespace, token, feats, connect_timeout, op_timeout,
                             verify, so_rcvbuf)

        self._streams = max(int(streams), 1)
        self._pool = Pool(factory, streams)
        # Prime one conn so limits are known and auth failures surface at construct.
        c = self._pool.checkout()
        self.limits = c.limits
        # The EFFECTIVE receive buffer the kernel reports (see _connect) —
        # exposed so rigs can stamp the real window, never an assumption.
        # With so_rcvbuf=None this is the autotune starting size.
        self.granted_rcvbuf = c.granted_rcvbuf
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
            nc, pk = self._run(lambda c: c.batch_exists(tile))  # noqa: B023 — lambda is invoked synchronously inside this iteration (Pool.run calls fn(conn) immediately); it never outlives the loop
            if pk is not None:
                per.extend(pk)
            if not broken:
                total_consec += nc
                if nc < len(tile):
                    broken = True
        return total_consec, (per or None)

    @staticmethod
    def _shard_bounds(n: int, nshards: int) -> list[tuple[int, int]]:
        """Split [0, n) into nshards CONTIGUOUS near-equal (start, end) ranges."""
        base, rem = divmod(n, nshards)
        bounds, s = [], 0
        for i in range(nshards):
            e = s + base + (1 if i < rem else 0)
            bounds.append((s, e))
            s = e
        return bounds

    def batch_get_scatter(self, keys, prefix_len, alloc, deadline=None):
        """Zero-copy batched GET, fanned out across up to 4 CONTIGUOUS key
        shards (bounded by the pool size), one pooled connection per shard,
        each shard internally tiled by the negotiated max_batch_keys.

        deadline (optional, time.monotonic()-based): overall wall-clock
        ceiling for the whole call. Each shard stops starting new tiles past
        it, and a tile caught mid-drain evicts its connection; either way the
        affected keys come back as misses — a slow store must degrade to
        recompute, never to an unbounded trickle.

        THREADING CONTRACT: `alloc` MUST be safe to call concurrently from
        multiple threads with DISTINCT idx values — the global idx is unique
        across shards/tiles by construction, so an allocator whose slots are
        disjoint by index (e.g. a slab) satisfies this for free. xxh3 verify
        runs inline per block, on the shard's thread, BEFORE that block's
        status becomes OK.

        WIRE-ORDER CONTRACT: within a frame, descriptors (statuses) always
        arrive before payloads; a non-OK descriptor is payload-free and never
        invokes alloc. A shard whose connection dies mid-drain yields misses
        for its remaining keys — blocks it already verified stay valid.
        Statuses come back stitched in key order."""
        n = len(keys)
        if n == 0:
            return []
        nshards = min(_MAX_GET_FANOUT, self._streams, n)
        if nshards <= 1:
            return self._scatter_shard(keys, prefix_len, alloc, 0, deadline)
        bounds = self._shard_bounds(n, nshards)
        statuses = [p.Status.NOT_FOUND] * n
        errs = []
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=nshards, thread_name_prefix="kvb-get") as ex:
            futs = [(s, ex.submit(self._scatter_shard, keys[s:e], prefix_len, alloc, s,
                                  deadline))
                    for s, e in bounds]
            for s, fut in futs:
                try:
                    part = fut.result()
                    statuses[s:s + len(part)] = part
                except Exception as exc:  # noqa: BLE001 — joined below; connection deaths were already downgraded to misses in _scatter_shard, so anything here is a real protocol/caller error
                    errs.append(exc)
        if errs:
            raise errs[0]
        return statuses

    def _scatter_shard(self, keys, prefix_len, alloc, idx_base, deadline=None):
        """One shard: current tiled drain. idx_base keeps alloc's index GLOBAL
        — without it, tiled batches collide in the caller's results keyed by
        index (the reproduced cross-key corruption). A dead connection turns
        the shard's REMAINING keys into misses (this shard only); tiles that
        already completed keep their verified statuses. An expired deadline
        stops STARTING tiles here; one caught mid-tile surfaces from _Conn as
        ConnectionLost and lands in the same remaining-keys-are-misses path."""
        out = []
        base = idx_base
        for tile in self._split(keys):
            if deadline is not None and time.monotonic() > deadline:
                out.extend([p.Status.NOT_FOUND] * (len(keys) - len(out)))
                return out
            b = base
            try:
                out.extend(self._run(
                    lambda c, t=tile, bb=b: c.batch_get_scatter(t, prefix_len, alloc, bb,
                                                                deadline)))
            except (ConnectionLost, FatalProtocol, OSError):
                out.extend([p.Status.NOT_FOUND] * (len(keys) - len(out)))
                return out
            base += len(tile)
        return out

    def batch_get_bytes(self, keys):
        if not keys:
            return [], []
        vals, sts = [], []
        for tile in self._split(keys):
            v, s = self._run(lambda c: c.batch_get_bytes(tile))  # noqa: B023 — lambda is invoked synchronously inside this iteration (Pool.run calls fn(conn) immediately); it never outlives the loop
            vals.extend(v)
            sts.extend(s)
        return vals, sts

    def put(self, key, data, ttl_ms=0):
        bufs = [data] if isinstance(data, (bytes, bytearray, memoryview)) else list(data)
        return self._run(lambda c: c.put(key, bufs, ttl_ms))

    def delete(self, keys, force=False):
        out = []
        for tile in self._split(keys):
            out.extend(self._run(lambda c: c.delete(tile, force)))  # noqa: B023 — lambda is invoked synchronously inside this iteration (Pool.run calls fn(conn) immediately); it never outlives the loop
        return out

    def touch_lease(self, keys, sub, ttl_ms=0):
        out = []
        for tile in self._split(keys):
            out.extend(self._run(lambda c: c.touch_lease(tile, sub, ttl_ms)))  # noqa: B023 — lambda is invoked synchronously inside this iteration (Pool.run calls fn(conn) immediately); it never outlives the loop
        return out

    def pin(self, keys, sub):
        out = []
        for tile in self._split(keys):
            out.extend(self._run(lambda c: c.pin(tile, sub)))  # noqa: B023 — lambda is invoked synchronously inside this iteration (Pool.run calls fn(conn) immediately); it never outlives the loop
        return out

    def stats(self):
        return self._run(lambda c: c.stats())
