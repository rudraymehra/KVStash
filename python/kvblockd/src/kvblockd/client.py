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
import threading
import time

import xxhash

from kvblockd import protocol as p
from kvblockd.errors import ConnectionLost, FatalProtocol
from kvblockd.pool import Pool

logger = logging.getLogger("kvblockd.client")

_MAX_SCRATCH = 1 << 20  # reused metadata-read buffer cap (mirrors Go maxReadReuse)
# Default batch_get_scatter shard ceiling (one pooled conn each). Overridable
# per Client via get_fanout — the knob exists for real-NIC rigs where one TCP
# flow caps at ~10 Gbps; iperf showed diminishing returns past 8 flows, so
# callers should stop there. The DEFAULT stays 4 until the loopback A/B rules
# out a regression from 8 recv+xxh3 threads on the syscall/membw-bound path.
_MAX_GET_FANOUT = 4
# Vectored send: header + body in one syscall with no payload copy. Absent
# only on platforms without sendmsg (Windows); the join fallback stays.
_HAS_SENDMSG = hasattr(socket.socket, "sendmsg")
# Rate limit for the degrade-to-miss disclosure lines (one per key per window).
_WARN_INTERVAL_S = 10.0


class ClientCounters:
    """Misses-by-cause telemetry for the degrade-to-miss machinery. The
    client's designed failure mode is a miss, so these counters are the only
    way an operator can tell 'cold cache' from 'store is dying' without log
    archaeology — the LMCache/vLLM connectors scrape them into their stats.

    evictions        pooled connections discarded after a non-StatusError
    deadline_misses  keys downgraded to misses by an expired per-call deadline
    corrupt_blocks   xxh3 digest mismatches (detected corruption; conn evicted)
    degraded_keys    keys downgraded to misses by a shard-level failure
    """

    __slots__ = ("_lock", "corrupt_blocks", "deadline_misses", "degraded_keys", "evictions")

    def __init__(self):
        self._lock = threading.Lock()
        self.evictions = 0
        self.deadline_misses = 0
        self.corrupt_blocks = 0
        self.degraded_keys = 0

    def bump(self, name: str, n: int = 1) -> None:
        with self._lock:
            setattr(self, name, getattr(self, name) + n)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "evictions": self.evictions,
                "deadline_misses": self.deadline_misses,
                "corrupt_blocks": self.corrupt_blocks,
                "degraded_keys": self.degraded_keys,
            }


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

    def __init__(self, sock: socket.socket, limits: Limits, namespace_id: int, verify: bool,
                 counters: ClientCounters | None = None):
        self._sock = sock
        self.limits = limits
        self.namespace_id = namespace_id
        self._verify = verify
        self._counters = counters  # shared Client-level telemetry (None in bare tests)
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
        """Send one frame. The vectored path hands the kernel the header and
        the caller's buffers as one iovec list — a 0.4–2.5MB blob is never
        duplicated under the GIL (the old join built a full payload copy per
        PUT_CHUNK on the store path) and header+body coalesce into one
        segment. The kernel may accept fewer bytes than offered, so resume
        by advancing through the iovec list — and because a Python resume
        loop would otherwise grant each sendmsg a FRESH op-timeout window
        (sendall bounds the WHOLE operation in C; a peer draining a few
        bytes per window would extend one frame forever), the loop arms one
        whole-frame send deadline: op_timeout from entry, clamped by any
        armed per-call deadline, enforced before every sendmsg and restored
        after (mirroring _recv_into / batch_get_scatter)."""
        views = [memoryview(b).cast("B") for b in bufs if len(b)]
        hdr.payload_len = sum(v.nbytes for v in views)
        try:
            if not _HAS_SENDMSG:
                payload = b"".join(views)
                self._sock.sendall(hdr.pack())
                if payload:
                    self._sock.sendall(payload)
                return
            deadline = self._deadline
            if self._op_timeout is not None:
                send_by = time.monotonic() + self._op_timeout
                deadline = send_by if deadline is None else min(deadline, send_by)
            pending = [memoryview(hdr.pack())] + views
            clamped = False
            try:
                while pending:
                    if deadline is not None:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise ConnectionLost("send deadline exceeded (conn evicted)")
                        self._sock.settimeout(remaining)
                        clamped = True
                    sent = self._sock.sendmsg(pending)
                    while pending and sent >= len(pending[0]):
                        sent -= len(pending[0])
                        pending.pop(0)
                    if sent:
                        pending[0] = pending[0][sent:]
            finally:
                if clamped:
                    try:
                        self._sock.settimeout(self._op_timeout)
                    except OSError:
                        pass  # dead socket: the conn is being evicted regardless
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
        op_timeout = self._op_timeout
        while got < n:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ConnectionLost("recv deadline exceeded (conn evicted)")
                # Re-arm the socket only when the remaining budget actually
                # binds. With a finite op timeout the clamp is a no-op for
                # the whole healthy life of a load (remaining >> op_timeout),
                # and remaining decreases monotonically within an armed call,
                # so once it binds it binds every iteration after — skipping
                # the no-op re-arms keeps this hot loop at one clock read per
                # recv. The per-iteration deadline CHECK above (the trickle
                # armor) is unconditional either way.
                if op_timeout is None or remaining < op_timeout:
                    try:
                        self._sock.settimeout(remaining)
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
    def batch_exists(self, keys: list[bytes], deadline: float | None = None):
        """deadline (optional, time.monotonic()-based): hard wall-clock
        ceiling for the WHOLE exchange, threaded into every send/recv exactly
        like batch_get_scatter's — the scheduler-side lookup runs on the
        engine's critical path, so a hung-but-accepting daemon must cost a
        bounded blip, never a per-recv op_timeout. Past the deadline this
        raises ConnectionLost (mid-exchange abandonment desyncs the stream,
        so eviction is the only safe degrade)."""
        want_bitmap = bool(self.limits.features & p.FEAT_EXISTS_BITMAP)
        self._deadline = deadline  # armed for _send_frame and every _recv_into
        try:
            self._send_frame(p.Header(p.Op.BATCH_EXISTS, ns=self.namespace_id, request_id=1),
                             [p.pack_keylist(keys)])
            h = self._next_header()
            body = self._read_body(h.payload_len)
        finally:
            self._deadline = None
            if deadline is not None:
                try:
                    self._sock.settimeout(self._op_timeout)
                except OSError:
                    pass  # dead socket: the conn is being evicted regardless
        status, count = p.parse_preamble(body)
        if not p.status_ok(status):
            raise p.StatusError(p.Op.BATCH_EXISTS, status)
        try:
            n_consec, _ = struct.unpack_from("<II", body, p.PREAMBLE_SIZE)
            per_key = None
            if want_bitmap:
                # Not a packed bitmap: one status byte per key (padded to 8), the
                # AppendExistsResp layout — OK/OK_EXISTS ⇒ present.
                off = p.PREAMBLE_SIZE + 8
                per_key = [p.status_ok(body[off + i]) for i in range(count)]
        except (struct.error, IndexError) as e:
            # Taxonomy boundary: a body shorter than its declared layout must
            # surface as a client error (`except KvblockdError` catches it),
            # never a raw struct/Index error — and it desyncs, so FrameError.
            raise p.FrameError(f"malformed BATCH_EXISTS body: {e}") from e
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
            # Detected corruption must not vanish into a silent miss: log at
            # ERROR (unconditional — this is rare and always actionable) and
            # count it before the eviction turns it into misses downstream.
            logger.error("kvblockd block %d: xxh3 mismatch (corrupt payload) — "
                         "evicting the connection", idx)
            if self._counters is not None:
                self._counters.bump("corrupt_blocks")
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
        try:
            # to_status (not Status()) — a forward-compat server may emit a per-key
            # code this client predates; decode tolerantly, never crash the stream.
            return [p.to_status(body[p.PREAMBLE_SIZE + i]) for i in range(count)]
        except IndexError as e:  # body shorter than its declared count: desync
            raise p.FrameError(f"malformed op {int(op):#x} body: {e}") from e

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
              so_rcvbuf=None, counters: ClientCounters | None = None) -> _Conn:
    host, port = addr
    sock, granted_rcvbuf = _connect(host, port, connect_timeout, so_rcvbuf)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    # The whole PRIMING exchange (HELLO send + response) stays on the DIAL
    # budget: a hung-but-ACCEPTING daemon (SIGSTOP, blackholed after accept)
    # must cost connect_timeout, not op_timeout — dialing runs on caller
    # threads (the connector's scheduler path among them), and op_timeout
    # (10s default) there is exactly the stall connect_timeout exists to
    # bound. The socket moves to op_timeout only once HELLO has parsed.
    hdr = p.Header(p.Op.HELLO, request_id=1)
    body = p.pack_hello_req(features, 0, 0, token.encode(), namespace.encode(), b"kvblockd-py")
    hdr.payload_len = len(body)
    try:
        sock.sendall(hdr.pack() + body)
    except OSError as e:
        sock.close()
        raise ConnectionLost(f"hello send: {e}") from e
    conn = _Conn(sock, None, 0, verify, counters)  # limits filled below
    try:
        h = conn._next_header()
        try:
            resp = p.HelloResp.parse(conn._read_body(h.payload_len))
        except struct.error as e:  # short HELLO body: taxonomy boundary (28)
            raise p.FrameError(f"malformed HELLO response body: {e}") from e
    except BaseException:
        conn.close()  # HELLO rejection (StatusError/Fatal/Frame) must not leak the fd
        raise
    conn.limits = Limits(resp)
    conn.namespace_id = resp.namespace_id
    conn.granted_rcvbuf = granted_rcvbuf
    # HELLO parsed: the dial budget ends here. Hand the socket its steady-
    # state per-recv budget AND refresh the _Conn snapshot (its __init__
    # read gettimeout() == connect_timeout; the deadline machinery restores
    # to _op_timeout after every clamped call, so both must move together —
    # or every recv for the conn's life stays silently on the dial budget).
    try:
        sock.settimeout(op_timeout)
    except OSError as e:  # dead fd already: surface as the usual eviction class
        conn.close()
        raise ConnectionLost(f"settimeout after hello: {e}") from e
    conn._op_timeout = op_timeout
    logger.debug("kvblockd conn to %s:%s — SO_RCVBUF asked %s, kernel reports %d",
                 host, port, "autotune (not set)" if so_rcvbuf is None else so_rcvbuf,
                 granted_rcvbuf)
    return conn


class Client:
    """A pool of connections to one kvblockd namespace."""

    def __init__(self, addr, *, namespace, token, streams=4, get_fanout=None,
                 connect_timeout=5.0, op_timeout=10.0, verify=True, so_rcvbuf=None):
        """so_rcvbuf: OPT-IN SO_RCVBUF override in bytes. Leave None (the
        default) to keep kernel receive-window autotuning — setting it
        disables autotuning and clamps at net.core.rmem_max (see _connect).

        get_fanout: batch_get_scatter shard ceiling. None (the default) keeps
        today's behavior — min(4, streams), silently clamped. An EXPLICIT
        value must satisfy 1 <= get_fanout <= streams: each shard drains on
        its own pooled connection, so a fan-out above the pool size would
        just serialize shards on connection checkout — refuse it loudly
        instead of quietly measuring the pool ceiling."""
        if isinstance(addr, str):
            host, _, port = addr.partition(":")
            addr = (host, int(port))
        feats = p.FEAT_EXISTS_BITMAP  # in-order (no OOO); bitmap for per-key EXISTS
        # Misses-by-cause telemetry (shared by every conn + the pool): the
        # designed failure mode is a miss, so this is the primary signal.
        self.counters = ClientCounters()
        # One rate-limit clock per disclosure key (degrade/deadline lines).
        self._warn_last: dict[str, float] = {}

        def factory():
            return _dial_one(addr, namespace, token, feats, connect_timeout, op_timeout,
                             verify, so_rcvbuf, self.counters)

        self._streams = max(int(streams), 1)
        if get_fanout is None:
            self._get_fanout = min(_MAX_GET_FANOUT, self._streams)
        else:
            gf = int(get_fanout)
            if gf < 1:
                raise ValueError(f"get_fanout must be >= 1, got {gf}")
            if gf > self._streams:
                raise ValueError(
                    f"get_fanout {gf} > streams {self._streams}: shards would "
                    "serialize on connection checkout — raise streams instead")
            self._get_fanout = gf
        self._pool = Pool(factory, streams,
                          on_evict=lambda: self.counters.bump("evictions"))
        # Prime one conn so limits are known and auth failures surface at construct.
        c = self._pool.checkout()
        self.limits = c.limits
        # The EFFECTIVE receive buffer the kernel reports (see _connect) —
        # exposed so rigs can stamp the real window, never an assumption.
        # With so_rcvbuf=None this is the autotune starting size.
        self.granted_rcvbuf = c.granted_rcvbuf
        self._pool.checkin(c)
        # Persistent drain workers for batch_get_scatter (created AFTER the
        # prime so a failed dial never leaks threads): a per-call executor
        # paid thread spawn + teardown on every load, on the latency path.
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self._get_fanout,  # already clamped to the pool size
            thread_name_prefix="kvb-get")

    def close(self):
        # Executor BEFORE the pool: Pool.close only closes IDLE conns, so a
        # shard mid-drain keeps its checked-out conn (recvs stay bounded by
        # op_timeout) and its checkin lands on the closed pool, which closes
        # the conn. wait=False keeps close() from blocking on an in-flight
        # drain — the workers exit as soon as their shard settles.
        self._executor.shutdown(wait=False)
        self._pool.close()

    def _run(self, fn, checkout_timeout: float | None = None):
        return self._pool.run(fn, checkout_timeout)

    def _warn_rl(self, key: str, msg: str) -> None:
        """Rate-limited WARNING (one line per key per interval): the degrade
        paths fire per-shard under load, and a warn storm is its own outage."""
        now = time.monotonic()
        if now - self._warn_last.get(key, 0.0) >= _WARN_INTERVAL_S:
            self._warn_last[key] = now
            logger.warning("%s", msg)

    def _split(self, keys):
        cap = self.limits.max_batch_keys or len(keys)
        for i in range(0, len(keys), cap):
            yield keys[i:i + cap]

    def batch_exists(self, keys, deadline: float | None = None):
        """deadline (optional, monotonic): wall-clock ceiling across EVERY
        tile — checkout wait, sends, and recvs included — for callers whose
        budget is smaller than op_timeout (the scheduler-side lookup). Past
        it this raises ConnectionLost; the conn mid-exchange is evicted."""
        if not keys:
            return 0, []
        # Split above the negotiated cap; consecutive-prefix stops at the first
        # miss, so a broken prefix in tile k makes later tiles irrelevant.
        total_consec, per = 0, []
        broken = False
        for tile in self._split(keys):
            co = None
            if deadline is not None:
                co = deadline - time.monotonic()
                if co <= 0:
                    raise ConnectionLost("EXISTS deadline exceeded before checkout")
            nc, pk = self._run(lambda c: c.batch_exists(tile, deadline), co)  # noqa: B023 — lambda is invoked synchronously inside this iteration (Pool.run calls fn(conn) immediately); it never outlives the loop
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
        """Zero-copy batched GET, fanned out across up to get_fanout (default
        4) CONTIGUOUS key shards (bounded by the pool size), one pooled
        connection per shard, each shard internally tiled by the negotiated
        max_batch_keys.

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
        nshards = min(self._get_fanout, self._streams, n)
        if nshards <= 1:
            return self._scatter_shard(keys, prefix_len, alloc, 0, deadline)
        bounds = self._shard_bounds(n, nshards)
        statuses = [p.Status.NOT_FOUND] * n
        errs = []
        futs = []
        try:
            for s, e in bounds:
                futs.append((s, self._executor.submit(self._scatter_shard, keys[s:e],
                                                      prefix_len, alloc, s, deadline)))
        except RuntimeError as exc:
            # Executor shut down == client closed: surface the same eviction-
            # class error the pool raises, so callers degrade to misses. But
            # NEVER settle with live writers: a shard already submitted keeps
            # its checked-out conn (Pool.close only closes idle ones) and
            # keeps writing caller-owned memoryviews via alloc; a raise that
            # returns first would let the caller re-map those bytes under a
            # later call while the orphan still writes them. Join the shards
            # — each is bounded by op_timeout/deadline, so the join is too.
            concurrent.futures.wait([f for _, f in futs])
            raise ConnectionLost(f"client closed: {exc}") from exc
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
            co = None
            if deadline is not None:
                co = deadline - time.monotonic()
                if co <= 0:
                    remaining = len(keys) - len(out)
                    self.counters.bump("deadline_misses", remaining)
                    self._warn_rl("deadline",
                                  f"kvblockd GET deadline expired — {remaining} remaining "
                                  "key(s) degraded to misses (recompute)")
                    out.extend([p.Status.NOT_FOUND] * remaining)
                    return out
            b = base
            try:
                out.extend(self._run(
                    lambda c, t=tile, bb=b: c.batch_get_scatter(t, prefix_len, alloc, bb,
                                                                deadline), co))
            except (ConnectionLost, FatalProtocol, OSError) as e:
                remaining = len(keys) - len(out)
                self.counters.bump("degraded_keys", remaining)
                self._warn_rl("degrade",
                              f"kvblockd GET shard failed ({len(keys)} key(s), {remaining} "
                              f"degraded to misses): {e!r}")
                out.extend([p.Status.NOT_FOUND] * remaining)
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
