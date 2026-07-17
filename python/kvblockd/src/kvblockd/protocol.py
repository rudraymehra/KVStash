"""KVB1 wire codec — the Python mirror of internal/protocol (frozen, §1–§12).

Every layout here is byte-identical to the Go server's; the golden hex
vectors in internal/protocol/testdata/frames are the shared oracle. All
integers are little-endian. The header CRC is CRC32C (Castagnoli) over the
first 60 bytes — NOT the IEEE crc32 in zlib, so we carry a tiny table.
"""

from __future__ import annotations

import struct
from enum import IntEnum

# --- CRC32C (Castagnoli, reflected, poly 0x1EDC6F41 → reflected 0x82F63B78) ---
# One CRC per 64-byte frame; a pure-Python table is plenty and dependency-free.
_CRC32C_POLY = 0x82F63B78
_CRC32C_TABLE = []
for _i in range(256):
    _c = _i
    for _ in range(8):
        _c = (_c >> 1) ^ _CRC32C_POLY if (_c & 1) else (_c >> 1)
    _CRC32C_TABLE.append(_c)


def crc32c(data: bytes) -> int:
    """CRC32C of data (init/final xor 0xFFFFFFFF), as the Go HeaderCRC computes."""
    crc = 0xFFFFFFFF
    for b in data:
        crc = (crc >> 8) ^ _CRC32C_TABLE[(crc ^ b) & 0xFF]
    return crc ^ 0xFFFFFFFF


# --- constants (mirror internal/protocol/ops.go + header.go) ---
MAGIC = b"KVB1"
VERSION1 = 0x01
HEADER_SIZE = 64
PREAMBLE_SIZE = 8
DESC_SIZE = 16
_CRC_OFFSET = 60


class Op(IntEnum):
    NOP = 0x00
    HELLO = 0x01
    BATCH_EXISTS = 0x02
    BATCH_GET = 0x03
    PUT_STREAM = 0x04
    TOUCH_LEASE = 0x05
    PIN = 0x06
    DELETE = 0x07
    STATS = 0x08


# Flags (bits 0–3); sub-op selector in bits 4–7.
F_RESP = 0x0001
F_MORE = 0x0002
F_FATAL = 0x0004
F_FORCE = 0x0008


def with_subop(flags: int, sub: int) -> int:
    return (flags & ~0xF0) | ((sub & 0xF) << 4)


def subop(flags: int) -> int:
    return (flags >> 4) & 0xF


# PUT_STREAM sub-ops.
PUT_BEGIN, PUT_CHUNK, PUT_COMMIT, PUT_ABORT = 0, 1, 2, 3
# TOUCH_LEASE sub-ops.
TOUCH_RECENCY, LEASE_GRANT, LEASE_RELEASE = 0, 1, 2
# PIN sub-ops.
PIN_SOFT, PIN_HARD, UNPIN = 0, 1, 2

# Feature bits (§10).
FEAT_OOO = 1 << 0
FEAT_EXISTS_BITMAP = 1 << 1


class Status(IntEnum):
    # Mirrors internal/protocol/ops.go §9 EXACTLY — the codes are wire law.
    OK = 0x00
    OK_EXISTS = 0x01
    NOT_FOUND = 0x10
    EVICTED = 0x11  # observability nicety; treat as NOT_FOUND
    ERR_AUTH_REQUIRED = 0x20
    ERR_AUTH_FAILED = 0x21
    ERR_NAMESPACE_UNKNOWN = 0x22
    ERR_FORBIDDEN = 0x23
    ERR_QUOTA_BYTES = 0x30
    ERR_PIN_QUOTA = 0x31
    ERR_TOO_LARGE = 0x32
    ERR_BATCH_TOO_LARGE = 0x33
    ERR_BUSY = 0x34
    ERR_CHECKSUM = 0x40
    ERR_SHORT_STREAM = 0x41
    ERR_STALE_STREAM = 0x42
    ERR_IMMUTABLE_CONFLICT = 0x43
    ERR_LEASED = 0x44
    ERR_PINNED = 0x45
    ERR_UNSUPPORTED = 0x50
    ERR_MALFORMED = 0x51
    ERR_INTERNAL = 0x60
    FATAL_PROTOCOL = 0xF0


def to_status(b: int) -> Status:
    """Decode a wire status byte tolerantly: an unknown code becomes
    ERR_INTERNAL rather than raising (a forward-compatible server may emit a
    status this client predates — a crash on decode would be worse than a
    generic error, and would leak a pool slot via a non-verb exception)."""
    try:
        return Status(b)
    except ValueError:
        return Status.ERR_INTERNAL


def status_ok(s: int) -> bool:
    return s in (Status.OK, Status.OK_EXISTS)


# --- struct layouts (all "<" little-endian, no padding) ---
_HEADER = struct.Struct("<4sBBHIIQ32sII")  # 64B; last I is the CRC slot
_PREAMBLE = struct.Struct("<B3xI")
_KEYLIST = struct.Struct("<II")  # n_keys, aux — then n×32 raw key bytes
_GET_IDX = struct.Struct("<II")  # first_index, total_keys
_DESC = struct.Struct("<B3xIQ")  # status, len, xxh3_64
_PUT_BEGIN = struct.Struct("<IIQII")  # total_len, ttl_ms, xxh3_hint, flags, rsvd
_PUT_COMMIT = struct.Struct("<Q")
_HELLO_REQ = struct.Struct("<BBHQIIQHHHH")  # 36B fixed
_HELLO_RESP = struct.Struct("<B3xQ8IHH")  # after preamble: proto|rsvd3|features u64|8×u32|name_len|rsvd
_EXISTS_HDR = struct.Struct("<II")  # n_consecutive, rsvd

assert _HEADER.size == HEADER_SIZE
assert _PREAMBLE.size == PREAMBLE_SIZE
assert _DESC.size == DESC_SIZE
assert _PUT_BEGIN.size == 24
assert _HELLO_REQ.size == 36


class Header:
    """A KVB1 frame header. `key` is the 32-byte per-frame key (zero for the
    batch verbs, whose keys live in the body)."""

    __slots__ = ("opcode", "flags", "ns", "credit", "request_id", "key", "payload_len")

    def __init__(self, opcode, flags=0, ns=0, credit=0, request_id=0, key=b"\x00" * 32, payload_len=0):
        self.opcode = opcode
        self.flags = flags
        self.ns = ns
        self.credit = credit
        self.request_id = request_id
        if len(key) != 32:
            # Never silently pad/truncate: a wrong-length key would store
            # under a silently different key (a permanent miss, not an error).
            raise ValueError(f"key must be 32 bytes, got {len(key)}")
        self.key = key
        self.payload_len = payload_len

    def pack(self) -> bytes:
        # Pack with a zero CRC slot, then overwrite the last 4 bytes with the
        # CRC32C of bytes 0..59 — exactly the Go MarshalTo order.
        buf = bytearray(
            _HEADER.pack(
                MAGIC, VERSION1, self.opcode, self.flags, self.ns,
                self.credit, self.request_id, self.key, self.payload_len, 0,
            )
        )
        struct.pack_into("<I", buf, _CRC_OFFSET, crc32c(bytes(buf[:_CRC_OFFSET])))
        return bytes(buf)

    @classmethod
    def parse(cls, buf: bytes) -> "Header":
        if len(buf) < HEADER_SIZE:
            raise FrameError(f"short header: {len(buf)} < {HEADER_SIZE}")
        magic, ver, op, flags, ns, credit, rid, key, plen, crc = _HEADER.unpack(buf[:HEADER_SIZE])
        if magic != MAGIC:
            raise FrameError(f"bad magic {magic!r}")
        if ver != VERSION1:
            raise FrameError(f"bad version {ver}")
        want = crc32c(buf[:_CRC_OFFSET])
        if crc != want:
            raise FrameError(f"header CRC mismatch: got {crc:#010x} want {want:#010x}")
        h = cls(op, flags, ns, credit, rid, key, plen)
        return h


class FrameError(Exception):
    """Malformed frame on the wire (bad magic/version/CRC or truncation)."""


# --- body codecs ---
def pack_preamble(status: int, count: int) -> bytes:
    return _PREAMBLE.pack(status & 0xFF, count)


def parse_preamble(body: bytes) -> tuple[int, int]:
    if len(body) < PREAMBLE_SIZE:
        raise FrameError("short preamble")
    return _PREAMBLE.unpack(body[:PREAMBLE_SIZE])


def pack_keylist(keys: list[bytes], aux: int = 0) -> bytes:
    out = bytearray(_KEYLIST.pack(len(keys), aux))
    for k in keys:
        if len(k) != 32:
            raise ValueError("keys must be 32 bytes")
        out += k
    return bytes(out)


def pack_put_begin(total_len: int, ttl_ms: int = 0, xxh3_hint: int = 0, flags: int = 0) -> bytes:
    return _PUT_BEGIN.pack(total_len, ttl_ms, xxh3_hint, flags, 0)


def pack_put_commit(xxh3_64: int) -> bytes:
    return _PUT_COMMIT.pack(xxh3_64)


def parse_desc(buf: bytes, off: int) -> tuple[int, int, int]:
    status, length, xxh3 = _DESC.unpack_from(buf, off)
    return status, length, xxh3


def get_resp_header_size(n_desc: int) -> int:
    return PREAMBLE_SIZE + _GET_IDX.size + DESC_SIZE * n_desc


def parse_get_region(body: bytes):
    """Parse a BATCH_GET response region → (batch_status, first_index,
    total_keys, [(status,len,xxh3)...]). A non-OK batch status is exactly the
    8-byte preamble; the caller must inspect it before reading further."""
    status, count = parse_preamble(body)
    if not status_ok(status):
        return status, 0, 0, []
    first_index, total_keys = _GET_IDX.unpack_from(body, PREAMBLE_SIZE)
    descs = []
    off = PREAMBLE_SIZE + _GET_IDX.size
    for _ in range(count):
        descs.append(parse_desc(body, off))
        off += DESC_SIZE
    return status, first_index, total_keys, descs


def _pad8(n: int) -> int:
    return (n + 7) & ~7


def pack_hello_req(
    features: int, max_batch_keys: int, max_frame_len: int,
    token: bytes, namespace: bytes, client_name: bytes,
) -> bytes:
    out = bytearray(
        _HELLO_REQ.pack(
            VERSION1, VERSION1, 0, features, max_batch_keys, max_frame_len, 0,
            len(token), len(namespace), len(client_name), 0,
        )
    )
    out += token + namespace + client_name
    out += b"\x00" * (_pad8(len(out)) - len(out))
    return bytes(out)


class HelloResp:
    __slots__ = (
        "proto", "features", "max_batch_keys", "max_frame_len", "max_blob_len",
        "namespace_id", "initial_credit", "lease_default_ms", "lease_max_ms",
        "stream_timeout_ms", "server_name",
    )

    @classmethod
    def parse(cls, body: bytes) -> "HelloResp":
        status, _ = parse_preamble(body)
        if not status_ok(status):
            raise StatusError(Op.HELLO, status)
        (proto, features, mbk, mfl, mbl, nsid, credit, ldef, lmax, stimeout,
         name_len, _rsvd) = _HELLO_RESP.unpack_from(body, PREAMBLE_SIZE)
        r = cls()
        r.proto, r.features = proto, features
        r.max_batch_keys, r.max_frame_len, r.max_blob_len = mbk, mfl, mbl
        r.namespace_id, r.initial_credit = nsid, credit
        r.lease_default_ms, r.lease_max_ms, r.stream_timeout_ms = ldef, lmax, stimeout
        off = PREAMBLE_SIZE + _HELLO_RESP.size
        r.server_name = body[off:off + name_len].decode("utf-8", "replace")
        return r


class StatusError(Exception):
    """A non-OK verb status. The connection stays in sync (the caller may
    re-pool it) — distinct from FrameError, which desyncs the stream."""

    def __init__(self, op: int, status: int):
        self.op = op
        self.status = status
        super().__init__(f"op {int(op):#x}: status {Status(status).name if status in Status._value2member_map_ else hex(status)}")
