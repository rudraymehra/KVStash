"""The 32-byte metadata prefix that rides INSIDE each stored block so a
GET can reconstruct the MemoryObj (LMCache carries memory_format + dtype +
shape out of band; the opaque server won't). Layout keeps the block fully
opaque to kvblockd (T3): the client-computed XXH3 covers prefix + tensor, the
server never parses either.

    <4s 4B 4I I I>  — exactly 32 bytes
      magic     4s   b"KVM1"
      version   B    1
      mem_fmt   B    LMCache MemoryFormat enum value
      dtype     B    pinned code (DTYPE_CODES below)
      ndim      B    number of used shape dims (<=4)
      shape     4I   dims, zero-filled past ndim
      body_len  I    tensor byte length following the prefix
      reserved  I    0
"""

from __future__ import annotations

import struct

MAGIC = b"KVM1"
VERSION = 1
PREFIX_LEN = 32

_META = struct.Struct("<4sBBBB4III")
assert _META.size == PREFIX_LEN

# Pinned dtype codes (versioned by the prefix `version` byte). Names are the
# torch dtype names; the connector maps to/from torch.dtype via this table.
DTYPE_CODES = {
    "float16": 0, "bfloat16": 1, "float32": 2, "float64": 3,
    "uint8": 4, "int8": 5, "int32": 6, "int64": 7,
    "float8_e4m3fn": 8, "float8_e5m2": 9,
}
CODE_DTYPES = {v: k for k, v in DTYPE_CODES.items()}


class MetaError(ValueError):
    """The prefix is unrecognized (bad magic/version) or carries an unknown
    dtype/format — the caller treats the block as a miss."""


def encode(mem_fmt: int, dtype_name: str, shape, body_len: int) -> bytes:
    if dtype_name not in DTYPE_CODES:
        raise MetaError(f"unsupported dtype {dtype_name!r}")
    dims = list(shape)
    if len(dims) > 4:
        raise MetaError(f"shape rank {len(dims)} > 4")
    padded = (dims + [0, 0, 0, 0])[:4]
    return _META.pack(MAGIC, VERSION, mem_fmt & 0xFF, DTYPE_CODES[dtype_name], len(dims),
                      padded[0], padded[1], padded[2], padded[3], body_len, 0)


def decode(prefix: bytes):
    """→ (mem_fmt, dtype_name, shape_tuple, body_len). Raises MetaError on any
    unrecognized prefix so the connector can treat it as a cache miss."""
    if len(prefix) < PREFIX_LEN:
        raise MetaError("prefix too short")
    magic, version, mem_fmt, dtype_code, ndim, s0, s1, s2, s3, body_len, _rsvd = _META.unpack(
        prefix[:PREFIX_LEN]
    )
    if magic != MAGIC:
        raise MetaError(f"bad magic {magic!r}")
    if version != VERSION:
        raise MetaError(f"unknown meta version {version}")
    if dtype_code not in CODE_DTYPES:
        raise MetaError(f"unknown dtype code {dtype_code}")
    if ndim > 4:
        raise MetaError(f"bad ndim {ndim}")
    shape = tuple([s0, s1, s2, s3][:ndim])
    return mem_fmt, CODE_DTYPES[dtype_code], shape, body_len
