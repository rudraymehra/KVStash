"""The 32-byte metadata prefix codec — no lmcache/torch needed."""

import struct

import pytest

from lmcache_kvblockd import meta


def test_roundtrip():
    blob = meta.encode(mem_fmt=2, dtype_name="bfloat16", shape=(16, 128), body_len=4096)
    assert len(blob) == meta.PREFIX_LEN
    fmt, dtype, shape, body_len = meta.decode(blob)
    assert fmt == 2 and dtype == "bfloat16" and shape == (16, 128) and body_len == 4096


def test_all_dtypes_stable():
    for name, code in meta.DTYPE_CODES.items():
        blob = meta.encode(0, name, (1,), 8)
        assert meta.decode(blob)[1] == name
        assert blob[6] == code  # dtype byte at offset 4(magic)+1(ver)+1(fmt) = 6


def test_unknown_dtype_rejected():
    with pytest.raises(meta.MetaError):
        meta.encode(0, "complex128", (1,), 8)


def test_bad_magic_and_version_rejected():
    blob = bytearray(meta.encode(0, "float16", (2, 2), 16))
    blob[0] ^= 0xFF
    with pytest.raises(meta.MetaError):
        meta.decode(bytes(blob))
    blob = bytearray(meta.encode(0, "float16", (2, 2), 16))
    blob[4] = 99  # version byte
    with pytest.raises(meta.MetaError):
        meta.decode(bytes(blob))


def test_rank_over_4_rejected():
    with pytest.raises(meta.MetaError):
        meta.encode(0, "float16", (1, 2, 3, 4, 5), 8)


def test_shape_zero_filled():
    blob = meta.encode(0, "float32", (7,), 28)
    # ndim=1, then shape[0]=7, rest zero.
    ndim = blob[7]
    assert ndim == 1
    s0, s1, s2, s3 = struct.unpack_from("<4I", blob, 8)
    assert (s0, s1, s2, s3) == (7, 0, 0, 0)
