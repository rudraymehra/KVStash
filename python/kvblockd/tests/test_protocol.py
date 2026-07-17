"""Wire-codec parity: the Python codec must parse and re-pack the SAME golden
frames the Go server pins (internal/protocol/testdata/frames), byte-for-byte,
and reproduce the pinned CRC32C values from PROTOCOL.md §11/§12."""

from __future__ import annotations

import binascii
import struct
from pathlib import Path

import pytest

from kvblockd import protocol as p

# Repo-relative golden dir; embedded fallback keeps wheel-built tests runnable.
_GOLDEN_DIR = Path(__file__).resolve().parents[3] / "internal/protocol/testdata/frames"
_EMBEDDED = {
    "example-a-request": (
        "4b5642310102000007000000000000000110000000000000000000000000000000000000"
        "000000000000000000000000000000000000000068000000ffa72e5f0300000000000000"
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbb"
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbcccccccccccccccc"
        "cccccccccccccccccccccccccccccccccccccccccccccccc"
    ),
    "example-a-response": (
        "4b5642310102010007000000000010000110000000000000000000000000000000000000"
        "000000000000000000000000000000000000000018000000232e93d60000000003000000"
        "02000000000000000000100000000000"
    ),
    "example-b-response-header": (
        "4b5642310103010007000000000010000210000000000000000000000000000000000000"
        "000000000000000000000000000000000000000040003800b99d4fc50000000003000000"
        "000000000300000000000000000010004ffc818960eb7dc400000000000028005dc466ba"
        "22169ed610000000000000000000000000000000"
    ),
}


def _load(name: str) -> bytes:
    f = _GOLDEN_DIR / f"{name}.hex"
    text = f.read_text().strip() if f.exists() else _EMBEDDED[name]
    return binascii.unhexlify(text)


def test_crc32c_pins():
    # The three PROTOCOL.md §11/§12 header CRCs, over bytes 0..59.
    for name, want in [
        ("example-a-request", 0x5F2EA7FF),
        ("example-a-response", 0xD6932E23),
        ("example-b-response-header", 0xC54F9DB9),
    ]:
        frame = _load(name)
        assert p.crc32c(frame[:60]) == want, name
        # And the CRC stored at offset 60 matches.
        (stored,) = struct.unpack_from("<I", frame, 60)
        assert stored == want, name


def test_example_a_request_fields():
    frame = _load("example-a-request")
    h = p.Header.parse(frame)
    assert h.opcode == p.Op.BATCH_EXISTS
    assert h.flags == 0
    assert h.ns == 7
    assert h.request_id == 0x1001  # the golden's request_id (§11)
    assert h.payload_len == 0x68  # 8 + 3*32
    # Body: 3 keys aa/bb/cc, aux 0.
    body = frame[p.HEADER_SIZE:]
    n_keys, aux = struct.unpack_from("<II", body, 0)
    assert (n_keys, aux) == (3, 0)
    assert body[8:40] == b"\xaa" * 32
    assert body[40:72] == b"\xbb" * 32
    assert body[72:104] == b"\xcc" * 32


def test_header_repack_is_byte_identical():
    frame = _load("example-a-request")
    h = p.Header.parse(frame)
    assert h.pack() == frame[:p.HEADER_SIZE]


def test_example_a_response_exists():
    frame = _load("example-a-response")
    h = p.Header.parse(frame)
    assert h.flags & p.F_RESP
    body = frame[p.HEADER_SIZE:]
    status, count = p.parse_preamble(body)
    assert status == p.Status.OK and count == 3
    n_consec, _ = struct.unpack_from("<II", body, p.PREAMBLE_SIZE)
    assert n_consec == 2  # hit, hit, miss → consecutive prefix 2


def test_example_b_get_region():
    frame = _load("example-b-response-header")
    h = p.Header.parse(frame)
    assert h.opcode == p.Op.BATCH_GET and h.flags & p.F_RESP
    body = frame[p.HEADER_SIZE:]
    status, first, total, descs = p.parse_get_region(body)
    assert status == p.Status.OK
    assert first == 0 and total == 3
    assert len(descs) == 3
    # desc0 OK 1 MiB, desc1 OK 640 KiB, desc2 NOT_FOUND (§12).
    assert descs[0][0] == p.Status.OK and descs[0][2] == 0xC47DEB608981FC4F
    assert descs[1][0] == p.Status.OK and descs[1][2] == 0xD69E1622BA66C45D
    assert descs[2][0] == p.Status.NOT_FOUND


def test_bad_magic_and_crc_rejected():
    frame = bytearray(_load("example-a-request"))
    frame[0] ^= 0xFF
    with pytest.raises(p.FrameError):
        p.Header.parse(bytes(frame))
    frame = bytearray(_load("example-a-request"))
    frame[60] ^= 0xFF  # corrupt the stored CRC
    with pytest.raises(p.FrameError):
        p.Header.parse(bytes(frame))


def test_keylist_roundtrip():
    keys = [bytes([i]) * 32 for i in range(5)]
    packed = p.pack_keylist(keys, aux=0xDEAD)
    n, aux = struct.unpack_from("<II", packed, 0)
    assert n == 5 and aux == 0xDEAD
    assert packed[8:40] == keys[0]
    assert len(packed) == 8 + 5 * 32


def test_embedded_fallbacks_are_valid():
    # The wheel/sdist safety net: with the repo hex files absent, _EMBEDDED
    # must still parse. SSE found 2 of 3 hand-transcribed wrong; this test
    # (which never touches the repo files) is the tripwire.
    for name, want in [
        ("example-a-request", 0x5F2EA7FF),
        ("example-a-response", 0xD6932E23),
        ("example-b-response-header", 0xC54F9DB9),
    ]:
        frame = binascii.unhexlify(_EMBEDDED[name])
        assert p.crc32c(frame[:60]) == want, f"{name} embedded fallback corrupt"
        assert p.Header.parse(frame).opcode in (p.Op.BATCH_EXISTS, p.Op.BATCH_GET)


def test_status_enum_matches_wire():
    # The codes are wire law (ops.go §9). Spot-check the ones the ladder found
    # wrong, and that to_status never raises on an unknown byte.
    assert p.Status.EVICTED == 0x11
    assert p.Status.ERR_AUTH_REQUIRED == 0x20 and p.Status.ERR_AUTH_FAILED == 0x21
    assert p.Status.ERR_NAMESPACE_UNKNOWN == 0x22 and p.Status.ERR_FORBIDDEN == 0x23
    assert p.Status.ERR_UNSUPPORTED == 0x50 and p.Status.ERR_MALFORMED == 0x51
    assert p.Status.ERR_INTERNAL == 0x60 and p.Status.FATAL_PROTOCOL == 0xF0
    assert p.to_status(0x11) == p.Status.EVICTED
    assert p.to_status(0x7A) == p.Status.ERR_INTERNAL  # unknown → tolerant, no raise


def test_header_rejects_wrong_length_key():
    import pytest as _pytest
    with _pytest.raises(ValueError):
        p.Header(p.Op.PUT_STREAM, key=b"\x00" * 16)
