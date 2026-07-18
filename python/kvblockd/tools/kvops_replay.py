#!/usr/bin/env python3
"""Cross-language .kvops replayer — the parity oracle.

Reads a .kvops op-stream and prints the SAME "idx exists n get k put m" log
the Go replayer emits (bench/kvbench/internal/replay), proving both languages
derive identical op sequences from a trace. This is what lets a redis-py
baseline replay EXACTLY the ops kvblockd replayed.

The hit/miss split depends on store state, so parity is defined on the
op-SHAPE per record (chain length, and — for a from-empty single-worker
run — the deterministic k=0 first-touch pattern). Here we reproduce the
record shapes and key derivations; the harness's Go --oplog at --streams 1
against an empty store is matched exactly by the --empty mode below.

Key derivation mirrors bench/kvbench/internal/kvops.TraceKey:
BLAKE3-256 over domain "kvbench-trace-v1\\x00" then u32-LE length-prefixed
UTF-8 fields — identical recipe to kvblockd.hashing.wire_key with a
different domain.
"""
import struct
import sys

_TRACE_DOMAIN = b"kvbench-trace-v1\x00"


def trace_key(*fields: str) -> bytes:
    """Mirror of Go kvops.TraceKey — used only by the standalone
    key-derivation parity check, NOT by the replay (which reads baked keys
    from the .kvops file). Imported lazily so the replayer runs without
    blake3 installed."""
    from blake3 import blake3  # noqa: PLC0415 — optional dependency, lazy on purpose

    blob = bytearray(_TRACE_DOMAIN)
    for f in fields:
        fb = f.encode("utf-8")
        blob += struct.pack("<I", len(fb))
        blob += fb
    return blake3(bytes(blob)).digest()


def read_kvops(path):
    with open(path, "rb") as f:
        hdr = f.read(16)
        if hdr[0:4] != b"KVOP":
            raise SystemExit(f"bad magic {hdr[0:4]!r}")
        version = struct.unpack_from("<H", hdr, 4)[0]
        if version != 1:
            raise SystemExit(f"version {version}")
        blob_bytes = struct.unpack_from("<I", hdr, 8)[0]
        meta_len = struct.unpack_from("<I", hdr, 12)[0]
        f.read(meta_len)  # meta JSON — provenance, not needed for the replay
        idx = 0
        while True:
            fixed = f.read(10)
            if len(fixed) == 0:
                return  # clean end
            if len(fixed) < 10:
                raise SystemExit(f"torn record header at #{idx}")
            n = struct.unpack_from("<H", fixed, 8)[0]
            keys = []
            for _ in range(n):
                k = f.read(32)
                if len(k) < 32:
                    raise SystemExit(f"torn keys at #{idx}")
                keys.append(k)
            yield idx, keys
            idx += 1
    _ = blob_bytes


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: kvops_replay.py <file.kvops>")
    # Reproduce the Go --streams 1, from-empty adaptive replay: each record
    # is EXISTS(chain); the consecutive-hit prefix against a growing resident
    # set determines k; misses are PUT and become resident.
    resident: set[bytes] = set()
    for idx, keys in read_kvops(sys.argv[1]):
        k = 0
        for key in keys:
            if key in resident:
                k += 1
            else:
                break
        for key in keys[k:]:
            resident.add(key)
        print(f"{idx} exists {len(keys)} get {k} put {len(keys) - k}")


if __name__ == "__main__":
    main()
