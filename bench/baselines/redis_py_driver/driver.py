#!/usr/bin/env python3
"""redis-py .kvops replayer — the 'path LMCache actually ships' bar.

Replays the SAME adaptive op stream the Go replayer does (EXISTS→GET→PUT,
hit rate an OUTPUT) via redis-py 8.0.1 pipelines, so the bar is apples-to-
apples with the go-redis and kvblockd bars. The gap between this and the
go-redis zero-copy bar is the two-client story (methodology rule 3).

Payloads: this THROUGHPUT bar writes a fixed-size incompressible-ish blob of
EXACTLY blob_bytes per key (not the full kvblockd-payload-v1 checksum spec —
this bar measures Redis bytes/s, the kvblockd bar owns the corruption
oracle). Byte count matches the other bars for a fair goodput comparison.

Output: one JSON line matching bench/report/schema.md (kind="replay",
store="redis-py"), so plot.py treats it like any other bar.

Usage:
  driver.py --kvops trace.kvops --addr 127.0.0.1:6379 --streams 8 \
            [--resp2] [--out out.jsonl]
"""
import argparse
import json
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor

try:
    import redis  # redis-py 8.0.1
except ImportError:
    sys.exit("redis-py not installed (pip install 'redis==8.0.1')")


def read_kvops(path):
    with open(path, "rb") as f:
        hdr = f.read(16)
        if hdr[0:4] != b"KVOP":
            sys.exit(f"bad magic {hdr[0:4]!r}")
        version = struct.unpack_from("<H", hdr, 4)[0]
        if version != 1:
            sys.exit(f"kvops version {version}")
        blob_bytes = struct.unpack_from("<I", hdr, 8)[0]
        meta_len = struct.unpack_from("<I", hdr, 12)[0]
        f.read(meta_len)
        records = []
        while True:
            fixed = f.read(10)
            if not fixed:
                break
            if len(fixed) < 10:
                sys.exit("torn record header")
            n = struct.unpack_from("<H", fixed, 8)[0]
            keys = [f.read(32) for _ in range(n)]
            if any(len(k) < 32 for k in keys):
                sys.exit("torn keys")
            records.append(keys)
        return blob_bytes, records


def fill_bytes(n):
    # Deterministic incompressible-ish payload of EXACTLY n bytes — the same
    # byte count every other bar moves (the ladder caught an oversized tile
    # that made redis move +4096 B/key and mis-measured goodput).
    tile = bytes((i * 2654435761) & 0xFF for i in range(4096))
    reps = n // 4096 + 1
    return (tile * reps)[:n]


def replay_record(client, keys, blob_bytes, payload):
    # EXISTS the chain → consecutive prefix; GET hits; SET misses.
    pipe = client.pipeline(transaction=False)
    for k in keys:
        pipe.exists(k)
    exists = pipe.execute()
    k = 0
    for e in exists:
        if e:
            k += 1
        else:
            break
    get_bytes = 0
    if k:
        pipe = client.pipeline(transaction=False)
        for key in keys[:k]:
            pipe.get(key)
        for v in pipe.execute():
            if v is not None:
                get_bytes += len(v)
    if k < len(keys):
        pipe = client.pipeline(transaction=False)
        for key in keys[k:]:
            pipe.set(key, payload)
        pipe.execute()
    return k, len(keys) - k, get_bytes


def getbench(client, blob_bytes, streams, secs, pool):
    """Closed-loop batched-GET throughput — the Chart-1 bar comparable to the
    go-redis and kvblockd GET bars (the ≥10x gate's redis-py denominator).
    Fills `pool` keys, then pipelines GETs back-to-back for `secs`."""
    payload = fill_bytes(blob_bytes)
    keys = [f"gb:{i}".encode() for i in range(pool)]
    pipe = client.pipeline(transaction=False)
    for k in keys:
        pipe.set(k, payload)
    pipe.execute()

    stop = time.time() + secs
    total_bytes = [0] * streams

    def worker(w):
        off = w
        while time.time() < stop:
            batch = [keys[(off + j) % pool] for j in range(32)]
            off += 32
            p = client.pipeline(transaction=False)
            for k in batch:
                p.get(k)
            for v in p.execute():
                if v is not None:
                    total_bytes[w] += len(v)

    start = time.time()
    with ThreadPoolExecutor(max_workers=streams) as ex:
        list(ex.map(worker, range(streams)))
    took = time.time() - start
    return sum(total_bytes) / 1e9 / took, took


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kvops", help="trace to replay (replay mode)")
    ap.add_argument("--getbench", action="store_true", help="closed-loop GET-throughput bar for Chart 1")
    ap.add_argument("--blob-bytes", type=int, default=462848, help="getbench blob size")
    ap.add_argument("--pool", type=int, default=4096, help="getbench key pool")
    ap.add_argument("--secs", type=int, default=30, help="getbench measure seconds")
    ap.add_argument("--addr", default="127.0.0.1:6379")
    ap.add_argument("--streams", type=int, default=8)
    ap.add_argument("--resp2", action="store_true", help="force RESP2 (default is RESP3 in redis-py 8)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    host, port = args.addr.split(":")
    if args.getbench:
        pool = redis.ConnectionPool(host=host, port=int(port), max_connections=args.streams,
                                    protocol=2 if args.resp2 else 3)
        client = redis.Redis(connection_pool=pool)
        client.ping()
        gbps, took = getbench(client, args.blob_bytes, args.streams, args.secs, args.pool)
        rec = {
            "schema_version": 1, "kind": "sweep", "store": "redis-py",
            "redis_py": redis.__version__, "python": sys.version.split()[0],
            "goos": sys.platform, "goarch": "python", "seed": 0,
            "cell": {"blob_bytes": args.blob_bytes, "streams": args.streams,
                     "mode": "closed", "mix": "get", "resp": 2 if args.resp2 else 3},
            "duration_s": took, "goodput_gbytes_s": gbps, "ops": {},
        }
        line = json.dumps(rec)
        (open(args.out, "a").write(line + "\n") if args.out else print(line))
        print(f"redis-py getbench: {gbps:.3f} GB/s over {took:.1f}s", file=sys.stderr)
        return

    if not args.kvops:
        sys.exit("replay mode needs --kvops (or use --getbench)")
    blob_bytes, records = read_kvops(args.kvops)
    payload = fill_bytes(blob_bytes)

    pool = redis.ConnectionPool(
        host=host, port=int(port), max_connections=args.streams,
        protocol=2 if args.resp2 else 3,
    )
    client = redis.Redis(connection_pool=pool)
    client.ping()

    hits = misses = get_bytes = 0
    start = time.time()
    # Records fan out across a thread pool (redis-py releases the GIL on
    # socket I/O); the op sequence per record is deterministic.
    with ThreadPoolExecutor(max_workers=args.streams) as ex:
        futs = [ex.submit(replay_record, client, keys, blob_bytes, payload) for keys in records]
        for fut in futs:
            h, m, gb = fut.result()
            hits += h
            misses += m
            get_bytes += gb
    took = time.time() - start

    keys_total = hits + misses
    rec = {
        "schema_version": 1, "kind": "replay", "store": "redis-py",
        "redis_py": redis.__version__, "python": sys.version.split()[0],
        "goos": sys.platform, "goarch": "python", "seed": 0,
        "cell": {"blob_bytes": blob_bytes, "streams": args.streams, "mode": "asap",
                 "resp": 2 if args.resp2 else 3},
        "duration_s": took,
        "ops_per_s": len(records) / took if took else 0,
        "goodput_gbytes_s": get_bytes / 1e9 / took if took else 0,
        "hit_rate": hits / keys_total if keys_total else 0,
        "ops": {}, "errors_total": 0, "verify_fails": 0,
    }
    line = json.dumps(rec)
    if args.out:
        with open(args.out, "a") as f:
            f.write(line + "\n")
    else:
        print(line)
    print(f"redis-py replay: {len(records)} records, hit_rate={rec['hit_rate']:.4f} (OUTPUT), "
          f"{rec['goodput_gbytes_s']:.3f} GB/s, {took:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
