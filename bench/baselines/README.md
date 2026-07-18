# bench/baselines — the comparison bars for Chart #1

Every store gets the **same host, same sysctls, same payloads, same op
stream, warmed identically** (methodology rule 7). Two clients per Redis-
family store (rule 3, the "two-client rule"):

- **go-redis v9.21.0 zero-copy** (`GetToBuffer`/`SetFromBuffer`) — the
  *strongest available* client bar, driven by `kvbench --target redis|valkey`.
- **redis-py 8.0.1** (`redis_py_driver/driver.py`) — the client **LMCache
  actually ships**; the gap between the two bars *is* the client-side story.

The `≥10× vs redis-py` gate compares kvblockd's batched-GET throughput
against the redis-py path — both configs published, no massaging (rule 3+4).

## Services

`docker-compose.yml` pins Valkey 8.1, Redis 7.4, etcd, and Mooncake-TCP by
digest (PENDING until the rig pull). One profile per store; `maxmemory` is
set per capacity cell on the command line.

## Drivers

- `redis_py_driver/driver.py` — two modes: `--getbench` (closed-loop
  pipelined batched GETs → the Chart-1 throughput bar, comparable to the
  go-redis and kvblockd GET bars, `mode:"closed"`) and the default `.kvops`
  replay (pipelined per-key EXISTS/GET/SET, same adaptive EXISTS→GET→PUT
  logic as the Go replayer). Payloads are exactly `blob_bytes` for a fair
  goodput comparison.
- **Mooncake driver — NOT YET WRITTEN.** The Mooncake bar is timeboxed to 3h
  on the rig; its driver (official Python API only, `store.setup(...,
  protocol="tcp")`, never a homemade Go client) is authored at session start
  against the pinned image. If no stable TCP config reproduces, the bar ships
  as unreproducible with configs + errors attached (rule 4).

## Honesty posture

- Mooncake bar is timeboxed to 3h. If no stable TCP config reproduces, the
  bar is published as unreproducible with configs + errors attached and a
  standing re-run offer — never silently dropped (rule 4).
- All numbers quote %-of-iperf3-ceiling beside the absolute (rule 2).
