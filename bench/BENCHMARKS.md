# kvblockd benchmark scoreboard

Honest, reproducible loopback numbers for the Week-2 wire path. **Loopback is
a developer sanity check, not the quotable figure** — the AWS 50 GbE pair is
the real gate (per `docs/notes/a1-log.md`). Every number here is a *ratio
against a same-shape ceiling* where one exists; absolute GB/s on a laptop
varies ±15% with thermal state.

## How to reproduce

```
# codec microbenchmarks (zero-alloc hot paths)
go test -run '^$' -bench Benchmark ./internal/protocol/

# in-package GET/PUT/EXISTS (shares one scheduler between client and server)
go test -run '^$' -bench Benchmark ./internal/server/

# out-of-process (production shape: daemon + load generator in separate procs)
go build -o /tmp/kvblockd ./cmd/kvblockd
go build -o /tmp/getbench ./bench/kvbench/getbench
/tmp/kvblockd --config <cfg> &            # namespaces_path with one tenant
/tmp/getbench -streams 4 -secs 4          # add -noverify to isolate hash cost

# raw-socket ceilings (the fair same-shape baselines)
go run ./bench/microbench/rawget -streams 4 -secs 3   # GET shape
```

## GET (read path) — the headline

| config (dev-machine loopback, two-process, interleaved) | GB/s | % of raw ceiling |
|---|---:|---:|
| kvblockd, verification off | **11.27** | **102%** |
| kvblockd, full xxh3 verification (default) | 9.75 | 88% |
| raw socket, no protocol/auth/checksum (`rawget`) | 11.04 | 100% |

The protocol stack (framing, auth, credit, descriptors, dispatch) runs at
parity-or-better with a plain raw socket doing the same work. The remaining
12% is integrity verification — real work, dialable via `SkipVerify`. The
ceiling is the kernel loopback copy path; even raw sockets can't beat it.
Started the week at 7.4 GB/s; +32% default / +52% wire-path after the tuning
passes.

## Block-size range (the real workload is 0.4–2.5 MB)

Flat 5–9 GB/s, no size cliff: 0.4 MiB ≈ 7.7–8.5 · 1 MiB ≈ 7.3–7.9 ·
2.5 MiB ≈ 5.4–7.6 GB/s (the 2.5 MiB dip is the memory-bandwidth wall).

## PUT (write path)

~5.9 GB/s at 4 streams; 1-RTT pipelined 277 µs/op vs 353 µs two-RTT (−20%),
25 vs 32 allocs. **PUT/GET ratio ~2×, better than every store studied**
(Redis's own SET/GET = 3.8× at 4 MB). pprof: 64% syscall, 18% goroutine-wakeup
handoffs, memmove 1.9%, xxh3 1.6% — syscall/handoff-bound, not copy-bound.
Per-op allocation cut 3.1 MB → 1.05 MB via ownership-transfer commit +
first-chunk exact-cap staging + incremental digest.

## EXISTS (metadata RTT)

Single-probe 25.8 µs (macOS) is at the interrupt-driven kernel floor
(~27 µs Linux TCP_RR; only busy-poll/kernel-bypass go lower — out of scope).
Pipelining is the lever: depth-16 = 5.0 µs = **5.1× probe throughput** (macOS;
~1.4× on lower-RTT Linux — the lever scales with the RTT it hides).

## Codecs (per-frame hot paths)

All zero-alloc: MarshalHeader 7.8 ns · ParseHeader 13.9 ns · DecodeKeyList(32)
123 ns · AppendGetRespHeader(32) 34 ns · DecodePutBegin 0.3 ns.

## Two-kernel confirmation

Linux 6.10 (Docker VM, carries virtualization overhead — shape only):
codecs zero-alloc; GET peaks at 4 streams and declines at 16 (the a1-log
inverted-stream curve, independent of macOS); ~2× PUT/GET ratio holds.

## Honesty notes (things caught and NOT reported as real)

- A 49 GB/s reading (4× the ceiling) came from sweeping block sizes against one
  reused daemon — write-once conflict credited new-size bytes against stale
  data. Discarded; sizes must sweep against fresh daemons.
- An "unexpected EOF" at 2.5 MB was a `pkill`→rebind race in the harness, not a
  block-path bug (5 clean consecutive runs, empty server log).
- Inline hashing (vs the sidecar) measured *slower* (6.9 vs 9.75) — rejected.
- The 14.1 GB/s Week-1 xferspike figure is a one-way single-buffer blast, a
  different workload shape; not comparable to the GET round trip.

## Verification levers deferred to the real-NIC rigs (evidence-backed)

`readv` for the PUT read path (golang/go#17607), MSG_ZEROCOPY at 1 MiB sends
(zero on loopback, 5–15% on NICs), OOO/pipelined client demux (loopback-flat,
network-decisive), LAN-BDP socket sizing. Memory bandwidth is the terminal
ceiling (Netflix: 30 GB/s on ~150 GB/s DRAM).
