# A1 transport log — raw run cells

**Kill-gate A1:** ≥12 GB/s Linux loopback AND ≥85% of measured iperf3 ceiling on the c6in.8xlarge pair. <6 GB/s after tuning ⇒ KILL, pivot headline.

**Status:** Mac-loopback shakedown done. The real verdict is the AWS c6in pair run — these Mac numbers are a **developer-machine sanity check, NOT the gate and NOT a claim.**

---

## Run 1 — Apple M2 (8 core), macOS, loopback (127.0.0.1), xferspike client→server, 2s/cell

| streams | frame | GB/s | frames/s | cpu cores |
|--------:|------:|-----:|---------:|----------:|
| 1  | 1 MB  | **14.11** | 13458 | 1.28 |
| 1  | 4 MB  | 12.35 | 2944 | 1.13 |
| 1  | 16 MB | 6.92  | 413  | 0.91 |
| 4  | 1 MB  | 13.13 | 12522 | 2.33 |
| 4  | 4 MB  | 10.03 | 2392 | 2.73 |
| 4  | 16 MB | 9.22  | 550  | 2.70 |
| 16 | 1 MB  | 9.28  | 8850 | 2.59 |
| 16 | 4 MB  | 8.02  | 1912 | 2.97 |
| 16 | 16 MB | 7.80  | 465  | 2.79 |
| 32 | 1 MB  | 5.51  | 5251 | 2.16 |
| 32 | 4 MB  | 5.47  | 1305 | 2.12 |
| 32 | 16 MB | 4.74  | —    | 1.98 |
| 64 | *any* | ENOBUFS ("writev: no buffer") | — | — |

Data integrity confirmed every cell: server received-bytes == client sent-bytes exactly.

## Reading the results (important — this shapes the cloud run)

1. **Peak 14.11 GB/s clears the ≥12 GB/s loopback bar.** The Go code path itself is fast enough; no structural problem.
2. **The curve is INVERTED vs a real network** — throughput is *highest at 1 stream* and *declines* as streams increase. This is expected and must not be misread: loopback has no real NIC — it's a kernel memcpy, and one client+one server goroutine already saturate memory bandwidth on ~2 cores. Extra streams only add scheduler contention. **The parallel-streams thesis (many conns beat one) only proves out on a real network**, where a single flow is limited by single-core packet processing while the NIC has spare capacity. That proof is the c6in job, not loopback's.
3. **64-stream loopback hits macOS ENOBUFS** (kernel loopback send-buffer exhaustion). This is a well-known macOS-loopback limitation, not an xferspike defect and not expected on the Linux 50 GbE gate rig. See finding below.
4. **Bigger frames ≠ faster on loopback** (16 MB slower than 1 MB): again a memory-bandwidth/latency artifact of loopback; on a real link, larger frames amortize syscall cost and help. Re-measure on the cloud pair.

## Rig findings to address (batched with the review-ladder verdict)

- **F-a1-1 (resilience):** a single client stream's write error (e.g. transient ENOBUFS) returns from `runClient` and prints NO result line — losing the data from all other working streams. A measurement rig should tolerate a stream dropping and still report aggregate over survivors, and/or treat transient ENOBUFS as retry-with-backoff rather than fatal. (Confirm/priority via the ladder.)
- **F-a1-2 (harness):** rapid sequential runs on loopback pile up TIME_WAIT sockets → ephemeral-port pressure. The sweep script needs `ulimit -n` raised + spacing between cells (already applied in the re-run). The real Day-4 rig uses long-lived connections on fresh instances, so this is a local-harness concern only.

## Next
- Next: run `bench/rigs/aws-transport/` on 2× c6in.8xlarge, measure iperf3 ceiling first, grade ≥85% of it. That is the A1 verdict of record.

## Review-ladder outcome (2026-07-15, 8-agent full ladder + CTO gate)

Verdict: FIX-FIRST → all applied. GB/s measurement verified HONEST (no path inflates it; client==server bytes on happy path).

Fixed:
- **[HIGH, blocker] no write deadline → stalled-peer hang + masked errors.** Added `SetWriteDeadline` per write; only `os.ErrDeadlineExceeded` is a clean stop, all other errors returned unconditionally. **Proven:** truly-stalled peer now exits at exactly 2.0s / exit 0 / honest ~0 GB/s (was: indefinite hang, no output).
- **[MED] unit mislabel.** `gbps` → `gbytes_per_s` (+ `gbit_per_s` = ×8). The A1 "12 GB/s" = 96 Gbit/s; iperf3 reports Gbit/s — compare correctly on the cloud run.
- **[MED] `cpu_cores` → `cpu_cores_sender`** (RUSAGE_SELF = client only, ~2× low as a whole-system figure; do not use for efficiency claims).
- **[LOW] wire byte-order locked:** magic now big-endian so the wire reads "KVB1" in a hexdump; numeric fields stay little-endian. Cheap fixes: client rejects frame-bytes > max-frame with a clear message; server logs desync drops; bounded `wg.Wait()` on server shutdown; style nits (`errors.New`, `var buf []byte`, handled encode error).

Deferred (not blocking A1; scheduled):
- **[MED] socket buffers set post-handshake** — for the Day-4 cloud %iperf3 run, either set via `ListenConfig.Control`/`Dialer.Control` (setsockopt before connect) or rely on kernel autotune and treat `-sndbuf/-rcvbuf` as loopback-only. Decide in the aws-transport rig.
- **[LOW] server memory amplification** (alloc up to max-frame before body) — firewall the cloud bind to the client IP; acceptable for a rig.
- **[LOW] bytes_total includes 16B header** (wire throughput, not goodput) — negligible at ≥1 MiB frames; report goodput separately if ever needed.
