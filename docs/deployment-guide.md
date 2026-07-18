# kvblockd deployment guide

The quickstart in the [README](../README.md) runs a DRAM-only daemon in one
minute. This guide is the production path: kernel prep, the NVMe tier,
systemd, metrics, and tenants. Everything here is the configuration the
published benchmarks actually ran — nothing aspirational.

## 1. Install

Pick one:

- **Script:** `curl -fsSL https://raw.githubusercontent.com/kvstash/kvblockd/main/scripts/install.sh | sh`
- **Tarball:** grab `kvblockd_<ver>_<os>_<arch>.tar.gz` from releases, verify
  against `checksums.txt`, untar; the binaries are static — copy them anywhere.
- **Docker:**
  ```sh
  docker run --rm -p 9440:9440 -p 9442:9442 \
    -v $PWD/config:/config \
    -e KVBLOCKD_NAMESPACES=/config/namespaces.yaml \
    -e KVBLOCKD_METRICS_ADDR=0.0.0.0:9442 \
    ghcr.io/kvstash/kvblockd:<ver>
  ```
  (`FROM scratch` — no shell inside. The metrics override matters: the
  default bind is container-loopback, dead through `-p`.)
- **systemd:** [`deploy/kvblockd.service`](../deploy/kvblockd.service) — the
  header comment is the install procedure. Mind `TimeoutStopSec`: SIGTERM
  starts a graceful drain and systemd must outwait it.

macOS is a **dev platform only** — the daemon runs for development but no
durability or performance claims apply (`F_NOCACHE` is not `O_DIRECT`).

## 2. Network sysctls

For ≥25 GbE links, apply the ESnet-derived profile the benchmark rigs use:
[`bench/rigs/sysctl-esnet.conf`](../bench/rigs/sysctl-esnet.conf)
(`sudo sysctl -p bench/rigs/sysctl-esnet.conf`). The daemon requests 16 MiB socket
buffers per connection (`sock_sndbuf`/`sock_rcvbuf`); untuned hosts silently
clamp them — the daemon logs the effective sizes at accept, so check the log
once. Jumbo MTU (9001) helped on the AWS rigs; measure your own ceiling with
`iperf3 -P 8` first and compare kvblockd against *that* number, not the NIC's
label.

## 3. Memory: the DRAM arena + hugepages

`dram_arena_bytes` is mmap'd OUTSIDE the Go heap — the arena is the working
set, the heap stays small metadata. Size the arena to the KV blocks you want
resident.

Hugepages (recommended above ~8 GiB arenas):

```sh
# 2 MiB pages: reserve BEFORE starting the daemon (pages × 2 MiB = arena size)
echo 8192 | sudo tee /proc/sys/vm/nr_hugepages   # 16 GiB
```

then `dram_hugepages: true`. If reservation fails the daemon tells you at
startup instead of degrading silently. The systemd unit ships
`LimitMEMLOCK=infinity` for this.

## 4. The NVMe tier

Give each physical device its own filesystem and one volume dir:

```sh
sudo mkfs.xfs -f /dev/nvme1n1 && sudo mkdir -p /mnt/nvme1/kvb
sudo mount -o noatime /dev/nvme1n1 /mnt/nvme1/kvb
```

```yaml
nvme_paths: ["/mnt/nvme1/kvb"]        # one entry per device; APPEND-ONLY list
nvme_max_bytes: 3400000000000          # REQUIRED: total budget across volumes
```

Notes that matter:

- The tier is **inert until `nvme_paths` is set** — a DRAM-only daemon
  behaves byte-for-byte identically.
- `nvme_paths` order is positional ownership: append new volumes, never
  reorder or remove entries between restarts.
- I/O is `O_DIRECT`, 4 KiB-aligned, log-structured 256 MiB segments; on
  filesystems that refuse `O_DIRECT` (tmpfs/overlay) the daemon falls back
  to buffered I/O and says so — fine for testing, not for endurance claims.
- The crash contract (what a COMMIT ack promises, what recovery restores) is
  specified and measured in [DESIGN.md](DESIGN.md); the kill -9 harness in
  `test/crash/` is runnable against your own hardware.
- With `ProtectSystem=strict` in the systemd unit, add every volume dir to
  `ReadWritePaths`.

## 5. Tenants (namespaces)

Auth model: a connection presents `(namespace, token)` once at HELLO and
lives inside that namespace. **No namespaces file = no one can connect**
(secure by default). Copy [`config/namespaces.yaml`](../config/namespaces.yaml),
replace the demo tenant, keep ids stable forever (they key on-disk ownership):

```yaml
namespaces:
  - { name: team-a, id: 2, token: "a-real-secret" }
```

`pinned_bytes_cap` bounds per-namespace pinned bytes today; full quota/QoS
enforcement ships v0.2. Transport is plaintext TCP in v1 — run on a trusted
segment or behind a TCP-terminating proxy; do not expose 9440 to the internet.

## 6. Metrics + health

`metrics_addr` (default `127.0.0.1:9442`) serves:

- `/metrics` — Prometheus (`kvb_*` instruments + process collector)
- `/healthz` — readiness (used by the systemd/docker examples)
- `/debug/pprof` — keep it on loopback; the daemon warns if you bind wider

Scrape config: `job_name: kvblockd`, `static_configs: [{targets: ["host:9442"]}]`.

## 7. Sizing quick reference

| Deployment | Arena | NVMe | Notes |
|---|---|---|---|
| Laptop demo | 1 GiB | none | the quickstart default |
| Single inference node | 16–64 GiB + hugepages | optional, 1 device | LMCache remote backend on loopback/ToR |
| Shared cache node | 64–256 GiB + hugepages | 1–8 devices | 25–100 GbE; apply §2 sysctls; measure the ceiling |

Capacity rule of thumb: blocks are ~0.4–2.5 MB; the DRAM tier should hold
your hot prefix set, NVMe the warm tail. When in doubt run `bench/kvbench`
against the candidate box — the harness exists so you never have to trust
our numbers over yours.
