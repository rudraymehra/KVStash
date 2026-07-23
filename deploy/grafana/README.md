# Grafana dashboard

`kvblockd-dashboard.json` — one importable dashboard over the daemon's
`kvb_*` Prometheus families (see `internal/metrics/metrics.go`; every query
uses only metric names the daemon actually exports).

## Import

Grafana 11: **Dashboards → New → Import → Upload JSON file**, then pick your
Prometheus datasource in the `Prometheus` variable (top left). Scrape config:

```yaml
scrape_configs:
  - job_name: kvblockd
    static_configs:
      - targets: ["<host>:9442"]   # metrics_addr
```

## What's on it

| Row | Panels |
|---|---|
| Ops & throughput | ops/s by verb, payload bytes in/out, GET hit ratio, mean + p99 latency |
| Tier residency | blocks/bytes by tier (DRAM/NVMe/S3), arena occupancy gauge |
| Tenants | resident bytes vs quota per (namespace, tier), served/pinned bytes by namespace id |
| Cold tier (S3) | spill/restore flow, errors & drops, s3-resident blocks |
| Reclaim & demote | demote/promote flow, pressure refusals, reclaims/evictions, NVMe capacity |

Notes:

- **p99 panel needs native histograms**: `kvb_op_seconds` is a native
  histogram (no classic buckets), so the p99 panel requires Prometheus
  running with `--enable-feature=native-histograms`. Without it that one
  panel shows *No data* — the mean-latency panel beside it works everywhere.
- The `$namespace` variable filters the tenant row; data-plane counters
  (`kvb_bytes_total`, `kvb_pinned_bytes`) are labeled with the numeric
  namespace **id** from `namespaces.yaml`, the tenant quota gauges with the
  namespace **name**.
- NVMe/S3 panels are empty on a DRAM-only daemon by design — those families
  only appear once `nvme_paths` / `s3_bucket` are configured.
- No S3-GC series exists yet; object GC is observable indirectly (restores
  vs spills). If a `kvb_s3_gc_*` counter lands, add it to the errors panel.
