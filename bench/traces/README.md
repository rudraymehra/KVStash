# bench/traces — production KV-cache traces → `.kvops`

Trace replay is the honest-hit-rate evidence: the store evicts itself, so
**hit rate is an output, never a knob** (methodology rule 9). Two sources,
both Apache-2.0:

| Trace | Source | Ideal hit rate | Counts |
|---|---|---|---|
| **Bailian** | `alibaba-edu/qwen-bailian-usagetraces-anon` (ATC'25, Alibaba/SJTU) | **62% chat / 54% API** — not the marketing 90% | 4 files |
| **Mooncake FAST25** | `kvcache-ai/Mooncake` `FAST25-release/traces/` | — | 12,031 / 23,608 / 3,993 |

## Fetch

```bash
bench/traces/fetch.sh            # clones at pinned SHAs, writes SHA256SUMS + PROVENANCE.txt
```

## Convert (count-exactness is the acceptance gate)

```bash
kvbench convert --format bailian  --in data/bailian/<file>.jsonl \
  --out data/bailian.kvops  --trace bailian-chat  --blob-bytes 462848 \
  --expect-requests <published-count>

kvbench convert --format mooncake --in data/mooncake/<file>.jsonl \
  --out data/mooncake.kvops --trace mooncake-a    --blob-bytes 462848 \
  --expect-requests 12031
```

`--expect-requests` makes a silently dropped line a hard failure — a wrong
converter invalidates every hit-rate claim, so it stops the run.

## Block-size mapping

- **Bailian** 16-token blocks → **one blob each**; `--blob-bytes` is the
  model parameter, published both ways: 462,848 B ("0.44 MiB", 16-token
  vLLM block for an 8B-class model) and 2.5 MiB (70B-class).
- **Mooncake** 512-token blocks → **32 sub-keys** `TraceKey(trace, id, i)`,
  each a 16-token-equivalent blob, preserving prefix-chain semantics (a
  512-token hit = 32 consecutive sub-hits).

Key derivation (`kvbench-trace-v1`): BLAKE3-256 over `"kvbench-trace-v1\0"`
then u32-LE length-prefixed UTF-8 fields — the same recipe as the product
wire key with a distinct domain, so a bench key can never collide with a
content key. The Python replayer (`python/kvblockd/tools/kvops_replay.py`)
reproduces the op sequence for cross-language parity.

## Sanity gate

Infinite-capacity Bailian replay must land in the **54–62%** band — if it
doesn't, the converter is wrong. Stop and fix before any rig time.
