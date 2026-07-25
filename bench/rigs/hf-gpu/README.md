# Rig HF — Chart-2 TTFT on Hugging Face Jobs (a10g-small)

**Status: NOT-RUN.** Nothing in this directory has produced numbers yet;
no JSONL exists in `bench/results/rig-e/`. Everything below is the built and
locally-verified harness, waiting on an approved paid run.

## What it measures

The project's core claim: **reloading KV cache over the network beats
recomputing it on the GPU**. For the same prompt, same model, same box:

- **cold / recompute** — a fresh prompt (unique nonce at token 0, so every
  LMCache chunk hash differs → guaranteed miss). vLLM does the full prefill;
  LMCache stores the KV blocks into kvblockd.
- **warm / cache hit** — the same prompt again. vLLM runs with
  `--no-enable-prefix-caching` and LMCache with `local_cpu: false`, so the
  KV can ONLY come back from kvblockd over TCP.

Sweep of prefix lengths (default 1k / 4k / 8k / 16k / 32k tokens), 5 measured
rep pairs per length plus 1 discarded warmup pair, median + p95 TTFT per arm,
and the speedup ratio at each length — including any lengths where recompute
wins. **TTFT = time from sending the request to the first streamed token**
(OpenAI-compatible SSE endpoint; the timer stops on the first non-empty token
event, never on response completion — proven by `run_ttft.py --selftest`
against a stub whose first token is deliberately delayed).

Honesty rules baked in:

- Every warm rep checks that kvblockd's `kvb_hits_total` grew; a rep where it
  didn't is recorded `warm_hits_verified: false`, flagged on the chart, and
  fails the job (exit 3) — a "warm" number that never touched the daemon is
  worthless.
- HF Jobs containers likely lack NET_ADMIN, so `tc` link shaping is only
  *attempted* if `TC_RATE_GBIT` is set, and the actual link state
  (`unshaped-loopback` in the expected case) is stamped into every JSONL
  record and printed on the chart. Loopback numbers are loopback numbers.
- The actual model id, vLLM/LMCache versions, GPU, and git ref are stamped
  into every record; `plot.py` reads the conditions box from the data.
- Actual prompt token counts come from vLLM's `/tokenize` + the streamed
  `usage` — targets are nominal, the stamped count is measured.

## Stack (in-container, one box)

`vllm/vllm-openai:v0.25.1` (amd64 digest
`sha256:f0b9a0dc75a9fca3b6811e3279367b2d6a448055a000bfd13859587d74cef268`,
CUDA torch + vLLM prebuilt, pins match `bench/e2e/cpu/versions.env`), plus in
`job.sh`: Go 1.26.5 → `kvblockd` **built from the checked-out source**
(release v0.2.0 lags the current code), `pip lmcache==0.5.1`, and the two
local packages `python/kvblockd` + `python/lmcache_kvblockd` (editable).
Config shapes are copied from `bench/e2e/cpu/` with two deliberate deltas
(`local_cpu: false`, DRAM arena 3 GiB) documented in `job.sh`.

Fit on a10g-small (4 vCPU / 15 GB RAM / 1× A10G 24 GB / 110 GB disk):
Qwen2.5-7B KV is 28 layers × 4 KV heads × 128 dim × 2 (K+V) × 2 B =
**56 KiB/token** → a 32k-token prompt is ~1.75 GiB of KV; fits the 3 GiB
daemon arena and the ~5 GiB of GPU KV space left after bf16 weights at
`--gpu-memory-utilization 0.90`. Host RAM budget: vLLM ~4–5 GB + arena 3 GiB
+ LMCache pinned staging pool 3 GB + driver ≈ 11–12 GB of 15.

## Model

Default `Qwen/Qwen2.5-7B-Instruct` (ungated). Swap with one flag:
`MODEL=meta-llama/Llama-3.1-8B-Instruct bench/rigs/hf-gpu/submit.sh` — gated;
requires granted access on the `HF_TOKEN` account (requested, not yet
confirmed). The model id is stamped into every record either way.

## How to run

Prereqs: `hf` CLI at `~/.kvb-hf/bin/hf`, logged in, positive credit balance,
and **this directory pushed to `main`** (the job clones the public GitHub
tarball — it cannot see local uncommitted files).

```bash
DRY_RUN=1 bench/rigs/hf-gpu/submit.sh   # print the exact hf command, spend nothing
bench/rigs/hf-gpu/submit.sh             # submit (interactive confirmation)
```

Knobs: `MODEL`, `GIT_REF`, `LENGTHS`, `REPS`, `TIMEOUT` (default 2h),
`RESULTS_REPO` (optional HF dataset repo for the JSONL). The job is
non-interactive and exits nonzero on any failure.

## How results come back

1. **Primary (always on):** every JSONL record is printed to the job log
   behind a `CHART2JSONL ` marker:

   ```bash
   ~/.kvb-hf/bin/hf jobs logs <job_id> | sed -n 's/^.*CHART2JSONL //p' \
     > bench/results/rig-e/chart2-ttft.jsonl
   python3 bench/report/plot.py chart2 \
     --in bench/results/rig-e/chart2-ttft.jsonl --out chart2.png
   ```

2. **Optional:** set `RESULTS_REPO=<user>/<dataset>` and `job.sh` also
   uploads the JSONL to that HF dataset via `huggingface_hub` (non-fatal on
   failure; the log lines remain authoritative).

## Cost and runtime

a10g-small is **$1.00/hr, billed per minute**. Expected wall clock: ~10 min
install (Go + pip + model download on HF's network) + ~5 min vLLM startup +
~20–30 min sweep ≈ **45–60 min ≈ $1**. Budget **~$4** to cover one debug
re-run and one model swap; the 2h timeout caps any single run at $2.

## Known risks (pre-run)

- **RAM (15 GB)**: the ledger above leaves ~3 GB slack; if the job OOMs,
  drop the 32k point (`LENGTHS=1024,4096,8192,16384`) or shrink
  `KVBD_ARENA_BYTES` / `LMC_MAX_LOCAL_CPU_GB`.
- **lmcache pip install** may try to move torch/vLLM or build CUDA ext
  without nvcc; `job.sh` retries with `NO_GPU_EXT=1` and hard-fails if the
  image's CUDA torch or vLLM version changed.
- **Entrypoint override**: HF Jobs runs the given command *instead of* the
  image entrypoint (K8s-style; HF's own docs run `command=["duckdb", ...]`
  on `duckdb/duckdb`, whose entrypoint would otherwise swallow it) — but
  this rig is the first time we exercise it with `vllm/vllm-openai`.
- **`tc` shaping** almost certainly unavailable (no NET_ADMIN): expected and
  disclosed, the run is unshaped loopback. Emulated-link points (25/50 Gbps)
  stay with the AWS rig plan (`bench/rigs/aws-gpu/`).
- **Gated Llama**: only after access is granted; Qwen is the default.
- The image is pinned by tag + recorded digest, but HF pulls by tag.
