# Rig HF — Chart-2 TTFT on Hugging Face Jobs (a10g-large)

**Status: NO ACCEPTED NUMBERS.** Three paid attempts so far: two died on
15 GB a10g-small RAM, and run 3 (a10g-large, Llama-3.1-8B) produced a real
COLD arm but a worthless WARM arm — every warm record had
`kvb_hit_delta_total: 0` / `warm_hits_verified: false` because the job's
`local_cpu: false` LMCache config severed the store path (LMCache stages
remote writes THROUGH the local CPU buffer), so kvblockd never received a
byte and the "3–16x speedups" were LMCache serving its own cache. The
harness was rebuilt around a two-phase populate → restart → measure design;
nothing below is accepted until a verified re-run lands in
`bench/results/rig-e/`.

## What it measures

The project's core claim: **reloading KV cache over the network beats
recomputing it on the GPU**. For the same prompt, same model, same box, in
two phases separated by a vLLM restart (mirroring `bench/e2e/cpu` CI
property (d), "hits persist across a vLLM restart"):

1. **populate** — vLLM #1 prefills every sweep prompt once (one prompt per
   (length, rep), unique nonce at token 0). LMCache, with the CI-proven
   `local_cpu: true` config, stores the KV locally AND into kvblockd. The
   driver polls kvblockd's `kvb_bytes_total{dir="in"}` / `kvb_blocks` after
   every prompt and **fails loudly if kvblockd received nothing** — the
   run-3 failure mode can no longer produce numbers. It then waits for the
   async put queue to drain before handing over.
2. **restart** — vLLM is killed and a fresh engine starts. LMCache's
   in-process/local tier dies with the process; kvblockd keeps the blocks.
3. **measure** —
   - **warm / kvblockd reload**: the exact populated prompts. The engine is
     fresh and runs `--no-enable-prefix-caching`, so the KV can ONLY come
     from kvblockd over TCP; each rep must grow `kvb_hits_total` or it is
     recorded `warm_hits_verified: false` (red on the chart, exit 3).
   - **cold / recompute**: fresh prompts (unique nonce at token 0 → every
     LMCache chunk hash differs → guaranteed miss → full prefill).
   - All warm reps run BEFORE any cold rep: cold prefills store junk into
     kvblockd, and once the arena fills, eviction targets unread blocks —
     exactly the not-yet-measured populated ones. Ordering makes
     measure-phase eviction harmless by construction.

Sweep of prefix lengths (default 1k / 4k / 8k / 16k tokens), 5 measured rep
pairs per length plus 1 discarded warmup pair, median + p95 TTFT per arm,
and the speedup ratio at each length — including any lengths where recompute
wins. **TTFT = time from sending the request to the first streamed token**
(OpenAI-compatible SSE endpoint; the timer stops on the first non-empty
token event, never on response completion).

Honesty rules baked in (all exercised by `run_ttft.py --selftest` against a
stub with a driver-controlled metrics endpoint — run before spending money):

- Populate FAILS LOUDLY (exit 2, nothing measured) if kvblockd's put
  counters never grow — a severed store path aborts the run instead of
  producing fiction.
- Every warm rep checks that kvblockd's `kvb_hits_total` grew; a rep where
  it didn't is recorded `warm_hits_verified: false`, flagged on the chart,
  and fails the job (exit 3).
- `job.sh` derives `MAX_MODEL_LEN` from the largest sweep length (run 3
  passed `LENGTHS=...,32000` against a 20480 cap — `submit.sh` now refuses
  that mismatch before billing starts) and sizes the kvblockd arena so every
  populated block stays resident across the restart, with a RAM-fit check
  against the box before anything heavy starts.
- HF Jobs containers likely lack NET_ADMIN, so `tc` link shaping is only
  *attempted* if `TC_RATE_GBIT` is set, and the actual link state
  (`unshaped-loopback` in the expected case) is stamped into every JSONL
  record and printed on the chart. Loopback numbers are loopback numbers.
- The actual model id, vLLM/LMCache versions, GPU (read from `nvidia-smi`),
  flavor, and git ref are stamped into every record; `plot.py` reads the
  conditions box from the data. `warm_isolation: "vllm-restart"` is stamped
  so the isolation mechanism travels with the numbers.
- Actual prompt token counts come from vLLM's `/tokenize` + the streamed
  `usage` — targets are nominal, the stamped count is measured.

## Stack (in-container, one box)

`vllm/vllm-openai:v0.25.1` (amd64 digest
`sha256:f0b9a0dc75a9fca3b6811e3279367b2d6a448055a000bfd13859587d74cef268`,
CUDA torch + vLLM prebuilt, pins match `bench/e2e/cpu/versions.env`), plus in
`job.sh`: Go 1.26.5 → `kvblockd` **built from the checked-out source**,
`pip lmcache==0.5.1`, and the two local packages `python/kvblockd` +
`python/lmcache_kvblockd` (editable). The LMCache config is the EXACT shape
`bench/e2e/cpu/lmcache_kvblockd.yaml` proves in CI (`local_cpu: true`), with
`max_local_cpu_size` raised to 8 GiB (run 3's 3 GiB threw 32 MiB
block-allocation failures on 16k contexts).

Fit on a10g-large (46 GB RAM / 1× A10G 24 GB): KV per token is
~128 KiB for Llama-3.1-8B (measured in run 3: LMCache allocated 32 MiB per
256-token chunk) and ~56 KiB for Qwen2.5-7B; `KV_BYTES_PER_TOKEN` defaults
to the 128 KiB upper bound. Default sweep (1k+4k+8k+16k) × 6 pairs →
~25 GiB derived arena + 8 GiB LMCache pool + 6 GiB vLLM reserve ≈ 39 GiB of
46; `job.sh` refuses to start if the budget exceeds 92% of the box (which is
also why a10g-small's 15 GB cannot run this).

## Model

Default `Qwen/Qwen2.5-7B-Instruct` (ungated). Swap with one flag:
`MODEL=meta-llama/Llama-3.1-8B-Instruct bench/rigs/hf-gpu/submit.sh` (gated;
needs granted access on the `HF_TOKEN` account). The model id is stamped
into every record either way.

## How to run

Prereqs: `hf` CLI at `~/.kvb-hf/bin/hf`, logged in, positive credit balance,
and **this directory pushed to `main`** (the job clones the public GitHub
tarball — it cannot see local uncommitted files).

```bash
python3 bench/rigs/hf-gpu/run_ttft.py --selftest   # free; proves timer + honesty gates
DRY_RUN=1 bench/rigs/hf-gpu/submit.sh   # print the exact hf command, spend nothing
bench/rigs/hf-gpu/submit.sh             # submit (interactive confirmation)
```

Knobs: `MODEL`, `GIT_REF`, `LENGTHS`, `REPS`, `WARMUP`, `GEN_TOKENS`,
`MAX_MODEL_LEN` (derived from `LENGTHS` when unset), `KVBD_ARENA_BYTES`
(derived when unset), `LMC_MAX_LOCAL_CPU_GB`, `KV_BYTES_PER_TOKEN`,
`TIMEOUT` (default 2h), `FLAVOR` (default a10g-large), `RESULTS_REPO`
(optional HF dataset repo for the JSONL). The job is non-interactive and
exits nonzero on any failure.

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

a10g-large is billed per minute (a10g-small was \$1.00/hr; large costs more
— check current HF Jobs pricing before submitting). Expected wall clock:
~10 min install + ~5 min first vLLM boot (weight download) + populate sweep
+ ~3 min restart + measure sweep ≈ **under 1h**; the 2h timeout caps any
single run.

## Known risks (pre-run)

- **RAM**: the arena is prefaulted; the derived budget leaves ~7 GB slack on
  a10g-large at defaults. If the fit check refuses or the box OOMs, cut
  `REPS` or drop the 16k point rather than shrinking `KVBD_ARENA_BYTES`
  below the populated-set size (eviction would silently unverify warm reps).
- **LMCache internals inferred, not verified** until the next paid run: that
  remote puts drain asynchronously (the driver polls counters rather than
  trusting request completion), and that a fresh vLLM process holds no
  reusable KV anywhere outside kvblockd. Both are exactly what the per-rep
  counter checks exist to catch.
- **lmcache pip install** may try to move torch/vLLM or build CUDA ext
  without nvcc; `job.sh` retries with `NO_GPU_EXT=1` and hard-fails if the
  image's CUDA torch or vLLM version changed.
- **`tc` shaping** almost certainly unavailable (no NET_ADMIN): expected and
  disclosed, the run is unshaped loopback. Emulated-link points (25/50 Gbps)
  stay with the AWS rig plan (`bench/rigs/aws-gpu/`).
- **Gated Llama**: only after access is granted; Qwen is the default.
- The image is pinned by tag + recorded digest, but HF pulls by tag.
