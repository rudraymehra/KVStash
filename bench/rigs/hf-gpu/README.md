# Rig HF — Chart-2 TTFT on Hugging Face Jobs (a10g-large)

**Status: NO ACCEPTED NUMBERS.** Paid attempts so far: two died on 15 GB
a10g-small RAM; run 3 (a10g-large, Llama-3.1-8B) produced a real COLD arm
but a worthless WARM arm — every warm record had `kvb_hit_delta_total: 0` /
`warm_hits_verified: false` because the LMCache config severed the store
path, so kvblockd never received a byte and the "3–16x speedups" were
LMCache serving its own cache; four further runs never moved a byte through
the LMCache plugin route at all. The rig now uses the **native connector**
(`vllm_kvblockd.KvblockdConnector`) — the one path with demonstrated bytes:
`bench/e2e/cpu/local-docker.sh` proves it end-to-end for free (real prefill
bytes into kvblockd, hits surviving an engine restart), and its
`ttft-rehearsal` mode dress-rehearses this rig's exact
populate → restart → measure flow before any money is spent. Nothing below
is accepted until a verified run lands in `bench/results/rig-e/`.

## What it measures

The project's core claim: **reloading KV cache over the network beats
recomputing it on the GPU**. For the same prompt, same model, same box, in
two phases separated by a vLLM restart (mirroring `bench/e2e/cpu` CI
property (d), "hits persist across a vLLM restart"):

1. **populate** — vLLM #1 prefills every sweep prompt once (one prompt per
   (length, rep), unique nonce at token 0). The native connector stores each
   complete KV block into kvblockd synchronously on the prefill pass. The
   driver polls kvblockd's `kvb_bytes_total{dir="in"}` / `kvb_blocks` after
   every prompt and **fails loudly if kvblockd received nothing** — the
   run-3 failure mode can no longer produce numbers. It then waits for the
   put counters to quiesce before handing over (instant for the sync native
   path; armor in case a connector goes async).
2. **restart** — vLLM is killed and a fresh engine starts. Any engine-local
   KV dies with the process; kvblockd keeps the blocks.
3. **measure** —
   - **warm / kvblockd reload**: the exact populated prompts. The engine is
     fresh and runs `--no-enable-prefix-caching`, so the KV can ONLY come
     from kvblockd over TCP; each rep must grow `kvb_hits_total` or it is
     recorded `warm_hits_verified: false` (red on the chart, exit 3).
   - **cold / recompute**: fresh prompts (unique nonce at token 0 → every
     block key in the prefix chain differs → guaranteed miss → full
     prefill).
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
- After `/health`, `job.sh` asserts the engine's log actually mentions
  `KvblockdConnector` — a dropped kv-transfer config (engine serving with no
  connector at all) fails in seconds, not GPU-minutes. The native path has
  no LMCache-style silently-degraded init mode: a broken connector kills the
  engine boot, and a mute daemon is caught by populate's put receipt.
- HF Jobs containers likely lack NET_ADMIN, so `tc` link shaping is only
  *attempted* if `TC_RATE_GBIT` is set, and the actual link state
  (`unshaped-loopback` in the expected case) is stamped into every JSONL
  record and printed on the chart. Loopback numbers are loopback numbers.
- The actual model id, vLLM version, connector version, GPU (read from
  `nvidia-smi`), flavor, and git ref are stamped into every record;
  `plot.py` reads the conditions box from the data.
  `warm_isolation: "vllm-restart"` is stamped so the isolation mechanism
  travels with the numbers.
- Actual prompt token counts come from vLLM's `/tokenize` + the streamed
  `usage` — targets are nominal, the stamped count is measured.

## Stack (in-container, one box)

`vllm/vllm-openai:v0.25.1` (amd64 digest
`sha256:f0b9a0dc75a9fca3b6811e3279367b2d6a448055a000bfd13859587d74cef268`,
CUDA torch + vLLM prebuilt, pin matches `bench/e2e/cpu/versions.env`), plus
in `job.sh`: Go 1.26.5 → `kvblockd` **built from the checked-out source**,
and the two local packages `python/kvblockd` + `python/vllm_kvblockd`
(editable). No LMCache: vLLM loads `KvblockdConnector` out-of-tree via
`kv_connector_module_path`, and the daemon endpoint/namespace/token/streams
travel in `kv_connector_extra_config` — the exact shape
`bench/e2e/cpu/local-docker.sh` proves. There is no library between the
engine and the daemon, so there is no second config channel to silently
miss.

Fit on a10g-large (46 GB RAM / 1× A10G 24 GB): KV per token is ~128 KiB for
Llama-3.1-8B (measured in run 3) and ~56 KiB for Qwen2.5-7B;
`KV_BYTES_PER_TOKEN` defaults to the 128 KiB upper bound. The native
connector's blob is the raw paged block + a 32-byte prefix, so the same
bound holds. Default sweep (1k+4k+8k+16k) × 6 pairs → ~25 GiB derived arena
+ 2 GiB connector staging headroom (`CONNECTOR_STAGING_GB`; the connector
moves one paged block at a time through a transient CPU tensor, ~MB scale)
+ 6 GiB vLLM reserve ≈ 33 GiB of 46; `job.sh` refuses to start if the
budget exceeds 92% of the box (which is also why a10g-small's 15 GB cannot
run this).

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
python3 bench/rigs/hf-gpu/run_ttft.py --selftest    # free; proves timer + honesty gates
bench/e2e/cpu/local-docker.sh ttft-rehearsal        # free; this rig's exact flow on CPU
DRY_RUN=1 bench/rigs/hf-gpu/submit.sh   # print the exact hf command, spend nothing
bench/rigs/hf-gpu/submit.sh             # submit (interactive confirmation)
```

Knobs: `MODEL`, `GIT_REF`, `LENGTHS`, `REPS`, `WARMUP`, `GEN_TOKENS`,
`MAX_MODEL_LEN` (derived from `LENGTHS` when unset), `KVBD_ARENA_BYTES`
(derived when unset), `CONNECTOR_STAGING_GB`, `KV_BYTES_PER_TOKEN`,
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

- **RAM**: the arena is prefaulted; the derived budget leaves ~13 GB slack
  on a10g-large at defaults. If the fit check refuses or the box OOMs, cut
  `REPS` or drop the 16k point rather than shrinking `KVBD_ARENA_BYTES`
  below the populated-set size (eviction would silently unverify warm reps).
- **The native connector's GPU serving path is CPU-validated, not
  GPU-validated**: its store path stages device tensors through host copies
  (`t.to("cpu")`) and its load path scatters CPU staging buffers into the
  paged tensors with cross-device `copy_` — sound in principle, proven on
  the CPU backend, but this rig's first green run IS the GPU validation.
  Every failure mode it could produce (no stores, no hits) is caught by the
  put-receipt and per-rep hit gates, not papered over.
- **The warm arm's timing includes the connector's per-block host→device
  copies** (one small blocking `copy_` per layer per block on the load
  path). That inflates warm TTFT — i.e. it UNDERSTATES kvblockd, which is
  the honest direction. Batching those copies is a known optimization, not
  a correction.
- **`tc` shaping** almost certainly unavailable (no NET_ADMIN): expected and
  disclosed, the run is unshaped loopback. Emulated-link points (25/50 Gbps)
  stay with the AWS rig plan (`bench/rigs/aws-gpu/`).
- **Gated Llama**: only after access is granted; Qwen is the default.
- The image is pinned by tag + recorded digest, but HF pulls by tag.
