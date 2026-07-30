#!/usr/bin/env bash
# Submit the Chart-2 TTFT job to Hugging Face Jobs (default a10g-large: 46 GB
# RAM — a10g-small's 15 GB got an earlier run OOM-killed; billed per minute).
# THIS SPENDS MONEY when run without DRY_RUN=1.
#
#   DRY_RUN=1 bench/rigs/hf-gpu/submit.sh     # print the exact command, run nothing
#   bench/rigs/hf-gpu/submit.sh               # submit (asks for confirmation)
#
# Knobs (env): MODEL, GIT_REF, LENGTHS, REPS, WARMUP, GEN_TOKENS,
# MAX_MODEL_LEN, GPU_MEM_UTIL, KVBD_ARENA_BYTES, CONNECTOR_STAGING_GB,
# KV_BYTES_PER_TOKEN, TIMEOUT, RESULTS_REPO, FLAVOR, HF_BIN, BASELINE_ONLY
# (1 = pure-recompute control run: no connector, no daemon, cold-only — the
# third chart series), JOB_NAME (job name shown by `hf jobs`, default
# chart2-ttft), KVB_SUBMIT_N_CONFIRMED (1 = skip the billing confirmation —
# set ONLY by submit-n.sh, per invocation, after it confirms the whole batch
# once; the deliberately unwieldy name keeps a stray `export ASSUME_YES=1`
# in someone's shell from silently green-lighting spend),
# KV_CACHE_DTYPE (vLLM --kv-cache-dtype; same dtype feeds BOTH arms of a run
# and stamps every record — the fp8 disclosure rule), FP8_PREFLIGHT (comma
# dtype list or 1: probe-only job, FP8PROBE verdict lines, exits before any
# measured run), EQUIV_N (token-identity prompts around the restart; job.sh
# defaults it to 8 on fp8 runs — the certification gate), FP8_CAMPAIGN
# (1 = ONE command submits the WHOLE certifiable fp8 story: the two-phase
# job — cold + warm arms, fp8 in both engine boots, token-identity hard
# gate — AND the pure-recompute baseline job under the SAME dtype, so all
# three chart arms share one kv_cache_dtype and the multiple measures the
# store, never the quantizer; each job runs its own fresh daemon — arms
# never share a store, and the fp8 keyspace forks from bf16 by fingerprint;
# a caller-set EQUIV_N goes to the two-phase job ONLY — the baseline control
# has no store to certify against — and an explicit EQUIV_N=0 is refused at
# submit time: it would strip the campaign's token-identity certification).
#
# The job container clones the PUBLIC repo tarball at GIT_REF — local
# uncommitted changes are NOT visible to the job; push first.
set -euo pipefail

HF_BIN="${HF_BIN:-$HOME/.kvb-hf/bin/hf}"
IMAGE="${IMAGE:-vllm/vllm-openai:v0.25.1}"   # amd64 digest sha256:f0b9a0dc75a9fca3b6811e3279367b2d6a448055a000bfd13859587d74cef268 (recorded 2026-07-26)
FLAVOR="${FLAVOR:-a10g-large}"               # 46 GB RAM; the derived kvblockd arena (~25 GiB at default
                                             # LENGTHS/REPS) plus vLLM does NOT fit a10g-small's 15 GB
TIMEOUT="${TIMEOUT:-2h}"
GIT_REF="${GIT_REF:-main}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
LENGTHS="${LENGTHS:-1024,4096,8192,16384}"   # job.sh derives max-model-len + arena from these
REPS="${REPS:-5}"
GEN_TOKENS="${GEN_TOKENS:-16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"           # optional; job.sh derives it from LENGTHS when unset
RESULTS_REPO="${RESULTS_REPO:-}"   # optional: HF dataset repo to receive the JSONL

[[ -x "$HF_BIN" ]] || { echo "hf CLI not found at $HF_BIN (set HF_BIN)" >&2; exit 1; }

# ---- FP8 campaign mode: one command -> two jobs, three same-dtype arms -------
FP8_CAMPAIGN="${FP8_CAMPAIGN:-0}"
if [[ "$FP8_CAMPAIGN" == "1" ]]; then
  if [[ "${BASELINE_ONLY:-0}" == "1" ]]; then
    echo "refusing: FP8_CAMPAIGN=1 submits its own baseline job — drop BASELINE_ONLY=1" >&2; exit 1
  fi
  if [[ -n "${FP8_PREFLIGHT:-}" ]]; then
    echo "refusing: FP8_PREFLIGHT is a probe job; run it (and read its FP8PROBE verdicts) BEFORE spending on a campaign" >&2; exit 1
  fi
  # e4m3 is the default on purpose: vLLM's published accuracy campaign backs
  # it. (The bare 'fp8' alias is fine too — job.sh normalizes the stamp.)
  KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}"
  case "$KV_CACHE_DTYPE" in
    fp8*) ;;
    *) echo "refusing: FP8_CAMPAIGN=1 with KV_CACHE_DTYPE='$KV_CACHE_DTYPE' (not an fp8 dtype)" >&2; exit 1;;
  esac
  # EQUIV_N=0 would silently strip the token-identity certification from the
  # campaign's fp8 two-phase job (job.sh defaults it to 8 there) — the whole
  # point of the campaign is the certifiable run, so refuse before billing.
  if [[ "${EQUIV_N:-}" == "0" ]]; then
    echo "refusing: FP8_CAMPAIGN=1 with EQUIV_N=0 strips the token-identity certification from the fp8 two-phase job — drop EQUIV_N (job.sh defaults it to 8) or set it >0" >&2; exit 1
  fi
fi

# ---- refuse incoherent knobs BEFORE billing starts --------------------------
# Run 3 passed LENGTHS=...,32000 while the job's MAX_MODEL_LEN default was
# 20480 — the 32k cell could never run. job.sh now derives max-model-len from
# LENGTHS; this mirror check catches an explicit-but-too-small override here,
# where it costs nothing.
MAX_LEN=0
IFS=',' read -ra _LENS <<< "$LENGTHS"
for l in "${_LENS[@]}"; do
  l="${l//[[:space:]]/}"
  [[ "$l" =~ ^[0-9]+$ ]] || { echo "refusing: LENGTHS entry '$l' is not a number (LENGTHS=$LENGTHS)" >&2; exit 1; }
  if (( l > MAX_LEN )); then MAX_LEN=$l; fi
done
(( MAX_LEN > 0 )) || { echo "refusing: LENGTHS is empty" >&2; exit 1; }
NEED_LEN=$((MAX_LEN + GEN_TOKENS + 384))
if [[ -n "$MAX_MODEL_LEN" ]] && (( MAX_MODEL_LEN < NEED_LEN )); then
  echo "refusing: MAX_MODEL_LEN=$MAX_MODEL_LEN < $NEED_LEN needed for LENGTHS=$LENGTHS (+GEN_TOKENS $GEN_TOKENS +384 headroom)." >&2
  echo "either drop the largest length or raise/unset MAX_MODEL_LEN (job.sh derives it when unset)." >&2
  exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
if [[ -n "$(git -C "$ROOT" status --porcelain -- bench/rigs/hf-gpu python bench/e2e scripts 2>/dev/null)" ]]; then
  echo "WARNING: uncommitted changes under bench/rigs/hf-gpu, python/, bench/e2e or scripts/." >&2
  echo "         The job runs from the GitHub tarball at '$GIT_REF' and will NOT see them." >&2
fi

# In-container bootstrap: fetch the public repo tarball at GIT_REF (no git
# dependency in the image), then hand off to the committed entrypoint.
BOOTSTRAP='set -euo pipefail
mkdir -p /work && cd /work
curl -fsSL "https://codeload.github.com/rudraymehra/KVStash/tar.gz/${GIT_REF}" | tar -xz
mv KVStash-* kvstash
bash /work/kvstash/bench/rigs/hf-gpu/job.sh'

mk_cmd() {  # $1 = job name; reads the CURRENT env (incl. BASELINE_ONLY)
  CMD=("$HF_BIN" jobs run
    --name "$1"
    --flavor "$FLAVOR"
    --timeout "$TIMEOUT"
    --detach
    --secrets HF_TOKEN
    -e GIT_REF="$GIT_REF"
    -e MODEL="$MODEL"
    -e LENGTHS="$LENGTHS"
    -e REPS="$REPS"
    -e GEN_TOKENS="$GEN_TOKENS"
    -e FLAVOR="$FLAVOR")
  # optional knobs: forward only when the caller set them (job.sh has the defaults/derivations)
  local v
  for v in MAX_MODEL_LEN GPU_MEM_UTIL WARMUP KVBD_ARENA_BYTES CONNECTOR_STAGING_GB KVBD_STORE_QUEUE_BYTES KV_BYTES_PER_TOKEN RESULTS_REPO BASELINE_ONLY HF_OVERRIDES KV_CACHE_DTYPE FP8_PREFLIGHT EQUIV_N EQUIV_PROMPT_SET EQUIV_MARGIN_FLOOR EQUIV_GEN_TOKENS \
           STRIP_DCA MAX_NUM_BATCHED_TOKENS REQUEST_TIMEOUT RIG GPU_ANNOT KVBD_STREAMS \
           KVBD_STORE_DRAIN_WORKERS KVBD_STORE_FLUSH_TIMEOUT_S KVBD_LOAD_DEADLINE_S \
           KVBD_EXISTS_TIMEOUT_S KVBD_GET_FANOUT; do
    if [[ -n "${!v:-}" ]]; then CMD+=(-e "$v=${!v}"); fi
  done
  CMD+=("$IMAGE" /bin/bash -c "$BOOTSTRAP")
}

# job specs: "<name>|<BASELINE_ONLY override or empty>". The single-job path
# is byte-for-byte the old behavior; the campaign adds the same-dtype
# baseline control as its second submission (two-phase first).
JOB_SPECS=("${JOB_NAME:-chart2-ttft}|")
if [[ "$FP8_CAMPAIGN" == "1" ]]; then
  CAMPAIGN_TAG="${JOB_NAME:-chart2-fp8}"
  JOB_SPECS=("${CAMPAIGN_TAG}-twophase|" "${CAMPAIGN_TAG}-baseline|1")
fi

# Per-spec env is set UNCONDITIONALLY from a snapshot of the caller's values.
# Both loops below call this: a set-only guard here once let the preview
# loop's baseline spec leak BASELINE_ONLY=1 into the submission loop's
# two-phase job — both billed jobs ran as pure-recompute baselines, and the
# echoed preview (printed before the pollution) could not show it.
ORIG_BASELINE_ONLY="${BASELINE_ONLY:-}"
ORIG_EQUIV_N="${EQUIV_N:-}"
apply_spec() {  # $1 = "<name>|<BASELINE_ONLY override>"; sets name, BASELINE_ONLY, EQUIV_N
  name="${1%%|*}"
  local b="${1#*|}"
  BASELINE_ONLY="${b:-$ORIG_BASELINE_ONLY}"
  if [[ "$FP8_CAMPAIGN" == "1" && "$BASELINE_ONLY" == "1" ]]; then
    # The campaign's baseline control never gets EQUIV_N: job.sh refuses
    # EQUIV_N>0 + BASELINE_ONLY=1 inside the billed container (no store to
    # prove token identity against).
    EQUIV_N=""
  else
    EQUIV_N="$ORIG_EQUIV_N"
  fi
}

echo "== hf jobs command(s) =="
for spec in "${JOB_SPECS[@]}"; do
  apply_spec "$spec"
  mk_cmd "$name"
  printf ' %q' "${CMD[@]}"; echo; echo
done

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1 — nothing submitted."
  exit 0
fi

echo "flavor $FLAVOR is billed per minute (a10g-small was \$1.00/hr; a10g-large costs more — check current HF Jobs pricing)."
echo "expected run <1h per job (two vLLM boots: populate, then a fresh measure engine); timeout $TIMEOUT caps the spend."
if [[ "$FP8_CAMPAIGN" == "1" ]]; then
  echo "FP8_CAMPAIGN=1: TWO jobs (two-phase + baseline control), both kv_cache_dtype=$KV_CACHE_DTYPE — ~2x the single-job cost."
fi
if [[ "${KVB_SUBMIT_N_CONFIRMED:-0}" == "1" ]]; then
  echo "KVB_SUBMIT_N_CONFIRMED=1 — confirmation was given upstream (submit-n.sh batch)."
else
  read -r -p "Submit ${#JOB_SPECS[@]} job(s) and start billing? [y/N] " ans
  [[ "$ans" == "y" || "$ans" == "Y" ]] || { echo "aborted."; exit 1; }
fi

JOB_IDS=()
for spec in "${JOB_SPECS[@]}"; do
  apply_spec "$spec"
  mk_cmd "$name"
  JOB_OUT="$("${CMD[@]}")"
  echo "$JOB_OUT"
  JOB_ID="$(printf '%s\n' "$JOB_OUT" | tail -n1 | awk '{print $NF}')"  # --detach prints the Job ID
  echo "submitted: $JOB_ID"
  JOB_IDS+=("$JOB_ID")
done
echo
for JOB_ID in "${JOB_IDS[@]}"; do
  echo "follow logs:     $HF_BIN jobs logs -f $JOB_ID"
done
echo "check status:    $HF_BIN jobs inspect <job_id>"
echo "cancel:          $HF_BIN jobs cancel <job_id>"
# The sed requires a '{' after the marker (drops the job's own hint line and
# any prose mentioning the marker); selftest stub records are already renamed
# SELFTESTJSONL by job.sh.
echo "fetch results:   mkdir -p bench/results/rig-e && $HF_BIN jobs logs ${JOB_IDS[0]} | sed -n 's/^.*CHART2JSONL \\({.*\\)$/\\1/p' > bench/results/rig-e/chart2-ttft.jsonl"
if [[ "$FP8_CAMPAIGN" == "1" ]]; then
  echo "                 (repeat for the baseline job ${JOB_IDS[1]:-<baseline_id>} into its own file; plot both together — same dtype, so the mixed-dtype refusal stays quiet)"
  echo "token identity:  $HF_BIN jobs logs ${JOB_IDS[0]} | sed -n 's/^.*EQUIVJSONL \\({.*\\)$/\\1/p' > bench/results/rig-e/chart2-fp8-equivalence.jsonl"
fi
echo "render chart:    python3 bench/report/plot.py chart2 --in bench/results/rig-e/chart2-ttft.jsonl --out chart2.png"
