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
# third chart series).
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

CMD=("$HF_BIN" jobs run
  --name chart2-ttft
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
for v in MAX_MODEL_LEN GPU_MEM_UTIL WARMUP KVBD_ARENA_BYTES CONNECTOR_STAGING_GB KV_BYTES_PER_TOKEN RESULTS_REPO BASELINE_ONLY HF_OVERRIDES; do
  if [[ -n "${!v:-}" ]]; then CMD+=(-e "$v=${!v}"); fi
done
CMD+=("$IMAGE" /bin/bash -c "$BOOTSTRAP")

echo "== hf jobs command =="
printf ' %q' "${CMD[@]}"; echo; echo

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1 — nothing submitted."
  exit 0
fi

echo "flavor $FLAVOR is billed per minute (a10g-small was \$1.00/hr; a10g-large costs more — check current HF Jobs pricing)."
echo "expected run <1h (two vLLM boots: populate, then a fresh measure engine); timeout $TIMEOUT caps the spend."
read -r -p "Submit and start billing? [y/N] " ans
[[ "$ans" == "y" || "$ans" == "Y" ]] || { echo "aborted."; exit 1; }

JOB_OUT="$("${CMD[@]}")"
echo "$JOB_OUT"
JOB_ID="$(printf '%s\n' "$JOB_OUT" | tail -n1 | awk '{print $NF}')"  # --detach prints the Job ID
echo "submitted: $JOB_ID"
echo
echo "follow logs:     $HF_BIN jobs logs -f $JOB_ID"
echo "check status:    $HF_BIN jobs inspect $JOB_ID"
echo "cancel:          $HF_BIN jobs cancel $JOB_ID"
# The sed requires a '{' after the marker (drops the job's own hint line and
# any prose mentioning the marker); selftest stub records are already renamed
# SELFTESTJSONL by job.sh.
echo "fetch results:   mkdir -p bench/results/rig-e && $HF_BIN jobs logs $JOB_ID | sed -n 's/^.*CHART2JSONL \\({.*\\)$/\\1/p' > bench/results/rig-e/chart2-ttft.jsonl"
echo "render chart:    python3 bench/report/plot.py chart2 --in bench/results/rig-e/chart2-ttft.jsonl --out chart2.png"
