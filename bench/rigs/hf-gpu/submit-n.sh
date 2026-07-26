#!/usr/bin/env bash
# Submit the SAME Chart-2 config N times (default 3) as INDEPENDENT HF Jobs —
# independent submissions mean independent engine boots, i.e. true runs for
# methodology rule 7 (median of >=3 with min/max), not reps inside one boot.
# THIS SPENDS N x THE SINGLE-JOB COST when run without DRY_RUN=1.
#
#   DRY_RUN=1 bench/rigs/hf-gpu/submit-n.sh          # print, submit nothing
#   N=3 TAG=chart2-ttft-llama16k \
#     MODEL=meta-llama/Llama-3.1-8B-Instruct bench/rigs/hf-gpu/submit-n.sh
#
# All submit.sh knobs (MODEL, LENGTHS, REPS, GIT_REF, BASELINE_ONLY, ...) pass
# through the environment unchanged. One confirmation covers the whole batch;
# each underlying submit.sh call then runs with KVB_SUBMIT_N_CONFIRMED=1 (set
# per invocation right here, never exported — submit.sh explains the name).
#
# Companion fetch loop — after the jobs finish, pull each job's CHART2JSONL
# lines into per-run results files:
#
#   bench/rigs/hf-gpu/submit-n.sh fetch <tag> <job_id_1> <job_id_2> ...
#     -> bench/results/rig-e/<tag>-run1.jsonl ... -runN.jsonl
#
# then aggregate across runs (10% spread gate, exits nonzero over tolerance):
#
#   python3 bench/report/aggregate.py bench/results/rig-e/<tag>-run*.jsonl
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
HF_BIN="${HF_BIN:-$HOME/.kvb-hf/bin/hf}"

# ---- fetch mode --------------------------------------------------------------
if [[ "${1:-}" == "fetch" ]]; then
  shift
  TAG="${1:-}"
  [[ -n "$TAG" && $# -ge 2 ]] || {
    echo "usage: $0 fetch <tag> <job_id> [<job_id> ...]" >&2; exit 2; }
  shift
  [[ -x "$HF_BIN" ]] || { echo "hf CLI not found at $HF_BIN (set HF_BIN)" >&2; exit 1; }
  mkdir -p "$ROOT/bench/results/rig-e"
  i=0 empty=0
  for id in "$@"; do
    i=$((i + 1))
    out="$ROOT/bench/results/rig-e/${TAG}-run${i}.jsonl"
    # Same extraction as submit.sh advertises: the sed requires a '{' after
    # the marker (drops hint lines); job.sh already renames selftest stub
    # records to SELFTESTJSONL so they can never land here.
    # A failed `hf logs` (expired job, network blip) must not abort the loop
    # under set -e/pipefail — warn, mark the batch incomplete, fetch the rest.
    if ! "$HF_BIN" jobs logs "$id" | sed -n 's/^.*CHART2JSONL \({.*\)$/\1/p' > "$out"; then
      echo "  WARN: 'hf jobs logs $id' failed — run $i not fetched" \
           "($HF_BIN jobs inspect $id); continuing with the remaining jobs" >&2
      rm -f "$out"
      empty=1
      continue
    fi
    n="$(wc -l < "$out" | tr -d ' ')"
    echo "run $i: job $id -> $out ($n records)"
    if [[ "$n" == "0" ]]; then
      echo "  WARN: no CHART2JSONL lines — job still running or failed" \
           "($HF_BIN jobs inspect $id); empty file left in place" >&2
      empty=1
    fi
  done
  echo
  echo "aggregate across the runs (median/min/max/spread per cell, 10% gate):"
  echo "  python3 bench/report/aggregate.py bench/results/rig-e/${TAG}-run*.jsonl"
  echo "render with whiskers:"
  echo "  python3 bench/report/plot.py chart2 --in bench/results/rig-e/${TAG}-run*.jsonl --out chart2.png"
  exit "$empty"
fi

# ---- submit mode -------------------------------------------------------------
N="${N:-3}"
TAG="${TAG:-chart2-ttft}"
[[ "$N" =~ ^[0-9]+$ && "$N" -ge 1 ]] || { echo "refusing: N='$N' is not a positive integer" >&2; exit 1; }

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1 — the single-job command below would be submitted $N times" \
       "(names ${TAG}-run1..${TAG}-run${N}):"
  JOB_NAME="${TAG}-run1" "$HERE/submit.sh"
  exit 0
fi

echo "About to submit $N INDEPENDENT jobs of the same config (N x per-minute billing)."
echo "tag=$TAG  model=${MODEL:-<submit.sh default>}  lengths=${LENGTHS:-<default>}  reps=${REPS:-<default>}"
read -r -p "Submit all $N and start billing? [y/N] " ans
[[ "$ans" == "y" || "$ans" == "Y" ]] || { echo "aborted."; exit 1; }

ids=()
for i in $(seq 1 "$N"); do
  echo "== submitting run $i/$N (${TAG}-run${i}) =="
  out="$(JOB_NAME="${TAG}-run${i}" KVB_SUBMIT_N_CONFIRMED=1 "$HERE/submit.sh")"
  printf '%s\n' "$out"
  id="$(printf '%s\n' "$out" | sed -n 's/^submitted: //p' | tail -n1)"
  [[ -n "$id" ]] || { echo "FATAL: could not parse job id from submit.sh output" >&2; exit 1; }
  ids+=("$id")
done

echo
echo "submitted ${#ids[@]} jobs: ${ids[*]}"
echo
echo "when they finish, fetch all runs:"
echo "  $0 fetch $TAG ${ids[*]}"
