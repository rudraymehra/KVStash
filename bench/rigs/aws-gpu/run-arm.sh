#!/usr/bin/env bash
# kvbench arm runner (laptop-side). Executes ONE pre-registered arm inside
# the pinned engine container on the GPU node, pulls every artifact back
# IMMEDIATELY (a terminate must never lose more than the in-flight rep),
# and runs the honesty-gate greps locally. Exit 0 = arm green.
#
#   bench/rigs/aws-gpu/run-arm.sh arm9 run1
#   DRY_RUN=1 bench/rigs/aws-gpu/run-arm.sh arm9 run1   # print, execute nothing
#
# Results land in $STATE_DIR/results/<arm>-<runtag>/ and the JSONL is
# additionally extracted to <arm>-<runtag>.jsonl for aggregate.py.
set -euo pipefail
log() { printf '[arm %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
die() { log "FATAL: $*"; exit 1; }

ARM="${1:?usage: run-arm.sh <arm> <runtag>}"
TAG="${2:?usage: run-arm.sh <arm> <runtag>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="${STATE_DIR:-$HOME/kvbench-dday}"
IP=$(cat "$STATE_DIR/gpu-ip") || die "no gpu-ip — provision first"
SSH=(ssh -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ServerAliveCountMax=10 "ubuntu@$IP")
IMG="vllm/vllm-openai@sha256:f0b9a0dc75a9fca3b6811e3279367b2d6a448055a000bfd13859587d74cef268"
DEST="$STATE_DIR/results/$ARM-$TAG"
mkdir -p "$DEST"

# Compose docker -e flags from the arm table (KEY=VALUE per line, spaces ok).
# arms.sh exit status is checked EXPLICITLY: process-substitution failures
# escape set -e, and a typo'd arm silently launching a full default job
# (stamped hf-jobs-*) from this box would be spend + falsified provenance.
ENV_LINES=$(bash "$HERE/arms.sh" "$ARM") || die "arms.sh rejected arm '$ARM'"
[[ -n "$ENV_LINES" ]] || die "arms.sh emitted nothing for '$ARM'"
ENV_ARGS=()
while IFS= read -r kv; do
  [[ -n "$kv" ]] && ENV_ARGS+=(-e "$kv")
done <<< "$ENV_LINES"
IS_BASELINE=0; printf '%s\n' "${ENV_ARGS[@]}" | grep -q 'BASELINE_ONLY=1' && IS_BASELINE=1
IS_FP8=0;      printf '%s\n' "${ENV_ARGS[@]}" | grep -q 'KV_CACHE_DTYPE=fp8' && IS_FP8=1

# GIT_REF: belt-and-braces provenance — job.sh prefers `git rev-parse` on the
# mounted clone and falls back to this if the image lacks git.
GIT_REF=$(git -C "$HERE/../../.." rev-parse --short=12 HEAD 2>/dev/null || echo unknown)
RUN_CMD=(sudo docker run --rm --gpus all --ipc=host --network host --entrypoint bash
  -v /opt/dlami/nvme:/nvme -v /opt/dlami/nvme/work/KVStash:/work/kvstash
  -e WORK=/nvme/work/ttft-"$ARM-$TAG" -e GIT_REF="$GIT_REF" "${ENV_ARGS[@]}"
  "$IMG" /work/kvstash/bench/rigs/hf-gpu/job.sh)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'DRY-RUN env for %s:\n' "$ARM"; bash "$HERE/arms.sh" "$ARM" | sed 's/^/  /'
  printf 'DRY-RUN>'; printf ' %q' "${RUN_CMD[@]}"; printf '\n'; exit 0
fi

T0=$(date +%s)
log "running $ARM ($TAG) on $IP — output streams to $DEST/job.log"
set +e
"${SSH[@]}" "$(printf '%q ' "${RUN_CMD[@]}")" > "$DEST/job.log" 2>&1
RC=$?
set -e
T1=$(date +%s); WALL_MIN=$(( (T1 - T0) / 60 ))

# Pull EVERYTHING back before judging anything. The engine logs are the
# gates' EVIDENCE — for a connector arm their absence is itself a gate
# failure (a missing log must never read as "no bad strings").
scp -q -r "ubuntu@$IP:/opt/dlami/nvme/work/ttft-$ARM-$TAG/results" "$DEST/" 2>/dev/null || true
VLLM_LOGS_OK=1
scp -q "ubuntu@$IP:/opt/dlami/nvme/work/ttft-$ARM-$TAG/vllm-*.log" "$DEST/" 2>/dev/null || VLLM_LOGS_OK=0
grep '^CHART2JSONL ' "$DEST/job.log" | sed 's/^CHART2JSONL //' > "$STATE_DIR/results/$ARM-$TAG.jsonl" || true
N_REC=$(wc -l < "$STATE_DIR/results/$ARM-$TAG.jsonl" 2>/dev/null || echo 0)

# Budget ledger line (console lags a day; wall-clock x rate is the truth).
printf '%s,%s,%s,%d,%.2f\n' "$ARM" "$TAG" "$(date -u +%FT%TZ)" "$WALL_MIN" \
  "$(echo "$WALL_MIN * 2.242 / 60" | bc -l)" >> "$STATE_DIR/ledger.csv"

# ---- honesty gates (runbook §5), judged from the pulled logs ----------------
GATE_FAIL=0
gate() { log "GATE FAIL: $*"; GATE_FAIL=1; }
if [[ $RC -ne 0 ]]; then gate "job exited rc=$RC (see $DEST/job.log)"; fi
# Engine-log strings live ONLY in the pulled vllm-*.log files (job.sh
# redirects all engine output there); job.log carries the driver receipts.
if [[ $IS_BASELINE -eq 0 ]]; then
  [[ $VLLM_LOGS_OK -eq 1 && -n $(ls "$DEST"/vllm-*.log 2>/dev/null) ]] \
    || gate "engine logs missing — gates have no evidence, arm cannot be judged GREEN"
  for BAD in 'load deadline exceeded' 'GET deadline expired' 'GET shard failed' 'latched OFF' 'switched from'; do
    grep -q "$BAD" "$DEST"/vllm-*.log 2>/dev/null && gate "zero-occurrence string present: '$BAD' (rep voided)"
  done
  # Positive receipt: the driver prints this to stdout for every healthy run.
  grep -q 'connector load path: pipelined-slab' "$DEST/job.log" \
    || gate "missing pipelined-slab path receipt in driver output"
  grep -q 'dropped=0 failed=0' "$DEST/job.log" \
    || log "WARN: store-queue shutdown line absent (known teardown race) — check populate gate output above"
else
  grep -q 'Creating v1 connector' "$DEST"/vllm-*.log 2>/dev/null && gate "BASELINE run booted a connector — run voided"
fi
if [[ $IS_FP8 -eq 1 && $IS_BASELINE -eq 0 ]]; then
  grep -qi 'flashinfer' "$DEST"/vllm-*.log 2>/dev/null || gate "fp8 arm without a FlashInfer boot line — backend gate"
fi

log "$ARM $TAG: rc=$RC wall=${WALL_MIN}min records=$N_REC gates=$([[ $GATE_FAIL == 0 ]] && echo GREEN || echo FAILED)"
exit $GATE_FAIL
