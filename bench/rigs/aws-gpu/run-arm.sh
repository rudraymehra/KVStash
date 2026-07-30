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

export AWS_PROFILE="${AWS_PROFILE:-kvbench}"
ARM="${1:?usage: run-arm.sh <arm> <runtag>}"
TAG="${2:?usage: run-arm.sh <arm> <runtag>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="${STATE_DIR:-$HOME/kvbench-dday}"
IP=$(cat "$STATE_DIR/gpu-ip") || die "no gpu-ip — provision first"
SSH_KEY="${SSH_KEY:-$STATE_DIR/kvbench.pem}"
SSH=(ssh -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ServerAliveCountMax=10 "ubuntu@$IP")
IMG="vllm/vllm-openai@sha256:f0b9a0dc75a9fca3b6811e3279367b2d6a448055a000bfd13859587d74cef268"
DEST="$STATE_DIR/results/$ARM-$TAG"
mkdir -p "$DEST"

# Compose docker -e flags from the arm table (KEY=VALUE per line, spaces ok).
# arms.sh exit status is checked EXPLICITLY: process-substitution failures
# escape set -e, and a typo'd arm silently launching a full default job
# (stamped hf-jobs-*) from this box would be spend + falsified provenance.
# nic arms: default the store ip from the session state so a forgotten
# export refuses in arms.sh rather than dialing a host named "unset".
if [[ "$ARM" == nic-* && -z "${STORE_PRIVATE_IP:-}" && -s "$STATE_DIR/store-private-ip" ]]; then
  export STORE_PRIVATE_IP; STORE_PRIVATE_IP=$(cat "$STATE_DIR/store-private-ip")
fi
ENV_LINES=$(bash "$HERE/arms.sh" "$ARM") || die "arms.sh rejected arm '$ARM'"
[[ -n "$ENV_LINES" ]] || die "arms.sh emitted nothing for '$ARM'"
ENV_ARGS=()
while IFS= read -r kv; do
  [[ -n "$kv" ]] && ENV_ARGS+=(-e "$kv")
done <<< "$ENV_LINES"
IS_BASELINE=0; printf '%s\n' "${ENV_ARGS[@]}" | grep -q 'BASELINE_ONLY=1' && IS_BASELINE=1
IS_FP8=0;      printf '%s\n' "${ENV_ARGS[@]}" | grep -q 'KV_CACHE_DTYPE=fp8' && IS_FP8=1

# GIT_REF: belt-and-braces provenance — job.sh prefers `git rev-parse` on the
# mounted clone and falls back to this if the image lacks git. The laptop
# HEAD is only the DRY-RUN preview value; live runs re-resolve GIT_REF from
# the NODE's clone below (that is the code that actually executes, and the
# in-container git path can silently fail on root-vs-ubuntu ownership).
GIT_REF=$(git -C "$HERE/../../.." rev-parse --short=12 HEAD 2>/dev/null || echo unknown)
compose_run_cmd() {
  RUN_CMD=(sudo docker run --rm --gpus all --ipc=host --network host --entrypoint bash
    -v /opt/dlami/nvme:/nvme -v /opt/dlami/nvme/work/KVStash:/work/kvstash
    -e WORK=/nvme/work/ttft-"$ARM-$TAG" -e GIT_REF="$GIT_REF" "${ENV_ARGS[@]}"
    "$IMG" /work/kvstash/bench/rigs/hf-gpu/job.sh)
}
compose_run_cmd

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'DRY-RUN env for %s:\n' "$ARM"; bash "$HERE/arms.sh" "$ARM" | sed 's/^/  /'
  printf 'DRY-RUN>'; printf ' %q' "${RUN_CMD[@]}"; printf '\n'; exit 0
fi

# ---- pre-spend preflights (live node required; DRY_RUN exits above) ---------
env_val() { printf '%s\n' "$ENV_LINES" | sed -n "s/^$1=//p" | head -1; }

# 0) Provenance from the code that RUNS: resolve sha + dirtiness on the node's
#    clone (host-side git works as ubuntu; the in-container check can fail the
#    dubious-ownership test as root and silently fall back). A dirty node
#    clone is stamped, never hidden — session 1 published rows whose
#    annotation strings existed in no committed tree.
NODE_GIT="$("${SSH[@]}" 'cd /opt/dlami/nvme/work/KVStash 2>/dev/null && printf "%s %s" "$(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)" "$(git status --porcelain 2>/dev/null | head -1)"' 2>/dev/null || true)"
if [[ -n "$NODE_GIT" ]]; then
  NODE_SHA="${NODE_GIT%% *}"
  [[ "$NODE_GIT" == *" "?* ]] && NODE_SHA="$NODE_SHA-dirty"
  if [[ "$NODE_SHA" != "$GIT_REF" ]]; then
    log "node clone is $NODE_SHA (laptop $GIT_REF) — stamping the node's state"
  fi
  GIT_REF="$NODE_SHA"
  compose_run_cmd
fi

# 1) Instance-type truth: the arm's GPU_ANNOT claims a box ("ec2 <type> ...");
#    the box's own IMDS answer must agree, or the stamp publishes a lie
#    (session 1 provisioned a g5 when g6e was capacity-refused region-wide —
#    the annot would have kept saying g6e).
ANNOT="$(env_val GPU_ANNOT)"
if [[ "$ANNOT" == ec2\ * ]]; then
  CLAIMED_TYPE="$(printf '%s' "$ANNOT" | awk '{print $2}')"
  ACTUAL_TYPE="$("${SSH[@]}" 'TOK=$(curl -sX PUT --max-time 2 http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 60" 2>/dev/null); curl -sf --max-time 2 ${TOK:+-H "X-aws-ec2-metadata-token: $TOK"} http://169.254.169.254/latest/meta-data/instance-type' 2>/dev/null || true)"
  [[ -n "$ACTUAL_TYPE" ]] || die "could not read the GPU node's instance type from IMDS — refusing to run an arm whose provenance stamp cannot be checked"
  [[ "$ACTUAL_TYPE" == "$CLAIMED_TYPE" ]] \
    || die "GPU_ANNOT claims $CLAIMED_TYPE but the box is $ACTUAL_TYPE — a run now would stamp falsified provenance (fix the arm table or provision the right box)"
fi

# 2) EXTERNAL_DAEMON arms: the store must be reachable, EMPTY (fresh-daemon-
#    per-arm invariant: S3-FIFO residue from a previous arm can evict this
#    arm's own populate and void it at full cost — the trap stays hidden on
#    small arms and fires on the expensive one), on the link state the arm
#    stamps, and big enough for the workload.
EXT_METRICS="$(env_val EXTERNAL_METRICS)"
if [[ -n "$EXT_METRICS" ]]; then
  # Fetch first, THEN parse: piping into awk would let its END block print 0
  # on empty input, turning "store down" into "store empty" — the exact
  # inversion this gate exists to refuse. ssh propagates curl's exit status.
  METRICS_BODY="$("${SSH[@]}" "curl -sf --max-time 5 http://$EXT_METRICS/metrics" 2>/dev/null)" \
    || die "store metrics at $EXT_METRICS unreachable from the GPU node — is the store up?"
  BLOCKS_NOW="$(printf '%s\n' "$METRICS_BODY" | awk '/^kvb_blocks([{ ])/ {s+=$NF} END {printf "%d", s}')"
  [[ "$BLOCKS_NOW" == "0" ]] \
    || die "store is NOT empty (kvb_blocks=$BLOCKS_NOW) — run 'store-node.sh restart' first; every arm pre-registered a fresh daemon"
  # tc label vs machine state: STORE_TC_GBIT is operator-typed and lands in
  # the published TC_LINK stamp; shape/unshape record the actual qdisc state.
  TC_STATE_FILE="$STATE_DIR/store-tc-state"
  [[ -s "$TC_STATE_FILE" ]] || die "no $TC_STATE_FILE — run store-node.sh up/shape/unshape (it records the machine tc state this gate checks)"
  TC_STATE="$(cat "$TC_STATE_FILE")"
  [[ "${STORE_TC_GBIT:-}" == "$TC_STATE" ]] \
    || die "STORE_TC_GBIT='${STORE_TC_GBIT:-}' disagrees with the store's recorded tc state '$TC_STATE' — the TC_LINK stamp would lie"
  # Arena fit: external mode skips job.sh's derived-arena check entirely, so
  # an oversized arm otherwise burns a full populate before the hit gate
  # trips. Need = sweep pairs + the equivalence prompts, with job.sh's own
  # 1.15 headroom convention.
  ARENA_FILE="$STATE_DIR/store-arena-bytes"
  [[ -s "$ARENA_FILE" ]] || die "no $ARENA_FILE — store-node.sh up/restart records the arena size this gate checks"
  ARENA_B="$(cat "$ARENA_FILE")"
  KVB_TOK="$(env_val KV_BYTES_PER_TOKEN)"; REPS_A="$(env_val REPS)"; WARM_A="$(env_val WARMUP)"
  SUM_TOK=0; IFS=',' read -ra _LS <<< "$(env_val LENGTHS)"
  for _l in "${_LS[@]}"; do SUM_TOK=$((SUM_TOK + _l)); done
  NEED_B=$(( SUM_TOK * (REPS_A + WARM_A) * KVB_TOK * 115 / 100 + 8 * 320 * KVB_TOK ))
  (( NEED_B <= ARENA_B * 95 / 100 )) \
    || die "workload needs ~$((NEED_B / 1073741824)) GiB vs store arena $((ARENA_B / 1073741824)) GiB (95% cap) — the populate would evict itself; shrink the arm or grow STORE_ARENA_BYTES"
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
scp -q -i "$SSH_KEY" -r "ubuntu@$IP:/opt/dlami/nvme/work/ttft-$ARM-$TAG/results" "$DEST/" 2>/dev/null || true
VLLM_LOGS_OK=1
scp -q -i "$SSH_KEY" "ubuntu@$IP:/opt/dlami/nvme/work/ttft-$ARM-$TAG/vllm-*.log" "$DEST/" 2>/dev/null || VLLM_LOGS_OK=0
grep '^CHART2JSONL ' "$DEST/job.log" | sed 's/^CHART2JSONL //' > "$STATE_DIR/results/$ARM-$TAG.jsonl" || true
N_REC=$(wc -l < "$STATE_DIR/results/$ARM-$TAG.jsonl" 2>/dev/null || echo 0)
# Mechanism/rider evidence banks like everything else (the selftest's stub
# lines are pre-renamed by job.sh and cannot match these anchors).
grep '^CALIBJSONL ' "$DEST/job.log" | sed 's/^CALIBJSONL //' \
  > "$STATE_DIR/results/$ARM-$TAG-calib.jsonl" 2>/dev/null || true
grep '^PRESCREENJSONL ' "$DEST/job.log" | sed 's/^PRESCREENJSONL //' \
  > "$STATE_DIR/results/$ARM-$TAG-prescreen.jsonl" 2>/dev/null || true
find "$STATE_DIR/results" -maxdepth 1 -name "$ARM-$TAG-*.jsonl" -size 0 -delete 2>/dev/null || true

# Budget ledger line (console lags a day; wall-clock x rate is the truth).
# Rate follows the box actually provisioned, not an assumed type.
case "$(cat "$STATE_DIR/gpu-type" 2>/dev/null)" in
  g5.2xlarge)  RATE=1.212 ;;
  g6e.xlarge)  RATE=1.861 ;;
  *)           RATE=2.242 ;;
esac
printf '%s,%s,%s,%d,%.2f\n' "$ARM" "$TAG" "$(date -u +%FT%TZ)" "$WALL_MIN" \
  "$(echo "$WALL_MIN * $RATE / 60" | bc -l)" >> "$STATE_DIR/ledger.csv"

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
# Two-node arms: the run must PROVE it entered external-daemon mode — a
# stale job.sh on the node would silently measure loopback under a NIC stamp.
if printf '%s\n' "${ENV_ARGS[@]}" | grep -q 'EXTERNAL_DAEMON='; then
  grep -q 'using EXTERNAL kvblockd at' "$DEST/job.log" \
    || gate "nic arm never entered external-daemon mode — loopback measured, run voided"
  grep -q 'starting kvblockd (DRAM arena' "$DEST/job.log" \
    && gate "nic arm started a LOCAL daemon — loopback measured, run voided"
fi
if [[ $IS_FP8 -eq 1 && $IS_BASELINE -eq 0 ]]; then
  grep -qi 'flashinfer' "$DEST"/vllm-*.log 2>/dev/null || gate "fp8 arm without a FlashInfer boot line — backend gate"
elif [[ $IS_FP8 -eq 1 ]]; then
  # An fp8 BASELINE that silently lost FlashInfer inflates the denominator
  # and every multiple built on it; same image + same flags make it unlikely,
  # so this warns rather than gates (the arm side stays a hard gate).
  grep -qi 'flashinfer' "$DEST"/vllm-*.log 2>/dev/null \
    || log "WARN: fp8 BASELINE without a FlashInfer boot line — check the backend before publishing multiples against this denominator"
fi

log "$ARM $TAG: rc=$RC wall=${WALL_MIN}min records=$N_REC gates=$([[ $GATE_FAIL == 0 ]] && echo GREEN || echo FAILED)"
exit $GATE_FAIL
