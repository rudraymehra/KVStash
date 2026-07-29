#!/usr/bin/env bash
# kvbench arm table — every pre-registered cell of the L40S session as an
# env emitter. `arm_env <name>` prints KEY=VALUE lines consumed by
# run-arm.sh. The table (predictions, REPS caps, fallback ladders) is
# pre-registered; engine args are never tuned mid-session.
#
# Session-level choices frozen by the calibration block and passed via env:
#   MNBT       — max_num_batched_tokens winner of the a0c sweep (default 16384;
#                chosen to MINIMIZE THE BASELINE's TTFT, then frozen in both
#                arms of every run — the inverse of the vendor move)
#   KVB_FANOUT — get_fanout winner of the a0d A/B (default 4)
set -euo pipefail

QWEN=/nvme/hf/qwen1m
LLAMA=/nvme/hf/llama31
MNBT="${MNBT:-16384}"
KVB_FANOUT="${KVB_FANOUT:-4}"

# store queue bytes = max(4 GiB, 3 x MNBT x KV bytes/token): one chunked-
# prefill step stages MNBT*bytes in a burst and the write-behind queue must
# absorb three steps or the populate tail-skips (gate: dropped=0 failed=0).
queue_bytes() { # $1 = kv bytes/token
  local need=$((3 * MNBT * $1)) four=$((4 * 1073741824))
  echo $(( need > four ? need : four ))
}

common() { # $1 model path, $2 kv bytes/token, $3 L
  cat <<EOF
MODEL=$1
KV_BYTES_PER_TOKEN=$2
LENGTHS=$3
GEN_TOKENS=16
MAX_NUM_BATCHED_TOKENS=$MNBT
RIG=ec2-g6e-l40s
GPU_ANNOT=ec2 g6e.2xlarge
KVBD_STREAMS=8
KVBD_STORE_DRAIN_WORKERS=4
KVBD_STORE_FLUSH_TIMEOUT_S=120
KVBD_LOAD_DEADLINE_S=120
KVBD_EXISTS_TIMEOUT_S=2.0
KVBD_GET_FANOUT=$KVB_FANOUT
KVBD_STORE_QUEUE_BYTES=$(queue_bytes "$2")
REQUEST_TIMEOUT=$(( $3 >= 196608 ? 900 : 600 ))
EOF
}

arm_env() {
  case "$1" in
    # ---- calibration block (runs first; re-anchors predictions) ----
    a0b)  common "$QWEN" 57344 16384,32768; echo "REPS=3"; echo "WARMUP=1"; echo "GPU_MEM_UTIL=0.90" ;;
    a0c-*) # baseline-only chunk-size fairness sweep at the top point
          local m="${1#a0c-}"
          common "$QWEN" 57344 262144 | sed "s/^MAX_NUM_BATCHED_TOKENS=.*/MAX_NUM_BATCHED_TOKENS=$m/"
          echo "BASELINE_ONLY=1"; echo "REPS=2"; echo "WARMUP=1"; echo "GPU_MEM_UTIL=0.90" ;;
    a0d)  common "$QWEN" 57344 16384 | sed 's/^KVBD_GET_FANOUT=.*/KVBD_GET_FANOUT=8/;s/^KVBD_STREAMS=.*/KVBD_STREAMS=8/'
          echo "REPS=3"; echo "WARMUP=1"; echo "GPU_MEM_UTIL=0.90" ;;
    a0e)  common "$QWEN" 28672 16384; echo "FP8_PREFLIGHT=fp8_e4m3"; echo "GPU_MEM_UTIL=0.90" ;;
    a0f)  common "$QWEN" 28672 131072; echo "KV_CACHE_DTYPE=fp8_e4m3"; echo "REPS=1"; echo "WARMUP=1"; echo "GPU_MEM_UTIL=0.90" ;;
    # ---- baselines (denominator series; NO connector; identical flags) ----
    base-qwen-bf16) common "$QWEN" 57344 98304,131072,163840,196608,262144
          echo "BASELINE_ONLY=1"; echo "REPS=3"; echo "WARMUP=1"; echo "GPU_MEM_UTIL=0.90" ;;
    base-qwen-fp8)  common "$QWEN" 28672 98304,131072,163840,196608,262144
          echo "BASELINE_ONLY=1"; echo "KV_CACHE_DTYPE=fp8_e4m3"; echo "REPS=3"; echo "WARMUP=1"; echo "GPU_MEM_UTIL=0.90" ;;
    base-llama)     common "$LLAMA" 131072 98304,129024
          echo "BASELINE_ONLY=1"; echo "REPS=3"; echo "WARMUP=1"; echo "GPU_MEM_UTIL=0.92" ;;
    # ---- the pre-registered arm table (numbers = runbook) ----
    arm1)  common "$QWEN" 28672 98304;  echo "KV_CACHE_DTYPE=fp8_e4m3"; echo "REPS=3"; echo "WARMUP=1"; echo "GPU_MEM_UTIL=0.90" ;;
    arm2)  common "$QWEN" 28672 131072; echo "KV_CACHE_DTYPE=fp8_e4m3"; echo "REPS=3"; echo "WARMUP=1"; echo "GPU_MEM_UTIL=0.90" ;;
    arm3)  common "$QWEN" 57344 98304;  echo "REPS=3"; echo "WARMUP=1"; echo "GPU_MEM_UTIL=0.90" ;;
    arm4)  common "$QWEN" 57344 131072; echo "REPS=3"; echo "WARMUP=1"; echo "GPU_MEM_UTIL=0.90" ;;
    arm5)  common "$QWEN" 28672 163840; echo "KV_CACHE_DTYPE=fp8_e4m3"; echo "REPS=3"; echo "WARMUP=1"; echo "GPU_MEM_UTIL=0.90" ;;
    arm6)  common "$QWEN" 57344 163840; echo "REPS=3"; echo "WARMUP=1"; echo "GPU_MEM_UTIL=0.90" ;;
    arm7)  common "$QWEN" 28672 196608; echo "KV_CACHE_DTYPE=fp8_e4m3"; echo "REPS=3"; echo "WARMUP=1"; echo "GPU_MEM_UTIL=0.90" ;;
    arm8)  common "$QWEN" 57344 196608; echo "REPS=3"; echo "WARMUP=1"; echo "GPU_MEM_UTIL=0.90" ;;
    arm9)  common "$QWEN" 28672 262144; echo "KV_CACHE_DTYPE=fp8_e4m3"; echo "REPS=3"; echo "WARMUP=1"; echo "GPU_MEM_UTIL=0.90" ;;
    # arm10 fallback ladder (pre-registered): 262144 -> 229376 -> 196608.
    # bf16 arena caps pairs at 3 for 256k (Appendix A) -> REPS=2+1.
    arm10) common "$QWEN" 57344 262144; echo "REPS=2"; echo "WARMUP=1"; echo "GPU_MEM_UTIL=0.90" ;;
    arm11) common "$LLAMA" 131072 98304; echo "REPS=2"; echo "WARMUP=1"; echo "GPU_MEM_UTIL=0.92" ;;
    # arm12 VRAM knife-edge, LAST; fallback 129024 -> 112640 -> 98304;
    # arena caps REPS at 1+1, n=3 via three fresh runs.
    arm12) common "$LLAMA" 131072 129024; echo "REPS=1"; echo "WARMUP=1"; echo "GPU_MEM_UTIL=0.92" ;;
    *) echo "unknown arm: $1" >&2; return 1 ;;
  esac
}

# Called directly: print the env for one arm.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then arm_env "${1:?usage: arms.sh <arm>}"; fi
