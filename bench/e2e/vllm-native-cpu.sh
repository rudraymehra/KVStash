#!/usr/bin/env bash
# vLLM NATIVE adapter CPU e2e — the Week-10 gate recipe, pattern-matched to
# the Week-5 LMCache leg (bench/e2e/cpu/* + .github/workflows/e2e-cpu.yml):
#
#   build kvblockd -> start daemon (DRAM-only) -> vllm serve opt-125m on the
#   CPU backend with KvblockdConnector -> verify.py (same 640-token prompt
#   twice at temperature=0: byte-identical completions + kvb_hits_total > 0)
#   -> RESTART vLLM (daemon stays up) -> verify.py --expect-hits (hits survive
#   the engine restart: property d).
#
# Runs on the Mac (darwin needs GLOO_SOCKET_IFNAME=lo0 and the OMP no-bind)
# and in CI (.github/workflows/vllm-native-cpu.yml calls the serve/verify
# phases of this script; install is a separate cached step there).
#
# Flags REQUIRED by the pinned vLLM v0.25 contract (UPSTREAM.lock):
#   --disable-hybrid-kv-cache-manager : the connector factory refuses non-HMA
#     connectors otherwise; KvblockdConnector v0.1 does not implement
#     SupportsHMA.
#   --no-enable-prefix-caching : round 2 must go through the connector, not
#     vLLM's local prefix cache — otherwise the hit assertion tests nothing.
#   PYTHONHASHSEED=0 : the adapter refuses to boot unpinned (determinism
#     footgun; the error names this fix).
#
# Usage:
#   bench/e2e/vllm-native-cpu.sh            # everything (local dev loop)
#   bench/e2e/vllm-native-cpu.sh install    # pip stack only
#   bench/e2e/vllm-native-cpu.sh serve      # daemon+vllm+verify (assumes install)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
CPU_DIR="$ROOT/bench/e2e/cpu"
source "$CPU_DIR/versions.env"

VLLM_PORT="${VLLM_PORT:-18000}"
KVBD_METRICS="127.0.0.1:9442"
PHASE="${1:-all}"

export PYTHONHASHSEED=0
export VLLM_CPU_KVCACHE_SPACE="${VLLM_CPU_KVCACHE_SPACE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
if [[ "$(uname)" == "Darwin" ]]; then
  export VLLM_CPU_OMP_THREADS_BIND="${VLLM_CPU_OMP_THREADS_BIND:-nobind}"
  export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo0}"
  export KMP_BLOCKTIME="${KMP_BLOCKTIME:-0}"
fi

install_stack() {
  echo "[install] CPU torch"
  pip install --index-url "$TORCH_INDEX" torch
  echo "[install] vllm==$VLLM_VERSION (wheel-first, CPU)"
  PIP_EXTRA_INDEX_URL="$TORCH_INDEX" pip install "vllm==$VLLM_VERSION"
  echo "[install] kvblockd + vllm_kvblockd (editable)"
  pip install -e "$ROOT/python/kvblockd" -e "$ROOT/python/vllm_kvblockd"
  python - <<'PY'
import vllm, kvblockd, vllm_kvblockd
print("imports OK:", vllm.__version__, vllm_kvblockd.__version__)
PY
}

DAEMON_PID="" VLLM_PID=""
cleanup() {
  [[ -n "$VLLM_PID" ]] && kill "$VLLM_PID" 2>/dev/null || true
  [[ -n "$DAEMON_PID" ]] && kill "$DAEMON_PID" 2>/dev/null || true
}

serve_vllm() {  # $1 = log file, $2 = prefix-caching flag
  vllm serve "$MODEL" --port "$VLLM_PORT" --dtype bfloat16 \
    --max-model-len 2048 --max-num-seqs 1 \
    "${2:---no-enable-prefix-caching}" \
    --disable-hybrid-kv-cache-manager \
    --kv-transfer-config "$(cat "$CPU_DIR/kv_transfer_vllm_kvblockd.json")" \
    > "$1" 2>&1 &
  VLLM_PID=$!
  bash "$ROOT/scripts/wait-http.sh" "http://127.0.0.1:$VLLM_PORT/health" 600
}

run_e2e() {
  trap cleanup EXIT
  echo "[e2e] build + start kvblockd (DRAM-only)"
  (cd "$ROOT" && go build -o bin/kvblockd ./cmd/kvblockd)
  (cd "$ROOT" && exec ./bin/kvblockd -config bench/e2e/cpu/kvblockd.yaml) > /tmp/kvbd-native.log 2>&1 &
  DAEMON_PID=$!
  bash "$ROOT/scripts/wait-healthz.sh" "$KVBD_METRICS" 30

  echo "[e2e] round A: cold engine — byte-identity + hits>0"
  serve_vllm /tmp/vllm-native.log
  python "$CPU_DIR/verify.py" --vllm "http://127.0.0.1:$VLLM_PORT" --metrics "http://$KVBD_METRICS" \
    --save-ref /tmp/vllm-native-ref.json

  echo "[e2e] round B: restart vLLM, daemon keeps the cache (property d)"
  kill "$VLLM_PID" 2>/dev/null || true
  VLLM_PID=""
  sleep 3
  serve_vllm /tmp/vllm-native2.log
  python "$CPU_DIR/verify.py" --vllm "http://127.0.0.1:$VLLM_PORT" --metrics "http://$KVBD_METRICS" --expect-hits

  echo "[e2e] round C: restart vLLM with prefix caching ENABLED — a prefix"
  echo "      request warms the LOCAL cache, then the full prompt takes the"
  echo "      mixed local+remote path (the tail range must load correctly)"
  kill "$VLLM_PID" 2>/dev/null || true
  VLLM_PID=""
  sleep 3
  serve_vllm /tmp/vllm-native3.log --enable-prefix-caching
  python "$CPU_DIR/verify.py" --vllm "http://127.0.0.1:$VLLM_PORT" --metrics "http://$KVBD_METRICS" \
    --prefix-mixed /tmp/vllm-native-ref.json

  echo "[e2e] PASS: vLLM native adapter CPU e2e green"
}

case "$PHASE" in
  install) install_stack ;;
  serve)   run_e2e ;;
  all)     install_stack; run_e2e ;;
  *) echo "usage: $0 [install|serve|all]" >&2; exit 2 ;;
esac
