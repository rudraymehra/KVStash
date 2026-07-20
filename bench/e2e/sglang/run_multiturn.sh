#!/usr/bin/env bash
# =============================================================================
# STATUS: NOT RUN — wk-12 SGLang spike DEFERRED before any GPU session
# (no GPU budget this sprint; docs/design/sglang-hicache-v1.1.md). The exact
# launch flags below were code-read against sglang v0.5.15.post1, not
# executed. Log every failure VERBATIM into NOTES.md.
# =============================================================================
# Drive the multi-turn benchmark against an SGLang server whose HiCache L3 is
# kvblockd (dynamic storage backend, zero-copy v1 interface).
#
# Success criteria (week-12 Day-3):
#   1. kvblockd /metrics shows PUTs during round 1, GETs (hits) on round 2+
#   2. get_stats() prefetch/backup rates nonzero (scheduler logs)
#   3. multi-turn outputs token-identical vs the no-L3 baseline run
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/versions.env"

KVB_PORT="${KVB_PORT:-9400}"
KVB_METRICS_PORT="${KVB_METRICS_PORT:-9401}"
KVB_TOKEN="${KVB_TOKEN:-sglang-e2e-token}"
SGL_PORT="${SGL_PORT:-30000}"
BASELINE="${BASELINE:-0}" # BASELINE=1 → no L3 (token-identity reference run)

EXTRA_CONFIG=$(cat <<EOF
{"backend_name":"kvblockd","module_path":"sglang_kvblockd.backend","class_name":"KvblockdHiCacheStorage","interface_v1":1,"endpoint":"kvblockd://127.0.0.1:${KVB_PORT}","namespace":"sglang-e2e","token":"${KVB_TOKEN}"}
EOF
)

LAUNCH=(python3 -m sglang.launch_server
  --model-path "$MODEL"
  --port "$SGL_PORT"
  --page-size 32
  --enable-hierarchical-cache
  --hicache-ratio 2)
if [[ "$BASELINE" != "1" ]]; then
  LAUNCH+=(--hicache-storage-backend dynamic
    --hicache-storage-backend-extra-config "$EXTRA_CONFIG")
fi

echo "[run] launching sglang (baseline=$BASELINE)"
"${LAUNCH[@]}" > "/tmp/sglang-$BASELINE.log" 2>&1 &
echo $! > /tmp/sglang.pid
for _ in $(seq 1 600); do
  curl -fsS "http://127.0.0.1:${SGL_PORT}/health" >/dev/null 2>&1 && break
  sleep 1
done

metrics_snapshot() {
  curl -fsS "http://127.0.0.1:${KVB_METRICS_PORT}/metrics" \
    | grep -E "kvblockd_(op_total|bytes_total)" || true
}

echo "[run] kvblockd metrics BEFORE:" && metrics_snapshot | tee /tmp/kvb-before.txt

# The upstream multi-turn driver ships in the sglang repo (not the wheel):
#   git clone --depth 1 --branch "v${SGLANG_VERSION}" \
#     https://github.com/sgl-project/sglang /tmp/sglang-src
BENCH="${BENCH:-/tmp/sglang-src/benchmark/hicache/bench_multiturn.py}"
python3 "$BENCH" --port "$SGL_PORT" \
  --num-clients "${NUM_CLIENTS:-16}" --num-rounds "${NUM_ROUNDS:-3}" \
  | tee "/tmp/multiturn-$BASELINE.txt"

echo "[run] kvblockd metrics AFTER:" && metrics_snapshot | tee /tmp/kvb-after.txt
echo "[run] diff the op counters: PUTs round 1, GET hits round 2+ = criterion 1"
echo "[run] token-identity: compare /tmp/multiturn-1.txt (BASELINE=1 run) vs"
echo "      /tmp/multiturn-0.txt output hashes = criterion 3"
kill "$(cat /tmp/sglang.pid)" 2>/dev/null || true
