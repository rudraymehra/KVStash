#!/usr/bin/env bash
# =============================================================================
# STATUS: NOT RUN — written for the wk-12 SGLang spike, DEFERRED before any
# GPU session (no GPU budget this sprint; docs/design/sglang-hicache-v1.1.md).
# Re-verify every pin in versions.env before the first real run.
# =============================================================================
# Provision the SGLang e2e box: RunPod RTX 4090 secure (~$0.69/hr), Linux
# x86_64, CUDA 12.x image (runpod/pytorch:*). Run THIS SCRIPT ON THE POD
# after the repo tree lands in ~/kvstash — the repo is PRIVATE, so that
# means a deploy-key/token `git clone`, or rsync of the working tree from
# the laptop. Everything (daemon included) is built from that tree; nothing
# is fetched from GitHub releases.
#
# Teardown is MANDATORY after every session: bench/e2e/sglang/runpod_down.sh
# (budget rule: every rig has a written teardown step; verify $0 residue).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
source "$HERE/versions.env"

ARENA_BYTES=$((4 * 1024 * 1024 * 1024)) # 4 GiB DRAM arena (week-12 Day-3 spec)
KVB_DIR="${KVB_DIR:-/opt/kvblockd}"
KVB_PORT="${KVB_PORT:-9400}"
KVB_METRICS_PORT="${KVB_METRICS_PORT:-9401}"
KVB_TOKEN="${KVB_TOKEN:-sglang-e2e-token}"

echo "[up] 1/5 build kvblockd from this tree (private repo — no release pull)"
mkdir -p "$KVB_DIR"
if ! command -v go >/dev/null 2>&1; then
  echo "[up]   installing go${GO_VERSION} (runpod/pytorch images ship none)"
  curl -fsSL -o /tmp/go.tgz "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz"
  rm -rf /usr/local/go
  tar -C /usr/local -xzf /tmp/go.tgz
  export PATH="/usr/local/go/bin:$PATH"
fi
(cd "$ROOT" && go build -o "$KVB_DIR/kvblockd" ./cmd/kvblockd)
echo "[up]   built from: $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo 'rsync tree (no .git)')"
"$KVB_DIR/kvblockd" --version

echo "[up] 2/5 daemon config (4GiB DRAM arena, namespace sglang-e2e)"
cat > "$KVB_DIR/ns.yaml" <<EOF
namespaces:
  - { name: sglang-e2e, id: 12, token: ${KVB_TOKEN} }
EOF
cat > "$KVB_DIR/cfg.yaml" <<EOF
listen_addr: "127.0.0.1:${KVB_PORT}"
metrics_addr: "127.0.0.1:${KVB_METRICS_PORT}"
dram_arena_bytes: ${ARENA_BYTES}
namespaces_path: "${KVB_DIR}/ns.yaml"
EOF

echo "[up] 3/5 start daemon"
nohup "$KVB_DIR/kvblockd" -config "$KVB_DIR/cfg.yaml" \
  > "$KVB_DIR/kvblockd.log" 2>&1 &
echo $! > "$KVB_DIR/kvblockd.pid"
for _ in $(seq 1 50); do
  curl -fsS "http://127.0.0.1:${KVB_METRICS_PORT}/healthz" >/dev/null 2>&1 && break
  sleep 0.2
done
curl -fsS "http://127.0.0.1:${KVB_METRICS_PORT}/healthz" >/dev/null

echo "[up] 4/5 sglang==${SGLANG_VERSION} (pinned) + flashinfer per its docs"
pip install "sglang[all]==${SGLANG_VERSION}"

echo "[up] 5/5 kvblockd python client + sglang backend (editable)"
pip install -e "$ROOT/python/kvblockd" -e "$ROOT/python/sglang_kvblockd"
python "$HERE/import_check.py" --require-sglang

echo "[up] done. Next: bench/e2e/sglang/run_multiturn.sh"
echo "[up] REMINDER: runpod_down.sh + \$0-residue check before leaving the box."
