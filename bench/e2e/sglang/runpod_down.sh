#!/usr/bin/env bash
# =============================================================================
# STATUS: NOT RUN — teardown twin of runpod_up.sh (budget rule: every rig has
# a written teardown step; verify $0 residue after every session).
# =============================================================================
set -euo pipefail

KVB_DIR="${KVB_DIR:-/opt/kvblockd}"

echo "[down] stop daemon + sglang on the pod (best-effort)"
[[ -f /tmp/sglang.pid ]] && kill "$(cat /tmp/sglang.pid)" 2>/dev/null || true
[[ -f "$KVB_DIR/kvblockd.pid" ]] && kill "$(cat "$KVB_DIR/kvblockd.pid")" 2>/dev/null || true

cat <<'EOF'
[down] NOW, FROM YOUR LAPTOP (the pod cannot delete itself):
  runpodctl get pod                       # list pods
  runpodctl stop pod <POD_ID>
  runpodctl remove pod <POD_ID>           # STOPPED pods still bill storage
  runpodctl get pod                       # MUST print no pods
[down] then check the RunPod billing page: balance delta this session only,
[down] zero running/stopped pods, zero network volumes = $0 residue. Log the
[down] session cost in the week file's budget table.
EOF
