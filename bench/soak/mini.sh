#!/usr/bin/env bash
# soak-mini: the 10-minute local endurance smoke (see Makefile). Boots a
# daemon with a 256 MiB arena, drives a ~384 MiB working set through it
# (continuous eviction), and fails on ANY byte-verification miss.
# Usage: bash bench/soak/mini.sh [duration] (default 10m)
set -euo pipefail

DUR="${1:-10m}"
ROOT="$(git rev-parse --show-toplevel)"
DPID=""
TMP="$(mktemp -d -t kvb-soak-mini.XXXXXX)"
trap '[ -n "${DPID:-}" ] && kill "$DPID" 2>/dev/null || true; rm -rf "$TMP"' EXIT

cd "$ROOT"
go build -o "$TMP/kvblockd" ./cmd/kvblockd
go build -o "$TMP/soak" ./bench/soak

cat > "$TMP/ns.yaml" <<'EOF'
namespaces:
  - { name: soak-a, id: 11, token: soak-token }
  - { name: soak-b, id: 12, token: soak-token }
  - { name: soak-c, id: 13, token: soak-token }
EOF

KVBLOCKD_DRAM_ARENA_BYTES=268435456 KVBLOCKD_PINNED_BYTES_CAP=16777216 \
KVBLOCKD_METRICS_ADDR=127.0.0.1:19542 \
  "$TMP/kvblockd" -listen 127.0.0.1:19540 -namespaces "$TMP/ns.yaml" 2> "$TMP/daemon.log" &
DPID=$!
sleep 1.5

# ~380 MiB working set: 340 keys × ~1.1 MiB mean over a 256 MiB arena (1.5×).
RC=0
"$TMP/soak" -addr 127.0.0.1:19540 -duration "$DUR" -workers 12 -keys 340 \
  -report 30s -out "$TMP/soak.jsonl" || RC=$?

echo "--- final line:"
tail -1 "$TMP/soak.jsonl"
echo "--- daemon evictions:"
EVICTIONS=$(curl -s 127.0.0.1:19542/metrics | awk '/^kvb_evictions_total/{print int($2)}')
echo "kvb_evictions_total=${EVICTIONS:-0}"
if [ "${EVICTIONS:-0}" -eq 0 ]; then
  echo "soak-mini: FAIL — no evictions observed (the pressure shape is broken)" >&2
  RC=1
fi

kill -TERM "$DPID" 2>/dev/null || true
wait "$DPID" 2>/dev/null || true
exit "$RC"
