#!/usr/bin/env bash
# Poll a kvblockd /healthz until 200 or timeout. Usage: wait-healthz.sh host:port [timeout_s]
set -euo pipefail
ADDR="${1:?usage: wait-healthz.sh host:port [timeout_s]}"
TIMEOUT="${2:-30}"
deadline=$(( $(date +%s) + TIMEOUT ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  if curl -fsS --max-time 2 "http://${ADDR}/healthz" >/dev/null 2>&1; then
    echo "healthz OK: $ADDR"; exit 0
  fi
  sleep 0.3
done
echo "wait-healthz: $ADDR never returned 200 within ${TIMEOUT}s" >&2
exit 1
