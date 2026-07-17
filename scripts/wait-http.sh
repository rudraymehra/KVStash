#!/usr/bin/env bash
# Poll any HTTP path until it responds (any 2xx/4xx = server up).
# Usage: wait-http.sh url [timeout_s]
set -euo pipefail
URL="${1:?usage: wait-http.sh url [timeout_s]}"
TIMEOUT="${2:-120}"
deadline=$(( $(date +%s) + TIMEOUT ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  code=$(curl -s -o /dev/null -w '%{http_code}' "$URL" 2>/dev/null || echo 000)
  if [ "$code" != "000" ]; then echo "up: $URL ($code)"; exit 0; fi
  sleep 0.5
done
echo "wait-http: $URL never responded within ${TIMEOUT}s" >&2
exit 1
