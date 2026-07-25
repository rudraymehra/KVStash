#!/usr/bin/env bash
# Poll an HTTP path until the server answers. Any real status line (2xx/3xx/
# 4xx/5xx) means something is listening and speaking HTTP.
# Usage: wait-http.sh url [timeout_s] [pid]
#
# If pid is given and that process exits, stop waiting immediately instead of
# burning the whole timeout on a server that is never coming back.
#
# Read success from curl's exit status, not from the text of %{http_code}: on a
# connection failure curl prints "000" AND exits nonzero, so `code=$(curl … ||
# echo 000)` yielded the string "000000", which is not equal to "000" and was
# therefore accepted as success — a wait that returned instantly while nothing
# was listening at all.
set -euo pipefail
URL="${1:?usage: wait-http.sh url [timeout_s] [pid]}"
TIMEOUT="${2:-120}"
WATCH_PID="${3:-}"

deadline=$(( $(date +%s) + TIMEOUT ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  if code=$(curl -s -o /dev/null --max-time 5 -w '%{http_code}' "$URL" 2>/dev/null); then
    case "$code" in
      000|"") ;;   # curl exited 0 without a status line: keep waiting
      *) echo "up: $URL ($code)"; exit 0 ;;
    esac
  fi
  if [ -n "$WATCH_PID" ] && ! kill -0 "$WATCH_PID" 2>/dev/null; then
    echo "wait-http: watched process $WATCH_PID exited before $URL came up" >&2
    exit 2
  fi
  sleep 1
done
echo "wait-http: $URL never responded within ${TIMEOUT}s" >&2
exit 1
