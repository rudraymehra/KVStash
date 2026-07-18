#!/usr/bin/env bash
# fetch.sh — clone the production traces at pinned SHAs, checksum every file,
# and record provenance. Traces are the honest-hit-rate evidence (methodology
# rule 9: production traces over synthetic wherever a claim is about hit rates
# or policies).
#
# The two sources (both Apache-2.0):
#   Bailian:  alibaba-edu/qwen-bailian-usagetraces-anon  (ATC'25; ideal hit
#             rate 62% chat / 54% API — NOT the marketing 90%)
#   Mooncake: kvcache-ai/Mooncake  FAST25-release/traces/ (12031/23608/3993)
#
# Pinning: pass BAILIAN_SHA / MOONCAKE_SHA (from bench/VERSIONS.lock) as env
# vars to check out exact commits; unset = origin HEAD (the SHA is then
# PRINTED at the end for you to paste into VERSIONS.lock — the script does
# not edit the lock file itself, to keep the pin a deliberate human commit).
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
DEST="${1:-$DIR/data}"
mkdir -p "$DEST"

BAILIAN_REPO="https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon"
MOONCAKE_REPO="https://github.com/kvcache-ai/Mooncake"

# clone_at <url> <dir> <sha-or-empty> — logs to STDERR so the function's
# stdout is ONLY the resolved HEAD (the old code let a log line pollute the
# captured SHA).
clone_at() {
  local url="$1" dir="$2" sha="${3:-}"
  if [ -d "$dir/.git" ]; then
    echo "[fetch] $dir already cloned" >&2
  else
    git clone "$url" "$dir" >&2
  fi
  if [ -n "$sha" ]; then
    ( cd "$dir" && git fetch origin "$sha" >&2 && git checkout -q "$sha" )
  fi
  ( cd "$dir" && git rev-parse HEAD )
}

echo "[fetch] Bailian traces" >&2
BSHA=$(clone_at "$BAILIAN_REPO" "$DEST/bailian" "${BAILIAN_SHA:-}")
echo "[fetch] Mooncake traces" >&2
MSHA=$(clone_at "$MOONCAKE_REPO" "$DEST/mooncake" "${MOONCAKE_SHA:-}")

echo "[fetch] checksums" >&2
: > "$DEST/SHA256SUMS"
sha_tool() { if command -v sha256sum >/dev/null; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }
find "$DEST/bailian" "$DEST/mooncake" -type f \( -name '*.jsonl' -o -name '*.json' -o -name '*.csv' \) \
  -not -path '*/.git/*' -print0 | sort -z | while IFS= read -r -d '' f; do
  sha_tool "$f" | sed "s|$DEST/||" >> "$DEST/SHA256SUMS"
done

cat > "$DIR/PROVENANCE.txt" <<EOF
Fetched $(date -u +%FT%TZ)
bailian  = $BAILIAN_REPO @ $BSHA   (Apache-2.0)
mooncake = $MOONCAKE_REPO @ $MSHA  (Apache-2.0)
checksums: bench/traces/data/SHA256SUMS
Block-size mapping rationale: bench/traces/README.md
Published counts (converter gate): mooncake 12031 / 23608 / 3993; bailian per-file (see README).
EOF
echo "[fetch] done. If these were UNPINNED (HEAD), paste into bench/VERSIONS.lock and commit:"
echo "  bailian_repo  = alibaba-edu/qwen-bailian-usagetraces-anon @ $BSHA"
echo "  mooncake_repo = kvcache-ai/Mooncake @ $MSHA"
echo "  (then re-run pinned: BAILIAN_SHA=$BSHA MOONCAKE_SHA=$MSHA $0)"
