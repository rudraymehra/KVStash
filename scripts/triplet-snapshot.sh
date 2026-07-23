#!/usr/bin/env bash
# triplet-snapshot.sh — record the exact (kvblockd, adapter, engine) triplet
# a deployment runs, as one JSON object on stdout.
#
# Partner-deploy provenance: every support conversation starts with "which
# three versions were you on?" — this answers it mechanically, so before/after
# numbers are always attributable to an exact triplet. Cron it daily next to
# the daemon (>> triplets.jsonl) or run it once per deploy.
#
#   scripts/triplet-snapshot.sh                      # auto-detect everything
#   KVB_BIN=/usr/local/bin/kvblockd \
#   KVB_PYTHON=/opt/venv/bin/python \
#     scripts/triplet-snapshot.sh > triplet.json
#
# What lands in the snapshot:
#   kvblockd  — binary version (`kvblockd --version`), plus git SHA/dirty
#               when run inside a source checkout (release binaries carry a
#               version, not a SHA; both are recorded when both exist)
#   adapters  — installed kvblockd / lmcache-kvblockd / vllm-kvblockd /
#               sglang-kvblockd package versions (importlib.metadata)
#   engines   — installed vllm / lmcache / sglang versions
# Missing pieces are null, never guessed. Authorship-time upstream pins live
# in per-adapter UPSTREAM.lock files where present (today only
# python/vllm_kvblockd/UPSTREAM.lock); this script records RUNTIME reality.
set -euo pipefail

KVB_BIN="${KVB_BIN:-kvblockd}"
KVB_PYTHON="${KVB_PYTHON:-python3}"

# --- kvblockd binary version ------------------------------------------------
KVB_VERSION=""
if command -v "$KVB_BIN" >/dev/null 2>&1 || [ -x "$KVB_BIN" ]; then
  # "kvblockd 0.5.0 (linux/amd64)" or "kvblockd dev (darwin/arm64)"
  KVB_VERSION=$("$KVB_BIN" --version 2>/dev/null | awk '{print $2}' || true)
fi

# --- git provenance (source deploys) -----------------------------------------
GIT_SHA=""
GIT_DIRTY=""
REPO_DIR="${KVB_REPO:-$(cd "$(dirname "$0")/.." 2>/dev/null && pwd)}"
if command -v git >/dev/null 2>&1 && git -C "$REPO_DIR" rev-parse --git-dir >/dev/null 2>&1; then
  GIT_SHA=$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || true)
  if [ -n "$(git -C "$REPO_DIR" status --porcelain 2>/dev/null)" ]; then
    GIT_DIRTY="true"
  else
    GIT_DIRTY="false"
  fi
fi

# --- assemble ----------------------------------------------------------------
# Python does the JSON (quoting-proof) and reads the installed adapter/engine
# packages from ITS OWN environment — point KVB_PYTHON at the venv the engine
# actually runs in, or the adapter fields report what that interpreter sees.
if command -v "$KVB_PYTHON" >/dev/null 2>&1; then
  KVB_VERSION="$KVB_VERSION" GIT_SHA="$GIT_SHA" GIT_DIRTY="$GIT_DIRTY" \
    "$KVB_PYTHON" - <<'PYEOF'
import json, os, platform, socket
from datetime import datetime, timezone
from importlib import metadata


def pkg(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


env = os.environ.get
snap = {
    "schema": 1,
    "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "host": socket.gethostname(),
    "kvblockd": {
        "version": env("KVB_VERSION") or None,
        "git_sha": env("GIT_SHA") or None,
        "git_dirty": {"true": True, "false": False}.get(env("GIT_DIRTY", "")),
    },
    "adapters": {
        "kvblockd": pkg("kvblockd"),
        "lmcache-kvblockd": pkg("lmcache-kvblockd"),
        "vllm-kvblockd": pkg("vllm-kvblockd"),
        "sglang-kvblockd": pkg("sglang-kvblockd"),
    },
    "engines": {
        "vllm": pkg("vllm"),
        "lmcache": pkg("lmcache"),
        "sglang": pkg("sglang"),
    },
    "python": platform.python_version(),
}
print(json.dumps(snap, sort_keys=True))
PYEOF
else
  # No python on the box: emit the shell-gathered fields honestly (adapters
  # and engines are python packages — without an interpreter they are
  # unknowable, and unknown must never print as a version).
  # Newlines/CRs become spaces first (a multi-line capture would break the
  # JSON string mid-token), then backslashes and quotes are escaped.
  esc() { printf '%s' "$1" | tr '\r\n' '  ' | sed 's/\\/\\\\/g; s/"/\\"/g'; }
  j_kvb_version="null"; [ -n "$KVB_VERSION" ] && j_kvb_version="\"$(esc "$KVB_VERSION")\""
  j_git_sha="null";     [ -n "$GIT_SHA" ] && j_git_sha="\"$(esc "$GIT_SHA")\""
  j_git_dirty="null";   [ -n "$GIT_DIRTY" ] && j_git_dirty="$GIT_DIRTY"
  printf '{"schema":1,"captured_at":"%s","host":"%s",' \
    "$(date -u +%Y-%m-%dT%H:%M:%S+00:00)" "$(esc "$(hostname)")"
  printf '"kvblockd":{"version":%s,"git_sha":%s,"git_dirty":%s},' \
    "$j_kvb_version" "$j_git_sha" "$j_git_dirty"
  printf '"adapters":null,"engines":null,"python":null}\n'
fi
