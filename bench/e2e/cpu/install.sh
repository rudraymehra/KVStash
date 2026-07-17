#!/usr/bin/env bash
# Install the CPU e2e stack from pinned versions (bench/e2e/cpu/versions.env).
# Reused verbatim by .github/workflows/e2e-cpu.yml. CPU-only: no CUDA.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
source "$HERE/versions.env"

echo "[install] CPU torch"
pip install --index-url "$TORCH_INDEX" torch

echo "[install] vllm==$VLLM_VERSION (wheel-first)"
pip install "vllm==$VLLM_VERSION"

echo "[install] lmcache==$LMCACHE_VERSION (no GPU ext)"
NO_GPU_EXT=1 pip install "lmcache==$LMCACHE_VERSION"

echo "[install] kvblockd + lmcache_kvblockd (editable)"
pip install -e "$ROOT/python/kvblockd" -e "$ROOT/python/lmcache_kvblockd"

echo "[install] done: $(python -c 'import vllm,lmcache,kvblockd,lmcache_kvblockd; print("imports OK")')"
