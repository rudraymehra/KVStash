#!/usr/bin/env bash
# Chart-2 TTFT rig — in-container entrypoint for a Hugging Face Job
# (a10g-small: 4 vCPU, 15 GB RAM, 1x A10G 24 GB). Runs unattended:
#
#   install Go + build kvblockd from THIS checkout  ->  pip lmcache + our two
#   adapter packages  ->  write DRAM-only daemon + LMCache configs  ->  start
#   kvblockd  ->  start vLLM with the LMCacheConnectorV1 kv-transfer config
#   ->  driver selftest  ->  run_ttft.py cold/warm sweep  ->  JSONL to stdout
#   (CHART2JSONL lines) + optional HF dataset upload  ->  exit nonzero on any
#   failure.
#
# Base image: vllm/vllm-openai:v0.25.1 (CUDA torch + vLLM prebuilt; the pin
# matches bench/e2e/cpu/versions.env). Submit with bench/rigs/hf-gpu/submit.sh.
# Config shapes are derived from bench/e2e/cpu/{kvblockd,lmcache_kvblockd,
# namespaces}.yaml with two deliberate deltas, both load-bearing for honesty:
#   - lmcache local_cpu: false  -> a warm hit can ONLY come from kvblockd over
#     TCP (never LMCache's local CPU tier),
#   - vLLM --no-enable-prefix-caching -> a warm hit can never come from vLLM's
#     own prefix cache. The driver additionally verifies kvb_hits_total grows
#     on every warm rep.
set -euo pipefail

log() { printf '[job %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
die() { log "FATAL: $*"; exit 1; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"   # repo root (bench/rigs/hf-gpu -> .)
WORK="${WORK:-/tmp/kvb-ttft}"
mkdir -p "$WORK/bin" "$WORK/results"

# ---- knobs (all overridable via job -e/--env) ------------------------------
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
LENGTHS="${LENGTHS:-1024,4096,8192,16384}"
REPS="${REPS:-5}"
WARMUP="${WARMUP:-1}"
GEN_TOKENS="${GEN_TOKENS:-16}"
SETTLE_S="${SETTLE_S:-3.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-20480}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
GO_VERSION="${GO_VERSION:-1.26.5}"
LMCACHE_VERSION="${LMCACHE_VERSION:-0.5.1}"
KVBD_ARENA_BYTES="${KVBD_ARENA_BYTES:-2147483648}"   # 2 GiB — holds the freshest 16k-token KV with headroom; the arena is prefaulted, so it competes with vLLM for system RAM
KVBD_STREAMS="${KVBD_STREAMS:-4}"
LMC_MAX_LOCAL_CPU_GB="${LMC_MAX_LOCAL_CPU_GB:-3.0}"  # pinned staging pool only (local_cpu tier is OFF)
TC_RATE_GBIT="${TC_RATE_GBIT:-}"                     # optional loopback shaping; HF Jobs likely lack NET_ADMIN — detected, not assumed
RESULTS_REPO="${RESULTS_REPO:-}"                     # optional HF DATASET repo for the JSONL (e.g. binarybhakt/kvstash-bench)
VLLM_PORT="${VLLM_PORT:-8000}"
KVBD_ADDR="127.0.0.1:9440"
KVBD_METRICS="127.0.0.1:9442"
KVBD_TOKEN="hf-gpu-token"
OUT_JSONL="$WORK/results/chart2-ttft.jsonl"

KVBD_PID="" VLLM_PID=""
cleanup() {
  [[ -n "$VLLM_PID" ]] && kill "$VLLM_PID" 2>/dev/null || true
  [[ -n "$KVBD_PID" ]] && kill "$KVBD_PID" 2>/dev/null || true
}
trap cleanup EXIT

tail_logs() {
  echo "=== kvblockd (tail) ==="; tail -n 40 "$WORK/kvbd.log" 2>/dev/null || true
  echo "=== vllm (tail) ==="; tail -n 60 "$WORK/vllm.log" 2>/dev/null || true
}
trap 'tail_logs' ERR

# ---- 0. sanity -------------------------------------------------------------
log "repo root: $ROOT"
[[ "$(uname -m)" == "x86_64" ]] || die "expected x86_64 (a10g-small), got $(uname -m)"
command -v nvidia-smi >/dev/null || die "nvidia-smi missing — not a GPU flavor?"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
free -h || true
df -h / /tmp 2>/dev/null | tail -n +1 || true
VLLM_VER="$(python3 -c 'import vllm; print(vllm.__version__)')" || die "vllm not importable in this image"
log "image vllm=$VLLM_VER"

GIT_SHA="unknown"
if git -C "$ROOT" rev-parse --short=12 HEAD >/dev/null 2>&1; then
  GIT_SHA="$(git -C "$ROOT" rev-parse --short=12 HEAD)"
elif [[ -n "${GIT_REF:-}" ]]; then
  GIT_SHA="$GIT_REF"   # tarball checkout: stamp the requested ref instead
fi
log "git_sha stamp: $GIT_SHA"

# ---- 1. Go toolchain + kvblockd from THIS checkout -------------------------
if ! command -v go >/dev/null 2>&1; then
  log "installing go$GO_VERSION"
  curl -fsSL "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz" | tar -C /usr/local -xz
  export PATH="/usr/local/go/bin:$PATH"
fi
go version
log "building kvblockd from source (latest release lags the current code)"
(cd "$ROOT" && go build -o "$WORK/bin/kvblockd" ./cmd/kvblockd)
"$WORK/bin/kvblockd" -version

# ---- 2. python stack (image already has CUDA torch + vLLM) ------------------
pipi() { pip install "$@" || pip install --break-system-packages "$@"; }  # PEP-668 images
log "pip: lmcache==$LMCACHE_VERSION + kvblockd + lmcache_kvblockd (editable)"
pipi "lmcache==$LMCACHE_VERSION" || {
  log "lmcache install failed (source build without nvcc?) — retrying with NO_GPU_EXT=1"
  NO_GPU_EXT=1 pipi "lmcache==$LMCACHE_VERSION"
}
pipi -e "$ROOT/python/kvblockd" -e "$ROOT/python/lmcache_kvblockd"
# lmcache's dependency tree must NOT have replaced the image's CUDA torch or vLLM:
python3 - <<PY
import torch, vllm, lmcache, kvblockd, lmcache_kvblockd
assert torch.version.cuda, "torch lost CUDA — lmcache install replaced it"
assert torch.cuda.is_available(), "CUDA not available to torch"
assert vllm.__version__ == "$VLLM_VER", f"vllm changed: {vllm.__version__} != $VLLM_VER"
print("imports OK: vllm", vllm.__version__, "lmcache", lmcache.__version__,
      "torch", torch.__version__, "cuda", torch.version.cuda)
PY

# ---- 3. configs (shape: bench/e2e/cpu/*, deltas documented in the header) ---
cat > "$WORK/namespaces.yaml" <<EOF
namespaces:
  - { name: lmcache, id: 1, token: $KVBD_TOKEN }
EOF
cat > "$WORK/kvblockd.yaml" <<EOF
# DRAM-only daemon for the TTFT rig (single box; NVMe/S3 tiers not exercised here).
listen_addr: "$KVBD_ADDR"
metrics_addr: "$KVBD_METRICS"
dram_arena_bytes: $KVBD_ARENA_BYTES
pinned_bytes_cap: 268435456
eviction_policy: "s3fifo"
namespaces_path: "$WORK/namespaces.yaml"
EOF
cat > "$WORK/lmcache_kvblockd.yaml" <<EOF
# LMCache -> kvblockd (plugin path). local_cpu is OFF so every warm hit is a
# REMOTE TCP fetch from kvblockd — the thing Chart 2 measures.
chunk_size: 256
local_cpu: false
max_local_cpu_size: $LMC_MAX_LOCAL_CPU_GB
remote_url: "kvblockd://$KVBD_ADDR?namespace=lmcache&streams=$KVBD_STREAMS"
remote_storage_plugins: ["kvblockd"]
extra_config:
  kvblockd_token: "$KVBD_TOKEN"
  remote_storage_plugin.kvblockd.module_path: "lmcache_kvblockd.adapter"
  remote_storage_plugin.kvblockd.class_name: "KvblockdConnectorAdapter"
EOF
cat > "$WORK/kv_transfer.json" <<EOF
{"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both",
 "kv_connector_extra_config": {"lmcache_config_file": "$WORK/lmcache_kvblockd.yaml"}}
EOF

# ---- 4. link shaping: attempt only if asked; record the truth either way ----
TC_LINK="unshaped-loopback (tc shaping not attempted)"
if [[ -n "$TC_RATE_GBIT" ]]; then
  if command -v tc >/dev/null 2>&1 \
     && tc qdisc replace dev lo root tbf rate "${TC_RATE_GBIT}gbit" burst 32mbit latency 1ms 2>/dev/null; then
    TC_LINK="tbf ${TC_RATE_GBIT}gbit loopback"
    log "tc shaping applied: $TC_LINK"
  else
    TC_LINK="unshaped-loopback (tc failed: likely no NET_ADMIN on HF Jobs)"
    log "tc shaping REQUESTED but failed — continuing unshaped, disclosed in JSONL"
  fi
fi

# ---- 5. start kvblockd ------------------------------------------------------
log "starting kvblockd (DRAM arena $KVBD_ARENA_BYTES bytes)"
"$WORK/bin/kvblockd" -config "$WORK/kvblockd.yaml" > "$WORK/kvbd.log" 2>&1 &
KVBD_PID=$!
bash "$ROOT/scripts/wait-healthz.sh" "$KVBD_METRICS" 30

# ---- 6. driver selftest BEFORE spending GPU-minutes on requests -------------
python3 "$HERE/run_ttft.py" --selftest

# ---- 7. start vLLM (LMCache connector; local caches off — see header) -------
log "starting vLLM: $MODEL (max_len=$MAX_MODEL_LEN, gpu_util=$GPU_MEM_UTIL)"
PYTHONHASHSEED=0 vllm serve "$MODEL" --port "$VLLM_PORT" --dtype bfloat16 \
  --max-model-len "$MAX_MODEL_LEN" --max-num-seqs 1 \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --no-enable-prefix-caching \
  --kv-transfer-config "$(cat "$WORK/kv_transfer.json")" \
  > "$WORK/vllm.log" 2>&1 &
VLLM_PID=$!
# generous: first boot downloads ~15 GB of weights. Passing the pid makes a
# crashed server fail in seconds instead of burning the whole deadline.
if ! bash "$ROOT/scripts/wait-http.sh" "http://127.0.0.1:$VLLM_PORT/health" 2400 "$VLLM_PID"; then
  # vLLM reports the engine-core failure as "See root cause above": the worker
  # process writes its real error well before the API-server traceback, so a
  # short tail hides exactly the line worth reading.
  log "FATAL: vLLM never became healthy. Memory at failure:"
  free -g >&2 || true
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv >&2 || true
  log "root-cause candidates from its log:"
  grep -nE 'Error|error|Killed|OOM|out of memory|CUDA|No space|Traceback|raise ' \
    "$WORK/vllm.log" | tail -40 >&2 || true
  log "last 250 lines of vllm.log:"
  tail -250 "$WORK/vllm.log" >&2
  exit 1
fi

# ---- 8. the sweep ------------------------------------------------------------
LMCACHE_VER="$(python3 -c 'import lmcache; print(lmcache.__version__)')"
log "sweep: lengths=$LENGTHS reps=$REPS(+$WARMUP warmup) gen=$GEN_TOKENS link='$TC_LINK'"
rc=0
python3 "$HERE/run_ttft.py" \
  --vllm "http://127.0.0.1:$VLLM_PORT" --metrics "http://$KVBD_METRICS" \
  --model "$MODEL" --lengths "$LENGTHS" --reps "$REPS" --warmup "$WARMUP" \
  --gen-tokens "$GEN_TOKENS" --settle-s "$SETTLE_S" --out "$OUT_JSONL" \
  --stamp gpu="NVIDIA A10G 24GB (hf-jobs a10g-small)" \
  --stamp vllm="$VLLM_VER" --stamp lmcache="$LMCACHE_VER" \
  --stamp tc_link="$TC_LINK" --stamp rig="hf-jobs-a10g" \
  --stamp git_sha="$GIT_SHA" || rc=$?

# ---- 9. results out ----------------------------------------------------------
log "JSONL at $OUT_JSONL ($(wc -l < "$OUT_JSONL" 2>/dev/null || echo 0) records); every record was also printed above as a CHART2JSONL line"
if [[ -n "$RESULTS_REPO" && -s "$OUT_JSONL" ]]; then
  log "uploading JSONL to dataset $RESULTS_REPO"
  python3 - "$RESULTS_REPO" "$OUT_JSONL" "$GIT_SHA" <<'PY' || log "WARN: upload failed (non-fatal — the CHART2JSONL log lines are the primary retrieval path)"
import sys, time
from huggingface_hub import HfApi
repo, path, sha = sys.argv[1:4]
api = HfApi()
api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
dest = f"chart2/{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{sha}.jsonl"
api.upload_file(path_or_fileobj=path, path_in_repo=dest, repo_id=repo, repo_type="dataset")
print(f"uploaded -> hf://datasets/{repo}/{dest}")
PY
fi

if [[ $rc -ne 0 ]]; then
  tail_logs
  die "driver exited rc=$rc (rc=3 means data was produced but some warm reps were UNVERIFIED)"
fi
log "DONE — retrieve with: hf jobs logs <job_id> | sed -n 's/^.*CHART2JSONL //p' > chart2-ttft.jsonl"
