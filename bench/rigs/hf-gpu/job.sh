#!/usr/bin/env bash
# Chart-2 TTFT rig — in-container entrypoint for a Hugging Face Job (default
# flavor a10g-large: 46 GB RAM, 1x A10G 24 GB; a10g-small's 15 GB got the run
# OOM-killed). Runs unattended:
#
#   install Go + build kvblockd from THIS checkout  ->  pip lmcache + our two
#   adapter packages  ->  write daemon + LMCache configs  ->  start kvblockd
#   ->  driver selftest  ->  PHASE 1 (populate): vLLM #1 prefills every sweep
#   prompt once; the driver asserts per prompt that kvblockd's put counters
#   grew  ->  RESTART: kill vLLM, start a fresh one (LMCache's in-process
#   local tier dies with the process; kvblockd keeps the blocks)  ->  PHASE 2
#   (measure): all warm reps (exact populated prompts; per-rep kvb_hits_total
#   growth proves the KV came from kvblockd over TCP), then all cold reps
#   (fresh-nonce prompts, full prefill)  ->  JSONL to stdout (CHART2JSONL
#   lines) + optional HF dataset upload  ->  exit nonzero on any failure.
#
# Honesty properties (run-3 post-mortem; details in run_ttft.py's docstring):
#   - lmcache local_cpu: true — the EXACT shape bench/e2e/cpu CI proves works.
#     Run 3 set local_cpu: false to "force" remote reads, but LMCache stages
#     remote writes THROUGH the local buffer, so that silently severed the
#     store path and the warm arm measured LMCache's own cache. Isolation now
#     comes from the vLLM restart (CI property (d): hits persist across a
#     restart), never from disabling tiers.
#   - vLLM --no-enable-prefix-caching in BOTH phases: a warm hit can never
#     come from vLLM's own prefix cache.
#   - populate FAILS LOUDLY if kvblockd received nothing; every measured warm
#     rep must grow kvb_hits_total or the job exits 3 with the cell flagged.
#   - the kvblockd arena is sized so every populated block stays resident
#     across the restart (derived below; eviction mid-run would silently turn
#     warm hits into misses).
#
# Base image: vllm/vllm-openai:v0.25.1 (CUDA torch + vLLM prebuilt; the pin
# matches bench/e2e/cpu/versions.env). Submit with bench/rigs/hf-gpu/submit.sh.
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
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"            # empty -> derived: max(LENGTHS) + GEN_TOKENS + 384
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
GO_VERSION="${GO_VERSION:-1.26.5}"
LMCACHE_VERSION="${LMCACHE_VERSION:-0.5.1}"
KV_BYTES_PER_TOKEN="${KV_BYTES_PER_TOKEN:-131072}"  # bf16 KV/token upper bound for the 7-8B class.
                                              # Llama-3.1-8B measured 128KiB/token (run 3: LMCache
                                              # allocated 32MiB per 256-token chunk); Qwen2.5-7B is 56KiB.
KVBD_ARENA_BYTES="${KVBD_ARENA_BYTES:-}"      # empty -> derived from LENGTHS x pairs x KV_BYTES_PER_TOKEN
KVBD_STREAMS="${KVBD_STREAMS:-4}"
LMC_MAX_LOCAL_CPU_GB="${LMC_MAX_LOCAL_CPU_GB:-8.0}" # LMCache local tier + pinned staging. One 16k-token
                                              # context at 128KiB/token = 2GiB, so 8GiB ≈ 4 contexts in
                                              # flight; run 3's 3.0 threw 32MiB allocation failures.
VLLM_HOST_RESERVE_GB="${VLLM_HOST_RESERVE_GB:-6}"   # RAM budgeted for the vLLM process in the fit check
PUT_WAIT_S="${PUT_WAIT_S:-120}"               # populate: max wait for kvblockd put counters per prompt
DRAIN_S="${DRAIN_S:-5}"                       # populate: async put queue must be quiet this long
TC_RATE_GBIT="${TC_RATE_GBIT:-}"              # optional loopback shaping; HF Jobs likely lack NET_ADMIN — detected, not assumed
RESULTS_REPO="${RESULTS_REPO:-}"              # optional HF DATASET repo for the JSONL (e.g. binarybhakt/kvstash-bench)
VLLM_PORT="${VLLM_PORT:-8000}"
KVBD_ADDR="127.0.0.1:9440"
KVBD_METRICS="127.0.0.1:9442"
KVBD_TOKEN="hf-gpu-token"
OUT_JSONL="$WORK/results/chart2-ttft.jsonl"
STATE_JSON="$WORK/results/chart2-state.json"

KVBD_PID="" VLLM_PID=""
cleanup() {
  [[ -n "$VLLM_PID" ]] && kill "$VLLM_PID" 2>/dev/null || true
  [[ -n "$KVBD_PID" ]] && kill "$KVBD_PID" 2>/dev/null || true
}
trap cleanup EXIT

tail_logs() {
  echo "=== kvblockd (tail) ==="; tail -n 40 "$WORK/kvbd.log" 2>/dev/null || true
  for f in "$WORK"/vllm-*.log; do
    [[ -e "$f" ]] || continue
    echo "=== $(basename "$f") (tail) ==="; tail -n 60 "$f" 2>/dev/null || true
  done
}
trap 'tail_logs' ERR

# ---- 0. sanity -------------------------------------------------------------
log "repo root: $ROOT"
[[ "$(uname -m)" == "x86_64" ]] || die "expected x86_64 (a10g flavor), got $(uname -m)"
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

# ---- 1. knob coherence: LENGTHS drives max-model-len AND the arena ----------
SUM_TOKENS=0
MAX_LEN=0
IFS=',' read -ra _LENS <<< "$LENGTHS"
for l in "${_LENS[@]}"; do
  l="${l//[[:space:]]/}"
  [[ "$l" =~ ^[0-9]+$ ]] || die "LENGTHS entry '$l' is not a number (LENGTHS=$LENGTHS)"
  SUM_TOKENS=$((SUM_TOKENS + l))
  if (( l > MAX_LEN )); then MAX_LEN=$l; fi
done
(( MAX_LEN > 0 )) || die "LENGTHS is empty"

# 384 headroom = nonce head + build_prompt's overshoot tolerance.
NEED_LEN=$((MAX_LEN + GEN_TOKENS + 384))
if [[ -z "$MAX_MODEL_LEN" ]]; then
  MAX_MODEL_LEN=$NEED_LEN
  log "derived MAX_MODEL_LEN=$MAX_MODEL_LEN (max length $MAX_LEN + $GEN_TOKENS gen + 384 headroom)"
elif (( MAX_MODEL_LEN < NEED_LEN )); then
  die "MAX_MODEL_LEN=$MAX_MODEL_LEN cannot fit the sweep: need >= $NEED_LEN (max LENGTHS $MAX_LEN + GEN_TOKENS $GEN_TOKENS + 384 headroom). Shrink LENGTHS or raise MAX_MODEL_LEN — run 3's 32000-vs-20480 mismatch must not recur silently."
fi

PAIRS=$((REPS + WARMUP))
if [[ -z "$KVBD_ARENA_BYTES" ]]; then
  # The warm arm only works if EVERY populated block is still resident when
  # its measured read arrives: arena >= all populated KV + 15% headroom.
  KVBD_ARENA_BYTES=$((SUM_TOKENS * PAIRS * KV_BYTES_PER_TOKEN * 115 / 100))
  log "derived KVBD_ARENA_BYTES=$KVBD_ARENA_BYTES (~$((KVBD_ARENA_BYTES / 1073741824))GiB = $SUM_TOKENS sweep tokens x $PAIRS pairs x ${KV_BYTES_PER_TOKEN}B/token x 1.15)"
fi

# Fit check: the arena is PREFAULTED, so an oversubscribed box OOMs mid-run.
if [[ -r /proc/meminfo ]]; then
  awk -v arena="$KVBD_ARENA_BYTES" -v lmc="$LMC_MAX_LOCAL_CPU_GB" -v resv="$VLLM_HOST_RESERVE_GB" '
    /^MemTotal:/ {
      total = $2 * 1024
      need = arena + (lmc + resv) * 1073741824
      printf "[job] RAM fit: arena %.1fGiB + lmcache %.1fGiB + vllm-reserve %.1fGiB = %.1fGiB of %.1fGiB total\n",
             arena / 1073741824, lmc, resv, need / 1073741824, total / 1073741824
      exit (need > total * 0.92) ? 1 : 0
    }' /proc/meminfo \
    || die "RAM budget exceeded (the kvblockd arena is prefaulted; the box would OOM mid-run). Cut REPS/LENGTHS, lower LMC_MAX_LOCAL_CPU_GB, or set KVBD_ARENA_BYTES explicitly."
fi

# ---- 2. Go toolchain + kvblockd from THIS checkout -------------------------
if ! command -v go >/dev/null 2>&1; then
  log "installing go$GO_VERSION"
  curl -fsSL "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz" | tar -C /usr/local -xz
  export PATH="/usr/local/go/bin:$PATH"
fi
go version
log "building kvblockd from source (latest release lags the current code)"
(cd "$ROOT" && go build -o "$WORK/bin/kvblockd" ./cmd/kvblockd)
"$WORK/bin/kvblockd" -version

# ---- 3. python stack (image already has CUDA torch + vLLM) ------------------
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

# ---- 4. configs (shape: bench/e2e/cpu/*, deltas documented in the header) ---
cat > "$WORK/namespaces.yaml" <<EOF
namespaces:
  - { name: lmcache, id: 1, token: $KVBD_TOKEN }
EOF
cat > "$WORK/kvblockd.yaml" <<EOF
# DRAM-only daemon for the TTFT rig (single box; NVMe/S3 tiers not exercised
# here). The arena must hold the ENTIRE populated sweep across the vLLM
# restart — sized/checked above.
listen_addr: "$KVBD_ADDR"
metrics_addr: "$KVBD_METRICS"
dram_arena_bytes: $KVBD_ARENA_BYTES
pinned_bytes_cap: 268435456
eviction_policy: "s3fifo"
namespaces_path: "$WORK/namespaces.yaml"
EOF
cat > "$WORK/lmcache_kvblockd.yaml" <<EOF
# LMCache -> kvblockd (plugin path) — the EXACT shape bench/e2e/cpu proves in
# CI. local_cpu stays ON: LMCache stages remote writes through the local
# buffer, so turning it off severs the store path (run-3 failure). Warm-arm
# isolation comes from the vLLM restart between populate and measure, not
# from this file.
chunk_size: 256
local_cpu: true
max_local_cpu_size: $LMC_MAX_LOCAL_CPU_GB
remote_storage_plugins: ["kvblockd"]
extra_config:
  kvblockd_token: "$KVBD_TOKEN"
  remote_storage_plugin.kvblockd.module_path: "lmcache_kvblockd.adapter"
  remote_storage_plugin.kvblockd.class_name: "KvblockdConnectorAdapter"
  # The plugin backend dials the virtual url "plugin://kvblockd"; the REAL
  # endpoint must be THIS extra_config key (remote_url would create a second,
  # deprecated backend and double every put — and was the silent zero-bytes
  # failure when it was the only endpoint the adapter understood).
  remote_storage_plugin.kvblockd.url: "kvblockd://$KVBD_ADDR?namespace=lmcache&streams=$KVBD_STREAMS"
EOF
cat > "$WORK/kv_transfer.json" <<EOF
{"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both",
 "kv_connector_extra_config": {"lmcache_config_file": "$WORK/lmcache_kvblockd.yaml"}}
EOF

# ---- 5. link shaping: attempt only if asked; record the truth either way ----
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

# ---- 6. start kvblockd (stays up across BOTH vLLM lifetimes) -----------------
log "starting kvblockd (DRAM arena $KVBD_ARENA_BYTES bytes)"
"$WORK/bin/kvblockd" -config "$WORK/kvblockd.yaml" > "$WORK/kvbd.log" 2>&1 &
KVBD_PID=$!
bash "$ROOT/scripts/wait-healthz.sh" "$KVBD_METRICS" 30

# ---- 7. driver selftest BEFORE spending GPU-minutes on requests -------------
# Proves the first-token timer AND the two-phase honesty gates (populate must
# fail on a severed store path; unproven warm hits must be flagged).
python3 "$HERE/run_ttft.py" --selftest

# ---- 8. vLLM lifecycle (started twice: populate engine, then a fresh measure engine)
VLLM_LOG=""
start_vllm() {  # $1 = log file, $2 = health timeout seconds
  VLLM_LOG="$1"
  log "starting vLLM: $MODEL (max_len=$MAX_MODEL_LEN, gpu_util=$GPU_MEM_UTIL) -> $VLLM_LOG"
  PYTHONHASHSEED=0 vllm serve "$MODEL" --port "$VLLM_PORT" --dtype bfloat16 \
    --max-model-len "$MAX_MODEL_LEN" --max-num-seqs 1 \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --no-enable-prefix-caching \
    --kv-transfer-config "$(cat "$WORK/kv_transfer.json")" \
    > "$VLLM_LOG" 2>&1 &
  VLLM_PID=$!
  # Passing the pid makes a crashed server fail in seconds instead of burning
  # the whole deadline (scripts/wait-http.sh).
  if ! bash "$ROOT/scripts/wait-http.sh" "http://127.0.0.1:$VLLM_PORT/health" "$2" "$VLLM_PID"; then
    # vLLM reports the engine-core failure as "See root cause above": the
    # worker process writes its real error well before the API-server
    # traceback, so a short tail hides exactly the line worth reading.
    log "FATAL: vLLM never became healthy. Memory at failure:"
    free -g >&2 || true
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv >&2 || true
    log "root-cause candidates from its log:"
    grep -nE 'Error|error|Killed|OOM|out of memory|CUDA|No space|Traceback|raise ' \
      "$VLLM_LOG" | tail -40 >&2 || true
    log "last 250 lines of $VLLM_LOG:"
    tail -250 "$VLLM_LOG" >&2
    exit 1
  fi
}

stop_vllm() {
  [[ -n "$VLLM_PID" ]] || return 0
  log "stopping vLLM pid $VLLM_PID"
  kill "$VLLM_PID" 2>/dev/null || true
  for _ in $(seq 1 60); do
    kill -0 "$VLLM_PID" 2>/dev/null || break
    sleep 1
  done
  kill -9 "$VLLM_PID" 2>/dev/null || true
  wait "$VLLM_PID" 2>/dev/null || true
  VLLM_PID=""
}

# ---- 9. PHASE 1: populate (vLLM #1 stores every sweep prompt into kvblockd) --
start_vllm "$WORK/vllm-populate.log" 2400   # generous: first boot downloads ~15 GB of weights
log "phase 1/2 populate: lengths=$LENGTHS x $PAIRS pairs; driver polls kvblockd's put counters per prompt and FAILS if they never grow"
python3 "$HERE/run_ttft.py" --phase populate \
  --vllm "http://127.0.0.1:$VLLM_PORT" --metrics "http://$KVBD_METRICS" \
  --model "$MODEL" --lengths "$LENGTHS" --reps "$REPS" --warmup "$WARMUP" \
  --state "$STATE_JSON" --put-wait-s "$PUT_WAIT_S" --drain-s "$DRAIN_S" \
  || die "populate failed — kvblockd never received the blocks, so there is nothing honest to measure (see FATAL(populate) above)"

# ---- 10. RESTART: fresh engine = no local KV anywhere; kvblockd keeps blocks -
log "restarting vLLM: a fresh engine has no local KV, so a warm hit can only be a kvblockd TCP read"
stop_vllm
start_vllm "$WORK/vllm-measure.log" 1200    # weights already on disk

# ---- 11. PHASE 2: measure (warm reps first, then cold — see run_ttft.py) -----
LMCACHE_VER="$(python3 -c 'import lmcache; print(lmcache.__version__)')"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | sed 's/^ *//;s/ *$//')"
log "phase 2/2 measure: reps=$REPS(+$WARMUP warmup) gen=$GEN_TOKENS link='$TC_LINK'"
rc=0
python3 "$HERE/run_ttft.py" --phase measure \
  --vllm "http://127.0.0.1:$VLLM_PORT" --metrics "http://$KVBD_METRICS" \
  --model "$MODEL" --gen-tokens "$GEN_TOKENS" \
  --state "$STATE_JSON" --out "$OUT_JSONL" \
  --stamp gpu="$GPU_NAME (hf-jobs ${FLAVOR:-flavor-unset})" \
  --stamp vllm="$VLLM_VER" --stamp lmcache="$LMCACHE_VER" \
  --stamp tc_link="$TC_LINK" --stamp rig="hf-jobs-${FLAVOR:-a10g}" \
  --stamp git_sha="$GIT_SHA" || rc=$?

# ---- 12. results out ---------------------------------------------------------
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
