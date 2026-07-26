#!/usr/bin/env bash
# Chart-2 TTFT rig — in-container entrypoint for a Hugging Face Job (default
# flavor a10g-large: 46 GB RAM, 1x A10G 24 GB; a10g-small's 15 GB got the run
# OOM-killed). Runs unattended:
#
#   install Go + build kvblockd from THIS checkout  ->  pip our client + the
#   NATIVE vLLM connector (vllm_kvblockd)  ->  write daemon + transfer configs
#   ->  start kvblockd  ->  driver selftest  ->  PHASE 1 (populate): vLLM #1
#   prefills every sweep prompt once; the driver asserts per prompt that
#   kvblockd's put counters grew  ->  RESTART: kill vLLM, start a fresh one
#   (any engine-local KV dies with the process; kvblockd keeps the blocks)
#   ->  PHASE 2 (measure): all warm reps (exact populated prompts; per-rep
#   kvb_hits_total growth proves the KV came from kvblockd over TCP), then
#   all cold reps (fresh-nonce prompts, full prefill)  ->  JSONL to stdout
#   (CHART2JSONL lines) + optional HF dataset upload  ->  exit nonzero on any
#   failure.
#
# WHY THE NATIVE CONNECTOR (vllm_kvblockd.KvblockdConnector), NOT LMCache:
#   the LMCache route ate seven GPU runs without moving one byte; the free CPU
#   Docker rig (bench/e2e/cpu/local-docker.sh) then proved the native
#   connector end-to-end — real prefill bytes into kvblockd, hits surviving an
#   engine restart, mixed local+remote — while `lmcache-probe` pinned that
#   LMCache cannot even construct its engine off-CUDA at the current releases.
#   This rig runs the one path with demonstrated bytes. The LMCache GPU leg
#   remains UNVALIDATED, not disproven — its configs live in git history and
#   docs/INTEGRATIONS.md.
#
# Honesty properties (run-3 post-mortem; details in run_ttft.py's docstring):
#   - isolation comes from the vLLM restart between populate and measure
#     (CI property (d): hits persist across a restart), never from disabling
#     tiers or trusting connector internals.
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
BASELINE_ONLY="${BASELINE_ONLY:-0}"           # 1 = pure-recompute control: NO connector, NO daemon,
                                              # one engine boot, cold-only sweep. The third chart
                                              # series; see run_ttft.py --phase baseline.
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
LENGTHS="${LENGTHS:-1024,4096,8192,16384}"
REPS="${REPS:-5}"
WARMUP="${WARMUP:-1}"
GEN_TOKENS="${GEN_TOKENS:-16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"            # empty -> derived: max(LENGTHS) + GEN_TOKENS + 384
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
GO_VERSION="${GO_VERSION:-1.26.5}"
KV_BYTES_PER_TOKEN="${KV_BYTES_PER_TOKEN:-131072}"  # bf16 KV/token upper bound for the 7-8B class.
                                              # Llama-3.1-8B measured 128KiB/token; Qwen2.5-7B is 56KiB.
                                              # The native connector's blob = the raw paged block + 32B
                                              # prefix, so the same per-token bound holds.
KVBD_ARENA_BYTES="${KVBD_ARENA_BYTES:-}"      # empty -> derived from LENGTHS x pairs x KV_BYTES_PER_TOKEN
KVBD_STREAMS="${KVBD_STREAMS:-4}"
CONNECTOR_STAGING_GB="${CONNECTOR_STAGING_GB:-2}"   # native-connector host staging: it moves ONE paged
                                              # block at a time through a transient CPU tensor (~MB
                                              # scale), so this is headroom, not a tier.
VLLM_HOST_RESERVE_GB="${VLLM_HOST_RESERVE_GB:-6}"   # RAM budgeted for the vLLM process in the fit check
PUT_WAIT_S="${PUT_WAIT_S:-120}"               # populate: max wait for kvblockd put counters per prompt
DRAIN_S="${DRAIN_S:-5}"                       # populate: put counters must be quiet this long (the
                                              # native store path is synchronous, so this passes
                                              # immediately; kept as armor if a connector goes async)
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
if [[ "$BASELINE_ONLY" == "1" ]]; then
  KVBD_ARENA_BYTES=0   # no daemon in this mode; silence the derivation below
fi
if [[ -z "$KVBD_ARENA_BYTES" ]]; then
  # The warm arm only works if EVERY populated block is still resident when
  # its measured read arrives: arena >= all populated KV + 15% headroom.
  KVBD_ARENA_BYTES=$((SUM_TOKENS * PAIRS * KV_BYTES_PER_TOKEN * 115 / 100))
  log "derived KVBD_ARENA_BYTES=$KVBD_ARENA_BYTES (~$((KVBD_ARENA_BYTES / 1073741824))GiB = $SUM_TOKENS sweep tokens x $PAIRS pairs x ${KV_BYTES_PER_TOKEN}B/token x 1.15)"
fi

# Fit check: the arena is PREFAULTED, so an oversubscribed box OOMs mid-run.
if [[ -r /proc/meminfo ]]; then
  awk -v arena="$KVBD_ARENA_BYTES" -v stage="$CONNECTOR_STAGING_GB" -v resv="$VLLM_HOST_RESERVE_GB" '
    /^MemTotal:/ {
      total = $2 * 1024
      need = arena + (stage + resv) * 1073741824
      printf "[job] RAM fit: arena %.1fGiB + connector-staging %.1fGiB + vllm-reserve %.1fGiB = %.1fGiB of %.1fGiB total\n",
             arena / 1073741824, stage, resv, need / 1073741824, total / 1073741824
      exit (need > total * 0.92) ? 1 : 0
    }' /proc/meminfo \
    || die "RAM budget exceeded (the kvblockd arena is prefaulted; the box would OOM mid-run). Cut REPS/LENGTHS or set KVBD_ARENA_BYTES explicitly."
fi

# ---- 2. Go toolchain + kvblockd from THIS checkout -------------------------
# (skipped in baseline mode: no daemon, no connector, nothing to build)
if [[ "$BASELINE_ONLY" != "1" ]]; then
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
log "pip: kvblockd client + vllm_kvblockd native connector (editable, from THIS checkout)"
pipi -e "$ROOT/python/kvblockd" -e "$ROOT/python/vllm_kvblockd"
# The installs must NOT have replaced the image's CUDA torch or vLLM:
python3 - <<PY
import torch, vllm, kvblockd, vllm_kvblockd
from vllm_kvblockd.connector import KvblockdConnector  # the class vLLM will load
assert torch.version.cuda, "torch lost CUDA — a dependency replaced it"
assert torch.cuda.is_available(), "CUDA not available to torch"
assert vllm.__version__ == "$VLLM_VER", f"vllm changed: {vllm.__version__} != $VLLM_VER"
print("imports OK: vllm", vllm.__version__, "vllm_kvblockd", vllm_kvblockd.__version__,
      "torch", torch.__version__, "cuda", torch.version.cuda)
PY

# ---- 4. configs (shape: bench/e2e/cpu/local-docker.sh, the proven rig) ------
cat > "$WORK/namespaces.yaml" <<EOF
namespaces:
  - { name: vllm, id: 1, token: $KVBD_TOKEN }
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
# The EXACT transfer-config shape the free Docker rig proved moves bytes
# (bench/e2e/cpu/local-docker.sh): vLLM loads KvblockdConnector out-of-tree
# via kv_connector_module_path and hands it the kvblockd_* extra-config keys.
# There is no library between the engine and the daemon — no second config
# channel to silently miss.
cat > "$WORK/kv_transfer.json" <<EOF
{"kv_connector": "KvblockdConnector", "kv_role": "kv_both",
 "kv_connector_module_path": "vllm_kvblockd.connector",
 "kv_connector_extra_config": {
   "kvblockd_endpoint": "kvblockd://$KVBD_ADDR",
   "kvblockd_namespace": "vllm",
   "kvblockd_token": "$KVBD_TOKEN",
   "kvblockd_streams": $KVBD_STREAMS}}
EOF
python3 -c "import json,sys; json.load(open('$WORK/kv_transfer.json'))" \
  || die "generated kv_transfer.json is not valid JSON"
fi  # BASELINE_ONLY skip of build/pip/configs

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
if [[ "$BASELINE_ONLY" != "1" ]]; then
  log "starting kvblockd (DRAM arena $KVBD_ARENA_BYTES bytes)"
  "$WORK/bin/kvblockd" -config "$WORK/kvblockd.yaml" > "$WORK/kvbd.log" 2>&1 &
  KVBD_PID=$!
  bash "$ROOT/scripts/wait-healthz.sh" "$KVBD_METRICS" 30
fi

# ---- 7. driver selftest BEFORE spending GPU-minutes on requests -------------
# Proves the first-token timer AND the two-phase honesty gates (populate must
# fail on a severed store path; unproven warm hits must be flagged). The sed
# renames the selftest's stub CHART2JSONL lines so the log-retrieval sed can
# never scoop stub records into a results file (it did once — the run-4
# extraction picked up 8 selftest records until filtered by rig).
python3 "$HERE/run_ttft.py" --selftest | sed 's/^CHART2JSONL /SELFTESTJSONL /'

# ---- 8. vLLM lifecycle (started twice: populate engine, then a fresh measure engine)
VLLM_LOG=""
start_vllm() {  # $1 = log file, $2 = health timeout seconds
  VLLM_LOG="$1"
  log "starting vLLM: $MODEL (max_len=$MAX_MODEL_LEN, gpu_util=$GPU_MEM_UTIL) -> $VLLM_LOG"
  # PYTHONHASHSEED=0: the connector's key chain seeds from a determinism check
  # (config.require_pinned_hashseed) — unpinned, it refuses to boot rather
  # than silently never sharing cache. The env var reaches every child, and
  # vLLM spawns the engine core in a separate process.
  # --disable-hybrid-kv-cache-manager: v0.25's factory refuses external
  # connectors without SupportsHMA otherwise (connector.py docstring; the
  # Docker rig serves with the same flag).
  # Baseline mode serves with NO --kv-transfer-config: the whole point of
  # that control series is an engine that cannot pay any connector cost.
  KV_XFER_ARGS=(--kv-transfer-config "$(cat "$WORK/kv_transfer.json" 2>/dev/null || echo '{}')")
  if [[ "$BASELINE_ONLY" == "1" ]]; then KV_XFER_ARGS=(); fi
  PYTHONHASHSEED=0 \
  vllm serve "$MODEL" --port "$VLLM_PORT" --dtype bfloat16 \
    --max-model-len "$MAX_MODEL_LEN" --max-num-seqs 1 \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --no-enable-prefix-caching \
    --disable-hybrid-kv-cache-manager \
    "${KV_XFER_ARGS[@]}" \
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
  # FAST-FAIL GATE. The native path has no LMCache-style "healthy but
  # silently degraded" init mode: if the connector class can't be imported or
  # constructed, the vLLM factory raises and /health never comes up. What CAN
  # still be silently wrong at this point is a dropped transfer config (vLLM
  # serving with no connector at all). Gate on the FACTORY'S construction
  # line — vendored at python/vllm_kvblockd/upstream_refs/
  # kv_connector_factory.py:62-66, logged at INFO strictly before serving —
  # NOT the bare class name: vLLM echoes its parsed config at API-server
  # startup, so the class name alone can appear in the log of an engine that
  # never built the connector. Byte-level proof stays where it belongs: the
  # populate phase's per-prompt put receipt (kvb_bytes_total{dir="in"} must
  # grow within PUT_WAIT_S or the driver aborts loudly).
  if [[ "$BASELINE_ONLY" == "1" ]]; then
    # No connector requested; assert the inverse — a leaked transfer config
    # here would contaminate the pure-recompute control with store costs.
    if grep -q "Creating v1 connector" "$VLLM_LOG"; then
      log "FATAL: baseline engine constructed a KV connector — the control is contaminated"
      exit 1
    fi
    log "baseline engine confirmed connector-free"
    return 0
  fi
  if ! grep -q "Creating v1 connector with name: KvblockdConnector" "$VLLM_LOG"; then
    log "FATAL: vLLM is healthy but the connector factory never constructed KvblockdConnector — the kv-transfer config was dropped or ignored and the engine is serving with no KV connector"
    log "connector/config lines from $VLLM_LOG:"
    grep -inE "kv_transfer|kv_connector|KVConnector|connector" "$VLLM_LOG" | tail -30 >&2 || true
    exit 1
  fi
  log "native connector confirmed: $(grep -m1 'Creating v1 connector with name: KvblockdConnector' "$VLLM_LOG")"
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

# ---- 8b. BASELINE mode: one boot, cold-only control, then straight to results
if [[ "$BASELINE_ONLY" == "1" ]]; then
  start_vllm "$WORK/vllm-baseline.log" 2400
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | sed 's/^ *//;s/ *$//')"
  log "baseline: lengths=$LENGTHS x $PAIRS pairs, NO connector — pure recompute control"
  rc=0
  python3 "$HERE/run_ttft.py" --phase baseline \
    --vllm "http://127.0.0.1:$VLLM_PORT" \
    --model "$MODEL" --lengths "$LENGTHS" --reps "$REPS" --warmup "$WARMUP" \
    --gen-tokens "$GEN_TOKENS" --out "$OUT_JSONL" \
    --stamp gpu="$GPU_NAME (hf-jobs ${FLAVOR:-flavor-unset})" \
    --stamp vllm="$VLLM_VER" --stamp connector="none (baseline control)" \
    --stamp tc_link="$TC_LINK" --stamp rig="hf-jobs-${FLAVOR:-a10g}" \
    --stamp git_sha="$GIT_SHA" || rc=$?
  log "JSONL at $OUT_JSONL ($(wc -l < "$OUT_JSONL" 2>/dev/null || echo 0) records)"
  [[ $rc -eq 0 ]] || { tail_logs; die "baseline driver exited rc=$rc"; }
  log "DONE (baseline)"
  exit 0
fi

# ---- 9. PHASE 1: populate (vLLM #1 stores every sweep prompt into kvblockd) --
start_vllm "$WORK/vllm-populate.log" 2400   # generous: first boot downloads ~15 GB of weights
log "phase 1/2 populate: lengths=$LENGTHS x $PAIRS pairs; driver polls kvblockd's put counters per prompt and FAILS if they never grow"
python3 "$HERE/run_ttft.py" --phase populate \
  --vllm "http://127.0.0.1:$VLLM_PORT" --metrics "http://$KVBD_METRICS" \
  --model "$MODEL" --lengths "$LENGTHS" --reps "$REPS" --warmup "$WARMUP" \
  --state "$STATE_JSON" --put-wait-s "$PUT_WAIT_S" --drain-s "$DRAIN_S" \
  || {
    # The engine's own log is where the cache library reports which backends it
    # built. Without this, a populate failure says only that no bytes arrived —
    # never whether a remote backend was created at all, which is the single
    # most useful fact when diagnosing a severed store path.
    log "engine log: backend/connector lines"
    grep -inE 'remote|plugin|connector|backend|Traceback|Error' "$WORK/vllm-populate.log" 2>/dev/null | tail -40 >&2 || true
    log "engine log: last 120 lines"
    tail -120 "$WORK/vllm-populate.log" >&2 || true
    die "populate failed — kvblockd never received the blocks, so there is nothing honest to measure (see FATAL(populate) above)"
  }

# ---- 10. RESTART: fresh engine = no local KV anywhere; kvblockd keeps blocks -
log "restarting vLLM: a fresh engine has no local KV, so a warm hit can only be a kvblockd TCP read"
stop_vllm
start_vllm "$WORK/vllm-measure.log" 1200    # weights already on disk

# ---- 11. PHASE 2: measure (warm reps first, then cold — see run_ttft.py) -----
CONNECTOR_VER="$(python3 -c 'import vllm_kvblockd; print(vllm_kvblockd.__version__)')"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | sed 's/^ *//;s/ *$//')"
log "phase 2/2 measure: reps=$REPS(+$WARMUP warmup) gen=$GEN_TOKENS link='$TC_LINK'"
rc=0
python3 "$HERE/run_ttft.py" --phase measure \
  --vllm "http://127.0.0.1:$VLLM_PORT" --metrics "http://$KVBD_METRICS" \
  --model "$MODEL" --gen-tokens "$GEN_TOKENS" \
  --state "$STATE_JSON" --out "$OUT_JSONL" \
  --stamp gpu="$GPU_NAME (hf-jobs ${FLAVOR:-flavor-unset})" \
  --stamp vllm="$VLLM_VER" \
  --stamp connector="vllm_kvblockd $CONNECTOR_VER (native)" \
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
# The retrieval hint must NOT contain the literal marker: an earlier version
# printed the sed command verbatim, the marker matched its own hint line, and
# the results file grew a junk row. Marker spelled with a gap on purpose.
log "DONE — retrieve: hf jobs logs <job_id> | sed -n 's/^.*CHART2''JSONL \({.*\)$/\1/p' > chart2-ttft.jsonl"
