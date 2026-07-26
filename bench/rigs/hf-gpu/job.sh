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
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"          # optional --kv-cache-dtype for vLLM (fp8 / fp8_e4m3 /
                                              # fp8_e5m2; empty = vLLM's auto -> bf16 KV here).
                                              # DISCLOSURE RULE (hard): an fp8 warm arm may never be
                                              # charted against a bf16 cold arm — one env var feeds
                                              # BOTH phases of a run, every JSONL record is stamped
                                              # kv_cache_dtype=..., and plot.py refuses to render a
                                              # chart that mixes dtypes. When set, the operator must
                                              # pass a matching KV_BYTES_PER_TOKEN (fp8 halves it) —
                                              # WARN below if it still looks like the bf16 default.
FP8_PREFLIGHT="${FP8_PREFLIGHT:-}"            # comma list of kv-cache dtypes to PROBE ("1" = fp8):
                                              # per dtype, boot the engine with the flag, one ~1k-token
                                              # prefill, verdict from kvblockd's OWN counters
                                              # (FP8PROBE dtype=... verdict=... lines), then EXIT.
                                              # A probe job, never a measured run.
KVBD_ARENA_BYTES="${KVBD_ARENA_BYTES:-}"      # empty -> derived from LENGTHS x pairs x KV_BYTES_PER_TOKEN
KVBD_STREAMS="${KVBD_STREAMS:-4}"
KVBD_VERIFY="${KVBD_VERIFY:-1}"               # 1 (default) = xxh3-verify every loaded block.
                                              # 0 = DIAGNOSTIC arm only: skips the client-side
                                              # digest to isolate hash cost; labelled in the
                                              # connector stamp, never a headline number.
CONNECTOR_STAGING_GB="${CONNECTOR_STAGING_GB:-2}"   # native-connector pinned staging CAP: the connector
                                              # keeps ONE persistent pinned host slab, grown to the
                                              # largest load up to this cap (bigger loads drain
                                              # through it in cap-sized passes). Plumbed to the
                                              # connector as kvblockd_staging_bytes below and counted
                                              # 1:1 in the RAM fit check.
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
# FP8_PREFLIGHT probes the daemon's counters; BASELINE_ONLY runs with no
# daemon at all. Together they used to silently drop the preflight and run a
# baseline — a paid job doing something other than what was asked. Refuse.
if [[ -n "$FP8_PREFLIGHT" && "$BASELINE_ONLY" == "1" ]]; then
  die "FP8_PREFLIGHT and BASELINE_ONLY=1 are mutually exclusive: the baseline control runs no daemon/connector, so there is nothing for the probe to witness — submit them as two separate jobs"
fi
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

# fp8 halves the KV bytes/token; an unchanged bf16 default silently doubles
# the derived arena (harmless) and falsifies any per-byte arithmetic quoted
# off this run. Loud WARN, not a die: the operator may have sized explicitly.
if [[ "$KV_CACHE_DTYPE" == fp8* && "$KV_BYTES_PER_TOKEN" == "131072" ]]; then
  log "WARN: KV_CACHE_DTYPE=$KV_CACHE_DTYPE but KV_BYTES_PER_TOKEN is still the bf16 default (131072). fp8 KV is half that — pass the fp8 figure (65536 for the Llama-8B class, 28672 for Qwen2.5-7B) or the arena/fit math and any per-byte numbers quoted from this run are ~2x off."
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
# kvblockd_verify maps KVBD_VERIFY onto the connector's existing extra-config
# key (config.py reads it; default true). 0 is a labelled diagnostic arm.
# kvblockd_staging_bytes caps the connector's persistent pinned staging slab
# at CONNECTOR_STAGING_GB — the same number the RAM fit check budgeted above.
KVBD_VERIFY_JSON=true
[[ "$KVBD_VERIFY" == "0" ]] && KVBD_VERIFY_JSON=false
CONNECTOR_STAGING_BYTES=$((CONNECTOR_STAGING_GB * 1073741824))
cat > "$WORK/kv_transfer.json" <<EOF
{"kv_connector": "KvblockdConnector", "kv_role": "kv_both",
 "kv_connector_module_path": "vllm_kvblockd.connector",
 "kv_connector_extra_config": {
   "kvblockd_endpoint": "kvblockd://$KVBD_ADDR",
   "kvblockd_namespace": "vllm",
   "kvblockd_token": "$KVBD_TOKEN",
   "kvblockd_streams": $KVBD_STREAMS,
   "kvblockd_staging_bytes": $CONNECTOR_STAGING_BYTES,
   "kvblockd_store_queue_bytes": ${KVBD_STORE_QUEUE_BYTES:-1073741824},
   "kvblockd_verify": $KVBD_VERIFY_JSON}}
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
  # HF_OVERRIDES: optional JSON merged into the HF model config. Needed for
  # Qwen2.5-*-1M at this vLLM pin: their config ships
  # dual_chunk_attention_config, whose model-code path passes layer_idx to a
  # FlashAttention backend that does not accept it (measured: engine-core
  # TypeError at boot). Dropping the key serves the model-card-sanctioned
  # standard-attention mode (valid <= 262,144 tokens):
  #   HF_OVERRIDES='{"dual_chunk_attention_config": null}'
  # The override is stamped into every JSONL record by the measure/baseline
  # phases (hf_overrides=), so a chart can never hide it.
  HF_OVR_ARGS=()
  if [[ -n "${HF_OVERRIDES:-}" ]]; then HF_OVR_ARGS=(--hf-overrides "$HF_OVERRIDES"); fi
  # ${KV_CACHE_DTYPE:+...}: the flag only exists when the knob is set; both
  # engine boots of a run read the same env var, so the two arms of a charted
  # pair can never disagree on KV dtype (the stamp makes it checkable).
  PYTHONHASHSEED=0 \
  vllm serve "$MODEL" --port "$VLLM_PORT" --dtype bfloat16 \
    ${KV_CACHE_DTYPE:+--kv-cache-dtype "$KV_CACHE_DTYPE"} \
    --max-model-len "$MAX_MODEL_LEN" --max-num-seqs 1 \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --no-enable-prefix-caching \
    --disable-hybrid-kv-cache-manager \
    "${HF_OVR_ARGS[@]}" \
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
    # PROBE_MODE=1 (FP8 preflight): a refused boot is a PROBE VERDICT, not a
    # job failure — soft-fail so the caller can classify it.
    if [[ "${PROBE_MODE:-0}" == "1" ]]; then return 1; fi
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
      if [[ "${PROBE_MODE:-0}" == "1" ]]; then return 1; fi
      exit 1
    fi
    log "baseline engine confirmed connector-free"
    return 0
  fi
  if ! grep -q "Creating v1 connector with name: KvblockdConnector" "$VLLM_LOG"; then
    log "FATAL: vLLM is healthy but the connector factory never constructed KvblockdConnector — the kv-transfer config was dropped or ignored and the engine is serving with no KV connector"
    log "connector/config lines from $VLLM_LOG:"
    grep -inE "kv_transfer|kv_connector|KVConnector|connector" "$VLLM_LOG" | tail -30 >&2 || true
    if [[ "${PROBE_MODE:-0}" == "1" ]]; then return 1; fi
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

# ---- 8a. FP8 PREFLIGHT: probe --kv-cache-dtype support, verdicts, then EXIT --
# Whether fp8 KV actually reaches kvblockd cannot be inferred from a healthy
# boot: the engine may accept the flag and the connector may still serialize
# bf16 (SILENT-FALLBACK), refuse the torch dtype at save time
# (CONNECTOR-DTYPE-UNMAPPED: BlobError "unsupported dtype", connector.py's
# pinned DTYPE_CODES table), or store nothing. The only trustworthy witness is
# kvblockd's own counters: bytes-in per committed block == the blob size, and
# the analytic blob is computed from the model's HF config, never hardcoded
# (blob = 2(K+V) x layers x kv_heads x head_dim x dtype_bytes x 16-token block
# + the connector's 32B prefix; Qwen2.5-7B: bf16 917,536 B, fp8 458,784 B).
# PASS additionally requires the READ path: a second identical prompt must
# grow kvb_hits_total. This is a probe job — it exits before any measurement
# (probe blobs would sit unbudgeted in the arena of a real run).
if [[ -n "$FP8_PREFLIGHT" ]]; then   # BASELINE_ONLY conflict already refused in §1
  DTYPES="$FP8_PREFLIGHT"
  [[ "$DTYPES" == "1" ]] && DTYPES="fp8"
  read -r BF16_BLOB FP8_BLOB <<< "$(python3 - "$MODEL" <<'PY'
import sys

from transformers import AutoConfig

cfg = AutoConfig.from_pretrained(sys.argv[1])
layers = cfg.num_hidden_layers
kvh = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
per_block = 2 * layers * kvh * hd * 16   # K+V elements in one 16-token block
print(per_block * 2 + 32, per_block * 1 + 32)  # bf16 blob, fp8 blob (+32B prefix)
PY
)"
  [[ "$BF16_BLOB" =~ ^[0-9]+$ && "$FP8_BLOB" =~ ^[0-9]+$ ]] \
    || die "FP8 preflight: could not compute analytic blob sizes from the HF config"
  log "FP8 preflight: dtypes [$DTYPES]; analytic blob bytes bf16=$BF16_BLOB fp8=$FP8_BLOB (16-token block + 32B prefix, from the HF config)"

  probe_counters() {  # echoes "put_bytes blocks hits" (same fields run_ttft gates on)
    curl -fsS "http://$KVBD_METRICS/metrics" | awk '
      /^kvb_bytes_total\{[^}]*dir="in"/ {pb += $NF}
      /^kvb_blocks/                     {bl += $NF}
      /^kvb_hits_total/                 {h  += $NF}
      END {printf "%.0f %.0f %.0f\n", pb + 0, bl + 0, h + 0}'
  }
  blob_within_2pct() {  # $1=measured $2=expected
    awk -v b="$1" -v e="$2" 'BEGIN {d = b - e; if (d < 0) d = -d; exit !(d <= e * 0.02)}'
  }

  PROBE_FAIL=0
  _SAVED_KV_DTYPE="$KV_CACHE_DTYPE"
  PROBE_MODE=1
  IFS=',' read -ra _PROBE_DTS <<< "$DTYPES"
  for d in "${_PROBE_DTS[@]}"; do
    d="${d//[[:space:]]/}"
    [[ -n "$d" ]] || continue
    verdict="" blob=0
    KV_CACHE_DTYPE="$d"
    plog="$WORK/vllm-probe-${d//[^A-Za-z0-9]/_}.log"
    if ! start_vllm "$plog" 2400; then
      if grep -qi "unsupported dtype" "$plog"; then
        verdict=CONNECTOR-DTYPE-UNMAPPED
      else
        verdict=ENGINE-REFUSED
      fi
    else
      # ~1k-token prefill via curl. The nonce carries the dtype: a prompt
      # reused across dtype probes would EXISTS-hit the other dtype's blob
      # and muddy both verdicts.
      python3 - "$MODEL" "$d" > "$WORK/fp8probe-req.json" <<'PY'
import json
import sys
import time

prompt = (f"kvstash-fp8probe {sys.argv[2]} {int(time.time())} :: "
          + "The quick brown fox jumps over the lazy dog. " * 110)
print(json.dumps({"model": sys.argv[1], "prompt": prompt, "max_tokens": 1,
                  "temperature": 0.0}))
PY
      read -r pb0 bl0 h0 <<< "$(probe_counters)"
      curl -fsS -m 300 -H 'Content-Type: application/json' \
        -d @"$WORK/fp8probe-req.json" -o /dev/null \
        "http://127.0.0.1:$VLLM_PORT/v1/completions" \
        || log "FP8 probe: prefill request failed (dtype=$d) — verdict from the counters"
      pb1=$pb0 bl1=$bl0 h1=$h0
      for _ in $(seq 1 60); do
        read -r pb1 bl1 h1 <<< "$(probe_counters)"
        (( pb1 > pb0 && bl1 > bl0 )) && break
        sleep 1
      done
      if (( pb1 <= pb0 || bl1 <= bl0 )); then
        if grep -qi "unsupported dtype" "$plog"; then
          verdict=CONNECTOR-DTYPE-UNMAPPED   # engine booted; connector refused at save
        else
          verdict=NO-BYTES
        fi
      else
        blob=$(( (pb1 - pb0) / (bl1 - bl0) ))
        if blob_within_2pct "$blob" "$BF16_BLOB"; then
          verdict=SILENT-FALLBACK            # flag accepted, bytes still bf16-sized
        elif blob_within_2pct "$blob" "$FP8_BLOB"; then
          # write path is fp8-sized; PASS still needs the read path — the
          # SECOND identical prompt must grow kvblockd's hit counter.
          curl -fsS -m 300 -H 'Content-Type: application/json' \
            -d @"$WORK/fp8probe-req.json" -o /dev/null \
            "http://127.0.0.1:$VLLM_PORT/v1/completions" || true
          h2=$h1
          for _ in $(seq 1 30); do
            read -r _ _ h2 <<< "$(probe_counters)"
            (( h2 > h1 )) && break
            sleep 1
          done
          if (( h2 > h1 )); then
            verdict=PASS
          else
            verdict=FP8-STORED-NO-HITS       # stored fp8-sized blobs, reload never hit
          fi
        else
          verdict=UNEXPECTED-BLOB-SIZE       # neither analytic size: investigate before use
        fi
      fi
    fi
    stop_vllm
    log "FP8 probe detail: dtype=$d blob_bytes=$blob expected_bf16=$BF16_BLOB expected_fp8=$FP8_BLOB engine_log=$plog"
    echo "FP8PROBE dtype=$d verdict=$verdict"
    [[ "$verdict" == "PASS" ]] || PROBE_FAIL=4
  done
  KV_CACHE_DTYPE="$_SAVED_KV_DTYPE"
  PROBE_MODE=0
  log "FP8 preflight complete (exit $PROBE_FAIL; 0 = every probed dtype PASSed). Probe job only — no measured run. For the real sweep submit with KV_CACHE_DTYPE=<passing dtype> and a matching KV_BYTES_PER_TOKEN; the disclosure rule keeps both arms of any charted pair on that same dtype."
  exit $PROBE_FAIL
fi

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
    --stamp kv_cache_dtype="${KV_CACHE_DTYPE:-auto-bf16}" \
    ${HF_OVERRIDES:+--stamp hf_overrides="$HF_OVERRIDES"} \
    --stamp git_sha="$GIT_SHA" || rc=$?
  log "JSONL at $OUT_JSONL ($(wc -l < "$OUT_JSONL" 2>/dev/null || echo 0) records)"
  [[ $rc -eq 0 ]] || { tail_logs; die "baseline driver exited rc=$rc"; }
  log "DONE (baseline)"
  exit 0
fi

# ---- 9. PHASE 1: populate (vLLM #1 stores every sweep prompt into kvblockd) --
start_vllm "$WORK/vllm-populate.log" 2400   # generous: first boot downloads ~15 GB of weights
log "phase 1/2 populate: lengths=$LENGTHS x $PAIRS pairs; driver polls kvblockd's put counters per prompt and FAILS if they never grow"
# --vllm-log: after the put counters quiesce the driver greps the engine log
# for the connector's write-behind disclosure and ABORTS on a nonzero
# `dropped=` — a lossy populate silently shrinks the warm set.
python3 "$HERE/run_ttft.py" --phase populate \
  --vllm "http://127.0.0.1:$VLLM_PORT" --metrics "http://$KVBD_METRICS" \
  --model "$MODEL" --lengths "$LENGTHS" --reps "$REPS" --warmup "$WARMUP" \
  --state "$STATE_JSON" --put-wait-s "$PUT_WAIT_S" --drain-s "$DRAIN_S" \
  --vllm-log "$WORK/vllm-populate.log" \
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
# The connector's shutdown flush emits its final store-queue disclosure only
# now, after the engine exits — re-check it: blocks dropped or FAILED AT
# SHUTDOWN (flush timeout / dead wire) escaped the driver's in-run grep but
# still shrink the warm set. The line is emitted UNCONDITIONALLY (zeros
# included) by connector.shutdown() — WHEN the engine calls it. Measured
# 2026-07-27 (session-1, 2 of 3 runs): vLLM v0.25's teardown sometimes
# force-kills the engine core before the connector hook runs, so the line's
# ABSENCE is a property of the engine's exit path, not of our data. Data
# safety at this point rests on gates that already passed: the per-prompt
# put receipts, the quiesce (counters stable => the async queue had fully
# drained BEFORE the engine was stopped), and — authoritatively — the
# measure phase's exact-count hit gate, which fails any warm rep whose
# blocks are missing. Absent line => WARN and continue; a PRESENT line with
# nonzero dropped/failed stays a hard failure.
DISCLOSURE_FINAL="$(grep -o 'kvblockd store queue: dropped=[0-9]* failed=[0-9]*' "$WORK/vllm-populate.log" 2>/dev/null | tail -1 || true)"
if [[ -z "$DISCLOSURE_FINAL" ]]; then
  log "WARN: shutdown disclosure line absent (engine force-killed before connector.shutdown() — a known vLLM teardown race). Quiesce already proved the store queue drained; the measure phase's exact-count hit gate remains the authority on warm-set completeness."
fi
if [[ -n "$DISCLOSURE_FINAL" ]]; then
  DROPPED_FINAL="$(printf '%s' "$DISCLOSURE_FINAL" | sed -n 's/.*dropped=\([0-9]*\).*/\1/p')"
  FAILED_FINAL="$(printf '%s' "$DISCLOSURE_FINAL" | sed -n 's/.*failed=\([0-9]*\).*/\1/p')"
  if [[ "$DROPPED_FINAL" != "0" ]]; then
    die "the connector's shutdown disclosure reports dropped=$DROPPED_FINAL store blocks — the populated warm set is silently incomplete (raise kvblockd_store_flush_timeout_s / kvblockd_store_queue_bytes and rerun)"
  fi
  if [[ "$FAILED_FINAL" != "0" ]]; then
    die "the connector's shutdown disclosure reports failed=$FAILED_FINAL store puts — blocks that errored on the wire and never reached kvblockd shrink the warm set exactly like drops (check daemon health and rerun)"
  fi
  log "populate store-queue disclosure: dropped=$DROPPED_FINAL failed=$FAILED_FINAL"
fi
start_vllm "$WORK/vllm-measure.log" 1200    # weights already on disk

# ---- 11. PHASE 2: measure (warm reps first, then cold — see run_ttft.py) -----
CONNECTOR_VER="$(python3 -c 'import vllm_kvblockd; print(vllm_kvblockd.__version__)')"
CONNECTOR_STAMP="vllm_kvblockd $CONNECTOR_VER (native)"
[[ "$KVBD_VERIFY" == "0" ]] && CONNECTOR_STAMP="$CONNECTOR_STAMP verify=off (DIAGNOSTIC arm)"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | sed 's/^ *//;s/ *$//')"
log "phase 2/2 measure: reps=$REPS(+$WARMUP warmup) gen=$GEN_TOKENS link='$TC_LINK'"
# --block-size 16: vLLM's CUDA default; the driver verifies each warm rep's
# kvb_hits_total delta EQUALS the expected block count (partial hits flagged).
# --vllm-log: after the warm pass the driver greps the engine log for the
# connector's 'kvblockd load path:' line and appends path=... to the
# connector stamp, so every JSONL record names the path that produced it.
rc=0
python3 "$HERE/run_ttft.py" --phase measure \
  --vllm "http://127.0.0.1:$VLLM_PORT" --metrics "http://$KVBD_METRICS" \
  --model "$MODEL" --gen-tokens "$GEN_TOKENS" --block-size 16 \
  --state "$STATE_JSON" --out "$OUT_JSONL" --vllm-log "$WORK/vllm-measure.log" \
  --stamp gpu="$GPU_NAME (hf-jobs ${FLAVOR:-flavor-unset})" \
  --stamp vllm="$VLLM_VER" \
  --stamp connector="$CONNECTOR_STAMP" \
  --stamp tc_link="$TC_LINK" --stamp rig="hf-jobs-${FLAVOR:-a10g}" \
  --stamp kv_cache_dtype="${KV_CACHE_DTYPE:-auto-bf16}" \
  ${HF_OVERRIDES:+--stamp hf_overrides="$HF_OVERRIDES"} \
  --stamp git_sha="$GIT_SHA" || rc=$?

# Path attribution receipt (the driver already stamped it into the JSONL):
# on a CUDA run the fast path is chunked-slab — per-block means the pinned
# slab or the GPU scratch never came up, so the warm numbers measured the
# slow lane. Loud WARN, never a failure: the numbers are honest, just slower.
LOAD_PATH="$(grep -h -o 'kvblockd load path: [a-z-]*' "$WORK"/vllm-populate.log "$WORK"/vllm-measure.log 2>/dev/null | tail -1 | awk '{print $NF}')"
if [[ -n "$LOAD_PATH" ]]; then
  log "connector load path: $LOAD_PATH (JSONL connector stamp carries path=$LOAD_PATH)"
  if [[ "$LOAD_PATH" != "chunked-slab" ]]; then
    log "WARN WARN WARN: warm loads ran on the '$LOAD_PATH' path on a CUDA run — the pinned-slab chunked scatter did NOT serve these numbers (pin/scratch alloc failed or the paged layout was not viewable). Attribute the results to path=$LOAD_PATH."
  fi
else
  log "WARN: no 'kvblockd load path:' line found in the engine logs — load path unattributed"
fi

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
