#!/usr/bin/env bash
# Local, FREE vLLM->kvblockd e2e in Docker (CPU only, no GPU, no cloud).
#
# Proven 2026-07-25 on an Apple M2 / Docker Desktop (3.8 GiB VM): a real vLLM
# prefill moved real KV bytes into kvblockd over TCP
# (kvb_bytes_total{dir="in"} 0 -> 4718624) and a fresh engine re-read them
# after a restart (property d) plus the mixed local+remote leg (round C).
#
# WHY THE NATIVE CONNECTOR AND NOT LMCACHE ON CPU (mode `lmcache-probe`):
#   lmcache 0.5.1 (and 0.5.2, latest checked) has no CPU GPU-connector:
#   lmcache/v1/gpu_connector/__init__.py dispatches on cuda/xpu/musa/hpu and
#   raises `RuntimeError: No supported cpu connector found.` on the vLLM CPU
#   backend. LMCacheManager catches it and "operate[s] in degraded mode
#   (recompute)" — storage backends are NEVER created, so zero bytes can ever
#   reach any remote backend, ours or anyone's. Our config is NOT the problem:
#   the same probe shows every `lmcache.`-prefixed key (including the nested
#   `lmcache.extra_config` dict) applied on both the scheduler and worker
#   processes ("Updated config remote_storage_plugins from vLLM extra
#   config", "Overridden extra_config"). `lmcache-probe` pins that fact: it
#   PASSES while the degraded-mode signature is present and FAILS the day an
#   lmcache release grows CPU support (then the LMCache CPU leg becomes
#   testable and e2e-cpu.yml can be revived).
#
# WHY DOCKER AND NOT `pip install vllm` (the e2e-cpu.yml failure):
#   the PyPI vllm wheel is the CUDA build; on any CPU-only box it dies at CLI
#   parse time with "Failed to infer device type" (this is why the e2e-cpu CI
#   serve leg never went green). The real CPU builds live in the
#   vllm/vllm-openai-cpu images (arm64 + x86_64) and the `+cpu` wheels on the
#   GitHub release page. x86_64 NOTE: the CPU kernels use AVX — the x86 image
#   SIGILLs under Apple-silicon Rosetta emulation; on ARM Macs this script
#   uses the native arm64 image.
#
# Usage:
#   bench/e2e/cpu/local-docker.sh                 # native connector e2e (default)
#   bench/e2e/cpu/local-docker.sh lmcache-probe   # pin the lmcache-CPU impossibility
#   bench/e2e/cpu/local-docker.sh ttft-rehearsal  # the Chart-2 GPU job's exact
#                                 populate->restart->measure flow, driven by
#                                 bench/rigs/hf-gpu/run_ttft.py itself — the
#                                 free gate before any paid GPU submission
#   bench/e2e/cpu/local-docker.sh equivalence     # output-equivalence suite
#                                 (bench/e2e/equivalence.py): greedy-decode a
#                                 prompt set on a fresh engine, RESTART, replay
#                                 the exact prompts warm from kvblockd — every
#                                 token sequence must match, incl. prompts at
#                                 block-boundary lengths (kB-1 / kB / kB+1)
#   KVB_E2E_KEEP=1 ... to keep the container around for debugging.
#
# Requires: Docker with >= 3.5 GiB VM memory, Go toolchain, network (first run
# pulls the image, ~a few GiB, and downloads facebook/opt-125m into a named
# volume; both are reused afterwards).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
source "$HERE/versions.env"   # VLLM_VERSION / LMCACHE_VERSION / MODEL pins

MODE="${1:-native}"
CTR="${KVB_E2E_CTR:-kvb-cpu-e2e}"
HF_VOL="${KVB_E2E_HF_VOL:-kvb-hf-cache}"

# The bare tag is arm64; x86_64 has its own suffix (checked on Docker Hub).
DOCKER_ARCH="$(docker version --format '{{.Server.Arch}}')"
case "$DOCKER_ARCH" in
  arm64) IMG="vllm/vllm-openai-cpu:v${VLLM_VERSION}-arm64"; GOA=arm64 ;;
  amd64) IMG="vllm/vllm-openai-cpu:v${VLLM_VERSION}-x86_64"; GOA=amd64 ;;
  *) echo "unsupported docker arch: $DOCKER_ARCH" >&2; exit 2 ;;
esac
echo "[local-docker] image $IMG (docker arch $DOCKER_ARCH)"

cleanup() {
  if [[ "${KVB_E2E_KEEP:-0}" != "1" ]]; then docker rm -f "$CTR" >/dev/null 2>&1 || true; fi
}
trap cleanup EXIT

# ---- 1. kvblockd cross-compiled for the container ---------------------------
BIN="$(mktemp -d)/kvblockd"
echo "[local-docker] building kvblockd (linux/$GOA)"
(cd "$ROOT" && GOOS=linux GOARCH="$GOA" CGO_ENABLED=0 go build -o "$BIN" ./cmd/kvblockd)

# ---- 2. container: everything (daemon + vLLM) shares one netns --------------
docker rm -f "$CTR" >/dev/null 2>&1 || true
docker run -d --name "$CTR" \
  -v "$ROOT":/repo \
  -v "$HF_VOL":/root/.cache/huggingface \
  --entrypoint bash "$IMG" -c 'sleep infinity' >/dev/null
docker cp "$BIN" "$CTR":/work/kvblockd 2>/dev/null || {
  docker exec "$CTR" mkdir -p /work; docker cp "$BIN" "$CTR":/work/kvblockd; }

# Sized for a ~3.8 GiB Docker VM: 256 MiB prefaulted arena, 384 MiB vLLM KV
# cache, 0.4 GiB LMCache CPU buffer (probe mode only). opt-125m KV is
# ~36 KiB/token, so the 161-token verify prompt fits many times over.
docker exec "$CTR" bash -c 'cat > /work/namespaces.yaml <<EOF
namespaces:
  - { name: lmcache, id: 1, token: e2e-token }
  - { name: vllm, id: 2, token: e2e-token }
EOF
cat > /work/kvblockd.yaml <<EOF
listen_addr: "127.0.0.1:9440"
metrics_addr: "127.0.0.1:9442"
dram_arena_bytes: 268435456
pinned_bytes_cap: 67108864
eviction_policy: "s3fifo"
namespaces_path: "/work/namespaces.yaml"
EOF
cat > /work/kv_transfer_native.json <<EOF
{"kv_connector": "KvblockdConnector", "kv_role": "kv_both",
 "kv_connector_module_path": "vllm_kvblockd.connector",
 "kv_connector_extra_config": {
   "kvblockd_endpoint": "kvblockd://127.0.0.1:9440",
   "kvblockd_namespace": "vllm",
   "kvblockd_token": "e2e-token",
   "kvblockd_streams": 4}}
EOF
# probe-only: read by mode lmcache-probe, generated unconditionally for simplicity
cat > /work/kv_transfer_lmcache.json <<EOF
{"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both",
 "kv_connector_extra_config": {
   "lmcache.remote_storage_plugins": "kvblockd",
   "lmcache.local_cpu": true,
   "lmcache.max_local_cpu_size": 0.4,
   "lmcache.chunk_size": 256,
   "lmcache.extra_config": {
     "kvblockd_token": "e2e-token",
     "remote_storage_plugin.kvblockd.module_path": "lmcache_kvblockd.adapter",
     "remote_storage_plugin.kvblockd.class_name": "KvblockdConnectorAdapter",
     "remote_storage_plugin.kvblockd.url": "kvblockd://127.0.0.1:9440?namespace=lmcache&streams=4"
   }}}
EOF
python3 -c "import json; [json.load(open(f)) for f in (\"/work/kv_transfer_native.json\",\"/work/kv_transfer_lmcache.json\")]"'

# ---- 3. python stack (image already has the CPU vllm + torch) ---------------
echo "[local-docker] installing python packages (mode $MODE)"
case "$MODE" in
  native|ttft-rehearsal|equivalence)
    docker exec "$CTR" bash -c 'pip install -q /repo/python/kvblockd /repo/python/vllm_kvblockd' ;;
  lmcache-probe)
    # lmcache has no aarch64 wheel: sdist build (NO_GPU_EXT skips CUDA ext).
    docker exec "$CTR" bash -c "NO_GPU_EXT=1 pip install -q 'lmcache==$LMCACHE_VERSION' \
      && pip install -q /repo/python/kvblockd /repo/python/lmcache_kvblockd" ;;
  *) echo "usage: $0 [native|lmcache-probe|ttft-rehearsal|equivalence]" >&2; exit 2 ;;
esac
docker exec "$CTR" bash -c 'python3 -c "import vllm; assert vllm.__version__ == \"'"$VLLM_VERSION"'\", vllm.__version__; print(\"vllm\", vllm.__version__, \"OK\")"'

# ---- 4. daemon ---------------------------------------------------------------
docker exec -d "$CTR" bash -c '/work/kvblockd -config /work/kvblockd.yaml > /work/kvbd.log 2>&1'
docker exec "$CTR" bash -c 'for i in $(seq 1 60); do
  curl -fsS -o /dev/null http://127.0.0.1:9442/healthz 2>/dev/null && exit 0; sleep 1; done
  echo "kvblockd never became healthy" >&2; cat /work/kvbd.log >&2; exit 1'
echo "[local-docker] kvblockd healthy"

serve() {  # $1 = kv_transfer json, $2 = log, $3 = prefix-caching flag (whole flag, unquoted on purpose)
  # LMCACHE_LOG_LEVEL is probe-only (lmcache isn't installed in the other
  # modes); harmless where unread.
  docker exec -d "$CTR" bash -c "cd /work && PYTHONHASHSEED=0 OMP_NUM_THREADS=2 LMCACHE_LOG_LEVEL=INFO \
    vllm serve \"$MODEL\" --port 18000 --dtype bfloat16 --max-model-len 2048 \
    --max-num-seqs 1 $3 --enforce-eager --disable-hybrid-kv-cache-manager \
    --kv-cache-memory-bytes 402653184 \
    --kv-transfer-config \"\$(cat $1)\" > $2 2>&1"
  docker exec "$CTR" bash -c 'for i in $(seq 1 180); do
    curl -fsS -o /dev/null http://127.0.0.1:18000/health 2>/dev/null && exit 0
    pgrep -f "vllm serv[e]" >/dev/null || { echo "vLLM died during boot:" >&2; tail -40 '"$2"' >&2; exit 1; }
    sleep 5
  done; echo "vLLM never became healthy" >&2; tail -40 '"$2"' >&2; exit 1'
}

# Wait for the process to actually die (SIGTERM is graceful and can outlive a
# fixed sleep): a still-running old engine would answer the next health check
# on :18000 while the new serve dies on the bind — the restart the rehearsal
# exists to prove would be silently skipped, with a green PASS.
stop_vllm() {
  docker exec "$CTR" bash -c 'pkill -f "vllm serv[e]" 2>/dev/null
    for _ in $(seq 1 60); do
      pgrep -f "vllm serv[e]" >/dev/null || exit 0
      sleep 1
    done
    pkill -9 -f "vllm serv[e]" 2>/dev/null; sleep 2
    ! pgrep -f "vllm serv[e]" >/dev/null' \
    || { echo "stop_vllm: old engine would not die" >&2; exit 1; }
}

if [[ "$MODE" == "lmcache-probe" ]]; then
  # ---- the pinned impossibility: capture the server log and assert it -------
  serve /work/kv_transfer_lmcache.json /work/vllm-lmcache.log --no-enable-prefix-caching
  docker exec "$CTR" bash -c '
    ok=1
    grep -q "Updated config remote_storage_plugins from vLLM extra config" /work/vllm-lmcache.log \
      || { echo "PROBE FAIL: extra-config keys did not reach LMCache" >&2; ok=0; }
    if grep -q "Created remote backend for plugin: kvblockd" /work/vllm-lmcache.log; then
      echo "PROBE TRIPPED: LMCache created the kvblockd remote backend on CPU!" >&2
      echo "lmcache gained CPU support — the LMCache CPU e2e leg is now testable; revisit e2e-cpu.yml." >&2
      ok=0
    elif ! grep -q "No supported cpu connector found" /work/vllm-lmcache.log; then
      echo "PROBE FAIL: neither the degraded-mode signature nor a remote backend found — investigate:" >&2
      grep -iE "lmcache (error|warning)|Failed to" /work/vllm-lmcache.log | tail -20 >&2
      ok=0
    fi
    [[ $ok == 1 ]] || exit 1
    echo "lmcache-probe: config reaches LMCache on both roles; CPU path still degraded-by-design"
    grep -m1 "No supported cpu connector found" /work/vllm-lmcache.log'
  echo "[local-docker] PASS (lmcache-probe)"
  exit 0
fi

if [[ "$MODE" == "ttft-rehearsal" ]]; then
  # ---- dress rehearsal of the EXACT Chart-2 GPU flow (bench/rigs/hf-gpu) ----
  # selftest -> populate (per-prompt put receipt) -> engine RESTART ->
  # measure (per-rep kvb_hits_total verification). rc=0 from measure means
  # every honesty gate held against a REAL engine — proven for free before a
  # paid GPU submission. The TTFT values themselves are NOT benchmarks here
  # (CPU backend, tiny opt-125m lengths); only the gates are the deliverable.
  #
  # LENGTHS must clear the CPU backend's block size with room: vLLM-on-CPU
  # uses 128-token blocks (GPU uses 16) and the connector stores only the
  # COMPLETE blocks of the prompt minus its last token
  # (align_to_block_size: (n-1)//bs*bs — a 99-token prompt stores NOTHING).
  # The first rehearsal ran 96,160 and populate's receipt gate correctly
  # aborted the run — that failure is the gate working, kept here as the
  # reason these lengths are 320,512.
  docker exec "$CTR" bash -c 'cd /repo && PYTHONHASHSEED=0 python3 bench/rigs/hf-gpu/run_ttft.py --selftest'
  echo "[local-docker] rehearsal phase 1: populate"
  serve /work/kv_transfer_native.json /work/vllm-rehearse1.log --no-enable-prefix-caching
  # The paid rig gates on the factory's construction line after /health
  # (job.sh); assert the same line here so a wording drift in a vLLM bump
  # fails this FREE run, never the paid one.
  docker exec "$CTR" bash -c 'grep -q "Creating v1 connector with name: KvblockdConnector" /work/vllm-rehearse1.log \
    || { echo "FATAL: factory construction line missing — job.sh fast-fail gate would break on this vLLM" >&2
         grep -inE "kv_connector|KVConnector" /work/vllm-rehearse1.log | tail -10 >&2; exit 1; }'
  docker exec "$CTR" bash -c "cd /repo && python3 bench/rigs/hf-gpu/run_ttft.py --phase populate \
    --vllm http://127.0.0.1:18000 --metrics http://127.0.0.1:9442 \
    --model \"$MODEL\" --lengths 320,512 --reps 1 --warmup 1 \
    --state /work/rehearsal-state.json --put-wait-s 60 --drain-s 2"
  echo "[local-docker] rehearsal restart: fresh engine, kvblockd keeps the blocks"
  stop_vllm
  serve /work/kv_transfer_native.json /work/vllm-rehearse2.log --no-enable-prefix-caching
  echo "[local-docker] rehearsal phase 2: measure (warm reps must grow kvb_hits_total)"
  docker exec "$CTR" bash -c "cd /repo && python3 bench/rigs/hf-gpu/run_ttft.py --phase measure \
    --vllm http://127.0.0.1:18000 --metrics http://127.0.0.1:9442 \
    --model \"$MODEL\" --gen-tokens 8 \
    --state /work/rehearsal-state.json --out /work/rehearsal.jsonl \
    --stamp gpu=cpu-rehearsal-not-a-benchmark --stamp rig=local-docker \
    --stamp connector=vllm_kvblockd-native --stamp tc_link=loopback"
  echo "[local-docker] PASS (ttft-rehearsal): populate->restart->measure held on the native connector"
  exit 0
fi

if [[ "$MODE" == "equivalence" ]]; then
  # ---- output-equivalence, end to end and FREE (bench/e2e/equivalence.py) ---
  # record (fresh engine, greedy, per-prompt store receipt) -> engine RESTART
  # -> compare (exact prompts, warm from kvblockd, per-prompt block-exact hit
  # attribution, token-sequence equality gated at 100%). vLLM-on-CPU uses
  # 128-token KV blocks, so the boundary trios sit at 255/256/257 and
  # 383/384/385 tokens (the connector stores only the complete blocks of the
  # first n-1 tokens — these straddle the store/recompute edge); 320 is the
  # plain mid-block cell and 512 the exactly-block-aligned one (4*128 — a
  # fourth boundary length, not mid-block). Greedy decode: temperature=0,
  # seed pinned by
  # the driver, --max-num-seqs 1 in serve() — determinism is the premise, so
  # any token mismatch is a store-path finding, not sampling noise.
  docker exec "$CTR" bash -c 'cd /repo && python3 bench/e2e/equivalence.py --selftest'
  echo "[local-docker] equivalence phase 1: record (fresh engine, connector on)"
  serve /work/kv_transfer_native.json /work/vllm-equiv1.log --no-enable-prefix-caching
  docker exec "$CTR" bash -c "cd /repo && python3 bench/e2e/equivalence.py --phase record \
    --vllm http://127.0.0.1:18000 --metrics http://127.0.0.1:9442 \
    --model \"$MODEL\" --n 20 --gen-tokens 16 --seed 0 \
    --block-size 128 --boundary-multiples 2,3 --lengths 320,512 \
    --state /work/equiv-state.json --put-wait-s 60"
  echo "[local-docker] equivalence restart: fresh engine, kvblockd keeps the blocks"
  stop_vllm
  serve /work/kv_transfer_native.json /work/vllm-equiv2.log --no-enable-prefix-caching
  echo "[local-docker] equivalence phase 2: compare (warm replay must be token-identical)"
  docker exec "$CTR" bash -c "cd /repo && python3 bench/e2e/equivalence.py --phase compare \
    --vllm http://127.0.0.1:18000 --metrics http://127.0.0.1:9442 \
    --model \"$MODEL\" --min-match 100 \
    --state /work/equiv-state.json --out /work/equivalence.jsonl \
    --stamp rig=local-docker --stamp connector=vllm_kvblockd-native \
    --stamp tc_link=loopback --stamp isolation=vllm-restart"
  # ^ isolation is stamped by THIS harness because this harness performed the
  #   restart (stop_vllm + fresh serve above); the driver itself cannot see
  #   that and defaults the summary to isolation=unverified.
  echo "[local-docker] PASS (equivalence): warm kvblockd reload is token-identical to recompute"
  exit 0
fi

# ---- 5. native e2e: rounds A (bytes in + hits), B (restart), C (mixed) ------
echo "[local-docker] round A: cold engine"
serve /work/kv_transfer_native.json /work/vllm-native.log --no-enable-prefix-caching
docker exec "$CTR" bash -c 'cd /repo && python3 bench/e2e/cpu/verify.py \
  --vllm http://127.0.0.1:18000 --metrics http://127.0.0.1:9442 --save-ref /work/native-ref.json'
# The load-bearing assertion of this rig: real KV bytes arrived over TCP.
docker exec "$CTR" bash -c 'in=$(curl -fsS http://127.0.0.1:9442/metrics \
    | awk -F" " "/^kvb_bytes_total\\{dir=\\\"in\\\"/ {s+=\$2} END {print s+0}")
  echo "kvb_bytes_total{dir=in} = $in"
  awk -v v="$in" "BEGIN{exit !(v>0)}" || { echo "FAIL: kvblockd received zero bytes" >&2; exit 1; }'

echo "[local-docker] round B: restart vLLM — hits must survive (property d)"
stop_vllm
serve /work/kv_transfer_native.json /work/vllm-native2.log --no-enable-prefix-caching
docker exec "$CTR" bash -c 'cd /repo && python3 bench/e2e/cpu/verify.py \
  --vllm http://127.0.0.1:18000 --metrics http://127.0.0.1:9442 --expect-hits'

echo "[local-docker] round C: prefix caching ON — mixed local+remote load"
stop_vllm
serve /work/kv_transfer_native.json /work/vllm-native3.log --enable-prefix-caching
docker exec "$CTR" bash -c 'cd /repo && python3 bench/e2e/cpu/verify.py \
  --vllm http://127.0.0.1:18000 --metrics http://127.0.0.1:9442 --prefix-mixed /work/native-ref.json'

echo "[local-docker] PASS: vLLM (CPU, Docker) moved real KV bytes through kvblockd"
docker exec "$CTR" bash -c 'curl -fsS http://127.0.0.1:9442/metrics | grep -E "^kvb_(blocks|bytes_total|hits_total)"'
