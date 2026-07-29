#!/usr/bin/env bash
# kvbench GPU-node setup (laptop-side; drives the node over SSH).
# Asserts the GPU/driver, waits out the backgrounded image pull, downloads
# the models onto instance NVMe, DCA-strips the Qwen-1M snapshot (the
# model-card-sanctioned standard-attention mode, disclosed per CLAIMS PR-6),
# and clones this repo at the CURRENT laptop HEAD sha.
#
#   bench/rigs/aws-gpu/node-setup.sh            # full setup
#   SKIP_LLAMA=1 ... node-setup.sh              # Qwen-only day
#
# HF_TOKEN: only the Llama download needs it; it is piped through the SSH
# session's stdin per invocation and never written to any file or user-data.
set -euo pipefail
log() { printf '[setup %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
die() { log "FATAL: $*"; exit 1; }

export AWS_PROFILE="${AWS_PROFILE:-kvbench}"
STATE_DIR="${STATE_DIR:-$HOME/kvbench-dday}"
IP=$(cat "$STATE_DIR/gpu-ip") || die "no gpu-ip in $STATE_DIR — run provision.sh first"
SSH_KEY="${SSH_KEY:-$STATE_DIR/kvbench.pem}"
SSH=(ssh -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "ubuntu@$IP")
GIT_SHA=$(git -C "$(dirname "${BASH_SOURCE[0]}")/../../.." rev-parse HEAD)

log "waiting for SSH on $IP"
for i in $(seq 1 30); do "${SSH[@]}" true 2>/dev/null && break; sleep 10; [[ $i == 30 ]] && die "SSH never came up"; done

log "GPU + docker-root assertions"
"${SSH[@]}" bash -s <<'REMOTE'
set -euo pipefail
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q L40S || { echo "NOT AN L40S"; exit 1; }
sudo docker info 2>/dev/null | grep -i 'docker root dir' | grep -q /opt/dlami/nvme/docker || { echo "docker root not on NVMe"; exit 1; }
# Wait out the user-data pull (backgrounded at boot):
for i in $(seq 1 60); do
  sudo docker image inspect vllm/vllm-openai@sha256:f0b9a0dc75a9fca3b6811e3279367b2d6a448055a000bfd13859587d74cef268 >/dev/null 2>&1 && { echo "image present"; break; }
  sleep 20; [[ $i == 60 ]] && { echo "image pull never finished"; tail -5 /opt/dlami/nvme/work/pull.log; exit 1; }
done
# GPU visible from INSIDE docker (a clobbered nvidia runtime config would
# pass nvidia-smi on the host and still fail every arm):
sudo docker run --rm --gpus all --entrypoint nvidia-smi \
  vllm/vllm-openai@sha256:f0b9a0dc75a9fca3b6811e3279367b2d6a448055a000bfd13859587d74cef268 \
  --query-gpu=name --format=csv,noheader | grep -q L40S \
  || { echo "GPU not visible inside docker"; exit 1; }
REMOTE

log "Qwen2.5-7B-Instruct-1M download + DCA strip (ungated, no token)"
"${SSH[@]}" bash -s <<'REMOTE'
set -euo pipefail
export HF_HUB_ENABLE_HF_TRANSFER=1 HF_HOME=/opt/dlami/nvme/hf
pip install -q -U "huggingface_hub[hf_transfer]" 2>/dev/null || pip install -q -U huggingface_hub
python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-7B-Instruct-1M', local_dir='/opt/dlami/nvme/hf/qwen1m')
PY
python3 - <<'PY'
import json
p = '/opt/dlami/nvme/hf/qwen1m/config.json'
c = json.load(open(p))
removed = c.pop('dual_chunk_attention_config', None)
json.dump(c, open(p, 'w'), indent=2)
print('dual_chunk_attention_config removed:', removed is not None)
PY
REMOTE

if [[ "${SKIP_LLAMA:-0}" != "1" ]]; then
  log "Llama-3.1-8B-Instruct download (gated; token via stdin only)"
  [[ -n "${HF_TOKEN:-}" ]] || die "export HF_TOKEN for the Llama download (or SKIP_LLAMA=1)"
  # The script rides the ssh ARGUMENT so stdin carries ONLY the token — a
  # heredoc would replace the pipe and read would swallow script text
  # instead. TOK=$(cat) is newline-agnostic on purpose.
  printf '%s\n' "$HF_TOKEN" | "${SSH[@]}" '
    set -euo pipefail
    TOK=$(cat)
    export HF_HUB_ENABLE_HF_TRANSFER=1 HF_HOME=/opt/dlami/nvme/hf HF_TOKEN="$TOK"
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download(\"meta-llama/Llama-3.1-8B-Instruct\", local_dir=\"/opt/dlami/nvme/hf/llama31\")"
  '
fi

log "repo clone at $GIT_SHA"
"${SSH[@]}" bash -s <<REMOTE
set -euo pipefail
rm -rf /opt/dlami/nvme/work/KVStash
git clone --quiet https://github.com/rudraymehra/KVStash.git /opt/dlami/nvme/work/KVStash
git -C /opt/dlami/nvme/work/KVStash checkout --quiet "$GIT_SHA"
echo "repo at \$(git -C /opt/dlami/nvme/work/KVStash rev-parse --short HEAD)"
REMOTE

log "node ready — run arms with bench/rigs/aws-gpu/run-arm.sh"
log "REMINDER: re-arm the dead-man for the session window:"
log "  ssh ubuntu@$IP 'sudo shutdown -c && sudo shutdown -h +540'"
