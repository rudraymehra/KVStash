#!/bin/bash
# kvbench GPU-node user-data: arm the dead-man FIRST, then move Docker's
# data-root onto the instance NVMe and background the (slow) engine-image
# pull so it overlaps the operator's SSH window.
#
# The dead-man is the bring-up window only (150 min). Extending it is a
# CONSCIOUS act after bring-up: `sudo shutdown -c && sudo shutdown -h +540`.
# With --instance-initiated-shutdown-behavior=terminate this halt IS a
# terminate — the box cannot outlive an operator who walked away.
set -eu
shutdown -h +150 "kvbench dead-man (bring-up window)"
mountpoint -q /opt/dlami/nvme || exit 1   # the DLAMI mounts instance NVMe here; no mount = wrong image
mkdir -p /opt/dlami/nvme/docker /opt/dlami/nvme/hf /opt/dlami/nvme/work
chown -R ubuntu:ubuntu /opt/dlami/nvme/hf /opt/dlami/nvme/work   # user-data runs as root; the operator does not
# MERGE the data-root in — the DLAMI's daemon.json carries the nvidia
# runtime registration; clobbering it silently breaks `docker --gpus`.
python3 - <<'PY'
import json, os
p = '/etc/docker/daemon.json'
c = json.load(open(p)) if os.path.exists(p) else {}
c['data-root'] = '/opt/dlami/nvme/docker'
json.dump(c, open(p, 'w'), indent=2)
PY
systemctl restart docker
# Pinned BY DIGEST: identical bits to the image that produced every published
# HF-Jobs table (bench/rigs/hf-gpu/submit.sh pin).
docker pull vllm/vllm-openai@sha256:f0b9a0dc75a9fca3b6811e3279367b2d6a448055a000bfd13859587d74cef268 \
  > /opt/dlami/nvme/work/pull.log 2>&1 &
