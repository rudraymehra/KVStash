#!/usr/bin/env bash
# Rig N: run the nvmeprobe matrix on the instance-store NVMe. A3 verdict input.
set -euo pipefail
STATE="$(dirname "$0")/.rig-state"; source "$STATE"
SSH="ssh -i $HOME/.ssh/kvbench.pem -o StrictHostKeyChecking=accept-new ec2-user"
ROOT="$(git rev-parse --show-toplevel)"; OUT="$(dirname "$0")/nvmeprobe-results.jsonl"; : > "$OUT"
( cd "$ROOT" && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -o /tmp/nvmeprobe-linux ./bench/microbench/nvmeprobe )
scp -i $HOME/.ssh/kvbench.pem -o StrictHostKeyChecking=accept-new /tmp/nvmeprobe-linux "ec2-user@$N_PUB:/tmp/nvmeprobe"
# mount an instance-store NVMe with XFS for the file-based run + raw device runs
$SSH@"$N_PUB" 'sudo mkfs.xfs -f /dev/nvme1n1 >/dev/null 2>&1 && sudo mkdir -p /mnt/nvme && sudo mount /dev/nvme1n1 /mnt/nvme && sudo chown ec2-user /mnt/nvme'
for op in read write; do for bs in 131072 1048576; do for qd in 8 32; do
  $SSH@"$N_PUB" "/tmp/nvmeprobe --backend=threadpool --path=/mnt/nvme/probe.dat --file-size=$((32*1024*1024*1024)) --op=$op --bs=$bs --qd=$qd --duration=30s" | tee -a "$OUT"
done; done; done
echo "[run] results in $OUT — A3 gate: any read config >= 6.0 gbytes_per_s (and compare vs fio ceiling)"
