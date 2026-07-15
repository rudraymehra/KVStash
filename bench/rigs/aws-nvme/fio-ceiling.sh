#!/usr/bin/env bash
# Rig N: measure the DEVICE ceiling with fio before running nvmeprobe — every
# nvmeprobe number is reported as % of this ceiling (same discipline as iperf3).
set -euo pipefail
STATE="$(dirname "$0")/.rig-state"; source "$STATE"
SSH="ssh -o StrictHostKeyChecking=accept-new ec2-user"
$SSH@"$N_PUB" 'sudo dnf install -y fio >/dev/null 2>&1 || true; fio --version'
# i4i instance-store devices are usually nvme1n1/nvme2n1 (nvme0 is the EBS root)
$SSH@"$N_PUB" 'lsblk -d -o NAME,SIZE,MODEL | grep -i nvme'
for rw in read write; do for bs in 128k 1m; do
  echo "== fio $rw bs=$bs qd=32 (raw device, direct) =="
  $SSH@"$N_PUB" "sudo fio --name=ceil --filename=/dev/nvme1n1 --direct=1 --rw=$rw \
    --bs=$bs --iodepth=32 --numjobs=1 --time_based --runtime=30 --ioengine=libaio \
    --group_reporting --readonly=$([ $rw = read ] && echo 1 || echo 0) 2>/dev/null \
    | grep -E 'READ:|WRITE:'" || true
done; done
echo "[fio] record the best value per op as the device ceiling"
