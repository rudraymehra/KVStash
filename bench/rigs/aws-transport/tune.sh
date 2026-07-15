#!/usr/bin/env bash
# Rig T: apply the ESnet sysctl profile + jumbo MTU + IRQ affinity to both nodes.
# NOT run today — Day 4.
set -euo pipefail
STATE="$(dirname "$0")/.rig-state"; source "$STATE"
SSH="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 ec2-user"
SYSCTL="$(git rev-parse --show-toplevel)/bench/rigs/sysctl-esnet.conf"

for host in "$A_PUB" "$B_PUB"; do
  echo "[tune] $host"
  scp -o StrictHostKeyChecking=accept-new "$SYSCTL" "ec2-user@$host:/tmp/sysctl-esnet.conf"
  $SSH@"$host" 'sudo sysctl -p /tmp/sysctl-esnet.conf'
  # Jumbo frames (ENA supports MTU 9001); primary iface is usually ens5/eth0.
  $SSH@"$host" 'IF=$(ip -o -4 route show to default | awk "{print \$5}"); sudo ip link set dev "$IF" mtu 9001; ip link show "$IF" | grep mtu'
  # IRQ affinity: spread NIC queues across cores (best-effort).
  $SSH@"$host" 'command -v set_irq_affinity >/dev/null 2>&1 && sudo set_irq_affinity all || echo "set_irq_affinity not present; skipping (kernel default)"'
done
echo "[tune] done. iperf3 version on each node:"
for host in "$A_PUB" "$B_PUB"; do $SSH@"$host" 'iperf3 --version 2>/dev/null | head -1 || echo "iperf3 not installed — sudo dnf install -y iperf3"'; done
