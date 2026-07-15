#!/usr/bin/env bash
# Rig T: measure the link ceiling with iperf3 (>=3.16) BEFORE any store bench.
# Server on B, client on A, over the PRIVATE (placement-group) network.
# Records best one-way Gbps per -P/MTU as the ceiling. NOT run today — Day 4.
set -euo pipefail
STATE="$(dirname "$0")/.rig-state"; source "$STATE"
SSH="ssh -o StrictHostKeyChecking=accept-new ec2-user"
OUT="$(dirname "$0")/iperf-ceiling.txt"; : > "$OUT"

$SSH@"$B_PUB" 'pkill iperf3 2>/dev/null; nohup iperf3 -s >/tmp/iperf3s.log 2>&1 &'; sleep 2
for P in 8 16 32 64; do
  echo "== -P $P ==" | tee -a "$OUT"
  $SSH@"$A_PUB" "iperf3 -c $B_PRIV -P $P -l 1M -t 60 -J" \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); print(f"{d[\"end\"][\"sum_sent\"][\"bits_per_second\"]/1e9:.1f} Gbit/s")' \
    | tee -a "$OUT"
done
$SSH@"$B_PUB" 'pkill iperf3 2>/dev/null || true'
echo "[iperf] ceiling recorded in $OUT (use the BEST value as the A1 denominator)"
