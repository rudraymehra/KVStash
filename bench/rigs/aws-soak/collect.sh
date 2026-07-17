#!/usr/bin/env bash
# Rig S collector: pull the soak artifacts, compute the verdict inputs, and
# TERMINATE the box (idempotent — safe if the 27h dead-man already fired).
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
STATE="$DIR/.rig-state"; source "$STATE"
SSHOPTS=(-i "$HOME/.ssh/kvbench.pem" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10)
OUT="$DIR/results"
mkdir -p "$OUT"

if aws ec2 describe-instances --region "$REGION" --instance-ids "$IID" \
     --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null | grep -q running; then
  echo "[collect] pulling artifacts from $PUB"
  # Stop the driver gracefully if still running (prints the final line).
  ssh "${SSHOPTS[@]}" "ec2-user@$PUB" \
    'kill -TERM $(cat /tmp/soakout/driver.pid) 2>/dev/null || true; sleep 5; kill -TERM $(cat /tmp/soakout/daemon.pid) 2>/dev/null || true; sleep 3' || true
  scp -r "${SSHOPTS[@]}" "ec2-user@$PUB:/tmp/soakout/*" "$OUT/" || echo "[collect] partial pull"
else
  echo "[collect] instance not running (dead-man fired?) — artifacts lost unless pulled earlier"
fi

echo "[collect] terminating $IID"
aws ec2 terminate-instances --region "$REGION" --instance-ids "$IID" >/dev/null 2>&1 || true
aws ec2 wait instance-terminated --region "$REGION" --instance-ids "$IID" 2>/dev/null || true
aws ec2 delete-security-group --region "$REGION" --group-id "$SGID" 2>/dev/null || true
LEFT=$(aws ec2 describe-instances --region "$REGION" \
  --filters Name=tag:kvbench,Values=soak Name=instance-state-name,Values=running,pending,stopping,stopped \
  --query 'Reservations[].Instances[].InstanceId' --output text)
echo "[collect] teardown done; residual soak instances: '${LEFT:-none}'"

if [ -f "$OUT/gctrace.log" ]; then
  echo "--- GC pause percentiles (wall-clock STW µs, from gctrace):"
  # gctrace line shape: "gc N @Ts C%: clock1+clock2+clock3 ms ..." — the
  # three clock segments; STW are the 1st and 3rd. Extract + percentile.
  grep '^gc ' "$OUT/gctrace.log" | awk '{split($6, a, "+"); print a[1]*1000; print a[3]*1000}' | sort -n > "$OUT/pauses-us.txt"
  n=$(wc -l < "$OUT/pauses-us.txt" | tr -d ' ')
  for q in 50 99 999; do
    idx=$(( (n * q + 999) / 1000 )); [ "$q" = 50 ] && idx=$(( (n + 1) / 2 )); [ "$q" = 99 ] && idx=$(( (n * 99 + 99) / 100 ))
    echo "p$q: $(sed -n "${idx}p" "$OUT/pauses-us.txt") µs (n=$n)"
  done
  echo "--- final soak line:"
  tail -1 "$OUT/soak.jsonl" 2>/dev/null
  echo "--- RSS samples (KiB, first/last):"
  head -1 "$OUT/rss.log" 2>/dev/null; tail -1 "$OUT/rss.log" 2>/dev/null
fi
