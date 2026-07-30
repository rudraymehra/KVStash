#!/usr/bin/env bash
# 4.5h certified-pipeline soak: loop full two-phase runs with all gates.
cd /Users/rudraym/Cashing
END=$(( $(date +%s) + 16200 ))
i=0
while [ "$(date +%s)" -lt "$END" ]; do
  i=$((i+1))
  # Liveness gate: an unreachable box must END the soak, not spin on it.
  # (Learned the hard way: without this, a dead-man terminate turned the
  # remaining window into 130,000 instant ssh failures in the log.)
  if ! ssh -i ~/kvbench-dday/kvbench.pem -o ConnectTimeout=10 -o BatchMode=yes \
       "ubuntu@$(cat ~/kvbench-dday/gpu-ip)" true 2>/dev/null; then
    echo "SOAK ABORT: box unreachable after $i cycles" >> ~/kvbench-dday/soak.log
    exit 0
  fi
  for arm in a10g-cal a10g-64k a10g-128k; do
    [ "$(date +%s)" -ge "$END" ] && break
    bash bench/rigs/aws-gpu/run-arm.sh "$arm" "soak$i" >> ~/kvbench-dday/soak.log 2>&1 || true
  done
done
echo "SOAK COMPLETE: $i cycles" >> ~/kvbench-dday/soak.log
