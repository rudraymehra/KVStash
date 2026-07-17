#!/usr/bin/env bash
# Rig S: the 24h mixed-load eviction soak on ONE on-demand arm64 box
# (m7g.xlarge: 4 vCPU / 16 GiB ≈ $0.17/hr → ~$4.5 for 26h).
#
#   daemon: 8 GiB arena, hugepages, s3fifo, GODEBUG=gctrace=1
#   driver: bench/soak — ~12 GiB working set (10800 × ~1.1 MiB) (nonstop eviction), xxh3 +
#           byte-regeneration verification on every GET hit
#   hourly: heap + goroutine pprof snapshots, /metrics scrape, RSS sample
#
# SAFETY: the instance runs `shutdown -h +1620` (27h) with
# instance-initiated-shutdown-behavior=terminate — it kills ITSELF even if
# collect.sh never runs. collect.sh (run within 27h) pulls artifacts and
# terminates early. Tagged kvbench=soak.
set -euo pipefail

REGION="${REGION:-us-east-1}"
ITYPE="${ITYPE:-m7g.xlarge}"
SG="${SG:-kvbench-soak-sg}"
KEY="${KEY:-kvbench}"
SOAK_HOURS="${SOAK_HOURS:-24}"
DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(git rev-parse --show-toplevel)"
STATE="$DIR/.rig-state"
SSHOPTS=(-i "$HOME/.ssh/kvbench.pem" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10)

trap 'echo "[rig-s] FAILED — check: aws ec2 describe-instances --region '"$REGION"' --filters Name=tag:kvbench,Values=soak Name=instance-state-name,Values=running,pending" >&2' ERR

echo "[rig-s] cross-compiling linux/arm64"
BINS=/tmp/kvb-soakrig; mkdir -p "$BINS"
( cd "$ROOT"
  GOOS=linux GOARCH=arm64 CGO_ENABLED=0 go build -o "$BINS/kvblockd" ./cmd/kvblockd
  GOOS=linux GOARCH=arm64 CGO_ENABLED=0 go build -o "$BINS/soak" ./bench/soak )
cat > "$BINS/ns.yaml" <<'EOF'
namespaces:
  - { name: soak-a, id: 11, token: soak-token }
  - { name: soak-b, id: 12, token: soak-token }
  - { name: soak-c, id: 13, token: soak-token }
EOF
cat > "$BINS/soak-daemon.yaml" <<'EOF'
listen_addr: "127.0.0.1:9440"
metrics_addr: "127.0.0.1:9442"
dram_arena_bytes: 8589934592   # 8 GiB
dram_hugepages: true
pinned_bytes_cap: 268435456    # 256 MiB per ns
eviction_policy: "s3fifo"
eviction_watermark_pct: 95
eviction_batch_pct: 5
EOF

AMI=$(aws ssm get-parameter --region "$REGION" \
  --name "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-arm64" \
  --query Parameter.Value --output text)
SGID=$(aws ec2 create-security-group --region "$REGION" --group-name "$SG" \
  --description "kvbench soak rig (ssh only)" --query GroupId --output text 2>/dev/null || \
  aws ec2 describe-security-groups --region "$REGION" --group-names "$SG" \
    --query 'SecurityGroups[0].GroupId' --output text)
MYIP=$(curl -s https://checkip.amazonaws.com | tr -d '[:space:]')
aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$SGID" \
  --protocol tcp --port 22 --cidr "${MYIP}/32" 2>/dev/null || true

echo "[rig-s] launching $ITYPE on-demand (self-terminates in 27h)"
IID=$(aws ec2 run-instances --region "$REGION" --image-id "$AMI" --count 1 \
    --instance-type "$ITYPE" --key-name "$KEY" --security-group-ids "$SGID" \
    --instance-initiated-shutdown-behavior terminate \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=kvbench,Value=soak}]' \
    --query 'Instances[0].InstanceId' --output text)
aws ec2 wait instance-running --region "$REGION" --instance-ids "$IID"
PUB=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$IID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
printf 'IID=%s\nPUB=%s\nSGID=%s\nREGION=%s\n' "$IID" "$PUB" "$SGID" "$REGION" > "$STATE"
echo "[rig-s] instance $IID at $PUB (state → $STATE)"
for i in $(seq 1 30); do
  ssh "${SSHOPTS[@]}" "ec2-user@$PUB" true 2>/dev/null && break
  sleep 5
done

scp "${SSHOPTS[@]}" "$BINS"/kvblockd "$BINS"/soak "$BINS"/ns.yaml "$BINS"/soak-daemon.yaml "ec2-user@$PUB:/tmp/"

ssh "${SSHOPTS[@]}" "ec2-user@$PUB" "SOAK_HOURS=$SOAK_HOURS bash -s" <<'REMOTE'
set -euo pipefail
sudo shutdown -h +1620   # the 27h dead-man switch (terminate-on-shutdown)
# Hugepages: 8 GiB arena / 2 MiB pages + slack.
sudo sysctl -qw vm.nr_hugepages=4300
sudo sysctl -qw net.core.rmem_max=67108864 net.core.wmem_max=67108864
mkdir -p /tmp/soakout

GODEBUG=gctrace=1 nohup /tmp/kvblockd -config /tmp/soak-daemon.yaml -namespaces /tmp/ns.yaml \
  > /tmp/soakout/daemon.out 2> /tmp/soakout/gctrace.log &
echo $! > /tmp/soakout/daemon.pid
sleep 20   # 8 GiB prefault

DUR_H="${SOAK_HOURS}"
nohup /tmp/soak -addr 127.0.0.1:9440 -duration "${DUR_H}h" -workers 24 -keys 10800 \
  -report 60s -out /tmp/soakout/soak.jsonl > /tmp/soakout/driver.out 2>&1 &
echo $! > /tmp/soakout/driver.pid

# Hourly snapshots: heap + goroutine pprof, metrics, RSS.
nohup bash -c '
  for h in $(seq 1 26); do
    sleep 3600
    ts=$(date -u +%Y%m%dT%H%M)
    curl -s -o /tmp/soakout/heap-$ts.pb.gz  "http://127.0.0.1:9442/debug/pprof/heap" || true
    curl -s -o /tmp/soakout/goro-$ts.txt "http://127.0.0.1:9442/debug/pprof/goroutine?debug=1" || true
    curl -s "http://127.0.0.1:9442/metrics" | grep -E "^(kvb_|go_goroutines|process_resident)" > /tmp/soakout/metrics-$ts.txt || true
    ps -o rss= -p $(cat /tmp/soakout/daemon.pid) >> /tmp/soakout/rss.log || true
  done' > /dev/null 2>&1 &
echo "[remote] soak running: ${DUR_H}h; box self-terminates in 27h"
REMOTE

echo "[rig-s] soak launched. Collect within 27h: bash $DIR/collect.sh"
