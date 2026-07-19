#!/usr/bin/env bash
# Rig S3S: the overnight (12–24h) three-tier spill soak on ONE on-demand
# arm64 box (m7g.xlarge ≈ $0.17/hr → ~$4.5 for 26h), MinIO on-box as the
# cold tier.
#
#   daemon: 6 GiB arena, s3fifo, NVMe tier on a gp3-backed dir (this soak
#           proves the SPILL machinery + no-leak, not device speed — say so
#           in the writeup), spill to MinIO, admit_min_hits 0 (the ingest
#           posture: a PUT-heavy soak at the default would endurance-delete
#           instead of demote and the cold tier would starve)
#   minio:  single static binary, loopback only, static creds via env
#   driver: bench/soak — working set ~2× the NVMe budget so demote → seal
#           → spill → retire-flip → cold-GET runs CONTINUOUSLY; every GET
#           hit is byte-regenerated + xxh3-verified (cold reads included)
#   hourly: heap/goroutine pprof, /metrics scrape (kvb_s3_* families), RSS
#
# SAFETY: `shutdown -h +1620` (27h) with terminate-on-shutdown; artifacts
# go to /var/soakout (EBS — /tmp is tmpfs on AL2023 and dies with the box).
# Tagged kvbench=s3soak. Collect: bash collect.sh (within 27h).
set -euo pipefail

REGION="${REGION:-us-east-1}"
ITYPE="${ITYPE:-m7g.xlarge}"
SG="${SG:-kvbench-soak-sg}"
KEY="${KEY:-kvbench}"
SOAK_HOURS="${SOAK_HOURS:-18}"
DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(git rev-parse --show-toplevel)"
STATE="$DIR/.rig-state"
SSHOPTS=(-i "$HOME/.ssh/kvbench.pem" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10)

trap 'echo "[rig-s3s] FAILED — check: aws ec2 describe-instances --region '"$REGION"' --filters Name=tag:kvbench,Values=s3soak Name=instance-state-name,Values=running,pending" >&2' ERR

echo "[rig-s3s] cross-compiling linux/arm64"
BINS=/tmp/kvb-s3soakrig; mkdir -p "$BINS"
( cd "$ROOT"
  GOOS=linux GOARCH=arm64 CGO_ENABLED=0 go build -o "$BINS/kvblockd" ./cmd/kvblockd
  GOOS=linux GOARCH=arm64 CGO_ENABLED=0 go build -o "$BINS/soak" ./bench/soak )
cat > "$BINS/ns.yaml" <<'EOF'
namespaces:
  - { name: soak-a, id: 11, token: soak-token }
  - { name: soak-b, id: 12, token: soak-token }
  - { name: soak-c, id: 13, token: soak-token }
EOF
cat > "$BINS/s3soak-daemon.yaml" <<'EOF'
listen_addr: "127.0.0.1:9440"
metrics_addr: "127.0.0.1:9442"
dram_arena_bytes: 6442450944   # 6 GiB (room for MinIO on the 16 GiB box)
dram_hugepages: true
pinned_bytes_cap: 268435456
eviction_policy: "s3fifo"
eviction_watermark_pct: 95
eviction_batch_pct: 5
nvme_paths: ["/var/kvbnvme"]
nvme_max_bytes: 8589934592     # 8 GiB — working set overflows it, reclaim never idles
nvme_segment_bytes: 67108864   # 64 MiB objects — realistic spill granularity
nvme_read_workers: 8
nvme_demote_watermark_pct: 90
nvme_demote_batch_pct: 5
nvme_admit_min_hits: 0         # ingest posture — see header
nvme_sync_every_bytes: 8388608
nvme_ckpt_every_segments: 4
s3_bucket: "kvbsoak"
s3_node_id: "soak1"
s3_endpoint_override: "http://127.0.0.1:9000"
s3_path_style: true
s3_spill_queue: 8
s3_read_timeout_ms: 2000
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

echo "[rig-s3s] launching $ITYPE on-demand, 40 GiB gp3 root (self-terminates in 27h)"
IID=$(aws ec2 run-instances --region "$REGION" --image-id "$AMI" --count 1 \
    --instance-type "$ITYPE" --key-name "$KEY" --security-group-ids "$SGID" \
    --instance-initiated-shutdown-behavior terminate \
    --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":40,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=kvbench,Value=s3soak}]' \
    --query 'Instances[0].InstanceId' --output text)
aws ec2 wait instance-running --region "$REGION" --instance-ids "$IID"
PUB=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$IID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
printf 'IID=%s\nPUB=%s\nSGID=%s\nREGION=%s\n' "$IID" "$PUB" "$SGID" "$REGION" > "$STATE"
echo "[rig-s3s] instance $IID at $PUB (state → $STATE)"
for i in $(seq 1 30); do
  ssh "${SSHOPTS[@]}" "ec2-user@$PUB" true 2>/dev/null && break
  sleep 5
done

scp "${SSHOPTS[@]}" "$BINS"/kvblockd "$BINS"/soak "$BINS"/ns.yaml "$BINS"/s3soak-daemon.yaml "ec2-user@$PUB:/tmp/"

ssh "${SSHOPTS[@]}" "ec2-user@$PUB" "SOAK_HOURS=$SOAK_HOURS bash -s" <<'REMOTE'
set -euo pipefail
sudo shutdown -h +1620   # the 27h dead-man switch (terminate-on-shutdown)
sudo sysctl -qw vm.nr_hugepages=3300   # 6 GiB / 2 MiB + slack
sudo mkdir -p /var/soakout /var/kvbnvme /var/minio-data
sudo chown ec2-user:ec2-user /var/soakout /var/kvbnvme /var/minio-data
sudo install -m 0755 /tmp/kvblockd /tmp/soak /usr/local/bin/ 2>/dev/null || { cp /tmp/kvblockd /tmp/soak ~/; }

# MinIO: one static binary, loopback only.
curl -sSf -o /tmp/minio https://dl.min.io/server/minio/release/linux-arm64/minio
chmod +x /tmp/minio
MINIO_ROOT_USER=kvbsoak MINIO_ROOT_PASSWORD=kvbsoak-secret nohup /tmp/minio server /var/minio-data \
  --address 127.0.0.1:9000 --console-address 127.0.0.1:9001 \
  > /var/soakout/minio.out 2>&1 &
echo $! > /var/soakout/minio.pid
sleep 5
AWS_ACCESS_KEY_ID=kvbsoak AWS_SECRET_ACCESS_KEY=kvbsoak-secret \
  aws --endpoint-url http://127.0.0.1:9000 s3 mb s3://kvbsoak

AWS_ACCESS_KEY_ID=kvbsoak AWS_SECRET_ACCESS_KEY=kvbsoak-secret AWS_REGION=us-east-1 \
GODEBUG=gctrace=1 nohup /tmp/kvblockd -config /tmp/s3soak-daemon.yaml -namespaces /tmp/ns.yaml \
  > /var/soakout/daemon.out 2> /var/soakout/gctrace.log &
echo $! > /var/soakout/daemon.pid
sleep 20   # arena prefault + volume open

# Working set ~16 GiB (14400 × ~1.1 MiB) over a 6 GiB arena + 8 GiB NVMe
# budget: all three tiers churn for the whole window.
DUR_H="${SOAK_HOURS}"
nohup /tmp/soak -addr 127.0.0.1:9440 -duration "${DUR_H}h" -workers 24 -keys 14400 \
  -report 60s -out /var/soakout/soak.jsonl > /var/soakout/driver.out 2>&1 &
echo $! > /var/soakout/driver.pid

nohup bash -c '
  for h in $(seq 1 26); do
    sleep 3600
    ts=$(date -u +%Y%m%dT%H%M)
    curl -s -o /var/soakout/heap-$ts.pb.gz  "http://127.0.0.1:9442/debug/pprof/heap" || true
    curl -s -o /var/soakout/goro-$ts.txt "http://127.0.0.1:9442/debug/pprof/goroutine?debug=1" || true
    curl -s "http://127.0.0.1:9442/metrics" | grep -E "^(kvb_|go_goroutines|process_resident)" > /var/soakout/metrics-$ts.txt || true
    ps -o rss= -p $(cat /var/soakout/daemon.pid) >> /var/soakout/rss.log || true
    du -sb /var/minio-data >> /var/soakout/minio-usage.log || true
  done' > /dev/null 2>&1 &
echo "[remote] s3 spill soak running: ${DUR_H}h; box self-terminates in 27h"
REMOTE

echo "[rig-s3s] spill soak launched. Collect within 27h: bash $DIR/collect.sh"
