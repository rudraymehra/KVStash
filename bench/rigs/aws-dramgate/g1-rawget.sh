#!/usr/bin/env bash
# Rig G, session 2: the HONEST G1 ceiling. xferspike is a one-way hot-buffer
# blast — a different shape (the Week-2 finding; DESIGN.md: "the goalpost
# does not move"). rawget is the same GET round-trip shape on raw sockets
# with no protocol/auth/checksums: the binding ceiling. Interleaved pairs,
# ratio >= 0.9x.
set -euo pipefail

REGION="${REGION:-us-east-1}"
ITYPE="${ITYPE:-c7i.4xlarge}"
SG="${SG:-kvbench-dramgate-sg}"
KEY="${KEY:-kvbench}"
DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(git rev-parse --show-toplevel)"
OUT="$DIR/g1-rawget-results.txt"
SSHOPTS=(-i "$HOME/.ssh/kvbench.pem" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10)

trap 'echo "[rig-g1] FAILED — check: aws ec2 describe-instances --region '"$REGION"' --filters Name=tag:kvbench,Values=dramgate Name=instance-state-name,Values=running,pending" >&2' ERR

echo "[rig-g1] cross-compiling"
BINS=/tmp/kvb-dramgate; mkdir -p "$BINS"
( cd "$ROOT"
  GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -o "$BINS/kvblockd" ./cmd/kvblockd
  GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -o "$BINS/getbench" ./bench/kvbench/getbench
  GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -o "$BINS/rawget"   ./bench/microbench/rawget )
cat > "$BINS/ns.yaml" <<'EOF'
namespaces:
  - name: bench
    id: 3
    token: bench-token
EOF

AMI=$(aws ssm get-parameter --region "$REGION" \
  --name "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64" \
  --query Parameter.Value --output text)
SGID=$(aws ec2 create-security-group --region "$REGION" --group-name "$SG" \
  --description "kvbench dramgate rig (ssh only)" --query GroupId --output text 2>/dev/null || \
  aws ec2 describe-security-groups --region "$REGION" --group-names "$SG" \
    --query 'SecurityGroups[0].GroupId' --output text)
MYIP=$(curl -s https://checkip.amazonaws.com | tr -d '[:space:]')
aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$SGID" \
  --protocol tcp --port 22 --cidr "${MYIP}/32" 2>/dev/null || true

TAG='ResourceType=instance,Tags=[{Key=kvbench,Value=dramgate}]'
IID=$(aws ec2 run-instances --region "$REGION" --image-id "$AMI" --count 1 \
    --instance-type "$ITYPE" --key-name "$KEY" --security-group-ids "$SGID" \
    --instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time}' \
    --tag-specifications "$TAG" \
    --query 'Instances[0].InstanceId' --output text 2>/dev/null) || \
IID=$(aws ec2 run-instances --region "$REGION" --image-id "$AMI" --count 1 \
    --instance-type "$ITYPE" --key-name "$KEY" --security-group-ids "$SGID" \
    --tag-specifications "$TAG" \
    --query 'Instances[0].InstanceId' --output text)
echo "[rig-g1] instance $IID"
aws ec2 wait instance-running --region "$REGION" --instance-ids "$IID"
PUB=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$IID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
for i in $(seq 1 30); do
  ssh "${SSHOPTS[@]}" "ec2-user@$PUB" true 2>/dev/null && break
  sleep 5
done

teardown() {
  echo "[rig-g1] terminating $IID"
  aws ec2 terminate-instances --region "$REGION" --instance-ids "$IID" >/dev/null
  aws ec2 wait instance-terminated --region "$REGION" --instance-ids "$IID"
  aws ec2 delete-security-group --region "$REGION" --group-id "$SGID" 2>/dev/null || true
  LEFT=$(aws ec2 describe-instances --region "$REGION" \
    --filters Name=tag:kvbench,Values=dramgate Name=instance-state-name,Values=running,pending,stopping,stopped \
    --query 'Reservations[].Instances[].InstanceId' --output text)
  echo "[rig-g1] teardown done; residual dramgate instances: '${LEFT:-none}'"
}
trap teardown EXIT

scp "${SSHOPTS[@]}" "$BINS"/kvblockd "$BINS"/getbench "$BINS"/rawget "$BINS"/ns.yaml "ec2-user@$PUB:/tmp/"

ssh "${SSHOPTS[@]}" "ec2-user@$PUB" 'bash -s' <<'REMOTE' | tee "$OUT"
set -euo pipefail
sudo sysctl -qw net.core.rmem_max=67108864 net.core.wmem_max=67108864
cd /tmp
KVBLOCKD_DRAM_ARENA_BYTES=6442450944 ./kvblockd -listen 127.0.0.1:19440 -namespaces /tmp/ns.yaml 2>/tmp/daemon.log &
DPID=$!
sleep 8
echo "=== G1 honest pairs: rawget same-shape ceiling | getbench vs DRAM daemon ==="
for i in 1 2 3; do
  R=$(./rawget -streams 8 -secs 10 -chunk 1048576 2>/dev/null | tail -1)
  sleep 2
  G=$(./getbench -addr 127.0.0.1:19440 -streams 8 -blocks 32 -block-kib 1024 -pool 4096 -secs 10 -noverify 2>/dev/null | tail -1)
  echo "pair $i: rawget: ${R} | ${G}"
  sleep 2
done
kill -TERM $DPID 2>/dev/null
REMOTE
echo "[rig-g1] results in $OUT"
