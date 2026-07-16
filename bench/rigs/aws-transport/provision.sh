#!/usr/bin/env bash
# Rig T (A1): provision 2x c6in.8xlarge in a cluster placement group.
# Spot with on-demand fallback. All resources tagged kvbench for teardown.
# Requires: aws cli configured, vCPU quota >= 64 (approved).
set -euo pipefail

REGION="${REGION:-us-east-1}"
ITYPE="${ITYPE:-c6in.8xlarge}"
PG="${PG:-kvbench-transport-pg}"
SG="${SG:-kvbench-transport-sg}"
KEY="${KEY:-kvbench}"          # EC2 key pair name (create once: aws ec2 create-key-pair)
STATE="$(dirname "$0")/.rig-state"
TAG='ResourceType=instance,Tags=[{Key=kvbench,Value=transport}]'

# If anything below fails mid-launch, an instance may already be billing.
# It is tagged kvbench=transport, so teardown.sh finds and kills it.
trap 'echo "[provision] FAILED mid-run — run teardown.sh NOW to kill any tagged instance" >&2' ERR

echo "[provision] region=$REGION type=$ITYPE"

# Amazon Linux 2023 x86_64 AMI (SSM public parameter — always current).
AMI=$(aws ssm get-parameter --region "$REGION" \
  --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 \
  --query Parameter.Value --output text)

# Cluster placement group = instances physically close = max network throughput.
aws ec2 create-placement-group --region "$REGION" \
  --group-name "$PG" --strategy cluster 2>/dev/null || echo "[provision] placement group exists"

# Security group: SSH (22) from THIS machine only, and all traffic between the
# two nodes (self-referential) for iperf3/xferspike over the private network.
# Without this the nodes are unreachable and the rig hangs while billing.
MYIP=$(curl -s https://checkip.amazonaws.com | tr -d '[:space:]')
SGID=$(aws ec2 create-security-group --region "$REGION" --group-name "$SG" \
  --description "kvbench transport rig" --query GroupId --output text 2>/dev/null || \
  aws ec2 describe-security-groups --region "$REGION" --group-names "$SG" \
    --query 'SecurityGroups[0].GroupId' --output text)
aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$SGID" \
  --protocol tcp --port 22 --cidr "${MYIP}/32" 2>/dev/null || true
aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$SGID" \
  --protocol -1 --source-group "$SGID" 2>/dev/null || true
echo "[provision] security group $SGID (ssh from ${MYIP}/32 + intra-node)"

launch() {  # $1 = node label (a|b)
  # Try spot first; fall back to on-demand if spot capacity/price fails.
  local market='InstanceMarketOptions={MarketType=spot,SpotOptions={SpotInstanceType=one-time}}'
  local id
  id=$(aws ec2 run-instances --region "$REGION" --image-id "$AMI" \
        --instance-type "$ITYPE" --key-name "$KEY" --count 1 \
        --placement "GroupName=$PG" --security-group-ids "$SGID" \
        --tag-specifications "$TAG" \
        --instance-market-options "$market" \
        --query 'Instances[0].InstanceId' --output text 2>/dev/null) || {
    echo "[provision] spot failed for node $1, using on-demand" >&2
    id=$(aws ec2 run-instances --region "$REGION" --image-id "$AMI" \
          --instance-type "$ITYPE" --key-name "$KEY" --count 1 \
          --placement "GroupName=$PG" --security-group-ids "$SGID" \
          --tag-specifications "$TAG" \
          --query 'Instances[0].InstanceId' --output text)
  }
  echo "$id"
}

A=$(launch a); B=$(launch b)
echo "[provision] launched A=$A B=$B; waiting for running..."
aws ec2 wait instance-running --region "$REGION" --instance-ids "$A" "$B"

ipof() { aws ec2 describe-instances --region "$REGION" --instance-ids "$1" \
  --query "Reservations[0].Instances[0].$2" --output text; }

{
  echo "REGION=$REGION"
  echo "PG=$PG"; echo "SG=$SG"; echo "SGID=$SGID"
  echo "A_ID=$A"; echo "B_ID=$B"
  echo "A_PUB=$(ipof "$A" PublicIpAddress)"; echo "B_PUB=$(ipof "$B" PublicIpAddress)"
  echo "A_PRIV=$(ipof "$A" PrivateIpAddress)"; echo "B_PRIV=$(ipof "$B" PrivateIpAddress)"
} > "$STATE"

echo "[provision] state written to $STATE:"
cat "$STATE"
echo "[provision] next: bash tune.sh, then iperf-ceiling.sh, run.sh, teardown.sh"
