#!/usr/bin/env bash
# Rig N (A3): provision 1x i4i.8xlarge (32 vCPU, 2x 3750GB Nitro NVMe).
# Spot with on-demand fallback; tagged kvbench=nvme. Run fio-ceiling.sh, run.sh,
# then teardown.sh SAME DAY.
#
# TIER SESSION (the daemon-level gate): ITYPE=i7i.8xlarge ./provision.sh —
# 32 vCPU, 2x 3750GB 3rd-gen Nitro NVMe (600k read IOPS/device vs i4i's
# 400k; ~$0.5–0.9/hr spot). Then fio-ceiling.sh (the honest denominator),
# run-tier.sh (daemon storm %-of-ceiling + 50-loop torture + warm restart),
# teardown.sh. NOTE the recorded A3 reality: no AWS instance-store device
# reaches the literal 6.0 GB/s line — the session quotes %-of-fio-ceiling
# and A3 stays formally open unless a device surprises ≥6.0.
set -euo pipefail
REGION="${REGION:-us-east-1}"
ITYPE="${ITYPE:-i4i.8xlarge}"
SG="${SG:-kvbench-nvme-sg}"
KEY="${KEY:-kvbench}"
STATE="$(dirname "$0")/.rig-state"
TAG='ResourceType=instance,Tags=[{Key=kvbench,Value=nvme}]'
trap 'echo "[provision] FAILED mid-run — run teardown.sh NOW to kill any tagged instance" >&2' ERR

AMI=$(aws ssm get-parameter --region "$REGION" \
  --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 \
  --query Parameter.Value --output text)

MYIP=$(curl -s https://checkip.amazonaws.com | tr -d '[:space:]')
SGID=$(aws ec2 create-security-group --region "$REGION" --group-name "$SG" \
  --description "kvbench nvme rig" --query GroupId --output text 2>/dev/null || \
  aws ec2 describe-security-groups --region "$REGION" --group-names "$SG" \
    --query 'SecurityGroups[0].GroupId' --output text)
aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$SGID" \
  --protocol tcp --port 22 --cidr "${MYIP}/32" 2>/dev/null || true

launch() {
  local market='InstanceMarketOptions={MarketType=spot,SpotOptions={SpotInstanceType=one-time}}'
  local id
  id=$(aws ec2 run-instances --region "$REGION" --image-id "$AMI" \
        --instance-type "$ITYPE" --key-name "$KEY" --count 1 \
        --security-group-ids "$SGID" --tag-specifications "$TAG" \
        --instance-market-options "$market" \
        --query 'Instances[0].InstanceId' --output text 2>/dev/null) || {
    echo "[provision] spot failed, using on-demand" >&2
    id=$(aws ec2 run-instances --region "$REGION" --image-id "$AMI" \
          --instance-type "$ITYPE" --key-name "$KEY" --count 1 \
          --security-group-ids "$SGID" --tag-specifications "$TAG" \
          --query 'Instances[0].InstanceId' --output text)
  }
  echo "$id"
}

N=$(launch)
echo "[provision] launched $N; waiting..."
aws ec2 wait instance-running --region "$REGION" --instance-ids "$N"
PUB=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$N" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
{ echo "REGION=$REGION"; echo "SG=$SG"; echo "N_ID=$N"; echo "N_PUB=$PUB"; } > "$STATE"
echo "[provision] node $N at $PUB; state in $STATE"
