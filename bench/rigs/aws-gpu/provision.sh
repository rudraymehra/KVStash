#!/usr/bin/env bash
# kvbench GPU-node provisioner (laptop-side). Launches ONE g6e.2xlarge
# (L40S 48 GB) in us-east-1 with the AZ ladder 1a -> 1c -> 1b -> 1d,
# dead-man shutdown armed from user-data, everything tagged kvbench.
#
#   DRY_RUN=1 bench/rigs/aws-gpu/provision.sh   # print every command, run none
#   bench/rigs/aws-gpu/provision.sh             # launch for real
#
# Env: INSTANCE_TYPE (default g6e.2xlarge), KEY_NAME (default kvbench),
#      VOLUME_GB (default 100). Writes instance id + IP to $STATE_DIR
#      (default ~/kvbench-dday) — teardown.sh reads the same files.
set -euo pipefail
log() { printf '[provision %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
die() { log "FATAL: $*"; exit 1; }
run() { if [[ "${DRY_RUN:-0}" == "1" ]]; then { printf 'DRY-RUN>'; printf ' %q' "$@"; printf '\n'; } >&2; else "$@"; fi; }

REGION=us-east-1
INSTANCE_TYPE="${INSTANCE_TYPE:-g6e.2xlarge}"
KEY_NAME="${KEY_NAME:-kvbench}"
VOLUME_GB="${VOLUME_GB:-100}"
STATE_DIR="${STATE_DIR:-$HOME/kvbench-dday}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$STATE_DIR"

aws ec2 describe-key-pairs --region "$REGION" --key-names "$KEY_NAME" >/dev/null 2>&1 \
  || die "key pair '$KEY_NAME' not found in $REGION"

# AMI: Deep Learning BASE OSS Nvidia Driver GPU AMI only — resolved live via
# SSM so we never bake a stale id. (Never a "PyTorch x.y" DLAMI: the Docker
# path needs only driver >= image userspace.)
AMI=$(aws ssm get-parameter --region "$REGION" \
  --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id \
  --query Parameter.Value --output text)
[[ "$AMI" == ami-* ]] || die "AMI resolution failed: $AMI"
log "AMI: $AMI"

# Security group: SSH from the operator's egress IP only, tagged kvbench.
MYIP=$(curl -fsS https://checkip.amazonaws.com)/32
SG=$(aws ec2 describe-security-groups --region "$REGION" \
  --filters Name=group-name,Values=kvbench-ssh --query 'SecurityGroups[0].GroupId' --output text)
if [[ "$SG" == "None" || -z "$SG" ]]; then
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    run aws ec2 create-security-group --region "$REGION" --group-name kvbench-ssh \
      --description "kvbench: SSH from operator IP only" \
      --tag-specifications 'ResourceType=security-group,Tags=[{Key=kvbench,Value=net}]' \
      --query GroupId --output text
    SG=sg-DRYRUN
  else
    SG=$(aws ec2 create-security-group --region "$REGION" --group-name kvbench-ssh \
      --description "kvbench: SSH from operator IP only" \
      --tag-specifications 'ResourceType=security-group,Tags=[{Key=kvbench,Value=net}]' \
      --query GroupId --output text)
  fi
fi
log "SG: $SG (authorizing $MYIP; duplicate-rule errors are fine)"
run aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$SG" \
  --protocol tcp --port 22 --cidr "$MYIP" 2>/dev/null || true

# AZ ladder: g6e exists only in these four; 1a shows the loosest capacity.
for AZ in us-east-1a us-east-1c us-east-1b us-east-1d; do
  SUBNET=$(aws ec2 describe-subnets --region "$REGION" \
    --filters "Name=availability-zone,Values=$AZ" "Name=default-for-az,Values=true" \
    --query 'Subnets[0].SubnetId' --output text)
  [[ "$SUBNET" == subnet-* ]] || continue
  log "attempting $INSTANCE_TYPE in $AZ ($SUBNET)"
  set +e
  IID=$(run aws ec2 run-instances --region "$REGION" \
    --image-id "$AMI" --instance-type "$INSTANCE_TYPE" --count 1 \
    --key-name "$KEY_NAME" --security-group-ids "$SG" --subnet-id "$SUBNET" \
    --associate-public-ip-address \
    --instance-initiated-shutdown-behavior terminate \
    --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$VOLUME_GB,\"VolumeType\":\"gp3\",\"Iops\":5000,\"Throughput\":300,\"DeleteOnTermination\":true}}]" \
    --tag-specifications \
      'ResourceType=instance,Tags=[{Key=kvbench,Value=gpu},{Key=Name,Value=kvbench-l40s-dday}]' \
      'ResourceType=volume,Tags=[{Key=kvbench,Value=gpu}]' \
    --user-data "file://$HERE/userdata.sh" \
    --query 'Instances[0].InstanceId' --output text 2>"$STATE_DIR/launch-err.txt")
  RC=$?
  set -e
  if [[ "${DRY_RUN:-0}" == "1" ]]; then log "dry-run: stopping after first AZ"; exit 0; fi
  if [[ $RC -eq 0 && "$IID" == i-* ]]; then
    log "LAUNCHED $IID in $AZ"
    echo "$IID" > "$STATE_DIR/gpu-instance-id"
    aws ec2 wait instance-running --region "$REGION" --instance-ids "$IID"
    IP=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$IID" \
      --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
    echo "$IP" > "$STATE_DIR/gpu-ip"
    # Assert the cost dead-men actually took (a silent default here is how a
    # box outlives its operator):
    ATTR=$(aws ec2 describe-instance-attribute --region "$REGION" --instance-id "$IID" \
      --attribute instanceInitiatedShutdownBehavior --query 'InstanceInitiatedShutdownBehavior.Value' --output text)
    [[ "$ATTR" == "terminate" ]] || die "shutdown-behavior is '$ATTR', not terminate — fix before proceeding"
    DOT=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$IID" \
      --query 'Reservations[0].Instances[0].BlockDeviceMappings[0].Ebs.DeleteOnTermination' --output text)
    [[ "$DOT" == "True" ]] || die "root volume DeleteOnTermination=$DOT — fix before proceeding"
    log "instance $IID at $IP — dead-man verified (terminate-on-shutdown + DeleteOnTermination)"
    log "next: bench/rigs/aws-gpu/node-setup.sh"
    exit 0
  fi
  if grep -q VcpuLimitExceeded "$STATE_DIR/launch-err.txt" 2>/dev/null; then
    die "VcpuLimitExceeded — the G/VT quota is not applied; reply to the AWS support case (see the correspondence email) before retrying"
  fi
  log "AZ $AZ failed (likely capacity): $(tail -c 200 "$STATE_DIR/launch-err.txt" 2>/dev/null)"
done
die "no AZ had capacity — retry later or fall back to g6e.xlarge (same L40S, 4 vCPU; pre-test client CPU headroom)"
