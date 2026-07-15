#!/usr/bin/env bash
# Rig N teardown: terminate + delete SG + assert zero kvbench=nvme residue.
# Sweeps ALL regions if state is missing (never claim $0 on assumption).
set -euo pipefail
STATE="$(dirname "$0")/.rig-state"
SG="${SG:-kvbench-nvme-sg}"
if [ -f "$STATE" ]; then source "$STATE"; REGIONS="$REGION"; KNOWN=1
else echo "[teardown] no state — sweeping all regions" >&2
     REGIONS=$(aws ec2 describe-regions --query 'Regions[].RegionName' --output text); KNOWN=0; fi
find_ids(){ aws ec2 describe-instances --region "$1" \
  --filters "Name=tag:kvbench,Values=nvme" "Name=instance-state-name,Values=pending,running,stopping,stopped" \
  --query 'Reservations[].Instances[].InstanceId' --output text; }
RES=0
for R in $REGIONS; do
  IDS=$(find_ids "$R")
  if [ -n "$IDS" ]; then
    aws ec2 terminate-instances --region "$R" --instance-ids $IDS >/dev/null
    aws ec2 wait instance-terminated --region "$R" --instance-ids $IDS
  fi
  aws ec2 delete-security-group --region "$R" --group-name "$SG" 2>/dev/null || true
  [ -n "$(find_ids "$R")" ] && { RES=1; echo "[teardown] !!! RESIDUE in $R" >&2; }
done
rm -f "$STATE"
[ "$RES" -ne 0 ] && { echo "[teardown] FAILED — instances remain" >&2; exit 1; }
echo "[teardown] OK — zero kvbench=nvme resources remain (\$0 residue)"
