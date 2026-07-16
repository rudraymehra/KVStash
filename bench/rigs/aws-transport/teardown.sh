#!/usr/bin/env bash
# Rig T: terminate instances, delete placement group + security group, and
# ASSERT zero residue. Safe to run anytime; the assert is the whole point.
#
# Money-honesty: the "$0 residue" claim is only trustworthy if we check the
# SAME region the instances were launched in. If .rig-state is missing we do
# NOT know that region, so we sweep ALL regions for the tag and refuse to
# assert clean on assumption.
set -euo pipefail
STATE="$(dirname "$0")/.rig-state"
PG="${PG:-kvbench-transport-pg}"
SG="${SG:-kvbench-transport-sg}"

if [ -f "$STATE" ]; then
  source "$STATE"                       # sets REGION (+ IDs) from provision
  REGIONS="$REGION"
  KNOWN_REGION=1
else
  echo "[teardown] no .rig-state — region unknown; sweeping ALL regions for the tag" >&2
  REGIONS=$(aws ec2 describe-regions --query 'Regions[].RegionName' --output text)
  KNOWN_REGION=0
fi

find_ids() {  # $1 = region
  aws ec2 describe-instances --region "$1" \
    --filters "Name=tag:kvbench,Values=transport" \
              "Name=instance-state-name,Values=pending,running,stopping,stopped" \
    --query 'Reservations[].Instances[].InstanceId' --output text
}

RESIDUE=0
for R in $REGIONS; do
  IDS=$(find_ids "$R")
  if [ -n "$IDS" ]; then
    echo "[teardown] terminating in $R: $IDS"
    aws ec2 terminate-instances --region "$R" --instance-ids $IDS >/dev/null
    aws ec2 wait instance-terminated --region "$R" --instance-ids $IDS
  fi
  aws ec2 delete-placement-group --region "$R" --group-name "$PG" 2>/dev/null || true
  aws ec2 delete-security-group --region "$R" --group-name "$SG" 2>/dev/null || true
  # Re-check this region after termination.
  if [ -n "$(find_ids "$R")" ]; then RESIDUE=1; echo "[teardown] !!! RESIDUE in $R" >&2; fi
done
rm -f "$STATE"

if [ "$RESIDUE" -ne 0 ]; then
  echo "[teardown] FAILED — tagged instances still present; investigate before trusting \$0" >&2
  exit 1
fi
if [ "$KNOWN_REGION" -eq 1 ]; then
  echo "[teardown] OK — zero kvbench=transport resources remain in $REGION (\$0 residue)"
else
  echo "[teardown] OK — swept all regions, zero kvbench=transport resources remain (\$0 residue)"
fi
