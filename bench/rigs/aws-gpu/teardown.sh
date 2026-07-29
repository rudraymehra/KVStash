#!/usr/bin/env bash
# kvbench teardown + $0-residue verification. Run after EVERY session; a
# session is not over until this prints ALL-CLEAR.
set -euo pipefail
log() { printf '[teardown %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
REGION=us-east-1
STATE_DIR="${STATE_DIR:-$HOME/kvbench-dday}"

IDS=$(aws ec2 describe-instances --region $REGION \
  --filters Name=tag-key,Values=kvbench Name=instance-state-name,Values=pending,running,stopping,stopped \
  --query 'Reservations[].Instances[].InstanceId' --output text)
if [[ -n "$IDS" ]]; then
  log "terminating: $IDS"
  # shellcheck disable=SC2086
  aws ec2 terminate-instances --region $REGION --instance-ids $IDS >/dev/null
  # shellcheck disable=SC2086
  aws ec2 wait instance-terminated --region $REGION --instance-ids $IDS
fi

FAIL=0
# Fail CLOSED: a describe that errors is NOT "clear" — ALL-CLEAR must mean
# every check ran and returned empty.
chk() {
  local what=$1; shift
  local got rc
  got=$("$@" 2>&1); rc=$?
  if [[ $rc -ne 0 ]]; then log "CHECK ERROR ($what): $got"; FAIL=1
  elif [[ -n "$got" && "$got" != "None" ]]; then log "RESIDUE ($what): $got"; FAIL=1
  else log "clear: $what"; fi
}
chk instances aws ec2 describe-instances --region $REGION --filters Name=tag-key,Values=kvbench Name=instance-state-name,Values=pending,running,stopping,stopped --query 'Reservations[].Instances[].InstanceId' --output text
chk volumes-available aws ec2 describe-volumes --region $REGION --query 'Volumes[?State==`available`].VolumeId' --output text
chk elastic-ips aws ec2 describe-addresses --region $REGION --query 'Addresses[].AllocationId' --output text
chk capacity-reservations aws ec2 describe-capacity-reservations --region $REGION --query 'CapacityReservations[?State==`active`].CapacityReservationId' --output text
chk placement-groups aws ec2 describe-placement-groups --region $REGION --query 'PlacementGroups[?starts_with(GroupName,`kvbench`)].GroupName' --output text
chk nat-gateways aws ec2 describe-nat-gateways --region $REGION --filter Name=state,Values=available --query 'NatGateways[].NatGatewayId' --output text

if [[ -f "$STATE_DIR/ledger.csv" ]]; then
  log "session spend (wall-clock x rate): \$$(awk -F, '{s+=$5} END {printf "%.2f", s}' "$STATE_DIR/ledger.csv")"
  log "verify against Cost Explorer (tag kvbench) tomorrow morning — console lags 8-24h"
fi
[[ $FAIL -eq 0 ]] && log "ALL-CLEAR: \$0 residue" || { log "RESIDUE FOUND — fix before walking away"; exit 1; }
