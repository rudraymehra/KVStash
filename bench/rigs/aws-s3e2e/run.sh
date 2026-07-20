#!/usr/bin/env bash
# Real-S3 e2e: run s3probe ON an in-region box (us-east-1) against the
# 1-day-lifecycle e2e bucket, pull the JSONL back, and print the summary.
# Credentials are the SCOPED kvb-e2e IAM user (bucket-only policy), shipped
# over stdin into a 0600 env file removed in the same session — never on a
# command line (ps-visible) and never in this repo.
#
# Usage: BOX_IP=<public-ip> CREDS_JSON=/tmp/kvb-e2e-creds.json bash run.sh
#
# Teardown (same session, verify $0):
#   aws s3 rm s3://$BUCKET --recursive && aws s3api delete-bucket --bucket $BUCKET
#   aws iam delete-access-key --user-name kvb-e2e --access-key-id <id>
#   aws iam delete-user-policy --user-name kvb-e2e --policy-name kvb-e2e-bucket-only
#   aws iam delete-user --user-name kvb-e2e
set -euo pipefail
cd "$(dirname "$0")"

BUCKET="${BUCKET:-kvb-week9-e2e-057996826223}"
BOX_IP="${BOX_IP:?set BOX_IP to the in-region box public IP}"
CREDS_JSON="${CREDS_JSON:-/tmp/kvb-e2e-creds.json}"
SSHOPTS=(-i "$HOME/.ssh/kvbench.pem" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10)

AKID=$(python3 -c "import json;print(json.load(open('$CREDS_JSON'))['AccessKey']['AccessKeyId'])")
SECRET=$(python3 -c "import json;print(json.load(open('$CREDS_JSON'))['AccessKey']['SecretAccessKey'])")

GOOS=linux GOARCH=arm64 go build -o /tmp/s3probe-arm64 .
scp "${SSHOPTS[@]}" /tmp/s3probe-arm64 "ec2-user@$BOX_IP:/tmp/s3probe"

# Never put the secret on a remote COMMAND LINE (visible to every user via
# `ps` and in shell history on the box): ship it over stdin into a 0600 env
# file, source it for the one command, and remove it in the same session.
printf 'export AWS_ACCESS_KEY_ID=%s\nexport AWS_SECRET_ACCESS_KEY=%s\nexport AWS_REGION=us-east-1\n' \
  "$AKID" "$SECRET" |
  ssh "${SSHOPTS[@]}" "ec2-user@$BOX_IP" 'umask 077 && cat > /tmp/kvb-e2e.env'
ssh "${SSHOPTS[@]}" "ec2-user@$BOX_IP" \
  ". /tmp/kvb-e2e.env && /tmp/s3probe -bucket $BUCKET -out /tmp/s3probe-results.jsonl; rc=\$?; \
   rm -f /tmp/kvb-e2e.env /tmp/s3probe; exit \$rc"

scp "${SSHOPTS[@]}" "ec2-user@$BOX_IP:/tmp/s3probe-results.jsonl" ./s3probe-results.jsonl
echo "[s3e2e] results: bench/rigs/aws-s3e2e/s3probe-results.jsonl"
