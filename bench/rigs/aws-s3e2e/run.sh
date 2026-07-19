#!/usr/bin/env bash
# Real-S3 e2e: run s3probe ON an in-region box (us-east-1) against the
# 1-day-lifecycle e2e bucket, pull the JSONL back, and print the summary.
# Credentials are the SCOPED kvb-e2e IAM user (bucket-only policy), passed
# as env for the one command — never written to the box's disk or this repo.
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
BOX_IP="${BOX_IP:?set BOX_IP to the in-region box's public IP}"
CREDS_JSON="${CREDS_JSON:-/tmp/kvb-e2e-creds.json}"
SSHOPTS=(-i "$HOME/.ssh/kvbench.pem" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10)

AKID=$(python3 -c "import json;print(json.load(open('$CREDS_JSON'))['AccessKey']['AccessKeyId'])")
SECRET=$(python3 -c "import json;print(json.load(open('$CREDS_JSON'))['AccessKey']['SecretAccessKey'])")

GOOS=linux GOARCH=arm64 go build -o /tmp/s3probe-arm64 .
scp "${SSHOPTS[@]}" /tmp/s3probe-arm64 "ec2-user@$BOX_IP:/tmp/s3probe"

# Env on the command line only; the probe writes results to /tmp on the box.
ssh "${SSHOPTS[@]}" "ec2-user@$BOX_IP" \
  "AWS_ACCESS_KEY_ID=$AKID AWS_SECRET_ACCESS_KEY=$SECRET AWS_REGION=us-east-1 \
   /tmp/s3probe -bucket $BUCKET -out /tmp/s3probe-results.jsonl && rm -f /tmp/s3probe"

scp "${SSHOPTS[@]}" "ec2-user@$BOX_IP:/tmp/s3probe-results.jsonl" ./s3probe-results.jsonl
echo "[s3e2e] results: bench/rigs/aws-s3e2e/s3probe-results.jsonl"
