#!/usr/bin/env bash
# Rig S3S collector: pull the spill-soak artifacts, print the verdict inputs
# (leak indicators + the kvb_s3_* activity record), and TERMINATE the box
# (idempotent — safe if the 27h dead-man already fired).
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
STATE="$DIR/.rig-state"; source "$STATE"
SSHOPTS=(-i "$HOME/.ssh/kvbench.pem" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10)
OUT="$DIR/results"
mkdir -p "$OUT"

if aws ec2 describe-instances --region "$REGION" --instance-ids "$IID" \
     --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null | grep -q running; then
  echo "[collect] pulling artifacts from $PUB"
  # Final scrape BEFORE stopping anything — the s3 counters die with the
  # daemon; then stop driver → daemon → minio gracefully.
  ssh "${SSHOPTS[@]}" "ec2-user@$PUB" \
    'curl -s http://127.0.0.1:9442/metrics | grep -E "^(kvb_|go_goroutines|process_resident)" > /var/soakout/metrics-final.txt || true
     kill -TERM $(cat /var/soakout/driver.pid) 2>/dev/null || true; sleep 5
     kill -TERM $(cat /var/soakout/daemon.pid) 2>/dev/null || true; sleep 3
     kill -TERM $(cat /var/soakout/minio.pid) 2>/dev/null || true' || true
  if ! scp -r "${SSHOPTS[@]}" "ec2-user@$PUB:/var/soakout/*" "$OUT/"; then
    # NEVER tear down on a failed pull — a transient ssh drop must not
    # destroy the night's artifacts. Re-run collect.sh; only the 27h
    # dead-man may kill an uncollected box.
    echo "[collect] PULL FAILED — box left RUNNING; fix connectivity and re-run collect.sh" >&2
    exit 1
  fi
else
  # The dead-man TERMINATES the instance and DeleteOnTermination removes the
  # root volume — a fired dead-man means the artifacts are GONE. There is no
  # recovery branch to write; collect within the 27h window or lose the run.
  echo "[collect] instance not running — dead-man fired; artifacts are unrecoverable (terminate deletes the volume)" >&2
  exit 1
fi

echo "[collect] terminating $IID"
aws ec2 terminate-instances --region "$REGION" --instance-ids "$IID" >/dev/null 2>&1 || true
aws ec2 wait instance-terminated --region "$REGION" --instance-ids "$IID" 2>/dev/null || true
aws ec2 delete-security-group --region "$REGION" --group-id "$SGID" 2>/dev/null || true
LEFT=$(aws ec2 describe-instances --region "$REGION" \
  --filters Name=tag:kvbench,Values=s3soak Name=instance-state-name,Values=running,pending,stopping,stopped \
  --query 'Reservations[].Instances[].InstanceId' --output text)
echo "[collect] teardown done; residual s3soak instances: '${LEFT:-none}'"

echo "--- final soak line:"
tail -1 "$OUT/soak.jsonl" 2>/dev/null
echo "--- RSS samples (KiB, first/last):"
head -1 "$OUT/rss.log" 2>/dev/null; tail -1 "$OUT/rss.log" 2>/dev/null
echo "--- s3 activity (final scrape):"
grep -E "^kvb_s3_|tier=\"s3\"" "$OUT/metrics-final.txt" 2>/dev/null || echo "(no metrics-final.txt)"
echo "--- goroutine count first/last hourly snapshot (leak check):"
for f in $(ls "$OUT"/goro-*.txt 2>/dev/null | sort | sed -n '1p;$p'); do
  echo "$f: $(head -1 "$f")"
done
echo "--- minio data growth (bytes, first/last):"
head -1 "$OUT/minio-usage.log" 2>/dev/null; tail -1 "$OUT/minio-usage.log" 2>/dev/null
