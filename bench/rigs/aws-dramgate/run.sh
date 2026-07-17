#!/usr/bin/env bash
# Rig G: the three DRAM-tier measured gates on ONE Linux box (same-host,
# loopback — the gates are relative-to-same-host by design):
#   G1 throughput : getbench vs xferspike same-shape ceiling, >= 0.9x
#   G2 exists p99 : 512-key BATCH_EXISTS < 1ms under 8 GET lanes, 60s
#   G3 zero-alloc : /debug/pprof/allocs under the GET storm -> no blob-band
#                   (0.4-2.5MB) alloc sites on the GET path
# One c7i.4xlarge (16 vCPU / 32 GiB), spot first, on-demand fallback.
# Everything cross-compiled locally; the box needs no toolchain.
# All resources tagged kvbench=dramgate; teardown at the end of THIS script.
set -euo pipefail

REGION="${REGION:-us-east-1}"
ITYPE="${ITYPE:-c7i.4xlarge}"
SG="${SG:-kvbench-dramgate-sg}"     # SSH-only; created if absent, deleted at teardown
KEY="${KEY:-kvbench}"
DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(git rev-parse --show-toplevel)"
OUT="$DIR/gate-results.txt"
SSHOPTS=(-i "$HOME/.ssh/kvbench.pem" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10)

trap 'echo "[rig-g] FAILED — check for a billing instance: aws ec2 describe-instances --region '"$REGION"' --filters Name=tag:kvbench,Values=dramgate Name=instance-state-name,Values=running,pending" >&2' ERR

echo "[rig-g] cross-compiling linux/amd64 binaries"
BINS=/tmp/kvb-dramgate; mkdir -p "$BINS"
( cd "$ROOT"
  GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -o "$BINS/kvblockd"  ./cmd/kvblockd
  GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -o "$BINS/getbench"  ./bench/kvbench/getbench
  GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -o "$BINS/xferspike" ./cmd/xferspike
  GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test -tags integration -c -o "$BINS/exists_gate" ./test/integration/ )
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

echo "[rig-g] launching $ITYPE (spot first)"
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
echo "[rig-g] instance $IID — waiting for running + ssh"
aws ec2 wait instance-running --region "$REGION" --instance-ids "$IID"
PUB=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$IID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
for i in $(seq 1 30); do
  ssh "${SSHOPTS[@]}" "ec2-user@$PUB" true 2>/dev/null && break
  sleep 5
done

teardown() {
  echo "[rig-g] terminating $IID"
  aws ec2 terminate-instances --region "$REGION" --instance-ids "$IID" >/dev/null
  aws ec2 wait instance-terminated --region "$REGION" --instance-ids "$IID"
  aws ec2 delete-security-group --region "$REGION" --group-id "$SGID" 2>/dev/null || \
    echo "[rig-g] SG $SGID not deleted (still referenced?) — SGs are free, but tidy up later"
  LEFT=$(aws ec2 describe-instances --region "$REGION" \
    --filters Name=tag:kvbench,Values=dramgate Name=instance-state-name,Values=running,pending,stopping,stopped \
    --query 'Reservations[].Instances[].InstanceId' --output text)
  echo "[rig-g] teardown done; residual dramgate instances: '${LEFT:-none}'"
}
trap teardown EXIT

echo "[rig-g] shipping binaries"
scp "${SSHOPTS[@]}" "$BINS"/kvblockd "$BINS"/getbench "$BINS"/xferspike "$BINS"/exists_gate "$BINS"/ns.yaml "ec2-user@$PUB:/tmp/"

echo "[rig-g] running gates on the box (logs stream below)"
ssh "${SSHOPTS[@]}" "ec2-user@$PUB" 'bash -s' <<'REMOTE' | tee "$OUT"
set -euo pipefail
sudo sysctl -qw net.core.rmem_max=67108864 net.core.wmem_max=67108864
cd /tmp
echo "=== box: $(nproc) vCPU, $(free -g | awk "/Mem:/{print \$2}") GiB, $(uname -r) ==="

# --- daemon + ceiling server up for the whole session ---
KVBLOCKD_DRAM_ARENA_BYTES=6442450944 KVBLOCKD_METRICS_ADDR=127.0.0.1:19442 \
  ./kvblockd -listen 127.0.0.1:19440 -namespaces /tmp/ns.yaml 2>/tmp/daemon.log &
DPID=$!
./xferspike -mode server -addr 127.0.0.1:19999 >/dev/null 2>&1 &
SPID=$!
sleep 8   # 6 GiB prefault

echo "=== G1: interleaved pairs (xferspike ceiling | getbench vs DRAM daemon), 4 GiB working set ==="
for i in 1 2 3; do
  C=$(./xferspike -mode client -addr 127.0.0.1:19999 -streams 8 -frame-bytes 4194304 -duration 10s 2>/dev/null \
      | python3 -c "import json,sys;print(f\"{json.load(sys.stdin)['gbytes_per_s']:.2f}\")")
  sleep 2
  G=$(./getbench -addr 127.0.0.1:19440 -streams 8 -blocks 32 -block-kib 1024 -pool 4096 -secs 10 -noverify 2>/dev/null | tail -1)
  echo "pair $i: ceiling ${C} GB/s | ${G}"
  sleep 2
done

echo "=== G3: allocs profile under a 40s GET storm (15s capture) ==="
./getbench -addr 127.0.0.1:19440 -streams 8 -blocks 32 -block-kib 1024 -pool 4096 -secs 40 -noverify >/tmp/storm.log 2>&1 &
GP=$!
sleep 5
curl -s -o /tmp/allocs.pb.gz 'http://127.0.0.1:19442/debug/pprof/allocs?seconds=15'
curl -s 'http://127.0.0.1:19442/healthz'; echo " <- healthz"
wait $GP; tail -1 /tmp/storm.log
kill -TERM $DPID $SPID 2>/dev/null; sleep 2

echo "=== G2: 512-key EXISTS p99 under 8 GET lanes, 60s (in-process daemon) ==="
./exists_gate -test.v -test.run TestExistsLatencyUnderGetLoad 2>&1 | tail -4
REMOTE

echo "[rig-g] pulling the allocs profile"
scp "${SSHOPTS[@]}" "ec2-user@$PUB:/tmp/allocs.pb.gz" "$DIR/allocs-linux.pb.gz"
echo "[rig-g] done — results in $OUT, profile in $DIR/allocs-linux.pb.gz"
