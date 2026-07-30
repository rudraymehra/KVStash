#!/usr/bin/env bash
# kvbench STORE-NODE provisioner + driver (laptop-side) for the two-node
# real-NIC session: a network-optimized CPU box that runs kvblockd only, in
# the same subnet as the GPU node, reached over the actual NIC.
#
#   bench/rigs/aws-gpu/store-node.sh up          # launch + install + serve
#   bench/rigs/aws-gpu/store-node.sh shape 5     # tc: shape egress to 5 Gbit
#   bench/rigs/aws-gpu/store-node.sh unshape
#   bench/rigs/aws-gpu/store-node.sh iperf       # ceiling FIRST, per rule 2
#   bench/rigs/aws-gpu/store-node.sh restart     # fresh daemon between arms
#
# Why r6in.2xlarge: 64 GiB RAM (one 128k bf16 corpus needs ~18 GiB; c6in's
# 16 GiB cannot hold an arm), 12.5 Gbps baseline / 40 burst, $0.697/hr.
set -euo pipefail
log() { printf '[store %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
die() { log "FATAL: $*"; exit 1; }

export AWS_PROFILE="${AWS_PROFILE:-kvbench}"
REGION=us-east-1
TYPE="${STORE_TYPE:-r6in.2xlarge}"
STATE_DIR="${STATE_DIR:-$HOME/kvbench-dday}"
SSH_KEY="${SSH_KEY:-$STATE_DIR/kvbench.pem}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARENA_BYTES="${STORE_ARENA_BYTES:-34359738368}"   # 32 GiB default
TOKEN="${KVBD_TOKEN:-hf-gpu-token}"

ssh_store() { ssh -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=10 "ubuntu@$(cat "$STATE_DIR/store-ip")" "$@"; }

cmd_up() {
  # Same AZ + subnet as the GPU node: private-IP traffic, one hop, no NAT.
  local gpu_id az subnet
  gpu_id=$(cat "$STATE_DIR/gpu-instance-id") || die "launch the GPU node first"
  read -r az subnet < <(aws ec2 describe-instances --region $REGION --instance-ids "$gpu_id" \
    --query 'Reservations[0].Instances[0].[Placement.AvailabilityZone,SubnetId]' --output text)
  log "GPU node is in $az/$subnet — placing the store node beside it"
  local ami sg
  ami=$(aws ec2 describe-images --region $REGION --owners 099720109477 \
    --filters 'Name=name,Values=ubuntu/images/hvm-ssd-gp3/ubuntu-jammy-22.04-amd64-server-*' \
              'Name=state,Values=available' \
    --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)
  sg=$(aws ec2 describe-security-groups --region $REGION \
    --filters Name=group-name,Values=kvbench-ssh --query 'SecurityGroups[0].GroupId' --output text)
  # The store's data + ops ports are reachable only from inside the SG (i.e.
  # from the GPU node), never from the internet: plaintext KVB1 + bearer token
  # belong on a private segment, which is exactly what we publish.
  aws ec2 authorize-security-group-ingress --region $REGION --group-id "$sg" \
    --protocol tcp --port 9440-9442 --source-group "$sg" 2>/dev/null || true
  # iperf3's ceiling run needs its own port intra-SG (rule 2: ceiling FIRST):
  aws ec2 authorize-security-group-ingress --region $REGION --group-id "$sg" \
    --protocol tcp --port 5201 --source-group "$sg" 2>/dev/null || true

  local iid
  iid=$(aws ec2 run-instances --region $REGION --image-id "$ami" --instance-type "$TYPE" \
    --count 1 --key-name kvbench --security-group-ids "$sg" --subnet-id "$subnet" \
    --associate-public-ip-address --instance-initiated-shutdown-behavior terminate \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":60,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=kvbench,Value=store},{Key=Name,Value=kvbench-store-node}]' \
      'ResourceType=volume,Tags=[{Key=kvbench,Value=store}]' \
    --user-data '#!/bin/bash
shutdown -h +300 "kvbench store-node dead-man"' \
    --query 'Instances[0].InstanceId' --output text) || die "store-node launch failed (capacity? quota is the STANDARD family, separate from G)"
  echo "$iid" > "$STATE_DIR/store-instance-id"
  aws ec2 wait instance-running --region $REGION --instance-ids "$iid"
  aws ec2 describe-instances --region $REGION --instance-ids "$iid" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text > "$STATE_DIR/store-ip"
  aws ec2 describe-instances --region $REGION --instance-ids "$iid" \
    --query 'Reservations[0].Instances[0].PrivateIpAddress' --output text > "$STATE_DIR/store-private-ip"
  log "store node $iid at $(cat "$STATE_DIR/store-ip") (private $(cat "$STATE_DIR/store-private-ip"))"

  for i in $(seq 1 30); do ssh_store true 2>/dev/null && break; sleep 10; done
  log "installing Go + building kvblockd from HEAD, applying the published sysctls"
  local sha; sha=$(git -C "$HERE/../../.." rev-parse HEAD)
  ssh_store "bash -s" <<REMOTE
set -euo pipefail
sudo apt-get -qq update && sudo apt-get -qq install -y git iperf3 build-essential >/dev/null
curl -fsSL https://go.dev/dl/go1.26.5.linux-amd64.tar.gz | sudo tar -C /usr/local -xz
export PATH=/usr/local/go/bin:\$PATH
git clone -q https://github.com/rudraymehra/KVStash.git ~/KVStash
git -C ~/KVStash checkout -q "$sha"
cd ~/KVStash && go build -o ~/kvblockd ./cmd/kvblockd
sudo sysctl -qw net.core.rmem_max=134217728 net.core.wmem_max=134217728 \
  net.ipv4.tcp_rmem="4096 87380 134217728" net.ipv4.tcp_wmem="4096 65536 134217728" \
  net.core.default_qdisc=fq net.ipv4.tcp_congestion_control=bbr
sudo ip link set dev \$(ip -o -4 route show to default | awk '{print \$5}') mtu 9001 || true
mkdir -p ~/kvb
cat > ~/kvb/ns.yaml <<NS
namespaces:
  - { name: vllm, id: 1, token: $TOKEN }
NS
cat > ~/kvb/kvblockd.yaml <<CFG
listen_addr: "0.0.0.0:9440"
metrics_addr: "0.0.0.0:9442"
namespaces_path: "/home/ubuntu/kvb/ns.yaml"
dram_arena_bytes: $ARENA_BYTES
max_conns: 1024
CFG
REMOTE
  cmd_restart
  log "store ready — GPU-side env: EXTERNAL_DAEMON=$(cat "$STATE_DIR/store-private-ip"):9440 EXTERNAL_METRICS=$(cat "$STATE_DIR/store-private-ip"):9442"
}

cmd_restart() {  # fresh daemon per arm: no cross-arm residency, gates stay exact
  # Wait for the OLD daemon to actually die (graceful drain races a bare
  # sleep: the old process would keep the ports and answer the health check,
  # making "fresh daemon" silently false), then assert the NEW pid serves.
  ssh_store 'bash -s' <<'RESTART'
set -euo pipefail
pkill -f "kvblockd -config" 2>/dev/null || true
for i in $(seq 1 20); do pgrep -f "kvblockd -config" >/dev/null || break; sleep 1; [ "$i" = 20 ] && { echo "old daemon refused to die"; exit 1; }; done
nohup ~/kvblockd -config ~/kvb/kvblockd.yaml > ~/kvb/kvbd.log 2>&1 &
NEW=$!
sleep 4
kill -0 "$NEW" 2>/dev/null || { echo "new daemon died at boot:"; tail -5 ~/kvb/kvbd.log; exit 1; }
curl -fsS http://127.0.0.1:9442/healthz >/dev/null
echo "fresh daemon pid $NEW healthy"
# Re-arm the store dead-man for the ongoing session window:
sudo shutdown -c 2>/dev/null || true
sudo shutdown -h +300 "kvbench store-node dead-man (re-armed)" >/dev/null 2>&1
RESTART
  log "daemon restarted (old proven dead, new pid asserted) + dead-man re-armed"
}

cmd_shape() {  # $1 = Gbit; htb+fq preserves the published fq pacing
  local rate="${1:?usage: shape <gbit>}"
  ssh_store "bash -s" <<REMOTE
set -euo pipefail
IF=\$(ip -o -4 route show to default | awk '{print \$5}')
sudo tc qdisc del dev "\$IF" root 2>/dev/null || true
sudo tc qdisc replace dev "\$IF" root handle 1: htb default 10
sudo tc class replace dev "\$IF" parent 1: classid 1:10 htb rate ${rate}gbit ceil ${rate}gbit burst 5mb cburst 5mb quantum 90010
sudo tc qdisc replace dev "\$IF" parent 1:10 handle 10: fq
tc qdisc show dev "\$IF" | head -3
REMOTE
  log "egress shaped to ${rate} Gbit (the reload direction)"
}

cmd_unshape() {
  ssh_store 'IF=$(ip -o -4 route show to default | awk "{print \$5}"); sudo tc qdisc del dev "$IF" root 2>/dev/null || true; tc qdisc show dev "$IF" | head -2'
  log "shaping removed"
}

cmd_iperf() {  # ceiling FIRST, store->GPU direction (the reload direction)
  local gpu_priv
  gpu_priv=$(aws ec2 describe-instances --region $REGION --instance-ids "$(cat "$STATE_DIR/gpu-instance-id")" \
    --query 'Reservations[0].Instances[0].PrivateIpAddress' --output text)
  log "iperf3: store -> GPU ($gpu_priv), 4 streams, 30 s — this is the published denominator"
  ssh -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new "ubuntu@$(cat "$STATE_DIR/gpu-ip")" \
    'pkill iperf3 2>/dev/null; (command -v iperf3 >/dev/null || sudo apt-get -qq install -y iperf3); nohup iperf3 -s -1 >/dev/null 2>&1 &' || true
  sleep 2
  ssh_store "iperf3 -c $gpu_priv -P 4 -l 1M -t 30 -O 3 -J" | tee "$STATE_DIR/iperf3-$(date -u +%H%M%S).json" \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); bps=d["end"]["sum_sent"]["bits_per_second"]; print(f"CEILING: {bps/1e9:.2f} Gbit/s = {bps/8e9:.2f} GB/s")'
}

case "${1:?usage: store-node.sh up|restart|shape <gbit>|unshape|iperf}" in
  up) cmd_up ;;
  restart) cmd_restart ;;
  shape) cmd_shape "${2:-}" ;;
  unshape) cmd_unshape ;;
  iperf) cmd_iperf ;;
  *) die "unknown command: $1" ;;
esac
