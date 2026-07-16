#!/usr/bin/env bash
# Rig T: run xferspike server on B, client sweep on A over the private network.
# The A1 verdict: xferspike GB/s vs the iperf3 ceiling (gate: >=85%).
set -euo pipefail
STATE="$(dirname "$0")/.rig-state"; source "$STATE"
SSH="ssh -i $HOME/.ssh/kvbench.pem -o StrictHostKeyChecking=accept-new ec2-user"
ROOT="$(git rev-parse --show-toplevel)"; OUT="$(dirname "$0")/xferspike-results.jsonl"; : > "$OUT"

echo "[run] cross-compiling xferspike for linux/amd64"
( cd "$ROOT" && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -o /tmp/xferspike-linux ./cmd/xferspike )
for host in "$A_PUB" "$B_PUB"; do scp -i $HOME/.ssh/kvbench.pem -o StrictHostKeyChecking=accept-new /tmp/xferspike-linux "ec2-user@$host:/tmp/xferspike"; done

$SSH@"$B_PUB" 'pkill xferspike 2>/dev/null; nohup /tmp/xferspike --mode=server --addr=0.0.0.0:9999 >/tmp/xf-srv.log 2>&1 &'; sleep 2
for streams in 8 16 32 64; do
  for fmb in 1 4 16; do
    fb=$((fmb*1048576))
    $SSH@"$A_PUB" "/tmp/xferspike --mode=client --addr=$B_PRIV:9999 --streams=$streams --frame-bytes=$fb --duration=30s --sndbuf=$((16*1024*1024))" \
      | tee -a "$OUT"
  done
done
$SSH@"$B_PUB" 'pkill xferspike 2>/dev/null || true'
echo "[run] results in $OUT — compare gbytes_per_s*8 vs the iperf3 Gbit/s ceiling"
