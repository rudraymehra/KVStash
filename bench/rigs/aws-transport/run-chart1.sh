#!/usr/bin/env bash
# Rig T — Chart #1 data: the kvbench GET matrix through the full daemon vs
# every baseline, on the SAME host pair, same sysctls, same payloads, warmed
# identically. Emits the >=10x-vs-redis-py verdict. run.sh (xferspike) still
# owns the A1 transport ceiling; this owns the storage comparison.
#
# Prereqs: provision.sh + tune.sh + iperf-ceiling.sh already run (.rig-state
# exists, iperf-ceiling.txt has the ceiling). Daemon + baselines run on node B,
# kvbench client on node A over the private NIC.
#
# After: teardown.sh (asserts zero tagged resources).
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
STATE="$DIR/.rig-state"; source "$STATE"
ROOT="$(git rev-parse --show-toplevel)"
RESULTS="$ROOT/bench/results/rig-t"; mkdir -p "$RESULTS"
SSHO=(-i "$HOME/.ssh/kvbench.pem" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10)
A() { ssh "${SSHO[@]}" "ec2-user@$A_PUB" "$@"; }
B() { ssh "${SSHO[@]}" "ec2-user@$B_PUB" "$@"; }

# The measured transport ceiling from iperf-ceiling.sh (best Gbit/s → GB/s).
CEIL_GBIT=$(awk '{print $1}' "$DIR/iperf-ceiling.txt" 2>/dev/null | sort -n | tail -1)
CEIL_GB=$(awk "BEGIN{print ${CEIL_GBIT:-0}/8}")
echo "[chart1] transport ceiling = ${CEIL_GBIT:-?} Gbit/s = ${CEIL_GB} GB/s"

echo "[chart1] cross-compiling"
BIN=/tmp/kvb-chart1; mkdir -p "$BIN"
( cd "$ROOT"
  GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -o "$BIN/kvblockd" ./cmd/kvblockd
  GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -o "$BIN/kvbench" ./bench/kvbench )
cat > "$BIN/ns.yaml" <<'EOF'
namespaces:
  - { name: bench, id: 3, token: bench-token }
EOF
cat > "$BIN/kvbd.yaml" <<EOF
listen_addr: "0.0.0.0:9440"
metrics_addr: "0.0.0.0:9442"
namespaces_path: /tmp/ns.yaml
dram_arena_bytes: 34359738368
sock_sndbuf: 16777216
sock_rcvbuf: 16777216
EOF
for h in "$A_PUB" "$B_PUB"; do scp "${SSHO[@]}" "$BIN/kvbench" "ec2-user@$h:/tmp/"; done
scp "${SSHO[@]}" "$BIN/kvblockd" "$BIN/ns.yaml" "$BIN/kvbd.yaml" "ec2-user@$B_PUB:/tmp/"
# The baselines dir (compose + drivers) MUST be on node B for docker + redis-py.
scp -r "${SSHO[@]}" "$ROOT/bench/baselines" "ec2-user@$B_PUB:/tmp/baselines"

# runcell <store> <port> <metrics-url-or-empty>: the store's OWN address +
# its OWN metrics source (kvblockd exposes /metrics; Redis/Valkey don't — an
# empty metrics arg means client-cores only). A cell failure ABORTS (rule 4:
# never silently drop a bar).
runcell() {
  local store="$1" port="$2" metrics="$3"; local mflag=""
  [ -n "$metrics" ] && mflag="--daemon-metrics $metrics"
  A "/tmp/kvbench sweep --headline --addr $B_PRIV:$port --target $store $mflag \
      --ceiling-gbytes $CEIL_GB --rig rig-t --duration 60s --warmup 10s --out /tmp/$store.jsonl"
  scp "${SSHO[@]}" "ec2-user@$A_PUB:/tmp/$store.jsonl" "$RESULTS/"
}

echo "[chart1] === kvblockd (DRAM) ==="
B 'pkill kvblockd 2>/dev/null; nohup /tmp/kvblockd -config /tmp/kvbd.yaml >/tmp/kvbd.log 2>&1 &'
B 'for i in $(seq 1 120); do curl -sf http://127.0.0.1:9442/healthz >/dev/null && break; sleep 0.5; done'
for run in 1 2 3; do runcell kvblockd 9440 "http://$B_PRIV:9442/metrics"; done  # median of 3
B 'pkill kvblockd 2>/dev/null || true'

echo "[chart1] === baselines: Redis 7 / Valkey 8 (go-redis zero-copy, port 6379) ==="
for svc in redis valkey; do
  B "cd /tmp && docker compose -f baselines/docker-compose.yml --profile $svc up -d"
  B 'for i in $(seq 1 60); do redis-cli -p 6379 ping 2>/dev/null | grep -q PONG && break; sleep 0.5; done'
  for run in 1 2 3; do runcell "$svc" 6379 ""; done  # Redis has no /metrics; client cores only
  B "docker compose -f baselines/docker-compose.yml --profile $svc down"
done

echo "[chart1] === redis-py bar (the LMCache-shipped path, closed GET throughput) ==="
B "docker compose -f baselines/docker-compose.yml --profile redis up -d"
B 'for i in $(seq 1 60); do redis-cli -p 6379 ping 2>/dev/null | grep -q PONG && break; sleep 0.5; done'
for blob in 462848 2621440; do
  for run in 1 2 3; do
    B "python3 baselines/redis_py_driver/driver.py --getbench --addr 127.0.0.1:6379 \
        --blob-bytes $blob --streams 32 --secs 30 --out /tmp/redis-py.jsonl"
  done
done
scp "${SSHO[@]}" "ec2-user@$B_PUB:/tmp/redis-py.jsonl" "$RESULTS/"
B "docker compose -f baselines/docker-compose.yml --profile redis down"

echo "[chart1] === the >=10x-vs-redis-py verdict (MEDIAN of runs, rule 7) ==="
python3 - "$RESULTS" <<'PY'
import glob, json, sys, os, statistics
d = sys.argv[1]
def med(store):
    vals=[]
    for f in glob.glob(os.path.join(d,"*.jsonl")):
        for line in open(f):
            line=line.strip()
            if not line: continue
            r=json.loads(line)
            if r.get("store")==store and r.get("cell",{}).get("mode")=="closed" \
               and r.get("cell",{}).get("mix","get")=="get":
                vals.append(r.get("goodput_gbytes_s",0))
    return statistics.median(vals) if vals else 0
kvb=med("kvblockd"); rpy=med("redis-py")
if kvb and rpy:
    mult=kvb/rpy
    verdict='PASS' if mult>=10 else f'REPORT HONEST {mult:.1f}x (below 10x — profile, do NOT massage)'
    print(f"GATE A-10x: kvblockd {kvb:.2f} GB/s vs redis-py {rpy:.2f} GB/s = {mult:.1f}x -> {verdict}")
else:
    print("GATE A-10x: incomplete — need median kvblockd AND redis-py closed-GET bars")
PY
echo "[chart1] done — JSONL in $RESULTS; render: python bench/report/plot.py chart1 --in bench/results/rig-t/*.jsonl --out chart1.png. Then teardown.sh."
