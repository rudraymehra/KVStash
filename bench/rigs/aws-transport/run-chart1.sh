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
SSHO=(-i "$HOME/.ssh/kvbench.pem" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10
      -o ServerAliveInterval=15 -o ServerAliveCountMax=8 -o TCPKeepAlive=yes)
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
# A still-running binary makes the overwrite fail with ETXTBSY — kill first.
A 'pkill -x kvbench 2>/dev/null; sleep 1; true'
B 'pkill -x kvblockd 2>/dev/null; pkill -x kvbench 2>/dev/null; sleep 1; true'
for h in "$A_PUB" "$B_PUB"; do scp "${SSHO[@]}" "$BIN/kvbench" "ec2-user@$h:/tmp/"; done
scp "${SSHO[@]}" "$BIN/kvblockd" "$BIN/ns.yaml" "$BIN/kvbd.yaml" "ec2-user@$B_PUB:/tmp/"
# The baselines dir (compose + drivers) MUST be on node B for docker + redis-py.
scp -r "${SSHO[@]}" "$ROOT/bench/baselines" "ec2-user@$B_PUB:/tmp/baselines"

# runcell <store> <port> <metrics-url-or-empty>: the store's OWN address +
# its OWN metrics source (kvblockd exposes /metrics; Redis/Valkey don't — an
# empty metrics arg means client-cores only). A cell failure ABORTS (rule 4:
# never silently drop a bar).
#
# The sweep is launched DETACHED on node A and polled with short fresh
# connections: a live ssh held across a NIC-saturating cell starves and
# dies with the link (observed twice — the sweep drowns its own control
# channel).
runcell() { # runcell <store> <port> <metrics-url-or-empty> [cell-id-filter]
  local store="$1" port="$2" metrics="$3" filt="${4:-}"; local mflag="" fflag=""
  [ -n "$metrics" ] && mflag="--daemon-metrics $metrics"
  [ -n "$filt" ] && fflag="--filter $filt"
  # --open=false: Chart 1 is the closed-GET bar matrix. The open-loop
  # latency sweep is its own session — 6 extra rates per cell would put
  # ~2h per store run on this box.
  A "rm -f /tmp/cell.done /tmp/cell.log; nohup sh -c '/tmp/kvbench sweep --headline --open=false $fflag --addr $B_PRIV:$port --target $store $mflag \
      --ceiling-gbytes $CEIL_GB --rig rig-t --duration 60s --warmup 10s --out /tmp/$store.jsonl \
      >>/tmp/cell.log 2>&1; echo \$? > /tmp/cell.done' >/dev/null 2>&1 &"
  local ok=""
  for i in $(seq 1 180); do
    sleep 10
    if A "test -f /tmp/cell.done" 2>/dev/null; then ok=1; break; fi
  done
  if [ -z "$ok" ]; then
    echo "[chart1] cell $store never finished (30 min cap) — killing the detached sweep"
    A "pkill -x kvbench 2>/dev/null; tail -5 /tmp/cell.log" || true
    exit 1
  fi
  # One un-retried ssh here once aborted a whole session on a transient
  # drop — retry the rc read once before giving up.
  local rc; rc=$(A "cat /tmp/cell.done" 2>/dev/null) || { sleep 5; rc=$(A "cat /tmp/cell.done"); }
  [ "$rc" = "0" ] || { echo "[chart1] cell $store FAILED rc=$rc; log tail:"; A "tail -8 /tmp/cell.log" || true; exit 1; }
  scp "${SSHO[@]}" "ec2-user@$A_PUB:/tmp/$store.jsonl" "$RESULTS/"
}

if [ -z "${SKIP_KVBLOCKD:-}" ]; then
  echo "[chart1] === kvblockd (DRAM) ==="
  A 'rm -f /tmp/kvblockd.jsonl'
  B 'pkill kvblockd 2>/dev/null; nohup /tmp/kvblockd -config /tmp/kvbd.yaml >/tmp/kvbd.log 2>&1 &'
  B 'for i in $(seq 1 120); do curl -sf http://127.0.0.1:9442/healthz >/dev/null && break; sleep 0.5; done'
  for run in 1 2 3; do runcell kvblockd 9440 "http://$B_PRIV:9442/metrics"; done  # median of 3
  B 'pkill kvblockd 2>/dev/null || true'
fi

echo "[chart1] === baselines: Redis 7 / Valkey 8 (go-redis zero-copy, port 6379) ==="
# Baselines run stream points {8,32,64} only: single-threaded redis serves
# ~1.2 GB/s, so each cell costs ~4 min (fill dominates) — the full 7-point
# sweep × 3 runs would be ~3h/store. Three points keep median-of-3 on every
# PUBLISHED baseline cell (the bar + a scaling sketch); disclosed in the
# ledger + chart conditions. kvblockd keeps the full 7-point curve.
for svc in ${BASELINE_SVCS:-redis valkey}; do
  A "rm -f /tmp/$svc.jsonl"
  B "cd /tmp && docker compose -f baselines/docker-compose.yml --profile $svc up -d"
  B 'for i in $(seq 1 60); do redis-cli -p 6379 ping 2>/dev/null | grep -q PONG && break; sleep 0.5; done'
  for run in 1 2 3; do
    for sp in _s8_ _s32_ _s64_; do runcell "$svc" 6379 "" "$sp"; done
  done
  B "cd /tmp && docker compose -f baselines/docker-compose.yml --profile $svc down"
done

echo "[chart1] === redis-py bar (the LMCache-shipped path, closed GET throughput) ==="
B "cd /tmp && docker compose -f baselines/docker-compose.yml --profile redis up -d"
B 'for i in $(seq 1 60); do redis-cli -p 6379 ping 2>/dev/null | grep -q PONG && break; sleep 0.5; done'
# python3.11: redis-py 8.0.1 (the pin LMCache users actually resolve)
# requires Python >=3.10 — AL2023's default python3 is 3.9.
B 'rm -f /tmp/redis-py.jsonl'
for blob in 462848 2621440; do
  for run in 1 2 3; do
    B "cd /tmp && python3.11 baselines/redis_py_driver/driver.py --getbench --addr 127.0.0.1:6379 \
        --blob-bytes $blob --streams 32 --secs 30 --out /tmp/redis-py.jsonl"
  done
done
scp "${SSHO[@]}" "ec2-user@$B_PUB:/tmp/redis-py.jsonl" "$RESULTS/"
B "cd /tmp && docker compose -f baselines/docker-compose.yml --profile redis down"

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
