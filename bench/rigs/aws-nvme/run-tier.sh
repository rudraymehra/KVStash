#!/usr/bin/env bash
# Rig N tier session (Week-6 gates, i7i.8xlarge): the DAEMON-level NVMe
# numbers, quoted against fio-ceiling.sh's device ceiling (run that FIRST —
# it is the honest denominator).
#
#   T1  NVMe-resident BATCH_GET storm through the full daemon:
#       per-device (one volume) and both-device aggregate GB/s.
#       Quote %-of-fio-ceiling; ≥95% = software-not-the-bottleneck (the A1
#       discipline). A3-as-written (≥6.0 GB/s/device) is expected to read
#       FAIL on any AWS instance-store device — record honestly, A3 OPEN.
#   T2  50-loop kill -9 torture on real NVMe (zero corrupt, zero phantom).
#   T3  warm restart: ~20 GB resident fill → SIGKILL → restart → recovery
#       seconds + seconds-per-GB + hits-survive storm.
#
# Prereqs: provision.sh (ITYPE=i7i.8xlarge) ran; .rig-state exists.
# After: teardown.sh SAME DAY; record numbers in docs/DESIGN.md.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
STATE="$DIR/.rig-state"; source "$STATE"
ROOT="$(git rev-parse --show-toplevel)"
OUT="$DIR/tier-results.txt"; : > "$OUT"
SSHOPTS=(-i "$HOME/.ssh/kvbench.pem" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10
         -o ServerAliveInterval=15 -o ServerAliveCountMax=8 -o TCPKeepAlive=yes)
S() { ssh "${SSHOPTS[@]}" "ec2-user@$N_PUB" "$@"; }
note() { echo "$*" | tee -a "$OUT"; }

echo "[tier] cross-compiling linux/amd64"
BINS=/tmp/kvb-tier; mkdir -p "$BINS"
( cd "$ROOT"
  GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -o "$BINS/kvblockd" ./cmd/kvblockd
  GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -o "$BINS/getbench" ./bench/kvbench/getbench
  GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -tags crashtest -o "$BINS/torture" ./test/crash )
cat > "$BINS/ns.yaml" <<'EOF'
namespaces:
  - name: bench
    id: 3
    token: bench-token
EOF
# A still-running binary makes the overwrite fail with ETXTBSY — kill first.
S 'pkill -x kvblockd 2>/dev/null; pkill -x getbench 2>/dev/null; pkill -x torture 2>/dev/null; sleep 1; true'
scp "${SSHOPTS[@]}" "$BINS/kvblockd" "$BINS/getbench" "$BINS/torture" "$BINS/ns.yaml" "ec2-user@$N_PUB:/tmp/"

echo "[tier] XFS on both instance-store devices"
# umount first (a rerun finds them mounted; mkfs -f refuses a mounted
# device) and DON'T eat mkfs stderr — a silent refusal killed a session.
S 'set -e; for i in 0 1; do d=/dev/nvme$((i+1))n1; m=/mnt/nvme$i;
     sudo umount $m 2>/dev/null || true
     sudo mkfs.xfs -qf $d && sudo mkdir -p $m && sudo mount $d $m && sudo chown ec2-user $m
   done'

# writeCfg <name> <paths-yaml> — 8 GiB arena so a 20 GiB pool is mostly
# NVMe-resident; promotion OFF so the storm measures the DEVICE path.
# admit_min_hits 0: the fill is pure PUTs (zero lifetime GETs) — the
# default 1 would DELETE never-read blocks at the demote watermark instead
# of demoting them (SSD-endurance rule), leaving the pool mostly gone.
# nvme_max_bytes is a RECLAIM reference only; 6 TB (> the single 3.75 TB
# device in T1a) simply means reclaim never fires during these fills —
# intended: the storm must not race reclamation.
writeCfg() { S "cat > /tmp/$1 <<EOF
listen_addr: \"127.0.0.1:9440\"
metrics_addr: \"127.0.0.1:9442\"
namespaces_path: /tmp/ns.yaml
dram_arena_bytes: 8589934592
nvme_paths: [$2]
nvme_max_bytes: 6000000000000
nvme_admit_min_hits: 0
nvme_read_workers: 32
nvme_demote_watermark_pct: 50
nvme_demote_batch_pct: 10
nvme_promote_window_ms: 0
EOF"; }
startd() { S "nohup /tmp/kvblockd -config /tmp/$1 > /tmp/kvbd.log 2>&1 & echo \$! > /tmp/kvbd.pid
             for i in \$(seq 1 400); do curl -sf http://127.0.0.1:9442/healthz >/dev/null && exit 0; sleep 0.25; done; exit 1"; }
stopd() { S 'kill "$(cat /tmp/kvbd.pid)" 2>/dev/null || true; sleep 2' ; }

storm() { # storm <pool-blocks> <label> <fill-mbps>
  # Fill (PUT pool; demotion streams it to NVMe), then the GET storm.
  # The fill is THROTTLED below the device write ceiling: an unthrottled
  # loopback fill outruns demotion and the arena EVICTS the pool instead
  # of demoting it (legal cache behavior; wrong experiment). Ceiling
  # evidence: fio-ceiling.sh (RAW device) + an ad-hoc FILE-level fio on
  # this box (2026-07-19: write bs=1m qd32 ≈ 3.48 GiB/s/device, recorded
  # in tier-results.txt); the committed conservative floor is nvmeprobe's
  # ~2.33 GB/s/device file-level write — the throttles below sit under it.
  S "/tmp/getbench -addr 127.0.0.1:9440 -streams 16 -secs 60 -blocks 32 -block-kib 1024 -pool $1 -fill-mbps $3" \
    | tee -a "$OUT" | sed "s/^/[$2] /"
}

note "== T1a: per-device storm (volume 0 only) =="
writeCfg one.yaml '"/mnt/nvme0/kvb"'
startd one.yaml
storm 20480 "T1a" 2000   # 20 GiB pool over an 8 GiB arena → mostly NVMe-resident
stopd
S 'rm -rf /mnt/nvme0/kvb'

note "== T1b: both-device aggregate storm =="
writeCfg two.yaml '"/mnt/nvme0/kvb", "/mnt/nvme1/kvb"'
startd two.yaml
storm 20480 "T1b" 3500

note "== T3: warm restart on the two-volume fill =="
S 'grep -c . /tmp/kvbd.log >/dev/null; kill -9 "$(cat /tmp/kvbd.pid)"; sleep 1'
T0=$(date +%s.%N)
startd two.yaml
T1=$(date +%s.%N)
note "warm-restart wall (kill -9 → healthz 200): $(echo "$T1 - $T0" | bc)s"
S 'grep "volume recovered" /tmp/kvbd.log' | tee -a "$OUT"
note "-- hits must survive the restart (a warm storm, short):"
storm 20480 "T3" 0 # re-fill is all write-once no-op acks — nothing to throttle
stopd
S 'rm -rf /mnt/nvme0/kvb /mnt/nvme1/kvb'

note "== T2: 50-loop kill -9 torture on real NVMe =="
S '/tmp/torture -loops 50 -dir /mnt/nvme0/tort -bin /tmp/kvblockd' | tee -a "$OUT" | tail -5

note ""
note "== verdict inputs =="
note "fio ceiling: see fio-ceiling.sh output (run separately, same box)."
note "A3-as-written (>=6.0 GB/s/device): compare T1a GB/s; expected FAIL on AWS hardware — record %-of-ceiling and keep A3 OPEN."
note "Zero-corruption gate: T2 must print '0 corrupt reads, 0 phantom keys' over 50 cycles."
echo "[tier] done — results in $OUT. RECORD THE DEMO (asciinema bench/e2e/kill9_demo.sh on this box) ONLY after all three read green, then teardown.sh."
