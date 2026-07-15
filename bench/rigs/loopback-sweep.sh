#!/usr/bin/env bash
# Local loopback sweep for xferspike — a developer-machine sanity check, NOT the
# A1 gate (that is bench/rigs/aws-transport on the c6in pair, Day 4).
#
# Raises the fd limit and spaces runs so rapid sequential loopback runs don't
# exhaust ephemeral ports / pile up TIME_WAIT sockets (finding F-a1-2). Emits CSV.
#
# Usage: bench/rigs/loopback-sweep.sh [duration] [port]
set -euo pipefail

DUR="${1:-2s}"
PORT="${2:-9990}"
BIN="$(mktemp -t xferspike.XXXX)"
ADDR="127.0.0.1:${PORT}"

ulimit -n 8192 || true

cd "$(git rev-parse --show-toplevel)"
go build -o "$BIN" ./cmd/xferspike

"$BIN" --mode=server --addr="$ADDR" >/dev/null 2>&1 &
SRV=$!
trap 'kill "$SRV" 2>/dev/null || true; rm -f "$BIN"' EXIT
sleep 1

echo "streams,frame_mb,gbytes_per_s,gbit_per_s,cpu_cores_sender,status"
for streams in 1 4 16 32; do
  for fmb in 1 4 16; do
    fb=$((fmb * 1048576))
    if out=$("$BIN" --mode=client --addr="$ADDR" --streams="$streams" \
              --frame-bytes="$fb" --duration="$DUR" 2>/dev/null) && [ -n "$out" ]; then
      echo "$out" | python3 -c "import json,sys;d=json.load(sys.stdin);print(f\"$streams,$fmb,{d['gbytes_per_s']:.2f},{d['gbit_per_s']:.1f},{d['cpu_cores_sender']:.2f},ok\")"
    else
      echo "$streams,$fmb,,,,ERR"
    fi
    sleep 2  # let loopback sockets drain out of TIME_WAIT between cells
  done
done
# NOTE: 64-stream loopback hits macOS ENOBUFS (kernel loopback buffer limit),
# a Mac artifact absent on the Linux gate rig; omitted here on purpose.
