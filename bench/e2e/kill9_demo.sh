#!/usr/bin/env bash
# kill9_demo.sh — the durability demo, asciinema-friendly.
#
# What it shows: a kvblockd daemon under a real PUT storm (0.4–2.5 MiB
# blocks, ~30% of streams deliberately abandoned mid-transfer), SIGKILLed
# mid-write, restarted — and the recovered cache serves ONLY verified bytes:
# zero corrupt reads, zero phantom keys, honest loss for whatever hadn't
# reached a flushed NVMe record.
#
# Usage: bench/e2e/kill9_demo.sh [scratch-dir] [storm-seconds]
#   Run from the repo root. Real numbers come from real NVMe (the rig);
#   laptops demonstrate the MECHANISM, not the throughput.
set -euo pipefail

DIR="${1:-$(mktemp -d /tmp/kvb-kill9-demo.XXXX)}"
STORM_S="${2:-8}"

say() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }

say "kvblockd kill -9 demo — scratch: $DIR, storm: ${STORM_S}s"
say "1/3  build the daemon + the torture driver (journal-after-ack oracle)"
go build -o "$DIR/kvblockd" ./cmd/kvblockd
go build -tags crashtest -o "$DIR/torture" ./test/crash
echo "built."

say "2/3  storm → kill -9 mid-write → restart → verify (watch the recovery line)"
# One cycle, fixed long storm. The driver prints:
#   - the daemon's own "nvme: volume recovered ..." line (RecoveryReport),
#   - the verify line: survived/lost/corrupt/phantom counts,
#   - the PASS verdict with the recovery-to-ready gate.
"$DIR/torture" -loops 1 -storm-ms "$((STORM_S * 1000))" -dir "$DIR/data" -bin "$DIR/kvblockd"

say "3/3  the claim"
cat <<'EOF'
Every block the recovered daemon still admits to EXISTS was re-read and
verified byte-for-byte (xxh3) against what the client wrote — 0 corrupt.
Every stream that never got its COMMIT ack is absent — 0 phantom.
Blocks that only lived in DRAM died with the process — honest loss,
never corruption. That is the whole durability contract of a cache.
EOF
