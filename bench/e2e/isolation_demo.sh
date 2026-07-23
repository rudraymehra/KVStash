#!/usr/bin/env bash
# isolation_demo.sh — two tenants, one daemon: isolation is structural.
#
# What it shows, live, in under a minute on a laptop:
#   1. Identity is (namespace, key): tenant A stores a block; tenant B asking
#      for the SAME key gets NOT_FOUND — not FORBIDDEN, not a permission
#      error. There is no cross-tenant read path to guard, and no existence
#      oracle to probe.
#   2. Quotas are per-tenant walls: A fills its DRAM quota and gets
#      ERR_QUOTA_BYTES; B keeps writing through the same daemon, unaffected.
#
# No GPUs, no external deps: go build + kvbctl + bash + curl. Companion to
# kill9_demo.sh (durability); this one is a planned security post's runnable claim.
#
# Usage: bench/e2e/isolation_demo.sh [scratch-dir]
#   Run from the repo root. Exits 0 with a PASS summary, nonzero with the
#   first violated claim and the daemon log tail.
set -euo pipefail

AUTO_DIR=0
if [ $# -ge 1 ]; then
  DIR="$1"
  mkdir -p "$DIR"
else
  DIR="$(mktemp -d /tmp/kvb-isolation-demo.XXXX)"
  AUTO_DIR=1
fi

# All three listeners sit off their defaults (9440/9441/9442) so the demo
# never collides with a real daemon — and the preflight below refuses to run
# if anything already listens on the demo ports.
ADDR="127.0.0.1:19440"  # data plane
ADMIN="127.0.0.1:19441" # admin/ops plane (namespace + quota admin)
OPS="127.0.0.1:19442"   # /metrics + /healthz

TOKEN_A="demo-token-a"
TOKEN_B="demo-token-b"
QUOTA_A=$((4 * 1024 * 1024)) # tenant A: 4 MiB DRAM quota — a tight wall we can hit fast
BLOCK=$((1 * 1024 * 1024))   # 1 MiB blocks

DAEMON_PID=""
say() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
die() {
  printf '\033[1mFAIL: %s\033[0m\n' "$*" >&2
  if [ -s "$DIR/kvblockd.log" ]; then
    echo "--- kvblockd log tail ---" >&2
    tail -20 "$DIR/kvblockd.log" >&2
  fi
  exit 1
}
cleanup() {
  if [ -n "$DAEMON_PID" ]; then
    kill "$DAEMON_PID" 2>/dev/null || true
    wait "$DAEMON_PID" 2>/dev/null || true
  fi
  # Plain `[ ... ] && rm` would make this EXIT trap return 1 when AUTO_DIR=0
  # (caller-supplied dir), turning a clean PASS into exit 1 under set -e.
  if [ "$AUTO_DIR" = 1 ]; then
    rm -rf "$DIR"
  fi
}
trap cleanup EXIT

START_S=$(date +%s)

say "kvblockd isolation demo — scratch: $DIR"

say "1/4  build kvblockd + kvbctl"
go build -o "$DIR/kvblockd" ./cmd/kvblockd
go build -o "$DIR/kvbctl" ./cmd/kvbctl
echo "built."

say "2/4  start one daemon with two namespaces (per-tenant DRAM quotas)"
# Preflight: if anything already listens on a demo port, every claim below
# could be measured against the wrong daemon — refuse to start instead.
for A_PORT in "$ADDR" "$ADMIN" "$OPS"; do
  if (exec 3<>"/dev/tcp/${A_PORT%%:*}/${A_PORT##*:}") 2>/dev/null; then
    die "port $A_PORT is already listening — another kvblockd (or something else) holds the demo ports; kill it or set different ports"
  fi
done
cat >"$DIR/namespaces.yaml" <<EOF
# Two tenants, one daemon. ids key on-disk ownership; tokens are the whole
# identity — a connection lives inside its namespace from HELLO onward.
namespaces:
  - { name: tenant-a, id: 1, token: "$TOKEN_A", quota_dram: $QUOTA_A }
  - { name: tenant-b, id: 2, token: "$TOKEN_B", quota_dram: $((32 * 1024 * 1024)) }
EOF
cat >"$DIR/config.yaml" <<EOF
listen_addr: "$ADDR"
admin_addr: "$ADMIN" # off the default 127.0.0.1:9441 — see the port comment above
metrics_addr: "$OPS"
dram_arena_bytes: 134217728 # 128 MiB — quotas bite long before the arena does
namespaces_path: "$DIR/namespaces.yaml"
EOF
"$DIR/kvblockd" --config "$DIR/config.yaml" >"$DIR/kvblockd.log" 2>&1 &
DAEMON_PID=$!
scripts/wait-healthz.sh "$OPS" 15 >/dev/null || die "daemon never became healthy"
# healthz answering is NOT enough — assert the answer came from OUR child.
# If it lost a bind race (a daemon grabbed the ports between preflight and
# launch), a pre-existing kvblockd would serve every request below and the
# demo would "PASS" against the wrong process. Our child prints "kvblockd
# listening" only after every listener bound; a lost race prints "address
# already in use" and exits — poll (<=5s) until one of the two happens.
OURS=0
for _ in $(seq 1 50); do
  if grep -q "bind: address already in use" "$DIR/kvblockd.log"; then
    die "another kvblockd holds the demo ports — kill it or set different ports"
  fi
  if ! kill -0 "$DAEMON_PID" 2>/dev/null; then
    die "our kvblockd exited at startup (healthz is answering anyway) — another kvblockd holds the demo ports; kill it or set different ports"
  fi
  if grep -q "kvblockd listening" "$DIR/kvblockd.log"; then
    OURS=1
    break
  fi
  sleep 0.1
done
if [ "$OURS" != 1 ]; then
  die "cannot confirm the healthz answer came from our daemon (no 'kvblockd listening' in its log) — is another kvblockd holding the demo ports?"
fi
echo "daemon up on $ADDR (tenant-a quota: $((QUOTA_A / 1024 / 1024)) MiB DRAM, tenant-b: 32 MiB)"

A() { "$DIR/kvbctl" "$1" -addr "$ADDR" -ns tenant-a -token "$TOKEN_A" "${@:2}"; }
B() { "$DIR/kvbctl" "$1" -addr "$ADDR" -ns tenant-b -token "$TOKEN_B" "${@:2}"; }

say "3/4  claim 1 — no cross-tenant reads, no existence oracle"
head -c 65536 /dev/urandom >"$DIR/secret.bin"
A put a-secret "$DIR/secret.bin" >/dev/null || die "tenant A could not store its own block"
echo "tenant-a: PUT a-secret (64 KiB)                          -> stored"

A get -o "$DIR/roundtrip.bin" a-secret || die "tenant A cannot read its own block back"
cmp -s "$DIR/secret.bin" "$DIR/roundtrip.bin" || die "tenant A's read-back is not byte-identical"
echo "tenant-a: GET a-secret                                   -> byte-identical"

# Tenant B asks for the exact same key. Identity is (namespace, key): the
# answer MUST be NOT_FOUND — indistinguishable from a key that never existed.
set +e
B_GET_OUT=$(B get a-secret 2>&1 >/dev/null)
B_GET_RC=$?
set -e
[ "$B_GET_RC" -ne 0 ] || die "cross-tenant GET succeeded — isolation broken"
echo "$B_GET_OUT" | grep -q "NOT_FOUND" || die "cross-tenant GET failed with '$B_GET_OUT', want NOT_FOUND"
echo "$B_GET_OUT" | grep -q "FORBIDDEN" && die "cross-tenant GET leaked an existence oracle (FORBIDDEN)"
echo "tenant-b: GET a-secret (same key)                        -> NOT_FOUND (no oracle)"

B_EXISTS=$(B exists a-secret || true)
echo "$B_EXISTS" | head -1 | grep -q "consecutive=0" || die "cross-tenant EXISTS saw the block: $B_EXISTS"
echo "tenant-b: EXISTS a-secret                                -> consecutive=0"

say "4/4  claim 2 — A's quota wall never touches B"
head -c "$BLOCK" /dev/urandom >"$DIR/block.bin"
A_STORED=0
A_STATUS=""
for i in $(seq 1 8); do
  set +e
  OUT=$(A put "a-blk-$i" "$DIR/block.bin" 2>&1)
  RC=$?
  set -e
  if [ "$RC" -eq 0 ]; then
    A_STORED=$((A_STORED + 1))
  else
    A_STATUS="$OUT"
    break
  fi
done
[ "$A_STORED" -ge 1 ] || die "tenant A stored nothing before the wall: $A_STATUS"
[ -n "$A_STATUS" ] || die "tenant A wrote 8 MiB past a 4 MiB quota — wall missing"
echo "$A_STATUS" | grep -q "ERR_QUOTA_BYTES" || die "tenant A refused with '$A_STATUS', want ERR_QUOTA_BYTES"
echo "tenant-a: stored $A_STORED x 1 MiB, then next PUT               -> ERR_QUOTA_BYTES (the wall)"

for i in $(seq 1 6); do
  B put "b-blk-$i" "$DIR/block.bin" >/dev/null || die "tenant B write $i failed while A was over quota"
done
echo "tenant-b: stored 6 x 1 MiB through the same daemon       -> all OK (unaffected)"

echo
echo "per-tenant accounting the daemon exports (/metrics):"
curl -fsS "http://$OPS/metrics" | grep '^kvb_tenant_bytes{' | sed 's/^/  /' || true

ELAPSED=$(($(date +%s) - START_S))
say "PASS — structural isolation, ${ELAPSED}s"
cat <<'EOF'
Same daemon, same wire, same block key:
  tenant-a wrote it and read it back byte-identical;
  tenant-b saw NOT_FOUND — the identity of every block is (namespace, key),
  so there is no cross-tenant read path to misconfigure and no probe that
  reveals another tenant's keys (reads never say FORBIDDEN).
  tenant-a hit its own quota wall (ERR_QUOTA_BYTES) while tenant-b kept
  writing — quotas, like keys, are namespace-scoped by construction.
EOF
