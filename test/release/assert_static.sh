#!/usr/bin/env bash
# assert_static.sh <binary-path> <goos> <goarch> — the release gate behind
# the "single static binary" claim. Runs as a goreleaser post-build hook on
# every artifact; ANY failure fails the release.
#
# linux binaries: `file` must say "statically linked"; `ldd` must refuse it
# (where ldd exists — ubuntu CI); size < 20MB HARD (the ~11MB claim);
# boots `--version` inside a FROM-scratch container when docker is present.
# darwin binaries: Mach-O check + the same size gate (dev-only platform,
# no static/scratch story to prove).
set -euo pipefail
BIN="$1"; GOOS="$2"; GOARCH="${3:-}"
NAME="$(basename "$BIN")"
MAX_BYTES=$((20 * 1024 * 1024))

fail() { echo "assert_static: FAIL [$NAME $GOOS/$GOARCH]: $*" >&2; exit 1; }

[ -f "$BIN" ] || fail "no such binary: $BIN"

# 1. Size gate (all platforms). README quotes the real number — print it.
SIZE=$(wc -c <"$BIN" | tr -d ' ')
if [ "$SIZE" -ge "$MAX_BYTES" ]; then
  fail "size $SIZE bytes >= 20MB gate"
fi
printf 'assert_static: %s %s/%s size %.1f MB\n' "$NAME" "$GOOS" "$GOARCH" \
  "$(awk "BEGIN{print $SIZE/1048576}")"

if [ "$GOOS" = "darwin" ]; then
  file "$BIN" | grep -q "Mach-O" || fail "darwin artifact is not Mach-O"
  echo "assert_static: OK (darwin: Mach-O + size gate only; dev platform)"
  exit 0
fi

# 2. Statically linked, per `file`.
FILE_OUT=$(file "$BIN")
echo "$FILE_OUT" | grep -q "statically linked" \
  || fail "file(1) does not say statically linked: $FILE_OUT"

# 3. ldd must FAIL on a static binary (linux hosts only; macOS has no ldd).
if command -v ldd >/dev/null 2>&1; then
  if ldd "$BIN" >/dev/null 2>&1; then
    fail "ldd succeeded — binary is dynamically linked"
  fi
fi

# 4. Boot in a FROM-scratch container: the binary must need NOTHING.
#    Docker Desktop / CI both run non-native arches via qemu/binfmt.
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  CTX=$(mktemp -d); trap 'rm -rf "$CTX"' EXIT
  cp "$BIN" "$CTX/app"
  printf 'FROM scratch\nCOPY app /app\nENTRYPOINT ["/app"]\n' >"$CTX/Dockerfile"
  IMG="kvb-static-test:$NAME-$GOARCH"
  docker build --platform "linux/$GOARCH" -q -t "$IMG" "$CTX" >/dev/null \
    || fail "scratch image build failed"
  OUT=$(docker run --rm --platform "linux/$GOARCH" "$IMG" --version 2>&1) \
    || { docker rmi -f "$IMG" >/dev/null 2>&1; fail "--version in scratch failed: $OUT"; }
  docker rmi -f "$IMG" >/dev/null 2>&1 || true
  echo "assert_static: scratch boot OK: $OUT"
elif [ -n "${GITHUB_ACTIONS:-}" ]; then
  # In release CI the scratch boot is a GATE, not best-effort — a broken
  # docker daemon must fail the release, not silently pass it.
  fail "docker unavailable in CI — scratch-boot gate cannot run"
else
  echo "assert_static: docker unavailable — scratch-boot check skipped (gated in release CI)"
fi

echo "assert_static: OK [$NAME $GOOS/$GOARCH]"
