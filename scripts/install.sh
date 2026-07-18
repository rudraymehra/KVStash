#!/usr/bin/env sh
# kvblockd installer: detect OS/arch, fetch the latest release tarball,
# install kvblockd + kvbctl into /usr/local/bin (or $KVB_INSTALL_DIR).
#
#   curl -fsSL https://raw.githubusercontent.com/kvstash/kvblockd/main/scripts/install.sh | sh
#
# Options via env:
#   KVB_VERSION      pin a version (default: latest release; prereleases need
#                    an explicit KVB_VERSION or the /releases fallback below)
#   KVB_INSTALL_DIR  target dir (default /usr/local/bin)
#   KVB_CONFIG_DIR   config dir (default /usr/local/etc/kvblockd)
#   KVB_TARBALL      install from a local tarball instead of downloading
set -eu

REPO="kvstash/kvblockd"
INSTALL_DIR="${KVB_INSTALL_DIR:-/usr/local/bin}"
CONF_DIR="${KVB_CONFIG_DIR:-/usr/local/etc/kvblockd}"

say()  { printf '%s\n' "$*"; }
die()  { printf 'install.sh: %s\n' "$*" >&2; exit 1; }

OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)
case "$OS" in
  linux|darwin) ;;
  *) die "unsupported OS: $OS (linux and darwin only)" ;;
esac
# A Rosetta shell reports x86_64 on Apple silicon — the arm64 build is right.
if [ "$OS" = "darwin" ] && [ "$ARCH" = "x86_64" ] \
  && [ "$(sysctl -n sysctl.proc_translated 2>/dev/null || echo 0)" = "1" ]; then
  ARCH=arm64
fi
case "$ARCH" in
  x86_64|amd64)  ARCH=amd64 ;;
  aarch64|arm64) ARCH=arm64 ;;
  *) die "unsupported arch: $ARCH (amd64 and arm64 only)" ;;
esac
[ "$OS" = "darwin" ] && [ "$ARCH" = "amd64" ] \
  && die "darwin/amd64 is not shipped (Apple-silicon and linux only)"

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

if [ -n "${KVB_TARBALL:-}" ]; then
  say "installing from local tarball $KVB_TARBALL"
  tar -xzf "$KVB_TARBALL" -C "$TMP"
else
  command -v curl >/dev/null 2>&1 || die "curl is required"
  if [ -n "${KVB_VERSION:-}" ]; then
    TAG="$KVB_VERSION"
  else
    # /releases/latest excludes prereleases (404 while only rcs exist) —
    # fall back to the newest entry in /releases.
    TAG=$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" 2>/dev/null \
      | grep '"tag_name"' | head -n1 | cut -d'"' -f4)
    if [ -z "$TAG" ]; then
      TAG=$(curl -fsSL "https://api.github.com/repos/$REPO/releases?per_page=1" 2>/dev/null \
        | grep '"tag_name"' | head -n1 | cut -d'"' -f4)
    fi
    [ -n "$TAG" ] || die "could not resolve latest release"
  fi
  VER=${TAG#v}
  URL="https://github.com/$REPO/releases/download/$TAG/kvblockd_${VER}_${OS}_${ARCH}.tar.gz"
  say "downloading $URL"
  curl -fsSL "$URL" -o "$TMP/kvb.tar.gz" || die "download failed: $URL"
  tar -xzf "$TMP/kvb.tar.gz" -C "$TMP"
fi

[ -f "$TMP/kvblockd" ] || die "tarball has no kvblockd binary"
[ -f "$TMP/kvbctl" ]   || die "tarball has no kvbctl binary"

SUDO=""
mkdir -p "$INSTALL_DIR" 2>/dev/null || true
if [ ! -d "$INSTALL_DIR" ] || [ ! -w "$INSTALL_DIR" ]; then
  command -v sudo >/dev/null 2>&1 || die "$INSTALL_DIR not writable and no sudo"
  SUDO="sudo"
  say "installing to $INSTALL_DIR (needs sudo)"
fi
$SUDO install -m 0755 "$TMP/kvblockd" "$TMP/kvbctl" "$INSTALL_DIR/"

# The tarball's config/ would die with $TMP — install it somewhere durable,
# with namespaces_path rewritten to an absolute path so the daemon starts
# from ANY cwd. Never clobber an existing namespaces.yaml (real tenants).
if [ -f "$TMP/config/example.yaml" ]; then
  $SUDO mkdir -p "$CONF_DIR"
  sed "s|^namespaces_path:.*|namespaces_path: \"$CONF_DIR/namespaces.yaml\"|" \
    "$TMP/config/example.yaml" >"$TMP/example.resolved.yaml"
  $SUDO install -m 0644 "$TMP/example.resolved.yaml" "$CONF_DIR/example.yaml"
  if [ ! -f "$CONF_DIR/namespaces.yaml" ]; then
    $SUDO install -m 0600 "$TMP/config/namespaces.yaml" "$CONF_DIR/namespaces.yaml"
  fi
fi

if [ "$OS" = "darwin" ]; then
  # Unsigned binaries: clear the quarantine bit so Gatekeeper lets them run.
  $SUDO xattr -d com.apple.quarantine "$INSTALL_DIR/kvblockd" "$INSTALL_DIR/kvbctl" 2>/dev/null || true
  say "note: macOS is a dev platform only — no durability claims (see docs)."
fi

say ""
say "installed: $("$INSTALL_DIR/kvblockd" --version)"
say ""
say "60-second start:"
say "  kvblockd --config $CONF_DIR/example.yaml &     # :9440, 1 GiB DRAM arena, demo tenant"
say "  echo hello | kvbctl put -ns demo -token demo-token demo-key -"
say "  kvbctl get -ns demo -token demo-token demo-key"
say "  kvbctl stats -ns demo -token demo-token"
say ""
say "a bare 'kvblockd' (no config) accepts NO clients — namespaces are the"
say "auth model; edit $CONF_DIR/namespaces.yaml for real tenants."
say "docs: https://github.com/$REPO#readme"
