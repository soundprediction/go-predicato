#!/usr/bin/env bash
# Downloads upstream native libraries for all supported platforms and stores
# compressed copies in cmd/vendor-libs/ so they can be committed to the repo.
#
# Usage: bash update-vendor-libs.sh [--ladybug-only] [--cozo-only]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENDOR_DIR="${SCRIPT_DIR}/vendor-libs"
TMPDIR_BASE="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_BASE"' EXIT

mkdir -p "$VENDOR_DIR"

LADYBUG_ONLY=false
COZO_ONLY=false
for arg in "$@"; do
  case "$arg" in
    --ladybug-only) LADYBUG_ONLY=true ;;
    --cozo-only) COZO_ONLY=true ;;
  esac
done

# ---------------------------------------------------------------------------
# Ladybug
# ---------------------------------------------------------------------------
update_ladybug() {
  local REPOSITORY="${LBUG_GITHUB_REPOSITORY:-LadybugDB/ladybug}"
  local VERSION_OVERRIDE="${LBUG_VERSION:-}"
  local EXT_REGISTRY="${LBUG_EXTENSION_REGISTRY:-http://extension.ladybugdb.com}"
  local LBUG_EXTENSIONS="${LBUG_EXTENSIONS:-vector fts}"

  local version
  if [ -n "$VERSION_OVERRIDE" ]; then
    version="$VERSION_OVERRIDE"
  else
    version="$(curl -sS "https://api.github.com/repos/${REPOSITORY}/releases/latest" \
      | grep -o '"tag_name": "v\([^"]*\)"' | cut -d'"' -f4 | cut -c2-)"
  fi
  echo "Ladybug version: v${version}"

  # Platform archives to download
  local -a PLATFORMS=(
    "darwin:arm64:liblbug-osx-arm64.tar.gz:liblbug.dylib"
    "darwin:x86_64:liblbug-osx-x86_64.tar.gz:liblbug.dylib"
    "linux:x86_64:liblbug-linux-x86_64.tar.gz:liblbug.so"
    "linux:aarch64:liblbug-linux-aarch64.tar.gz:liblbug.so"
  )

  for entry in "${PLATFORMS[@]}"; do
    IFS=: read -r os arch archive lib_name <<< "$entry"
    local url="https://github.com/${REPOSITORY}/releases/download/v${version}/${archive}"
    local tmpdir="${TMPDIR_BASE}/lbug-${os}-${arch}"
    mkdir -p "$tmpdir"

    echo "  Downloading ladybug ${os}/${arch}..."
    if ! curl -fSL "$url" -o "$tmpdir/$archive" 2>/dev/null; then
      echo "  WARNING: Failed to download $archive — skipping"
      continue
    fi

    tar xzf "$tmpdir/$archive" -C "$tmpdir"

    # Upstream archives ship liblbug.so as a symlink to liblbug.so.<version>;
    # match the symlink (or the versioned real file) and resolve to the real file.
    local lib_file
    lib_file="$(find "$tmpdir" \( -name "$lib_name" -o -name "${lib_name}.*" -o -name "${lib_name%.*}.*.${lib_name##*.}" \) | head -n1)"
    lib_file="$(readlink -f "$lib_file" 2>/dev/null || echo "$lib_file")"
    if [ -z "$lib_file" ] || [ ! -f "$lib_file" ]; then
      echo "  WARNING: $lib_name not found in $archive — skipping"
      continue
    fi

    local ext="${lib_name##*.}"
    local out="liblbug-${os}-${arch}.${ext}.gz"
    gzip -c "$lib_file" > "$VENDOR_DIR/$out"
    echo "  Stored $out ($(du -h "$VENDOR_DIR/$out" | cut -f1))"

    # Bundle the official extensions used by the driver (vector, fts) at the
    # SAME pinned version, so downstream loads them offline instead of fetching
    # from the extension registry at runtime. The registry uses lbug's own
    # platform names (linux_amd64/linux_arm64/osx_amd64/osx_arm64).
    local lbug_plat=""
    case "${os}:${arch}" in
      darwin:arm64)  lbug_plat="osx_arm64" ;;
      darwin:x86_64) lbug_plat="osx_amd64" ;;
      linux:x86_64)  lbug_plat="linux_amd64" ;;
      linux:aarch64) lbug_plat="linux_arm64" ;;
    esac
    if [ -n "$lbug_plat" ]; then
      local extn
      for extn in $LBUG_EXTENSIONS; do
        local exturl="${EXT_REGISTRY}/v${version}/${lbug_plat}/${extn}/lib${extn}.lbug_extension"
        local extout="lib${extn}-${os}-${arch}.lbug_extension.gz"
        if curl -fSL "$exturl" -o "$tmpdir/lib${extn}.lbug_extension" 2>/dev/null; then
          gzip -c "$tmpdir/lib${extn}.lbug_extension" > "$VENDOR_DIR/$extout"
          echo "  Stored $extout ($(du -h "$VENDOR_DIR/$extout" | cut -f1))"
        else
          echo "  WARNING: failed to download '$extn' extension for ${lbug_plat} ($exturl)"
        fi
      done
    fi

    # go-ladybug >= v0.17 dropped lbug.h from its Go module; vendor the C API
    # header (platform-independent) so the build can find it via CGO_CFLAGS.
    if [ ! -f "$VENDOR_DIR/lbug.h" ]; then
      local hdr
      hdr="$(find "$tmpdir" -name 'lbug.h' -type f | head -n1)"
      if [ -n "$hdr" ]; then
        cp -f "$hdr" "$VENDOR_DIR/lbug.h"
        echo "  Stored lbug.h"
      fi
    fi
  done

  echo "$version" > "$VENDOR_DIR/ladybug.version"
}

# ---------------------------------------------------------------------------
# CozoDB
# ---------------------------------------------------------------------------
update_cozo() {
  local COZO_VERSION="${COZO_VERSION:-0.7.5}"
  echo "CozoDB version: v${COZO_VERSION}"

  local -a PLATFORMS=(
    "darwin:arm64:aarch64-apple-darwin"
    "darwin:x86_64:x86_64-apple-darwin"
    "linux:x86_64:x86_64-unknown-linux-gnu"
    "linux:aarch64:aarch64-unknown-linux-gnu"
  )

  for entry in "${PLATFORMS[@]}"; do
    IFS=: read -r os arch platform <<< "$entry"
    local url="https://github.com/cozodb/cozo/releases/download/v${COZO_VERSION}/libcozo_c-${COZO_VERSION}-${platform}.a.gz"

    echo "  Downloading cozo ${os}/${arch}..."
    local out="libcozo_c-${os}-${arch}.a.gz"
    if ! curl -fSL "$url" -o "$VENDOR_DIR/$out" 2>/dev/null; then
      echo "  WARNING: Failed to download cozo for ${platform} — skipping"
      continue
    fi
    echo "  Stored $out ($(du -h "$VENDOR_DIR/$out" | cut -f1))"
  done

  echo "$COZO_VERSION" > "$VENDOR_DIR/cozo.version"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if [ "$COZO_ONLY" = false ]; then
  echo "=== Updating Ladybug ==="
  update_ladybug
fi

if [ "$LADYBUG_ONLY" = false ]; then
  echo "=== Updating CozoDB ==="
  update_cozo
fi

echo "Done. Vendor libs updated in $VENDOR_DIR"
