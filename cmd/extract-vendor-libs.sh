#!/usr/bin/env bash
# Extracts pre-vendored native libraries for the current platform.
# Called during the build to populate lib-ladybug/ and lib-cozo/ from
# the compressed archives committed in vendor-libs/.
#
# Usage: bash extract-vendor-libs.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENDOR_DIR="${SCRIPT_DIR}/vendor-libs"

OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
  Darwin)
    NORM_OS="darwin"
    case "$ARCH" in
      arm64)   NORM_ARCH="arm64" ;;
      x86_64)  NORM_ARCH="x86_64" ;;
      *) echo "Unsupported macOS architecture: $ARCH" >&2; exit 1 ;;
    esac
    LBUG_EXT="dylib"
    ;;
  Linux)
    NORM_OS="linux"
    case "$ARCH" in
      x86_64)          NORM_ARCH="x86_64" ;;
      aarch64|arm64)   NORM_ARCH="aarch64" ;;
      *) echo "Unsupported Linux architecture: $ARCH" >&2; exit 1 ;;
    esac
    LBUG_EXT="so"
    ;;
  *)
    echo "Unsupported OS: $OS" >&2
    exit 1
    ;;
esac

# ---------------------------------------------------------------------------
# Ladybug
# ---------------------------------------------------------------------------
LBUG_GZ="$VENDOR_DIR/liblbug-${NORM_OS}-${NORM_ARCH}.${LBUG_EXT}.gz"
LBUG_DIR="${SCRIPT_DIR}/lib-ladybug"

if [ ! -f "$LBUG_GZ" ]; then
  echo "ERROR: Vendored ladybug not found: $LBUG_GZ" >&2
  echo "Run 'bash cmd/update-vendor-libs.sh' to download it." >&2
  exit 1
fi

mkdir -p "$LBUG_DIR"
LBUG_OUT="${LBUG_DIR}/liblbug.${LBUG_EXT}"
if [ ! -f "$LBUG_OUT" ]; then
  echo "Extracting ladybug (${NORM_OS}/${NORM_ARCH})..."
  gunzip -c "$LBUG_GZ" > "$LBUG_OUT"
  chmod +x "$LBUG_OUT"
  # Create the versioned SONAME symlink the runtime linker resolves. Binaries are
  # linked against @rpath/liblbug.0.dylib (macOS) / liblbug.so.0 (Linux), so
  # without this they build but abort at load with "Library not loaded". Only the
  # Linux case existed, which is why the CGO driver tests could never run on macOS.
  if [ "$LBUG_EXT" = "so" ]; then
    ln -sf "liblbug.so" "${LBUG_DIR}/liblbug.so.0"
  else
    ln -sf "liblbug.dylib" "${LBUG_DIR}/liblbug.0.dylib"
  fi
  echo "  -> $LBUG_OUT"
else
  echo "Ladybug library already extracted."
fi

# Ladybug C API header (lbug.h). go-ladybug >= v0.17 no longer ships lbug.h in
# its Go module, so the build needs our vendored copy on the include path (see
# CGO_CFLAGS in the Makefile, which adds -I$(LIB_PATH)).
if [ -f "$VENDOR_DIR/lbug.h" ]; then
  cp -f "$VENDOR_DIR/lbug.h" "${LBUG_DIR}/lbug.h"
fi

# Ladybug official extensions (vector, fts) bundled at the pinned version. The
# driver copies these into lbug's home extension dir at runtime so LOAD works
# offline (no registry fetch). Extracted next to liblbug under extensions/.
LBUG_EXT_DIR="${LBUG_DIR}/extensions"
for extn in vector fts reasoning; do
  EXT_GZ="$VENDOR_DIR/lib${extn}-${NORM_OS}-${NORM_ARCH}.lbug_extension.gz"
  EXT_OUT="${LBUG_EXT_DIR}/lib${extn}.lbug_extension"
  if [ -f "$EXT_GZ" ] && [ ! -f "$EXT_OUT" ]; then
    mkdir -p "$LBUG_EXT_DIR"
    gunzip -c "$EXT_GZ" > "$EXT_OUT"
    echo "  Extracted extension: $EXT_OUT"
  fi
done

# ---------------------------------------------------------------------------
# CozoDB
# ---------------------------------------------------------------------------
COZO_GZ="$VENDOR_DIR/libcozo_c-${NORM_OS}-${NORM_ARCH}.a.gz"
COZO_DIR="${SCRIPT_DIR}/lib-cozo"

if [ ! -f "$COZO_GZ" ]; then
  echo "ERROR: Vendored cozo not found: $COZO_GZ" >&2
  echo "Run 'bash cmd/update-vendor-libs.sh' to download it." >&2
  exit 1
fi

mkdir -p "$COZO_DIR"
COZO_OUT="${COZO_DIR}/libcozo_c.a"
if [ ! -f "$COZO_OUT" ]; then
  echo "Extracting cozo (${NORM_OS}/${NORM_ARCH})..."
  gunzip -c "$COZO_GZ" > "$COZO_OUT"
  echo "  -> $COZO_OUT"
else
  echo "CozoDB library already extracted."
fi

echo "Native libraries ready."
