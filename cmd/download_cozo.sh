#!/usr/bin/env bash
#
# Downloads the CozoDB C library for the current platform.
# Installs into both a local directory and the cozo-lib-go module cache
# so the linker can find it via the module's own #cgo LDFLAGS: -lcozo_c.
#
# Usage: bash download_cozo.sh [-out DIR]
#
set -euo pipefail

COZO_VERSION="0.7.5"
OUT_DIR="lib-cozo"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -out) OUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# Detect platform
OS="$(uname -s)"
ARCH="$(uname -m)"

case "${OS}" in
    Linux)
        case "${ARCH}" in
            x86_64)  PLATFORM="x86_64-unknown-linux-gnu" ;;
            aarch64) PLATFORM="aarch64-unknown-linux-gnu" ;;
            *) echo "Unsupported Linux architecture: ${ARCH}" >&2; exit 1 ;;
        esac
        ;;
    Darwin)
        case "${ARCH}" in
            x86_64)  PLATFORM="x86_64-apple-darwin" ;;
            arm64)   PLATFORM="aarch64-apple-darwin" ;;
            *) echo "Unsupported macOS architecture: ${ARCH}" >&2; exit 1 ;;
        esac
        ;;
    MINGW*|MSYS*|CYGWIN*|Windows_NT)
        PLATFORM="x86_64-pc-windows-gnu"
        ;;
    *)
        echo "Unsupported OS: ${OS}" >&2; exit 1
        ;;
esac

URL="https://github.com/cozodb/cozo/releases/download/v${COZO_VERSION}/libcozo_c-${COZO_VERSION}-${PLATFORM}.a.gz"

mkdir -p "${OUT_DIR}"

LIB_FILE="${OUT_DIR}/libcozo_c.a"
if [[ -f "${LIB_FILE}" ]]; then
    echo "libcozo_c.a already exists in ${OUT_DIR}, skipping download"
else
    echo "Downloading libcozo_c v${COZO_VERSION} for ${PLATFORM}..."
    curl -sL "${URL}" | gunzip > "${LIB_FILE}"
    echo "Downloaded to ${LIB_FILE} ($(du -h "${LIB_FILE}" | cut -f1))"
fi

# Also install into the cozo-lib-go module cache so the module's own
# #cgo LDFLAGS: -lcozo_c can find it without CGO_LDFLAGS.
GOMODCACHE="$(go env GOMODCACHE 2>/dev/null || echo "${GOPATH:-$HOME/go}/pkg/mod")"
COZO_MOD_DIR="${GOMODCACHE}/github.com/cozodb/cozo-lib-go@v${COZO_VERSION}"

if [[ -d "${COZO_MOD_DIR}" ]]; then
    COZO_MOD_LIB="${COZO_MOD_DIR}/libs"
    # Module cache is read-only; make writable, copy, restore
    chmod u+w "${COZO_MOD_DIR}" 2>/dev/null || true
    mkdir -p "${COZO_MOD_LIB}"
    if [[ ! -f "${COZO_MOD_LIB}/libcozo_c.a" ]]; then
        cp "${LIB_FILE}" "${COZO_MOD_LIB}/libcozo_c.a"
        echo "Installed libcozo_c.a into ${COZO_MOD_LIB}"
    fi
    chmod u-w "${COZO_MOD_DIR}" 2>/dev/null || true
fi
