#!/usr/bin/env bash
set -euo pipefail

REPO="heartbit-ai/heartbit"
BINARY_NAME="heartbit"
INSTALL_DIR="/usr/local/bin"

main() {
    local os arch target

    os="$(uname -s)"
    arch="$(uname -m)"

    case "$os" in
        Linux)  os="unknown-linux-gnu" ;;
        Darwin) os="apple-darwin" ;;
        *)
            echo "Error: unsupported operating system: $os" >&2
            exit 1
            ;;
    esac

    case "$arch" in
        x86_64)         arch="x86_64" ;;
        aarch64|arm64)  arch="aarch64" ;;
        *)
            echo "Error: unsupported architecture: $arch" >&2
            exit 1
            ;;
    esac

    target="${arch}-${os}"

    # Only these targets have pre-built binaries
    case "$target" in
        x86_64-unknown-linux-gnu|x86_64-apple-darwin|aarch64-apple-darwin) ;;
        *)
            echo "Error: no pre-built binary for ${target}" >&2
            echo "Build from source: cargo install --git https://github.com/${REPO} heartbit-cli" >&2
            exit 1
            ;;
    esac

    local url="https://github.com/${REPO}/releases/latest/download/heartbit-${target}.tar.gz"

    echo "Downloading heartbit for ${target}..."

    local tmpdir
    tmpdir="$(mktemp -d)"
    trap 'rm -rf "$tmpdir"' EXIT

    if ! curl -fsSL "$url" -o "${tmpdir}/heartbit.tar.gz"; then
        echo "Error: failed to download from ${url}" >&2
        exit 1
    fi

    tar -xzf "${tmpdir}/heartbit.tar.gz" -C "$tmpdir"

    if [ ! -f "${tmpdir}/${BINARY_NAME}" ]; then
        echo "Error: ${BINARY_NAME} not found in archive" >&2
        exit 1
    fi

    chmod +x "${tmpdir}/${BINARY_NAME}"

    if [ -w "$INSTALL_DIR" ]; then
        mv "${tmpdir}/${BINARY_NAME}" "${INSTALL_DIR}/${BINARY_NAME}"
    else
        echo "Installing to ${INSTALL_DIR} (requires sudo)..."
        sudo mv "${tmpdir}/${BINARY_NAME}" "${INSTALL_DIR}/${BINARY_NAME}"
    fi

    echo "Installed ${BINARY_NAME} to ${INSTALL_DIR}/${BINARY_NAME}"

    if command -v "$BINARY_NAME" >/dev/null 2>&1; then
        echo "Version: $("$BINARY_NAME" --version)"
    else
        echo "Warning: ${INSTALL_DIR} may not be in your PATH" >&2
    fi
}

main
