#!/usr/bin/env bash
# Build a heartbit release binary for the `prebuilt` install mode, inside a glibc
# Docker builder so its glibc/OpenSSL ABI matches Debian/Ubuntu-derived TB2 images.
#
# Output: target/release/heartbit  (the adapter's default HEARTBIT_BIN path)
#
# Usage: benchmarks/terminal-bench-2/scripts/build_prebuilt.sh [base-image]
#   base-image defaults to rust:1-bookworm. Pick one whose glibc/OpenSSL major
#   matches the TB2 task base images you intend to run.
set -euo pipefail

BASE="${1:-rust:1-bookworm}"
REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

echo "==> Building heartbit (release) in $BASE for glibc ABI portability"
docker run --rm \
  -v "$REPO_ROOT:/src" \
  -w /src \
  -e CARGO_TERM_COLOR=always \
  "$BASE" \
  bash -c '
    set -euo pipefail
    apt-get update
    apt-get install -y cmake libssl-dev libcurl4-openssl-dev pkg-config build-essential
    cargo build --release -p heartbit-cli
  '

echo "==> Built $REPO_ROOT/target/release/heartbit"
ldd "$REPO_ROOT/target/release/heartbit" || true
echo "==> Note its required .so versions (libssl.so.3 / libcrypto.so.3 / libcurl / libz)"
echo "    The TB2 base image must provide a compatible glibc + these libs"
echo "    (the adapter apt-installs: ca-certificates libssl3 libcurl4 zlib1g)."
