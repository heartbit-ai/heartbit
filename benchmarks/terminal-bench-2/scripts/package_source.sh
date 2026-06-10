#!/usr/bin/env bash
# Build an OFFLINE source bundle of the heartbit workspace for the build-in-image
# install mode: the committed source tree + all crates vendored + a .cargo/config
# that points cargo at the vendored crates. The in-container build then runs
# `cargo build --release --offline` with no crates.io access.
#
# Output: benchmarks/terminal-bench-2/dist/heartbit-src.tar.gz
#
# Usage: benchmarks/terminal-bench-2/scripts/package_source.sh [git-ref]
set -euo pipefail

REF="${1:-HEAD}"
REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
DIST="$REPO_ROOT/benchmarks/terminal-bench-2/dist"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

echo "==> Exporting source tree at $REF"
git -C "$REPO_ROOT" archive --format=tar "$REF" | tar -x -C "$STAGE"

echo "==> Vendoring crates (offline build)"
cd "$STAGE"
mkdir -p .cargo
# `cargo vendor` writes the config snippet to stdout; capture it.
cargo vendor --quiet vendor > .cargo/config.toml

echo "==> Packing bundle"
mkdir -p "$DIST"
tar -czf "$DIST/heartbit-src.tar.gz" .
echo "==> Wrote $DIST/heartbit-src.tar.gz ($(du -h "$DIST/heartbit-src.tar.gz" | cut -f1))"
