#!/usr/bin/env bash
# Build a SLIM, fully STATIC x86_64-unknown-linux-musl `heartbit` binary that
# runs in ARBITRARY containers (scratch/distroless/alpine included) with no
# shared libs and no system CA store (rustls + compiled-in webpki-roots).
#
# Requires the slim feature set (run/chat env-path only): no teloxide/kafka/
# restate/postgres/openssl — pure rustls+ring, so musl links statically and no
# cmake is needed.
#
# Output: target/x86_64-unknown-linux-musl/release/heartbit  (static)
#
# Usage: benchmarks/terminal-bench-2/scripts/build_musl.sh
#   Builds inside rust:alpine (target is already musl, crt-static by default).
set -euo pipefail

REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
OUT="target/x86_64-unknown-linux-musl/release/heartbit"

echo "==> Building slim static musl binary in rust:alpine"
docker run --rm \
  -v "$REPO_ROOT:/src" \
  -w /src \
  -e CARGO_TERM_COLOR=always \
  rust:alpine \
  sh -c '
    set -eu
    # cc/musl-gcc for ring; git for any build-script that probes the repo.
    apk add --no-cache build-base git
    cargo build --release --target x86_64-unknown-linux-musl \
      --no-default-features --features slim -p heartbit-cli --bin heartbit
  '

BIN="$REPO_ROOT/$OUT"
echo "==> Built $BIN"
echo "==> Portability check"
# A musl release build links static-PIE; both "statically linked" and
# "static-pie linked" are fully static (no dynamic interpreter / no .so deps).
file "$BIN" | grep -qE "statically linked|static-pie" \
  && echo "OK: $(file "$BIN" | grep -oE 'static(-pie)? linked')" \
  || { echo "FAIL: not statically linked"; file "$BIN"; exit 1; }
# ldd on a static binary prints "statically linked" / "not a dynamic executable".
ldd "$BIN" 2>&1 | head -3 || true

echo "==> Smoke test in a bare busybox container (no libs, no CA store)"
docker run --rm -v "$BIN:/heartbit:ro" busybox /heartbit --version \
  && echo "OK: runs in scratch-class container" \
  || echo "WARN: smoke test failed — inspect above"

echo
echo "Use it: set HEARTBIT_INSTALL_MODE=prebuilt (the adapter auto-detects the"
echo "musl binary); it needs no apt runtime libs and works in scratch/distroless."
