#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Launch agentgateway + heartbit together.
#
# Usage:
#   ./gateway/launch.sh "Describe the files in /workspace"
#   ./gateway/launch.sh --file heartbit.toml "Your task"
#   ./gateway/launch.sh chat
#
# What it does:
#   1. Starts agentgateway in the background (port 3000)
#   2. Waits until the MCP endpoint is ready
#   3. Sets HEARTBIT_MCP_SERVERS=http://localhost:3000/mcp
#   4. Runs heartbit with all arguments passed through
#   5. Stops agentgateway on exit
# ──────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
AG_PID=""
AG_PORT="${AG_PORT:-3000}"

cleanup() {
    if [ -n "$AG_PID" ] && kill -0 "$AG_PID" 2>/dev/null; then
        echo ""
        echo "Stopping agentgateway (pid $AG_PID)..."
        kill "$AG_PID" 2>/dev/null || true
        wait "$AG_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

red()   { printf '\033[1;31m%s\033[0m\n' "$*" >&2; }
green() { printf '\033[1;32m%s\033[0m\n' "$*" >&2; }
bold()  { printf '\033[1m%s\033[0m\n' "$*" >&2; }

# ─── Load .env ──────────────────────────────────────────────

if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi

# ─── Find binaries ──────────────────────────────────────────

AG="${AGENTGATEWAY_BIN:-agentgateway}"
if ! command -v "$AG" >/dev/null 2>&1; then
    red "agentgateway not found in PATH."
    red "Install from: https://github.com/agentgateway/agentgateway"
    red "Or set AGENTGATEWAY_BIN=/path/to/agentgateway"
    exit 1
fi

HEARTBIT="${HEARTBIT_BIN:-}"
if [ -z "$HEARTBIT" ]; then
    if [ -x "$ROOT_DIR/target/release/heartbit" ]; then
        HEARTBIT="$ROOT_DIR/target/release/heartbit"
    elif [ -x "$ROOT_DIR/target/debug/heartbit" ]; then
        HEARTBIT="$ROOT_DIR/target/debug/heartbit"
    else
        bold "No heartbit binary found, building..."
        (cd "$ROOT_DIR" && cargo build --release 2>&1) || { red "Build failed"; exit 1; }
        HEARTBIT="$ROOT_DIR/target/release/heartbit"
    fi
fi

if [ $# -eq 0 ]; then
    red "Usage: ./gateway/launch.sh [heartbit args...]"
    red ""
    red "Examples:"
    red "  ./gateway/launch.sh \"List files in /workspace\""
    red "  ./gateway/launch.sh --file heartbit.toml \"Your task\""
    red "  ./gateway/launch.sh chat"
    red "  ./gateway/launch.sh run -v \"Analyze something\""
    exit 1
fi

# ─── Start agentgateway ────────────────────────────────────

bold "Starting agentgateway on port $AG_PORT..."
"$AG" --file "$SCRIPT_DIR/config.yaml" > "$SCRIPT_DIR/.ag_stdout.log" 2> "$SCRIPT_DIR/.ag_stderr.log" &
AG_PID=$!

# Wait for readiness (up to 30s)
waited=0
while [ "$waited" -lt 30 ]; do
    if curl -sf -o /dev/null "http://localhost:$AG_PORT/mcp" \
        -H "Content-Type: application/json" \
        -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"launch","version":"0.1"}}}' 2>/dev/null; then
        green "Agentgateway ready at http://localhost:$AG_PORT/mcp"
        break
    fi
    # Check if process died
    if ! kill -0 "$AG_PID" 2>/dev/null; then
        red "Agentgateway exited unexpectedly."
        if [ -f "$SCRIPT_DIR/.ag_stderr.log" ]; then
            red "Stderr:"
            cat "$SCRIPT_DIR/.ag_stderr.log" >&2
        fi
        exit 1
    fi
    sleep 1
    waited=$((waited + 1))
done

if [ "$waited" -ge 30 ]; then
    red "Agentgateway failed to become ready within 30s."
    if [ -f "$SCRIPT_DIR/.ag_stderr.log" ]; then
        red "Stderr:"
        cat "$SCRIPT_DIR/.ag_stderr.log" >&2
    fi
    exit 1
fi

# ─── Run heartbit ──────────────────────────────────────────

export HEARTBIT_MCP_SERVERS="http://localhost:$AG_PORT/mcp"
bold "Running: heartbit $*"
bold "MCP endpoint: $HEARTBIT_MCP_SERVERS"
echo "" >&2

exec "$HEARTBIT" "$@"
