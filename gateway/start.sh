#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Load .env if present
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi

# Find agentgateway binary
AG="${AGENTGATEWAY_BIN:-agentgateway}"
if ! command -v "$AG" >/dev/null 2>&1; then
    echo "Error: agentgateway not found in PATH."
    echo "Install from: https://github.com/agentgateway/agentgateway"
    echo "Or set AGENTGATEWAY_BIN=/path/to/agentgateway"
    exit 1
fi

echo "Starting agentgateway on port 3000..."
echo "Heartbit endpoint: http://localhost:3000/mcp"
exec "$AG" --file "$SCRIPT_DIR/config.yaml"
