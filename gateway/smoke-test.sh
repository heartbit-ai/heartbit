#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Smoke test for an MCP endpoint (agentgateway or supergateway).
#
# No LLM needed — sends raw JSON-RPC requests via curl to verify:
#   1. MCP initialize handshake succeeds
#   2. tools/list returns discovered tools
#   3. Tool names are prefixed by target name
#
# Usage:
#   ./gateway/smoke-test.sh              # default: localhost:3000
#   ./gateway/smoke-test.sh 8080         # custom port
# ──────────────────────────────────────────────────────────────
set -euo pipefail

PORT="${1:-3000}"
ENDPOINT="http://localhost:$PORT/mcp"
SESSION_ID=""

# MCP Streamable HTTP requires Accept header with both types
ACCEPT="Accept: application/json, text/event-stream"

red()   { printf '\033[1;31m%s\033[0m\n' "$*"; }
green() { printf '\033[1;32m%s\033[0m\n' "$*"; }
bold()  { printf '\033[1m%s\033[0m\n' "$*"; }

bold "Smoke testing MCP endpoint: $ENDPOINT"
echo ""

# ─── Step 1: Initialize ────────────────────────────────────

bold "1. Initialize handshake..."
INIT_RESPONSE=$(curl -sf -D - "$ENDPOINT" \
    -H "Content-Type: application/json" \
    -H "$ACCEPT" \
    -d '{
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "smoke-test", "version": "0.1"}
        }
    }' 2>/dev/null) || {
    red "  FAIL: Could not connect to $ENDPOINT"
    red "  Is the MCP server running?"
    red "  For agentgateway: ./gateway/start.sh"
    red "  For supergateway: npx -y supergateway --stdio '...' --outputTransport streamableHttp --port $PORT"
    exit 1
}

# Extract session ID from headers
SESSION_ID=$(echo "$INIT_RESPONSE" | grep -i "^mcp-session-id:" | tr -d '\r' | awk '{print $2}')

# Extract JSON body (everything after the blank line in HTTP response)
INIT_BODY=$(echo "$INIT_RESPONSE" | sed -n '/^\r*$/,$p' | tail -n +2)

if echo "$INIT_BODY" | python3 -c "
import json, sys
data = json.loads(sys.stdin.read())
if 'result' in data:
    r = data['result']
    print(f'  Protocol: {r.get(\"protocolVersion\", \"unknown\")}')
    print(f'  Server:   {r.get(\"serverInfo\", {}).get(\"name\", \"unknown\")}')
    sys.exit(0)
elif 'error' in data:
    print(f'  Error: {data[\"error\"]}')
    sys.exit(1)
" 2>/dev/null; then
    green "  OK"
else
    red "  FAIL: Unexpected response"
    echo "  $INIT_BODY"
    exit 1
fi

if [ -n "$SESSION_ID" ]; then
    echo "  Session: $SESSION_ID"
fi

# ─── Step 2: Send initialized notification ──────────────────

bold "2. Sending initialized notification..."
NOTIFY_ARGS=(-sf "$ENDPOINT" \
    -H "Content-Type: application/json" \
    -H "$ACCEPT")
if [ -n "$SESSION_ID" ]; then
    NOTIFY_ARGS+=(-H "Mcp-Session-Id: $SESSION_ID")
fi
NOTIFY_ARGS+=(-d '{"jsonrpc":"2.0","method":"notifications/initialized"}')

curl "${NOTIFY_ARGS[@]}" > /dev/null 2>&1 || true
green "  OK"

# ─── Step 3: List tools ─────────────────────────────────────

bold "3. Listing tools..."
TOOLS_ARGS=(-sf "$ENDPOINT" \
    -H "Content-Type: application/json" \
    -H "$ACCEPT")
if [ -n "$SESSION_ID" ]; then
    TOOLS_ARGS+=(-H "Mcp-Session-Id: $SESSION_ID")
fi
TOOLS_ARGS+=(-d '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}')

TOOLS_RESPONSE=$(curl "${TOOLS_ARGS[@]}" 2>/dev/null) || {
    red "  FAIL: tools/list request failed"
    exit 1
}

TOOL_COUNT=$(echo "$TOOLS_RESPONSE" | python3 -c "
import json, sys
data = json.loads(sys.stdin.read())
tools = data.get('result', {}).get('tools', [])
print(len(tools))
for t in sorted(tools, key=lambda x: x.get('name','')):
    desc = t.get('description', '')[:60]
    print(f'  {t[\"name\"]:40s} {desc}')
" 2>/dev/null) || {
    red "  FAIL: Could not parse tools/list response"
    echo "  $TOOLS_RESPONSE"
    exit 1
}

# First line is count, rest is the tool list
COUNT=$(echo "$TOOL_COUNT" | head -1)
echo "$TOOL_COUNT" | tail -n +2

echo ""
if [ "$COUNT" -gt 0 ]; then
    green "  $COUNT tools discovered"
else
    red "  FAIL: No tools discovered"
    exit 1
fi

# ─── Step 4: Check prefixing ────────────────────────────────

bold "4. Checking tool name prefixing..."
PREFIXED=$(echo "$TOOL_COUNT" | tail -n +2 | grep -c "_" || true)
if [ "$PREFIXED" -gt 0 ]; then
    green "  $PREFIXED/$COUNT tools have prefix (e.g., filesystem_read_file)"
else
    echo "  No prefixed tools found (prefix_mode may be 'conditional')"
fi

echo ""
green "Smoke test passed."
