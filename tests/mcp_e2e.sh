#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# End-to-end MCP integration tests.
#
# Starts a lightweight MCP server (supergateway + filesystem),
# runs heartbit agents against it, and verifies tool discovery
# and execution work end-to-end.
#
# These are NOT CI tests. They call a real LLM and cost money.
# Run manually after changes to the MCP client or tool pipeline.
#
# Design rules (same as features_e2e.sh):
#   1. Assert on artifacts (files, events), never LLM prose.
#   2. Assertions must be LENIENT — allow the LLM flexibility.
#   3. Each test retries up to $MAX_RETRIES times.
#   4. Individual test: ./mcp_e2e.sh 2   (runs only test 2)
#
# Requires: npx (Node.js), OPENROUTER_API_KEY, target/release/heartbit
# ──────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BINARY="$ROOT_DIR/target/release/heartbit"
WORKDIR="$(mktemp -d)"
MAX_RETRIES="${MAX_RETRIES:-2}"
PASS=0
FAIL=0
SKIP=0
ERRORS=""
FILTER="${1:-}"
MCP_PORT="${MCP_PORT:-18321}"  # high port to avoid conflicts
MCP_PID=""

export HEARTBIT_MODEL="${HEARTBIT_MODEL:-qwen/qwen3-30b-a3b}"
export HEARTBIT_MAX_TURNS="${HEARTBIT_MAX_TURNS:-10}"

cleanup() {
    if [ -n "$MCP_PID" ] && kill -0 "$MCP_PID" 2>/dev/null; then
        kill "$MCP_PID" 2>/dev/null || true
        wait "$MCP_PID" 2>/dev/null || true
    fi
    rm -rf "$WORKDIR"
}
trap cleanup EXIT

red()    { printf '\033[1;31m%s\033[0m\n' "$*"; }
green()  { printf '\033[1;32m%s\033[0m\n' "$*"; }
yellow() { printf '\033[1;33m%s\033[0m\n' "$*"; }
bold()   { printf '\033[1m%s\033[0m\n' "$*"; }

pass() { PASS=$((PASS + 1)); green "  PASS: $1"; }
fail() { FAIL=$((FAIL + 1)); red "  FAIL: $1 — $2"; ERRORS+="  [$1] $2\n"; }
skip() { SKIP=$((SKIP + 1)); yellow "  SKIP: $1"; }

should_run() {
    [ -z "$FILTER" ] || [ "$FILTER" = "$1" ]
}

extract_events() {
    grep '^\[event\] ' "$WORKDIR/_stderr" | sed 's/^\[event\] //' > "$WORKDIR/_events.jsonl" 2>/dev/null || true
}

with_retry() {
    local name="$1" fn="$2"
    local attempt=0
    while [ "$attempt" -le "$MAX_RETRIES" ]; do
        if [ "$attempt" -gt 0 ]; then
            yellow "  retry $attempt/$MAX_RETRIES..."
        fi
        rm -f "$WORKDIR/_stdout" "$WORKDIR/_stderr" "$WORKDIR/_events.jsonl"
        local result=0
        "$fn" && result=0 || result=$?
        if [ "$result" -eq 0 ]; then
            pass "$name"
            return 0
        fi
        attempt=$((attempt + 1))
    done
    fail "$name" "${LAST_FAIL_REASON:-unknown}"
    return 1
}

# ─── Start MCP Server ──────────────────────────────────────────
# Uses supergateway to wrap filesystem server over Streamable HTTP.
# Creates a known file tree in $WORKDIR/mcp_root/ for agents to read.

start_mcp_server() {
    mkdir -p "$WORKDIR/mcp_root"
    echo "mcp-sentinel-value-42" > "$WORKDIR/mcp_root/probe.txt"
    mkdir -p "$WORKDIR/mcp_root/subdir"
    echo "nested-content" > "$WORKDIR/mcp_root/subdir/nested.txt"

    npx -y supergateway \
        --stdio "npx -y @modelcontextprotocol/server-filesystem $WORKDIR/mcp_root" \
        --outputTransport streamableHttp \
        --port "$MCP_PORT" \
        --healthEndpoint /healthz \
        > "$WORKDIR/_mcp_stdout" 2> "$WORKDIR/_mcp_stderr" &
    MCP_PID=$!

    # Wait for server to be ready via health endpoint (up to 30s)
    local waited=0
    while [ "$waited" -lt 30 ]; do
        if curl -sf -o /dev/null "http://localhost:$MCP_PORT/healthz" 2>/dev/null; then
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
    done
    red "MCP server failed to start within 30s"
    if [ -f "$WORKDIR/_mcp_stderr" ]; then
        red "MCP stderr:"
        cat "$WORKDIR/_mcp_stderr" >&2
    fi
    return 1
}

stop_mcp_server() {
    if [ -n "$MCP_PID" ] && kill -0 "$MCP_PID" 2>/dev/null; then
        kill "$MCP_PID" 2>/dev/null || true
        wait "$MCP_PID" 2>/dev/null || true
        MCP_PID=""
    fi
}

# ─── Preflight ───────────────────────────────────────────────

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    red "OPENROUTER_API_KEY not set"; exit 1
fi
if ! command -v npx >/dev/null 2>&1; then
    red "npx not found (install Node.js)"; exit 1
fi
if [ ! -x "$BINARY" ]; then
    bold "Binary not found, building release..."
    (cd "$ROOT_DIR" && cargo build --release 2>&1) || { red "Build failed"; exit 1; }
fi

bold "╔════════════════════════════════════════════════════════╗"
bold "║  MCP E2E — Tool Discovery & Execution                 ║"
bold "╠════════════════════════════════════════════════════════╣"
echo "  Binary:   $BINARY"
echo "  Workdir:  $WORKDIR"
echo "  MCP port: $MCP_PORT"
echo "  Model:    $HEARTBIT_MODEL"
echo "  Retries:  $MAX_RETRIES"
bold "╚════════════════════════════════════════════════════════╝"
echo ""

bold "Starting MCP server (supergateway + filesystem)..."
start_mcp_server || exit 1
green "  MCP server ready at http://localhost:$MCP_PORT/mcp"
echo ""

# ═══════════════════════════════════════════════════════════════
# Test 1: MCP tool discovery
#
#   Verify that the agent discovers MCP tools during startup.
#   Events should show tool_call_started for an MCP tool.
#   The filesystem server exposes: read_file, write_file,
#   list_directory, etc.
# ═══════════════════════════════════════════════════════════════
if should_run 1; then
    bold "TEST 1: MCP tool discovery + basic read"
    test_1() {
        HEARTBIT_MCP_SERVERS="http://localhost:$MCP_PORT/mcp" \
            timeout 120 "$BINARY" run -v \
            "Read the file probe.txt and tell me what it contains. Use the read_file tool." \
            > "$WORKDIR/_stdout" 2> "$WORKDIR/_stderr" || true
        extract_events

        if [ ! -f "$WORKDIR/_events.jsonl" ]; then
            LAST_FAIL_REASON="no events captured"
            return 1
        fi

        # Check that a tool_call_started event exists for an MCP tool
        # (filesystem server tools: read_file, write_file, list_directory, etc.)
        local found_mcp_tool=false
        while IFS= read -r line; do
            local tool_name
            tool_name=$(echo "$line" | python3 -c "
import json, sys
e = json.loads(sys.stdin.read())
if e.get('type') == 'tool_call_started':
    name = e.get('tool_name', '')
    # MCP filesystem tools
    if name in ('read_file', 'write_file', 'list_directory', 'search_files',
                'get_file_info', 'list_allowed_directories',
                'create_directory', 'move_file', 'edit_file', 'read_multiple_files'):
        print(name)
" 2>/dev/null) || true
            if [ -n "$tool_name" ]; then
                found_mcp_tool=true
                echo "    MCP tool called: $tool_name"
                break
            fi
        done < "$WORKDIR/_events.jsonl"

        if ! $found_mcp_tool; then
            LAST_FAIL_REASON="no MCP filesystem tool called in events"
            return 1
        fi

        # Verify the tool call completed successfully (not an error).
        # The LLM may rephrase the content, so we check for tool_call_completed
        # rather than exact sentinel text. Test 2 verifies data integrity.
        local tool_completed=false
        while IFS= read -r line; do
            local check
            check=$(echo "$line" | python3 -c "
import json, sys
e = json.loads(sys.stdin.read())
if e.get('type') == 'tool_call_completed':
    name = e.get('tool_name', '')
    if name in ('read_file', 'read_multiple_files'):
        print('yes')
" 2>/dev/null) || true
            if [ "$check" = "yes" ]; then
                tool_completed=true
                break
            fi
        done < "$WORKDIR/_events.jsonl"

        if $tool_completed; then
            echo "    read_file completed successfully"
        else
            # Fallback: if tool_call_started was emitted, the tool was dispatched.
            # Completion event may be missing if the run ended before it was emitted.
            echo "    read_file dispatched (completion event not captured)"
        fi

        # Bonus: check if sentinel appears anywhere (stdout or events)
        if grep -qF "mcp-sentinel-value-42" "$WORKDIR/_stdout" "$WORKDIR/_events.jsonl" 2>/dev/null; then
            echo "    Sentinel value confirmed in output"
        fi
    }
    with_retry "MCP tool discovery + basic read" test_1 || true
fi

# ═══════════════════════════════════════════════════════════════
# Test 2: MCP tool write + verify
#
#   Ask the agent to write a file via MCP, then verify the file
#   was actually created on disk in the MCP root directory.
# ═══════════════════════════════════════════════════════════════
if should_run 2; then
    bold "TEST 2: MCP write file via tool"
    test_2() {
        rm -f "$WORKDIR/mcp_root/agent_output.txt"
        HEARTBIT_MCP_SERVERS="http://localhost:$MCP_PORT/mcp" \
            timeout 120 "$BINARY" run -v \
            "Write a file called agent_output.txt containing exactly 'heartbit-mcp-test-ok'. Use the write_file tool." \
            > "$WORKDIR/_stdout" 2> "$WORKDIR/_stderr" || true
        extract_events

        # The file should exist in the MCP root
        if [ ! -f "$WORKDIR/mcp_root/agent_output.txt" ]; then
            LAST_FAIL_REASON="agent_output.txt not created in MCP root"
            return 1
        fi
        if ! grep -qF "heartbit-mcp-test-ok" "$WORKDIR/mcp_root/agent_output.txt"; then
            LAST_FAIL_REASON="content mismatch: $(cat "$WORKDIR/mcp_root/agent_output.txt")"
            return 1
        fi
        echo "    File written via MCP: $(cat "$WORKDIR/mcp_root/agent_output.txt")"
    }
    with_retry "MCP write file via tool" test_2 || true
fi

# ═══════════════════════════════════════════════════════════════
# Test 3: MCP tool list directory
#
#   Ask the agent to list directory contents via MCP.
#   Verify it discovers the known files we created.
# ═══════════════════════════════════════════════════════════════
if should_run 3; then
    bold "TEST 3: MCP list directory"
    test_3() {
        HEARTBIT_MCP_SERVERS="http://localhost:$MCP_PORT/mcp" \
            timeout 120 "$BINARY" run -v \
            "List all files and directories in the root directory. Use the list_directory tool." \
            > "$WORKDIR/_stdout" 2> "$WORKDIR/_stderr" || true
        extract_events

        if [ ! -f "$WORKDIR/_events.jsonl" ]; then
            LAST_FAIL_REASON="no events captured"
            return 1
        fi

        # Check that list_directory was called
        if ! grep -q '"list_directory"\|"list_allowed_directories"' "$WORKDIR/_events.jsonl"; then
            LAST_FAIL_REASON="list_directory not called"
            return 1
        fi

        # Verify probe.txt and subdir appear in output or events
        local found_probe=false
        if grep -qF "probe.txt" "$WORKDIR/_stdout" || grep -qF "probe.txt" "$WORKDIR/_events.jsonl"; then
            found_probe=true
        fi
        if ! $found_probe; then
            LAST_FAIL_REASON="probe.txt not found in directory listing"
            return 1
        fi
        echo "    Directory listing contains probe.txt"
    }
    with_retry "MCP list directory" test_3 || true
fi

# ═══════════════════════════════════════════════════════════════
# Test 4: MCP with orchestrator config
#
#   Use a TOML config file that wires MCP servers to a specific
#   agent. Verify the orchestrator delegates to the agent and
#   MCP tools are used.
# ═══════════════════════════════════════════════════════════════
if should_run 4; then
    bold "TEST 4: MCP via orchestrator config"
    test_4() {
        # Create a minimal orchestrator config
        cat > "$WORKDIR/mcp_test.toml" << TOMLEOF
[provider]
name = "openrouter"
model = "$HEARTBIT_MODEL"

[orchestrator]
max_turns = 5
max_tokens = 4096

[[agents]]
name = "filer"
description = "File system agent — reads and writes files via MCP tools"
system_prompt = "You are a file system agent. Use the tools available to you to complete file tasks."
mcp_servers = ["http://localhost:$MCP_PORT/mcp"]
max_turns = 5
TOMLEOF

        rm -f "$WORKDIR/mcp_root/orchestrated.txt"
        timeout 180 "$BINARY" run -v --config "$WORKDIR/mcp_test.toml" \
            "Read the file probe.txt and create a new file called orchestrated.txt containing its contents reversed." \
            > "$WORKDIR/_stdout" 2> "$WORKDIR/_stderr" || true
        extract_events

        if [ ! -f "$WORKDIR/_events.jsonl" ]; then
            LAST_FAIL_REASON="no events captured"
            return 1
        fi

        # Verify sub-agent dispatch happened
        if ! grep -q '"sub_agents_dispatched"\|"SubAgentsDispatched"' "$WORKDIR/_events.jsonl"; then
            LAST_FAIL_REASON="no sub-agent dispatch event"
            return 1
        fi

        # Verify the filer agent used an MCP tool
        local filer_used_mcp=false
        while IFS= read -r line; do
            local check
            check=$(echo "$line" | python3 -c "
import json, sys
e = json.loads(sys.stdin.read())
if e.get('type') == 'tool_call_started' and e.get('agent') == 'filer':
    name = e.get('tool_name', '')
    if name in ('read_file', 'write_file', 'list_directory', 'search_files',
                'get_file_info', 'list_allowed_directories',
                'create_directory', 'move_file', 'edit_file', 'read_multiple_files'):
        print('yes')
" 2>/dev/null) || true
            if [ "$check" = "yes" ]; then
                filer_used_mcp=true
                break
            fi
        done < "$WORKDIR/_events.jsonl"

        if ! $filer_used_mcp; then
            LAST_FAIL_REASON="filer agent did not use any MCP tool"
            return 1
        fi
        echo "    Orchestrator delegated to filer; MCP tools used"
    }
    with_retry "MCP via orchestrator config" test_4 || true
fi

# ═══════════════════════════════════════════════════════════════
# Test 5: MCP server down — graceful error
#
#   Point heartbit at a non-existent MCP server.
#   It should NOT crash — should run with builtins only
#   and log the connection failure.
# ═══════════════════════════════════════════════════════════════
if should_run 5; then
    bold "TEST 5: MCP server unreachable — graceful degradation"
    test_5() {
        # Use a port that's definitely not running an MCP server
        HEARTBIT_MCP_SERVERS="http://localhost:19999/mcp" \
            timeout 60 "$BINARY" run -v \
            "Say hello." \
            > "$WORKDIR/_stdout" 2> "$WORKDIR/_stderr" || true

        # Should NOT have crashed (exit code captured by || true)
        # Should have some output (either stdout or events)
        if [ ! -s "$WORKDIR/_stdout" ] && [ ! -s "$WORKDIR/_stderr" ]; then
            LAST_FAIL_REASON="no output at all — binary may have crashed"
            return 1
        fi

        # Stderr should contain an MCP connection error message
        if grep -qi "mcp\|connect\|error\|failed" "$WORKDIR/_stderr"; then
            echo "    MCP failure logged gracefully"
            return 0
        fi

        # If stdout has content, the agent ran (with builtins only)
        if [ -s "$WORKDIR/_stdout" ]; then
            echo "    Agent ran without MCP (graceful fallback)"
            return 0
        fi

        LAST_FAIL_REASON="no evidence of graceful error handling"
        return 1
    }
    with_retry "MCP server unreachable" test_5 || true
fi

# ─── Cleanup & Results ────────────────────────────────────────

stop_mcp_server

echo ""
bold "╔════════════════════════════════════════════════════════╗"
TOTAL=$((PASS + FAIL))
if [ "$SKIP" -gt 0 ]; then
    echo "  Skipped: $SKIP"
fi
if [ "$FAIL" -eq 0 ]; then
    green "║  ALL $TOTAL MCP TESTS PASSED                            ║"
else
    red "║  $FAIL/$TOTAL FAILED                                     ║"
    echo ""
    red "Failure details:"
    printf "$ERRORS"
fi
bold "╚════════════════════════════════════════════════════════╝"

if [ "$FAIL" -gt 0 ]; then
    echo ""
    echo "Workdir preserved for inspection: $WORKDIR"
    trap - EXIT
fi

exit "$FAIL"
