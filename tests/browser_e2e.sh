#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# End-to-end browser integration test: multi-agent build-test-fix.
#
# Exercises the full orchestrator pipeline on a realistic task:
#   1. builder  — creates an Axum web server using built-in tools
#   2. tester   — tests it via Chrome DevTools MCP tools
#   3. fixer    — fixes any bugs found, recompiles
#
# Validates: orchestrator delegation, built-in tools, MCP tool
# discovery + execution, inter-agent coordination via blackboard.
#
# These are NOT CI tests. They call a real LLM and cost money.
# Run manually after changes to the orchestrator or MCP pipeline.
#
# Design rules (same as mcp_e2e.sh / squad_e2e.sh):
#   1. Assert on artifacts (files, events), never LLM prose.
#   2. Assertions must be LENIENT — allow the LLM flexibility.
#   3. Each test retries up to $MAX_RETRIES times.
#   4. Individual test: ./browser_e2e.sh 2   (runs only test 2)
#
# Requires: Chrome/Chromium, npx (Node.js), OPENROUTER_API_KEY,
#           target/release/heartbit
# ──────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BINARY="$ROOT_DIR/target/release/heartbit"
WORKDIR="$(mktemp -d)"
MAX_RETRIES="${MAX_RETRIES:-1}"
PASS=0
FAIL=0
SKIP=0
ERRORS=""
FILTER="${1:-}"
CDP_MCP_PORT="${CDP_MCP_PORT:-18400}"
WEBAPP_PORT="${WEBAPP_PORT:-18401}"
CHROME_DEBUG_PORT="${CHROME_DEBUG_PORT:-9222}"
CHROME_PID=""
MCP_PID=""

export HEARTBIT_MODEL="${HEARTBIT_MODEL:-anthropic/claude-sonnet-4}"
export HEARTBIT_BUILDER_MODEL="${HEARTBIT_BUILDER_MODEL:-$HEARTBIT_MODEL}"
export HEARTBIT_TESTER_MODEL="${HEARTBIT_TESTER_MODEL:-$HEARTBIT_MODEL}"
export HEARTBIT_FIXER_MODEL="${HEARTBIT_FIXER_MODEL:-$HEARTBIT_MODEL}"

# ─── Helpers ──────────────────────────────────────────────────

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
            kill_webapp_port
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

# ─── Chrome + MCP lifecycle ──────────────────────────────────

find_chrome() {
    for candidate in \
        google-chrome \
        google-chrome-stable \
        chromium \
        chromium-browser \
        /usr/bin/google-chrome \
        /usr/bin/chromium \
        /usr/bin/chromium-browser \
        /snap/bin/chromium \
        /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome; do
        if command -v "$candidate" >/dev/null 2>&1 || [ -x "$candidate" ]; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

start_chrome() {
    local chrome_bin
    chrome_bin=$(find_chrome) || { red "Chrome/Chromium not found"; return 1; }
    bold "  Starting Chrome headless: $chrome_bin"

    "$chrome_bin" \
        --headless=new \
        --remote-debugging-port="$CHROME_DEBUG_PORT" \
        --no-sandbox \
        --disable-gpu \
        --disable-dev-shm-usage \
        --no-first-run \
        --no-default-browser-check \
        > "$WORKDIR/_chrome_stdout" 2> "$WORKDIR/_chrome_stderr" &
    CHROME_PID=$!

    # Wait for DevTools port to accept connections (up to 30s)
    local waited=0
    while [ "$waited" -lt 30 ]; do
        if curl -sf -o /dev/null "http://localhost:$CHROME_DEBUG_PORT/json/version" 2>/dev/null; then
            green "  Chrome headless ready (PID $CHROME_PID, port $CHROME_DEBUG_PORT)"
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
    done
    red "Chrome failed to start within 30s"
    if [ -f "$WORKDIR/_chrome_stderr" ]; then
        red "Chrome stderr:"
        cat "$WORKDIR/_chrome_stderr" >&2
    fi
    return 1
}

start_cdp_mcp() {
    bold "  Starting Chrome DevTools MCP via supergateway..."

    npx -y supergateway \
        --stdio "npx -y chrome-devtools-mcp@latest --browserUrl http://localhost:$CHROME_DEBUG_PORT" \
        --outputTransport streamableHttp \
        --stateful \
        --port "$CDP_MCP_PORT" \
        --healthEndpoint /healthz \
        > "$WORKDIR/_mcp_stdout" 2> "$WORKDIR/_mcp_stderr" &
    MCP_PID=$!

    # Wait for health endpoint (up to 60s — npm install can be slow)
    local waited=0
    while [ "$waited" -lt 60 ]; do
        if curl -sf -o /dev/null "http://localhost:$CDP_MCP_PORT/healthz" 2>/dev/null; then
            green "  CDP MCP ready at http://localhost:$CDP_MCP_PORT/mcp (PID $MCP_PID)"
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
    done
    red "CDP MCP server failed to start within 60s"
    if [ -f "$WORKDIR/_mcp_stderr" ]; then
        red "MCP stderr:"
        cat "$WORKDIR/_mcp_stderr" >&2
    fi
    return 1
}

kill_webapp_port() {
    # Kill any process listening on the webapp port
    local pid
    pid=$(lsof -ti :"$WEBAPP_PORT" 2>/dev/null || true)
    if [ -n "$pid" ]; then
        kill "$pid" 2>/dev/null || true
        sleep 1
        kill -9 "$pid" 2>/dev/null || true
    fi
    # Kill any webapp binary from the workdir (catches wrong-port zombies)
    pkill -f "$WORKDIR/webapp/target" 2>/dev/null || true
}

cleanup() {
    bold "Cleaning up..."
    if [ -n "$MCP_PID" ] && kill -0 "$MCP_PID" 2>/dev/null; then
        kill "$MCP_PID" 2>/dev/null || true
        wait "$MCP_PID" 2>/dev/null || true
    fi
    if [ -n "$CHROME_PID" ] && kill -0 "$CHROME_PID" 2>/dev/null; then
        kill "$CHROME_PID" 2>/dev/null || true
        wait "$CHROME_PID" 2>/dev/null || true
    fi
    kill_webapp_port
    if [ "$FAIL" -eq 0 ]; then
        rm -rf "$WORKDIR"
    fi
}
trap cleanup EXIT

# ─── Generate TOML config ────────────────────────────────────

generate_config() {
    cat > "$WORKDIR/browser_e2e.toml" << TOMLEOF
[provider]
name = "openrouter"
model = "$HEARTBIT_MODEL"

[provider.retry]
max_retries = 3
base_delay_ms = 500
max_delay_ms = 30000

[orchestrator]
max_turns = 30
max_tokens = 8192
enable_squads = false
dispatch_mode = "sequential"

[[agents]]
name = "builder"
description = "Rust web developer who creates Axum web applications using bash and file tools"
system_prompt = """You are a Rust developer. Build a minimal Axum web app. Use ABSOLUTE paths. Project: $WORKDIR/webapp

Steps (follow EXACTLY, do not add extra steps):
1. Run: cargo init $WORKDIR/webapp
2. Read $WORKDIR/webapp/Cargo.toml
3. Write $WORKDIR/webapp/Cargo.toml with this content:
   [package]
   name = "webapp"
   version = "0.1.0"
   edition = "2021"
   [dependencies]
   axum = "0.8"
   tokio = { version = "1", features = ["full"] }
   serde = { version = "1", features = ["derive"] }
4. Read $WORKDIR/webapp/src/main.rs
5. Write $WORKDIR/webapp/src/main.rs with this EXACT content:

use axum::{routing::{get, post}, Router, Form, response::Html};
use serde::Deserialize;

#[derive(Deserialize)]
struct SubmitForm { message: String }

async fn index() -> Html<String> {
    Html("<h1>Hello from Heartbit</h1><form method=\\"post\\" action=\\"/submit\\"><input name=\\"message\\"><button type=\\"submit\\">Submit</button></form>".to_string())
}

async fn submit(Form(form): Form<SubmitForm>) -> String {
    format!("Received: {}", form.message)
}

#[tokio::main]
async fn main() {
    let app = Router::new()
        .route("/", get(index))
        .route("/submit", post(submit));
    let listener = tokio::net::TcpListener::bind("0.0.0.0:$WEBAPP_PORT").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

6. Run: cargo build --manifest-path $WORKDIR/webapp/Cargo.toml

If build fails, fix and retry. Your task is COMPLETE once cargo build succeeds. Do nothing else after that."""
max_turns = 10

[agents.provider]
name = "openrouter"
model = "$HEARTBIT_BUILDER_MODEL"

[agents.session_prune]
keep_recent_n = 2
pruned_tool_result_max_bytes = 200
preserve_task = true

[[agents]]
name = "tester"
description = "QA engineer who tests web applications using Chrome DevTools browser automation via MCP tools"
system_prompt = """You are a QA tester. Test the web app using Chrome DevTools MCP tools.

IMPORTANT: Use ABSOLUTE paths. Binary: $WORKDIR/webapp/target/debug/webapp

Step 1 — Start server (use this EXACT bash command, all lines together):
  $WORKDIR/webapp/target/debug/webapp > $WORKDIR/server.log 2>&1 &
  sleep 3
  echo "server started on port $WEBAPP_PORT"

Step 2 — Test with Chrome DevTools MCP tools ONLY (not curl, not bash for HTTP):
  a) navigate_page to http://localhost:$WEBAPP_PORT/
  b) take_snapshot — verify "Hello from Heartbit" is present
  c) fill the input field with "test message"
  d) click the submit button
  e) take_snapshot — verify response page

If a tool returns "No snapshot found", take_snapshot again then retry. Never kill Chrome or browser processes.

Step 3 — Write PASS/FAIL results to $WORKDIR/test_results.txt
Step 4 — Kill server: pkill -f "$WORKDIR/webapp/target/debug/webapp" || true

RULES: Never read source code. Never run cargo build. Never use curl. Only use MCP tools for browser interaction."""
mcp_servers = ["http://localhost:$CDP_MCP_PORT/mcp"]
max_turns = 25

[agents.provider]
name = "openrouter"
model = "$HEARTBIT_TESTER_MODEL"

[agents.session_prune]
keep_recent_n = 2
pruned_tool_result_max_bytes = 200
preserve_task = true

[[agents]]
name = "fixer"
description = "Senior Rust developer who fixes bugs found during testing"
system_prompt = """You are a senior Rust developer. Read test results and fix any bugs.

IMPORTANT: Use ABSOLUTE paths everywhere.
Test results: $WORKDIR/test_results.txt
Source code: $WORKDIR/webapp/src/main.rs

Steps:
1. Read $WORKDIR/test_results.txt
2. If all tests passed, write "ALL TESTS PASSED" to $WORKDIR/fix_report.txt
3. If tests failed: read source, fix bugs, run cargo build --manifest-path $WORKDIR/webapp/Cargo.toml, write fix summary to $WORKDIR/fix_report.txt
4. If no test results exist, write "NO TEST RESULTS FOUND" to $WORKDIR/fix_report.txt"""
max_turns = 15

[agents.provider]
name = "openrouter"
model = "$HEARTBIT_FIXER_MODEL"

[agents.session_prune]
keep_recent_n = 2
pruned_tool_result_max_bytes = 200
preserve_task = true
TOMLEOF
}

# ─── Preflight ────────────────────────────────────────────────

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    red "OPENROUTER_API_KEY not set"; exit 1
fi
if ! command -v npx >/dev/null 2>&1; then
    red "npx not found (install Node.js)"; exit 1
fi
if ! find_chrome >/dev/null 2>&1; then
    red "Chrome/Chromium not found"; exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
    red "python3 not found"; exit 1
fi
if [ ! -x "$BINARY" ]; then
    bold "Binary not found, building release..."
    (cd "$ROOT_DIR" && cargo build --release 2>&1) || { red "Build failed"; exit 1; }
fi

bold "╔════════════════════════════════════════════════════════════╗"
bold "║  Browser E2E — Multi-Agent Build-Test-Fix Pipeline        ║"
bold "╠════════════════════════════════════════════════════════════╣"
echo "  Binary:      $BINARY"
echo "  Workdir:     $WORKDIR"
echo "  CDP MCP:     http://localhost:$CDP_MCP_PORT/mcp"
echo "  Webapp port: $WEBAPP_PORT"
echo "  Chrome dbg:  $CHROME_DEBUG_PORT"
echo "  Model:       $HEARTBIT_MODEL"
echo "  Builder:     $HEARTBIT_BUILDER_MODEL"
echo "  Tester:      $HEARTBIT_TESTER_MODEL"
echo "  Fixer:       $HEARTBIT_FIXER_MODEL"
echo "  Dispatch:    sequential"
echo "  Retries:     $MAX_RETRIES"
bold "╚════════════════════════════════════════════════════════════╝"
echo ""

# ─── Start Chrome + CDP MCP ───────────────────────────────────

bold "Starting Chrome headless..."
start_chrome || exit 1

bold "Starting Chrome DevTools MCP server..."
start_cdp_mcp || exit 1
echo ""

# ═══════════════════════════════════════════════════════════════
# Test 1: Full build-test-fix pipeline
#
#   Run the 3-agent orchestrator: builder creates axum app,
#   tester verifies via CDP, fixer patches any issues.
#   Assert on artifacts and event stream.
# ═══════════════════════════════════════════════════════════════
if should_run 1; then
    bold "TEST 1: Full build-test-fix pipeline"
    test_1() {
        # Clean up any prior webapp artifacts
        rm -rf "$WORKDIR/webapp" "$WORKDIR/test_results.txt" "$WORKDIR/fix_report.txt"
        kill_webapp_port

        generate_config

        TASK="First delegate to 'builder' to create and build the Axum web app. After builder completes, delegate to 'tester' to test the app using Chrome DevTools MCP tools. After tester completes, delegate to 'fixer' to review and fix any issues."

        timeout 600 "$BINARY" run -v --config "$WORKDIR/browser_e2e.toml" "$TASK" \
            > "$WORKDIR/_stdout" 2> "$WORKDIR/_stderr" || true
        extract_events

        # Save stable copy immediately (before assertions) so tests 2 & 3
        # can analyze events even if test 1 fails on a later assertion.
        if [ -f "$WORKDIR/_events.jsonl" ] && [ -s "$WORKDIR/_events.jsonl" ]; then
            cp "$WORKDIR/_events.jsonl" "$WORKDIR/events_stable.jsonl"
        fi

        kill_webapp_port

        # ── Sub-assertions ──

        # 1. Events captured
        if [ ! -f "$WORKDIR/_events.jsonl" ] || [ ! -s "$WORKDIR/_events.jsonl" ]; then
            LAST_FAIL_REASON="no events captured"
            return 1
        fi
        local event_count
        event_count=$(wc -l < "$WORKDIR/_events.jsonl" | tr -d ' ')
        echo "    Events captured: $event_count"

        # 2. Cargo.toml exists (builder created the project)
        if [ ! -f "$WORKDIR/webapp/Cargo.toml" ]; then
            LAST_FAIL_REASON="webapp/Cargo.toml not created — builder failed"
            return 1
        fi
        echo "    webapp/Cargo.toml exists"

        # 3. main.rs exists and contains axum
        if [ ! -f "$WORKDIR/webapp/src/main.rs" ]; then
            LAST_FAIL_REASON="webapp/src/main.rs not created"
            return 1
        fi
        if ! grep -qi "axum" "$WORKDIR/webapp/src/main.rs"; then
            LAST_FAIL_REASON="main.rs does not reference axum"
            return 1
        fi
        echo "    main.rs contains axum"

        # 4. Delegation events present
        if ! grep -q '"sub_agents_dispatched"' "$WORKDIR/_events.jsonl"; then
            LAST_FAIL_REASON="no sub_agents_dispatched events"
            return 1
        fi
        echo "    Delegation events present"

        # 5. Builder agent events present
        if ! grep -q '"agent":"builder"' "$WORKDIR/_events.jsonl" && \
           ! grep -q '"agent": "builder"' "$WORKDIR/_events.jsonl"; then
            LAST_FAIL_REASON="no events from builder agent"
            return 1
        fi
        echo "    Builder agent events present"

        # 6. Tester agent events present
        if ! grep -q '"agent":"tester"' "$WORKDIR/_events.jsonl" && \
           ! grep -q '"agent": "tester"' "$WORKDIR/_events.jsonl"; then
            LAST_FAIL_REASON="no events from tester agent"
            return 1
        fi
        echo "    Tester agent events present"

        # 7. Chrome DevTools MCP tools used by tester
        # Check for any non-builtin tool call from tester (MCP tools discovered dynamically)
        local cdp_tool_used
        cdp_tool_used=$(python3 -c "
import json, sys
builtins = {'bash', 'read', 'write', 'edit', 'list', 'glob', 'grep', 'patch', 'todo',
            'skill', 'question', 'delegate_task', 'form_squad',
            'blackboard_read', 'blackboard_write', 'blackboard_list',
            'memory_store', 'memory_recall', 'memory_update', 'memory_forget', 'memory_consolidate'}
found = set()
with open(sys.argv[1]) as f:
    for line in f:
        try:
            e = json.loads(line)
            if e.get('type') == 'tool_call_started' and e.get('agent') == 'tester':
                name = e.get('tool_name', '')
                if name and name not in builtins:
                    found.add(name)
        except Exception:
            pass
if found:
    print(','.join(sorted(found)))
else:
    print('')
" "$WORKDIR/_events.jsonl" 2>/dev/null) || true

        if [ -n "$cdp_tool_used" ]; then
            echo "    CDP MCP tools used by tester: $cdp_tool_used"
        else
            echo "    WARNING: No non-builtin tool calls from tester (CDP tools not invoked)"
        fi

        # 8. Fixer agent ran
        if ! grep -q '"agent":"fixer"' "$WORKDIR/_events.jsonl" && \
           ! grep -q '"agent": "fixer"' "$WORKDIR/_events.jsonl"; then
            LAST_FAIL_REASON="fixer agent never ran"
            return 1
        fi
        echo "    Fixer agent events present"

        # 9. test_results.txt created by tester (warning only)
        if [ ! -f "$WORKDIR/test_results.txt" ]; then
            echo "    WARNING: test_results.txt not created by tester"
        fi

        # 10. fix_report.txt created by fixer (warning only)
        if [ ! -f "$WORKDIR/fix_report.txt" ]; then
            echo "    WARNING: fix_report.txt not created by fixer"
        fi

        # 11. Final cargo build passes
        if cargo build --manifest-path "$WORKDIR/webapp/Cargo.toml" 2>/dev/null; then
            echo "    Final cargo build: OK"
        else
            LAST_FAIL_REASON="webapp does not compile after fixer ran"
            return 1
        fi

    }
    with_retry "Full build-test-fix pipeline" test_1 || true
fi

# ═══════════════════════════════════════════════════════════════
# Test 2: Event stream integrity
#
#   Verify event stream from Test 1 is well-formed:
#   - All events are valid JSON
#   - Required event types present
#   - Orchestrator agent in events
# ═══════════════════════════════════════════════════════════════
if should_run 2; then
    bold "TEST 2: Event stream integrity"
    test_2() {
        # Reuse stable events from test 1
        local evfile="$WORKDIR/events_stable.jsonl"
        if [ ! -f "$evfile" ] || [ ! -s "$evfile" ]; then
            LAST_FAIL_REASON="no events file (run test 1 first)"
            return 1
        fi

        # All events are valid JSON
        local invalid_count
        invalid_count=$(python3 -c "
import json, sys
count = 0
with open(sys.argv[1]) as f:
    for line in f:
        try:
            json.loads(line)
        except Exception:
            count += 1
print(count)
" "$evfile")

        if [ "$invalid_count" -ne 0 ]; then
            LAST_FAIL_REASON="$invalid_count events are not valid JSON"
            return 1
        fi
        echo "    All events are valid JSON"

        # Required event types present (run_completed may be absent if timed out)
        local types_found
        types_found=$(python3 -c "
import json, sys
types = set()
with open(sys.argv[1]) as f:
    for line in f:
        try:
            e = json.loads(line)
            types.add(e.get('type', ''))
        except Exception:
            pass
required = {'run_started', 'llm_response', 'tool_call_started', 'tool_call_completed'}
missing = required - types
if missing:
    print('MISSING:' + ','.join(sorted(missing)))
else:
    print('OK')
" "$evfile")

        if [ "$types_found" != "OK" ]; then
            LAST_FAIL_REASON="$types_found"
            return 1
        fi
        echo "    All required event types present"

        # Orchestrator agent in events (use flag variable, not sys.exit inside try/except)
        local has_orchestrator
        has_orchestrator=$(python3 -c "
import json, sys
found = False
with open(sys.argv[1]) as f:
    for line in f:
        try:
            e = json.loads(line)
            if e.get('agent') == 'orchestrator':
                found = True
                break
        except Exception:
            pass
print('yes' if found else 'no')
" "$evfile")

        if [ "$has_orchestrator" != "yes" ]; then
            LAST_FAIL_REASON="no orchestrator agent in events"
            return 1
        fi
        echo "    Orchestrator agent present in events"
    }
    with_retry "Event stream integrity" test_2 || true
fi

# ═══════════════════════════════════════════════════════════════
# Test 3: Token accounting
#
#   Verify at least one run_completed event has non-zero tokens.
#   (Orchestrator may not complete if agents time out.)
# ═══════════════════════════════════════════════════════════════
if should_run 3; then
    bold "TEST 3: Token accounting"
    test_3() {
        local evfile="$WORKDIR/events_stable.jsonl"
        if [ ! -f "$evfile" ] || [ ! -s "$evfile" ]; then
            LAST_FAIL_REASON="no events file (run test 1 first)"
            return 1
        fi

        # Check any run_completed event (not just orchestrator — it may not finish)
        local token_info
        token_info=$(python3 -c "
import json, sys
best_agent = ''
best_tokens = 0
with open(sys.argv[1]) as f:
    for line in f:
        try:
            e = json.loads(line)
            if e.get('type') == 'run_completed':
                usage = e.get('total_usage', {})
                total = usage.get('input_tokens', 0) + usage.get('output_tokens', 0)
                agent = e.get('agent', '?')
                # Prefer orchestrator, but accept any agent
                if agent == 'orchestrator' or total > best_tokens:
                    best_agent = agent
                    best_tokens = total
                    inp = usage.get('input_tokens', 0)
                    out = usage.get('output_tokens', 0)
                    if agent == 'orchestrator':
                        break
        except Exception:
            pass
if best_tokens > 0:
    print(f'{best_agent},{inp},{out}')
else:
    print(',0,0')
" "$evfile")

        local agent input_tokens output_tokens
        agent=$(echo "$token_info" | cut -d, -f1)
        input_tokens=$(echo "$token_info" | cut -d, -f2)
        output_tokens=$(echo "$token_info" | cut -d, -f3)

        if [ "$input_tokens" -eq 0 ] && [ "$output_tokens" -eq 0 ]; then
            LAST_FAIL_REASON="no run_completed event with non-zero tokens"
            return 1
        fi
        echo "    Tokens ($agent): $input_tokens in / $output_tokens out"
    }
    with_retry "Token accounting" test_3 || true
fi

# ─── Event timeline (informational) ──────────────────────────

# Prefer stable copy from test 1, fall back to transient
EVENTS_FILE="$WORKDIR/events_stable.jsonl"
[ -f "$EVENTS_FILE" ] || EVENTS_FILE="$WORKDIR/_events.jsonl"

if [ -f "$EVENTS_FILE" ] && [ -s "$EVENTS_FILE" ]; then
    echo ""
    bold "── Event Timeline ──"
    python3 - "$EVENTS_FILE" <<'PYEOF'
import json, sys

events = []
with open(sys.argv[1]) as f:
    for line in f:
        try:
            events.append(json.loads(line))
        except Exception:
            pass

if events:
    print(f"  {len(events)} events total:")
    hdr = f"  {'#':>3} {'Agent':<16} {'Type':<24} Details"
    print(hdr)
    print(f"  {'─'*3} {'─'*16} {'─'*24} {'─'*50}")
    for i, e in enumerate(events):
        agent = e.get("agent", "?")[:16]
        etype = e.get("type", "?")[:24]
        details = ""
        if etype == "llm_response":
            text = e.get("text", "")[:50].replace("\n", "\\n")
            lat = e.get("latency_ms", 0)
            model = (e.get("model") or "")[:30]
            details = f'lat={lat}ms model={model} "{text}"'
        elif etype == "tool_call_started":
            tool = e.get("tool_name", "?")
            inp = e.get("input", "")[:40].replace("\n", "\\n")
            details = f'tool={tool} input="{inp}"'
        elif etype == "tool_call_completed":
            tool = e.get("tool_name", "?")
            out = e.get("output", "")[:40].replace("\n", "\\n")
            dur = e.get("duration_ms", 0)
            err = " [ERR]" if e.get("is_error") else ""
            details = f'tool={tool} dur={dur}ms{err} out="{out}"'
        elif etype == "run_started":
            task = e.get("task", "")[:50].replace("\n", "\\n")
            details = f'task="{task}"'
        elif etype == "run_completed":
            u = e.get("total_usage", {})
            details = f"in={u.get('input_tokens',0)} out={u.get('output_tokens',0)} tools={e.get('tool_calls_made',0)}"
        elif "sub_agent" in etype:
            agents = e.get("agents", [e.get("agent", "")])
            success = ""
            if "success" in e:
                success = " OK" if e["success"] else " FAIL"
            details = f"agents={agents}{success}"
        print(f"  {i:>3} {agent:<16} {etype:<24} {details[:70]}")
PYEOF
fi

# ─── Token summary (informational) ───────────────────────────

if [ -f "$EVENTS_FILE" ] && [ -s "$EVENTS_FILE" ]; then
    echo ""
    bold "── Token Usage ──"
    python3 - "$EVENTS_FILE" <<'PYEOF'
import json, sys
with open(sys.argv[1]) as f:
    for line in f:
        try:
            e = json.loads(line)
            if e.get("type") == "run_completed":
                agent = e.get("agent", "?")
                usage = e.get("total_usage", {})
                inp = usage.get("input_tokens", 0)
                out = usage.get("output_tokens", 0)
                tools = e.get("tool_calls_made", 0)
                print(f"  {agent}: {inp} in / {out} out | tools: {tools}")
        except Exception:
            pass
PYEOF
fi

# ─── Results ──────────────────────────────────────────────────

echo ""
echo "  Event log: $WORKDIR/_events.jsonl"
echo "  Stdout:    $WORKDIR/_stdout"
echo "  Stderr:    $WORKDIR/_stderr"
echo "  Config:    $WORKDIR/browser_e2e.toml"
echo ""

bold "╔════════════════════════════════════════════════════════════╗"
TOTAL=$((PASS + FAIL))
if [ "$SKIP" -gt 0 ]; then
    echo "  Skipped: $SKIP"
fi
if [ "$FAIL" -eq 0 ]; then
    green "║  ALL $TOTAL BROWSER E2E TESTS PASSED                        ║"
else
    red "║  $FAIL/$TOTAL FAILED                                          ║"
    echo ""
    red "Failure details:"
    printf "$ERRORS"
fi
bold "╚════════════════════════════════════════════════════════════╝"

if [ "$FAIL" -gt 0 ]; then
    echo ""
    echo "Workdir preserved for inspection: $WORKDIR"
    trap - EXIT
fi

exit "$FAIL"
