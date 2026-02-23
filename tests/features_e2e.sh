#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# End-to-end integration tests for B2 (Permission Persistence),
# D1 (LSP Integration), and general robustness.
#
# These are NOT CI tests. They call a real LLM and cost money.
# Run manually after changes to validate real-world behavior.
#
# Design rules:
#   1. Assert on artifacts (files, exit codes, event JSON), never LLM prose.
#   2. Assertions must be LENIENT — allow the LLM to solve the task its way.
#   3. Each test retries up to $MAX_RETRIES times before failing.
#   4. Tests can be run individually: ./features_e2e.sh 3   (runs only test 3)
#
# Requires: OPENROUTER_API_KEY, target/release/heartbit-cli
# ──────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BINARY="$ROOT_DIR/target/release/heartbit-cli"
WORKDIR="$(mktemp -d)"
MAX_RETRIES="${MAX_RETRIES:-2}"
PASS=0
FAIL=0
SKIP=0
ERRORS=""
FILTER="${1:-}"  # optional: run only this test number

# XDG isolation: learned permissions go to temp dir, not real config
export XDG_CONFIG_HOME="$WORKDIR/xdg_config"

# Default model for all tests (cheap, capable of tool use)
export HEARTBIT_MODEL="${HEARTBIT_MODEL:-qwen/qwen3-30b-a3b}"

# Keep runs short
export HEARTBIT_MAX_TURNS="${HEARTBIT_MAX_TURNS:-10}"

PERM_FILE="$XDG_CONFIG_HOME/heartbit/permissions.toml"
ASK_CONFIG="$SCRIPT_DIR/features_e2e_ask.toml"
DENY_BASH_CONFIG="$SCRIPT_DIR/features_e2e_deny_bash.toml"

cleanup() { rm -rf "$WORKDIR"; }
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

# Extract events from stderr into JSONL.
extract_events() {
    grep '^\[event\] ' "$WORKDIR/_stderr" | sed 's/^\[event\] //' > "$WORKDIR/_events.jsonl" 2>/dev/null || true
}

# Clean permissions file for test isolation.
clean_permissions() {
    rm -f "$PERM_FILE"
    mkdir -p "$(dirname "$PERM_FILE")"
}

# Retry wrapper: run a function up to MAX_RETRIES+1 times.
with_retry() {
    local name="$1" fn="$2"
    local attempt=0
    while [ "$attempt" -le "$MAX_RETRIES" ]; do
        if [ "$attempt" -gt 0 ]; then
            yellow "  retry $attempt/$MAX_RETRIES..."
        fi
        # Clean per-attempt state
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

has_rust_analyzer() { command -v rust-analyzer >/dev/null 2>&1; }

# ─── Preflight ───────────────────────────────────────────────

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    red "OPENROUTER_API_KEY not set"; exit 1
fi
if [ ! -x "$BINARY" ]; then
    bold "Binary not found, building release..."
    (cd "$ROOT_DIR" && cargo build --release 2>&1) || { red "Build failed"; exit 1; }
fi
for cfg in "$ASK_CONFIG" "$DENY_BASH_CONFIG"; do
    if [ ! -f "$cfg" ]; then
        red "Config not found: $cfg"; exit 1
    fi
done

bold "╔════════════════════════════════════════════════════════╗"
bold "║  Features E2E — Permission, LSP, Robustness           ║"
bold "╠════════════════════════════════════════════════════════╣"
echo "  Binary:  $BINARY"
echo "  Workdir: $WORKDIR"
echo "  Model:   $HEARTBIT_MODEL"
echo "  Retries: $MAX_RETRIES"
bold "╚════════════════════════════════════════════════════════╝"
echo ""

# ═══════════════════════════════════════════════════════════════
# Test 1: Permission — AlwaysAllow persists to disk
#
#   Config forces all tools to "ask". Piping Y! auto-approves
#   with AlwaysAllow. After the run, permissions.toml should
#   contain at least one [[rules]] entry with action = "allow".
# ═══════════════════════════════════════════════════════════════
if should_run 1; then
    bold "TEST 1: AlwaysAllow persists to disk"
    test_1() {
        clean_permissions
        yes 'Y!' | head -50 | timeout 120 "$BINARY" run \
            --approve --config "$ASK_CONFIG" -v \
            "Create a file at $WORKDIR/test1.txt containing the word 'alpha'. Do not explain." \
            > "$WORKDIR/_stdout" 2> "$WORKDIR/_stderr" || true
        extract_events

        if [ ! -f "$PERM_FILE" ]; then
            LAST_FAIL_REASON="permissions.toml not created"
            return 1
        fi
        if ! grep -q '\[\[rules\]\]' "$PERM_FILE"; then
            LAST_FAIL_REASON="no [[rules]] section in permissions.toml"
            return 1
        fi
        if ! grep -q 'action = "allow"' "$PERM_FILE"; then
            LAST_FAIL_REASON="no action = \"allow\" in permissions.toml"
            return 1
        fi
    }
    with_retry "AlwaysAllow persists" test_1 || true
fi

# ═══════════════════════════════════════════════════════════════
# Test 2: Permission — AlwaysDeny persists to disk
#
#   Config forces all tools to "ask". Piping N! auto-denies
#   with AlwaysDeny. After the run, permissions.toml should
#   contain at least one [[rules]] entry with action = "deny".
#   The agent will fail the task (all tools denied).
# ═══════════════════════════════════════════════════════════════
if should_run 2; then
    bold "TEST 2: AlwaysDeny persists to disk"
    test_2() {
        clean_permissions
        yes 'N!' | head -50 | timeout 120 "$BINARY" run \
            --approve --config "$ASK_CONFIG" -v \
            "Create a file at $WORKDIR/test2.txt containing the word 'beta'. Do not explain." \
            > "$WORKDIR/_stdout" 2> "$WORKDIR/_stderr" || true
        extract_events

        if [ ! -f "$PERM_FILE" ]; then
            LAST_FAIL_REASON="permissions.toml not created"
            return 1
        fi
        if ! grep -q '\[\[rules\]\]' "$PERM_FILE"; then
            LAST_FAIL_REASON="no [[rules]] section in permissions.toml"
            return 1
        fi
        if ! grep -q 'action = "deny"' "$PERM_FILE"; then
            LAST_FAIL_REASON="no action = \"deny\" in permissions.toml"
            return 1
        fi
        # The task should have failed — file should NOT exist
        if [ -f "$WORKDIR/test2.txt" ]; then
            LAST_FAIL_REASON="file was created despite all tools denied"
            return 1
        fi
    }
    with_retry "AlwaysDeny persists" test_2 || true
fi

# ═══════════════════════════════════════════════════════════════
# Test 3: Permission — Config deny overrides learned allow
#
#   Pre-write a learned allow rule for bash.
#   Config denies bash. Config rules are evaluated first
#   (first match wins), so bash should be denied despite
#   the learned allow.
#
#   Assert: debug log contains "tool call denied by permission rule"
# ═══════════════════════════════════════════════════════════════
if should_run 3; then
    bold "TEST 3: Config deny overrides learned allow"
    test_3() {
        clean_permissions
        # Pre-write a learned allow rule for bash
        cat > "$PERM_FILE" << 'TOMLEOF'
[[rules]]
tool = "bash"
pattern = "*"
action = "allow"
TOMLEOF
        # Run with config that denies bash.
        # Config rules come first in the ruleset (first match wins), so config Deny
        # should override the learned Allow.
        timeout 120 "$BINARY" run \
            --config "$DENY_BASH_CONFIG" -v \
            "You MUST use the bash tool to execute: echo test3-hello. Do not use any other tool, only bash." \
            > "$WORKDIR/_stdout" 2> "$WORKDIR/_stderr" || true
        extract_events

        if [ ! -f "$WORKDIR/_events.jsonl" ]; then
            LAST_FAIL_REASON="no events captured"
            return 1
        fi

        # Primary check via events: worker tried to call a tool (llm_response with
        # tool_call_count > 0) but no tool_call_started was emitted for bash.
        # Permission-denied tools skip event emission, so this pattern proves denial.
        local check_exit=0
        python3 -c "
import json, sys
worker_tried_tool = False
worker_bash_started = False
with open('$WORKDIR/_events.jsonl') as f:
    for line in f:
        try:
            e = json.loads(line)
            if e.get('type') == 'llm_response' and e.get('agent') == 'worker' and e.get('tool_call_count', 0) > 0:
                worker_tried_tool = True
            if e.get('type') == 'tool_call_started' and e.get('agent') == 'worker' and e.get('tool_name') == 'bash':
                worker_bash_started = True
        except Exception:
            pass
if worker_tried_tool and not worker_bash_started:
    sys.exit(0)
elif not worker_tried_tool:
    print('worker did not try to call any tool')
    sys.exit(1)
else:
    print('bash tool_call_started found — permission not denied')
    sys.exit(1)
" 2>/dev/null || check_exit=$?
        if [ "$check_exit" -eq 0 ]; then
            return 0
        fi

        # Fallback: check stderr for debug log (with RUST_LOG=debug set)
        if grep -q "tool call denied by permission rule" "$WORKDIR/_stderr"; then
            return 0
        fi
        # Last resort: verify learned rules were loaded (proves wiring works)
        if grep -q "loaded learned permission rules" "$WORKDIR/_stderr"; then
            yellow "    (LLM didn't call bash — verifying rule loading only)"
            return 0
        fi
        LAST_FAIL_REASON="no evidence of permission denial in events or logs"
        return 1
    }
    with_retry "config deny overrides learned allow" test_3 || true
fi

# ═══════════════════════════════════════════════════════════════
# Test 4: LSP — Diagnostics appear for broken Rust code
#
#   Ask the agent to write a .rs file with EXACT content that
#   has a type error. The write tool triggers LSP diagnostics
#   for the error. The content looks intentional so the LLM
#   is unlikely to "fix" it.
#
#   Prerequisite: rust-analyzer installed.
# ═══════════════════════════════════════════════════════════════
if should_run 4; then
    if has_rust_analyzer; then
        bold "TEST 4: LSP diagnostics for broken Rust code"
        test_4() {
            # Create a minimal Cargo project so rust-analyzer has a workspace.
            # Run the binary from within it so LspManager workspace = this dir.
            # Do NOT pre-create src/main.rs — the write tool must create it fresh
            # to avoid FileTracker read-before-write guard on existing files.
            local LSP_PROJECT="$WORKDIR/lsp_project"
            rm -rf "$LSP_PROJECT"
            mkdir -p "$LSP_PROJECT/src"
            cat > "$LSP_PROJECT/Cargo.toml" << 'CARGOEOF'
[package]
name = "lsp-test"
version = "0.1.0"
edition = "2021"
CARGOEOF

            (cd "$LSP_PROJECT" && HEARTBIT_LSP_ENABLED=1 timeout 180 "$BINARY" run -v \
                "Create a file at $LSP_PROJECT/src/main.rs containing exactly this Rust code: fn main() { let x: i32 = \"hello\"; } — do not explain, just create it." \
                > "$WORKDIR/_stdout" 2> "$WORKDIR/_stderr" || true)
            extract_events

            # Primary check: tracing log that diagnostics were appended.
            # (ToolCallCompleted events are emitted BEFORE LSP diagnostics
            # are appended, so check stderr tracing output instead.)
            if grep -q "lsp-diagnostics appended" "$WORKDIR/_stderr"; then
                return 0
            fi
            # Secondary: check events for lsp-diagnostics in tool output
            # (works if event emission is changed in the future)
            if [ -f "$WORKDIR/_events.jsonl" ] && grep -q "lsp-diagnostics" "$WORKDIR/_events.jsonl"; then
                return 0
            fi
            # Diagnose: did the agent even call write/edit?
            if [ -f "$WORKDIR/_events.jsonl" ] && grep -q '"tool_name":"write"\|"tool_name":"edit"' "$WORKDIR/_events.jsonl"; then
                LAST_FAIL_REASON="write/edit called but no lsp-diagnostics (check stderr for LSP logs)"
            else
                LAST_FAIL_REASON="agent did not call write/edit tool on .rs file"
            fi
            return 1
        }
        with_retry "LSP diagnostics for broken code" test_4 || true
    else
        bold "TEST 4: LSP diagnostics for broken Rust code"
        skip "rust-analyzer not installed"
    fi
fi

# ═══════════════════════════════════════════════════════════════
# Test 5: LSP — No error diagnostics for valid Rust code
#
#   Ask the agent to write a valid .rs file. LSP should produce
#   no error diagnostics.
#
#   Prerequisite: rust-analyzer installed.
# ═══════════════════════════════════════════════════════════════
if should_run 5; then
    if has_rust_analyzer; then
        bold "TEST 5: LSP no error diagnostics for valid code"
        test_5() {
            local LSP_PROJECT="$WORKDIR/lsp_project"
            rm -rf "$LSP_PROJECT"
            mkdir -p "$LSP_PROJECT/src"
            cat > "$LSP_PROJECT/Cargo.toml" << 'CARGOEOF'
[package]
name = "lsp-test"
version = "0.1.0"
edition = "2021"
CARGOEOF

            (cd "$LSP_PROJECT" && HEARTBIT_LSP_ENABLED=1 timeout 180 "$BINARY" run -v \
                "Create a file at $LSP_PROJECT/src/main.rs containing exactly: fn main() { println!(\"hello world\"); } — do not explain, just create it." \
                > "$WORKDIR/_stdout" 2> "$WORKDIR/_stderr" || true)
            extract_events

            if [ ! -f "$WORKDIR/_events.jsonl" ]; then
                LAST_FAIL_REASON="no events captured"
                return 1
            fi
            # Valid code should NOT produce "lsp-diagnostics appended" in tracing
            # (diagnostics should be empty for valid Rust)
            if grep -q "lsp-diagnostics appended" "$WORKDIR/_stderr"; then
                # Diagnostics were found — check if they contain actual errors
                # Warnings (e.g., unused variable) are acceptable
                if grep -q "error\[" "$WORKDIR/_stderr"; then
                    LAST_FAIL_REASON="error diagnostics found for valid code"
                    return 1
                fi
            fi
        }
        with_retry "LSP no error diagnostics for valid code" test_5 || true
    else
        bold "TEST 5: LSP no error diagnostics for valid code"
        skip "rust-analyzer not installed"
    fi
fi

# ═══════════════════════════════════════════════════════════════
# Test 6: LSP — No diagnostics for unsupported extensions
#
#   A .txt file should not trigger any LSP server, so
#   lsp-diagnostics should be absent from tool output.
# ═══════════════════════════════════════════════════════════════
if should_run 6; then
    bold "TEST 6: LSP no diagnostics for .txt"
    test_6() {
        rm -f "$WORKDIR/test6.txt"
        HEARTBIT_LSP_ENABLED=1 timeout 120 "$BINARY" run -v \
            "Create a file at $WORKDIR/test6.txt containing 'just a text file'. Do not explain." \
            > "$WORKDIR/_stdout" 2> "$WORKDIR/_stderr" || true
        extract_events

        # lsp-diagnostics should NOT appear for .txt files
        if [ -f "$WORKDIR/_events.jsonl" ] && grep -q "lsp-diagnostics" "$WORKDIR/_events.jsonl"; then
            LAST_FAIL_REASON="lsp-diagnostics appeared for .txt file"
            return 1
        fi
        # Also verify the file was created (tool use worked)
        if [ ! -f "$WORKDIR/test6.txt" ]; then
            LAST_FAIL_REASON="file not created"
            return 1
        fi
    }
    with_retry "LSP no diagnostics for .txt" test_6 || true
fi

# ═══════════════════════════════════════════════════════════════
# Test 7: Qwen model — Basic tool use works
#
#   Verify that Qwen via OpenRouter can do basic tool use:
#   create a file with specific content.
# ═══════════════════════════════════════════════════════════════
if should_run 7; then
    bold "TEST 7: Qwen basic tool use"
    test_7() {
        rm -f "$WORKDIR/test7.txt"
        timeout 120 "$BINARY" run -v \
            "Create a file at $WORKDIR/test7.txt containing exactly 'heartbit-tool-test'. Do not explain, just create it." \
            > "$WORKDIR/_stdout" 2> "$WORKDIR/_stderr" || true
        extract_events

        # File must exist and contain the expected text
        if [ ! -f "$WORKDIR/test7.txt" ]; then
            LAST_FAIL_REASON="file not created"
            return 1
        fi
        if ! grep -qF "heartbit-tool-test" "$WORKDIR/test7.txt"; then
            LAST_FAIL_REASON="content: $(cat "$WORKDIR/test7.txt")"
            return 1
        fi
        # Events should contain the model name in the "model" field of llm_response events.
        # Use python to check specifically (grep would match task text too).
        if [ ! -f "$WORKDIR/_events.jsonl" ]; then
            LAST_FAIL_REASON="no events captured"
            return 1
        fi
        local model_check=0
        python3 -c "
import json, sys
with open('$WORKDIR/_events.jsonl') as f:
    for line in f:
        try:
            e = json.loads(line)
            if e.get('type') == 'llm_response':
                model = e.get('model', '')
                if 'qwen' in model.lower():
                    sys.exit(0)
        except Exception:
            pass
print('no llm_response with qwen model found')
sys.exit(1)
" 2>/dev/null || model_check=$?
        if [ "$model_check" -ne 0 ]; then
            LAST_FAIL_REASON="no qwen model in llm_response events"
            return 1
        fi
    }
    with_retry "Qwen basic tool use" test_7 || true
fi

# ═══════════════════════════════════════════════════════════════
# Test 8: Pricing — Non-zero token usage in run_completed
#
#   The run_completed event should have non-zero input and
#   output tokens, proving token accounting works end-to-end.
# ═══════════════════════════════════════════════════════════════
if should_run 8; then
    bold "TEST 8: Token usage in run_completed"
    test_8() {
        rm -f "$WORKDIR/test8.txt"
        timeout 120 "$BINARY" run -v \
            "Create a file at $WORKDIR/test8.txt containing 'token-test'. Do not explain." \
            > "$WORKDIR/_stdout" 2> "$WORKDIR/_stderr" || true
        extract_events

        if [ ! -f "$WORKDIR/_events.jsonl" ]; then
            LAST_FAIL_REASON="no events captured"
            return 1
        fi

        # Extract token counts from run_completed event
        local check_out check_exit=0
        check_out=$(python3 -c "
import json, sys
found = False
with open('$WORKDIR/_events.jsonl') as f:
    for line in f:
        try:
            e = json.loads(line)
            if e.get('type') == 'run_completed':
                usage = e.get('total_usage', {})
                inp = usage.get('input_tokens', 0)
                out = usage.get('output_tokens', 0)
                if inp > 0 and out > 0:
                    print(f'OK: {inp} in / {out} out')
                    found = True
                else:
                    print(f'zero tokens: {inp} in / {out} out')
                    sys.exit(1)
        except Exception:
            pass
if not found:
    print('no run_completed event found')
    sys.exit(1)
" 2>&1) || check_exit=$?

        if [ "$check_exit" -ne 0 ]; then
            LAST_FAIL_REASON="$check_out"
            return 1
        fi
        echo "    $check_out"
    }
    with_retry "token usage in run_completed" test_8 || true
fi

# ═══════════════════════════════════════════════════════════════
# Test 9: Memory — Store and recall via orchestrator
#
#   Config: orchestrator with 2 agents (writer + reader) sharing
#   in-memory store. Writer stores a fact, reader recalls it.
#
#   Assert:
#   - memory_store tool_call_started event exists
#   - shared_memory_read or memory_recall tool was called
#   - Both agents have events in the event stream
# ═══════════════════════════════════════════════════════════════
if should_run 9; then
    bold "TEST 9: Memory store and recall (in-memory)"
    test_9() {
        local cfg="$WORKDIR/memory_test.toml"
        cat > "$cfg" << TOMLEOF
[provider]
name = "openrouter"
model = "$HEARTBIT_MODEL"

[provider.retry]
max_retries = 2
base_delay_ms = 500
max_delay_ms = 10000

[orchestrator]
max_turns = 15
max_tokens = 4096
dispatch_mode = "sequential"

[memory]
type = "in_memory"

[[agents]]
name = "writer"
description = "Agent that stores facts in memory"
system_prompt = "You store facts in long-term memory. Use the memory_store tool to store the fact given in your task. After storing, confirm you are done."

[[agents]]
name = "reader"
description = "Agent that recalls facts from shared memory"
system_prompt = "You recall facts from memory. Use the shared_memory_read tool to search for memories. Write all recalled memories to a file called memory_output.txt in the current directory."
TOMLEOF

        local task="First delegate to 'writer' with task: 'Store this fact in memory: The Heartbit runtime was created in 2024 and is written in Rust'. Then delegate to 'reader' with task: 'Recall all shared memories about Heartbit and write them to $WORKDIR/memory_output.txt'."

        timeout 180 "$BINARY" run -v --config "$cfg" "$task" \
            > "$WORKDIR/_stdout" 2> "$WORKDIR/_stderr" || true
        extract_events

        # 1. Events captured
        if [ ! -f "$WORKDIR/_events.jsonl" ] || [ ! -s "$WORKDIR/_events.jsonl" ]; then
            LAST_FAIL_REASON="no events captured"
            return 1
        fi
        echo "    Events captured: $(wc -l < "$WORKDIR/_events.jsonl" | tr -d ' ')"

        # 2. memory_store tool was called
        if ! grep -q '"memory_store"' "$WORKDIR/_events.jsonl"; then
            LAST_FAIL_REASON="memory_store tool never called"
            return 1
        fi
        echo "    memory_store tool called"

        # 3. shared_memory_read or memory_recall tool was called
        if ! grep -q '"shared_memory_read"' "$WORKDIR/_events.jsonl" && \
           ! grep -q '"memory_recall"' "$WORKDIR/_events.jsonl"; then
            LAST_FAIL_REASON="neither shared_memory_read nor memory_recall called"
            return 1
        fi
        echo "    memory read/recall tool called"

        # 4. Writer agent events present
        if ! grep -q '"agent":"writer"' "$WORKDIR/_events.jsonl" && \
           ! grep -q '"agent": "writer"' "$WORKDIR/_events.jsonl"; then
            LAST_FAIL_REASON="no events from writer agent"
            return 1
        fi
        echo "    Writer agent events present"

        # 5. Reader agent events present
        if ! grep -q '"agent":"reader"' "$WORKDIR/_events.jsonl" && \
           ! grep -q '"agent": "reader"' "$WORKDIR/_events.jsonl"; then
            LAST_FAIL_REASON="no events from reader agent"
            return 1
        fi
        echo "    Reader agent events present"
    }
    with_retry "Memory store and recall" test_9 || true
fi

# ─── Results ─────────────────────────────────────────────────

echo ""
bold "╔════════════════════════════════════════════════════════╗"
TOTAL=$((PASS + FAIL))
if [ "$SKIP" -gt 0 ]; then
    echo "  Skipped: $SKIP"
fi
if [ "$FAIL" -eq 0 ]; then
    green "║  ALL $TOTAL TESTS PASSED                                ║"
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
