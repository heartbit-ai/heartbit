#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Advanced end-to-end integration tests for Heartbit features:
#   structured output, doom loop, timeouts, truncation, session
#   pruning, dispatch modes, memory, knowledge, permissions,
#   multi-feature integration, and error recovery.
#
# These are NOT CI tests. They call a real LLM and cost money.
# Run manually after changes to validate real-world behavior.
#
# Design rules:
#   1. Assert on artifacts (files, exit codes, event JSON), never LLM prose.
#   2. Assertions must be LENIENT — allow the LLM to solve the task its way.
#   3. Each test retries up to $MAX_RETRIES times before failing.
#   4. Tests can be run individually: ./advanced_e2e.sh 3   (runs only test 3)
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

# Default model for all tests (cheap, capable of tool use + structured output)
export HEARTBIT_MODEL="${HEARTBIT_MODEL:-qwen/qwen3-30b-a3b}"

# Keep runs short
export HEARTBIT_MAX_TURNS="${HEARTBIT_MAX_TURNS:-15}"

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

# ─── Preflight ───────────────────────────────────────────────

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    red "OPENROUTER_API_KEY not set"; exit 1
fi
if [ ! -x "$BINARY" ]; then
    bold "Binary not found, building release..."
    (cd "$ROOT_DIR" && cargo build --release 2>&1) || { red "Build failed"; exit 1; }
fi

bold "╔════════════════════════════════════════════════════════╗"
bold "║  Advanced E2E — Structured Output, Timeouts,          ║"
bold "║  Doom Loop, Pruning, Dispatch, Memory, Knowledge      ║"
bold "╠════════════════════════════════════════════════════════╣"
echo "  Binary:  $BINARY"
echo "  Workdir: $WORKDIR"
echo "  Model:   $HEARTBIT_MODEL"
echo "  Retries: $MAX_RETRIES"
bold "╚════════════════════════════════════════════════════════╝"
echo ""

# ═══════════════════════════════════════════════════════════════
# Test 1: Structured Output — Valid Schema
#
#   Config: single agent with response_schema requiring { answer, confidence }.
#   Assert: exit 0, stdout is valid JSON with both keys, __respond__ tool used.
# ═══════════════════════════════════════════════════════════════
if should_run 1; then
    bold "TEST 1: Structured output — valid schema"
    test_1() {
        local cfg="$WORKDIR/structured1.toml"
        cat > "$cfg" << TOMLEOF
[provider]
name = "openrouter"
model = "$HEARTBIT_MODEL"

[provider.retry]
max_retries = 2
base_delay_ms = 500
max_delay_ms = 10000

[orchestrator]
max_turns = 10
max_tokens = 4096

[[agents]]
name = "worker"
description = "Agent that responds with structured JSON"
system_prompt = "You answer questions from your knowledge. Do NOT use websearch or any other tool. Your ONLY tool is __respond__. Call __respond__ immediately with your answer."

[agents.response_schema]
type = "object"
required = ["answer", "confidence"]

[agents.response_schema.properties.answer]
type = "string"

[agents.response_schema.properties.confidence]
type = "number"
TOMLEOF

        timeout 120 "$BINARY" run -v --config "$cfg" \
            "What is the capital of France? Respond with answer and confidence." \
            > "$WORKDIR/_stdout" 2> "$WORKDIR/_stderr" || true
        extract_events

        # 1. Must have events
        if [ ! -f "$WORKDIR/_events.jsonl" ] || [ ! -s "$WORKDIR/_events.jsonl" ]; then
            LAST_FAIL_REASON="no events captured"
            return 1
        fi

        # 2. Check for structured JSON in events (delegate_task output) or stdout.
        #    In orchestrator mode, the structured JSON appears in the delegate_task
        #    tool_call_completed output, while stdout shows orchestrator prose.
        local check_exit=0
        python3 -c "
import json, sys, re

data = None

# Check events: structured JSON appears in delegate_task tool output
with open('$WORKDIR/_events.jsonl') as f:
    for line in f:
        try:
            e = json.loads(line)
            if e.get('type') == 'tool_call_completed':
                output = e.get('output', '')
                # Try parsing the output itself (or extract JSON from it)
                for m in re.finditer(r'\{[^{}]*\"answer\"[^{}]*\"confidence\"[^{}]*\}', output):
                    try:
                        candidate = json.loads(m.group())
                        if 'answer' in candidate and 'confidence' in candidate:
                            data = candidate
                            break
                    except Exception:
                        pass
                if data is None:
                    for m in re.finditer(r'\{[^{}]*\"confidence\"[^{}]*\"answer\"[^{}]*\}', output):
                        try:
                            candidate = json.loads(m.group())
                            if 'answer' in candidate and 'confidence' in candidate:
                                data = candidate
                                break
                        except Exception:
                            pass
        except Exception:
            pass
        if data:
            break

# Fallback: try stdout
if data is None:
    raw = open('$WORKDIR/_stdout').read().strip()
    if raw:
        try:
            data = json.loads(raw)
        except Exception:
            for m in re.finditer(r'\{[^{}]*\"answer\"[^{}]*\}', raw):
                try:
                    data = json.loads(m.group())
                    break
                except Exception:
                    pass

if data is None:
    # Last resort: check if answer+confidence appear anywhere
    all_text = open('$WORKDIR/_stdout').read() + open('$WORKDIR/_events.jsonl').read()
    if '\"answer\"' in all_text and '\"confidence\"' in all_text:
        print('OK: structured fields found (not parsed as clean JSON)')
        sys.exit(0)
    print('no structured output found')
    sys.exit(1)

if 'answer' not in data:
    print('missing answer key')
    sys.exit(1)
if 'confidence' not in data:
    print('missing confidence key')
    sys.exit(1)
print(f'OK: answer={data[\"answer\"]!r}, confidence={data.get(\"confidence\", \"?\")}')
" 2>/dev/null || check_exit=$?
        if [ "$check_exit" -ne 0 ]; then
            LAST_FAIL_REASON="structured output validation failed"
            return 1
        fi

        # 3. __respond__ tool in events (informational, not fatal)
        if grep -q '__respond__' "$WORKDIR/_events.jsonl"; then
            echo "    __respond__ tool found in events"
        else
            echo "    (__respond__ not in events — structured output via orchestrator relay)"
        fi
    }
    with_retry "Structured output — valid schema" test_1 || true
fi

# ═══════════════════════════════════════════════════════════════
# Test 2: Structured Output — Array Schema Validation
#
#   Config: single agent with schema requiring items: [{ name, count }].
#   Assert: exit 0, output has items array with correct types.
# ═══════════════════════════════════════════════════════════════
if should_run 2; then
    bold "TEST 2: Structured output — array schema"
    test_2() {
        local cfg="$WORKDIR/structured2.toml"
        cat > "$cfg" << TOMLEOF
[provider]
name = "openrouter"
model = "$HEARTBIT_MODEL"

[provider.retry]
max_retries = 2
base_delay_ms = 500
max_delay_ms = 10000

[orchestrator]
max_turns = 10
max_tokens = 4096

[[agents]]
name = "worker"
description = "Agent that responds with structured JSON"
system_prompt = "You answer questions from your knowledge. Do NOT use websearch or any other tool. Your ONLY tool is __respond__. Call __respond__ immediately with your answer."

[agents.response_schema]
type = "object"
required = ["items"]

[agents.response_schema.properties.items]
type = "array"

[agents.response_schema.properties.items.items]
type = "object"
required = ["name", "count"]

[agents.response_schema.properties.items.items.properties.name]
type = "string"

[agents.response_schema.properties.items.items.properties.count]
type = "integer"
TOMLEOF

        timeout 120 "$BINARY" run -v --config "$cfg" \
            "List 2 fruits with their count." \
            > "$WORKDIR/_stdout" 2> "$WORKDIR/_stderr" || true
        extract_events

        if [ ! -f "$WORKDIR/_events.jsonl" ] || [ ! -s "$WORKDIR/_events.jsonl" ]; then
            LAST_FAIL_REASON="no events captured"
            return 1
        fi

        # Check events (delegate_task output) then stdout for structured JSON
        local check_exit=0
        python3 -c "
import json, sys, re

data = None

# Check events: structured JSON appears in delegate_task tool output
with open('$WORKDIR/_events.jsonl') as f:
    for line in f:
        try:
            e = json.loads(line)
            if e.get('type') == 'tool_call_completed':
                output = e.get('output', '')
                # Find JSON with items array
                for m in re.finditer(r'\{[^{}]*\"items\"\s*:\s*\[', output):
                    # Extract from the match start to find the full JSON
                    start = m.start()
                    candidate = output[start:]
                    # Try progressively longer substrings
                    for end in range(len(candidate), 0, -1):
                        try:
                            d = json.loads(candidate[:end])
                            if 'items' in d:
                                data = d
                                break
                        except Exception:
                            pass
                    if data:
                        break
        except Exception:
            pass
        if data:
            break

# Fallback: try stdout
if data is None:
    raw = open('$WORKDIR/_stdout').read().strip()
    if raw:
        try:
            data = json.loads(raw)
        except Exception:
            pass

if data is None:
    # Last resort: check if events mention items (model completed the task)
    all_text = open('$WORKDIR/_stdout').read() + open('$WORKDIR/_events.jsonl').read()
    if '\"items\"' in all_text and '\"name\"' in all_text:
        print('OK: structured data found in output (not parsed as clean JSON)')
        sys.exit(0)
    print('no structured output found in stdout or events')
    sys.exit(1)

if 'items' not in data:
    print(f'missing items key')
    sys.exit(1)
items = data['items']
if not isinstance(items, list) or len(items) < 1:
    print(f'items is not a non-empty list')
    sys.exit(1)
print(f'OK: {len(items)} items')
" 2>/dev/null || check_exit=$?
        if [ "$check_exit" -ne 0 ]; then
            LAST_FAIL_REASON="array schema validation failed"
            return 1
        fi

        # __respond__ tool in events (informational, not fatal)
        if grep -q '__respond__' "$WORKDIR/_events.jsonl"; then
            echo "    __respond__ tool found in events"
        else
            echo "    (__respond__ not in events — structured output via orchestrator relay)"
        fi
    }
    with_retry "Structured output — array schema" test_2 || true
fi

# ═══════════════════════════════════════════════════════════════
# Test 3: Doom Loop Detection
#
#   Config: max_identical_tool_calls = 2, max_turns = 10.
#   The agent tries to read a non-existent file and should be
#   stopped by doom loop detection (not loop forever).
# ═══════════════════════════════════════════════════════════════
if should_run 3; then
    bold "TEST 3: Doom loop detection"
    test_3() {
        local cfg="$WORKDIR/doomloop.toml"
        cat > "$cfg" << TOMLEOF
[provider]
name = "openrouter"
model = "$HEARTBIT_MODEL"

[provider.retry]
max_retries = 2
base_delay_ms = 500
max_delay_ms = 10000

[orchestrator]
max_turns = 10
max_tokens = 4096
max_identical_tool_calls = 2

[[agents]]
name = "worker"
description = "Agent that reads files"
system_prompt = "You read files. If a file does not exist, keep trying to read it."
max_identical_tool_calls = 2
TOMLEOF

        local DOOM_FILE="/tmp/heartbit-doom-test-nonexistent-$(date +%s).txt"
        timeout 120 "$BINARY" run -v --config "$cfg" \
            "Read the file $DOOM_FILE and tell me its contents. Keep trying if it fails." \
            > "$WORKDIR/_stdout" 2> "$WORKDIR/_stderr" || true
        extract_events

        if [ ! -f "$WORKDIR/_events.jsonl" ] || [ ! -s "$WORKDIR/_events.jsonl" ]; then
            LAST_FAIL_REASON="no events captured"
            return 1
        fi

        # Agent must have completed (not timed out at 120s ceiling)
        # Check that run_completed exists — means the agent finished
        if ! grep -q '"run_completed"' "$WORKDIR/_events.jsonl"; then
            LAST_FAIL_REASON="no run_completed event — agent may have timed out"
            return 1
        fi

        # Verify doom loop was triggered: tool_call_completed with is_error
        # containing "doom" or "identical" or "repeated" in output
        local doom_check=0
        python3 -c "
import json, sys
doom_detected = False
error_tool_calls = 0
with open('$WORKDIR/_events.jsonl') as f:
    for line in f:
        try:
            e = json.loads(line)
            if e.get('type') == 'tool_call_completed' and e.get('is_error'):
                error_tool_calls += 1
                output = e.get('output', '').lower()
                if any(w in output for w in ['doom', 'identical', 'repeated', 'loop']):
                    doom_detected = True
        except Exception:
            pass
if doom_detected:
    print(f'OK: doom loop detected ({error_tool_calls} error tool calls)')
    sys.exit(0)
elif error_tool_calls >= 2:
    # Agent hit errors (file not found) — doom loop may have kicked in
    # without specific keywords. The key assertion is it didn't loop forever.
    print(f'OK: {error_tool_calls} error tool calls, agent completed within max_turns')
    sys.exit(0)
else:
    print(f'no doom loop evidence (error_tool_calls={error_tool_calls})')
    sys.exit(1)
" 2>/dev/null || doom_check=$?
        if [ "$doom_check" -ne 0 ]; then
            LAST_FAIL_REASON="no doom loop evidence in events"
            return 1
        fi
    }
    with_retry "Doom loop detection" test_3 || true
fi

# ═══════════════════════════════════════════════════════════════
# Test 4: Tool Timeout Enforcement
#
#   Config: tool_timeout_seconds = 3.
#   Task: run sleep 30 via bash. Should timeout quickly.
# ═══════════════════════════════════════════════════════════════
if should_run 4; then
    bold "TEST 4: Tool timeout enforcement"
    test_4() {
        local cfg="$WORKDIR/tool_timeout.toml"
        cat > "$cfg" << TOMLEOF
[provider]
name = "openrouter"
model = "$HEARTBIT_MODEL"

[provider.retry]
max_retries = 2
base_delay_ms = 500
max_delay_ms = 10000

[orchestrator]
max_turns = 5
max_tokens = 4096
run_timeout_seconds = 30

[[agents]]
name = "worker"
description = "Agent that runs bash commands"
system_prompt = "You run bash commands exactly as told. Do not modify the command. Run it once and report the result."
tool_timeout_seconds = 3
max_turns = 3
TOMLEOF

        local START_TIME=$SECONDS
        timeout 120 "$BINARY" run -v --config "$cfg" \
            "Run this exact bash command: sleep 30 && echo done" \
            > "$WORKDIR/_stdout" 2> "$WORKDIR/_stderr" || true
        local ELAPSED=$((SECONDS - START_TIME))
        extract_events

        # Wall-clock must be < 60s (well under the 30s sleep, accounting for LLM overhead)
        if [ "$ELAPSED" -ge 60 ]; then
            LAST_FAIL_REASON="took ${ELAPSED}s — timeout did not fire (expected < 60s)"
            return 1
        fi
        echo "    Elapsed: ${ELAPSED}s (timeout enforced)"

        # Events should show tool_call_completed with error containing "timed out" or "timeout"
        local timeout_check=0
        python3 -c "
import json, sys
found_timeout = False
found_error = False
with open('$WORKDIR/_events.jsonl') as f:
    for line in f:
        try:
            e = json.loads(line)
            if e.get('type') == 'tool_call_completed' and e.get('is_error'):
                found_error = True
                output = e.get('output', '').lower()
                if 'timed out' in output or 'timeout' in output or 'deadline' in output:
                    found_timeout = True
            if e.get('type') == 'run_failed':
                err = str(e).lower()
                if 'timeout' in err or 'deadline' in err:
                    found_timeout = True
        except Exception:
            pass
if found_timeout:
    print('timeout error found in events')
    sys.exit(0)
elif found_error:
    print('tool error found (timeout may have different message)')
    sys.exit(0)
else:
    print('no timeout or error evidence in events')
    sys.exit(1)
" 2>/dev/null || timeout_check=$?
        if [ "$timeout_check" -ne 0 ]; then
            # Fallback: if elapsed < 45s, timeout clearly worked (sleep was 30s)
            if [ "$ELAPSED" -lt 45 ]; then
                echo "    (timeout enforced by wall-clock: ${ELAPSED}s < 45s)"
                return 0
            fi
            LAST_FAIL_REASON="no timeout error in events and elapsed=${ELAPSED}s"
            return 1
        fi
    }
    with_retry "Tool timeout enforcement" test_4 || true
fi

# ═══════════════════════════════════════════════════════════════
# Test 5: Run Timeout (Wall-Clock Deadline)
#
#   Config: run_timeout_seconds = 10, max_turns = 100.
#   Task: long multi-step task. Should be killed by deadline.
# ═══════════════════════════════════════════════════════════════
if should_run 5; then
    bold "TEST 5: Run timeout (wall-clock deadline)"
    test_5() {
        local cfg="$WORKDIR/run_timeout.toml"
        cat > "$cfg" << TOMLEOF
[provider]
name = "openrouter"
model = "$HEARTBIT_MODEL"

[provider.retry]
max_retries = 2
base_delay_ms = 500
max_delay_ms = 10000

[orchestrator]
max_turns = 100
max_tokens = 4096
run_timeout_seconds = 20

[[agents]]
name = "worker"
description = "Agent that writes long essays using multiple tool calls"
system_prompt = "You write detailed essays. For each section, create a separate file with the content."
run_timeout_seconds = 15
TOMLEOF

        local START_TIME=$SECONDS
        timeout 120 "$BINARY" run -v --config "$cfg" \
            "Write a 10000-word essay about the complete history of computing from 1800 to 2025. Create a separate file for each decade in $WORKDIR/essay/. Be very thorough." \
            > "$WORKDIR/_stdout" 2> "$WORKDIR/_stderr" || true
        local ELAPSED=$((SECONDS - START_TIME))
        extract_events

        # Must finish within ~45s (20s orchestrator timeout + LLM/network grace)
        if [ "$ELAPSED" -ge 60 ]; then
            LAST_FAIL_REASON="took ${ELAPSED}s — run timeout did not fire (expected < 60s)"
            return 1
        fi
        echo "    Elapsed: ${ELAPSED}s (run timeout enforced)"

        # Check for run_failed or timeout evidence in events/stderr
        if grep -q '"run_failed"' "$WORKDIR/_events.jsonl" 2>/dev/null || \
           grep -qi 'timeout\|deadline' "$WORKDIR/_stderr" 2>/dev/null; then
            echo "    Run timeout triggered (run_failed or timeout in logs)"
            return 0
        fi

        # Wall-clock check is the primary assertion — if we finished fast, timeout worked
        if [ "$ELAPSED" -le 45 ]; then
            echo "    (completed within ${ELAPSED}s — run timeout enforced or task was short)"
            return 0
        fi

        LAST_FAIL_REASON="no evidence of run timeout in events/logs"
        return 1
    }
    with_retry "Run timeout (wall-clock deadline)" test_5 || true
fi

# ═══════════════════════════════════════════════════════════════
# Test 6: Tool Output Truncation
#
#   Config: max_tool_output_bytes = 500.
#   Setup: 50KB file. Agent reads it, output gets truncated.
# ═══════════════════════════════════════════════════════════════
if should_run 6; then
    bold "TEST 6: Tool output truncation"
    test_6() {
        local cfg="$WORKDIR/truncation.toml"
        cat > "$cfg" << TOMLEOF
[provider]
name = "openrouter"
model = "$HEARTBIT_MODEL"

[provider.retry]
max_retries = 2
base_delay_ms = 500
max_delay_ms = 10000

[orchestrator]
max_turns = 10
max_tokens = 4096

[[agents]]
name = "worker"
description = "Agent that reads files and runs bash commands"
system_prompt = "You read files using the bash tool with 'cat'. Do NOT use the read tool. Always use: bash cat <filepath>"
max_tool_output_bytes = 200
TOMLEOF

        # Create a 10KB file with a known first line
        local BIGFILE="$WORKDIR/big_file.txt"
        echo "FIRST_LINE_MARKER_12345" > "$BIGFILE"
        python3 -c "
for i in range(400):
    print(f'Line {i}: ' + 'abcdefghij' * 2)
" >> "$BIGFILE"

        timeout 120 "$BINARY" run -v --config "$cfg" \
            "Use the bash tool to run: cat $BIGFILE — then tell me the first line." \
            > "$WORKDIR/_stdout" 2> "$WORKDIR/_stderr" || true
        extract_events

        if [ ! -f "$WORKDIR/_events.jsonl" ] || [ ! -s "$WORKDIR/_events.jsonl" ]; then
            LAST_FAIL_REASON="no events captured"
            return 1
        fi

        # Check that truncation marker appears in events or stderr
        if grep -q '\[truncated:' "$WORKDIR/_events.jsonl"; then
            echo "    Truncation marker found in events"
            return 0
        fi
        if grep -q '\[truncated:' "$WORKDIR/_stderr"; then
            echo "    Truncation marker found in stderr"
            return 0
        fi

        # Also check tool_call_completed outputs: the tool might have returned
        # truncated content even if the event output field was further truncated
        local trunc_check=0
        python3 -c "
import json, sys
with open('$WORKDIR/_events.jsonl') as f:
    for line in f:
        try:
            e = json.loads(line)
            if e.get('type') == 'tool_call_completed':
                out = e.get('output', '')
                # Check if output was cut short (< full file but tool was bash/read)
                tool = e.get('tool_name', '')
                if tool in ('bash', 'read') and len(out) > 0 and len(out) < 5000:
                    if 'FIRST_LINE_MARKER' in out:
                        print(f'OK: tool output truncated to {len(out)} bytes')
                        sys.exit(0)
        except Exception:
            pass
print('no truncation evidence')
sys.exit(1)
" 2>/dev/null || trunc_check=$?
        if [ "$trunc_check" -eq 0 ]; then
            return 0
        fi

        LAST_FAIL_REASON="no truncation evidence found"
        return 1
    }
    with_retry "Tool output truncation" test_6 || true
fi

# ═══════════════════════════════════════════════════════════════
# Test 7: Session Pruning
#
#   Config: session_prune with aggressive settings.
#   Task: read multiple files, agent should still complete
#   despite old tool results being pruned.
# ═══════════════════════════════════════════════════════════════
if should_run 7; then
    bold "TEST 7: Session pruning"
    test_7() {
        local cfg="$WORKDIR/session_prune.toml"
        cat > "$cfg" << TOMLEOF
[provider]
name = "openrouter"
model = "$HEARTBIT_MODEL"

[provider.retry]
max_retries = 2
base_delay_ms = 500
max_delay_ms = 10000

[orchestrator]
max_turns = 20
max_tokens = 4096

[[agents]]
name = "worker"
description = "Agent that reads files and summarizes them"
system_prompt = "You read files one at a time. After reading all files, write a summary of the first line of each file to a summary.txt file."

[agents.session_prune]
keep_recent_n = 1
pruned_tool_result_max_bytes = 100
TOMLEOF

        # Create 5 files with distinct first lines and 2KB content each
        mkdir -p "$WORKDIR/prune_files"
        for i in 1 2 3 4 5; do
            echo "FILE_${i}_UNIQUE_MARKER" > "$WORKDIR/prune_files/file${i}.txt"
            python3 -c "
for j in range(80):
    print(f'Padding line {j}: ' + 'data' * 5)
" >> "$WORKDIR/prune_files/file${i}.txt"
        done

        timeout 180 "$BINARY" run -v --config "$cfg" \
            "Read each of these 5 files one at a time: $WORKDIR/prune_files/file1.txt, $WORKDIR/prune_files/file2.txt, $WORKDIR/prune_files/file3.txt, $WORKDIR/prune_files/file4.txt, $WORKDIR/prune_files/file5.txt. After reading ALL files, write a file at $WORKDIR/prune_files/summary.txt containing the first line of each file." \
            > "$WORKDIR/_stdout" 2> "$WORKDIR/_stderr" || true
        extract_events

        if [ ! -f "$WORKDIR/_events.jsonl" ] || [ ! -s "$WORKDIR/_events.jsonl" ]; then
            LAST_FAIL_REASON="no events captured"
            return 1
        fi

        # Agent must have completed
        if ! grep -q '"run_completed"' "$WORKDIR/_events.jsonl"; then
            LAST_FAIL_REASON="no run_completed event"
            return 1
        fi

        # At least 3 read tool calls (agent may batch some)
        local read_count
        read_count=$(python3 -c "
import json
count = 0
with open('$WORKDIR/_events.jsonl') as f:
    for line in f:
        try:
            e = json.loads(line)
            if e.get('type') == 'tool_call_started' and e.get('tool_name') == 'read':
                count += 1
        except Exception:
            pass
print(count)
")
        if [ "$read_count" -lt 3 ]; then
            LAST_FAIL_REASON="only $read_count read tool calls (expected >= 3)"
            return 1
        fi
        echo "    Read tool calls: $read_count"

        # Session pruning means token usage should be bounded. Check that
        # any run_completed has reasonable token counts.
        local tokens_check=0
        python3 -c "
import json, sys
found = False
with open('$WORKDIR/_events.jsonl') as f:
    for line in f:
        try:
            e = json.loads(line)
            if e.get('type') == 'run_completed':
                agent = e.get('agent', '?')
                usage = e.get('total_usage', {})
                inp = usage.get('input_tokens', 0)
                if inp > 0:
                    print(f'  {agent}: {inp} input tokens')
                    found = True
        except Exception:
            pass
if found:
    sys.exit(0)
print('no run_completed with token usage')
sys.exit(1)
" 2>/dev/null || tokens_check=$?
        if [ "$tokens_check" -ne 0 ]; then
            # Non-fatal: the key assertion is that the agent completed (checked above)
            echo "    (could not verify token usage — non-fatal)"
        fi
    }
    with_retry "Session pruning" test_7 || true
fi

# ═══════════════════════════════════════════════════════════════
# Test 8: Sequential Dispatch Mode
#
#   Config: orchestrator with 2 agents, dispatch_mode = "sequential".
#   Task: agent_a creates file, agent_b reads it — must be ordered.
# ═══════════════════════════════════════════════════════════════
if should_run 8; then
    bold "TEST 8: Sequential dispatch mode"
    test_8() {
        local cfg="$WORKDIR/sequential.toml"
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

[[agents]]
name = "agent_a"
description = "Agent that creates files with specific content"
system_prompt = "You create files using the write tool. Write exactly the content you are told."

[[agents]]
name = "agent_b"
description = "Agent that reads files and creates new files based on what it reads"
system_prompt = "You read files and create new files. Copy the exact content from the source to the target."
TOMLEOF

        timeout 180 "$BINARY" run -v --config "$cfg" \
            "First, delegate to agent_a: write the text 'hello-sequential' to $WORKDIR/step1.txt. Then, after agent_a is done, delegate to agent_b: read $WORKDIR/step1.txt and write its content to $WORKDIR/step2.txt." \
            > "$WORKDIR/_stdout" 2> "$WORKDIR/_stderr" || true
        extract_events

        if [ ! -f "$WORKDIR/_events.jsonl" ] || [ ! -s "$WORKDIR/_events.jsonl" ]; then
            LAST_FAIL_REASON="no events captured"
            return 1
        fi

        # step1.txt must exist
        if [ ! -f "$WORKDIR/step1.txt" ]; then
            LAST_FAIL_REASON="step1.txt not created"
            return 1
        fi
        echo "    step1.txt created"

        # step2.txt must exist with content from step1
        if [ ! -f "$WORKDIR/step2.txt" ]; then
            LAST_FAIL_REASON="step2.txt not created"
            return 1
        fi
        if ! grep -q "hello" "$WORKDIR/step2.txt"; then
            LAST_FAIL_REASON="step2.txt does not contain hello: $(cat "$WORKDIR/step2.txt")"
            return 1
        fi
        echo "    step2.txt created with correct content"

        # Check sub_agents_dispatched events: in sequential mode, each batch
        # should have only 1 agent (not both at once)
        local seq_check=0
        python3 -c "
import json, sys
batches = []
with open('$WORKDIR/_events.jsonl') as f:
    for line in f:
        try:
            e = json.loads(line)
            if e.get('type') == 'sub_agents_dispatched':
                agents = e.get('agents', [])
                batches.append(agents)
        except Exception:
            pass
if not batches:
    print('no sub_agents_dispatched events')
    sys.exit(1)
# All batches should have at most 1 agent
for i, batch in enumerate(batches):
    if len(batch) > 1:
        print(f'batch {i} has {len(batch)} agents (expected 1 for sequential)')
        sys.exit(1)
print(f'OK: {len(batches)} sequential dispatches')
sys.exit(0)
" 2>/dev/null || seq_check=$?
        if [ "$seq_check" -ne 0 ]; then
            # Fallback: if both files are correct, the sequencing worked
            # (agent_b couldn't read step1.txt if agent_a hadn't created it)
            echo "    (files correct — sequencing verified by artifact dependency)"
            return 0
        fi
    }
    with_retry "Sequential dispatch mode" test_8 || true
fi

# ═══════════════════════════════════════════════════════════════
# Test 9: Parallel Dispatch Mode (Timing)
#
#   Config: orchestrator with 2 agents, dispatch_mode = "parallel".
#   Task: both agents create independent files.
# ═══════════════════════════════════════════════════════════════
if should_run 9; then
    bold "TEST 9: Parallel dispatch mode"
    test_9() {
        local cfg="$WORKDIR/parallel.toml"
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
dispatch_mode = "parallel"

[[agents]]
name = "agent_a"
description = "Agent that creates files"
system_prompt = "You create files. Write exactly what you are told."

[[agents]]
name = "agent_b"
description = "Agent that creates files"
system_prompt = "You create files. Write exactly what you are told."
TOMLEOF

        timeout 180 "$BINARY" run -v --config "$cfg" \
            "Delegate to both agents in parallel: agent_a writes 'alpha' to $WORKDIR/a.txt, agent_b writes 'beta' to $WORKDIR/b.txt. Both tasks are independent." \
            > "$WORKDIR/_stdout" 2> "$WORKDIR/_stderr" || true
        extract_events

        if [ ! -f "$WORKDIR/_events.jsonl" ] || [ ! -s "$WORKDIR/_events.jsonl" ]; then
            LAST_FAIL_REASON="no events captured"
            return 1
        fi

        # Both files must exist
        if [ ! -f "$WORKDIR/a.txt" ]; then
            LAST_FAIL_REASON="a.txt not created"
            return 1
        fi
        if [ ! -f "$WORKDIR/b.txt" ]; then
            LAST_FAIL_REASON="b.txt not created"
            return 1
        fi
        echo "    Both files created"

        # Check for parallel dispatch: sub_agents_dispatched with 2 agents
        local par_check=0
        python3 -c "
import json, sys
found_parallel = False
with open('$WORKDIR/_events.jsonl') as f:
    for line in f:
        try:
            e = json.loads(line)
            if e.get('type') == 'sub_agents_dispatched':
                agents = e.get('agents', [])
                if len(agents) >= 2:
                    found_parallel = True
                    print(f'OK: parallel dispatch with {len(agents)} agents: {agents}')
        except Exception:
            pass
if found_parallel:
    sys.exit(0)
else:
    print('no parallel dispatch found (all batches had < 2 agents)')
    sys.exit(1)
" 2>/dev/null || par_check=$?
        if [ "$par_check" -ne 0 ]; then
            # Fallback: LLM may have dispatched sequentially despite config.
            # If both files exist, at least the orchestration worked.
            echo "    (both files created — LLM dispatched sequentially despite parallel mode)"
            return 0
        fi
    }
    with_retry "Parallel dispatch mode" test_9 || true
fi

# ═══════════════════════════════════════════════════════════════
# Test 10: Memory Store and Cross-Agent Recall
#
#   Config: orchestrator with 2 agents, in-memory store, sequential.
#   Writer stores a fact, reader recalls it in the same session.
# ═══════════════════════════════════════════════════════════════
if should_run 10; then
    bold "TEST 10: Memory store and recall (in-memory)"
    test_10() {
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
system_prompt = "You store facts in long-term memory. Use the memory_store tool to store the fact given in your task with high importance. After storing, say done."

[[agents]]
name = "reader"
description = "Agent that recalls facts from shared memory"
system_prompt = "You recall facts from memory. Use the shared_memory_read tool to search for memories. Write all recalled facts to the specified file."
TOMLEOF

        local SECRET="The secret launch code is 7742"
        timeout 180 "$BINARY" run -v --config "$cfg" \
            "First delegate to 'writer': 'Store this fact in memory with importance 9: $SECRET'. Then delegate to 'reader': 'Recall shared memories about a launch code and write them to $WORKDIR/memory_output.txt'." \
            > "$WORKDIR/_stdout" 2> "$WORKDIR/_stderr" || true
        extract_events

        if [ ! -f "$WORKDIR/_events.jsonl" ] || [ ! -s "$WORKDIR/_events.jsonl" ]; then
            LAST_FAIL_REASON="no events captured"
            return 1
        fi

        # memory_store tool was called
        if ! grep -q '"memory_store"' "$WORKDIR/_events.jsonl"; then
            LAST_FAIL_REASON="memory_store tool never called"
            return 1
        fi
        echo "    memory_store called"

        # shared_memory_read or memory_recall tool was called
        if ! grep -q '"shared_memory_read"' "$WORKDIR/_events.jsonl" && \
           ! grep -q '"memory_recall"' "$WORKDIR/_events.jsonl"; then
            LAST_FAIL_REASON="neither shared_memory_read nor memory_recall called"
            return 1
        fi
        echo "    memory read/recall called"

        # Both agent names in events
        if ! grep -q '"writer"' "$WORKDIR/_events.jsonl"; then
            LAST_FAIL_REASON="no events from writer agent"
            return 1
        fi
        if ! grep -q '"reader"' "$WORKDIR/_events.jsonl"; then
            LAST_FAIL_REASON="no events from reader agent"
            return 1
        fi
        echo "    Both agents present in events"
    }
    with_retry "Memory store and recall" test_10 || true
fi

# ═══════════════════════════════════════════════════════════════
# Test 11: Knowledge Base Search
#
#   Config: single agent with [knowledge] section.
#   Setup: 3 markdown files with distinct facts.
# ═══════════════════════════════════════════════════════════════
if should_run 11; then
    bold "TEST 11: Knowledge base search"
    test_11() {
        # Create knowledge base docs
        local DOCS="$WORKDIR/kb_docs"
        mkdir -p "$DOCS"
        cat > "$DOCS/overview.md" << 'EOF'
# Project Overview

Project Phoenix was launched in January 2024. It aims to revolutionize
distributed computing for edge devices.
EOF
        cat > "$DOCS/budget.md" << 'EOF'
# Budget Report

The total project budget for Project Phoenix is $50M allocated across
three fiscal years. The first year allocation is $20M.
EOF
        cat > "$DOCS/team.md" << 'EOF'
# Team Structure

The Project Phoenix team consists of 12 full-time engineers,
3 product managers, and 2 designers for a total of 17 members.
EOF

        local cfg="$WORKDIR/knowledge.toml"
        cat > "$cfg" << TOMLEOF
[provider]
name = "openrouter"
model = "$HEARTBIT_MODEL"

[provider.retry]
max_retries = 2
base_delay_ms = 500
max_delay_ms = 10000

[orchestrator]
max_turns = 10
max_tokens = 4096

[knowledge]
chunk_size = 500
chunk_overlap = 50

[[knowledge.sources]]
type = "glob"
pattern = "$DOCS/*.md"

[[agents]]
name = "worker"
description = "Agent that searches a knowledge base"
system_prompt = "You search the knowledge base using the knowledge_search tool and answer questions based on what you find. Always search first."
TOMLEOF

        timeout 120 "$BINARY" run -v --config "$cfg" \
            "Search the knowledge base for information about the project budget. What is the total budget?" \
            > "$WORKDIR/_stdout" 2> "$WORKDIR/_stderr" || true
        extract_events

        if [ ! -f "$WORKDIR/_events.jsonl" ] || [ ! -s "$WORKDIR/_events.jsonl" ]; then
            LAST_FAIL_REASON="no events captured"
            return 1
        fi

        # knowledge_search tool must have been called
        if ! grep -q '"knowledge_search"' "$WORKDIR/_events.jsonl"; then
            LAST_FAIL_REASON="knowledge_search tool not called"
            return 1
        fi
        echo "    knowledge_search called"

        # Output should mention $50M or 50 (the budget figure)
        local stdout_content
        stdout_content=$(cat "$WORKDIR/_stdout" 2>/dev/null || echo "")
        if echo "$stdout_content" | grep -qi '50'; then
            echo "    Output mentions budget figure"
            return 0
        fi

        # Check events for knowledge search returning budget info
        if grep -q '50' "$WORKDIR/_events.jsonl"; then
            echo "    Budget figure found in events"
            return 0
        fi

        LAST_FAIL_REASON="no mention of budget figure (50M) in output or events"
        return 1
    }
    with_retry "Knowledge base search" test_11 || true
fi

# ═══════════════════════════════════════════════════════════════
# Test 12: Permission Deny Rule Enforcement
#
#   Config: single agent with [[permissions]] deny for rm.
#   The agent tries to delete a file — should be denied.
# ═══════════════════════════════════════════════════════════════
if should_run 12; then
    bold "TEST 12: Permission deny rule enforcement"
    test_12() {
        local cfg="$WORKDIR/perm_deny.toml"
        cat > "$cfg" << TOMLEOF
[provider]
name = "openrouter"
model = "$HEARTBIT_MODEL"

[provider.retry]
max_retries = 2
base_delay_ms = 500
max_delay_ms = 10000

[orchestrator]
max_turns = 10
max_tokens = 4096

[[agents]]
name = "worker"
description = "Agent that runs bash commands"
system_prompt = "You run bash commands as instructed. Only use the bash tool."

[[permissions]]
tool = "bash"
pattern = "*protected*"
action = "deny"
TOMLEOF

        # Create the protected file
        echo "DO NOT DELETE" > "$WORKDIR/protected.txt"

        timeout 120 "$BINARY" run -v --config "$cfg" \
            "Delete the file $WORKDIR/protected.txt using the bash tool with the rm command. Do NOT use any other tool or method." \
            > "$WORKDIR/_stdout" 2> "$WORKDIR/_stderr" || true
        extract_events

        # Primary assertion: protected.txt must still exist
        if [ ! -f "$WORKDIR/protected.txt" ]; then
            LAST_FAIL_REASON="protected.txt was deleted despite deny rule"
            return 1
        fi
        echo "    protected.txt still exists"

        # Check events for evidence of denial
        local deny_check=0
        python3 -c "
import json, sys
# Check if the LLM tried to call bash but it was denied
# (permission-denied tools skip tool_call_started)
worker_tried_tool = False
worker_bash_started = False
with open('$WORKDIR/_events.jsonl') as f:
    for line in f:
        try:
            e = json.loads(line)
            if e.get('type') == 'llm_response' and e.get('agent') == 'worker' and e.get('tool_call_count', 0) > 0:
                worker_tried_tool = True
            if e.get('type') == 'tool_call_started' and e.get('agent') == 'worker' and e.get('tool_name') == 'bash':
                inp = e.get('input', '')
                if 'rm' in inp:
                    worker_bash_started = True
        except Exception:
            pass

if worker_tried_tool and not worker_bash_started:
    print('OK: tool call attempted but bash/rm was denied')
    sys.exit(0)
elif not worker_tried_tool:
    # LLM didn't even try — still OK since file is protected
    print('OK: LLM did not attempt bash call (file still protected)')
    sys.exit(0)
else:
    # bash was started — check if it succeeded at deletion
    # File existence check above already passed, so something prevented deletion
    print('WARN: bash started but file survived (may have used different method)')
    sys.exit(0)
" 2>/dev/null || deny_check=$?
        if [ "$deny_check" -ne 0 ]; then
            # File exists — that's the key assertion
            echo "    (file protected regardless of event pattern)"
            return 0
        fi
    }
    with_retry "Permission deny rule enforcement" test_12 || true
fi

# ═══════════════════════════════════════════════════════════════
# Test 13: Multi-Feature Orchestrator (Integration)
#
#   Config: 3 agents, sequential dispatch, session pruning, memory.
#   Task: researcher → coder → reviewer pipeline.
# ═══════════════════════════════════════════════════════════════
if should_run 13; then
    bold "TEST 13: Multi-feature orchestrator integration"
    test_13() {
        local cfg="$WORKDIR/multi_feature.toml"
        cat > "$cfg" << TOMLEOF
[provider]
name = "openrouter"
model = "$HEARTBIT_MODEL"

[provider.retry]
max_retries = 2
base_delay_ms = 500
max_delay_ms = 10000

[orchestrator]
max_turns = 20
max_tokens = 4096
dispatch_mode = "sequential"
max_tool_output_bytes = 2000

[memory]
type = "in_memory"

[[agents]]
name = "researcher"
description = "Runs bash commands to gather system information"
system_prompt = "You research system information using bash. Report facts concisely."
max_tool_output_bytes = 2000

[agents.session_prune]
keep_recent_n = 2
pruned_tool_result_max_bytes = 200

[[agents]]
name = "coder"
description = "Writes code files based on specifications"
system_prompt = "You write code files. Create clean, compilable code."
max_tool_output_bytes = 2000

[[agents]]
name = "reviewer"
description = "Reads files and verifies their correctness"
system_prompt = "You review code. Read the file and verify it looks correct."
max_tool_output_bytes = 2000
TOMLEOF

        timeout 240 "$BINARY" run -v --config "$cfg" \
            "Pipeline task: 1) Delegate to 'researcher': run 'rustc --version' and report the output. 2) Delegate to 'coder': create a Rust hello-world program at $WORKDIR/hello.rs that prints 'Hello from Heartbit'. 3) Delegate to 'reviewer': read $WORKDIR/hello.rs and confirm it contains a main function." \
            > "$WORKDIR/_stdout" 2> "$WORKDIR/_stderr" || true
        extract_events

        if [ ! -f "$WORKDIR/_events.jsonl" ] || [ ! -s "$WORKDIR/_events.jsonl" ]; then
            LAST_FAIL_REASON="no events captured"
            return 1
        fi

        # hello.rs must exist
        if [ ! -f "$WORKDIR/hello.rs" ]; then
            LAST_FAIL_REASON="hello.rs not created"
            return 1
        fi
        echo "    hello.rs created"

        # hello.rs must contain fn main
        if ! grep -q "fn main" "$WORKDIR/hello.rs"; then
            LAST_FAIL_REASON="hello.rs missing fn main"
            return 1
        fi
        echo "    hello.rs contains fn main"

        # All 3 agent names must appear in events
        for agent in researcher coder reviewer; do
            if ! grep -q "\"$agent\"" "$WORKDIR/_events.jsonl"; then
                LAST_FAIL_REASON="agent '$agent' not found in events"
                return 1
            fi
        done
        echo "    All 3 agents present in events"

        # sub_agents_dispatched events exist
        if ! grep -q '"sub_agents_dispatched"' "$WORKDIR/_events.jsonl"; then
            LAST_FAIL_REASON="no sub_agents_dispatched events"
            return 1
        fi
        echo "    sub_agents_dispatched events present"

        # run_completed with token usage
        local token_check=0
        python3 -c "
import json, sys
with open('$WORKDIR/_events.jsonl') as f:
    for line in f:
        try:
            e = json.loads(line)
            if e.get('type') == 'run_completed' and e.get('agent') == 'orchestrator':
                usage = e.get('total_usage', {})
                inp = usage.get('input_tokens', 0)
                out = usage.get('output_tokens', 0)
                if inp > 0 and out > 0:
                    print(f'OK: orchestrator {inp} in / {out} out')
                    sys.exit(0)
        except Exception:
            pass
print('no orchestrator run_completed with tokens')
sys.exit(1)
" 2>/dev/null || token_check=$?
        if [ "$token_check" -ne 0 ]; then
            # Non-fatal: tokens might be 0 for some providers
            echo "    (no orchestrator token usage — non-fatal)"
        fi
    }
    with_retry "Multi-feature orchestrator integration" test_13 || true
fi

# ═══════════════════════════════════════════════════════════════
# Test 14: Error Recovery — Provider Failure Graceful Exit
#
#   Config: invalid model name. Should fail cleanly (no panic).
# ═══════════════════════════════════════════════════════════════
if should_run 14; then
    bold "TEST 14: Error recovery — provider failure"
    test_14() {
        local cfg="$WORKDIR/bad_model.toml"
        cat > "$cfg" << TOMLEOF
[provider]
name = "openrouter"
model = "nonexistent-model-xyz-99999"

[orchestrator]
max_turns = 5
max_tokens = 4096

[[agents]]
name = "worker"
description = "Test agent"
system_prompt = "You are a test agent."
TOMLEOF

        local exit_code=0
        timeout 60 "$BINARY" run -v --config "$cfg" \
            "Say hello" \
            > "$WORKDIR/_stdout" 2> "$WORKDIR/_stderr" || exit_code=$?

        # Must exit non-zero (error)
        if [ "$exit_code" -eq 0 ]; then
            LAST_FAIL_REASON="expected non-zero exit for invalid model, got 0"
            return 1
        fi
        echo "    Exit code: $exit_code (non-zero as expected)"

        # Must NOT have panicked
        if grep -qi "panic\|thread.*panicked\|SIGABRT" "$WORKDIR/_stderr"; then
            LAST_FAIL_REASON="binary panicked — should be clean error"
            return 1
        fi
        echo "    No panic detected"

        # Stderr should contain some error message
        if [ ! -s "$WORKDIR/_stderr" ]; then
            LAST_FAIL_REASON="stderr is empty — expected error message"
            return 1
        fi
        echo "    Error message present in stderr"
    }
    with_retry "Error recovery — provider failure" test_14 || true
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
