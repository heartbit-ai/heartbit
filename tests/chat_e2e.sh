#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Smoke tests for `heartbit chat`
#
# These are NOT CI tests. They call a real LLM and cost money.
# Run manually after changes to the chat path to sanity-check.
#
# Design rules:
#   1. Assert on artifacts (files, exit codes, script output), never LLM prose.
#   2. Assertions must be LENIENT — allow the LLM to solve the task its way.
#   3. Each test retries up to $MAX_RETRIES times before failing.
#   4. Tests can be run individually: ./chat_e2e.sh 3   (runs only test 3)
#
# Requires: OPENROUTER_API_KEY, target/release/heartbit
# ──────────────────────────────────────────────────────────────
set -euo pipefail

BINARY="$(cd "$(dirname "$0")/.." && pwd)/target/release/heartbit"
WORKDIR="$(mktemp -d)"
MAX_RETRIES="${MAX_RETRIES:-2}"
PASS=0
FAIL=0
SKIP=0
ERRORS=""
FILTER="${1:-}"  # optional: run only this test number

cleanup() { rm -rf "$WORKDIR"; }
trap cleanup EXIT

red()   { printf '\033[1;31m%s\033[0m\n' "$*"; }
green() { printf '\033[1;32m%s\033[0m\n' "$*"; }
yellow(){ printf '\033[1;33m%s\033[0m\n' "$*"; }
bold()  { printf '\033[1m%s\033[0m\n' "$*"; }

pass() { PASS=$((PASS + 1)); green "  PASS: $1"; }
fail() { FAIL=$((FAIL + 1)); red "  FAIL: $1 — $2"; ERRORS+="  [$1] $2\n"; }
skip() { SKIP=$((SKIP + 1)); yellow "  SKIP: $1"; }

should_run() {
    [ -z "$FILTER" ] || [ "$FILTER" = "$1" ]
}

# Run chat, separate stdout/stderr, return exit code.
run_chat() {
    local input="$1" timeout_s="${2:-90}" exit_code=0
    printf '%s' "$input" | timeout "$timeout_s" "$BINARY" chat \
        > "$WORKDIR/_stdout" 2> "$WORKDIR/_stderr" || exit_code=$?
    return "$exit_code"
}

# Retry wrapper: run a function up to MAX_RETRIES+1 times.
# Usage: with_retry test_name test_function
with_retry() {
    local name="$1" fn="$2"
    local attempt=0
    while [ "$attempt" -le "$MAX_RETRIES" ]; do
        if [ "$attempt" -gt 0 ]; then
            yellow "  retry $attempt/$MAX_RETRIES..."
        fi
        # Clean per-attempt state
        rm -f "$WORKDIR/_stdout" "$WORKDIR/_stderr"
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
    red "Binary not found: $BINARY"; exit 1
fi

bold "Workdir: $WORKDIR"
bold "Retries: $MAX_RETRIES"
echo ""

# ═══════════════════════════════════════════════════════════════
# Test 1: Empty input exits cleanly
#   Deterministic — no LLM call, just tests the on_input path.
# ═══════════════════════════════════════════════════════════════
if should_run 1; then
    bold "TEST 1: empty input exits"
    test_1() {
        run_chat "
" 15
        local ec=$?
        if [ "$ec" -ne 0 ]; then
            LAST_FAIL_REASON="exit code $ec"
            return 1
        fi
    }
    with_retry "empty input exits" test_1 || true
fi

# ═══════════════════════════════════════════════════════════════
# Test 2: File creation
#   Agent creates a file. We check the file exists and contains
#   the required text. LENIENT: we don't demand exact content
#   because the LLM might add a trailing newline.
# ═══════════════════════════════════════════════════════════════
if should_run 2; then
    bold "TEST 2: file creation"
    test_2() {
        rm -f "$WORKDIR/hello.txt"
        run_chat "Create a file at $WORKDIR/hello.txt containing the text 'heartbit works'. Do not explain, just create the file.
" 60 || true  # don't fail on nonzero exit — check artifact

        if [ ! -f "$WORKDIR/hello.txt" ]; then
            LAST_FAIL_REASON="file not created"
            return 1
        fi
        if ! grep -qF "heartbit works" "$WORKDIR/hello.txt"; then
            LAST_FAIL_REASON="content: $(cat "$WORKDIR/hello.txt")"
            return 1
        fi
    }
    with_retry "file creation" test_2 || true
fi

# ═══════════════════════════════════════════════════════════════
# Test 3: Write code (artifact verification)
#   Agent writes a Python script. We run it ourselves.
#   LENIENT: we check that stdout contains each expected number
#   on its own line, but allow extra whitespace or labels.
# ═══════════════════════════════════════════════════════════════
if should_run 3; then
    bold "TEST 3: write code"
    test_3() {
        rm -f "$WORKDIR/squares.py"
        run_chat "Write a Python script at $WORKDIR/squares.py that prints the squares of 1 through 5, each on its own line: 1, 4, 9, 16, 25. No labels, just the numbers. Do not explain, just write the file.
" 60 || true

        if [ ! -f "$WORKDIR/squares.py" ]; then
            LAST_FAIL_REASON="file not created"
            return 1
        fi
        local py_out py_exit=0
        py_out=$(python3 "$WORKDIR/squares.py" 2>&1) || py_exit=$?
        if [ "$py_exit" -ne 0 ]; then
            LAST_FAIL_REASON="script crashed: $py_out"
            return 1
        fi
        # Check each square appears in output (grep -w = word boundary)
        for n in 1 4 9 16 25; do
            if ! echo "$py_out" | grep -qw "$n"; then
                LAST_FAIL_REASON="output missing $n: $py_out"
                return 1
            fi
        done
    }
    with_retry "write code" test_3 || true
fi

# ═══════════════════════════════════════════════════════════════
# Test 4: Bug fix (artifact verification)
#   Pre-broken script with a div-by-zero. Agent must fix it.
#   LENIENT: we check the script runs (exit 0) and output contains
#   expected values — but we don't demand exact line positions,
#   because the agent might refactor print statements.
# ═══════════════════════════════════════════════════════════════
if should_run 4; then
    bold "TEST 4: bug fix"
    test_4() {
        cat > "$WORKDIR/broken.py" << 'PYEOF'
import sys

def double(x):
    return x * 2

def safe_divide(a, b):
    return a / b

if __name__ == "__main__":
    print(double(21))
    print(safe_divide(10, 0))
    print(safe_divide(10, 2))
PYEOF
        run_chat "The script $WORKDIR/broken.py crashes with a ZeroDivisionError. Fix safe_divide to return None when b is 0. Do not ask questions, just fix it.
" 90 || true

        local py_out py_exit=0
        py_out=$(python3 "$WORKDIR/broken.py" 2>&1) || py_exit=$?
        if [ "$py_exit" -ne 0 ]; then
            LAST_FAIL_REASON="script still crashes (exit $py_exit): $py_out"
            return 1
        fi
        # double(21) must still work
        if ! echo "$py_out" | grep -qw "42"; then
            LAST_FAIL_REASON="double(21)=42 missing from: $py_out"
            return 1
        fi
        # safe_divide(10,0) should produce None
        if ! echo "$py_out" | grep -qi "None"; then
            LAST_FAIL_REASON="None missing from: $py_out"
            return 1
        fi
        # safe_divide(10,2) should produce 5.0 or 5
        if ! echo "$py_out" | grep -qwE "5\.0|^5$"; then
            LAST_FAIL_REASON="5.0 missing from: $py_out"
            return 1
        fi
    }
    with_retry "bug fix" test_4 || true
fi

# ═══════════════════════════════════════════════════════════════
# Test 5: Edit existing file (artifact verification)
#   Pre-create JSON. Agent edits 3 fields. We parse with Python
#   and check all fields — both changed and unchanged.
# ═══════════════════════════════════════════════════════════════
if should_run 5; then
    bold "TEST 5: edit file"
    test_5() {
        cat > "$WORKDIR/config.json" << 'JSONEOF'
{
    "name": "myapp",
    "version": "1.0.0",
    "debug": true,
    "port": 8080
}
JSONEOF
        run_chat "Edit $WORKDIR/config.json: change version to \"2.0.0\", set debug to false, change port to 9090. Do not change the name field. Do not ask questions.
" 60 || true

        if [ ! -f "$WORKDIR/config.json" ]; then
            LAST_FAIL_REASON="file deleted"
            return 1
        fi
        local check_out check_exit=0
        check_out=$(python3 -c "
import json, sys
try:
    d = json.load(open('$WORKDIR/config.json'))
except Exception as e:
    print(f'invalid JSON: {e}')
    sys.exit(1)
errors = []
if d.get('name') != 'myapp':      errors.append(f'name changed to {d.get(\"name\")!r}')
if d.get('version') != '2.0.0':   errors.append(f'version={d.get(\"version\")!r}')
if d.get('debug') is not False:   errors.append(f'debug={d.get(\"debug\")!r}')
if d.get('port') != 9090:         errors.append(f'port={d.get(\"port\")!r}')
if errors:
    print('; '.join(errors))
    sys.exit(1)
" 2>&1) || check_exit=$?
        if [ "$check_exit" -ne 0 ]; then
            LAST_FAIL_REASON="$check_out"
            return 1
        fi
    }
    with_retry "edit file" test_5 || true
fi

# ═══════════════════════════════════════════════════════════════
# Test 6: Multi-turn with file state
#   Turn 1: write file A. Turn 2: modify same file.
#   We verify the file reflects BOTH turns, proving context
#   carried across on_input boundary.
#   The second prompt does NOT repeat the path — the agent must
#   remember it from turn 1.
# ═══════════════════════════════════════════════════════════════
if should_run 6; then
    bold "TEST 6: multi-turn context"
    test_6() {
        rm -f "$WORKDIR/multi.txt"
        run_chat "Write the word 'alpha' to $WORKDIR/multi.txt. Nothing else. Do not explain.
Now append the word 'beta' on a new line to that file. Do not explain.
" 90 || true

        if [ ! -f "$WORKDIR/multi.txt" ]; then
            LAST_FAIL_REASON="file not created"
            return 1
        fi
        local content
        content=$(cat "$WORKDIR/multi.txt")
        if ! echo "$content" | grep -qF "alpha"; then
            LAST_FAIL_REASON="missing 'alpha' in: $content"
            return 1
        fi
        if ! echo "$content" | grep -qF "beta"; then
            LAST_FAIL_REASON="missing 'beta' in: $content"
            return 1
        fi
    }
    with_retry "multi-turn context" test_6 || true
fi

# ═══════════════════════════════════════════════════════════════
# Test 7: Bash + negative check
#   Agent creates files in a directory. We check:
#   - expected files exist (positive)
#   - a decoy file was NOT touched (negative)
# ═══════════════════════════════════════════════════════════════
if should_run 7; then
    bold "TEST 7: bash + negative check"
    test_7() {
        rm -rf "$WORKDIR/project"
        mkdir -p "$WORKDIR/project"
        echo "do not touch" > "$WORKDIR/project/decoy.txt"

        run_chat "Inside $WORKDIR/project, create a subdirectory called 'src' and inside it create a file called 'main.rs' with the content 'fn main() {}'. Do not modify any existing files. Do not explain.
" 60 || true

        if [ ! -f "$WORKDIR/project/src/main.rs" ]; then
            LAST_FAIL_REASON="src/main.rs not created"
            return 1
        fi
        if ! grep -qF "fn main()" "$WORKDIR/project/src/main.rs"; then
            LAST_FAIL_REASON="main.rs content: $(cat "$WORKDIR/project/src/main.rs")"
            return 1
        fi
        # Negative: decoy must be untouched
        local decoy
        decoy=$(cat "$WORKDIR/project/decoy.txt")
        if [ "$decoy" != "do not touch" ]; then
            LAST_FAIL_REASON="decoy.txt modified: $decoy"
            return 1
        fi
    }
    with_retry "bash + negative check" test_7 || true
fi

# ─── Results ─────────────────────────────────────────────────

echo ""
bold "════════════════════════════════"
TOTAL=$((PASS + FAIL))
if [ "$SKIP" -gt 0 ]; then
    echo "  Skipped: $SKIP"
fi
if [ "$FAIL" -eq 0 ]; then
    green "ALL $TOTAL TESTS PASSED"
else
    red "$FAIL/$TOTAL FAILED"
    echo ""
    red "Failure details:"
    printf "$ERRORS"
fi
bold "════════════════════════════════"

exit "$FAIL"
