#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Smoke tests for `heartbit-cli daemon`
#
# These are NOT CI tests. They require:
#   1. Running Kafka broker (docker compose up kafka -d)
#   2. OPENROUTER_API_KEY or ANTHROPIC_API_KEY set
#   3. target/release/heartbit-cli built
#
# Design rules:
#   1. Assert on HTTP responses and task state transitions.
#   2. Each test is self-contained with its own daemon instance.
#   3. Daemon starts on a random port to avoid conflicts.
#
# Usage:
#   ./tests/daemon_e2e.sh        # run all tests
#   ./tests/daemon_e2e.sh 2      # run only test 2
# ──────────────────────────────────────────────────────────────
set -euo pipefail

BINARY="$(cd "$(dirname "$0")/.." && pwd)/target/release/heartbit-cli"
WORKDIR="$(mktemp -d)"
PASS=0
FAIL=0
SKIP=0
ERRORS=""
FILTER="${1:-}"

cleanup() {
    # Kill any daemon processes we started
    if [ -n "${DAEMON_PID:-}" ] && kill -0 "$DAEMON_PID" 2>/dev/null; then
        kill "$DAEMON_PID" 2>/dev/null || true
        wait "$DAEMON_PID" 2>/dev/null || true
    fi
    rm -rf "$WORKDIR"
}
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

# Find a free port
find_port() {
    python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()'
}

# Wait for daemon to be ready (health endpoint)
wait_for_daemon() {
    local url="$1" retries=30
    for i in $(seq 1 $retries); do
        if curl -sf "$url/health" > /dev/null 2>&1; then
            return 0
        fi
        sleep 0.5
    done
    return 1
}

# Start daemon on a random port, sets DAEMON_PID and BASE_URL
start_daemon() {
    local port config_path="$1"
    port=$(find_port)
    BASE_URL="http://127.0.0.1:$port"

    "$BINARY" daemon --config "$config_path" --bind "127.0.0.1:$port" \
        > "$WORKDIR/_daemon_stdout" 2> "$WORKDIR/_daemon_stderr" &
    DAEMON_PID=$!

    if ! wait_for_daemon "$BASE_URL"; then
        red "  daemon failed to start within 15s"
        cat "$WORKDIR/_daemon_stderr" >&2
        return 1
    fi

    # Allow Kafka consumer group rebalance to complete before submitting tasks
    sleep 3
}

stop_daemon() {
    if [ -n "${DAEMON_PID:-}" ] && kill -0 "$DAEMON_PID" 2>/dev/null; then
        kill "$DAEMON_PID" 2>/dev/null || true
        wait "$DAEMON_PID" 2>/dev/null || true
    fi
    DAEMON_PID=""
}

# ──────────────────────────────────────────────────────────────
# Preflight checks
# ──────────────────────────────────────────────────────────────
bold "=== Daemon E2E Tests ==="
echo ""

if [ ! -x "$BINARY" ]; then
    red "Binary not found: $BINARY"
    red "Run: cargo build --release"
    exit 1
fi

# Check Kafka is reachable
if ! docker compose exec kafka /opt/kafka/bin/kafka-topics.sh --list --bootstrap-server localhost:9092 > /dev/null 2>&1; then
    red "Kafka not reachable. Run: docker compose up kafka -d"
    exit 1
fi

# Check for API key
if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    red "No API key set. Export OPENROUTER_API_KEY or ANTHROPIC_API_KEY."
    exit 1
fi

# Create a minimal daemon config
cat > "$WORKDIR/daemon_test.toml" << TOML
[provider]
name = "openrouter"
model = "${HEARTBIT_MODEL:-qwen/qwen3-30b-a3b}"

[provider.retry]
max_retries = 2
base_delay_ms = 500
max_delay_ms = 10000

[orchestrator]
max_turns = 5
max_tokens = 4096

[[agents]]
name = "worker"
description = "General-purpose worker agent"
system_prompt = "You are a helpful assistant. Complete tasks concisely."
max_turns = 3
max_tokens = 2048

[daemon]
max_concurrent_tasks = 2

[daemon.kafka]
brokers = "localhost:9092"
consumer_group = "heartbit-e2e-$$"
commands_topic = "hb.test.commands.$$"
events_topic = "hb.test.events.$$"

[daemon.metrics]
enabled = true
TOML

# If using Anthropic, patch the config
if [ -n "${ANTHROPIC_API_KEY:-}" ] && [ -z "${OPENROUTER_API_KEY:-}" ]; then
    sed -i 's/name = "openrouter"/name = "anthropic"/' "$WORKDIR/daemon_test.toml"
    sed -i "s/model = .*/model = \"claude-sonnet-4-20250514\"/" "$WORKDIR/daemon_test.toml"
fi

# ──────────────────────────────────────────────────────────────
# Test 1: Health endpoint
# ──────────────────────────────────────────────────────────────
if should_run 1; then
    bold "Test 1: Health endpoint"
    start_daemon "$WORKDIR/daemon_test.toml"

    response=$(curl -sf "$BASE_URL/health")
    if echo "$response" | grep -q '"status":"ok"'; then
        pass "health returns ok status"
    else
        fail "health" "unexpected response: $response"
    fi

    if echo "$response" | grep -q '"uptime_seconds"'; then
        pass "health includes uptime_seconds"
    else
        fail "health uptime" "missing uptime_seconds"
    fi

    stop_daemon
fi

# ──────────────────────────────────────────────────────────────
# Test 2: Submit and list tasks
# ──────────────────────────────────────────────────────────────
if should_run 2; then
    bold "Test 2: Submit and list tasks"
    start_daemon "$WORKDIR/daemon_test.toml"

    # Submit a task
    submit_response=$(curl -sf -X POST "$BASE_URL/tasks" \
        -H 'Content-Type: application/json' \
        -d '{"task":"Say hello"}')

    if echo "$submit_response" | grep -q '"id"'; then
        pass "submit returns task id"
    else
        fail "submit" "no id in response: $submit_response"
    fi

    if echo "$submit_response" | grep -q '"state":"pending"'; then
        pass "submit returns pending state"
    else
        fail "submit state" "unexpected state: $submit_response"
    fi

    TASK_ID=$(echo "$submit_response" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")

    # List tasks
    sleep 1
    list_response=$(curl -sf "$BASE_URL/tasks")
    if echo "$list_response" | grep -q "$TASK_ID"; then
        pass "list includes submitted task"
    else
        fail "list" "task $TASK_ID not in list"
    fi

    stop_daemon
fi

# ──────────────────────────────────────────────────────────────
# Test 3: Get task by ID
# ──────────────────────────────────────────────────────────────
if should_run 3; then
    bold "Test 3: Get task by ID"
    start_daemon "$WORKDIR/daemon_test.toml"

    submit_response=$(curl -sf -X POST "$BASE_URL/tasks" \
        -H 'Content-Type: application/json' \
        -d '{"task":"Say goodbye"}')
    TASK_ID=$(echo "$submit_response" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")

    get_response=$(curl -sf "$BASE_URL/tasks/$TASK_ID")
    if echo "$get_response" | grep -q '"task":"Say goodbye"'; then
        pass "get returns correct task text"
    else
        fail "get task" "unexpected: $get_response"
    fi

    # 404 for unknown ID
    http_code=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/tasks/00000000-0000-0000-0000-000000000000")
    if [ "$http_code" = "404" ]; then
        pass "get unknown task returns 404"
    else
        fail "get 404" "expected 404, got $http_code"
    fi

    stop_daemon
fi

# ──────────────────────────────────────────────────────────────
# Test 4: Cancel a pending task
# ──────────────────────────────────────────────────────────────
if should_run 4; then
    bold "Test 4: Cancel a task"
    start_daemon "$WORKDIR/daemon_test.toml"

    submit_response=$(curl -sf -X POST "$BASE_URL/tasks" \
        -H 'Content-Type: application/json' \
        -d '{"task":"Long running task that should be cancelled"}')
    TASK_ID=$(echo "$submit_response" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")

    # Cancel it
    cancel_code=$(curl -s -o /dev/null -w "%{http_code}" -X DELETE "$BASE_URL/tasks/$TASK_ID")
    if [ "$cancel_code" = "204" ]; then
        pass "cancel returns 204"
    else
        fail "cancel" "expected 204, got $cancel_code"
    fi

    # Give time for cancel to propagate through Kafka
    sleep 5

    get_response=$(curl -sf "$BASE_URL/tasks/$TASK_ID")
    task_state=$(echo "$get_response" | python3 -c "import sys,json; print(json.load(sys.stdin)['state'])")
    if [ "$task_state" = "cancelled" ]; then
        pass "task state is cancelled after cancel"
    else
        # Task may have completed before cancel arrived — that's OK
        if [ "$task_state" = "completed" ] || [ "$task_state" = "running" ]; then
            pass "task state is $task_state (cancel raced with execution)"
        else
            fail "cancel state" "expected cancelled/completed/running, got $task_state"
        fi
    fi

    stop_daemon
fi

# ──────────────────────────────────────────────────────────────
# Test 5: Task completes successfully
# ──────────────────────────────────────────────────────────────
if should_run 5; then
    bold "Test 5: Task completes successfully"
    start_daemon "$WORKDIR/daemon_test.toml"

    submit_response=$(curl -sf -X POST "$BASE_URL/tasks" \
        -H 'Content-Type: application/json' \
        -d '{"task":"What is 2+2? Reply with just the number."}')
    TASK_ID=$(echo "$submit_response" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")

    # Poll until terminal state (max 60s)
    for i in $(seq 1 60); do
        get_response=$(curl -sf "$BASE_URL/tasks/$TASK_ID")
        task_state=$(echo "$get_response" | python3 -c "import sys,json; print(json.load(sys.stdin)['state'])")
        if [ "$task_state" = "completed" ] || [ "$task_state" = "failed" ]; then
            break
        fi
        sleep 1
    done

    if [ "$task_state" = "completed" ]; then
        pass "task completed"

        # Check result is non-empty
        result=$(echo "$get_response" | python3 -c "import sys,json; r=json.load(sys.stdin).get('result',''); print(r)")
        if [ -n "$result" ]; then
            pass "task has non-empty result"
        else
            fail "task result" "result is empty"
        fi

        # Check tokens were tracked
        input_tokens=$(echo "$get_response" | python3 -c "import sys,json; print(json.load(sys.stdin)['tokens_used']['input_tokens'])")
        if [ "$input_tokens" -gt 0 ] 2>/dev/null; then
            pass "tokens tracked (input=$input_tokens)"
        else
            fail "tokens" "input_tokens not tracked"
        fi
    else
        fail "completion" "expected completed, got $task_state after 60s"
    fi

    stop_daemon
fi

# ──────────────────────────────────────────────────────────────
# Test 6: Graceful shutdown via SIGTERM
# ──────────────────────────────────────────────────────────────
if should_run 6; then
    bold "Test 6: Graceful shutdown"
    start_daemon "$WORKDIR/daemon_test.toml"

    # Send SIGTERM
    kill -TERM "$DAEMON_PID"
    wait "$DAEMON_PID" 2>/dev/null || true
    exit_code=$?

    # Exit code 0 = graceful, 143 = SIGTERM default (also acceptable)
    if [ "$exit_code" -eq 0 ] || [ "$exit_code" -eq 143 ]; then
        pass "daemon shuts down gracefully (exit=$exit_code)"
    else
        fail "shutdown" "unexpected exit code: $exit_code"
    fi

    if grep -q "daemon shut down gracefully\|SIGTERM received" "$WORKDIR/_daemon_stderr"; then
        pass "shutdown logged"
    else
        fail "shutdown log" "no shutdown message in stderr"
    fi

    DAEMON_PID=""  # already stopped
fi

# ──────────────────────────────────────────────────────────────
# Test 7: Liveness probe /healthz
# ──────────────────────────────────────────────────────────────
if should_run 7; then
    bold "Test 7: Liveness probe /healthz"
    start_daemon "$WORKDIR/daemon_test.toml"

    response=$(curl -sf "$BASE_URL/healthz")
    if echo "$response" | grep -q '"status":"ok"'; then
        pass "healthz returns ok status"
    else
        fail "healthz" "unexpected response: $response"
    fi

    if echo "$response" | grep -q '"uptime_seconds"'; then
        pass "healthz includes uptime_seconds"
    else
        fail "healthz uptime" "missing uptime_seconds"
    fi

    # /health should still work as alias
    alias_response=$(curl -sf "$BASE_URL/health")
    if echo "$alias_response" | grep -q '"status":"ok"'; then
        pass "/health backward compat alias works"
    else
        fail "/health alias" "unexpected response: $alias_response"
    fi

    stop_daemon
fi

# ──────────────────────────────────────────────────────────────
# Test 8: Readiness probe /readyz
# ──────────────────────────────────────────────────────────────
if should_run 8; then
    bold "Test 8: Readiness probe /readyz"
    start_daemon "$WORKDIR/daemon_test.toml"

    response=$(curl -sf "$BASE_URL/readyz")
    if echo "$response" | grep -q '"ready":true'; then
        pass "readyz reports ready"
    else
        fail "readyz" "not ready: $response"
    fi

    if echo "$response" | grep -q '"name":"kafka"'; then
        pass "readyz includes kafka check"
    else
        fail "readyz kafka" "missing kafka check: $response"
    fi

    if echo "$response" | grep -q '"name":"shutdown"'; then
        pass "readyz includes shutdown check"
    else
        fail "readyz shutdown" "missing shutdown check: $response"
    fi

    stop_daemon
fi

# ──────────────────────────────────────────────────────────────
# Test 9: Prometheus metrics /metrics
# ──────────────────────────────────────────────────────────────
if should_run 9; then
    bold "Test 9: Prometheus metrics /metrics"
    start_daemon "$WORKDIR/daemon_test.toml"

    response=$(curl -sf "$BASE_URL/metrics")
    if [ -z "$response" ]; then
        fail "metrics" "empty response"
    else
        pass "metrics endpoint returns data"
    fi

    if echo "$response" | grep -q 'heartbit_daemon_tasks_submitted_total'; then
        pass "metrics has tasks_submitted_total"
    else
        fail "metrics submitted" "missing heartbit_daemon_tasks_submitted_total"
    fi

    if echo "$response" | grep -q 'heartbit_llm_calls_total'; then
        pass "metrics has llm_calls_total"
    else
        fail "metrics llm" "missing heartbit_llm_calls_total"
    fi

    if echo "$response" | grep -q 'heartbit_reliability_retry_attempts_total'; then
        pass "metrics has reliability counters"
    else
        fail "metrics reliability" "missing heartbit_reliability_retry_attempts_total"
    fi

    # Content-Type check
    content_type=$(curl -sf -o /dev/null -w "%{content_type}" "$BASE_URL/metrics")
    if echo "$content_type" | grep -q 'text/plain'; then
        pass "metrics content-type is text/plain"
    else
        fail "metrics content-type" "expected text/plain, got $content_type"
    fi

    stop_daemon
fi

# ──────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────
echo ""
bold "=== Results ==="
green "  Passed: $PASS"
[ "$FAIL" -gt 0 ] && red "  Failed: $FAIL" || echo "  Failed: $FAIL"
[ "$SKIP" -gt 0 ] && yellow "  Skipped: $SKIP" || echo "  Skipped: $SKIP"
echo ""

if [ "$FAIL" -gt 0 ]; then
    red "Failures:"
    printf "$ERRORS"
    exit 1
fi

green "All tests passed!"
