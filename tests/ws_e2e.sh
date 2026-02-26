#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Smoke tests for WebSocket channel (`/ws` endpoint)
#
# These are NOT CI tests. They require:
#   1. Running Kafka broker (docker compose up kafka -d)
#   2. OPENROUTER_API_KEY or ANTHROPIC_API_KEY set
#   3. target/release/heartbit built
#
# Design rules:
#   1. Assert on WS frame contents and session state.
#   2. Each test is self-contained with its own daemon instance.
#   3. Uses inline Python (stdlib only) for WebSocket framing.
#
# Usage:
#   ./tests/ws_e2e.sh        # run all tests
#   ./tests/ws_e2e.sh 2      # run only test 2
# ──────────────────────────────────────────────────────────────
set -euo pipefail

BINARY="$(cd "$(dirname "$0")/.." && pwd)/target/release/heartbit"
WORKDIR="$(mktemp -d)"
PASS=0
FAIL=0
SKIP=0
ERRORS=""
FILTER="${1:-}"

cleanup() {
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

find_port() {
    python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()'
}

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

start_daemon() {
    local port config_path="$1"
    port=$(find_port)
    BASE_URL="http://127.0.0.1:$port"
    WS_HOST="127.0.0.1"
    WS_PORT="$port"

    "$BINARY" daemon --config "$config_path" --bind "127.0.0.1:$port" \
        > "$WORKDIR/_daemon_stdout" 2> "$WORKDIR/_daemon_stderr" &
    DAEMON_PID=$!

    if ! wait_for_daemon "$BASE_URL"; then
        red "  daemon failed to start within 15s"
        cat "$WORKDIR/_daemon_stderr" >&2
        return 1
    fi

    # Allow Kafka consumer group rebalance
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
# Inline Python WebSocket client (stdlib only, no pip deps)
# ──────────────────────────────────────────────────────────────
# Creates $WORKDIR/ws_client.py — a minimal WS client that can
# send JSON frames and receive responses.
# ──────────────────────────────────────────────────────────────
create_ws_client() {
cat > "$WORKDIR/ws_client.py" << 'PYEOF'
"""Minimal WebSocket client using only Python stdlib.

Usage: python3 ws_client.py <host> <port> <command> [args...]

Commands:
    handshake           — test upgrade returns 101
    send_recv <json>    — send a JSON text frame, print all received frames for 2s
    session_flow        — full flow: create session, send chat, collect events
"""
import hashlib
import base64
import json
import os
import select
import socket
import struct
import sys
import time


def ws_connect(host, port, path="/ws"):
    """Perform WebSocket handshake, return connected socket."""
    sock = socket.create_connection((host, int(port)), timeout=10)
    key = base64.b64encode(os.urandom(16)).decode()
    request = (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        f"Upgrade: websocket\r\n"
        f"Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        f"Sec-WebSocket-Version: 13\r\n"
        f"\r\n"
    )
    sock.sendall(request.encode())

    # Read response headers
    response = b""
    while b"\r\n\r\n" not in response:
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("connection closed during handshake")
        response += chunk

    header_text = response.split(b"\r\n\r\n")[0].decode()
    status_line = header_text.split("\r\n")[0]
    if "101" not in status_line:
        raise ConnectionError(f"handshake failed: {status_line}")

    return sock


def ws_send(sock, text):
    """Send a masked text frame."""
    data = text.encode("utf-8")
    frame = bytearray()
    frame.append(0x81)  # FIN + text opcode
    mask_key = os.urandom(4)
    length = len(data)
    if length < 126:
        frame.append(0x80 | length)
    elif length < 65536:
        frame.append(0x80 | 126)
        frame.extend(struct.pack("!H", length))
    else:
        frame.append(0x80 | 127)
        frame.extend(struct.pack("!Q", length))
    frame.extend(mask_key)
    masked = bytearray(b ^ mask_key[i % 4] for i, b in enumerate(data))
    frame.extend(masked)
    sock.sendall(frame)


def ws_recv(sock, timeout_sec=5.0):
    """Receive one text frame. Returns string or None on timeout/close."""
    sock.setblocking(False)
    deadline = time.time() + timeout_sec
    buf = bytearray()

    while time.time() < deadline:
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        ready, _, _ = select.select([sock], [], [], min(remaining, 0.1))
        if ready:
            try:
                chunk = sock.recv(65536)
            except BlockingIOError:
                continue
            if not chunk:
                return None  # connection closed
            buf.extend(chunk)
            # Try to parse a frame
            if len(buf) >= 2:
                opcode = buf[0] & 0x0F
                masked = buf[1] & 0x80
                payload_len = buf[1] & 0x7F
                offset = 2
                if payload_len == 126:
                    if len(buf) < 4:
                        continue
                    payload_len = struct.unpack("!H", buf[2:4])[0]
                    offset = 4
                elif payload_len == 127:
                    if len(buf) < 10:
                        continue
                    payload_len = struct.unpack("!Q", buf[2:10])[0]
                    offset = 10
                if masked:
                    offset += 4  # skip mask key
                if len(buf) < offset + payload_len:
                    continue
                payload = buf[offset:offset + payload_len]
                if masked:
                    mask_key = buf[offset - 4:offset]
                    payload = bytearray(b ^ mask_key[i % 4] for i, b in enumerate(payload))
                if opcode == 0x08:  # close
                    return None
                if opcode == 0x09:  # ping -> pong
                    pong = bytearray([0x8A, 0x80]) + os.urandom(4)
                    sock.sendall(pong)
                    buf = buf[offset + payload_len:]
                    continue
                if opcode == 0x01:  # text
                    return payload.decode("utf-8", errors="replace")
                buf = buf[offset + payload_len:]
                continue
    return None


def ws_recv_all(sock, timeout_sec=3.0):
    """Receive all frames within timeout. Returns list of strings."""
    frames = []
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        frame = ws_recv(sock, timeout_sec=remaining)
        if frame is None:
            break
        frames.append(frame)
    return frames


def ws_close(sock):
    """Send close frame and shutdown."""
    try:
        close_frame = bytearray([0x88, 0x80]) + os.urandom(4)
        sock.sendall(close_frame)
    except Exception:
        pass
    try:
        sock.close()
    except Exception:
        pass


def main():
    if len(sys.argv) < 4:
        print("Usage: ws_client.py <host> <port> <command> [args...]", file=sys.stderr)
        sys.exit(1)

    host, port, command = sys.argv[1], sys.argv[2], sys.argv[3]

    if command == "handshake":
        try:
            sock = ws_connect(host, port)
            ws_close(sock)
            print("OK")
        except Exception as e:
            print(f"FAIL: {e}", file=sys.stderr)
            sys.exit(1)

    elif command == "send_recv":
        payload = sys.argv[4] if len(sys.argv) > 4 else "{}"
        timeout = float(sys.argv[5]) if len(sys.argv) > 5 else 3.0
        sock = ws_connect(host, port)
        ws_send(sock, payload)
        frames = ws_recv_all(sock, timeout_sec=timeout)
        ws_close(sock)
        for f in frames:
            print(f)

    elif command == "session_flow":
        timeout = float(sys.argv[4]) if len(sys.argv) > 4 else 60.0
        sock = ws_connect(host, port)
        results = {}

        # Step 1: Create session
        ws_send(sock, json.dumps({
            "type": "req", "id": "1",
            "method": "session.create",
            "params": {"title": "E2E Test Session"}
        }))
        resp = ws_recv(sock, timeout_sec=5)
        if resp:
            parsed = json.loads(resp)
            results["session_create"] = parsed
            session_id = parsed.get("payload", {}).get("session_id")
        else:
            print(json.dumps({"error": "no response to session.create"}))
            sys.exit(1)

        # Step 2: List sessions
        ws_send(sock, json.dumps({
            "type": "req", "id": "2",
            "method": "session.list",
            "params": {}
        }))
        resp = ws_recv(sock, timeout_sec=5)
        if resp:
            results["session_list"] = json.loads(resp)

        # Step 3: Send chat message
        ws_send(sock, json.dumps({
            "type": "req", "id": "3",
            "method": "chat.send",
            "params": {"session_id": session_id, "message": "What is 2+2? Reply with just the number."}
        }))
        resp = ws_recv(sock, timeout_sec=5)
        if resp:
            parsed = json.loads(resp)
            results["chat_send"] = parsed
            task_id = parsed.get("payload", {}).get("task_id")

        # Step 4: Collect events (deltas, final, error)
        events = []
        deadline = time.time() + timeout
        got_final = False
        while time.time() < deadline:
            frame = ws_recv(sock, timeout_sec=min(deadline - time.time(), 2.0))
            if frame is None:
                break
            parsed = json.loads(frame)
            events.append(parsed)
            event_type = parsed.get("event", "")
            if event_type in ("chat.final", "chat.error"):
                got_final = True
                break
        results["events"] = events
        results["got_final"] = got_final

        # Step 5: Get chat history
        ws_send(sock, json.dumps({
            "type": "req", "id": "4",
            "method": "chat.history",
            "params": {"session_id": session_id}
        }))
        resp = ws_recv(sock, timeout_sec=5)
        if resp:
            results["chat_history"] = json.loads(resp)

        # Step 6: Delete session
        ws_send(sock, json.dumps({
            "type": "req", "id": "5",
            "method": "session.delete",
            "params": {"session_id": session_id}
        }))
        resp = ws_recv(sock, timeout_sec=5)
        if resp:
            results["session_delete"] = json.loads(resp)

        ws_close(sock)
        print(json.dumps(results))

    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
PYEOF
}

# ──────────────────────────────────────────────────────────────
# Preflight checks
# ──────────────────────────────────────────────────────────────
bold "=== WebSocket E2E Tests ==="
echo ""

if [ ! -x "$BINARY" ]; then
    red "Binary not found: $BINARY"
    red "Run: cargo build --release"
    exit 1
fi

if ! docker compose exec kafka /opt/kafka/bin/kafka-topics.sh --list --bootstrap-server localhost:9092 > /dev/null 2>&1; then
    red "Kafka not reachable. Run: docker compose up kafka -d"
    exit 1
fi

if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    red "No API key set. Export OPENROUTER_API_KEY or ANTHROPIC_API_KEY."
    exit 1
fi

if ! python3 -c "import socket, select, struct" 2>/dev/null; then
    red "Python3 with stdlib not available"
    exit 1
fi

create_ws_client

# Create daemon config with WS enabled
cat > "$WORKDIR/ws_test.toml" << TOML
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
consumer_group = "heartbit-ws-e2e-$$"
commands_topic = "hb.ws.test.commands.$$"
events_topic = "hb.ws.test.events.$$"

[daemon.ws]
enabled = true
interaction_timeout_seconds = 30
max_connections = 10
TOML

# If using Anthropic, patch the config
if [ -n "${ANTHROPIC_API_KEY:-}" ] && [ -z "${OPENROUTER_API_KEY:-}" ]; then
    sed -i 's/name = "openrouter"/name = "anthropic"/' "$WORKDIR/ws_test.toml"
    sed -i "s/model = .*/model = \"claude-sonnet-4-20250514\"/" "$WORKDIR/ws_test.toml"
fi

# ──────────────────────────────────────────────────────────────
# Test 1: WebSocket handshake succeeds
# ──────────────────────────────────────────────────────────────
if should_run 1; then
    bold "Test 1: WebSocket handshake"
    start_daemon "$WORKDIR/ws_test.toml"

    result=$(python3 "$WORKDIR/ws_client.py" "$WS_HOST" "$WS_PORT" handshake 2>&1)
    if [ "$result" = "OK" ]; then
        pass "WS handshake returns 101 Switching Protocols"
    else
        fail "handshake" "$result"
    fi

    stop_daemon
fi

# ──────────────────────────────────────────────────────────────
# Test 2: Session CRUD over WebSocket
# ──────────────────────────────────────────────────────────────
if should_run 2; then
    bold "Test 2: Session CRUD"
    start_daemon "$WORKDIR/ws_test.toml"

    # Create a session
    response=$(python3 "$WORKDIR/ws_client.py" "$WS_HOST" "$WS_PORT" send_recv \
        '{"type":"req","id":"1","method":"session.create","params":{"title":"Test"}}' 5)

    if echo "$response" | grep -q '"ok":true'; then
        pass "session.create returns ok"
    else
        fail "session.create" "unexpected: $response"
    fi

    if echo "$response" | grep -q '"session_id"'; then
        pass "session.create returns session_id"
    else
        fail "session.create id" "missing session_id: $response"
    fi

    SESSION_ID=$(echo "$response" | python3 -c "import sys,json; print(json.loads(sys.stdin.readline())['payload']['session_id'])" 2>/dev/null || echo "")

    if [ -n "$SESSION_ID" ]; then
        # List sessions
        response=$(python3 "$WORKDIR/ws_client.py" "$WS_HOST" "$WS_PORT" send_recv \
            '{"type":"req","id":"2","method":"session.list","params":{}}' 5)

        if echo "$response" | grep -q "$SESSION_ID"; then
            pass "session.list includes created session"
        else
            fail "session.list" "session $SESSION_ID not found: $response"
        fi

        # Delete session
        response=$(python3 "$WORKDIR/ws_client.py" "$WS_HOST" "$WS_PORT" send_recv \
            "{\"type\":\"req\",\"id\":\"3\",\"method\":\"session.delete\",\"params\":{\"session_id\":\"$SESSION_ID\"}}" 5)

        if echo "$response" | grep -q '"deleted":true'; then
            pass "session.delete returns deleted:true"
        else
            fail "session.delete" "unexpected: $response"
        fi
    else
        fail "session CRUD" "could not extract session_id"
    fi

    stop_daemon
fi

# ──────────────────────────────────────────────────────────────
# Test 3: Invalid method returns error
# ──────────────────────────────────────────────────────────────
if should_run 3; then
    bold "Test 3: Invalid method error"
    start_daemon "$WORKDIR/ws_test.toml"

    response=$(python3 "$WORKDIR/ws_client.py" "$WS_HOST" "$WS_PORT" send_recv \
        '{"type":"req","id":"err1","method":"nonexistent.method","params":{}}' 5)

    if echo "$response" | grep -q '"ok":false'; then
        pass "unknown method returns ok:false"
    else
        fail "unknown method" "expected ok:false: $response"
    fi

    if echo "$response" | grep -q 'unknown method'; then
        pass "error message mentions unknown method"
    else
        fail "error msg" "expected 'unknown method' in: $response"
    fi

    stop_daemon
fi

# ──────────────────────────────────────────────────────────────
# Test 4: Invalid JSON returns error frame
# ──────────────────────────────────────────────────────────────
if should_run 4; then
    bold "Test 4: Malformed JSON error"
    start_daemon "$WORKDIR/ws_test.toml"

    response=$(python3 "$WORKDIR/ws_client.py" "$WS_HOST" "$WS_PORT" send_recv \
        'not valid json at all' 5)

    if echo "$response" | grep -q '"ok":false'; then
        pass "malformed JSON returns error frame"
    else
        fail "malformed json" "expected error frame: $response"
    fi

    stop_daemon
fi

# ──────────────────────────────────────────────────────────────
# Test 5: Full chat flow — send message, receive events
# ──────────────────────────────────────────────────────────────
if should_run 5; then
    bold "Test 5: Full chat flow (send + events + history)"
    start_daemon "$WORKDIR/ws_test.toml"

    flow_result=$(python3 "$WORKDIR/ws_client.py" "$WS_HOST" "$WS_PORT" session_flow 90)

    if [ -z "$flow_result" ]; then
        fail "chat flow" "empty result from session_flow"
    else
        # Check session.create
        if echo "$flow_result" | python3 -c "
import sys,json
d=json.loads(sys.stdin.read())
assert d['session_create']['ok'] == True
" 2>/dev/null; then
            pass "session created successfully"
        else
            fail "session create" "session_create not ok"
        fi

        # Check session.list includes our session
        if echo "$flow_result" | python3 -c "
import sys,json
d=json.loads(sys.stdin.read())
sessions=d['session_list']['payload']['sessions']
assert len(sessions) >= 1
" 2>/dev/null; then
            pass "session listed"
        else
            fail "session list" "empty session list"
        fi

        # Check chat.send returned task_id
        if echo "$flow_result" | python3 -c "
import sys,json
d=json.loads(sys.stdin.read())
assert 'task_id' in d['chat_send']['payload']
" 2>/dev/null; then
            pass "chat.send returns task_id"
        else
            fail "chat.send" "missing task_id in response"
        fi

        # Check we received events
        event_count=$(echo "$flow_result" | python3 -c "
import sys,json
d=json.loads(sys.stdin.read())
print(len(d.get('events', [])))
" 2>/dev/null || echo "0")
        if [ "$event_count" -gt 0 ] 2>/dev/null; then
            pass "received $event_count events"
        else
            fail "events" "no events received"
        fi

        # Check we got a terminal event (chat.final or chat.error)
        if echo "$flow_result" | python3 -c "
import sys,json
d=json.loads(sys.stdin.read())
assert d['got_final'] == True
" 2>/dev/null; then
            pass "received terminal event (chat.final or chat.error)"
        else
            fail "terminal event" "no chat.final/chat.error received within timeout"
        fi

        # Check chat.delta events contain text
        has_delta=$(echo "$flow_result" | python3 -c "
import sys,json
d=json.loads(sys.stdin.read())
deltas=[e for e in d['events'] if e.get('event')=='chat.delta']
print('yes' if deltas else 'no')
" 2>/dev/null || echo "no")
        if [ "$has_delta" = "yes" ]; then
            pass "received chat.delta streaming events"
        else
            # Not all LLMs stream — this is a soft check
            yellow "  NOTE: no chat.delta events (model may not stream)"
        fi

        # Check chat history has at least user + assistant messages
        if echo "$flow_result" | python3 -c "
import sys,json
d=json.loads(sys.stdin.read())
if 'chat_history' in d:
    history=d['chat_history']['payload']
    assert len(history) >= 2, f'expected >=2 messages, got {len(history)}'
else:
    raise Exception('no chat_history in result')
" 2>/dev/null; then
            pass "chat history has user + assistant messages"
        else
            fail "chat history" "expected at least 2 messages in history"
        fi

        # Check session.delete
        if echo "$flow_result" | python3 -c "
import sys,json
d=json.loads(sys.stdin.read())
assert d['session_delete']['ok'] == True
assert d['session_delete']['payload']['deleted'] == True
" 2>/dev/null; then
            pass "session deleted after flow"
        else
            fail "session delete" "session delete failed"
        fi
    fi

    stop_daemon
fi

# ──────────────────────────────────────────────────────────────
# Test 6: Chat on non-existent session returns error
# ──────────────────────────────────────────────────────────────
if should_run 6; then
    bold "Test 6: Chat on non-existent session"
    start_daemon "$WORKDIR/ws_test.toml"

    response=$(python3 "$WORKDIR/ws_client.py" "$WS_HOST" "$WS_PORT" send_recv \
        '{"type":"req","id":"x","method":"chat.send","params":{"session_id":"00000000-0000-0000-0000-000000000000","message":"hello"}}' 5)

    if echo "$response" | grep -q '"ok":false'; then
        pass "chat.send on missing session returns error"
    else
        fail "missing session" "expected error: $response"
    fi

    if echo "$response" | grep -q 'session not found'; then
        pass "error message says session not found"
    else
        fail "error msg" "expected 'session not found' in: $response"
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
