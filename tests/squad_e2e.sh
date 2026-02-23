#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# End-to-end integration test for squad execution + audit trail
#
# NOT a CI test. Calls a real LLM and costs money.
# Run manually: ./tests/squad_e2e.sh
#
# Requires: OPENROUTER_API_KEY, target/release/heartbit-cli
# ──────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BINARY="$ROOT_DIR/target/release/heartbit-cli"
CONFIG="$SCRIPT_DIR/squad_e2e.toml"
WORKDIR="$(mktemp -d)"
PASS=0
FAIL=0
WARN=0
TOTAL_CHECKS=0

cleanup() { rm -rf "$WORKDIR"; }
trap cleanup EXIT

red()    { printf '\033[1;31m%s\033[0m\n' "$*"; }
green()  { printf '\033[1;32m%s\033[0m\n' "$*"; }
yellow() { printf '\033[1;33m%s\033[0m\n' "$*"; }
bold()   { printf '\033[1m%s\033[0m\n' "$*"; }

check() {
    local name="$1" result="$2" detail="${3:-}"
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    if [ "$result" = "pass" ]; then
        PASS=$((PASS + 1))
        green "  ✓ $name"
    elif [ "$result" = "warn" ]; then
        WARN=$((WARN + 1))
        yellow "  ⚠ $name: $detail"
    else
        FAIL=$((FAIL + 1))
        red "  ✗ $name: $detail"
    fi
}

# ─── Preflight ───────────────────────────────────────────────

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    red "OPENROUTER_API_KEY not set"; exit 1
fi
if [ ! -x "$BINARY" ]; then
    bold "Binary not found, building release..."
    (cd "$ROOT_DIR" && cargo build --release 2>&1) || { red "Build failed"; exit 1; }
fi
if [ ! -f "$CONFIG" ]; then
    red "Config not found: $CONFIG"; exit 1
fi

bold "╔════════════════════════════════════════════════════╗"
bold "║  Squad E2E Integration Test — Audit Trail Check   ║"
bold "╠════════════════════════════════════════════════════╣"
echo "  Binary:  $BINARY"
echo "  Config:  $CONFIG"
echo "  Workdir: $WORKDIR"
bold "╚════════════════════════════════════════════════════╝"
echo ""

# ─── Run the task ────────────────────────────────────────────

TASK="Create a Rust project analysis: First, use bash to run 'rustc --version' and examine the Cargo.toml at $ROOT_DIR/Cargo.toml. Then create a file $WORKDIR/analysis.md with a brief analysis of the project structure. Include the Rust version, workspace members, and a one-paragraph assessment."

bold "PHASE 1: Running orchestrator with verbose events"
bold "Task: $(echo "$TASK" | head -c 100)..."
echo ""

EXIT_CODE=0
timeout 180 "$BINARY" run --config "$CONFIG" -v $TASK \
    > "$WORKDIR/stdout.txt" 2> "$WORKDIR/stderr.txt" || EXIT_CODE=$?

# Extract event lines from stderr
grep '^\[event\] ' "$WORKDIR/stderr.txt" | sed 's/^\[event\] //' > "$WORKDIR/events.jsonl" || true
EVENT_COUNT=$(wc -l < "$WORKDIR/events.jsonl" | tr -d ' ')

echo ""
bold "PHASE 2: Analyzing event stream ($EVENT_COUNT events captured)"
echo ""

# ═══════════════════════════════════════════════════════════════
# CHECK 1: Basic execution
# ═══════════════════════════════════════════════════════════════
bold "── Basic Execution ──"

if [ "$EXIT_CODE" -eq 0 ]; then
    check "CLI exited successfully" "pass"
else
    check "CLI exited successfully" "fail" "exit code $EXIT_CODE"
fi

if [ "$EVENT_COUNT" -gt 0 ]; then
    check "Events captured ($EVENT_COUNT)" "pass"
else
    check "Events captured" "fail" "no events in stderr"
    bold "No events to analyze. Dumping stderr:"
    cat "$WORKDIR/stderr.txt"
    exit 1
fi

# ═══════════════════════════════════════════════════════════════
# CHECK 2: All events are valid JSON
# ═══════════════════════════════════════════════════════════════
bold "── JSON Validity ──"

INVALID_JSON=$(python3 - "$WORKDIR/events.jsonl" <<'PYEOF'
import json, sys
count = 0
with open(sys.argv[1]) as f:
    for line in f:
        try:
            json.loads(line)
        except:
            count += 1
print(count)
PYEOF
)

if [ "$INVALID_JSON" -eq 0 ]; then
    check "All events are valid JSON" "pass"
else
    check "All events are valid JSON" "fail" "$INVALID_JSON invalid lines"
fi

# ═══════════════════════════════════════════════════════════════
# CHECK 3: Required event types present (snake_case serde names)
# ═══════════════════════════════════════════════════════════════
bold "── Event Types ──"

python3 - "$WORKDIR/events.jsonl" "$WORKDIR/event_types.txt" <<'PYEOF'
import json, sys
types = set()
with open(sys.argv[1]) as f:
    for line in f:
        try:
            e = json.loads(line)
            types.add(e.get("type", "unknown"))
        except:
            pass
with open(sys.argv[2], "w") as out:
    for t in sorted(types):
        out.write(t + "\n")
PYEOF

echo "  Event types found:"
while IFS= read -r t; do
    echo "    - $t"
done < "$WORKDIR/event_types.txt"

# Events are serde-serialized as snake_case
for required_type in run_started turn_started llm_response tool_call_started tool_call_completed run_completed; do
    if grep -q "^${required_type}$" "$WORKDIR/event_types.txt"; then
        check "Event type $required_type present" "pass"
    else
        check "Event type $required_type present" "fail" "not found in event stream"
    fi
done

if grep -q "^sub_agents_dispatched$" "$WORKDIR/event_types.txt"; then
    check "sub_agents_dispatched events present" "pass"
else
    check "sub_agents_dispatched events present" "warn" "orchestrator may not have delegated"
fi
if grep -q "^sub_agent_completed$" "$WORKDIR/event_types.txt"; then
    check "sub_agent_completed events present" "pass"
else
    check "sub_agent_completed events present" "warn" "orchestrator may not have delegated"
fi

# ═══════════════════════════════════════════════════════════════
# CHECK 4: Agent names in events
# ═══════════════════════════════════════════════════════════════
bold "── Agent Names ──"

python3 - "$WORKDIR/events.jsonl" "$WORKDIR/agents.txt" <<'PYEOF'
import json, sys
agents = set()
with open(sys.argv[1]) as f:
    for line in f:
        try:
            e = json.loads(line)
            a = e.get("agent", "")
            if a:
                agents.add(a)
        except:
            pass
with open(sys.argv[2], "w") as out:
    for a in sorted(agents):
        out.write(a + "\n")
PYEOF

echo "  Agents observed:"
while IFS= read -r a; do
    echo "    - $a"
done < "$WORKDIR/agents.txt"

if grep -q "^orchestrator$" "$WORKDIR/agents.txt"; then
    check "Orchestrator agent in events" "pass"
else
    check "Orchestrator agent in events" "fail" "no events with agent=orchestrator"
fi

SUB_AGENT_COUNT=$(grep -cv "^orchestrator$" "$WORKDIR/agents.txt" || echo 0)
if [ "$SUB_AGENT_COUNT" -gt 0 ]; then
    check "Sub-agent events forwarded ($SUB_AGENT_COUNT agents)" "pass"
else
    check "Sub-agent events forwarded" "warn" "no sub-agent events (on_event not forwarded?)"
fi

# ═══════════════════════════════════════════════════════════════
# CHECK 5: LlmResponse enrichment (text, latency_ms, model)
# ═══════════════════════════════════════════════════════════════
bold "── LlmResponse Enrichment ──"

python3 - "$WORKDIR/events.jsonl" "$WORKDIR/llm_analysis.json" <<'PYEOF'
import json, sys

results = {
    "total": 0, "with_text": 0, "with_latency": 0, "with_model": 0,
    "empty_text": 0, "zero_latency": 0, "latencies_ms": [],
    "models": [], "agents": [], "sample_texts": [],
}

agents_set = set()
models_set = set()

with open(sys.argv[1]) as f:
    for line in f:
        try:
            e = json.loads(line)
            if e.get("type") != "llm_response":
                continue
            results["total"] += 1
            agents_set.add(e.get("agent", ""))

            text = e.get("text", "")
            if text:
                results["with_text"] += 1
                results["sample_texts"].append(text[:200])
            else:
                results["empty_text"] += 1

            latency = e.get("latency_ms", 0)
            if latency > 0:
                results["with_latency"] += 1
                results["latencies_ms"].append(latency)
            else:
                results["zero_latency"] += 1

            model = e.get("model")
            if model:
                results["with_model"] += 1
                models_set.add(model)
        except:
            pass

results["models"] = sorted(models_set)
results["agents"] = sorted(agents_set)
with open(sys.argv[2], "w") as out:
    json.dump(results, out, indent=2)
PYEOF

LLM_TOTAL=$(python3 -c "import json; d=json.load(open('$WORKDIR/llm_analysis.json')); print(d['total'])")
LLM_WITH_TEXT=$(python3 -c "import json; d=json.load(open('$WORKDIR/llm_analysis.json')); print(d['with_text'])")
LLM_WITH_LATENCY=$(python3 -c "import json; d=json.load(open('$WORKDIR/llm_analysis.json')); print(d['with_latency'])")
LLM_WITH_MODEL=$(python3 -c "import json; d=json.load(open('$WORKDIR/llm_analysis.json')); print(d['with_model'])")

if [ "$LLM_TOTAL" -gt 0 ]; then
    check "llm_response events found ($LLM_TOTAL)" "pass"
else
    check "llm_response events found" "fail" "none"
fi

if [ "$LLM_WITH_TEXT" -gt 0 ]; then
    check "llm_response has text field ($LLM_WITH_TEXT/$LLM_TOTAL)" "pass"
else
    check "llm_response has text field" "fail" "0/$LLM_TOTAL have text"
fi

if [ "$LLM_WITH_LATENCY" -gt 0 ]; then
    check "llm_response has latency_ms ($LLM_WITH_LATENCY/$LLM_TOTAL)" "pass"
else
    check "llm_response has latency_ms" "fail" "0/$LLM_TOTAL have latency"
fi

if [ "$LLM_WITH_MODEL" -gt 0 ]; then
    check "llm_response has model field ($LLM_WITH_MODEL/$LLM_TOTAL)" "pass"
else
    check "llm_response has model field" "fail" "0/$LLM_TOTAL have model"
fi

# Print latency stats
python3 - "$WORKDIR/llm_analysis.json" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
lats = d.get("latencies_ms", [])
if lats:
    print(f"  Latency stats: min={min(lats)}ms, max={max(lats)}ms, avg={sum(lats)//len(lats)}ms")
models = d.get("models", [])
if models:
    print(f"  Models seen: {', '.join(models)}")
texts = d.get("sample_texts", [])
for i, t in enumerate(texts[:3]):
    preview = t[:80].replace("\n", "\\n")
    print(f'  Sample text [{i}]: "{preview}..."')
PYEOF

# ═══════════════════════════════════════════════════════════════
# CHECK 6: ToolCall enrichment (input/output)
# ═══════════════════════════════════════════════════════════════
bold "── ToolCall Enrichment ──"

python3 - "$WORKDIR/events.jsonl" "$WORKDIR/tool_analysis.json" <<'PYEOF'
import json, sys

results = {
    "started_total": 0, "started_with_input": 0,
    "completed_total": 0, "completed_with_output": 0,
    "tools_used": {}, "agents_with_tools": [], "truncated_outputs": 0,
}

agents_set = set()

with open(sys.argv[1]) as f:
    for line in f:
        try:
            e = json.loads(line)
            evt_type = e.get("type", "")

            if evt_type == "tool_call_started":
                results["started_total"] += 1
                tool_name = e.get("tool_name", "")
                agent = e.get("agent", "")
                results["tools_used"][tool_name] = results["tools_used"].get(tool_name, 0) + 1
                agents_set.add(agent)
                inp = e.get("input", "")
                if inp:
                    results["started_with_input"] += 1

            elif evt_type == "tool_call_completed":
                results["completed_total"] += 1
                out = e.get("output", "")
                if out:
                    results["completed_with_output"] += 1
                    if "[truncated:" in out:
                        results["truncated_outputs"] += 1
        except:
            pass

results["agents_with_tools"] = sorted(agents_set)
with open(sys.argv[2], "w") as out:
    json.dump(results, out, indent=2)
PYEOF

TC_STARTED=$(python3 -c "import json; d=json.load(open('$WORKDIR/tool_analysis.json')); print(d['started_total'])")
TC_WITH_INPUT=$(python3 -c "import json; d=json.load(open('$WORKDIR/tool_analysis.json')); print(d['started_with_input'])")
TC_COMPLETED=$(python3 -c "import json; d=json.load(open('$WORKDIR/tool_analysis.json')); print(d['completed_total'])")
TC_WITH_OUTPUT=$(python3 -c "import json; d=json.load(open('$WORKDIR/tool_analysis.json')); print(d['completed_with_output'])")

if [ "$TC_STARTED" -gt 0 ]; then
    check "tool_call_started events found ($TC_STARTED)" "pass"
else
    check "tool_call_started events found" "warn" "no tool calls"
fi

if [ "$TC_STARTED" -gt 0 ] && [ "$TC_WITH_INPUT" -eq "$TC_STARTED" ]; then
    check "All tool_call_started have input ($TC_WITH_INPUT/$TC_STARTED)" "pass"
elif [ "$TC_WITH_INPUT" -gt 0 ]; then
    check "tool_call_started have input" "warn" "$TC_WITH_INPUT/$TC_STARTED have input"
else
    check "tool_call_started have input" "fail" "0/$TC_STARTED have input"
fi

if [ "$TC_COMPLETED" -gt 0 ] && [ "$TC_WITH_OUTPUT" -eq "$TC_COMPLETED" ]; then
    check "All tool_call_completed have output ($TC_WITH_OUTPUT/$TC_COMPLETED)" "pass"
elif [ "$TC_WITH_OUTPUT" -gt 0 ]; then
    check "tool_call_completed have output" "warn" "$TC_WITH_OUTPUT/$TC_COMPLETED have output"
else
    check "tool_call_completed have output" "fail" "0/$TC_COMPLETED have output"
fi

# Print tool usage summary
python3 - "$WORKDIR/tool_analysis.json" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
tools = d.get("tools_used", {})
if tools:
    print("  Tools used:")
    for name, count in sorted(tools.items(), key=lambda x: -x[1]):
        print(f"    {name}: {count}x")
agents = d.get("agents_with_tools", [])
if agents:
    print(f"  Agents using tools: {', '.join(agents)}")
trunc = d.get("truncated_outputs", 0)
if trunc:
    print(f"  Truncated outputs: {trunc}")
PYEOF

# ═══════════════════════════════════════════════════════════════
# CHECK 7: Event ordering (per agent)
# ═══════════════════════════════════════════════════════════════
bold "── Event Ordering ──"

python3 - "$WORKDIR/events.jsonl" "$WORKDIR/ordering.json" <<'PYEOF'
import json, sys
from collections import Counter

agent_events = {}
with open(sys.argv[1]) as f:
    for i, line in enumerate(f):
        try:
            e = json.loads(line)
            agent = e.get("agent", "unknown")
            if agent not in agent_events:
                agent_events[agent] = []
            agent_events[agent].append({"idx": i, "type": e.get("type", "")})
        except:
            pass

results = {"agents": {}, "errors": [], "skipped": []}
for agent, events in agent_events.items():
    types = [e["type"] for e in events]
    results["agents"][agent] = types
    # Squad-leader and squad composite agents emit only synthetic dispatch/completion
    # markers (sub_agents_dispatched, sub_agent_completed) — not full agent lifecycles.
    # Skip ordering checks for these.
    if agent == "squad-leader" or agent.startswith("squad["):
        results["skipped"].append(agent)
        continue
    if types and types[0] != "run_started":
        results["errors"].append(f"{agent}: first event is {types[0]}, expected run_started")
    # Last event: run_completed, run_failed, OR sub_agent_completed (emitted after run_completed for sub-agents)
    valid_last = ("run_completed", "run_failed", "sub_agent_completed")
    if types and types[-1] not in valid_last:
        results["errors"].append(f"{agent}: last event is {types[-1]}, expected one of {valid_last}")

with open(sys.argv[2], "w") as out:
    json.dump(results, out, indent=2)

# Print event sequence per agent
for agent, types in sorted(results["agents"].items()):
    counts = Counter(types)
    summary = ", ".join(f"{t}:{c}" for t, c in counts.most_common())
    print(f"  {agent} ({len(types)} events): {summary}")
PYEOF

ORDER_ERRORS=$(python3 -c "import json; d=json.load(open('$WORKDIR/ordering.json')); print(len(d.get('errors',[])))")
if [ "$ORDER_ERRORS" -eq 0 ]; then
    check "Event ordering correct per agent" "pass"
else
    python3 -c "import json; d=json.load(open('$WORKDIR/ordering.json')); [print(f'    {e}') for e in d['errors']]"
    check "Event ordering" "fail" "$ORDER_ERRORS ordering violations"
fi

# ═══════════════════════════════════════════════════════════════
# CHECK 8: Delegation events
# ═══════════════════════════════════════════════════════════════
bold "── Delegation & Squad Events ──"

python3 - "$WORKDIR/events.jsonl" "$WORKDIR/delegation.json" <<'PYEOF'
import json, sys

dispatched = []
completed = []
squad_count = 0

with open(sys.argv[1]) as f:
    for line in f:
        try:
            e = json.loads(line)
            t = e.get("type", "")
            if t == "sub_agents_dispatched":
                # Squad dispatches use agent="squad-leader",
                # delegate dispatches use agent="orchestrator"
                is_squad = e.get("agent", "") == "squad-leader"
                dispatched.append({
                    "agents": e.get("agents", []),
                    "tasks": e.get("tasks", []),
                    "is_squad": is_squad,
                })
                if is_squad:
                    squad_count += 1
            elif t == "sub_agent_completed":
                completed.append({
                    "agent": e.get("agent", ""),
                    "success": e.get("success", False),
                })
        except:
            pass

results = {
    "delegate_count": len(dispatched),
    "complete_count": len(completed),
    "squad_count": squad_count,
    "dispatched": dispatched,
    "completed": completed,
}
with open(sys.argv[2], "w") as out:
    json.dump(results, out, indent=2)

# Print delegation details
for d in dispatched:
    agents = d.get("agents", [])
    tasks = d.get("tasks", [])
    mode = "SQUAD" if d.get("is_squad") else "DELEGATE"
    task_preview = "; ".join(t[:60] for t in tasks[:3])
    print(f"  [{mode}] agents={agents} task=\"{task_preview}\"")

for c in completed:
    status = "OK" if c["success"] else "FAIL"
    print(f"  [{status}] {c['agent']} completed")
PYEOF

DISPATCH_COUNT=$(python3 -c "import json; d=json.load(open('$WORKDIR/delegation.json')); print(d['delegate_count'])")
COMPLETE_COUNT=$(python3 -c "import json; d=json.load(open('$WORKDIR/delegation.json')); print(d['complete_count'])")
SQUAD_COUNT=$(python3 -c "import json; d=json.load(open('$WORKDIR/delegation.json')); print(d['squad_count'])")

if [ "$DISPATCH_COUNT" -gt 0 ]; then
    check "Delegation dispatched ($DISPATCH_COUNT batches)" "pass"
else
    check "Delegation dispatched" "warn" "no delegation occurred"
fi

if [ "$COMPLETE_COUNT" -gt 0 ]; then
    check "Sub-agents completed ($COMPLETE_COUNT)" "pass"
fi

if [ "$SQUAD_COUNT" -gt 0 ]; then
    check "Squad formation triggered ($SQUAD_COUNT squads)" "pass"
else
    check "Squad formation" "warn" "not triggered (LLM chose delegate_task instead)"
fi

# ═══════════════════════════════════════════════════════════════
# CHECK 9: Token usage
# ═══════════════════════════════════════════════════════════════
bold "── Token Usage ──"

python3 - "$WORKDIR/events.jsonl" <<'PYEOF'
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
        except:
            pass
PYEOF

ORCH_TOKENS=$(python3 -c "
import json
with open('$WORKDIR/events.jsonl') as f:
    for line in f:
        e = json.loads(line)
        if e.get('type') == 'run_completed' and e.get('agent') == 'orchestrator':
            u = e.get('total_usage', {})
            print(u.get('input_tokens', 0) + u.get('output_tokens', 0))
            break
    else:
        print(0)
")

if [ "$ORCH_TOKENS" -gt 0 ]; then
    check "Orchestrator reports aggregated token usage ($ORCH_TOKENS)" "pass"
else
    check "Orchestrator token usage" "warn" "no RunCompleted or zero tokens"
fi

# ═══════════════════════════════════════════════════════════════
# CHECK 10: Artifact verification
# ═══════════════════════════════════════════════════════════════
bold "── Task Completion ──"

if [ -f "$WORKDIR/analysis.md" ]; then
    check "Output file created (analysis.md)" "pass"
    FILE_SIZE=$(wc -c < "$WORKDIR/analysis.md" | tr -d ' ')
    if [ "$FILE_SIZE" -gt 50 ]; then
        check "Output file has content (${FILE_SIZE} bytes)" "pass"
    else
        check "Output file has content" "warn" "only ${FILE_SIZE} bytes"
    fi
else
    check "Output file created" "warn" "analysis.md not found (LLM may have chosen different path)"
fi

# ═══════════════════════════════════════════════════════════════
# Full event timeline
# ═══════════════════════════════════════════════════════════════
bold "── Event Timeline ──"

python3 - "$WORKDIR/events.jsonl" <<'PYEOF'
import json, sys

events = []
with open(sys.argv[1]) as f:
    for line in f:
        try:
            events.append(json.loads(line))
        except:
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
            agents = e.get("agents", [e.get("agent","")])
            squad = " [SQUAD]" if agent == "squad-leader" else ""
            success = ""
            if "success" in e:
                success = " OK" if e["success"] else " FAIL"
            details = f"agents={agents}{squad}{success}"
        print(f"  {i:>3} {agent:<16} {etype:<24} {details[:70]}")
PYEOF

# Save event log path
echo ""
echo "  Event log: $WORKDIR/events.jsonl"
echo "  Stdout: $WORKDIR/stdout.txt"
echo "  Stderr: $WORKDIR/stderr.txt"

# ─── Results ─────────────────────────────────────────────────

echo ""
bold "╔════════════════════════════════════════════════════╗"
if [ "$FAIL" -eq 0 ]; then
    green "║  ALL $PASS/$TOTAL_CHECKS CHECKS PASSED (${WARN} warnings)               ║"
else
    red "║  $FAIL/$TOTAL_CHECKS CHECKS FAILED ($PASS passed, $WARN warnings)     ║"
fi
bold "╚════════════════════════════════════════════════════╝"
echo ""

if [ "$FAIL" -gt 0 ]; then
    echo "Workdir preserved for inspection: $WORKDIR"
    trap - EXIT
fi

exit "$FAIL"
