#!/usr/bin/env bash
# Run Terminal-Bench 2.0 through the Heartbit Harbor adapter, pointed at the
# Koyeb Qwen OpenAI-compat endpoint, using the TUI "brain" (orchestrator).
#
# Usage:
#   benchmarks/terminal-bench-2/scripts/run-tb2-qwen.sh            # smoke: 2 easy tasks
#   benchmarks/terminal-bench-2/scripts/run-tb2-qwen.sh --full     # all 89 (long)
#   TASKS="build-cython-ext" benchmarks/terminal-bench-2/scripts/run-tb2-qwen.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
TB2="$ROOT/benchmarks/terminal-bench-2"
cd "$TB2"

# Secrets + endpoint: prefer the harness .env (gitignored).
if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

: "${HEARTBIT_OPENAI_BASE_URL:?set HEARTBIT_OPENAI_BASE_URL (e.g. in $ROOT/.env)}"
: "${HEARTBIT_OPENAI_API_KEY:?set HEARTBIT_OPENAI_API_KEY}"

export HEARTBIT_BASE_URL="$HEARTBIT_OPENAI_BASE_URL"
export HEARTBIT_API_KEY="$HEARTBIT_OPENAI_API_KEY"
# TUI-brain parity (entry-agent orchestrator + squad).
export HEARTBIT_ORCHESTRATOR=1
export HEARTBIT_SUB_AGENT_MAX_TURNS="${HEARTBIT_SUB_AGENT_MAX_TURNS:-200}"
export HEARTBIT_MAX_TURNS="${HEARTBIT_MAX_TURNS:-100}"
# Reasoning models burn tokens before content; 4096 truncates (smoke 2026-09-22).
export HEARTBIT_MAX_TOKENS="${HEARTBIT_MAX_TOKENS:-8192}"
# Koyeb cold start can exceed the OpenAiCompat 120s default.
export HEARTBIT_OPENAI_TIMEOUT_SECS="${HEARTBIT_OPENAI_TIMEOUT_SECS:-300}"
export HEARTBIT_PROMPT_CACHING=0
# Prefer the portable slim musl binary if present.
MUSL="$ROOT/target/x86_64-unknown-linux-musl/release/heartbit"
ALT_MUSL="/home/pleclech/projects/heartbit/target/x86_64-unknown-linux-musl/release/heartbit"
if [[ -x "$MUSL" ]]; then
  export HEARTBIT_BIN="$MUSL"
elif [[ -x "$ALT_MUSL" ]]; then
  export HEARTBIT_BIN="$ALT_MUSL"
else
  echo "No musl heartbit binary. Run: benchmarks/terminal-bench-2/scripts/build_musl.sh" >&2
  exit 1
fi
export HEARTBIT_INSTALL_MODE=prebuilt
export HEARTBIT_BIN_STATIC=1

MODEL="${HEARTBIT_TB2_MODEL:-qwen/qwen3.8-27b}"
JOB_NAME="${HEARTBIT_TB2_JOB_NAME:-tb2-qwen-orchestrator}"

# Warm the endpoint (Koyeb cold start) so the first TB2 trial doesn't burn its
# OpenAiCompat timeout on a sleeping instance.
echo "warming $HEARTBIT_BASE_URL …"
curl -sS -m 240 -o /dev/null -w "warm models HTTP %{http_code} in %{time_total}s\n" \
  -H "Authorization: Bearer $HEARTBIT_API_KEY" \
  "$HEARTBIT_BASE_URL/models" || true

INCLUDE=()
if [[ "${1:-}" == "--full" ]]; then
  shift || true
  echo "FULL run: all TB2 tasks (expect many hours)"
else
  # Tasks that previously scored 1.0 under gpt-5.5 — cheap capability smoke.
  # Harbor 0.13 expects the dataset-qualified name.
  TASKS="${TASKS:-terminal-bench/build-cython-ext terminal-bench/constraints-scheduling}"
  for t in $TASKS; do
    # Accept short names too.
    case "$t" in
      terminal-bench/*) INCLUDE+=(-i "$t") ;;
      *) INCLUDE+=(-i "terminal-bench/$t") ;;
    esac
  done
  echo "SMOKE tasks: ${INCLUDE[*]}"
fi

echo "bin=$HEARTBIT_BIN"
echo "model=$MODEL orchestrator=$HEARTBIT_ORCHESTRATOR max_tokens=$HEARTBIT_MAX_TOKENS timeout=${HEARTBIT_OPENAI_TIMEOUT_SECS}s base=$HEARTBIT_BASE_URL"

harbor run \
  -d terminal-bench/terminal-bench-2 \
  --agent-import-path heartbit_tb2.agent:HeartbitAgent \
  -m "$MODEL" \
  -n 1 \
  -k 1 \
  -y \
  --job-name "$JOB_NAME" \
  -o "$TB2/jobs" \
  "${INCLUDE[@]}" \
  "$@"
status=$?

# Opt-in post-run Langfuse ingest (host-side — never inside the musl jail).
# Set LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY (and optionally LANGFUSE_HOST).
if [[ $status -eq 0 && -n "${LANGFUSE_PUBLIC_KEY:-}" && -n "${LANGFUSE_SECRET_KEY:-}" ]]; then
  JOB_DIR="$TB2/jobs/$JOB_NAME"
  if [[ -d "$JOB_DIR" ]]; then
    echo "ingest → Langfuse ($JOB_DIR) …"
    "$TB2/scripts/ingest-langfuse.sh" "$JOB_DIR" || echo "warn: Langfuse ingest failed (harness run still OK)" >&2
  fi
fi
exit "$status"