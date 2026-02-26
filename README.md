[![CI](https://github.com/heartbit-ai/heartbit/actions/workflows/ci.yml/badge.svg)](https://github.com/heartbit-ai/heartbit/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)

# Heartbit

Multi-agent enterprise runtime in Rust. Orchestrator spawns sub-agents that execute LLM-powered reasoning loops with parallel tool execution.

Three execution paths:
- **Standalone** — in-process via `tokio::JoinSet`, zero infrastructure
- **Durable** — replay-safe via [Restate](https://restate.dev/) workflows, crash-resilient
- **Daemon** — long-running Kafka-backed task execution with HTTP API and SSE streaming

## Installation

### Pre-built binaries

```bash
curl -fsSL https://raw.githubusercontent.com/heartbit-ai/heartbit/main/install.sh | bash
```

### From source

```bash
cargo install --git https://github.com/heartbit-ai/heartbit heartbit-cli
```

### Docker

```bash
docker pull ghcr.io/heartbit-ai/heartbit:latest
```

### Prerequisites

Building from source requires:
- Rust stable (latest)
- cmake, libssl-dev, pkg-config (for rdkafka)

## Quick start

```bash
# Standalone mode (no config file needed)
export ANTHROPIC_API_KEY=sk-...
cargo run --release -p heartbit-cli -- "Analyze the Rust ecosystem"

# With OpenRouter
export OPENROUTER_API_KEY=sk-...
cargo run --release -p heartbit-cli -- "Analyze the Rust ecosystem"

# Interactive chat
cargo run --release -p heartbit-cli -- chat
```

Without a config file, a single agent runs with 14 built-in tools (bash, read, write, edit, grep, etc.).

## CLI

```
heartbit [run|chat|serve|daemon|submit|status|approve|result] <args>
heartbit <task>                  # shorthand for 'run'
```

| Command | Description |
|---------|-------------|
| `run <task>` | Execute in standalone mode (no Restate) |
| `chat` | Start an interactive chat session (multi-turn REPL) |
| `serve` | Start the Restate HTTP worker |
| `daemon` | Run as a long-lived Kafka-backed daemon with HTTP API |
| `submit <task>` | Submit to Restate for durable execution |
| `status <id>` | Query workflow status |
| `approve <id>` | Send approval signal to a child agent workflow |
| `result <id>` | Get result of a completed workflow |

**Flags:**

| Flag | Commands | Description |
|------|----------|-------------|
| `--config <path>` | all | Path to `heartbit.toml` |
| `--approve` | `run`, `chat`, `submit` | Enable human-in-the-loop approval |
| `-v`, `--verbose` | `run`, `chat`, `daemon` | Emit structured agent events as JSON to stderr |
| `--bind <addr>` | `serve`, `daemon` | Bind address (serve: `0.0.0.0:9080`, daemon: `127.0.0.1:3000`) |
| `--restate-url <url>` | `submit`, `status`, `approve`, `result` | Restate ingress URL |

## Configuration

```toml
[provider]
name = "anthropic"                    # or "openrouter"
model = "claude-sonnet-4-20250514"
prompt_caching = true                 # Anthropic only; default false

[provider.retry]                      # optional: retry transient failures
max_retries = 3
base_delay_ms = 500
max_delay_ms = 30000

[orchestrator]
max_turns = 10
max_tokens = 4096
run_timeout_seconds = 300             # wall-clock deadline for the entire run

[[agents]]
name = "researcher"
description = "Research specialist"
system_prompt = "You are a research specialist."
mcp_servers = ["http://localhost:8000/mcp"]

# All optional:
max_turns = 20                        # override orchestrator default
max_tokens = 16384
tool_timeout_seconds = 60
max_tool_output_bytes = 16384
run_timeout_seconds = 120             # per-agent wall-clock deadline
summarize_threshold = 80000
context_strategy = { type = "sliding_window", max_tokens = 100000 }
# context_strategy = { type = "summarize", threshold = 80000 }
# context_strategy = { type = "unlimited" }

# MCP server with authentication (alternative to bare URL)
# mcp_servers = [{ url = "http://localhost:8000/mcp", auth_header = "Bearer tok_xxx" }]

# Per-agent LLM provider override (optional)
[agents.provider]
name = "anthropic"
model = "claude-opus-4-20250514"
prompt_caching = true

# Structured JSON output (optional)
[agents.response_schema]
type = "object"
[agents.response_schema.properties.score]
type = "number"
[agents.response_schema.properties.summary]
type = "string"

[[agents]]
name = "writer"
description = "Writing specialist"
system_prompt = "You are a writing specialist."

# Optional sections
[memory]
type = "in_memory"                    # or: type = "postgres", database_url = "..."

[knowledge]
chunk_size = 1000                     # max bytes per chunk (default: 1000)
chunk_overlap = 200                   # overlap bytes between chunks (default: 200)

[[knowledge.sources]]
type = "file"
path = "README.md"

[[knowledge.sources]]
type = "glob"
pattern = "docs/**/*.md"

[[knowledge.sources]]
type = "url"
url = "https://docs.example.com/api"

[restate]
endpoint = "http://localhost:9070"

[daemon]
bind = "127.0.0.1:3000"            # HTTP API bind address
max_concurrent_tasks = 4            # bounded concurrency

[daemon.kafka]
brokers = "localhost:9092"
consumer_group = "heartbit-daemon"  # default
commands_topic = "heartbit.commands"
events_topic = "heartbit.events"

[[daemon.schedules]]
name = "daily-review"
cron = "0 0 9 * * *"               # 6-field cron (sec min hr dom mon dow)
task = "Review yesterday's work"

[telemetry]
otlp_endpoint = "http://localhost:4317"
service_name = "heartbit"
```

## Environment variables

When running without a config file, the CLI reads these environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | Anthropic API key (required for anthropic provider) |
| `OPENROUTER_API_KEY` | — | OpenRouter API key (required for openrouter provider) |
| `HEARTBIT_PROVIDER` | auto-detect | Force provider (`anthropic` / `openrouter`) |
| `HEARTBIT_MODEL` | `claude-sonnet-4-20250514` | Override model name |
| `HEARTBIT_MAX_TURNS` | `50` (`run`) / `200` (`chat`) | Max agent turns |
| `HEARTBIT_PROMPT_CACHING` | `false` | Enable Anthropic prompt caching (`1` or `true`) |
| `HEARTBIT_SUMMARIZE_THRESHOLD` | `80000` | Token count to trigger context summarization |
| `HEARTBIT_MAX_TOOL_OUTPUT_BYTES` | `32768` | Max bytes per tool output before truncation |
| `HEARTBIT_TOOL_TIMEOUT` | `120` | Tool execution timeout in seconds |
| `HEARTBIT_MCP_SERVERS` | — | Comma-separated MCP server URLs |
| `EXA_API_KEY` | — | Exa AI API key (for `websearch` built-in tool) |
| `RUST_LOG` | — | Tracing filter (e.g. `info`, `debug`) |

## Architecture

```
                    heartbit-cli (bin)
                         |
                    heartbit (lib)
                    /           \
          Standalone            Durable (Restate)
          AgentRunner            AgentWorkflow
          Orchestrator           OrchestratorWorkflow
          tokio::JoinSet         Restate SDK 0.8
```

### Key components

**Orchestrator** — an `AgentRunner` with two delegation tools: `delegate_task` (independent parallel subtasks) and `form_squad` (collaborative subtasks with a shared private blackboard). Sub-agents do NOT spawn further agents (flat hierarchy). Squads auto-enable when >= 2 agents are registered; disable with `orchestrator.enable_squads = false`.

**AgentRunner** — the ReAct loop: LLM call -> tool execution -> repeat until done or max turns. Tools execute in parallel via `JoinSet`. Panicked tasks produce error results without crashing the loop.

**Tool trait** — `definition() -> ToolDefinition` + `execute(Value) -> Future<Result<ToolOutput>>`. Input validated against JSON Schema before dispatch.

**MCP client** — Streamable HTTP client (protocol `2025-03-26`). `McpClient::connect(url)` discovers tools automatically. Supports optional `auth_header` for authenticated servers.

**LLM providers** — `AnthropicProvider` and `OpenRouterProvider` with SSE streaming. `RetryingProvider` wraps any provider with exponential backoff on 429/5xx. `BoxedProvider` for type-erased usage. Per-agent provider overrides allow routing different agents to different models.

**Prompt caching** — `AnthropicProvider::with_prompt_caching()` sets `cache_control: {"type": "ephemeral"}` on system messages and tool definitions. Cache reads cost 10% of input rate, cache writes 125%.

### Guardrails

`Guardrail` trait with four async hooks for intercepting the agent loop (standalone path only):

- `pre_llm(&mut CompletionRequest)` — modify or validate requests before LLM calls
- `post_llm(&CompletionResponse) -> GuardAction` — allow or deny LLM responses
- `pre_tool(&ToolCall) -> GuardAction` — allow or deny individual tool calls
- `post_tool(&ToolCall, &mut ToolOutput)` — inspect or modify tool outputs

Registered as `Vec<Arc<dyn Guardrail>>` — first `Deny` wins. Denied `post_llm` responses insert a synthetic assistant placeholder to maintain alternating message roles.

### Agent events

13 structured `AgentEvent` variants emitted via `OnEvent` callback:

`RunStarted`, `TurnStarted`, `LlmResponse`, `ToolCallStarted`, `ToolCallCompleted`, `ApprovalRequested`, `ApprovalDecision`, `SubAgentsDispatched`, `SubAgentCompleted`, `ContextSummarized`, `RunCompleted`, `GuardrailDenied`, `RunFailed`

Use `--verbose` to emit events as JSON to stderr.

### Cost tracking

`estimate_cost(model, usage) -> Option<f64>` returns estimated USD cost for known models (Claude 4, 3.5, and 3 generations, including OpenRouter aliases). Accounts for cache read/write token rates. Displayed in CLI output after each run.

### Built-in tools

14 tools available by default in env-based mode (no config file):

| Tool | Description |
|------|-------------|
| `bash` | Execute bash commands. Working directory persists between calls. Default timeout: 120s, max: 600s. |
| `read` | Read a file with line numbers. Detects binary files. Max size: 256 KB. |
| `write` | Write content to a file. Creates parent directories. Read-before-write guard. |
| `edit` | Replace an exact string in a file (must appear exactly once). Read-before-write guard. |
| `patch` | Apply unified diff patches to one or more files. Single-pass hunk application. |
| `glob` | Find files matching a glob pattern. Skips hidden files. |
| `grep` | Search file contents with regex. Uses `rg` when available, falls back to built-in. |
| `list` | List directory contents as an indented tree. Skips common build artifacts. |
| `webfetch` | Fetch content from a URL via HTTP GET. Supports text, markdown, HTML. Max: 5 MB. |
| `websearch` | Search the web via Exa AI. Requires `EXA_API_KEY`. |
| `todowrite` | Write/replace the full todo list. Only 1 item in progress at a time. |
| `todoread` | Read the current todo list. |
| `skill` | Load skill definitions from `SKILL.md` files. |
| `question` | Ask the user structured questions (only when `on_question` callback is set). |

### Cross-agent coordination

**Blackboard** — shared `Key -> Value` store. Sub-agents get `blackboard_read`, `blackboard_write`, `blackboard_list` tools. After each sub-agent completes, its result is written to `"agent:{name}"`.

**Memory** — `Memory` trait with `store`, `recall`, `update`, `forget`. Implementations: `InMemoryStore`, `PostgresMemoryStore`. Agents get 5 memory tools including `memory_consolidate` (MemGPT pattern). Recall scoring uses Park et al. composite: `recency + importance + relevance`.

**Knowledge** — `KnowledgeBase` trait for document retrieval. `InMemoryKnowledgeBase` provides keyword search over indexed chunks. Loaders: file, glob, URL (with HTML stripping). Paragraph-aware chunking with configurable size and overlap. Agents get a `knowledge_search` tool. Standalone path only.

### Context management

- `Unlimited` — no trimming (default)
- `SlidingWindow { max_tokens }` — keeps first message + recent messages within budget; tool use/result pairs kept together
- `Summarize { threshold }` — LLM-generated summary injected when context exceeds threshold

### Structured output

Set `response_schema` (JSON Schema) on an agent. A synthetic `__respond__` tool is injected and `tool_choice` forced to `Any`. The agent calls `__respond__` to produce structured JSON in `AgentOutput::structured`.

### Human-in-the-loop

`--approve` flag enables interactive approval before each tool execution round. Denied tools receive error results — the LLM can adjust and retry. In Restate path, approval uses per-turn promise keys.

### Streaming

`on_text` callback receives text deltas as they arrive from the LLM. Both Anthropic and OpenRouter providers implement SSE streaming. Sub-agents don't stream — only the orchestrator.

### Run timeout

`run_timeout_seconds` sets a wall-clock deadline for the entire agent run. If exceeded, the run stops with `Error::RunTimeout` and returns partial token usage. Configurable at both orchestrator and per-agent level.

### OpenTelemetry

Add a `[telemetry]` section to your config to export traces via OTLP. Works with all commands (`run`, `chat`, `serve`). When absent, a simple `tracing_subscriber::fmt` subscriber is used instead.

## Library usage

```rust
use std::sync::Arc;
use std::time::Duration;
use heartbit::{
    AnthropicProvider, BoxedProvider, RetryingProvider,
    InMemoryBlackboard, Blackboard, Orchestrator, SubAgentConfig,
    ContextStrategy, McpClient,
};

let provider = Arc::new(BoxedProvider::new(
    RetryingProvider::with_defaults(
        AnthropicProvider::new(api_key, "claude-sonnet-4-20250514")
    )
));

let blackboard: Arc<dyn Blackboard> = Arc::new(InMemoryBlackboard::new());

let tools = McpClient::connect("http://localhost:8000/mcp")
    .await?.into_tools();

let mut orchestrator = Orchestrator::builder(provider.clone())
    .sub_agent_full(SubAgentConfig {
        name: "researcher".into(),
        description: "Research specialist".into(),
        system_prompt: "You research.".into(),
        tools,
        context_strategy: Some(ContextStrategy::SlidingWindow { max_tokens: 100_000 }),
        summarize_threshold: None,
        tool_timeout: Some(Duration::from_secs(30)),
        max_tool_output_bytes: Some(16384),
        max_turns: None,
        max_tokens: None,
        response_schema: None,
        guardrails: vec![],
        provider: None, // inherits orchestrator's provider; or Some(Arc::new(BoxedProvider::new(...)))
    })
    .sub_agent("writer", "Writing specialist", "You write.")
    .blackboard(blackboard)
    .on_text(Arc::new(|text| print!("{text}")))
    .build()?;

let output = orchestrator.run("Research the Rust ecosystem").await?;
println!("\nTokens: {} in / {} out", output.tokens_used.input_tokens,
    output.tokens_used.output_tokens);
```

## Durable execution (Restate)

```bash
# Start Restate + worker
docker compose up -d

# Register the worker with Restate
curl -X POST http://localhost:9070/deployments -H 'content-type: application/json' \
  -d '{"uri": "http://heartbit:9080"}'

# Submit a task
heartbit submit --config heartbit.toml "Analyze the Rust ecosystem"

# Check status
heartbit status <workflow-id>

# Approve tool execution (when --approve was used)
heartbit approve <child-workflow-id>

# Get result
heartbit result <workflow-id>
```

Restate provides: durable execution with replay, crash recovery, exactly-once tool execution, token budget tracking, circuit breaker for LLM providers, and recurring task scheduling.

## Daemon mode

Long-running Kafka-backed task execution with HTTP API and SSE event streaming.

```bash
# Start Kafka
docker compose up kafka -d

# Run the daemon
heartbit daemon --config heartbit.toml

# Submit a task
curl -X POST http://localhost:3000/tasks \
  -H 'Content-Type: application/json' \
  -d '{"task":"Analyze the codebase"}'

# List tasks
curl http://localhost:3000/tasks

# Stream events (SSE)
curl -N http://localhost:3000/tasks/<id>/events

# Cancel a task
curl -X DELETE http://localhost:3000/tasks/<id>
```

Features: bounded concurrency, per-task cancellation, cron scheduling, dual event delivery (Kafka durable + in-process SSE), graceful shutdown.

## Docker

```bash
docker compose up -d
```

Services:
- `restate` — Restate server (ports 8080 ingress, 9070 admin)
- `heartbit` — worker (port 9080)
- `kafka` — KRaft-mode Kafka broker (port 9092)

Mount your `heartbit.toml` and set `ANTHROPIC_API_KEY` / `OPENROUTER_API_KEY` in the environment.

## MCP via agentgateway

Instead of connecting to each MCP server individually, use [agentgateway](https://github.com/agentgateway/agentgateway) as a single aggregation layer. Heartbit connects to one endpoint, agentgateway fans out to all upstream MCP servers (local stdio + remote HTTP).

```
heartbit agent(s) → http://localhost:3000/mcp → agentgateway → filesystem, git, github, playwright, ...
```

```bash
# Start agentgateway with the bundled config
cd gateway && ./start.sh

# With config file
# heartbit.toml: mcp_servers = ["http://localhost:3000/mcp"]
cargo run --release -p heartbit-cli -- --config heartbit.toml run "your task"

# Without config file (env var)
HEARTBIT_MCP_SERVERS=http://localhost:3000/mcp cargo run --release -p heartbit-cli -- run "your task"
```

See `gateway/config.example.yaml` for the full list of MCP servers and setup instructions.

## Development

```bash
# Quality gate (must pass before every commit)
cargo fmt -- --check && cargo clippy -- -D warnings && cargo test

# Run tests only
cargo test

# Test MCP locally
npx -y supergateway \
  --stdio "npx -y @modelcontextprotocol/server-filesystem /tmp/mcp-test" \
  --outputTransport streamableHttp --port 8000
# Server at http://localhost:8000/mcp
```

2374 tests. TDD mandatory -- red/green/refactor for every feature.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## Project structure

```
crates/
  heartbit/           # Library crate
    src/
      agent/          # AgentRunner, Orchestrator, context management, blackboard
        events.rs     # AgentEvent enum (13 variants)
        guardrail.rs  # Guardrail trait (pre/post LLM/tool hooks)
      knowledge/      # KnowledgeBase trait, InMemoryKnowledgeBase, chunker, loaders
      llm/            # LlmProvider trait, Anthropic, OpenRouter, retry, SSE parser
        pricing.rs    # Cost estimation for known models
      memory/         # Memory trait, InMemoryStore, PostgresMemoryStore, scoring
      tool/           # Tool trait, MCP client, validation
        builtins/     # 14 built-in tools (bash, read, write, edit, patch, glob, grep, etc.)
      workflow/       # Restate workflows, services, objects
      config.rs       # TOML configuration
      error.rs        # Error types (thiserror)
      lib.rs          # Public API re-exports
  heartbit-cli/       # Binary crate
    src/
      main.rs         # CLI entry point, standalone runner
      serve.rs        # Restate HTTP worker
      submit.rs       # Restate task submission
tests/                # Integration tests
deploy/               # Systemd service units
Dockerfile            # Multi-stage build
docker-compose.yml    # Restate + worker
```
