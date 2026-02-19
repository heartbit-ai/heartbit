# Heartbit

Multi-agent enterprise runtime in Rust. Orchestrator spawns sub-agents that execute LLM-powered reasoning loops with parallel tool execution.

Two execution paths:
- **Standalone** — in-process via `tokio::JoinSet`, zero infrastructure
- **Durable** — replay-safe via [Restate](https://restate.dev/) workflows, crash-resilient

## Quick start

```bash
# Standalone mode (no config file needed)
export ANTHROPIC_API_KEY=sk-...
cargo run --release -p heartbit-cli -- "Analyze the Rust ecosystem"

# With OpenRouter
export OPENROUTER_API_KEY=sk-...
cargo run --release -p heartbit-cli -- "Analyze the Rust ecosystem"
```

Default agents (researcher, analyst, writer) are created automatically when no config file is provided.

## CLI

```
heartbit [run|serve|submit|status|approve|result] <args>
heartbit <task>                  # shorthand for 'run'
```

| Command | Description |
|---------|-------------|
| `run <task>` | Execute in standalone mode (no Restate) |
| `serve` | Start the Restate HTTP worker |
| `submit <task>` | Submit to Restate for durable execution |
| `status <id>` | Query workflow status |
| `approve <id>` | Send approval signal to a child agent workflow |
| `result <id>` | Get result of a completed workflow |

**Flags:**

| Flag | Commands | Description |
|------|----------|-------------|
| `--config <path>` | all | Path to `heartbit.toml` |
| `--approve` | `run`, `submit` | Enable human-in-the-loop approval |
| `--bind <addr>` | `serve` | Worker bind address (default: `0.0.0.0:9080`) |
| `--restate-url <url>` | `submit`, `status`, `approve`, `result` | Restate ingress URL |

## Configuration

```toml
[provider]
name = "anthropic"                    # or "openrouter"
model = "claude-sonnet-4-20250514"

[provider.retry]                      # optional: retry transient failures
max_retries = 3
base_delay_ms = 500
max_delay_ms = 30000

[orchestrator]
max_turns = 10
max_tokens = 4096

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
context_strategy = { type = "sliding_window", max_tokens = 100000 }
# context_strategy = { type = "summarize", threshold = 80000 }
# context_strategy = { type = "unlimited" }

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

[restate]
endpoint = "http://localhost:9070"

[telemetry]
otlp_endpoint = "http://localhost:4317"
service_name = "heartbit"
```

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | For anthropic provider | Anthropic API key |
| `OPENROUTER_API_KEY` | For openrouter provider | OpenRouter API key |
| `HEARTBIT_PROVIDER` | No | Force provider (`anthropic` / `openrouter`) |
| `HEARTBIT_MODEL` | No | Override model name |
| `RUST_LOG` | No | Tracing filter (e.g. `info`, `debug`) |

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

**Orchestrator** — an `AgentRunner` with a `DelegateTaskTool`. Dispatches sub-agents in parallel. Sub-agents do NOT spawn further agents (flat hierarchy).

**AgentRunner** — the ReAct loop: LLM call -> tool execution -> repeat until done or max turns. Tools execute in parallel via `JoinSet`. Panicked tasks produce error results without crashing the loop.

**Tool trait** — `definition() -> ToolDefinition` + `execute(Value) -> Future<Result<ToolOutput>>`. Input validated against JSON Schema before dispatch.

**MCP client** — Streamable HTTP client (protocol `2025-03-26`). `McpClient::connect(url)` discovers tools automatically.

**LLM providers** — `AnthropicProvider` and `OpenRouterProvider` with SSE streaming. `RetryingProvider` wraps any provider with exponential backoff on 429/5xx. `BoxedProvider` for type-erased usage.

### Cross-agent coordination

**Blackboard** — shared `Key -> Value` store. Sub-agents get `blackboard_read`, `blackboard_write`, `blackboard_list` tools. After each sub-agent completes, its result is written to `"agent:{name}"`.

**Memory** — `Memory` trait with `store`, `recall`, `update`, `forget`. Implementations: `InMemoryStore`, `PostgresMemoryStore`. Agents get 5 memory tools including `memory_consolidate` (MemGPT pattern). Recall scoring uses Park et al. composite: `recency + importance + relevance`.

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

let mut orchestrator = Orchestrator::builder(provider)
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

## Docker

```bash
docker compose up -d
```

Services:
- `restate` — Restate server (ports 8080 ingress, 9070 admin)
- `heartbit` — worker (port 9080)

Mount your `heartbit.toml` and set `ANTHROPIC_API_KEY` / `OPENROUTER_API_KEY` in the environment.

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

375 tests. TDD mandatory — red/green/refactor for every feature.

## Project structure

```
crates/
  heartbit/           # Library crate
    src/
      agent/          # AgentRunner, Orchestrator, context management, blackboard
      llm/            # LlmProvider trait, Anthropic, OpenRouter, retry, SSE parser
      memory/         # Memory trait, InMemoryStore, PostgresMemoryStore, scoring
      tool/           # Tool trait, MCP client, validation
      workflow/       # Restate workflows, services, objects
      config.rs       # TOML configuration
      error.rs        # Error types (thiserror)
      lib.rs          # Public API re-exports
  heartbit-cli/       # Binary crate
    src/
      main.rs         # CLI entry point, standalone runner
      serve.rs        # Restate HTTP worker
      submit.rs       # Restate task submission
deploy/               # Systemd service units
Dockerfile            # Multi-stage build
docker-compose.yml    # Restate + worker
```
