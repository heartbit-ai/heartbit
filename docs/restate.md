# Durable Execution with Restate

Crash-resilient agent workflows via [Restate SDK 0.8](https://restate.dev/).

## Components

| Component | Type | Description |
|-----------|------|-------------|
| `AgentService` | `#[restate_sdk::service]` | `llm_call` + `tool_call` activities |
| `AgentWorkflow` | `#[restate_sdk::workflow]` | Durable ReAct loop with replay |
| `OrchestratorWorkflow` | workflow | Delegates to child `AgentWorkflow` instances |
| `Blackboard` | virtual object | Shared state across agents |
| `Budget` | virtual object | Token tracking |
| `CircuitBreaker` | virtual object | LLM fault tolerance |
| `Scheduler` | virtual object | Recurring task scheduling |

## Quick Start

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

## What Restate Provides

- **Durable execution** with replay — agent loops survive crashes
- **Exactly-once** tool execution — tools don't re-execute on recovery
- **Token budget tracking** — persistent budget across workflow restarts
- **Circuit breaker** — LLM fault tolerance with automatic recovery
- **Recurring tasks** — scheduler for periodic agent invocations

## Configuration

```toml
[restate]
endpoint = "http://localhost:9070"
```

## Docker

```bash
docker compose up -d
```

Services:
- `restate` — Restate server (ports 8080 ingress, 9070 admin)
- `heartbit` — worker (port 9080)

## Feature Flag

Restate support requires the `restate` feature flag:

```bash
cargo build --features restate
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `heartbit serve` | Start the Restate HTTP worker |
| `heartbit submit <task>` | Submit task for durable execution |
| `heartbit status <id>` | Query workflow status |
| `heartbit approve <id>` | Send approval signal |
| `heartbit result <id>` | Get completed workflow result |
