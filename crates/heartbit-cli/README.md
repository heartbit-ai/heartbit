# heartbit-cli — the Heartbit binary

The `heartbit` CLI is the operator-facing entry point for the Heartbit
multi-tenant runtime. Library users should target
[`heartbit-core`](../heartbit-core/README.md); use this crate when you want
to **run** agents — locally, durably, or as a long-lived multi-tenant
service.

## Install

### From source

```bash
cargo install --git https://github.com/heartbit-ai/heartbit heartbit-cli
```

### Pre-built binaries

```bash
curl -fsSL https://raw.githubusercontent.com/heartbit-ai/heartbit/main/install.sh | bash
```

### Docker

```bash
docker pull ghcr.io/heartbit-ai/heartbit:latest
docker compose up -d   # Restate + worker + Kafka
```

### Prerequisites

Building from source requires: Rust stable, `cmake`, `libssl-dev`,
`pkg-config` (for `rdkafka`).

## Subcommands

```
heartbit [run|chat|serve|daemon|submit|status|approve|result|templates|skills|init] <args>
heartbit <task>                  # shorthand for `run`
```

| Command | Description |
|---|---|
| `run <task>` | Single-shot agent run in standalone mode (env-config; uses `ANTHROPIC_API_KEY` etc.) |
| `chat` | Interactive multi-turn REPL with the default agent |
| `serve` | Restate worker for **durable** agent execution |
| `daemon` | Kafka-backed multi-tenant runtime with HTTP API + SSE event streaming |
| `submit <task>` | Submit a task for durable execution |
| `status <id>` | Query workflow status |
| `approve <id>` | Send approval signal (human-in-the-loop) |
| `result <id>` | Get completed workflow result |
| `templates list\|show` | Browse built-in agent templates |
| `skills list\|show` | Browse built-in domain skills |
| `init <template>` | Generate a starter `heartbit.toml` from a template |
| `eval` | Run eval suites against agents |

| Flag | Description |
|---|---|
| `--config <path>` | Path to `heartbit.toml` |
| `--approve` | Enable human-in-the-loop approval for tool execution |
| `-v`, `--verbose` | Emit agent events as JSON to stderr |

## Quick start

```bash
# Standalone, no config
export ANTHROPIC_API_KEY=sk-...
heartbit "Analyze the Rust ecosystem"

# Interactive chat
heartbit chat

# With a config (orchestrator + sub-agents, MCP servers, guardrails…)
heartbit --config heartbit.toml run "Plan our next sprint"
```

## Three execution paths

| Path | Infrastructure | Use case |
|---|---|---|
| **Standalone** | None (in-process) | CLI tasks, scripts, library embedding |
| **Durable** (`serve` / `submit`) | [Restate](https://restate.dev/) server | Crash-resilient workflows, exactly-once tool execution |
| **Daemon** | Kafka + Axum (+ optional Postgres) | Long-running services, cron jobs, event-driven multi-tenant tasks |

## Daemon mode

`heartbit daemon` brings up a Kafka-backed multi-tenant runtime:

- Kafka consumer loop dispatches commands to agent workers.
- Axum HTTP API for `submit` / `status` / `cancel` / `events` (SSE) plus
  the A2A agent card at `GET /.well-known/agent.json`.
- Cron scheduler for recurring tasks.
- Heartbeat pulse + idle-session memory consolidation.
- WebSocket + chat-channel adapters (Telegram / Discord / Slack), each
  enforcing tenant boundaries.

### Multi-tenancy

A single daemon serves multiple users with per-request isolation:

- **JWT/JWKS auth** — `JwksClient` (5-minute TTL, auto-refetch on key
  rotation), `JwtValidator` for RS256 tokens. `UserContext` carries
  `user_id`, `tenant_id`, and `roles` through every request.
- **Per-user memory namespacing** — daemon wraps the memory store with
  `NamespacedMemory` using `tenant:{tid}:user:{uid}` prefix.
- **Per-user workspace isolation** — workspace root becomes
  `{base}/{tenant_id}/{user_id}/`; path traversal prevention enforced.
- **Tenant-scoped store queries** — `TaskStore::list_filtered()` and
  `stats()` push tenant filters to the store level.
- **Audit trail enrichment** — `AuditRecord` carries `user_id`,
  `tenant_id`, and `delegation_chain`.
- **Dynamic MCP token injection** (RFC 8693) — `TokenExchangeAuthProvider`
  exchanges the user's JWT for a scoped MCP token, cached in-memory with
  TTL.

### Channel adapters

When configured, the daemon hosts:

- **Telegram bot** — DM support, streaming responses, multimodal input
  (photos / voice / documents), idle-session pruning.
- **Discord bot** — channel- or DM-bound agent.
- **Slack bot** — channel- or DM-bound agent.
- **WebSocket** (`/ws`) — interactive sessions with tenant-scoped session
  store (`InMemorySessionStore` or `PostgresSessionStore`).

### Sensor pipeline

`heartbit-gateway` (separate binary, see
[`crates/heartbit-gateway`](../heartbit-gateway)) ingests events from 7
sources (RSS, Email/JMAP, Webhook, Weather, Audio, Image, MCP) and
publishes to Kafka commands consumed by the daemon. Triage, deduplication,
and story correlation happen in the gateway. See
[`docs/sensors.md`](../../docs/sensors.md).

## Configuration

The CLI reads `heartbit.toml` (path via `--config`). Minimal example:

```toml
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "researcher"
description = "Research specialist"
system_prompt = "You research."

[[agents]]
name = "writer"
description = "Writing specialist"
system_prompt = "You write."
```

A daemon-mode config additionally adds a `[daemon]` section (Kafka,
Postgres, JWT auth, memory namespacing, sensor wiring). See
`daemon-dev.toml.example` for a fully-annotated reference.

Full TOML schema and environment variables: [`docs/configuration.md`](../../docs/configuration.md).

## Feature flags

| Feature | What it enables |
|---|---|
| (default) | Agent runner, orchestrator, providers, tools, memory, config |
| `daemon` | Daemon command + Kafka + Axum HTTP API + cron + Prometheus |
| `local-embedding` | Offline ONNX embeddings (fastembed) — no API keys |

The CLI links against the umbrella `heartbit` crate with `features = ["full"]`,
so `restate`, `postgres`, `a2a`, `telegram`, `discord`, `slack`, `vault`,
`sensor` are all enabled in the binary by default.

## Operations

- **Logs / metrics** — Tracing via `tracing-subscriber`; OpenTelemetry
  export via OTLP when `[telemetry]` is configured. Daemon exposes
  Prometheus metrics on `/metrics`.
- **Cost tracking** — Per-model pricing for Claude 4 / 3.5 / 3
  generations; `output.estimated_cost_usd` and per-tier accounting for
  cascading providers.
- **Permission system** — `--approve` flag enables human-in-the-loop with
  persistent learned rules (allow / deny / always-allow / always-deny) and
  glob patterns.
- **Sandbox** — Linux Landlock filesystem ACLs for `bash` subprocesses
  (unprivileged, works in Docker) when the `sandbox` feature is enabled.
- **Vault** — AES-256-GCM encrypted credential store with Argon2 KDF
  (`~/.heartbit/vault.enc`).

## Architecture

For a deeper architecture overview of the platform side (daemon /
gateway / multi-tenancy / Kafka topics / Postgres schema), see
[`docs/platform.md`](../../docs/platform.md).
