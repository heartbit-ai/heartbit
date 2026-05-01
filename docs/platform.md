# Heartbit Platform Architecture

## What "the platform" means

Heartbit ships in two shapes:

1. **The framework** — [`heartbit-core`](../crates/heartbit-core), a library
   you `cargo add` and embed in your own application. Single-process, no
   infrastructure dependencies.
2. **The platform** — daemon mode in [`heartbit-cli`](../crates/heartbit-cli)
   plus the [`heartbit-gateway`](../crates/heartbit-gateway) ingestion
   binary, providing a multi-tenant Agents-as-a-Service runtime.

This document covers the platform. For the framework, see the top-level
[`README.md`](../README.md) and [docs.rs/heartbit-core](https://docs.rs/heartbit-core).

## Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                         heartbit-cli (bin)                          │
│  Commands: run | chat | serve | daemon | submit | status | approve  │
│            result | templates | skills | init | eval                │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────┐
│                  heartbit (umbrella crate, lib)                     │
│                                                                     │
│  ┌──────────────────┐ ┌────────────────┐ ┌───────────────────────┐ │
│  │   Standalone     │ │     Durable    │ │       Daemon          │ │
│  │                  │ │                │ │                       │ │
│  │  AgentRunner     │ │ AgentWorkflow  │ │  Kafka consumer       │ │
│  │  Orchestrator    │ │ OrchestratorWf │ │  Axum HTTP API + SSE  │ │
│  │  tokio::JoinSet  │ │ Restate SDK    │ │  WS + Telegram/etc.   │ │
│  │                  │ │                │ │  Cron + heartbeat     │ │
│  └────────┬─────────┘ └───────┬────────┘ └──────────┬────────────┘ │
│           │                   │                     │              │
│           ▼                   ▼                     ▼              │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                heartbit-core (framework)                      │ │
│  │  LlmProvider · Memory · Tool · Guardrail · Workspace · Eval  │ │
│  └───────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                       heartbit-gateway (bin)                        │
│  RSS · Email/JMAP · Webhook · Weather · Audio · Image · MCP         │
│  triage → dedup → stories → Kafka commands ──────► daemon           │
└─────────────────────────────────────────────────────────────────────┘
```

### Daemon

Long-running Kafka-backed task executor. Consumes a commands topic,
publishes events. Hosts:

- **Axum HTTP API** — `submit` / `status` / `cancel` / `events` (SSE),
  WebSocket interactive sessions (`/ws`), A2A agent card at
  `/.well-known/agent.json`, Prometheus metrics at `/metrics`.
- **Cron scheduler** — recurring tasks defined in config.
- **Heartbeat pulse** — periodic memory consolidation, idle-session
  pruning, todo digest.
- **Channel adapters** — Telegram / Discord / Slack bots, each with
  tenant-scoped sessions.

### Gateway

Separate binary (`heartbit-gateway`) that ingests events from external
sources, runs them through triage and story-correlation, and emits
Kafka commands consumed by the daemon. Sources:

| Sensor | Source | Trust level |
|---|---|---|
| `RssSensor` | RSS / Atom feeds | configurable |
| `EmailSensor` | JMAP mailboxes | configurable |
| `WebhookSensor` | HTTP webhooks (HMAC-verified) | configurable |
| `WeatherSensor` | Weather APIs | configurable |
| `AudioSensor` | Audio file ingestion + transcription | configurable |
| `ImageSensor` | Image ingestion + vision | configurable |
| `McpSensor` | MCP server polling | configurable |

See [`sensors.md`](sensors.md) for the full pipeline.

### Restate (durable execution)

Optional. The `serve` subcommand registers Restate
services / workflows / virtual objects:

- `agent_workflow` — durable ReAct loop (replay-safe).
- `orchestrator_workflow` — delegates to child `agent_workflow`s.
- Virtual objects: `blackboard`, `budget`, `circuit_breaker`,
  `scheduler`.

Tool calls become Restate Activities; the loop is replay-safe across
crashes. See [`restate.md`](restate.md).

### Kafka topics

| Topic | Producer | Consumer | Payload |
|---|---|---|---|
| `commands` (configurable) | gateway, HTTP API, cron | daemon | `DaemonCommand` (submit, cancel, …) |
| `events` (configurable) | daemon | dashboard, observers | `AgentEvent` JSON |

The daemon supports a **channel-mode** fallback (`CommandProducer` =
in-process `tokio::mpsc`) when `[daemon.kafka]` is omitted, useful for
local development.

### Postgres (optional)

When configured, Postgres backs:

- **`PostgresMemoryStore`** — persistent agent memory (pgvector for
  semantic recall).
- **`PostgresSessionStore`** — chat sessions (tenant-scoped via
  `WHERE tenant_id = $1`).
- **`PostgresTaskStore`** — task lifecycle + audit (with `user_id` /
  `tenant_id` columns).
- **`PostgresAuditTrail`** — append-only audit log of agent decisions,
  tool calls, and guardrail outcomes.

Migrations are append-only `ALTER TABLE` to preserve backward
compatibility.

## Multi-tenancy

A single daemon serves multiple users; isolation is enforced at every
boundary:

- **Authentication** — `[daemon.auth]` config selects between static
  bearer tokens (with rotation) and JWT/JWKS validation. Auth middleware
  extracts `UserContext { user_id, tenant_id, roles, raw_token }` from
  the token and threads it through the request.
- **Memory namespacing** — `NamespacedMemory` with prefix
  `tenant:{tid}:user:{uid}`. `recall()` always forces its own namespace,
  so agents cannot read across users via prompt injection.
  Institutional ("shared") memory is gated by the `shared_memory_read`
  tool and write access is role-gated via
  `DaemonMemoryConfig.shared_write_roles`.
- **Workspaces** — workspace root becomes `{base}/{tenant_id}/{user_id}/`.
  `WorkspaceTool` enforces path-boundary checks and rejects `..`-traversal
  and symlink escape.
- **Sessions** — `Session` carries `user_id` / `tenant_id`. WebSocket and
  channel-bound sessions enforce the same boundaries; session listing is
  tenant-scoped at the SQL layer.
- **Task store** — every task carries `user_id` / `tenant_id`. Listing,
  status, cancel, and stats endpoints reject unauthenticated callers and
  filter by tenant.
- **MCP token exchange (RFC 8693)** — when configured, the daemon
  exchanges the user's JWT for a scoped MCP server token via
  `TokenExchangeAuthProvider`. Cache is keyed by `(tenant_id, user_id)`
  and TTL-capped at 3600 s.
- **Audit trail** — `AuditRecord` records `user_id`, `tenant_id`, and the
  full `delegation_chain` (which agent acted on whose behalf, through
  which sub-agents).
- **Telegram pruning** — idle-session memory pruning is namespace-scoped
  (`tg:{user_id}` prefix) so cross-user memory is never deleted.

See [`memory.md`](memory.md) and the `[daemon.auth]` section of
[`configuration.md`](configuration.md) for full details.

## Running locally

```bash
docker compose up -d            # Restate + Kafka + Postgres
heartbit daemon --config daemon-dev.toml
```

`daemon-dev.toml` (see `daemon-dev.toml.example` at the repo root) wires
up the full stack: Anthropic provider, Kafka, Postgres, JWT-disabled
local auth, channel-mode fallbacks, prompt caching enabled.

The dashboard (separate repo) listens at `http://localhost:5173` and
talks to the daemon over the HTTP API + SSE.

## Production deployment

### Topology

- **N daemons** — horizontally scaled by Kafka consumer-group
  partitioning. Each daemon registers heartbeat in Postgres for liveness.
- **M gateways** — independently scaled; each gateway owns specific
  sensors (e.g. dedicated email-ingest gateway).
- **1+ Restate worker** — required only when using `serve` for durable
  execution; otherwise omit.
- **Postgres** — primary + read-replica recommended. Connection pool sized
  per daemon.
- **Kafka** — partition the commands topic by `tenant_id` for ordered
  per-tenant dispatch.

### Scaling

- Daemon work is per-task; horizontal scaling is bounded by Kafka
  partitions.
- LLM-bound throughput dominates; Anthropic / OpenRouter rate limits are
  the typical ceiling. Use the `RetryingProvider` (built-in) and
  `CascadingProvider` to absorb 429s.
- Memory recall is BM25 + composite scoring + optional pgvector cosine.
  pgvector with `ivfflat` indexes scales to millions of memories per
  namespace.

### Observability stack

- **Tracing** — OpenTelemetry OTLP export (`[telemetry]` config).
- **Metrics** — Prometheus on `/metrics`. Key counters:
  `heartbit_cascade_escalations_total`, `heartbit_guardrail_denied_total`,
  task lifecycle histograms.
- **Events** — `AgentEvent` SSE stream per task; archive to Kafka `events`
  topic for replay / audit.
- **Audit** — `PostgresAuditTrail` for compliance-grade retention.

### Postgres schema migrations

Schema initialization runs inline at store startup
(`PostgresMemoryStore`, `PostgresSessionStore`, `PostgresTaskStore`,
`PostgresAuditTrail`) using `CREATE TABLE IF NOT EXISTS` and additive
`ALTER TABLE ADD COLUMN` statements. New struct fields carry
`#[serde(default)]` for backward compatibility, so rolling upgrades are
safe.

## Cross-references

- [`daemon.md`](daemon.md) — daemon-specific config and behavior
- [`memory.md`](memory.md) — memory architecture
- [`sensors.md`](sensors.md) — sensor pipeline
- [`restate.md`](restate.md) — durable execution path
- [`telegram.md`](telegram.md) — Telegram bot specifics
- [`configuration.md`](configuration.md) — full TOML reference
