# Daemon Mode

Long-running Kafka-backed task execution with HTTP API, cron scheduling, and interactive channels.

## Overview

- **Kafka consumer** loop processes `DaemonCommand` messages
- **Axum HTTP API** — submit/list/cancel tasks, stream events via SSE
- **Cron scheduler** — 6-field cron expressions for recurring tasks
- **Heartbeat pulse** — periodic awareness loop that reads `HEARTBIT.md` standing orders, checks todos, submits tasks with idle backoff
- **Bounded concurrency** — `max_concurrent_tasks` limits parallel agent runs
- **WebSocket + Telegram** — interactive channels via `InteractionBridge`
- **Multi-tenant isolation** — JWT/JWKS authentication extracts `UserContext` per request
- **A2A Agent Card** — `GET /.well-known/agent.json` for agent discovery

## Quick Start

```bash
# Start Kafka
docker compose up kafka -d

# Run the daemon
heartbit daemon --config heartbit.toml

# Submit a task (with bearer token auth)
curl -X POST http://localhost:3000/tasks \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer $YOUR_API_KEY' \
  -d '{"task":"Analyze the codebase"}'

# Submit a task (with JWT auth — multi-tenant)
curl -X POST http://localhost:3000/tasks \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <jwt-token>' \
  -d '{"task":"Analyze the codebase"}'

# List tasks
curl http://localhost:3000/tasks \
  -H 'Authorization: Bearer $YOUR_API_KEY'

# Stream events (SSE)
curl -N http://localhost:3000/tasks/<id>/events

# Agent discovery
curl http://localhost:3000/.well-known/agent.json

# Cancel a task
curl -X DELETE http://localhost:3000/tasks/<id>
```

## Configuration

```toml
[daemon]
bind = "127.0.0.1:3000"
max_concurrent_tasks = 4

[daemon.kafka]
brokers = "localhost:9092"
consumer_group = "heartbit-daemon"
commands_topic = "heartbit.commands"
events_topic = "heartbit.events"

[[daemon.schedules]]
name = "daily-review"
cron = "0 0 9 * * *"               # 6-field cron (sec min hr dom mon dow)
task = "Review yesterday's work"
```

## Multi-Tenant Authentication

The daemon supports two auth modes (both can be active simultaneously):

| Mode | Config key | How it works |
|------|-----------|-------------|
| **Bearer tokens** | `daemon.auth.bearer_tokens` | Static API keys; supports multiple tokens for rotation |
| **JWT/JWKS** | `daemon.auth.jwks_url` | RS256 tokens verified against a JWKS endpoint; extracts `UserContext` |

JWT claim names are configurable:

```toml
[daemon.auth]
bearer_tokens = ["$YOUR_API_KEY"]
jwks_url = "https://idp.example.com/.well-known/jwks.json"
issuer = "https://idp.example.com"
audience = "heartbit-daemon"
user_id_claim = "sub"
tenant_id_claim = "tid"
roles_claim = "roles"
```

When JWT auth is configured, authenticated requests carry a `UserContext` through the entire lifecycle:
- **Memory** — wrapped with `NamespacedMemory` using `tenant:{tid}:user:{uid}` prefix
- **Workspace** — scoped to `{base}/{tenant_id}/{user_id}/`
- **Tasks** — filtered by authenticated tenant
- **Audit** — records include `user_id`, `tenant_id`, and `delegation_chain`

## Dynamic MCP Authentication

For multi-tenant MCP server access, the `AuthProvider` trait enables per-user token injection:

- **`StaticAuthProvider`** — returns the same auth header for all users (backward-compatible)
- **`TokenExchangeAuthProvider`** — implements RFC 8693 token exchange against an IdP, caching per-user tokens
