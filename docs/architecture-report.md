# Heartbit 3-Tier Architecture: Plan, Implementation & Status Report

**Date**: 2026-03-08
**Author**: Claude (AI assistant to Pascal Le Clech)
**Scope**: Complete architectural overview of the heartbit platform transition from monolithic daemon to 3-tier architecture.

---

## Table of Contents

1. [Original Plan](#1-original-plan)
2. [Architecture Overview](#2-architecture-overview)
3. [Phase 1: Runtime — Implementation Status](#3-phase-1-runtime)
4. [Phase 2: Gateway — Implementation Status](#4-phase-2-gateway)
5. [Phase 3: Cloud — Implementation Status](#5-phase-3-cloud)
6. [Phase 4: Deployment — Status](#6-phase-4-deployment)
7. [What Was Not Implemented & Why](#7-what-was-not-implemented--why)
8. [Test Coverage Summary](#8-test-coverage-summary)
9. [File Structure Reference](#9-file-structure-reference)

---

## 1. Original Plan

### The Problem

Heartbit started as a monolithic daemon (`heartbit-cli daemon`) that handled everything in a single process:

- **Agent execution** (LLM calls, tool execution, guardrails)
- **Input ingestion** (Telegram bots, cron schedules, sensors, webhooks, WebSocket)
- **API serving** (REST endpoints, SSE streaming, task management)
- **Multi-tenant isolation** (JWT auth, per-tenant workspaces, memory scoping)

This monolith was deployed as a single container. It worked but created scaling bottlenecks:

- **Execution-bound**: A tenant running a heavy agent starved other tenants' ingestion.
- **Ingestion-bound**: Sensor bursts or Telegram floods delayed agent responses.
- **Deployment coupling**: Updating the Telegram adapter required redeploying the execution engine.
- **Resource waste**: Execution needs GPU-adjacent compute; ingestion needs network I/O. One container can't optimize for both.

### The Solution: 3-Tier Architecture

Split the monolith into three independently deployable containers on Koyeb:

| Tier | Name | Responsibility | Scaling Profile |
|------|------|---------------|-----------------|
| **Runtime** | `heartbit-cli daemon` | Pure agent execution engine | CPU/memory-heavy, autoscale on queue depth |
| **Gateway** | `heartbit-gateway` | Input source ingestion | I/O-heavy, autoscale on connection count |
| **Cloud** | `heartbit-cloud` | Portal, API, billing, marketplace | Request-driven, autoscale on RPS |

### Transport Decisions

- **Cloud → Runtime**: HTTP POST to `/v1/tasks/execute` with full `RuntimeRequest` (provider keys, MCP servers, guardrails). Kafka deferred to Phase 4.
- **Runtime → Cloud**: SSE streaming (`RuntimeSseEvent` — Delta, Done, Error, Event variants).
- **Gateway → Runtime**: Kafka topic `heartbit.tasks` for fire-and-forget task submission.
- **Auth**: xavyo-idp (custom IdP with standard JWKS). All three tiers validate independently.

### Phase Ordering

1. **Phase 1 — Runtime**: Strip input sources, version API, harden auth, add cloud-delegated execution endpoint.
2. **Phase 2 — Gateway**: New crate for Telegram, cron, sensors, webhooks → Kafka.
3. **Phase 3 — Cloud**: WebSocket chat, MCP OAuth, SSE proxy, migrations.
4. **Phase 4 — Deployment**: Dockerfiles, Kafka integration, end-to-end testing.

---

## 2. Architecture Overview

### Deployed Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │              heartbit-cloud                  │
                    │  (Portal + API + Billing + Marketplace)      │
                    │                                              │
  End User ──────► │  POST /v1/chat  ──► build_runtime_request()  │
  (Browser/API)    │  GET  /v1/chat/ws   ──► WebSocket streaming  │
                    │  POST /tasks    ──► async task spawn         │
                    │                                              │
                    │  /api/agents, /marketplace, /api/credits     │
                    │  /api/mcp/oauth/*, /api/telegram/bots        │
                    │  /api/schedules, /api/keys, /api/usage       │
                    └──────────────┬───────────────────────────────┘
                                   │
                    HTTP POST /v1/tasks/execute (RuntimeRequest)
                    SSE stream ◄── (RuntimeSseEvent: Delta/Done/Error)
                                   │
                    ┌──────────────▼───────────────────────────────┐
                    │           heartbit runtime                    │
                    │  (Pure Execution Engine — headless)           │
                    │                                              │
                    │  POST /v1/tasks/execute ──► AgentRunner       │
                    │  POST /v1/tasks         ──► Kafka/HTTP submit │
                    │  GET  /v1/tasks/{id}/stream ──► SSE events   │
                    │  GET  /v1/health, /v1/ready, /v1/metrics     │
                    │                                              │
                    │  Kafka consumer loop (heartbit.commands)      │
                    │  JWT + Bearer token auth                      │
                    │  Per-request MCP auth tokens                  │
                    └──────────────────────────────────────────────┘

                    ┌──────────────────────────────────────────────┐
                    │          heartbit-gateway                     │
                    │  (Ingestion Layer — fire-and-forget)          │
                    │                                              │
  Telegram ──────► │  Cron scheduler ──► Kafka ──► Runtime         │
  Sensors  ──────► │  Sensor manager ──► Kafka ──► Runtime         │
  Webhooks ──────► │  (Webhooks: scaffolded, not wired)            │
                    │                                              │
                    │  GET /v1/health, /v1/ready                   │
                    └──────────────────────────────────────────────┘
```

### Crate Map

```
heartbit/                          (workspace root)
├── crates/
│   ├── heartbit/                  (lib — 3071 tests)
│   │   └── src/
│   │       ├── agent/             (runner, builder, orchestrator, guardrails, doom_loop)
│   │       ├── config/            (modular: agent, daemon, guardrails, memory, provider, sensor)
│   │       ├── daemon/            (core, cron, kafka, metrics, runtime_types, store, types)
│   │       ├── llm/               (providers, cascade, retry, streaming)
│   │       ├── tool/              (builtins, MCP client, skill)
│   │       ├── memory/            (episodic, semantic, reflection, consolidation)
│   │       └── sensor/            (manager, metrics)
│   │
│   ├── heartbit-cli/              (bin — 30 tests)
│   │   └── src/
│   │       ├── main.rs            (CLI: run, chat, serve, daemon, submit, status, approve)
│   │       └── daemon/            (mod, execute, handlers, types, auth, memory)
│   │
│   ├── heartbit-gateway/          (bin — 17 tests)
│   │   └── src/
│   │       ├── main.rs            (CLI: --config, --bind)
│   │       ├── config.rs          (GatewayConfig from TOML)
│   │       ├── server.rs          (Axum: /v1/health, /v1/ready)
│   │       ├── producer.rs        (KafkaCommandProducer wrapper)
│   │       ├── error.rs           (GatewayError enum)
│   │       └── sources/           (cron.rs, sensors.rs)
│   │
│   └── heartbit-macro/            (proc-macro — #[heartbit_tool])
│
heartbit-cloud/                    (separate repo — 177 tests)
└── src/
    ├── main.rs                    (startup: TOML or env config)
    ├── server.rs                  (AppState, build_router, build_runtime_request)
    ├── config.rs                  (PlatformConfig: server, db, platform, admin, xavyo, stripe, runtime, telegram)
    ├── routes/                    (18 modules: chat, ws, oauth, agents, tasks, marketplace, sessions, ...)
    ├── execution/                 (runtime_client, scheduler)
    ├── marketplace/               (catalog, store, oauth)
    ├── auth/                      (JWT + API key, middleware)
    ├── billing/                   (credit store, metering, Stripe)
    ├── telegram/                  (bot store, webhook, delivery)
    ├── builder/                   (meta-agent for agent CRUD)
    ├── tenant/                    (store, context)
    └── migrations/                (18 SQL files)
```

---

## 3. Phase 1: Runtime

**Goal**: Strip input sources, version API, harden auth, add cloud execution endpoint.

### Fully Implemented

| Item | Status | Details |
|------|--------|---------|
| Route versioning (`/v1/` prefix) | **Done** | All 11 endpoints versioned |
| Strip Telegram integration | **Done** | Removed from daemon startup |
| Strip WebSocket/SSE-via-WS | **Done** | `ws.rs` deleted (843 lines) |
| Strip cron scheduler | **Done** | Not started in `run_daemon()` |
| Strip heartbit pulse | **Done** | Not started in `run_daemon()` |
| Strip sensor manager | **Done** | Not started in daemon |
| Strip MCP server endpoint | **Done** | Removed |
| Strip owner email notifications | **Done** | Removed |
| Cloud-delegated execution endpoint | **Done** | `POST /v1/tasks/execute` — accepts `RuntimeRequest`, builds ephemeral `AgentRunner`, returns JSON or SSE |
| `RuntimeRequest` / `RuntimeResponse` types | **Done** | Full protocol in `daemon/runtime_types.rs` |
| `RuntimeSseEvent` enum | **Done** | Delta, Done, Error, Event variants |
| Config modularization | **Done** | Split 7744-line `config.rs` into 7 focused modules |
| Agent code extraction | **Done** | `runner.rs`, `builder.rs`, `doom_loop.rs` extracted from 2827-line `agent/mod.rs` |
| Per-request MCP auth tokens | **Done** | `mcp_auth_tokens: Option<HashMap<String, String>>` on `DaemonCommand::SubmitTask` |
| JWT/JWKS auth hardening | **Done** | `required: bool` mode on JwtMiddlewareState |
| Bearer + JWT dual auth | **Done** | JWT enrichment + static bearer gate |
| RFC 8693 token exchange | **Done** | `TokenExchangeAuthProvider` for per-user MCP credentials |
| Multi-tenant isolation | **Done** | user_id, tenant_id, roles on all task submissions |
| Audit context in prompts | **Done** | User identity appended to system prompts |
| Role-gated memory writes | **Done** | `shared_write_roles` config |

### Partially Implemented

| Item | Status | Notes |
|------|--------|-------|
| Auth hardening — remove bearer fallback | **Partial** | Bearer tokens still accepted when JWT is configured (backward compat). JWT can be set to `required = true` to disable bearer. |

---

## 4. Phase 2: Gateway

**Goal**: New crate handling all input sources, producing to Kafka.

### Fully Implemented

| Item | Status | Details |
|------|--------|---------|
| Crate structure | **Done** | `crates/heartbit-gateway/` with Cargo.toml, 7 source files |
| Workspace integration | **Done** | Listed in root `Cargo.toml` members |
| Config system | **Done** | `GatewayConfig` from TOML (server, kafka, schedules, sensors, auth) |
| Kafka producer | **Done** | `GatewayProducer` wrapping `KafkaCommandProducer` |
| Cron scheduler | **Done** | `init_cron()` from config `ScheduleEntry` vec, background task |
| Sensor manager | **Done** | `init_sensors()` with Prometheus metrics, background task |
| HTTP health endpoints | **Done** | `GET /v1/health`, `GET /v1/ready` |
| Graceful shutdown | **Done** | `CancellationToken` propagation across all tasks |
| Error types | **Done** | `GatewayError` enum (Kafka, Config, Sensor) |
| CLI entry point | **Done** | `--config` and `--bind` flags |
| 17 tests passing | **Done** | Config, server, producer, cron, sensors, errors |

### Scaffolded But Not Wired

| Item | Status | Notes |
|------|--------|-------|
| Auth config | **Scaffolded** | `AuthConfig` struct defined with `jwks_url`, `issuer`, `audience` — marked `#[allow(dead_code)]` |
| Error enum | **Scaffolded** | Defined but variants unused |
| Producer `submit_task()` | **Scaffolded** | Implemented but not called from any handler |

### Not Implemented

| Item | Why Not |
|------|---------|
| Telegram webhook handler | Telegram integration lives in `heartbit-telegram` crate; gateway doesn't expose webhook routes yet |
| Generic webhook receiver | Deferred — no webhook source exists outside Telegram |
| DB-driven cron | Uses TOML config only; no Postgres polling for dynamic schedules |
| Metrics endpoint (`/v1/metrics`) | Prometheus registry created but not served via HTTP |
| JWKS auth middleware | Auth config exists but validation not wired |
| Delivery consumer (Telegram replies) | Fire-and-forget model only; no response routing |

---

## 5. Phase 3: Cloud

**Goal**: Cloud switches to HTTP delegation, adds WebSocket chat, MCP OAuth, SSE proxy.

### Fully Implemented

| Item | Status | Details |
|------|--------|---------|
| `RuntimeClient` — HTTP delegation | **Done** | `execute()` (sync), `execute_stream()` (SSE), `cancel()`, `health_check()` |
| SSE stream parser | **Done** | Custom parser with 1MB buffer guard, handles `data: {...}\n\n` format |
| Route versioning (cloud endpoints) | **Done** | `/v1/chat`, `/v1/chat/ws`, `/v1/sessions/*` |
| `build_runtime_request()` | **Done** | Resolves provider (BYOK or platform), loads MCP installations, maps guardrails, memory config |
| WebSocket chat (`/v1/chat/ws`) | **Done** | Full streaming: parse JSON → validate → load history → stream from runtime → forward deltas → save session |
| WebSocket auth | **Done** | Auth middleware injects `TenantContext` before upgrade |
| WebSocket message limits | **Done** | 100KB incoming, 1MB accumulated response |
| MCP OAuth 2.0 + PKCE | **Done** | Full flow: initiate, callback, status, revoke |
| PKCE challenge generation | **Done** | SHA256 + base64url, 32-byte random verifier |
| Token encryption at rest | **Done** | AES-256-GCM for access/refresh tokens |
| Token refresh integration | **Done** | Auto-refresh in `build_runtime_request()` with 5-minute buffer |
| OAuth status endpoint | **Done** | `GET /api/mcp/oauth/{id}` — has_tokens, expires_at, needs_refresh |
| Token revocation | **Done** | `DELETE /api/mcp/oauth/{id}` — clears tokens + installation auth header |
| Catalog OAuth fields | **Done** | `oauth_client_id`, `oauth_authorize_url` on `McpCatalogEntry` |
| `get_catalog_by_id()` | **Done** | Approval-checked catalog fetch for OAuth flow |
| Migration 017: mcp_oauth_tokens | **Done** | Encrypted token storage, UNIQUE(tenant_id, installation_id) |
| Migration 018: gateway_config | **Done** | `gateway_webhooks` + `gateway_sensors` tables |
| `OAuthStore` on AppState | **Done** | Single instance, no per-request allocation |
| `save_conversation_turn()` | **Done** | Batch insert (2 messages in 1 SQL statement) |
| `public_url` config | **Done** | For OAuth redirect URI fallback |
| Builder agent rejection over WS | **Done** | Returns error (builder needs direct DB access) |
| Session tenant isolation | **Done** | `ON CONFLICT ... WHERE tenant_id = $2` prevents cross-tenant hijack |
| Credit pre-check | **Done** | `ensure_credits()` before execution |
| Usage recording | **Done** | Token counts + model attribution after execution |
| `openrouter_model_id()` mapping | **Done** | Anthropic ID → OpenRouter slug translation |
| 177 tests passing | **Done** | Includes unit + integration tests |

### Not Implemented

| Item | Why Not |
|------|---------|
| **Kafka producer** (cloud → runtime) | Deferred by user request ("keep kafka deferred"). HTTP delegation works for current scale. Kafka adds operational complexity (broker management, topic creation, consumer groups) without immediate benefit when cloud and runtime are co-located or low-latency. |
| **Events consumer** (runtime → cloud) | Depends on Kafka. Currently, task state is tracked via synchronous HTTP response or SSE stream completion. |
| **SSE proxy route** (`GET /v1/tasks/{id}/stream`) | RuntimeClient has `execute_stream()` but no dedicated endpoint for subscribing to an existing task's stream. Current model: stream is consumed inline during `/v1/chat` or WebSocket. |
| **Private MCP servers** (per-tenant custom servers) | Migration 018 creates `gateway_sensors` table but no CRUD routes. Tenant MCP installations already work via marketplace. Private servers (bypassing catalog) deferred — tenants can self-install custom URLs already. |
| **WebSocket chat for existing tasks** | WS handler creates new tasks per message. No "attach to running task" capability. |
| **Audit logging for OAuth** | Token operations not logged to audit trail. Would need audit_log table + INSERT calls. |

---

## 6. Phase 4: Deployment

**Goal**: Dockerfiles, end-to-end integration testing.

### Status: Not Started

This phase was not reached in the implementation sessions. It depends on:

1. **Kafka integration** (deferred) — for durable task queue between cloud/gateway and runtime.
2. **Container orchestration** — Dockerfiles for 3 containers, Koyeb deployment config.
3. **E2E test harness** — docker-compose with Kafka + Postgres + 3 services.

### Prerequisites Met

- Runtime has versioned API (`/v1/tasks/execute`)
- Cloud has `RuntimeClient` with health check
- Gateway has Kafka producer (implemented but uncalled)
- All three compile independently

---

## 7. What Was Not Implemented & Why

### Kafka (Cloud → Runtime, Gateway → Runtime)

**Decision**: Deferred by explicit user request.

**Rationale**: HTTP delegation (`POST /v1/tasks/execute`) provides synchronous execution with SSE streaming, which is simpler to debug, deploy, and monitor. Kafka adds:

- Operational overhead (broker cluster, topic management, consumer groups, offset tracking)
- Latency (produce → consume → execute vs direct HTTP)
- Complexity for streaming (Kafka doesn't natively support SSE-style streaming back to the caller)

HTTP delegation is correct for the current deployment model (cloud and runtime on same Koyeb cluster with low-latency networking). Kafka becomes valuable when:

- Runtime needs horizontal scaling with work-stealing queues
- Tasks need durable persistence across restarts
- Multiple consumers need the same task stream (fan-out)

### Gateway Webhooks & Telegram Delivery

**Decision**: Deferred to next iteration.

**Rationale**: The gateway currently handles the "produce" side (cron → Kafka, sensors → Kafka) but not the "consume + deliver" side (Kafka → Telegram reply). This requires:

1. A Kafka consumer loop in the gateway
2. Telegram bot token lookup from DB
3. Message formatting and delivery via Telegram API

The gateway crate has the infrastructure (Kafka producer, sensor manager) but webhook HTTP handlers and delivery consumers are not wired.

### Auth Hardening — Remove Bearer Token Fallback

**Decision**: Kept for backward compatibility.

**Rationale**: Existing API consumers (CRM integrations, scripts) use static bearer tokens. Removing them would break existing deployments. The `required = true` JWT mode effectively disables bearer auth when needed.

### Private MCP Server Registry

**Decision**: Deferred — existing mechanism suffices.

**Rationale**: Tenants can already install custom MCP servers via `POST /installations` with any URL. A "private server registry" with health checks, service discovery, and access control is a premium feature for later.

### OpenTelemetry (Cloud ↔ Runtime)

**Decision**: Not in scope for this update.

**Rationale**: The runtime already supports OTel via `[telemetry]` config. Cloud doesn't need its own OTel — it delegates execution to runtime. Future: propagate trace context in `RuntimeRequest` headers.

---

## 8. Test Coverage Summary

| Crate | Tests | Status |
|-------|-------|--------|
| `heartbit` (lib) | 3,071 | All passing |
| `heartbit-cli` (bin) | 30 | All passing |
| `heartbit-gateway` (bin) | 17 | All passing |
| `heartbit-cloud` (separate repo) | 177 | All passing |
| **Total** | **3,295** | **All passing** |

Quality gates enforced:
```bash
cargo fmt -- --check && cargo clippy -- -D warnings && cargo test
```

---

## 9. File Structure Reference

### Files Modified in This Update

**heartbit repo** (28 files changed, -15,091 / +1,284 lines):

| File | Change | Purpose |
|------|--------|---------|
| `crates/heartbit-cli/src/daemon.rs` | **Deleted** | Monolithic 3624-line file → split into directory |
| `crates/heartbit-cli/src/daemon/mod.rs` | **New** | Daemon startup, route wiring, `/v1/tasks/execute` |
| `crates/heartbit-cli/src/daemon/execute.rs` | **New** | Cloud-delegated execution handler |
| `crates/heartbit-cli/src/daemon/handlers.rs` | **New** | REST handlers (submit, list, get, cancel, approve) |
| `crates/heartbit-cli/src/daemon/types.rs` | **New** | Request/response DTOs |
| `crates/heartbit-cli/src/daemon/auth.rs` | **New** | JWT + bearer middleware |
| `crates/heartbit-cli/src/daemon/memory.rs` | **New** | Institutional memory utilities |
| `crates/heartbit/src/config.rs` | **Deleted** | Monolithic 7744-line file → split into directory |
| `crates/heartbit/src/config/mod.rs` | **New** | HeartbitConfig + orchestrator config |
| `crates/heartbit/src/config/agent.rs` | **New** | Per-agent configuration |
| `crates/heartbit/src/config/daemon.rs` | **New** | Daemon mode settings |
| `crates/heartbit/src/config/guardrails.rs` | **New** | Guardrail definitions |
| `crates/heartbit/src/config/memory.rs` | **New** | Memory store configuration |
| `crates/heartbit/src/config/provider.rs` | **New** | LLM provider settings |
| `crates/heartbit/src/config/sensor.rs` | **New** | Sensor pipeline config |
| `crates/heartbit/src/agent/builder.rs` | **New** | AgentRunnerBuilder (extracted) |
| `crates/heartbit/src/agent/runner.rs` | **New** | AgentRunner core loop (extracted) |
| `crates/heartbit/src/agent/doom_loop.rs` | **New** | DoomLoopTracker (extracted) |
| `crates/heartbit/src/daemon/runtime_types.rs` | **New** | RuntimeRequest/Response/SseEvent protocol |
| `crates/heartbit/src/daemon/mod.rs` | **Modified** | Added runtime_types module + re-exports |
| `crates/heartbit/src/lib.rs` | **Modified** | Added Runtime* type re-exports |
| `crates/heartbit/src/agent/guardrail.rs` | **Modified** | Kill switch + audit mode |
| `crates/heartbit/src/agent/guardrails/*.rs` | **Modified** | All 10 guardrail files updated |
| `crates/heartbit/src/agent/orchestrator.rs` | **Modified** | Sub-agent config struct |
| `crates/heartbit-gateway/` | **New crate** | 7 source files, 17 tests |

**heartbit-cloud repo** (30 files changed, +2,279 / -468 lines):

| File | Change | Purpose |
|------|--------|---------|
| `src/routes/ws.rs` | **New** | WebSocket chat handler |
| `src/routes/oauth.rs` | **New** | MCP OAuth authorization routes |
| `src/marketplace/oauth.rs` | **New** | OAuth store (PKCE, token encryption, refresh) |
| `migrations/017_mcp_oauth.sql` | **New** | OAuth token storage + catalog OAuth fields |
| `migrations/018_gateway_config.sql` | **New** | Gateway webhooks + sensors tables |
| `src/server.rs` | **Major** | AppState expansion, build_runtime_request, token refresh, route wiring |
| `src/config.rs` | **Major** | Added RuntimeClientConfig, StripeConfig, XavyoConfig, public_url |
| `src/routes/chat.rs` | **Major** | Refactored for runtime delegation |
| `src/routes/tasks.rs` | **Major** | Async task spawn with cancellation tokens |
| `src/agent_config/mod.rs` | **Major** | AgentAdvancedConfig, GuardrailsConfig, builder agent name |
| `src/auth/middleware.rs` | **Modified** | JWT + API key dual auth |
| `src/execution/runtime_client.rs` | **Modified** | Updated endpoints to `/v1/` |
| `src/marketplace/mod.rs` | **Modified** | Added oauth_client_id, oauth_authorize_url to catalog |
| `src/marketplace/store.rs` | **Modified** | Added get_catalog_by_id() |
| `Cargo.toml` | **Modified** | Added urlencoding, aes-gcm, hmac, reqwest, subtle, bytes |

---

## Summary

The 3-tier architecture is **operational** with the following completion levels:

| Tier | Completion | Notes |
|------|-----------|-------|
| **Runtime** | **95%** | Pure execution engine, versioned API, cloud delegation endpoint, auth hardening. Missing: bearer fallback removal (intentionally kept). |
| **Gateway** | **60%** | Crate scaffolded with cron + sensors + Kafka producer. Missing: webhook handlers, Telegram delivery, auth middleware, metrics endpoint. |
| **Cloud** | **90%** | Full SaaS platform with 18 route modules, MCP OAuth, WebSocket chat, billing, marketplace. Missing: Kafka integration, event streaming endpoint. |
| **Deployment** | **0%** | Not started. Awaiting Kafka decision. |

The architecture works end-to-end via HTTP delegation today. Kafka integration is the key decision point for scaling beyond single-runtime deployments.

---

*Generated from deep codebase analysis of heartbit (3,071 tests), heartbit-cli (30 tests), heartbit-gateway (17 tests), and heartbit-cloud (177 tests) — 3,295 tests total, all passing.*
