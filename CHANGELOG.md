# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [2026.228.2] - 2026-02-28

### Added

- **`[daemon.auth.token_exchange]` config** — RFC 8693 Token Exchange configuration (`exchange_url`, `client_id`, `client_secret`, `agent_token`, `scopes`) for per-user MCP auth delegation. Config validation rejects empty required fields.
- **Per-task MCP tool loading** — when `token_exchange` is configured and user context is present, each daemon task creates fresh MCP connections with a user-scoped delegated token instead of shared static auth.
- **`audit_delegation_chain`** on `AgentRunner`, `Orchestrator`, `SubAgentConfig` — records which agent(s) are in the delegation path when acting on behalf of a user. Populated automatically in multi-tenant mode.
- **Tenant-scoped store queries** — `TaskStore::list_filtered()` and `stats()` accept `tenant_id` parameter, pushing filter to store level. Fixes pagination counts and prevents cross-tenant data in stats.
- 11 new tests: 6 tenant-filtered store/core, 5 token exchange config validation.

## [2026.228.1] - 2026-02-28

### Added

- **Multi-tenant daemon** — single daemon instance serves multiple users with per-request tenant isolation. JWT-authenticated API ensures tasks, memory, and workspaces are scoped per user/tenant.
- **JWT/JWKS authentication** (`auth/jwt.rs`): `JwksClient` fetches and caches JWKS keys (5-minute TTL, auto-refetch on key rotation). `JwtValidator` verifies RS256 tokens, extracts `UserContext` (user_id, tenant_id, roles) from configurable claim names.
- **`UserContext` struct** (`daemon/types.rs`): carries `user_id`, `tenant_id`, and `roles` through every request. Extracted from JWT claims by auth middleware, injected into request extensions.
- **`[daemon.auth]` config section** (`config.rs`): `bearer_tokens` (static API keys with rotation support), `jwks_url` (JWKS endpoint), `issuer`/`audience` (JWT validation), `user_id_claim`/`tenant_id_claim`/`roles_claim` (configurable claim names for different IdPs).
- **Per-user memory namespacing** — daemon wraps memory store with `NamespacedMemory` using `tenant:{tid}:user:{uid}` prefix. Users cannot access each other's memories. Institutional memory remains shared via `shared_memory_read` tool.
- **Dynamic MCP token injection** (`tool/mcp.rs`): `AuthProvider` trait with `auth_header_for(user_id, tenant_id)` enables per-request authentication. `StaticAuthProvider` for backward-compatible static headers. `TokenExchangeAuthProvider` implements RFC 8693 token exchange against an IdP (e.g. xavyo-idp) to obtain user-scoped MCP tokens with in-memory caching.
- **Per-user workspace isolation** — workspace root becomes `{base}/{tenant_id}/{user_id}/` in multi-tenant mode. Path traversal prevention already enforced.
- **Audit trail enrichment** (`agent/audit.rs`): `AuditRecord` gains `user_id`, `tenant_id`, and `delegation_chain` fields. `AuditTrail` trait gains `entries_for_tenant()` for tenant-scoped queries.
- **A2A Agent Card** — daemon serves `GET /.well-known/agent.json` for agent discovery. Card includes agent name, description, skills (from config agents), auth schemes (bearer/JWT), and endpoint URL.
- `Error::Auth` variant for authentication-specific errors (distinct from infrastructure errors).
- `PostgresTaskStore` gains `user_id` and `tenant_id` columns with ALTER TABLE migration for existing databases.
- 32 new tests for JWT validation, claim extraction, cross-namespace isolation, and token exchange hardening.

### Changed

- All JWT/auth errors now use `Error::Auth` (previously used `Error::Agent`, making 401 vs 502 indistinguishable).
- `NamespacedMemory::recall()` always forces own namespace — ignores caller-supplied agent parameter to prevent cross-namespace reads via prompt injection.
- `TokenExchangeAuthProvider` hardened: token cache with TTL (30s early expiry), 10-second HTTP timeout, error body truncated to 512 bytes, respects `token_type` from response.
- `JwtValidator::validate()` rejects empty tokens and tokens exceeding 16 KiB.
- `extract_string_claim()` rejects null, boolean, object, and array claim values (accepts only string and number).

## [2026.227.1] - 2026-02-27

### Added

- **Local embedding provider** (`LocalEmbeddingProvider`): offline ONNX-based text embeddings via fastembed. No API keys, no network, zero cost per query. Supports 9 models (all-MiniLM-L6-v2 default, BGE variants, nomic variants, plus quantized `-q` suffixes). Feature-gated behind `local-embedding`.
- **Eval framework** (`eval/mod.rs`): built-in agent behavior testing with `EvalCase`, `EvalRunner`, and pluggable scorers (`TrajectoryScorer`, `KeywordScorer`, `SimilarityScorer`). Concurrent evaluation with per-case and aggregate scoring.
- **Workflow agents** (`agent/workflow.rs`): deterministic orchestration without LLM cost — `SequentialAgent` (chains output→input), `ParallelAgent` (concurrent via `JoinSet`), `LoopAgent` (repeat until condition).
- **Audit trail** (`agent/audit.rs`): `AuditTrail` trait with `InMemoryAuditTrail` and `PostgresAuditTrail` for logging agent decisions, tool calls, and guardrail outcomes.
- **Injection classifier guardrail** (`guardrails/injection.rs`): detect prompt injection attempts with warn or deny mode.
- **PII guardrail** (`guardrails/pii.rs`): detect PII (email, phone, SSN, credit card) with redact, warn, or deny actions.
- **Tool policy guardrail** (`guardrails/tool_policy.rs`): declarative per-tool allow/deny rules with input constraints (patterns, max length).
- **LLM-as-judge guardrail** (`guardrails/llm_judge.rs`): safety evaluation via a cheap judge model with criteria-based prompts. Fail-open on timeout.
- **Guardrail composition** (`guardrails/compose.rs`): `ConditionalGuardrail`, `GuardrailChain`, and `WarnToDeny` escalation.
- **`GuardrailMeta` trait**: optional guardrail identification for debugging and audit.
- **`GuardrailsConfig`** in config: top-level `[guardrails]` section with injection, PII, tool policy, and LLM judge sub-configs. Per-agent `guardrails` override.
- `cache_dir` field on `EmbeddingConfig` for local provider model cache directory.
- `local-embedding` feature flag on `heartbit` and `heartbit-cli` crates.
- `RoutingStrategy` trait and `KeywordRoutingStrategy` for pluggable task routing.
- `TrustLevel` enum moved to `config.rs` (always available, not sensor-gated).
- `SensorModality` re-exported from config (always available).
- Examples: `simple_agent.rs`, `mcp_agent.rs`, `custom_tool.rs`.
- `llms.md`: LLM-friendly project context file (llmstxt.org pattern).
- `install.md`: comprehensive installation guide with troubleshooting.

### Changed

- **Feature-gated modules**: `daemon`, `sensor`, `workflow` modules now require their respective feature flags. Previously always compiled.
- **Feature-gated re-exports**: `PostgresMemoryStore`, `PostgresSessionStore`, `PostgresTaskStore`, `PostgresStore`, `PostgresAuditTrail` gated behind `postgres`; `A2aClient` behind `a2a`; `SensorSecurityGuardrail` behind `sensor`; sensor re-exports behind `sensor`; `LocalEmbeddingProvider` behind `local-embedding`.
- Agent events expanded from 13 to 18 variants (added `GuardrailWarned`, `LlmRetry`, `ModelEscalated`, `ToolDeselected`, `ReflectionTriggered`).
- Guardrail re-exports expanded: all 8 guardrails, `GuardrailMode`, `PiiAction`, `PiiDetector`, `InputConstraint`, `ToolRule`, `WarnToDeny` now re-exported from crate root.
- Config re-exports expanded: `GuardrailsConfig`, `InjectionConfig`, `PiiConfig`, `ToolPolicyConfig`, `InputConstraintConfig`, `ToolPolicyRuleConfig`, `SensorModality`, `TrustLevel`.
- README comprehensively updated: expanded guardrails (2→8 with table), memory section (embedding providers, hybrid retrieval, confidentiality), feature flags section, workflow agents section, eval framework section, audit trail section, environment variables (12→27), config example (cascade, routing, dispatch_mode, session_prune), test count (2374→2665+).

## [2026.226.2] - 2026-02-26

### Fixed

- Repetitive pulse notifications dumping entire todo list (including completed items) every 30 minutes.
- Added `snoozed_until` field to `TodoEntry` for suppressing items from pulse.
- `format_for_pulse_prompt()` filters terminal/snoozed entries.
- Snooze action added to `TodoManageTool` (default 24h, validates hours > 0).
- Kafka event serialization: log-and-skip instead of sending empty payload.
- `subscribe_events`: log on lock poison instead of silent `None`.
- Validate Kafka `consumer_group`, `commands_topic`, `events_topic` for non-empty.

## [2026.226.1] - 2026-02-26

### Added

- Institutional memory: daemon task results auto-persist to shared `"institutional"` namespace, enabling cross-context knowledge flow from sensor pipeline to Telegram chat agents.
- Telegram dual recall: `preload_memories` now queries both user-private and institutional namespaces concurrently via `tokio::join!`.
- `institutional_recall_limit` config field for Telegram (default: 3).
- `story_id` field on `TaskOutcome` for story-scoped provenance tracking.

### Changed

- README rewritten with architecture diagrams, contributor guide, and disclaimer.
- README updated to highlight Telegram and Google Workspace integrations.
- README updated with Telegram community link.

## [2026.2.26] - 2026-02-26

### Added

- Multi-agent runtime with orchestrator and sub-agents (flat hierarchy, parallel dispatch via `tokio::JoinSet`).
- Three execution paths: standalone (in-process), Restate (durable with replay), and daemon (Kafka-backed with HTTP API).
- 14 built-in tools: bash, read, write, edit, patch, glob, grep, list, webfetch, websearch, todowrite, todoread, skill, question.
- MCP Streamable HTTP client (protocol `2025-03-26`) with automatic tool discovery and optional authentication.
- LLM providers: Anthropic and OpenRouter with SSE streaming.
- Retry provider with exponential backoff on 429/5xx errors.
- Cascading provider: tries cheapest model first, escalates on gate rejection or error.
- Prompt caching for Anthropic (cache reads at 10% input rate, writes at 125%).
- Structured output via synthetic `__respond__` tool with JSON Schema validation.
- Context management strategies: unlimited, sliding window, and LLM-generated summarization.
- Memory system with in-memory and PostgreSQL backends (store, recall, update, forget, consolidate).
- Composite recall scoring (recency, importance, relevance, strength) with Ebbinghaus decay.
- Knowledge base with paragraph-aware chunking, keyword search, and file/glob/URL loaders.
- Guardrails: pre/post LLM and tool hooks with allow/deny actions.
- Human-in-the-loop approval for tool execution (`--approve` flag).
- Sensor pipeline with 6 sources, triage, deduplication, and story grouping.
- Telegram bot integration with DM support, streaming responses, and multimodal input (photos, voice, documents).
- Daemon mode: Kafka consumer loop, Axum HTTP API, SSE event streaming, cron scheduler, heartbeat pulse.
- Cross-agent coordination via shared blackboard and memory tools.
- Dynamic task routing based on complexity heuristics.
- Agent workspace with path traversal prevention.
- Cost tracking with per-model pricing for Claude 4, 3.5, and 3 generations.
- 13 structured agent event variants with JSON stderr output (`--verbose`).
- OpenTelemetry tracing via OTLP exporter.
- Interactive chat mode with multi-turn REPL.
- Docker support with multi-stage build.
- Doom loop detection and auto-compaction on context overflow.
- Tool output truncation with UTF-8 safe boundaries.
- Tool name repair via Levenshtein distance matching.
