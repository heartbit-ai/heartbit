[![CI](https://github.com/heartbit-ai/heartbit/actions/workflows/ci.yml/badge.svg)](https://github.com/heartbit-ai/heartbit/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)
[![Telegram](https://img.shields.io/badge/Telegram-heartbitagent-blue?logo=telegram)](https://t.me/heartbitagent)

# Heartbit

Multi-agent enterprise runtime in Rust. Orchestrator spawns sub-agents that execute LLM-powered reasoning loops with parallel tool execution.

**Why Heartbit?**
- **Zero-copy agent loops** — ReAct cycle in pure Rust, no Python/Node overhead
- **Three execution paths** — standalone (zero infra), durable (Restate), daemon (Kafka)
- **Flat agent hierarchy** — orchestrator delegates to sub-agents, sub-agents never spawn further
- **Parallel tool execution** — `tokio::JoinSet` runs tools concurrently within each turn
- **Production-grade** — 8 guardrails, context management, MemGPT-style memory, cost tracking, OpenTelemetry
- **Local-first embeddings** — offline semantic search via ONNX Runtime (fastembed), no API keys required
- **Built-in eval framework** — trajectory scoring, keyword matching, and similarity scoring for agent behavior testing
- **Built-in integrations** — Telegram bot, Google Workspace (JMAP email), RSS, webhooks, and more via sensor pipeline

> **Not an OpenClaw fork or clone.** Heartbit is an independent project built from scratch. It shares no code, architecture, or lineage with [OpenClaw](https://github.com/anthropics/openclaw) or any other agent framework. Different design goals, different codebase.

> **Early-stage software — capability over security.** Heartbit prioritizes **capability and velocity** at this stage of development. Security hardening is ongoing but not yet comprehensive. Agents execute tools (including shell commands) with the permissions of the host process. **Do not run untrusted workloads in production environments without your own sandboxing and access controls.** See [Disclaimer](#disclaimer) below.

## Architecture Overview

### System Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        heartbit-cli (bin)                          │
│  Commands: run | chat | serve | daemon | submit | status | approve │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────┐
│                          heartbit (lib)                             │
│                                                                     │
│  ┌─────────────────┐  ┌────────────────┐  ┌──────────────────────┐ │
│  │   Standalone     │  │    Durable      │  │      Daemon          │ │
│  │                  │  │                 │  │                      │ │
│  │  AgentRunner     │  │  AgentWorkflow  │  │  Kafka consumer      │ │
│  │  Orchestrator    │  │  OrchestratorWf │  │  Axum HTTP API       │ │
│  │  tokio::JoinSet  │  │  Restate SDK    │  │  SSE + WebSocket     │ │
│  │                  │  │                 │  │  Cron scheduler      │ │
│  └────────┬─────────┘  └───────┬────────┘  │  Heartbeat pulse     │ │
│           │                    │           └──────────┬───────────┘ │
│           ▼                    ▼                      ▼             │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                      Shared Core                                ││
│  │                                                                 ││
│  │  LlmProvider (Anthropic, OpenRouter)    Tool trait + MCP client ││
│  │  Memory (InMemory, Postgres)            KnowledgeBase           ││
│  │  Guardrails (pre/post LLM & tool)       Sensor pipeline         ││
│  │  Context strategies                     Channel adapters         ││
│  │  Cost tracking + OTel                   Permission system        ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

### Three Execution Paths

| Path | Infrastructure | Use case |
|------|---------------|----------|
| **Standalone** | None (in-process) | CLI tasks, scripts, library embedding |
| **Durable** | [Restate](https://restate.dev/) server | Crash-resilient workflows, exactly-once execution |
| **Daemon** | Kafka + Axum | Long-running services, cron jobs, event-driven tasks |

### Core Concepts

| Concept | What it does | Key type |
|---------|-------------|----------|
| **AgentRunner** | Executes one agent's ReAct loop (LLM → tools → repeat) | `AgentRunner<P>` |
| **Orchestrator** | Dispatches tasks to sub-agents via `delegate_task` / `form_squad` | `Orchestrator<P>` |
| **Tool** | A capability the agent can invoke (bash, read, MCP, custom) | `Arc<dyn Tool>` |
| **LlmProvider** | Sends completion requests to an LLM (Anthropic, OpenRouter) | `Arc<BoxedProvider>` |
| **Guardrail** | Intercepts the agent loop at 4 hook points | `Arc<dyn Guardrail>` |
| **Memory** | Persistent agent memory with recall scoring | `Arc<dyn Memory>` |

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

## Quick Start

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

Without a config file, a single agent runs with 14 built-in tools (bash, read, write, edit, patch, glob, grep, etc.).

## How It Works

### The Agent Loop (ReAct Cycle)

Every agent runs the same loop. The orchestrator is just an agent whose tools include `delegate_task` and `form_squad`.

```
                    ┌──────────────────────┐
                    │   AgentRunner::      │
                    │   execute(task)       │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  Build messages       │
                    │  [system + history    │
                    │   + task/tool results]│
                    └──────────┬───────────┘
                               │
              ┌────────────────▼────────────────┐
              │         LLM call                 │
              │  (stream_complete / complete)     │
              │  Provider → Retry → Cascade      │
              └────────────────┬────────────────┘
                               │
                    ┌──────────▼───────────┐
                    │  Response has         │
                    │  tool calls?          │
                    └──┬────────────────┬──┘
                       │ Yes            │ No
            ┌──────────▼──────────┐     │
            │  Guardrail:         │     │
            │  pre_tool checks    │     │
            └──────────┬──────────┘     │
                       │                │
            ┌──────────▼──────────┐     │
            │  Execute tools in   │     │
            │  PARALLEL via       │     │
            │  tokio::JoinSet     │     │
            └──────────┬──────────┘     │
                       │                │
            ┌──────────▼──────────┐     │
            │  Guardrail:         │     │
            │  post_tool checks   │     │
            └──────────┬──────────┘     │
                       │                │
            ┌──────────▼──────────┐     │
            │  Append results     │     │
            │  to messages        │     │
            └──────────┬──────────┘     │
                       │                │
                       │     ┌──────────▼───────────┐
                       │     │  Return AgentOutput   │
                       │     │  { result, tokens,    │
                       │     │    cost, structured } │
                       │     └──────────────────────┘
                       │
              (loop back to LLM call)
```

Key behaviors:
- **Parallel tool execution** — all tool calls in a single turn run concurrently via `JoinSet`
- **Panic recovery** — a panicked tool task produces an error result without crashing the loop
- **Context management** — `SlidingWindow` or `Summarize` strategies trim messages before each LLM call
- **Auto-compaction** — on `ContextOverflow`, the agent summarizes history and retries (max once per turn pair)
- **Doom loop detection** — detects repeated identical tool-call batches and breaks the cycle
- **Tool name repair** — Levenshtein distance ≤ 2 auto-corrects misspelled tool names

### Multi-Agent Orchestration

The orchestrator is an `AgentRunner` with two delegation tools. Sub-agents do NOT spawn further agents (flat hierarchy).

```
                    ┌───────────────────────┐
                    │     Orchestrator       │
                    │  (AgentRunner + tools) │
                    └───────────┬───────────┘
                                │
                   LLM decides delegation
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                  │
   ┌──────────▼──────┐  ┌──────▼────────┐  ┌─────▼──────────┐
   │  delegate_task   │  │  delegate_task │  │  form_squad     │
   │  (independent)   │  │  (independent) │  │  (collaborative)│
   └──────────┬──────┘  └──────┬────────┘  └─────┬──────────┘
              │                │                  │
   ┌──────────▼──────┐  ┌─────▼─────────┐  ┌────▼───────────┐
   │  Sub-Agent A     │  │  Sub-Agent B   │  │  Squad of C, D  │
   │  (own tools,     │  │  (own tools,   │  │  (shared        │
   │   own provider)  │  │   own provider)│  │   blackboard)   │
   └──────────┬──────┘  └──────┬────────┘  └─────┬──────────┘
              │                │                  │
              └────────────────┼──────────────────┘
                               │
                    ┌──────────▼───────────┐
                    │  Results aggregated   │
                    │  back to orchestrator │
                    │  + written to         │
                    │  blackboard           │
                    └──────────────────────┘
```

- **`delegate_task`** — independent parallel subtasks, each sub-agent gets its own `AgentRunner`
- **`form_squad`** — collaborative subtasks sharing a `Blackboard` (key-value store)
- **Routing** — `Auto` mode uses heuristic + capability matching; `AlwaysOrchestrate` / `SingleAgent` overrides
- **Per-agent providers** — each sub-agent can use a different LLM model/provider

### LLM Provider Stack

Providers compose via wrapping. The cascade tries cheaper models first.

```
     ┌────────────────────────────────────┐
     │  CascadingProvider (optional)       │
     │  tier 1: haiku (cheapest)           │
     │  tier 2: sonnet                     │
     │  tier 3: opus (most capable)        │
     │                                     │
     │  Gate: confidence check after each  │
     │  tier — escalate on rejection       │
     └──────────────┬─────────────────────┘
                    │ wraps
     ┌──────────────▼─────────────────────┐
     │  RetryingProvider                   │
     │  Exponential backoff on:            │
     │  429, 500, 502, 503, 529            │
     │  + network errors                   │
     └──────────────┬─────────────────────┘
                    │ wraps
     ┌──────────────▼─────────────────────┐
     │  AnthropicProvider / OpenRouter     │
     │  SSE streaming, prompt caching      │
     │  ToolChoice (Auto/Any/Tool)         │
     └────────────────────────────────────┘
```

### Sensor Pipeline (Daemon)

Sensors gather data from external sources, triage classifies it, stories aggregate related events into actionable commands.

```
  Sources                  Triage              Stories         Daemon
  ───────                  ──────              ───────         ──────
  ┌──────────┐
  │  RSS      │──┐
  ├──────────┤  │
  │  Email    │──┤      ┌────────────┐     ┌────────────┐   ┌─────────┐
  │  (JMAP/   │  ├────▶│  Triage     │───▶│  Story      │──▶│ Command  │
  │  Google)  │──┤      │  (per-type  │    │  Builder    │   │ Producer │
  ├──────────┤  │      │   scoring)  │    │  (dedup,    │   │ → Kafka  │
  │  Webhook  │──┤      └────────────┘    │   merge)    │   └─────────┘
  ├──────────┤  │                         └────────────┘
  │  Weather  │──┤
  ├──────────┤  │
  │  Audio    │──┤
  ├──────────┤  │
  │  Image    │──┘
  └──────────┘
```

## Module Map

### Crate Structure

```
crates/
  heartbit/                    # Library crate — all core logic
    src/
      agent/
        mod.rs                 # AgentRunner: the ReAct loop
        orchestrator.rs        # Orchestrator: multi-agent dispatch
        context.rs             # Context strategies (sliding window, summarize)
        guardrail.rs           # Guardrail trait definition
        guardrails/            # Built-in guardrail implementations
        blackboard.rs          # Blackboard trait + InMemoryBlackboard
        blackboard_tools.rs    # Blackboard read/write/list tools
        events.rs              # AgentEvent enum (13 variants)
        routing.rs             # Task routing (Auto/AlwaysOrchestrate/SingleAgent)
        observability.rs       # Span tracking, TTFT, retry events
        permission.rs          # Tool permission rules + learned permissions
        pruner.rs              # Session pruning (trim old tool results)
        tool_filter.rs         # ToolProfile-based tool pre-filtering
        instructions.rs        # System prompt construction + resourcefulness
        token_estimator.rs     # Token count estimation
      llm/
        mod.rs                 # LlmProvider trait, BoxedProvider, DynLlmProvider
        anthropic.rs           # Anthropic provider (SSE, prompt caching)
        openrouter.rs          # OpenRouter provider (OpenAI-compat SSE)
        cascade.rs             # CascadingProvider + ConfidenceGate
        retry.rs               # RetryingProvider (exponential backoff)
        pricing.rs             # Cost estimation for known models
        error_class.rs         # Error classification (ContextOverflow, RateLimit, ...)
        types.rs               # CompletionRequest, CompletionResponse, ToolCall, ...
      tool/
        mod.rs                 # Tool trait, ToolDefinition, ToolOutput
        mcp.rs                 # MCP Streamable HTTP client
        a2a.rs                 # Agent-to-Agent protocol
        builtins/
          mod.rs               # builtin_tools() factory, BuiltinToolsConfig
          bash.rs              # Shell command execution
          read.rs              # File reading with line numbers
          write.rs             # File writing with read-before-write guard
          edit.rs              # Exact string replacement
          patch.rs             # Unified diff application
          glob.rs              # File pattern matching
          grep.rs              # Content search (regex)
          list.rs              # Directory tree listing
          webfetch.rs          # HTTP GET with content extraction
          websearch.rs         # Exa AI web search
          todo.rs              # Todo list management
          question.rs          # Structured agent-to-user questions
          skill.rs             # SKILL.md-based skill loading
          file_tracker.rs      # Shared mtime-based read-before-write guard
      memory/
        mod.rs                 # Memory trait, MemoryEntry, MemoryQuery, MemoryType
        in_memory.rs           # InMemoryStore
        postgres.rs            # PostgresMemoryStore
        bm25.rs                # BM25 text scoring
        scoring.rs             # Composite recall scoring (Park et al.)
        reflection.rs          # ReflectionTracker (importance threshold)
        consolidation.rs       # ConsolidationPipeline (Jaccard clustering)
        pruning.rs             # Ebbinghaus strength decay + weak entry pruning
        namespaced.rs          # NamespacedMemory wrapper
        embedding.rs           # EmbeddingProvider trait + OpenAI, Local (fastembed), Noop
        hybrid.rs              # Hybrid recall (BM25 + embedding)
        tools.rs               # 5 memory tools (store, recall, update, forget, consolidate)
        shared_tools.rs        # Shared memory for multi-agent
      knowledge/
        mod.rs                 # KnowledgeBase trait, Chunk, SearchResult
        in_memory.rs           # InMemoryKnowledgeBase (keyword search)
        chunker.rs             # Paragraph-aware chunking with overlap
        loader.rs              # File, glob, URL loaders
        tools.rs               # knowledge_search tool
      sensor/
        mod.rs                 # Sensor trait, SensorEvent, SensorModality
        manager.rs             # SensorManager (lifecycle)
        stories.rs             # Story builder (dedup, merge)
        routing.rs             # Event routing
        metrics.rs             # Sensor metrics
        sources/               # 7 sensor sources (RSS, JMAP/Google Workspace, webhook, weather, audio, image, MCP)
        triage/                # Per-modality triage classifiers
        compression/           # Event compression rules
        perception/            # Perceivers for multimodal input
      channel/
        mod.rs                 # Channel module
        bridge.rs              # InteractionBridge (A2H adapter)
        session.rs             # Session management
        types.rs               # WebSocket frame types
        telegram/              # Telegram bot adapter (DMs, streaming, HITL)
      daemon/
        core.rs                # DaemonCore: Kafka consumer + task runner
        kafka.rs               # CommandProducer trait, KafkaCommandProducer
        cron.rs                # Cron scheduler (6-field expressions)
        heartbit_pulse.rs      # Periodic awareness loop
        store.rs               # Task store
        todo.rs                # FileTodoStore
        notify.rs              # Notification dispatch
        metrics.rs             # Daemon metrics
        types.rs               # DaemonCommand, DaemonEvent
      workflow/
        mod.rs                 # Restate module entry
        agent_service.rs       # Restate service (llm_call + tool_call activities)
        agent_workflow.rs      # Restate workflow (durable ReAct loop)
        orchestrator_workflow.rs  # Durable orchestrator
        blackboard.rs          # Restate virtual object (shared state)
        budget.rs              # Token budget tracking
        circuit_breaker.rs     # LLM circuit breaker
        scheduler.rs           # Recurring task scheduler
        types.rs               # Workflow types
      auth/
        jwt.rs                 # JwksClient, JwtValidator → UserContext
      lsp/                     # LSP client integration
      store/                   # PostgreSQL task/audit store
      config.rs                # HeartbitConfig from TOML (incl. AuthConfig)
      error.rs                 # Error types (thiserror)
      workspace.rs             # Agent workspace (sandboxed file access)
      lib.rs                   # Public API re-exports
  heartbit-cli/                # Binary crate
    src/
      main.rs                  # CLI entry point (clap), standalone runner
      serve.rs                 # Restate HTTP worker
      submit.rs                # Restate task submission
      daemon.rs                # Daemon mode entry point
```

### Key Traits

| Trait | Location | Purpose |
|-------|----------|---------|
| `Tool` | `tool/mod.rs` | Define a tool the agent can call |
| `LlmProvider` | `llm/mod.rs` | Send completions to an LLM |
| `Memory` | `memory/mod.rs` | Persistent agent memory (store, recall, update, forget) |
| `KnowledgeBase` | `knowledge/mod.rs` | Document retrieval (index, search) |
| `Guardrail` | `agent/guardrail.rs` | Intercept agent loop (pre/post LLM/tool hooks) |
| `Sensor` | `sensor/mod.rs` | Gather data from external sources |
| `Blackboard` | `agent/blackboard.rs` | Shared key-value store for squad coordination |
| `AuthProvider` | `tool/mcp.rs` | Per-user auth header resolution for MCP servers |
| `CommandProducer` | `daemon/kafka.rs` | Produce commands to Kafka topics |

### Contributor Pathfinder

| I want to... | Start here |
|---|---|
| Understand the agent loop | `agent/mod.rs` → `AgentRunner::execute()` |
| Add a new built-in tool | `tool/builtins/` → implement `Tool` trait |
| Add an LLM provider | `llm/mod.rs` → implement `LlmProvider` trait |
| Add a sensor source | `sensor/sources/` → implement `Sensor` trait |
| Add a guardrail | `agent/guardrail.rs` → implement `Guardrail` trait |
| Understand multi-agent dispatch | `agent/orchestrator.rs` → `Orchestrator` |
| Add a memory backend | `memory/mod.rs` → implement `Memory` trait |
| Understand context management | `agent/context.rs` → `ContextStrategy` |
| Add a channel adapter | `channel/bridge.rs` → `InteractionBridge` |
| Understand the daemon | `daemon/core.rs` → `DaemonCore` |
| Add JWT/multi-tenant auth | `auth/jwt.rs` → `JwtValidator` |
| Add an MCP auth provider | `tool/mcp.rs` → `AuthProvider` trait |
| Add a Restate workflow | `workflow/` → `agent_workflow.rs` |
| Understand config loading | `config.rs` → `HeartbitConfig` |
| Understand cost tracking | `llm/pricing.rs` → `estimate_cost()` |
| Add a triage classifier | `sensor/triage/` → match on `SensorModality` |
| Understand task routing | `agent/routing.rs` → `RoutingMode` |

> All paths are relative to `crates/heartbit/src/`.

## Feature Flags

The `heartbit` crate uses feature flags to keep the default build lightweight. Only `core` is enabled by default.

| Feature | Dependencies | What it enables |
|---------|-------------|-----------------|
| `core` (default) | — | Agent runner, orchestrator, LLM providers, tools, memory, config |
| `kafka` | `rdkafka` | Kafka consumer/producer |
| `daemon` | kafka + `cron`, `prometheus` | Long-running daemon with HTTP API, cron scheduling, metrics |
| `sensor` | daemon + `quick-xml`, `hmac`, `sha2`, `hex`, `subtle` | 7 sensor sources, triage pipeline, story correlation |
| `restate` | `restate-sdk 0.8` | Durable workflow execution (services, workflows, virtual objects) |
| `postgres` | `sqlx`, `pgvector` | PostgreSQL-backed memory store + task store with vector search |
| `a2a` | `a2a-sdk` | Agent-to-Agent protocol (endpoint discovery, remote tool invocation) |
| `telegram` | `teloxide` | Telegram bot (DMs, streaming, HITL approval, keyboard menus) |
| `local-embedding` | `fastembed` | Local ONNX-based text embeddings (no API keys, offline) |
| `full` | all of the above (except `local-embedding`) | Everything enabled — used by the CLI |

```bash
# Default build (core only — no Kafka, no Postgres, no Telegram)
cargo build -p heartbit

# Full CLI build
cargo build -p heartbit-cli

# With local embeddings
cargo build --features local-embedding
```

## Key Subsystems

### Memory

MemGPT-inspired memory with composite recall scoring and hybrid retrieval.

- **Storage**: `InMemoryStore`, `PostgresMemoryStore` (pgvector), or `NamespacedMemory` (3-tier: user/agent/session)
- **Memory types**: `Episodic` (default), `Semantic`, `Reflection`
- **Confidentiality**: `Public`, `Internal`, `Confidential`, `Restricted` — controls visibility in LLM context
- **Recall scoring**: BM25 keyword search (2x boost) + Park et al. composite (recency + importance + relevance + strength)
- **Hybrid retrieval**: BM25 + vector cosine similarity fused via Reciprocal Rank Fusion (RRF)
- **Ebbinghaus decay**: `effective_strength()` with decay rate of 0.005/hr (~6-day half-life); strength reinforced +0.2 on access
- **Reflection**: `ReflectionTracker` triggers reflection prompts when cumulative importance exceeds threshold
- **Consolidation**: `ConsolidationPipeline` clusters entries by Jaccard keyword similarity, merges into `Semantic` entries
- **Pruning**: auto-prune weak memories at session end; configurable min strength + min age
- **Session pruning**: `SessionPruneConfig` auto-trims old tool results before LLM calls
- **Pre-compaction flush**: extracts tool results to episodic memory before context summarization

5 agent-facing tools: `memory_store`, `memory_recall`, `memory_update`, `memory_forget`, `memory_consolidate`.

#### Embedding Providers

Embeddings enable hybrid retrieval (BM25 + vector cosine) for improved recall quality.

| Provider | Config `provider =` | Requirements | Dimension |
|----------|-------------------|--------------|-----------|
| `NoopEmbedding` | `"none"` | None | 0 (BM25-only fallback) |
| `OpenAiEmbedding` | `"openai"` | API key (`OPENAI_API_KEY`) | 1536 (small) / 3072 (large) |
| `LocalEmbeddingProvider` | `"local"` | `local-embedding` feature flag | 384 (MiniLM) and others |

**Local embeddings** run entirely offline via [fastembed](https://github.com/Anush008/fastembed-rs) (ONNX Runtime) — no API keys, no network, zero cost per query. Models are downloaded once on first use (~30MB).

Supported local models: `all-MiniLM-L6-v2` (default), `all-MiniLM-L12-v2`, `BGE-small-en-v1.5`, `BGE-base-en-v1.5`, `BGE-large-en-v1.5`, `nomic-embed-text-v1`, `nomic-embed-text-v1.5` (plus quantized variants with `-q` suffix).

```toml
[memory]
type = "postgres"
database_url = "postgresql://localhost/heartbit"

[memory.embedding]
provider = "local"              # or "openai" or "none"
model = "all-MiniLM-L6-v2"     # optional (this is the default for local)
cache_dir = "/tmp/fastembed"    # optional model cache directory
```

Build with local embedding support:
```bash
cargo build --features local-embedding
```

### Sensors

Data ingestion pipeline from 7 external sources → triage → stories → daemon commands.

**Sources**: RSS, JMAP (Google Workspace / email), Webhook, Weather, Audio, Image, MCP — each implements the `Sensor` trait with `name()`, `modality()`, and `run()`.

**Triage**: per-modality classifiers score urgency and relevance. **Stories**: aggregate related events, deduplicate, and produce actionable `DaemonCommand` entries sent to Kafka.

### Daemon

Long-running Kafka-backed task execution with HTTP API.

- **Kafka consumer** loop processes `DaemonCommand` messages
- **Axum HTTP API** — submit/list/cancel tasks, stream events via SSE
- **Cron scheduler** — 6-field cron expressions for recurring tasks
- **Heartbeat pulse** — periodic awareness loop that reads `HEARTBIT.md` standing orders, checks todos, submits tasks with idle backoff
- **Bounded concurrency** — `max_concurrent_tasks` limits parallel agent runs
- **WebSocket + Telegram** — interactive channels via `InteractionBridge`
- **Multi-tenant isolation** — JWT/JWKS authentication extracts `UserContext` per request; tasks, memory, and workspaces scoped per user/tenant
- **A2A Agent Card** — `GET /.well-known/agent.json` for agent discovery (name, skills, auth schemes, endpoint)

#### Multi-Tenant Authentication

The daemon supports two auth modes (both can be active simultaneously):

| Mode | Config key | How it works |
|------|-----------|-------------|
| **Bearer tokens** | `daemon.auth.bearer_tokens` | Static API keys; supports multiple tokens for rotation |
| **JWT/JWKS** | `daemon.auth.jwks_url` | RS256 tokens verified against a JWKS endpoint; extracts `UserContext` |

JWT claim names are configurable to accommodate different identity providers:

```toml
[daemon.auth]
bearer_tokens = ["$YOUR_API_KEY"]          # static API keys
jwks_url = "https://idp.example.com/.well-known/jwks.json"
issuer = "https://idp.example.com"       # optional: validate iss claim
audience = "heartbit-daemon"             # optional: validate aud claim
user_id_claim = "sub"                    # default: "sub"
tenant_id_claim = "tid"                  # default: "tid"
roles_claim = "roles"                    # default: "roles"
```

When JWT auth is configured, authenticated requests carry a `UserContext` (user_id, tenant_id, roles) through the entire request lifecycle:
- **Memory** — wrapped with `NamespacedMemory` using `tenant:{tid}:user:{uid}` prefix; users never see each other's memories
- **Workspace** — scoped to `{base}/{tenant_id}/{user_id}/`
- **Tasks** — filtered by authenticated tenant
- **Audit** — records include `user_id`, `tenant_id`, and `delegation_chain`

#### Dynamic MCP Authentication

For multi-tenant MCP server access, the `AuthProvider` trait enables per-user token injection:

```rust
pub trait AuthProvider: Send + Sync {
    fn auth_header_for(&self, user_id: &str, tenant_id: &str)
        -> Pin<Box<dyn Future<Output = Result<Option<String>, Error>> + Send + '_>>;
}
```

- **`StaticAuthProvider`** — returns the same auth header for all users (backward-compatible)
- **`TokenExchangeAuthProvider`** — implements RFC 8693 token exchange against an IdP, caching per-user tokens

### Channels

Interactive agent-human communication layer.

`InteractionBridge` adapts callbacks (`OnText`, `OnInput`, `OnApproval`, `OnQuestion`) into async message channels. Currently integrated with:
- **WebSocket** sessions with session management
- **Telegram** bot adapter (DMs, streaming responses, HITL approval, keyboard menus)

### Guardrails

Four async hooks intercept the agent loop (standalone path only):

| Hook | When | Can do |
|------|------|--------|
| `pre_llm` | Before each LLM call | Modify the `CompletionRequest` |
| `post_llm` | After LLM response | `Allow` or `Deny { reason }` |
| `pre_tool` | Before each tool call | `Allow` or `Deny { reason }` |
| `post_tool` | After each tool call | Inspect or modify `ToolOutput` |

Registered as `Vec<Arc<dyn Guardrail>>` — first `Deny` wins.

Built-in guardrails:

| Guardrail | What it does |
|-----------|-------------|
| `ContentFenceGuardrail` | Block/allow based on content patterns |
| `InjectionClassifierGuardrail` | Detect prompt injection attempts (warn or deny mode) |
| `PiiGuardrail` | Detect PII (email, phone, SSN, credit card) — redact, warn, or deny |
| `ToolPolicyGuardrail` | Per-tool allow/deny rules with input constraints (patterns, max length) |
| `LlmJudgeGuardrail` | Safety evaluation via a cheap judge model (fail-open on timeout) |
| `SensorSecurityGuardrail` | Sensor-specific security rules |
| `ConditionalGuardrail` | Apply guardrails conditionally based on agent/context |
| `GuardrailChain` | Compose multiple guardrails; `WarnToDeny` escalates warnings |

### Context Management

Three strategies control how message history is trimmed before LLM calls:

| Strategy | Behavior |
|----------|----------|
| `Unlimited` | No trimming (default) |
| `SlidingWindow { max_tokens }` | Keep system + recent messages within budget; tool use/result pairs kept together |
| `Summarize { threshold }` | LLM-generated summary injected when context exceeds threshold |

Additional features: session pruning (trim old tool results), recursive summarization (cluster-then-summarize for long conversations), auto-compaction on context overflow.

### Knowledge Base

Document retrieval for agent RAG:

- `InMemoryKnowledgeBase` — keyword search over indexed chunks
- Paragraph-aware chunking with configurable `chunk_size` and `chunk_overlap`
- Loaders: file, glob, URL (with HTML tag stripping)
- FNV-1a hash for deterministic chunk IDs
- Agent gets a `knowledge_search` tool (standalone path only)

### Workflow Agents (Deterministic Orchestration)

Three workflow agent types provide deterministic pipelines without LLM cost:

| Agent | What it does |
|-------|-------------|
| `SequentialAgent` | Chains agents: output of one becomes input of the next |
| `ParallelAgent` | Runs agents concurrently via `tokio::JoinSet`, merges results |
| `LoopAgent` | Repeats an agent until `should_stop(text)` returns true or max iterations |

All return `AgentOutput` with accumulated `TokenUsage`. Builder pattern: `.agent()`, `.agents()`, `.max_iterations()`, `.should_stop()`.

```rust
use heartbit::{SequentialAgent, ParallelAgent, LoopAgent};

// Pipeline: researcher → writer → reviewer
let pipeline = SequentialAgent::builder()
    .agent(researcher)
    .agent(writer)
    .agent(reviewer)
    .build()?;
let output = pipeline.run("Analyze Rust ecosystem").await?;
```

### Eval Framework

Built-in evaluation framework for testing agent behavior:

```rust
use heartbit::{EvalCase, EvalRunner, TrajectoryScorer, KeywordScorer};

let case = EvalCase::new("research-task", "Find info about Rust")
    .expect_tool("websearch")
    .expect_output_contains("Rust")
    .reference_output("Rust is a systems programming language");

let runner = EvalRunner::new(vec![case])
    .scorer(TrajectoryScorer)    // tool call sequence matching
    .scorer(KeywordScorer)       // output keyword checking
    .scorer(SimilarityScorer);   // cosine similarity to reference

let summary = runner.run(agent).await?;
println!("Pass rate: {:.0}%", summary.pass_rate * 100.0);
```

### Audit Trail

`AuditTrail` logs agent decisions, tool calls, and guardrail outcomes. `InMemoryAuditTrail` for development, `PostgresAuditTrail` for production persistence. In multi-tenant mode, audit records include `user_id`, `tenant_id`, and `delegation_chain` (RFC 8693 actor chain) for full provenance tracking.

### Durable Execution (Restate)

Crash-resilient agent workflows via [Restate SDK 0.8](https://restate.dev/):

- `AgentService` — `#[restate_sdk::service]` with `llm_call` + `tool_call` activities
- `AgentWorkflow` — `#[restate_sdk::workflow]` durable ReAct loop with replay
- `OrchestratorWorkflow` — delegates to child `AgentWorkflow` instances
- Virtual objects: `Blackboard` (shared state), `Budget` (token tracking), `CircuitBreaker` (LLM fault tolerance), `Scheduler` (recurring tasks)

## CLI Reference

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

[provider.cascade]                    # optional: try cheaper models first
enabled = true
[[provider.cascade.tiers]]
model = "anthropic/claude-3.5-haiku"  # cheapest tier tried first
[provider.cascade.gate]
type = "heuristic"                    # escalate if response is low-quality
min_output_tokens = 10                # escalate on very short responses
accept_tool_calls = false             # escalate if cheap model wants to use tools
escalate_on_max_tokens = false        # escalate on max_tokens stop reason

[orchestrator]
max_turns = 10
max_tokens = 4096
run_timeout_seconds = 300             # wall-clock deadline for the entire run
routing = "auto"                      # "auto", "always_orchestrate", or "single_agent"
dispatch_mode = "parallel"            # "parallel" or "sequential" (sub-agent dispatch)
reasoning_effort = "high"             # "high", "medium", "low", or "none"
tool_profile = "standard"             # "conversational", "standard", or "full"

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
reasoning_effort = "medium"           # per-agent override
tool_profile = "full"                 # per-agent override
context_strategy = { type = "sliding_window", max_tokens = 100000 }
# context_strategy = { type = "summarize", threshold = 80000 }
# context_strategy = { type = "unlimited" }

[agents.session_prune]                # optional: trim old tool results before LLM calls
keep_recent_n = 2                     # keep N most recent message pairs at full fidelity
pruned_tool_result_max_bytes = 200    # truncate older tool results to this size
preserve_task = true                  # keep the first user message (task) intact

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

[memory.embedding]                    # optional: enables hybrid retrieval (BM25 + vector)
provider = "local"                    # "openai", "local", or "none" (default)
model = "all-MiniLM-L6-v2"           # model name (provider-specific)
cache_dir = "/tmp/fastembed"          # local provider only: model cache directory
# api_key_env = "OPENAI_API_KEY"     # openai provider only

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

[daemon.auth]                       # optional: daemon API authentication
bearer_tokens = ["$YOUR_API_KEY"]     # static API keys (multiple for rotation)
jwks_url = "https://idp.example.com/.well-known/jwks.json"  # JWT/JWKS auth
issuer = "https://idp.example.com"  # optional: validate iss claim
audience = "heartbit-daemon"        # optional: validate aud claim
# user_id_claim = "sub"             # JWT claim for user ID (default: "sub")
# tenant_id_claim = "tid"           # JWT claim for tenant ID (default: "tid")
# roles_claim = "roles"             # JWT claim for roles (default: "roles")

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

## Environment Variables

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
| `HEARTBIT_A2A_AGENTS` | — | Comma-separated A2A agent URLs |
| `HEARTBIT_REASONING_EFFORT` | — | Reasoning effort level (`high`, `medium`, `low`, `none`) |
| `HEARTBIT_ENABLE_REFLECTION` | `false` | Enable reflective reasoning (`1` or `true`) |
| `HEARTBIT_COMPRESSION_THRESHOLD` | — | Token threshold for context compression |
| `HEARTBIT_MAX_TOOLS_PER_TURN` | — | Max tool calls per turn |
| `HEARTBIT_TOOL_PROFILE` | — | Tool pre-filtering (`conversational`, `standard`, `full`) |
| `HEARTBIT_MAX_IDENTICAL_TOOL_CALLS` | — | Doom loop detection threshold |
| `HEARTBIT_SESSION_PRUNE` | `false` | Enable session pruning of old tool results (`1` or `true`) |
| `HEARTBIT_RECURSIVE_SUMMARIZATION` | `false` | Enable cluster-then-summarize for long conversations (`1` or `true`) |
| `HEARTBIT_REFLECTION_THRESHOLD` | — | Cumulative importance threshold to trigger reflection |
| `HEARTBIT_CONSOLIDATE_ON_EXIT` | `false` | Consolidate memories at session end (`1` or `true`) |
| `HEARTBIT_MEMORY` | — | Memory backend (`in_memory` or a PostgreSQL URL) |
| `HEARTBIT_LSP_ENABLED` | `false` | Enable LSP integration (`1` or `true`) |
| `HEARTBIT_OBSERVABILITY` | `production` | Observability mode (`production`, `analysis`, `debug`, `off`) |
| `HEARTBIT_TELEGRAM_TOKEN` | — | Telegram bot token (daemon mode) |
| `HEARTBIT_API_KEY` | — | API key for daemon HTTP authentication |
| `EXA_API_KEY` | — | Exa AI API key (for `websearch` built-in tool) |
| `RUST_LOG` | — | Tracing filter (e.g. `info`, `debug`) |

## Library Usage

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

## Extending Heartbit

### Adding a New Tool

Implement the `Tool` trait and register it with the agent builder.

```rust
use heartbit::{Tool, ToolDefinition, ToolOutput, Error};
use serde_json::Value;
use std::pin::Pin;
use std::future::Future;

pub struct MyTool;

impl Tool for MyTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new("my_tool", "Does something useful")
            .with_parameter("input", "string", "The input value", true)
    }

    fn execute(
        &self,
        input: Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>> {
        Box::pin(async move {
            let input_str = input["input"].as_str().unwrap_or_default();
            Ok(ToolOutput::text(format!("Processed: {input_str}")))
        })
    }
}

// Register with an agent:
// AgentRunnerBuilder::new(provider).tools(vec![Arc::new(MyTool)]).build()
```

### Adding an LLM Provider

Implement the `LlmProvider` trait. For dynamic dispatch, the framework uses `BoxedProvider` which wraps a `DynLlmProvider` adapter.

```rust
use heartbit::llm::{LlmProvider, CompletionRequest, CompletionResponse};
use heartbit::Error;

pub struct MyProvider { /* ... */ }

impl LlmProvider for MyProvider {
    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, Error> {
        // Send request to your LLM, return response
        todo!()
    }

    // Optional: override for streaming support
    // async fn stream_complete(&self, request, on_text) -> Result<CompletionResponse, Error>

    fn model_name(&self) -> Option<&str> {
        Some("my-model")
    }
}

// Wrap for use: Arc::new(BoxedProvider::new(MyProvider { ... }))
```

### Adding a Sensor Source

Implement the `Sensor` trait for a new data source in the daemon pipeline.

```rust
use heartbit::sensor::{Sensor, SensorModality};
use heartbit::Error;
use rdkafka::producer::FutureProducer;
use tokio_util::sync::CancellationToken;
use std::pin::Pin;
use std::future::Future;

pub struct MySensor { /* config */ }

impl Sensor for MySensor {
    fn name(&self) -> &str { "my_source" }

    fn modality(&self) -> SensorModality { SensorModality::Text }

    fn kafka_topic(&self) -> &str { "heartbit.sensors.my_source" }

    fn run(
        &self,
        producer: FutureProducer,
        cancel: CancellationToken,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>> {
        Box::pin(async move {
            // Poll your source, produce SensorEvent messages to Kafka
            // Check cancel.is_cancelled() periodically
            Ok(())
        })
    }
}
```

### Adding a Guardrail

Implement the `Guardrail` trait. Override only the hooks you need — all default to `Allow`.

```rust
use heartbit::agent::guardrail::{Guardrail, GuardAction};
use heartbit::llm::types::ToolCall;
use heartbit::tool::ToolOutput;
use heartbit::Error;
use std::pin::Pin;
use std::future::Future;

pub struct MyGuardrail;

impl Guardrail for MyGuardrail {
    fn pre_tool(
        &self,
        call: &ToolCall,
    ) -> Pin<Box<dyn Future<Output = Result<GuardAction, Error>> + Send + '_>> {
        let name = call.name.clone();
        Box::pin(async move {
            if name == "bash" {
                Ok(GuardAction::Deny {
                    reason: "Bash execution is not allowed".into(),
                })
            } else {
                Ok(GuardAction::Allow)
            }
        })
    }
}

// Register: AgentRunnerBuilder::new(provider).guardrails(vec![Arc::new(MyGuardrail)]).build()
```

## Built-in Tools

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

### Cross-Agent Coordination

**Blackboard** — shared `Key -> Value` store. Sub-agents get `blackboard_read`, `blackboard_write`, `blackboard_list` tools. After each sub-agent completes, its result is written to `"agent:{name}"`.

### Structured Output

Set `response_schema` (JSON Schema) on an agent. A synthetic `__respond__` tool is injected and `tool_choice` forced to `Any`. The agent calls `__respond__` to produce structured JSON in `AgentOutput::structured`.

### Human-in-the-Loop

`--approve` flag enables interactive approval before each tool execution round. Denied tools receive error results — the LLM can adjust and retry. In Restate path, approval uses per-turn promise keys.

### Streaming

`on_text` callback receives text deltas as they arrive from the LLM. Both Anthropic and OpenRouter providers implement SSE streaming. Sub-agents don't stream — only the orchestrator.

### Agent Events

13 structured `AgentEvent` variants emitted via `OnEvent` callback:

`RunStarted`, `TurnStarted`, `LlmResponse`, `ToolCallStarted`, `ToolCallCompleted`, `ApprovalRequested`, `ApprovalDecision`, `SubAgentsDispatched`, `SubAgentCompleted`, `ContextSummarized`, `RunCompleted`, `GuardrailDenied`, `RunFailed`

Use `--verbose` to emit events as JSON to stderr.

### Cost Tracking

`estimate_cost(model, usage) -> Option<f64>` returns estimated USD cost for known models (Claude 4, 3.5, and 3 generations, including OpenRouter aliases). Accounts for cache read/write token rates. Displayed in CLI output after each run.

### OpenTelemetry

Add a `[telemetry]` section to your config to export traces via OTLP. Works with all commands (`run`, `chat`, `serve`). When absent, a simple `tracing_subscriber::fmt` subscriber is used instead.

## Durable Execution (Restate)

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

## Daemon Mode

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

2720+ tests. TDD mandatory — red/green/refactor for every feature.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

Join the community on [Telegram](https://t.me/heartbitagent).

## Disclaimer

Heartbit is **early-stage, capability-first software**. It is provided "as is" without warranty of any kind, express or implied.

**Security is not the primary design driver today.** The project optimizes for agent capability, extensibility, and developer velocity. While guardrails and permission systems exist, they have not been audited and should not be relied upon as a security boundary. Specifically:

- Agents can execute arbitrary shell commands, read/write files, and make network requests with the full permissions of the host process.
- LLM outputs are inherently unpredictable. Tool calls generated by the model may produce unintended side effects.
- MCP servers, sensors, and other external integrations expand the attack surface.
- There is no sandboxing, privilege separation, or capability-based security built in.

**Early adopters are responsible for:**
- Running Heartbit in appropriately isolated environments (containers, VMs, restricted user accounts).
- Implementing their own access controls, network policies, and monitoring.
- Evaluating the risk profile before deploying against sensitive data or production systems.

**The maintainers accept no liability** for data loss, security incidents, unintended actions, costs incurred from LLM API usage, or any other damages arising from the use of this software. Use at your own risk.

If you discover a security vulnerability, please report it privately via GitHub Security Advisories rather than opening a public issue.

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE).
