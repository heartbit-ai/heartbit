[![CI](https://github.com/heartbit-ai/heartbit/actions/workflows/ci.yml/badge.svg)](https://github.com/heartbit-ai/heartbit/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/heartbit)](https://crates.io/crates/heartbit)
[![docs.rs](https://img.shields.io/docsrs/heartbit)](https://docs.rs/heartbit)
[![Downloads](https://img.shields.io/crates/d/heartbit)](https://crates.io/crates/heartbit)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)
[![Telegram](https://img.shields.io/badge/Telegram-heartbitagent-blue?logo=telegram)](https://t.me/heartbitagent)

# Heartbit

Multi-agent enterprise runtime in Rust. Orchestrator spawns sub-agents that execute LLM-powered reasoning loops with parallel tool execution.

**Why Heartbit?**
- **Zero-copy agent loops** — ReAct cycle in pure Rust, no Python/Node overhead
- **Three execution paths** — standalone (zero infra), durable (Restate), daemon (Kafka)
- **Flat agent hierarchy** — orchestrator delegates to sub-agents, sub-agents never spawn further
- **8 orchestration patterns** — Sequential, Parallel, Loop, Debate, Voting, Mixture-of-Agents, DAG, Batch
- **Parallel tool execution** — `tokio::JoinSet` runs tools concurrently within each turn
- **4 LLM providers** — Anthropic, OpenRouter, Google Gemini, and any OpenAI-compatible endpoint
- **Production-grade** — 12 guardrails, workspace jailing, Landlock sandboxing, permission system, vault encryption, context management, MemGPT-style memory, cost tracking, OpenTelemetry
- **Smart context optimization** — ToolProfile-based filtering, per-agent builtin tool selection, recursive summarization, response caching, auto-compaction
- **15 agent templates + 10 skills** — pre-built agent archetypes and domain expertise modules
- **10 MCP presets** — one-line integrations for GitHub, Slack, Notion, Jira, and more
- **Reasoning & structured output** — extended thinking, reflection prompts, `__respond__` tool for schema-validated output
- **Local-first embeddings** — offline semantic search via ONNX Runtime (fastembed), no API keys required
- **Built-in eval framework** — 7 scorers (trajectory, keyword, similarity, cost, latency, tool count, safety) + A/B comparison
- **29 observable events** — streaming event system with OpenTelemetry, TTFT tracking, observability modes
- **Channel adapters** — Telegram, Discord, and Slack bot integrations
- **Dynamic agent spawning** — orchestrator creates specialist agents at runtime via `spawn_agent`

> **Early-stage software — capability over security.** Heartbit prioritizes **capability and velocity** at this stage of development. Security hardening is ongoing but not yet comprehensive. Agents execute tools (including shell commands) with the permissions of the host process. **Do not run untrusted workloads in production environments without your own sandboxing and access controls.** See [Disclaimer](#disclaimer) below.

## Quick Start

```bash
# Install
cargo install --git https://github.com/heartbit-ai/heartbit heartbit-cli

# Or use pre-built binaries
curl -fsSL https://raw.githubusercontent.com/heartbit-ai/heartbit/main/install.sh | bash

# Run (standalone mode, no config needed)
export ANTHROPIC_API_KEY=sk-...
heartbit "Analyze the Rust ecosystem"

# Interactive chat
heartbit chat
```

### As a Library

```rust
use std::sync::Arc;
use heartbit::{
    AnthropicProvider, BoxedProvider, RetryingProvider,
    AgentRunner,
};

let provider = Arc::new(BoxedProvider::new(
    RetryingProvider::with_defaults(
        AnthropicProvider::new(api_key, "claude-sonnet-4-20250514")
    )
));

let mut agent = AgentRunner::builder(provider)
    .system_prompt("You are a helpful assistant.")
    .on_text(Arc::new(|text| print!("{text}")))
    .build()?;

let output = agent.execute("Analyze the Rust ecosystem").await?;
println!("\nTokens: {} in / {} out", output.tokens_used.input_tokens,
    output.tokens_used.output_tokens);
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        heartbit-cli (bin)                          │
│  Commands: run | chat | serve | daemon | submit | status | approve │
│            result | templates | skills | init                      │
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
│  │  LlmProvider (Anthropic, OpenRouter, Gemini, OpenAI-compat)    ││
│  │  Memory (InMemory, Postgres, Namespaced)   Tool trait + MCP    ││
│  │  Guardrails (pre/post LLM & tool)          KnowledgeBase       ││
│  │  Context strategies + caching              Sensor pipeline      ││
│  │  Cost tracking + OTel                      Channel adapters     ││
│  │  Templates + Skills                        Permission system    ││
│  │  Vault (encrypted credentials)             Landlock sandbox     ││
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
| **Orchestrator** | Dispatches tasks to sub-agents via `delegate_task` / `form_squad` / `spawn_agent` | `Orchestrator<P>` |
| **Tool** | A capability the agent can invoke (bash, read, MCP, custom) | `Arc<dyn Tool>` |
| **LlmProvider** | Sends completion requests to an LLM (Anthropic, OpenRouter, Gemini, OpenAI-compat) | `Arc<BoxedProvider>` |
| **Guardrail** | Intercepts the agent loop at 4 hook points | `Arc<dyn Guardrail>` |
| **Memory** | Persistent agent memory with recall scoring | `Arc<dyn Memory>` |
| **Template** | Pre-built agent archetype with system prompt, tools, and settings | `AgentTemplate` |

## Features

<details>
<summary><strong>Multi-Agent Orchestration</strong></summary>

The orchestrator is an `AgentRunner` with three delegation tools. Sub-agents do NOT spawn further (flat hierarchy).

- **`delegate_task`** — independent parallel subtasks
- **`form_squad`** — collaborative subtasks sharing a `Blackboard`
- **`spawn_agent`** — dynamic specialist agents created at runtime (with token budget tracking)
- **Routing** — `Auto`, `AlwaysOrchestrate`, or `SingleAgent` modes with failure escalation
- **Per-agent providers** — each sub-agent can use a different LLM model
- **Per-agent builtins** — each sub-agent can load a different subset of builtin tools

```rust
use heartbit::{Orchestrator, SubAgentConfig};

let mut orchestrator = Orchestrator::builder(provider.clone())
    .sub_agent("researcher", "Research specialist", "You research.")
    .sub_agent("writer", "Writing specialist", "You write.")
    .on_text(Arc::new(|text| print!("{text}")))
    .build()?;

let output = orchestrator.run("Write an article about Rust").await?;
```

</details>

<details>
<summary><strong>LLM Provider Stack</strong></summary>

Four providers plus composable wrappers:

- **`AnthropicProvider`** — Claude models with SSE streaming and prompt caching
- **`OpenRouterProvider`** — 100+ models via OpenRouter API
- **`GeminiProvider`** — Google Gemini models
- **`OpenAICompatProvider`** — any OpenAI-compatible endpoint (vLLM, Ollama, Azure, etc.)
- **`CascadingProvider`** — tries cheaper models first, escalates on gate rejection
- **`RetryingProvider`** — exponential backoff on 429/500/502/503/529
- **Reasoning mode** — extended thinking with configurable `ReasoningEffort` (low/medium/high) and token budgets
- **Reflection** — automatic self-critique prompts for improved response quality
- **Structured output** — `__respond__` tool injection with JSON schema validation

</details>

<details>
<summary><strong>Templates & Skills</strong></summary>

15 built-in agent templates and 10 domain skills for rapid agent composition.

**Templates** — pre-configured agent archetypes:
`coder`, `researcher`, `planner`, `reviewer`, `debugger`, `writer`, `ops`, `orchestrator`, `security-auditor`, `test-engineer`, `architect`, `data-scientist`, `analyst`, `customer-support`, `translator`

**Skills** — domain expertise injected into system prompts:
`rust-expert`, `python-expert`, `typescript-expert`, `docker`, `kubernetes`, `security`, `sql-expert`, `api-design`, `testing`, `git-expert`

```toml
[[agents]]
name = "dev"
template = "coder"
skills = ["rust-expert", "docker"]
```

Discovery: bundled → `~/.config/heartbit/templates/` → `.heartbit/templates/` (walk to git root)

</details>

<details>
<summary><strong>MCP Presets</strong></summary>

10 bundled MCP server presets for one-line integrations:

`github`, `gitlab`, `slack`, `notion`, `postgresql`, `brave-search`, `sentry`, `linear`, `google-calendar`, `jira`

```toml
[[agents]]
name = "dev"
mcp_servers = [{ command = "npx", args = ["-y", "@modelcontextprotocol/server-github"] }]
```

MCP client supports: tools, resources (as tools or context), prompts, sampling, roots, token exchange (RFC 8693). Protocol version: `2025-11-25`.

</details>

<details>
<summary><strong>Per-Agent Builtin Tool Selection</strong></summary>

Control which builtin tools each agent receives. When absent, all builtins load (backward compatible). Empty list disables all builtins for MCP-only agents.

```toml
[[agents]]
name = "researcher"
builtin_tools = ["websearch", "webfetch"]  # only these 2

[[agents]]
name = "publisher"
builtin_tools = []  # MCP-only, no builtins
```

Library API:
```rust
let config = BuiltinToolsConfig {
    allowlist: Some(vec!["websearch".into(), "webfetch".into()]),
    ..Default::default()
};
let tools = builtin_tools(config);
```

18 known builtins: `bash`, `read`, `write`, `edit`, `grep`, `glob`, `list`, `patch`, `webfetch`, `websearch`, `image_generate`, `tts`, `skill`, `todoread`, `todowrite`, `question`, `twitter_post`, `todo_manage`. Unknown names are rejected at config parse time.

</details>

<details>
<summary><strong>Memory System</strong></summary>

MemGPT-inspired memory with composite recall scoring and hybrid retrieval.

- **Backends**: `InMemoryStore`, `PostgresMemoryStore` (pgvector), `NamespacedMemory`
- **Types**: Episodic, Semantic, Reflection
- **Recall**: BM25 + Park et al. composite + optional vector cosine (hybrid RRF)
- **Decay**: Ebbinghaus strength decay (~6-day half-life)
- **Local embeddings**: offline ONNX (fastembed), no API keys needed
- **Institutional memory**: 3-tier namespace (user/agent/session), cross-context knowledge bridge, role-gated shared writes

See [docs/memory.md](docs/memory.md) for full details.

</details>

<details>
<summary><strong>Guardrails</strong></summary>

Four async hooks intercept the agent loop: `pre_llm`, `post_llm`, `pre_tool`, `post_tool`.

12 built-in guardrail types:

| Guardrail | Description |
|-----------|-------------|
| `ContentFenceGuardrail` | Content boundary enforcement with keyword/regex matching |
| `InjectionClassifierGuardrail` | Multi-mode prompt injection detection |
| `PiiGuardrail` | PII detection and redaction |
| `ToolPolicyGuardrail` | Tool-level authorization with input constraints |
| `LlmJudgeGuardrail` | LLM-as-judge for output quality validation |
| `SecretScannerGuardrail` | API key, token, and credential detection in outputs |
| `ActionBudgetGuardrail` | Per-action execution cost/rate limiting |
| `BehavioralMonitorGuardrail` | Behavioral rule monitoring and enforcement |
| `SensorSecurityGuardrail` | Sensor pipeline trust-level enforcement |
| `GuardrailChain` | Composable guardrail pipelines |
| `ConditionalGuardrail` | Conditional execution based on context |
| `WarnToDeny` | Escalate warnings to hard denials |

</details>

<details>
<summary><strong>Security</strong></summary>

- **Workspace jailing** — all filesystem tools enforce path boundaries, reject `../` escapes and symlink traversal
- **Landlock sandboxing** — Linux kernel-level filesystem ACLs for bash subprocesses (unprivileged, works in Docker)
- **Vault encryption** — AES-256-GCM encrypted credential store with Argon2 key derivation (`~/.heartbit/vault.enc`)
- **Permission system** — human-in-the-loop approval with persistent learned rules and glob patterns
- **Secret scanning** — detects API keys, tokens, and credentials in agent outputs
- **Behavioral monitoring** — rule-based behavioral analysis with anomaly detection
- **Action budgets** — per-action execution limits to prevent runaway agents
- **Kill switch** — immediate agent termination on guardrail trigger
- **Doom loop detection** — exact and fuzzy duplicate tool-call detection with configurable thresholds
- **SSRF protection** — HTTP redirects disabled on MCP and token-exchange clients
- **Environment policy** — restrictive env var allowlists for bash in daemon mode

</details>

<details>
<summary><strong>Context Management</strong></summary>

- **Unlimited** — no trimming (default)
- **SlidingWindow** — keep system + recent messages within token budget
- **Summarize** — LLM-generated summary when context exceeds threshold, recursive summarization for long conversations
- **ToolProfile** — Conversational/Standard/Full query classification to reduce input tokens
- **Per-agent builtin selection** — load only needed tools to reduce context usage
- Auto-compaction on context overflow, doom loop detection, tool name repair
- Session pruning, response caching (LRU with FNV-1a hash keys)

</details>

<details>
<summary><strong>Workflow Agents (Deterministic)</strong></summary>

Eight orchestration patterns — three simple, five advanced:

- `SequentialAgent` — chains output → input across agents
- `ParallelAgent` — concurrent execution via `tokio::JoinSet`
- `LoopAgent` — repeats until stop condition or max iterations
- `DebateAgent` — multi-round debate between agents with a judge that picks the winner
- `VotingAgent` — parallel proposals with configurable majority vote and tie-breaking
- `MixtureOfAgentsAgent` — proposer → synthesizer layers (Mixture-of-Agents pattern)
- `DagAgent` — directed acyclic graph with conditional edges and transforms (petgraph)
- `BatchExecutor` — semaphore-controlled concurrent batch processing

Workflow observability via `WorkflowNodeStarted`, `WorkflowNodeCompleted`, `WorkflowNodeFailed` events.

</details>

<details>
<summary><strong>Eval Framework</strong></summary>

Built-in evaluation for testing agent behavior with 7 scorers:

- `TrajectoryScorer` — tool call sequence matching
- `KeywordScorer` — output keyword checking
- `SimilarityScorer` — cosine similarity to reference
- `CostScorer` — token cost budget enforcement
- `LatencyScorer` — execution time budget enforcement
- `ToolCallCountScorer` — tool call count budget enforcement
- `SafetyScorer` — guardrail violation detection

Plus `EvalComparison` for A/B regression testing between agent versions. All types derive `Serialize` for JSON reports.

See [docs/eval.md](docs/eval.md) for full details.

</details>

<details>
<summary><strong>Daemon Mode</strong></summary>

Long-running Kafka-backed task execution: HTTP API, cron scheduling, WebSocket + Telegram/Discord/Slack channels, multi-tenant JWT auth, A2A agent card.

See [docs/daemon.md](docs/daemon.md) for full details.

</details>

<details>
<summary><strong>Sensor Pipeline</strong></summary>

7 sources (RSS, Email/JMAP, Webhook, Weather, Audio, Image, MCP) → triage → story correlation → daemon commands.

See [docs/sensors.md](docs/sensors.md) for full details.

</details>

<details>
<summary><strong>Durable Execution (Restate)</strong></summary>

Crash-resilient workflows via Restate SDK 0.8: durable ReAct loops, exactly-once tool execution, token budget tracking, circuit breaker.

See [docs/restate.md](docs/restate.md) for full details.

</details>

<details>
<summary><strong>Observability</strong></summary>

29 `AgentEvent` variants across 6 categories (lifecycle, LLM, tool, guardrail, orchestration, workflow). Streaming via `OnEvent` callback, JSON output with `--verbose`, OpenTelemetry OTLP export, TTFT tracking, payload truncation (64KB).

</details>

## Installation

### Pre-built binaries

```bash
curl -fsSL https://raw.githubusercontent.com/heartbit-ai/heartbit/main/install.sh | bash
```

### From source

```bash
cargo install --git https://github.com/heartbit-ai/heartbit heartbit-cli
```

### As a library

```bash
cargo add heartbit
```

### Docker

```bash
docker pull ghcr.io/heartbit-ai/heartbit:latest
docker compose up -d   # Restate + worker + Kafka
```

### Prerequisites

Building from source requires: Rust stable, cmake, libssl-dev, pkg-config (for rdkafka).

## Feature Flags

| Feature | Dependencies | What it enables |
|---------|-------------|-----------------|
| `core` (default) | — | Agent runner, orchestrator, LLM providers, tools, memory, config |
| `kafka` | `rdkafka` | Kafka consumer/producer |
| `daemon` | kafka + `cron`, `prometheus` | Daemon with HTTP API, cron, metrics |
| `sensor` | daemon + `quick-xml`, `hmac` | 7 sensor sources, triage, stories |
| `restate` | `restate-sdk 0.8` | Durable workflow execution |
| `postgres` | `sqlx`, `pgvector` | PostgreSQL memory + task store |
| `a2a` | `a2a-sdk` | Agent-to-Agent protocol |
| `telegram` | `teloxide` | Telegram bot channel |
| `discord` | — | Discord bot channel |
| `slack` | — | Slack bot channel |
| `sandbox` | — | Linux Landlock filesystem sandboxing |
| `vault` | `aes-gcm`, `argon2` | Encrypted credential vault |
| `macro` | `heartbit-macro` | `#[heartbit_tool]` proc-macro for deriving Tool trait |
| `local-embedding` | `fastembed` | Local ONNX embeddings (no API keys) |
| `full` | all above (except `local-embedding`, `macro`, `sandbox`) | Everything |

## Configuration

See [docs/configuration.md](docs/configuration.md) for the full TOML reference and environment variables.

Minimal config example:

```toml
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "researcher"
description = "Research specialist"
system_prompt = "You are a research specialist."

[[agents]]
name = "writer"
description = "Writing specialist"
system_prompt = "You are a writing specialist."
```

With templates, skills, and per-agent tool selection:

```toml
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "dev"
description = "Full-stack developer"
template = "coder"
skills = ["rust-expert", "docker"]
builtin_tools = ["read", "write", "edit", "grep", "glob", "bash", "patch"]

[[agents]]
name = "researcher"
description = "Web researcher"
template = "researcher"
builtin_tools = ["websearch", "webfetch"]
```

## CLI Reference

```
heartbit [run|chat|serve|daemon|submit|status|approve|result|templates|skills|init] <args>
heartbit <task>                  # shorthand for 'run'
```

| Command | Description |
|---------|-------------|
| `run <task>` | Execute in standalone mode |
| `chat` | Interactive multi-turn REPL |
| `serve` | Start Restate HTTP worker |
| `daemon` | Run Kafka-backed daemon |
| `submit <task>` | Submit for durable execution |
| `status <id>` | Query workflow status |
| `approve <id>` | Send approval signal |
| `result <id>` | Get completed workflow result |
| `templates list\|show` | Browse built-in agent templates |
| `skills list\|show` | Browse built-in domain skills |
| `init <template>` | Generate starter config from a template |

| Flag | Description |
|------|-------------|
| `--config <path>` | Path to `heartbit.toml` |
| `--approve` | Enable human-in-the-loop approval |
| `-v`, `--verbose` | Emit agent events as JSON to stderr |

## Protocol Coverage

| Protocol | Version | Coverage |
|----------|---------|----------|
| MCP client | 2025-11-25 | Tools, resources, prompts, sampling, roots (Streamable HTTP + stdio) |
| A2A | 0.2.x | Agent card + 8-state task lifecycle |
| RFC 8693 | stable | Token exchange for MCP on-behalf-of |

## Documentation

Full documentation is available at **[heartbit.ai](https://heartbit.ai)**.

| Guide | Description |
|-------|-------------|
| [Configuration](docs/configuration.md) | Full TOML reference + environment variables |
| [Built-in Tools](docs/tools.md) | 18 tools reference + custom tool guide |
| [Memory](docs/memory.md) | MemGPT-style memory architecture |
| [Daemon](docs/daemon.md) | Kafka-backed daemon setup |
| [Sensors](docs/sensors.md) | Sensor pipeline |
| [Telegram](docs/telegram.md) | Telegram bot integration |
| [Restate](docs/restate.md) | Durable execution |
| [Eval](docs/eval.md) | Eval framework |

## Examples

See [`crates/heartbit/examples/`](crates/heartbit/examples/) for runnable examples:

```bash
cargo run --example hello_agent -p heartbit     # Minimal single agent
cargo run --example simple_agent -p heartbit    # Agent with builtin tools
cargo run --example multi_agent -p heartbit     # Orchestrator with sub-agents
cargo run --example custom_tool -p heartbit     # Implementing the Tool trait
cargo run --example mcp_agent -p heartbit       # MCP tool integration
cargo run --example guardrails -p heartbit      # LLM judge guardrail
cargo run --example memory -p heartbit          # Memory-enabled agent
cargo run --example eval -p heartbit            # Running evals
```

## Development

```bash
# Quality gate (must pass before every commit)
cargo fmt -- --check && cargo clippy -- -D warnings && cargo test
```

2000+ tests. TDD mandatory — red/green/refactor for every feature.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

Join the community on [Telegram](https://t.me/heartbitagent).

## Disclaimer

Heartbit is **early-stage, capability-first software**. It is provided "as is" without warranty of any kind, express or implied.

**Security is not the primary design driver today.** The project optimizes for agent capability, extensibility, and developer velocity. While guardrails and permission systems exist, they have not been audited and should not be relied upon as a security boundary. Specifically:

- Agents can execute arbitrary shell commands, read/write files, and make network requests with the full permissions of the host process.
- LLM outputs are inherently unpredictable. Tool calls generated by the model may produce unintended side effects.
- MCP servers, sensors, and other external integrations expand the attack surface.
- Landlock sandboxing and workspace jailing reduce but do not eliminate risk.

**Early adopters are responsible for:**
- Running Heartbit in appropriately isolated environments (containers, VMs, restricted user accounts).
- Implementing their own access controls, network policies, and monitoring.
- Evaluating the risk profile before deploying against sensitive data or production systems.

**The maintainers accept no liability** for data loss, security incidents, unintended actions, costs incurred from LLM API usage, or any other damages arising from the use of this software. Use at your own risk.

If you discover a security vulnerability, please report it privately via [GitHub Security Advisories](https://github.com/heartbit-ai/heartbit/security/advisories) rather than opening a public issue.

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE).
