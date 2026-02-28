# Heartbit

> Multi-agent enterprise runtime in Rust. Orchestrator spawns sub-agents that execute LLM-powered reasoning loops with parallel tool execution.

## Key Facts

- 2 crates: `heartbit` (lib), `heartbit-cli` (bin)
- 3 execution paths: standalone (zero infra), durable (Restate SDK 0.8), daemon (Kafka-backed)
- Flat agent hierarchy: orchestrator delegates to sub-agents, sub-agents never spawn further
- Parallel tool execution via `tokio::JoinSet`
- 14 built-in tools, 8 guardrails, MemGPT-style memory with Ebbinghaus decay
- Local-first ONNX embeddings (fastembed) or OpenAI API
- Built-in eval framework for agent behavior testing
- Multi-tenant daemon with JWT/JWKS auth, per-user memory namespacing, dynamic MCP token injection
- A2A Agent Card (`/.well-known/agent.json`) for agent discovery
- Integrations: Telegram bot, Google Workspace (JMAP email), RSS, webhooks via sensor pipeline
- 2720+ tests, TDD mandatory, zero `unwrap()` in library code

## Feature Flags

| Feature | Dependencies | Purpose |
|---------|-------------|---------|
| `core` (default) | — | Agent runner, orchestrator, LLM providers, tools, memory, config |
| `kafka` | `rdkafka` | Kafka consumer/producer |
| `daemon` | kafka + `cron`, `prometheus` | Long-running daemon with HTTP API, cron, metrics |
| `sensor` | daemon + `quick-xml`, `hmac`, `sha2` | 7 sensor sources, triage pipeline, story correlation |
| `restate` | `restate-sdk 0.8` | Durable workflow execution |
| `postgres` | `sqlx`, `pgvector` | PostgreSQL memory store + task store with vector search |
| `a2a` | `a2a-sdk` | Agent-to-Agent protocol |
| `telegram` | `teloxide` | Telegram bot (DMs, streaming, HITL) |
| `local-embedding` | `fastembed` | Local ONNX-based text embeddings (offline) |
| `full` | all above except `local-embedding` | Everything enabled — used by the CLI |

## Public Traits

### Tool

```rust
pub trait Tool: Send + Sync {
    fn definition(&self) -> ToolDefinition;
    fn execute(&self, input: Value) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>>;
}
```

### LlmProvider (RPITIT — not dyn-compatible)

```rust
pub trait LlmProvider: Send + Sync {
    fn complete(&self, request: CompletionRequest) -> impl Future<Output = Result<CompletionResponse, Error>> + Send;
    fn stream_complete(&self, request: CompletionRequest, on_text: &OnText) -> impl Future<Output = Result<CompletionResponse, Error>> + Send;
    fn model_name(&self) -> Option<&str>;
}
```

Use `BoxedProvider` for dynamic dispatch: `Arc::new(BoxedProvider::new(provider))`.

### DynLlmProvider (object-safe wrapper)

```rust
pub trait DynLlmProvider: Send + Sync {
    fn complete<'a>(&'a self, request: CompletionRequest) -> Pin<Box<dyn Future<Output = Result<CompletionResponse, Error>> + Send + 'a>>;
    fn stream_complete<'a>(&'a self, request: CompletionRequest, on_text: &'a OnText) -> Pin<Box<dyn Future<Output = Result<CompletionResponse, Error>> + Send + 'a>>;
    fn model_name(&self) -> Option<&str>;
}
```

Blanket impl for all `LlmProvider` types.

### Memory

```rust
pub trait Memory: Send + Sync {
    fn store(&self, entry: MemoryEntry) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>>;
    fn recall(&self, query: MemoryQuery) -> Pin<Box<dyn Future<Output = Result<Vec<MemoryEntry>, Error>> + Send + '_>>;
    fn update(&self, id: &str, content: String) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>>;
    fn forget(&self, id: &str) -> Pin<Box<dyn Future<Output = Result<bool, Error>> + Send + '_>>;
    fn add_link(&self, id: &str, related_id: &str) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>>;
    fn prune(&self, min_strength: f64, min_age: chrono::Duration) -> Pin<Box<dyn Future<Output = Result<usize, Error>> + Send + '_>>;
}
```

### KnowledgeBase

```rust
pub trait KnowledgeBase: Send + Sync {
    fn index(&self, chunk: Chunk) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>>;
    fn search(&self, query: KnowledgeQuery) -> Pin<Box<dyn Future<Output = Result<Vec<SearchResult>, Error>> + Send + '_>>;
    fn chunk_count(&self) -> Pin<Box<dyn Future<Output = Result<usize, Error>> + Send + '_>>;
}
```

### Guardrail

```rust
pub trait Guardrail: Send + Sync {
    fn pre_llm(&self, request: &mut CompletionRequest) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>>;
    fn post_llm(&self, response: &CompletionResponse) -> Pin<Box<dyn Future<Output = Result<GuardAction, Error>> + Send + '_>>;
    fn pre_tool(&self, call: &ToolCall) -> Pin<Box<dyn Future<Output = Result<GuardAction, Error>> + Send + '_>>;
    fn post_tool(&self, call: &ToolCall, output: &mut ToolOutput) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>>;
}
```

All hooks default to `Allow`. Override only what you need.

### Blackboard

```rust
pub trait Blackboard: Send + Sync {
    fn write(&self, key: &str, value: Value) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>>;
    fn read(&self, key: &str) -> Pin<Box<dyn Future<Output = Result<Option<Value>, Error>> + Send + '_>>;
    fn list_keys(&self) -> Pin<Box<dyn Future<Output = Result<Vec<String>, Error>> + Send + '_>>;
}
```

### EmbeddingProvider

```rust
pub trait EmbeddingProvider: Send + Sync {
    fn embed(&self, texts: &[&str]) -> Pin<Box<dyn Future<Output = Result<Vec<Vec<f32>>, Error>> + Send + '_>>;
    fn dimension(&self) -> usize;
}
```

### Sensor (feature: `sensor`)

```rust
pub trait Sensor: Send + Sync {
    fn name(&self) -> &str;
    fn modality(&self) -> SensorModality;
    fn kafka_topic(&self) -> &str;
    fn run(&self, producer: FutureProducer, cancel: CancellationToken) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>>;
}
```

### AuthProvider

```rust
pub trait AuthProvider: Send + Sync {
    fn auth_header_for<'a>(
        &'a self,
        user_id: &'a str,
        tenant_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<String>, Error>> + Send + 'a>>;
}
```

Two implementations: `StaticAuthProvider` (backward-compatible static header) and `TokenExchangeAuthProvider` (RFC 8693 token exchange with in-memory caching).

### CommandProducer (feature: `daemon`)

```rust
pub trait CommandProducer: Send + Sync {
    fn send_command<'a>(&'a self, topic: &'a str, key: &'a str, payload: &'a [u8]) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + 'a>>;
}
```

## Key Types

### Agent & Orchestration

- `AgentRunner<P>` — Executes one agent's ReAct loop (LLM → tools → repeat)
- `AgentOutput` — `result: String`, `tool_calls_made: usize`, `tokens_used: TokenUsage`, `structured: Option<Value>`, `estimated_cost_usd: Option<f64>`
- `Orchestrator<P>` — Multi-agent dispatch via `delegate_task` / `form_squad`
- `SubAgentConfig` — Builder struct: name, description, system_prompt, tools, max_turns, max_tokens, provider, guardrails, memory config, etc.
- `SequentialAgent<P>` — Chains agents: output of one becomes input of the next
- `ParallelAgent<P>` — Runs agents concurrently via `JoinSet`, merges results
- `LoopAgent<P>` — Repeats until `should_stop(text)` or max iterations

### LLM

- `CompletionRequest` — system, messages, tools, max_tokens, tool_choice, reasoning_effort
- `CompletionResponse` — content (Vec<ContentBlock>), stop_reason, usage, model
- `Message` — role (User/Assistant), content (Vec<ContentBlock>)
- `ContentBlock` — `Text`, `ToolUse`, `ToolResult`, `Image`, `Audio`
- `TokenUsage` — input_tokens, output_tokens, cache_creation_input_tokens, cache_read_input_tokens, reasoning_tokens (derives `Copy`)
- `StopReason` — `EndTurn`, `ToolUse`, `MaxTokens` (derives `Copy`)
- `ToolChoice` — `Auto`, `Any`, `Tool { name }`
- `ReasoningEffort` — `High`, `Medium`, `Low`, `None`

### Tool

- `ToolDefinition` — name, description, input_schema (JSON Schema)
- `ToolOutput` — content: String, is_error: bool; constructors: `.success()`, `.error()`, `.truncated(max_bytes)`
- `ToolCall` — name, id, input

### Memory

- `MemoryEntry` — id, agent, content, category, tags, created_at, last_accessed, access_count, importance (1-10), memory_type, keywords, summary, strength, related_ids, source_ids, embedding, confidentiality
- `MemoryQuery` — text, category, tags, agent, limit, memory_type, min_strength, query_embedding, max_confidentiality
- `MemoryType` — `Episodic` (default), `Semantic`, `Reflection`
- `Confidentiality` — `Public`, `Internal`, `Confidential`, `Restricted`

### Guardrail

- `GuardAction` — `Allow`, `Deny { reason }`, `Warn { reason }`

### Events & Callbacks

- `AgentEvent` — 18 variants: RunStarted, TurnStarted, LlmResponse, ToolCallStarted, ToolCallCompleted, ApprovalRequested, ApprovalDecision, SubAgentsDispatched, SubAgentCompleted, ContextSummarized, RunCompleted, GuardrailDenied, GuardrailWarned, RunFailed, LlmRetry, ModelEscalated, ToolDeselected, ReflectionTriggered
- `OnText = dyn Fn(&str) + Send + Sync` — streaming text callback
- `OnApproval = dyn Fn(&[ToolCall]) -> ApprovalDecision` — HITL approval
- `ApprovalDecision` — `Allow`, `Deny`, `AlwaysAllow`, `AlwaysDeny`
- `OnInput = dyn Fn() -> Pin<Box<dyn Future<Output = Option<String>> + Send>>` — interactive mode
- `OnEvent = Arc<dyn Fn(AgentEvent)>` — event callback
- `OnQuestion` — structured agent-to-user questions

### Multi-Tenant (feature: `daemon`)

- `UserContext` — user_id, tenant_id, roles (extracted from JWT claims)
- `JwksClient` — fetches and caches JWKS keys (5-min TTL, auto-refetch on key rotation)
- `JwtValidator` — validates RS256 JWTs against JWKS, extracts `UserContext` with configurable claim names
- `AuditRecord` — agent, turn, event_type, payload, usage, timestamp, user_id, tenant_id, delegation_chain

### Eval

- `EvalCase` — name, input, expected_tools, output_contains, reference_output
- `EvalResult` — case_name, passed, scores, actual_tools, actual_output, error
- `EvalRunner` — executor with pluggable scorers
- `EvalSummary` — total, passed, failed, errors, avg_score

### Error

```rust
pub enum Error {
    HTTP, Json, Api { status, message }, Agent, MaxTurnsExceeded, Truncated, RunTimeout,
    Mcp, A2a, Config, Store, Memory, Knowledge, Guardrail, Daemon, Sensor, Channel, Telegram,
    BudgetExceeded { used, limit },
    WithPartialUsage { source: Box<Error>, usage: TokenUsage },
}
```

## Builders

### AgentRunnerBuilder

```rust
AgentRunner::builder(provider)
    .name("agent")
    .system_prompt("You are...")
    .tools(vec![Arc::new(my_tool)])
    .max_turns(20).max_tokens(4096)
    .context_strategy(ContextStrategy::SlidingWindow { max_tokens: 100_000 })
    .summarize_threshold(80000)
    .tool_timeout(Duration::from_secs(60))
    .max_tool_output_bytes(16384)
    .run_timeout(Duration::from_secs(300))
    .on_text(Arc::new(|t| print!("{t}")))
    .on_approval(Arc::new(|_| ApprovalDecision::Allow))
    .on_event(Arc::new(|e| eprintln!("{e:?}")))
    .on_input(Arc::new(|| Box::pin(async { Some("user input".into()) })))
    .structured_schema(json_schema)
    .guardrails(vec![Arc::new(my_guardrail)])
    .memory(Arc::new(InMemoryStore::new()))
    .knowledge_base(Arc::new(InMemoryKnowledgeBase::new()))
    .reasoning_effort(ReasoningEffort::High)
    .enable_reflection(true)
    .tool_profile(ToolProfile::Standard)
    .max_identical_tool_calls(3)
    .session_prune_config(config)
    .build()?
```

### OrchestratorBuilder

```rust
Orchestrator::builder(provider)
    .sub_agent(SubAgentConfig { name, description, system_prompt, tools, ..Default::default() })
    .max_turns(10).max_tokens(4096)
    .blackboard(Arc::new(InMemoryBlackboard::new()))
    .shared_memory(Arc::new(InMemoryStore::new()))
    .dispatch_mode(DispatchMode::Parallel)
    .on_text(Arc::new(|t| print!("{t}")))
    .build()?
```

### Workflow Agents

```rust
// Sequential: output chains through agents
SequentialAgent::builder().agent(a1).agent(a2).agent(a3).build()?;

// Parallel: concurrent execution, merged results
ParallelAgent::builder().agent(a1).agent(a2).build()?;

// Loop: repeat until condition or max iterations
LoopAgent::builder().agent(a1).max_iterations(5).should_stop(Box::new(|text| text.contains("DONE"))).build()?;
```

### LlmJudgeGuardrailBuilder

```rust
LlmJudgeGuardrail::builder(judge_provider)
    .criterion("No harmful content")
    .criterion("Factually accurate")
    .timeout(Duration::from_secs(10))
    .evaluate_tool_inputs(true)
    .max_judge_tokens(256)
    .build()?
```

### Eval

```rust
let case = EvalCase::new("test", "Find info about Rust")
    .expect_tool("websearch")
    .expect_output_contains("Rust")
    .reference_output("Rust is a systems programming language");

let runner = EvalRunner::new(vec![case])
    .scorer(TrajectoryScorer)
    .scorer(KeywordScorer)
    .scorer(SimilarityScorer);

let summary = runner.run(agent).await?;
```

## LLM Provider Stack

Providers compose via wrapping:

```
CascadingProvider (optional, tries cheaper models first)
  └─ RetryingProvider (429/500/502/503/529 + network errors)
       └─ AnthropicProvider or OpenRouterProvider (SSE streaming, prompt caching)
```

- **CascadingProvider**: Tiers from cheapest to most expensive. HeuristicGate checks min_output_tokens, accept_tool_calls, escalate_on_max_tokens. Non-final tiers use `complete()` (not streaming).
- **RetryingProvider**: Exponential backoff (base_delay_ms, max_delay_ms). OnRetry callback.
- **AnthropicProvider**: `.with_prompt_caching()` — 3 cache breakpoints: system prompt, last tool def, second-to-last user message.
- **OpenRouterProvider**: OpenAI-compatible SSE format.

## Built-in Tools (14)

| Tool | Description |
|------|-------------|
| `bash` | Shell commands, working directory persists, timeout 120s/max 600s |
| `read` | File reading with line numbers, binary detection, max 256KB |
| `write` | File writing, creates parent dirs, read-before-write guard |
| `edit` | Exact string replacement (must appear exactly once) |
| `patch` | Unified diff application, single-pass hunk processing |
| `glob` | File pattern matching, skips hidden files |
| `grep` | Regex content search, uses `rg` when available |
| `list` | Directory tree listing, skips build artifacts |
| `webfetch` | HTTP GET with content extraction, max 5MB |
| `websearch` | Web search via Exa AI (requires `EXA_API_KEY`) |
| `todowrite` | Write/replace todo list, 1 in-progress item max |
| `todoread` | Read current todo list |
| `skill` | Load skills from `SKILL.md` files |
| `question` | Structured agent-to-user questions (requires `on_question` callback) |

## Built-in Guardrails (8)

| Guardrail | Purpose |
|-----------|---------|
| `ContentFenceGuardrail` | Block/allow based on content patterns |
| `InjectionClassifierGuardrail` | Detect prompt injection (warn or deny mode) |
| `PiiGuardrail` | Detect PII (email, phone, SSN, credit card) — redact, warn, or deny |
| `ToolPolicyGuardrail` | Per-tool allow/deny rules with input constraints |
| `LlmJudgeGuardrail` | Safety evaluation via cheap judge model (fail-open on timeout) |
| `SensorSecurityGuardrail` | Sensor-specific security rules (feature: `sensor`) |
| `ConditionalGuardrail` | Apply guardrails conditionally based on agent/context |
| `GuardrailChain` | Compose multiple guardrails; `WarnToDeny` escalates warnings |

## Memory System

- **Storage**: `InMemoryStore`, `PostgresMemoryStore` (pgvector), `NamespacedMemory` (3-tier: user/agent/session)
- **Recall**: BM25 keyword search (2x boost) + Park et al. composite (recency + importance + relevance + strength)
- **Hybrid retrieval**: BM25 + vector cosine similarity fused via Reciprocal Rank Fusion (RRF)
- **Ebbinghaus decay**: `effective_strength()` with decay rate 0.005/hr (~6-day half-life); +0.2 on access, capped at 1.0
- **Reflection**: `ReflectionTracker` triggers when cumulative importance exceeds threshold
- **Consolidation**: `ConsolidationPipeline` clusters by Jaccard keyword similarity, merges into Semantic entries
- **Pruning**: auto-prune weak memories at session end (min strength + min age)
- **Session pruning**: `SessionPruneConfig` trims old tool results before LLM calls
- **Embedding providers**: `NoopEmbedding`, `OpenAiEmbedding`, `LocalEmbeddingProvider` (fastembed, feature: `local-embedding`)
- **5 agent tools**: `memory_store`, `memory_recall`, `memory_update`, `memory_forget`, `memory_consolidate`

## Agent Loop Behavior

- **Parallel tool execution**: all tool calls in a turn run concurrently via `JoinSet`
- **Panic recovery**: panicked tool task produces error result without crashing
- **Context management**: `Unlimited`, `SlidingWindow`, `Summarize` strategies
- **Auto-compaction**: on `ContextOverflow`, summarizes history and retries (max once per turn pair)
- **Doom loop detection**: hashes tool-call batches; breaks cycle at threshold
- **Tool name repair**: Levenshtein distance ≤ 2 auto-corrects misspelled tool names
- **Tool pre-filtering**: `ToolProfile` (Conversational/Standard/Full) reduces tools per turn

## Configuration (TOML)

```toml
[provider]
name = "anthropic"                    # or "openrouter"
model = "claude-sonnet-4-20250514"
prompt_caching = true

[provider.retry]
max_retries = 3
base_delay_ms = 500

[provider.cascade]
enabled = true
[[provider.cascade.tiers]]
model = "anthropic/claude-3.5-haiku"
[provider.cascade.gate]
type = "heuristic"
min_output_tokens = 10

[orchestrator]
max_turns = 10
max_tokens = 4096
routing = "auto"                      # auto | always_orchestrate | single_agent
dispatch_mode = "parallel"            # parallel | sequential
reasoning_effort = "high"             # high | medium | low | none
tool_profile = "standard"             # conversational | standard | full

[[agents]]
name = "researcher"
description = "Research specialist"
system_prompt = "You are a research specialist."
mcp_servers = ["http://localhost:8000/mcp"]
max_turns = 20
context_strategy = { type = "sliding_window", max_tokens = 100000 }

[agents.session_prune]
keep_recent_n = 2
pruned_tool_result_max_bytes = 200

[memory]
type = "in_memory"                    # or "postgres" with database_url

[memory.embedding]
provider = "local"                    # openai | local | none
model = "all-MiniLM-L6-v2"

[daemon]
bind = "127.0.0.1:3000"
max_concurrent_tasks = 4

[daemon.auth]
bearer_tokens = ["your-api-key-1"]
jwks_url = "https://idp.example.com/.well-known/jwks.json"
issuer = "https://idp.example.com"
audience = "heartbit-daemon"
# user_id_claim = "sub"              # configurable claim names
# tenant_id_claim = "tid"
# roles_claim = "roles"

[telemetry]
otlp_endpoint = "http://localhost:4317"
```

## CLI

```
heartbit run <task>       # Standalone execution
heartbit chat             # Interactive REPL (200 max turns)
heartbit serve            # Restate HTTP worker
heartbit daemon           # Kafka-backed daemon with HTTP API
heartbit submit <task>    # Submit to Restate
heartbit status <id>      # Query workflow status
heartbit approve <id>     # Send approval signal
heartbit result <id>      # Get workflow result
```

Flags: `--config <path>`, `--approve` (HITL), `-v`/`--verbose` (events as JSON to stderr), `--bind <addr>`.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `OPENROUTER_API_KEY` | — | OpenRouter API key |
| `HEARTBIT_PROVIDER` | auto-detect | Force provider |
| `HEARTBIT_MODEL` | `claude-sonnet-4-20250514` | Model name |
| `HEARTBIT_MAX_TURNS` | `50`/`200` | Max agent turns |
| `HEARTBIT_PROMPT_CACHING` | `false` | Anthropic prompt caching |
| `HEARTBIT_SUMMARIZE_THRESHOLD` | `80000` | Context summarization threshold |
| `HEARTBIT_MAX_TOOL_OUTPUT_BYTES` | `32768` | Tool output truncation |
| `HEARTBIT_TOOL_TIMEOUT` | `120` | Tool timeout (seconds) |
| `HEARTBIT_MCP_SERVERS` | — | Comma-separated MCP URLs |
| `HEARTBIT_A2A_AGENTS` | — | Comma-separated A2A URLs |
| `HEARTBIT_REASONING_EFFORT` | — | high/medium/low/none |
| `HEARTBIT_TOOL_PROFILE` | — | conversational/standard/full |
| `HEARTBIT_SESSION_PRUNE` | `false` | Enable session pruning |
| `HEARTBIT_RECURSIVE_SUMMARIZATION` | `false` | Cluster-then-summarize |
| `HEARTBIT_REFLECTION_THRESHOLD` | — | Reflection importance threshold |
| `HEARTBIT_CONSOLIDATE_ON_EXIT` | `false` | Consolidate memories at exit |
| `HEARTBIT_MEMORY` | — | in_memory or PostgreSQL URL |
| `HEARTBIT_OBSERVABILITY` | `production` | production/analysis/debug/off |

## Module Map

```
crates/heartbit/src/
  agent/           AgentRunner, Orchestrator, context, guardrails, routing, events, workflow, audit
  auth/            JwksClient, JwtValidator → UserContext (JWT/JWKS authentication)
  llm/             LlmProvider, Anthropic, OpenRouter, cascade, retry, pricing, error_class
  tool/            Tool trait, MCP client (AuthProvider), A2A client, builtins/ (14 tools)
  memory/          Memory trait, BM25, scoring, reflection, consolidation, embedding, hybrid, namespaced
  knowledge/       KnowledgeBase trait, chunker, loader
  sensor/          Sensor trait, 7 sources, triage, stories, compression
  channel/         InteractionBridge, sessions, telegram/
  daemon/          DaemonCore, Kafka, cron, heartbeat pulse, store, metrics, types (UserContext)
  workflow/        Restate services, workflows, virtual objects
  store/           PostgreSQL task/audit store
  eval/            EvalCase, EvalRunner, scorers
  config.rs        HeartbitConfig from TOML (incl. AuthConfig)
  error.rs         Error enum (thiserror)
  lib.rs           Public API re-exports
```

## Re-exports (crate root)

**Always available**: `AgentRunner`, `AgentRunnerBuilder`, `AgentOutput`, `Orchestrator`, `OrchestratorBuilder`, `SubAgentConfig`, `SequentialAgent`, `ParallelAgent`, `LoopAgent`, `BoxedProvider`, `AnthropicProvider`, `OpenRouterProvider`, `CascadingProvider`, `RetryingProvider`, `Tool`, `ToolDefinition`, `ToolOutput`, `builtin_tools`, `McpClient`, `Memory`, `MemoryEntry`, `MemoryQuery`, `MemoryType`, `InMemoryStore`, `EmbeddingProvider`, `EmbeddingMemory`, `NoopEmbedding`, `OpenAiEmbedding`, `KnowledgeBase`, `InMemoryKnowledgeBase`, `Guardrail`, `GuardAction`, `ContentFenceGuardrail`, `InjectionClassifierGuardrail`, `PiiGuardrail`, `ToolPolicyGuardrail`, `LlmJudgeGuardrail`, `ConditionalGuardrail`, `GuardrailChain`, `Blackboard`, `InMemoryBlackboard`, `HeartbitConfig`, `ContextStrategy`, `CompletionRequest`, `CompletionResponse`, `Message`, `TokenUsage`, `ToolCall`, `ToolChoice`, `ReasoningEffort`, `AgentEvent`, `OnText`, `OnApproval`, `OnInput`, `OnEvent`, `ApprovalDecision`, `EvalCase`, `EvalRunner`, `EvalSummary`, `Error`

**Feature-gated**: `LocalEmbeddingProvider` (local-embedding), `PostgresMemoryStore` (postgres), `A2aClient` (a2a), `SensorSecurityGuardrail` (sensor), `DaemonCore` (daemon)
