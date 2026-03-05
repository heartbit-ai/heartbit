# Configuration Reference

Heartbit uses TOML configuration files. Pass via `--config heartbit.toml` or place in the working directory.

## Provider

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
```

## Orchestrator

```toml
[orchestrator]
max_turns = 10
max_tokens = 4096
run_timeout_seconds = 300             # wall-clock deadline for the entire run
routing = "auto"                      # "auto", "always_orchestrate", or "single_agent"
dispatch_mode = "parallel"            # "parallel" or "sequential" (sub-agent dispatch)
reasoning_effort = "high"             # "high", "medium", "low", or "none"
tool_profile = "standard"             # "conversational", "standard", or "full"
```

## Agents

```toml
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
```

## Memory

```toml
[memory]
type = "in_memory"                    # or: type = "postgres", database_url = "..."

[memory.embedding]                    # optional: enables hybrid retrieval (BM25 + vector)
provider = "local"                    # "openai", "local", or "none" (default)
model = "all-MiniLM-L6-v2"           # model name (provider-specific)
cache_dir = "/tmp/fastembed"          # local provider only: model cache directory
# api_key_env = "OPENAI_API_KEY"     # openai provider only
```

## Knowledge Base

```toml
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
```

## Restate

```toml
[restate]
endpoint = "http://localhost:9070"
```

## Daemon

```toml
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
```

## Telemetry

```toml
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
| `HEARTBIT_RECURSIVE_SUMMARIZATION` | `false` | Enable cluster-then-summarize (`1` or `true`) |
| `HEARTBIT_REFLECTION_THRESHOLD` | — | Cumulative importance threshold to trigger reflection |
| `HEARTBIT_CONSOLIDATE_ON_EXIT` | `false` | Consolidate memories at session end (`1` or `true`) |
| `HEARTBIT_MEMORY` | — | Memory backend (`in_memory` or a PostgreSQL URL) |
| `HEARTBIT_LSP_ENABLED` | `false` | Enable LSP integration (`1` or `true`) |
| `HEARTBIT_OBSERVABILITY` | `production` | Observability mode (`production`, `analysis`, `debug`, `off`) |
| `HEARTBIT_TELEGRAM_TOKEN` | — | Telegram bot token (daemon mode) |
| `HEARTBIT_API_KEY` | — | API key for daemon HTTP authentication |
| `EXA_API_KEY` | — | Exa AI API key (for `websearch` built-in tool) |
| `RUST_LOG` | — | Tracing filter (e.g. `info`, `debug`) |
