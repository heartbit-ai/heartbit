[![CI](https://github.com/heartbit-ai/heartbit/actions/workflows/ci.yml/badge.svg)](https://github.com/heartbit-ai/heartbit/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/heartbit-core)](https://crates.io/crates/heartbit-core)
[![docs.rs](https://img.shields.io/docsrs/heartbit-core)](https://docs.rs/heartbit-core)
[![Book](https://img.shields.io/badge/book-docs.heartbit.ai-blue)](https://docs.heartbit.ai)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)
[![Telegram](https://img.shields.io/badge/Telegram-heartbitagent-blue?logo=telegram)](https://t.me/heartbitagent)

# Heartbit — the Rust agentic framework

A production-grade framework for building LLM-powered agents in Rust.
Type-safe, async-native, and runtime-agnostic. Zero-copy ReAct loops with
parallel tool execution via `tokio::JoinSet`, no Python/Node overhead.

> **Early-stage, capability-first software.** Heartbit prioritizes capability and
> velocity at this stage. Agents execute tools (including shell commands) with
> the permissions of the host process. Do not run untrusted workloads in
> production without your own sandboxing and access controls. See
> [Disclaimer](#disclaimer).

## Quickstart

```bash
cargo add heartbit-core
```

```rust
use std::sync::Arc;
use heartbit_core::{AgentRunner, AnthropicProvider, BoxedProvider, RetryingProvider};

#[tokio::main]
async fn main() -> Result<(), heartbit_core::Error> {
    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .expect("set ANTHROPIC_API_KEY environment variable");

    let provider = Arc::new(BoxedProvider::new(
        RetryingProvider::with_defaults(
            AnthropicProvider::new(api_key, "claude-sonnet-4-20250514"),
        ),
    ));

    let agent = AgentRunner::builder(provider)
        .system_prompt("You are a helpful assistant.")
        .build()?;

    let output = agent.execute("What is Rust?").await?;
    println!("{}", output.result);
    Ok(())
}
```

## Features

- **ReAct agent loop** with parallel tool execution via `tokio::JoinSet`.
- **LLM providers**: Anthropic, OpenRouter, Google Gemini, any OpenAI-compatible
  endpoint. Composable wrappers for retry (exponential backoff on 429/5xx),
  cascade (try cheap models first, escalate on gate rejection), and prompt
  caching.
- **Built-in tools**: `bash`, `read`/`write`/`edit`/`patch`, `grep`/`glob`/`list`,
  `web_fetch` (with SSRF defense), `web_search`, `image_generate`, `tts`,
  `twitter_post`, `todo_manage`, `skill`, `question`, plus the MCP client and
  A2A protocol.
- **Memory** — `Memory` trait + `InMemoryStore` + `NamespacedMemory`. Postgres /
  pgvector backend via the [`heartbit`](crates/heartbit) umbrella crate. MemGPT-
  style composite recall, Ebbinghaus decay, hybrid retrieval.
- **Guardrails** — 12 ready-made: LLM-as-judge, secret scanner, PII, content
  fence, action budget, behavioral monitor, tool policy, injection classifier,
  sensor-security, plus composition (chain, conditional, warn-to-deny).
- **Workflow agents** — `Sequential`, `Parallel`, `Loop`, `DAG`, `Voting`,
  `Debate`, `MixtureOfAgents`, `Batch` — deterministic orchestration with no
  LLM cost.
- **Eval framework** — `EvalRunner`, `EvalCase`, 7 scorers (trajectory, keyword,
  similarity, cost, latency, tool count, safety) and `EvalComparison` for A/B
  regression testing.
- **Multi-tenant primitives** — workspace jails, namespaced memory, guardrail
  kill-switch, constant-time auth helpers.
- **Templates & skills** — 15 agent templates (`coder`, `researcher`, …) and 10
  domain skills (`rust-expert`, `kubernetes`, …) for rapid composition.
- **Observability** — 29 streaming `AgentEvent` variants, OpenTelemetry OTLP
  export, TTFT tracking, JSON-to-stderr verbose mode.

## Crate layout

| Crate | What it is |
|---|---|
| [`heartbit-core`](crates/heartbit-core) | The framework. ← `cargo add` this. |
| [`heartbit`](crates/heartbit) | Umbrella + platform integrations: Postgres, Telegram/Discord/Slack adapters, Restate workflows, fastembed local embeddings, vault, JWT validator, daemon mode. |
| [`heartbit-cli`](crates/heartbit-cli) | The binary: `heartbit run`, `heartbit chat`, `heartbit serve`, `heartbit daemon`. |
| [`heartbit-gateway`](crates/heartbit-gateway) | Ingestion gateway — cron, sensors, webhooks to Kafka. |
| [`heartbit-macro`](crates/heartbit-macro) | Proc macros for tool definitions. |

The umbrella's `pub use heartbit_core::*;` means existing imports
(`use heartbit::AgentRunner;`) keep working. Library-only users can target
`heartbit-core` directly and skip the platform dependencies.

## Want the full multi-tenant runtime / platform?

The platform side — daemon mode, Kafka-backed task queue, Axum HTTP API,
multi-tenant JWT auth, sandboxed workspaces, sensor pipeline, Telegram /
Discord / Slack channels — is documented separately:

- [`crates/heartbit-cli/README.md`](crates/heartbit-cli/README.md) — operator-facing guide.
- [`docs/platform.md`](docs/platform.md) — architecture overview.

## Documentation

- **[API reference (heartbit-core)](https://docs.rs/heartbit-core)**
- [Configuration reference](docs/configuration.md) — full TOML schema and env vars
- [Built-in tools](docs/tools.md)
- [Memory architecture](docs/memory.md)
- [Daemon mode](docs/daemon.md)
- [Sensor pipeline](docs/sensors.md)
- [Restate / durable execution](docs/restate.md)
- [Eval framework](docs/eval.md)
- [Telegram integration](docs/telegram.md)

## Examples

See [`crates/heartbit/examples/`](crates/heartbit/examples/):

```bash
cargo run --example hello_agent  -p heartbit    # Minimal single agent
cargo run --example simple_agent -p heartbit    # Agent with builtin tools
cargo run --example multi_agent  -p heartbit    # Orchestrator with sub-agents
cargo run --example custom_tool  -p heartbit    # Implementing the Tool trait
cargo run --example mcp_agent    -p heartbit    # MCP tool integration
cargo run --example guardrails   -p heartbit    # LLM judge guardrail
cargo run --example memory       -p heartbit    # Memory-enabled agent
cargo run --example eval         -p heartbit    # Running evals
```

## Development

```bash
cargo fmt -- --check && cargo clippy -- -D warnings && cargo test
```

All three must pass. TDD mandatory — red / green / refactor for every feature.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Security-related reports: see
[SECURITY.md](SECURITY.md). Join the community on
[Telegram](https://t.me/heartbitagent).

## Disclaimer

Heartbit is **early-stage, capability-first software**. It is provided "as is"
without warranty of any kind, express or implied. Security is not the primary
design driver today; while guardrails and permission systems exist, they have
not been audited and should not be relied upon as a security boundary.

Specifically: agents can execute arbitrary shell commands, read/write files,
and make network requests with the full permissions of the host process. LLM
outputs are inherently unpredictable; tool calls generated by the model may
produce unintended side effects. MCP servers, sensors, and other external
integrations expand the attack surface. Landlock sandboxing and workspace
jailing reduce but do not eliminate risk.

The maintainers accept no liability for data loss, security incidents,
unintended actions, costs incurred from LLM API usage, or any other damages
arising from the use of this software. Use at your own risk.

If you discover a security vulnerability, please report it privately via
[GitHub Security Advisories](https://github.com/heartbit-ai/heartbit/security/advisories)
rather than opening a public issue.

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at
your option.

## Acknowledgements

`heartbit-core` is the engine for heartbit-cloud, an Agents-as-a-Service
platform. The framework is independently usable and licensed under
MIT OR Apache-2.0.
