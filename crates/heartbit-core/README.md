# heartbit-core

The Rust agentic framework — agents, tools, LLM providers, memory,
guardrails, workflow agents, evaluation. Type-safe, async-native,
runtime-agnostic.

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

## What's in the box

- **ReAct agent loop** with parallel tool execution via `tokio::JoinSet`.
- **LLM providers** — Anthropic, OpenRouter, Google Gemini, any
  OpenAI-compatible endpoint. Composable wrappers: `RetryingProvider`
  (exponential backoff on 429/5xx), `CascadingProvider` (cheapest model
  first, escalate on gate rejection), Anthropic prompt caching.
- **Tools** — `Tool` trait, `ToolDefinition`, `ToolOutput`. The crate
  ships a small core set used by the agent loop; the umbrella `heartbit`
  crate adds the full builtin suite (`bash`, `read`/`write`/`edit`/
  `patch`, `web_fetch`, `web_search`, MCP, …).
- **Memory** — `Memory` trait, `InMemoryStore`, `NamespacedMemory`.
  MemGPT-style composite recall (recency + importance + relevance +
  strength), Ebbinghaus decay, optional vector retrieval.
- **Guardrails** — `Guardrail` trait with four hooks (`pre_llm`,
  `post_llm`, `pre_tool`, `post_tool`). Built-ins: LLM-as-judge, secret
  scanner, PII, content fence, action budget, behavioral monitor, tool
  policy, injection classifier, sensor security, plus composition
  (chain, conditional, warn-to-deny).
- **Workflow agents** — deterministic orchestration with no LLM cost:
  `SequentialAgent`, `ParallelAgent`, `LoopAgent`, `DagAgent`,
  `VotingAgent`, `DebateAgent`, `MixtureOfAgentsAgent`, `BatchExecutor`.
- **Orchestrator** — `Orchestrator<P>` for multi-agent runs with flat
  hierarchy. Three delegation tools: `delegate_task` (parallel
  subtasks), `form_squad` (collaborative via `Blackboard`), `spawn_agent`
  (dynamic specialists).
- **Eval** — `EvalRunner`, `EvalCase`, scorers (trajectory, keyword,
  similarity, cost, latency, tool count, safety), `EvalComparison` for
  A/B regression testing.
- **Templates & skills** — 15 agent archetypes and 10 domain skills
  ready to compose into prompts.
- **Observability** — 29 streaming `AgentEvent` variants, `OnEvent`
  callback, `OnText` for token streaming, `OnApproval` for
  human-in-the-loop.

## Optional integrations

For Postgres-backed memory, Telegram / Discord / Slack chat adapters,
fastembed local embeddings, sandboxed workspaces, JWT auth, vault,
multi-tenant daemon mode, and Restate-durable execution, add the
[`heartbit`](https://crates.io/crates/heartbit) umbrella crate:

```bash
cargo add heartbit --features full
```

The umbrella does `pub use heartbit_core::*;`, so library code remains
import-compatible regardless of which crate you depend on.

## Repository

- Source: [github.com/heartbit-ai/heartbit](https://github.com/heartbit-ai/heartbit)
- API reference: [docs.rs/heartbit-core](https://docs.rs/heartbit-core)
- Top-level README, examples, full docs: see the
  [repository root](https://github.com/heartbit-ai/heartbit#readme).

## License

Dual-licensed under [MIT](https://github.com/heartbit-ai/heartbit/blob/main/LICENSE-MIT)
or [Apache-2.0](https://github.com/heartbit-ai/heartbit/blob/main/LICENSE-APACHE),
at your option.
