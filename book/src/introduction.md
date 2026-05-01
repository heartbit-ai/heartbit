# Introduction

## What is heartbit-core

`heartbit-core` is a Rust framework for building agentic LLM applications.
You give an agent a system prompt, a list of tools, and a user task; the
agent runs a [ReAct](https://arxiv.org/abs/2210.03629) loop — alternating
LLM calls and tool executions — until it returns a final answer or hits
one of your configured limits.

The framework is built around four trait abstractions: `LlmProvider` for
swappable model backends, `Tool` for capabilities you grant the agent,
`Memory` for cross-turn and cross-session state, and `Guardrail` for
safety and policy hooks. Tool calls within a single turn run in parallel
on a `tokio::JoinSet`. The whole stack is async-native, type-safe, and
runtime-agnostic — it works equally well embedded in a web service, a
CLI, or a long-running daemon.

This is a production framework, not a research demo. The crate ships
with retry, prompt caching, model cascading, sandboxing, MCP client,
session pruning, auto-compaction on context overflow, and
[29 streaming `AgentEvent` variants](https://docs.rs/heartbit-core) for
observability. Reach for it when you want type-checked tool I/O and a
single binary, not when you want to copy-paste a notebook.

## What you can build with it

- **Chat agents** that hold a conversation, search the web, and remember
  what you told them last week — see [Recipes](./recipes/README.md).
- **Code-aware agents** that read, edit, patch, and run shell commands
  inside a sandboxed workspace — see the
  [Tools chapter](./tools/README.md).
- **Multi-agent research workflows** with an orchestrator delegating to
  parallel specialists via a shared blackboard — see
  [Multi-Agent Orchestration](./orchestration/README.md).
- **Eval-driven prompt iteration** with trajectory, similarity, cost,
  and safety scorers and A/B regression comparison — see the
  [Eval Framework](./eval/README.md).
- **MCP server integrations** that expose any
  [Model Context Protocol](https://modelcontextprotocol.io) server's
  tools to your agent with a single call — see the
  [Tools chapter](./tools/README.md).

## How this book is organized

The book has twelve chapters. Read the first five in order — each
builds on the one before it:

1. **Introduction** (this chapter).
2. **[Getting Started](./getting-started/README.md)** — install, run a
   hello-world agent, pick a provider, set up API keys.
3. **[Agents](./agents/README.md)** — the ReAct loop, builder API,
   token budgets, streaming, events.
4. **[Tools](./tools/README.md)** — built-ins, writing your own,
   approval hooks, MCP.
5. **[Memory](./memory/README.md)** — episodic, semantic, and
   reflection memory; recall scoring; consolidation.

The remaining chapters can be read by need:
**[Guardrails](./guardrails/README.md)**,
**[Workflow Agents](./workflow-agents/README.md)**,
**[Multi-Agent Orchestration](./orchestration/README.md)**,
**[Configuration](./configuration/README.md)**,
**[Eval Framework](./eval/README.md)**,
**[Recipes](./recipes/README.md)**, and
**[Production Considerations](./production/README.md)**.

## Why Rust

Tool inputs and outputs are JSON values, but everything around them is
strongly typed: tool definitions, agent configuration, provider
responses, event streams. Bugs that would be runtime crashes in a
dynamic language are compile errors here.

The Tokio runtime scales an agent server to thousands of concurrent
agents on a single machine without per-agent threads, and the binary
ships as a single statically-linked file with no GC pauses to worry
about. When latency or throughput matter — production chat, real-time
trading workflows, large-scale evaluations — the runtime cost
disappears.

## Prerequisites

- **Rust 1.85 or later** (edition 2024).
- **An LLM API key** for at least one of: Anthropic, OpenRouter,
  Google Gemini, or any OpenAI-compatible endpoint (vLLM, Ollama,
  local servers).
- **Basic familiarity with async Rust** — `async fn`, `.await`,
  `Arc`, `tokio::main`. The book does not teach the language.

If you are comfortable reading the Rust standard library docs, you are
ready for this book.

## Where to go from here

- API reference: [heartbit-core on docs.rs](https://docs.rs/heartbit-core).
- Source code, examples, issue tracker:
  [heartbit on GitHub](https://github.com/heartbit-ai/heartbit).
- The next chapter: [Getting Started](./getting-started/README.md).
