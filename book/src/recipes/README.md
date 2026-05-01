# Recipes

Practical, task-focused walkthroughs. Each recipe shows how to build one
thing, end to end, with code that compiles.

## Recipes in this cookbook

- [Chat agent with web search](./chat-with-search.md) — a conversational
  agent that browses the web and answers questions with sources.
- [Code-aware agent](./code-aware.md) — wire the file tools and the LSP
  client so the agent can navigate, read, and edit a real codebase.
- [Multi-agent research workflow](./multi-agent-research.md) — orchestrate
  a researcher and a writer that hand off work.
- [Long-running agent with persistent memory](./persistent-memory.md) —
  use the heartbit umbrella crate's Postgres-backed memory so context
  survives across runs.
- [Eval-driven prompt iteration](./eval-driven.md) — write a fixed test
  set, iterate the system prompt against it, gate CI on the score.
- [MCP server integration](./mcp-integration.md) — connect to an MCP
  server (GitHub, Slack, Postgres, …) and expose its tools to the agent.

If you're new to heartbit, read [Getting Started](../getting-started/README.md)
first. Recipes assume you've read chapters 3 ([Agents](../agents/README.md))
and 4 ([Tools](../tools/README.md)).
