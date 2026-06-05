# Multi-Agent Context Enablement — Design

**Date:** 2026-06-05 · **Branch:** demo/goal-qwen-live · **Status:** approved (user chose "build it now" + per-agent isolation)

## Goal

Make the harness's context-management features — long-horizon planning
(recitation + replan-on-verify-fail), restore-on-demand (`ContextRecallStore`),
and proactive compaction (`context_window_tokens`) — work for **sub-agents** in
the multi-agent (orchestrator) path, not just the single-agent path. Today all
four are single-agent-only, leaving both point 3 (planning) and point 4 (context
finish) half-done in a "multi-agent enterprise runtime."

## Core decision (user)

**Per-sub-agent isolation.** Each sub-agent gets its OWN `TodoStore` and
`ContextRecallStore` — it recites/restores only its own plan and tool outputs.
Mirrors the existing flat-hierarchy, per-run model (each sub-agent already has
its own `FileTracker`, tools, memory namespace).

## Architecture — caller-wired config, orchestrator forwards

The orchestrator cannot reach into a sub-agent's pre-built
`SubAgentConfig.tools` to share a store with the runner. So the **caller** (e.g.
the TUI's `default_sub_agents`) creates each sub-agent's stores, builds its tools
from them, and hands BOTH to the config — exactly as `build_engine` already does
for the single agent. The orchestrator then forwards the config's context fields
to the sub-agent's `AgentRunnerBuilder`, just as it already forwards
`audit_trail`, `permission_rules`, `lsp_manager`, and `tenant_tracker`.

This keeps the load-bearing invariant intact: *whoever builds the tools owns the
store and is responsible for sharing it with the runner.* No new ownership model.

## Components

### 1. `SubAgentContextConfig` (new struct, `agent/orchestrator.rs`)

```rust
/// Per-sub-agent context-management wiring (multi-agent enablement). All
/// fields opt-in: a default value reproduces today's behavior exactly.
#[derive(Clone, Default)]
pub struct SubAgentContextConfig {
    /// Shared todo store the sub-agent recites at the context tail each turn
    /// (long-horizon recitation). MUST be the same store backing the
    /// sub-agent's todowrite/todoread tools.
    pub todo_store: Option<Arc<crate::tool::builtins::TodoStore>>,
    /// Per-run recall store for restore-on-demand. MUST be the same store
    /// passed into the sub-agent's `BuiltinToolsConfig.context_recall_store`.
    pub context_recall_store: Option<Arc<crate::agent::context_recall::ContextRecallStore>>,
    /// Model context window (tokens) for the proactive compaction backstop.
    pub context_window_tokens: Option<u32>,
    /// When true, a RED verify blocks the sub-agent's natural completion
    /// (bounded replan).
    pub replan_on_verify_fail: bool,
}
```

> **Session-prune pairing (impl decision):** the struct holds 4 fields, NOT 5.
> Restore-on-demand needs pruning to produce restore markers, but rather than
> duplicate a `session_prune_config` here, the caller sets the EXISTING
> `SubAgentConfig::session_prune_config` field (already forwarded by the
> orchestrator). The TUI pairs the recall store with `gentle_prune_config()`
> there. Reusing the existing field keeps the new struct minimal and the
> caller-wired invariant intact.

### 2. `SubAgentConfig` gains ONE field

`pub context: SubAgentContextConfig`. A single nested field (not four bare
fields) keeps the ~15 existing struct-literal sites to a one-line change
(`context: SubAgentContextConfig::default(),`) and lets the context travel WITH
the agent definition — no name-keyed map and its silent-mismatch footgun.

### 3. Orchestrator forwarding (`agent/orchestrator.rs`, sub-agent build block)

In the per-sub-agent `builder = builder.X(...)` assembly (alongside the existing
audit/permission/lsp/tenant forwards), add:

```rust
// agent_def.context fields are Option<Arc>/bool (cheap); clone per field.
if let Some(store) = agent_def.context.todo_store.clone() { builder = builder.todo_store(store); }
if agent_def.context.replan_on_verify_fail { builder = builder.replan_on_verify_fail(true); }
if let Some(w) = agent_def.context.context_window_tokens { builder = builder.context_window_tokens(w); }
if let Some(store) = agent_def.context.context_recall_store.clone() { builder = builder.context_recall_store(store); }
```

The prune pairing is NOT here — the caller sets the existing
`SubAgentConfig::session_prune_config`, already forwarded earlier in the same
block. There are **two** sub-agent build paths (`delegate_task` ~540 and
`form_squad` ~1000 — confirmed: F-AGENT-2 showed they diverge, so the block is
added to BOTH and each gets its own propagation test). The forwarding also
requires threading `context` through the internal `SubAgentDef` (struct field +
`new()` + `From<SubAgentConfig>` conversion), since the orchestrator builds
runners from `SubAgentDef`, not `SubAgentConfig`.

### 4. Callers

- **TUI `default_sub_agents`**: for each sub-agent, create a per-agent
  `TodoStore` + `ContextRecallStore`, build tools via
  `fresh_builtins(cwd, Some(&recall), Some(&todo))`, and set
  `context: SubAgentContextConfig { todo_store: Some(todo), context_recall_store:
  Some(recall), context_window_tokens: <window>, replan_on_verify_fail: <verify
  active>, session_prune_config: Some(gentle) }`. Gate on the same
  `context_recall` / `verify_command` toggles the single-agent path uses. Remove
  the `!multi_agent` gates that currently disable these features.
- **`RuntimeSubAgentConfig` → `SubAgentConfig`** conversion (daemon): set
  `context: SubAgentContextConfig::default()` (daemon enablement is out of scope
  for this pass; keep it behavior-preserving).
- **All other literal sites** (orchestrator tests, CLI): add
  `context: SubAgentContextConfig::default(),`.

## Data flow

caller creates per-agent `TodoStore`/`ContextRecallStore` → builds todo/recall
tools FROM them (into `SubAgentConfig.tools`) → sets the SAME stores on
`SubAgentConfig.context` → orchestrator forwards them to the sub-agent's
`AgentRunnerBuilder` → recitation / restore-on-demand / proactive compaction /
replan run per sub-agent, isolated.

## Error handling

Fully opt-in. `SubAgentContextConfig::default()` (all `None`/`false`) reproduces
today's behavior byte-for-byte. No orchestrator API breakage beyond the additive
struct field. Existing sub-agents with no context config are unaffected.

## Testing (TDD)

1. **`SubAgentContextConfig` default** is all-None/false (compile + trivial assert).
2. **Recitation propagates**: build an `Orchestrator` with a sub-agent whose
   `SubAgentConfig.context.todo_store` is pre-populated with open items, a
   request-capturing provider, the orchestrator delegates a task to it; assert
   the sub-agent's captured LLM request's last message contains the
   `[plan — open items` recitation block. (Mirrors the single-agent
   `recites_open_todos_at_context_tail` test, through the orchestrator.)
3. **Replan propagates**: a sub-agent whose `context.replan_on_verify_fail` is
   true and whose transcript shows `VERIFY_RESULT: FAIL` does not complete on the
   first EndTurn (bounded). Mirrors the single-agent replan test through the
   orchestrator.
4. **Defaults unchanged**: a sub-agent with `context: Default::default()`
   produces NO recitation block and identical behavior (regression guard).
5. **Gate**: `cargo fmt --all -- --check && cargo clippy --workspace --exclude
   mini-crm --all-targets -- -D warnings && cargo test --workspace --exclude
   mini-crm` green; existing 3000+ tests still pass (additive change).

## Out of scope

Daemon (`RuntimeSubAgentConfig`) enablement; cross-agent shared plans (rejected
— per-agent isolation chosen); a name-keyed registry (rejected — struct field is
more robust); orchestrator-level recitation of its own router plan (the
orchestrator is a thin router; sub-agents hold the work).
