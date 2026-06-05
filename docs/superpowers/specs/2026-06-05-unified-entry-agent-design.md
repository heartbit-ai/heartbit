# Unified Dynamic-Execution Entry Agent ("option C") — Design

**Date:** 2026-06-05 · **Branch:** demo/goal-qwen-live · **Status:** approved (user: "go pour C", "fais les évolutions jusqu'au bout"). Locked decisions: **émergent routing** (the LLM decides by choosing tools — `routing.rs` stays unwired) + **named workflow registry** (recipes by name + args).

## Problem

The TUI picks single-agent vs orchestrator via a STATIC `multi_agent` config flag. The orchestrator's prompt forbids direct answers ("Never respond to the user directly without delegating first"), so a trivial turn like "que sais-tu" triggers a delegation/self-description — inappropriate. The `flow/` dynamic workflows are developer-Rust only, unreachable by the agent.

## Target

ONE capable entry agent, always built (no mode flag), that decides per request via tool choice:
- conversational / simple question → answer DIRECTLY;
- concrete work it can do → do it with its own tools;
- multi-part work → delegate (`delegate_task` / `form_squad`);
- structured repeatable job matching a recipe → launch it (`run_workflow`).

Routing is emergent (Claude-Code model: one agent + delegation tools + a workflow tool). Realized by **evolving the orchestrator** (already an `AgentRunner` with delegation tools) — additively, so its 30+ existing tests stay green.

## Components

### 1. Capable entry-agent prompt + direct tools (`orchestrator.rs`)

- `OrchestratorBuilder::entry_agent(direct_tools: Vec<Arc<dyn Tool>>)` — opt-in. Sets a flag `entry_mode = true` and stores `direct_tools`.
- `build()`: when `entry_mode`, (a) build the system prompt via a NEW `build_entry_agent_prompt(...)` variant instead of `build_system_prompt`, and (b) add `direct_tools` to the orchestrator's own `runner_builder.tools(...)` alongside the delegation tools. Default path (entry_mode false) is byte-for-byte unchanged.
- `build_entry_agent_prompt`: a non-absolute contract —
  - answer conversational/simple turns DIRECTLY (no delegation);
  - do concrete work with your own tools when you can in a few steps;
  - delegate ONLY multi-part work needing different expertise (delegate_task / form_squad);
  - launch a workflow recipe (run_workflow) for structured, repeatable fan-out;
  - prefer the simplest path; never force-split a simple task.
  Still lists the available sub-agents + (if present) workflow recipes.

### 2. Workflow registry + `run_workflow` tool (`agent/workflow_tool.rs`, new)

- `WorkflowRecipe`: `{ name: String, description: String, args_schema: serde_json::Value, run: Arc<dyn Fn(WorkflowCtx, serde_json::Value) -> Pin<Box<dyn Future<Output = Result<String, Error>> + Send>> + Send + Sync> }`. The `run` closure drives `flow/` combinators (`agent`/`parallel`/`pipeline`) against a `WorkflowCtx` and returns a text result.
- `WorkflowRegistry`: `Vec<WorkflowRecipe>` + `get(name)`; builder `register(recipe)`.
- `RunWorkflowTool` (`impl Tool`, name `"run_workflow"`): `definition()` advertises the recipe names + descriptions + a `{recipe: enum[names], args: object}` input schema so the LLM picks one; `execute()` looks up the recipe, builds a `WorkflowCtx` (from a shared `BoxedProvider` + an event sink), runs it, returns the result (or a clear error if the name is unknown / args invalid). Empty registry → the tool is simply not registered.
- **First recipe (`recipes::parallel_review`):** fan out N independent review lenses (correctness, security, clarity) over a provided target via `flow::parallel`, then synthesize — genuinely useful for the coding TUI and exercises the parallel combinator + budget. Args: `{ target: string, lenses?: [string] }`.

### 3. Context features on the orchestrator's OWN runner (`orchestrator.rs`)

The entry agent must get the same context stack as the single-agent path. Add `OrchestratorBuilder` setters that apply to its own `runner_builder`: `todo_store`, `context_recall_store`, `context_window_tokens`, `replan_on_verify_fail`, `session_prune_config` (mirroring the sub-agent forwarding, but for the orchestrator's own runner). All opt-in.

### 4. TUI wiring (`heartbit-tui/src/main.rs`)

- `build_engine` ALWAYS builds the unified entry agent: `Orchestrator::builder(provider)` + `.entry_agent(fresh_builtins(...))` (the entry agent's own direct tools, with its own TodoStore + ContextRecallStore) + the workflow registry (`.workflow_registry(default_registry())`) + the default squad as sub-agents + the orchestrator-runner context setters (todo/recall/window/replan) + the capable prompt. No `if multi_agent` branch.
- The single-agent `else` branch is removed; `Engine::Single` may be retired or kept as a thin alias. The `multi_agent` config field + `/agents` toggle become **deprecated no-ops** (kept for config back-compat; a Notice explains the unification) — do NOT hard-error on existing configs.
- MCP tools, verify, context_recall, context_window flags all thread into the single unified builder.

## Data flow

user turn → unified entry `AgentRunner` (builtins + delegate_task/form_squad + run_workflow + recitation/recall) → LLM picks: direct answer | direct tool use | delegate | run_workflow → result synthesized in the SAME loop (one `run()`).

## Error handling

All additive/opt-in: `entry_mode=false` and an empty workflow registry reproduce today's orchestrator exactly. Unknown recipe / invalid args → `ToolOutput::error` (agent self-corrects). Empty squad → delegation tools have no agents, agent falls back to its own tools (graceful single-agent degradation). Existing configs with `multi_agent` still load (no-op + Notice).

## Testing (TDD)

1. `build_entry_agent_prompt` ALLOWS direct answers (asserts it does NOT contain the "never respond directly" clause; DOES contain the "answer simple turns directly" contract).
2. Entry-mode orchestrator with direct tools: a mock provider returning a plain text answer to "hi" completes directly (no delegation, no error) — and its runner actually holds the direct tools.
3. `WorkflowRegistry::get` hit/miss; `RunWorkflowTool::definition()` lists registered recipe names; `execute()` runs a stub recipe and returns its result; unknown name → error output.
4. `parallel_review` recipe: with a mock provider, fans out the lenses (parallel) and returns a synthesized string (assert all lenses appear / N agent calls).
5. Orchestrator own-runner context setters: entry agent with a populated `todo_store` recites at its tail (mirror the sub-agent recitation test on the orchestrator's own runner).
6. Regression: default orchestrator (no entry_agent, empty registry) — prompt + behavior unchanged; the existing 30+ orchestrator tests stay green.
7. Gate: `cargo fmt --all -- --check && cargo clippy --workspace --exclude mini-crm --all-targets -- -D warnings && cargo test --workspace --exclude mini-crm`.

## Out of scope (documented follow-ups)

Daemon/Restate unification (TUI only this pass); more recipes beyond the first; LLM-authored workflows (locked out — no DSL/IR); retiring `routing.rs` (left in core, unwired); auto-tuning the squad size.

## Sequencing (implementation order, each its own commit)

1. Component 1 (prompt + direct tools) — core of the fix.
2. Component 3 (orchestrator own-runner context setters) — small, unblocks full capability.
3. Component 2 (workflow registry + tool + `parallel_review`).
4. Component 4 (TUI wiring + drop the mode flag).
