# Multi-Agent Orchestration

## When to use the Orchestrator

The [`Orchestrator`](https://docs.rs/heartbit-core/latest/heartbit_core/struct.Orchestrator.html)
is the right tool when sub-agent dispatch needs to be LLM-driven —
that is, when the *which* sub-agent to call is itself a question the
model should answer at runtime, based on the user's request. The
orchestrator runs its own ReAct loop with a `delegate_task` tool
exposing each registered sub-agent by name and description, plus
optional `form_squad` and `spawn_agent` tools for dynamic team
assembly. The model picks, the runner dispatches.

The trade-off versus the deterministic
[Workflow Agents](../workflow-agents/README.md) is the usual one:
flexibility against predictability and cost. The orchestrator pays
LLM tokens at the dispatcher level on top of every sub-agent run. Use
it when the structure isn't knowable in advance, or when you genuinely
want the model to reason about routing. If a `SequentialAgent` or
`DagAgent` already encodes your dispatch graph, prefer that.

## The Orchestrator and OrchestratorBuilder

You build an orchestrator through
[`OrchestratorBuilder`](https://docs.rs/heartbit-core/latest/heartbit_core/struct.OrchestratorBuilder.html).
The two most common shapes are `.sub_agent(name, description, prompt)`
for the simple case and `.sub_agent_full(SubAgentConfig { … })` when
you need per-agent tools, turn caps, or memory namespaces. Sub-agents
must have unique non-empty names; `build()` errors otherwise.

When `agents.len() == 1` the orchestrator detects a single-agent fast
path and bypasses the dispatch loop, calling the lone sub-agent
directly with the input task. This means you can wire a config that
*could* grow into a multi-agent setup without paying dispatch cost
until you actually register a second agent.

## Sub-agent dispatch tools

The orchestrator's LLM has up to three dispatch tools available, all
auto-wired by the builder.

### DelegateTaskTool
`delegate_task` is the default. The LLM picks a sub-agent by name,
hands it a task string, and the orchestrator runs that sub-agent's
`AgentRunner` to completion. The sub-agent's token usage flows back
into the orchestrator's accumulated `AgentOutput`. Use it for the
common "pick the right specialist" pattern.

### FormSquadTool
`form_squad` is auto-enabled when at least two sub-agents are
registered (toggle via `.enable_squads(bool)`). The LLM names a
subset of sub-agents and a single task; the orchestrator runs them in
parallel on a private in-memory blackboard so squad members can share
notes for that one assignment, then aggregates their outputs into a
single result. Use it when independent specialists should collaborate
on the same question — research + critique + risk assessment, run at
once.

### SpawnAgentTool
`spawn_agent` is opt-in via `.spawn_config(...)`. It lets the
orchestrator's LLM define *new* sub-agents at runtime with a custom
system prompt and a subset of an allowlisted tool pool. A spawn-count
cap, name uniqueness within a run, and a shared token budget keep
runaway recursion from happening. Reach for it when the right
specialist isn't predictable in advance and you'd rather have the
model describe the agent it needs than enumerate every possibility
upfront.

## Example

A three-sub-agent orchestrator — researcher, coder, reviewer — wired
through `OrchestratorBuilder` and run on a single task:

```rust,no_run
{{#include ../../../crates/heartbit-core/examples/multi_agent.rs}}
```

The orchestrator's LLM sees three sub-agents in `delegate_task`'s
schema and routes the task to whichever one best fits — or chains
several in sequence within its own loop.

## Blackboard for shared state

Sub-agents are otherwise isolated: each one runs its own
`AgentRunner` with its own message history. When they need to share
intermediate state — a research finding the reviewer should also see,
a plan the coder produced — wire a `Blackboard` via
`OrchestratorBuilder::blackboard(Arc<dyn Blackboard>)` and the runner
exposes three tools to every sub-agent: `blackboard_read`,
`blackboard_write`, and `blackboard_list`. The default
`InMemoryBlackboard` is a `RwLock<HashMap<String, String>>`. Each
sub-agent's `AgentOutput` is also written automatically under
`agent:{name}` so downstream agents can pick it up by key. The
`form_squad` flow uses the same blackboard pattern internally — but
on a private instance so squad scratch work doesn't leak to the
outer run.
