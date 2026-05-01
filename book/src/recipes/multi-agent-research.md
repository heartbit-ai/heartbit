# Multi-agent research workflow

## Goal

A workflow where a researcher gathers facts and a writer turns them
into prose. You can wire it as a deterministic
[SequentialAgent](../workflow-agents/README.md#sequential) or, if the
work order isn't fixed, dispatch through a dynamic
[Orchestrator](../orchestration/README.md).

## Solution: deterministic pipeline

When the order is known — research first, write second — a
`SequentialAgent` keeps the wiring boring and free of LLM-driven
routing. The output of agent _N_ becomes the input of agent _N+1_; the
final `AgentOutput` accumulates token usage from every step.

```rust,no_run
{{#include ../../../crates/heartbit-core/examples/sequential_agent.rs}}
```

Two `AgentRunner`s share the same provider, both built with focused
system prompts. `SequentialAgent::builder().agent(...).agent(...).build()?`
chains them; `.execute("topic")` runs the pipeline end to end and
returns one combined output.

## Solution: dynamic orchestrator

When the right next sub-agent depends on what the user asked,
hand off to an `Orchestrator`. Each sub-agent is registered with a
name, a one-line description, and its own system prompt. The
orchestrator's LLM sees those descriptions and picks who to delegate
to via the built-in `delegate_task` tool.

```rust,no_run
{{#include ../../../crates/heartbit-core/examples/multi_agent.rs}}
```

## Choosing between them

Pick the deterministic pipeline when the order is known; pick the
orchestrator when sub-agent selection depends on the task at runtime.
See [Workflow Agents](../workflow-agents/README.md) and
[Orchestration](../orchestration/README.md) for the full menu.
