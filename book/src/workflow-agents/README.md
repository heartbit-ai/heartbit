# Workflow Agents

## When to use a workflow vs a single agent

A single [`AgentRunner`](https://docs.rs/heartbit-core/latest/heartbit_core/struct.AgentRunner.html)
is the right tool when one ReAct loop can finish the task — the LLM
decides what to do, calls tools, and returns. A workflow agent is the
right tool when the structure of the work is known in advance and you
want it executed deterministically, with no LLM cost paid at the
dispatcher level. The dispatcher is plain Rust code; only the leaves
call the model.

Workflow agents compose other `AgentRunner`s (or other workflow
agents) into pipelines, fan-outs, refinement loops, conditional DAGs,
and consensus structures. They share the same `AgentOutput` shape —
result text plus accumulated `TokenUsage` — so they slot into any
caller that accepts an agent. The trade-off versus a single LLM-driven
agent is predictability and cost (no dispatcher tokens) against
flexibility (the structure is fixed before the run starts). Reach for
the [Multi-Agent Orchestration](../orchestration/README.md) chapter
when the dispatch itself needs to be LLM-driven instead.

## Sequential

[`SequentialAgent`](https://docs.rs/heartbit-core/latest/heartbit_core/struct.SequentialAgent.html)
chains agents in declared order, piping each step's output as the
next step's input and accumulating token usage across the whole run.
Use it for pipelines where stage `n+1` literally consumes stage `n`'s
text — research → write, draft → critique → revise. Fail-fast: any
sub-agent error aborts the pipeline and the partial usage so far is
preserved on the returned error.

```rust,no_run
{{#include ../../../crates/heartbit-core/examples/sequential_agent.rs}}
```

## Parallel

[`ParallelAgent`](https://docs.rs/heartbit-core/latest/heartbit_core/struct.ParallelAgent.html)
runs every agent concurrently on a `tokio::JoinSet` and merges the
results in declared order. Token usage is summed; the first error
fails the whole run. Use it when sub-tasks are independent — three
search backends, four file-summarisers — and total wall time matters.

```rust,ignore
let workflow = ParallelAgent::builder()
    .agent(searcher_a)
    .agent(searcher_b)
    .agent(searcher_c)
    .build()?;
let output = workflow.execute("query").await?;
```

## Loop

[`LoopAgent`](https://docs.rs/heartbit-core/latest/heartbit_core/struct.LoopAgent.html)
re-runs a single sub-agent on its own output until either
`should_stop(text)` returns `true` or `max_iterations` is reached.
Useful for refinement and self-correction passes — keep critiquing a
draft until a separate quality check signals "good enough".
`should_stop` is plain Rust (`Fn(&str) -> bool`), so the termination
predicate runs at zero LLM cost.

```rust,ignore
let refiner = LoopAgent::builder()
    .agent(reviewer)
    .max_iterations(3)
    .should_stop(|text: &str| text.contains("APPROVED"))
    .build()?;
```

## DAG

[`DagAgent`](https://docs.rs/heartbit-core/latest/heartbit_core/struct.DagAgent.html)
generalises the patterns above into an arbitrary directed acyclic
graph. Declare nodes with `.node(name, agent)` and edges with
`.edge(from, to)`; the builder rejects cycles at build time.

```rust,no_run
{{#include ../../../crates/heartbit-core/examples/dag_agent.rs}}
```

Execution is petgraph-backed BFS. Topological order is respected,
nodes within the same tier run in parallel, and a node receives the
concatenated outputs of its incoming edges as input. The result of
the DAG is the output of the unique terminal node — the builder
errors if there is more than one. This is the right shape when work
has true branching structure rather than fitting into pure pipeline,
fan-out, or refinement.

## Voting / Debate / Mixture-of-agents

[`VotingAgent`](https://docs.rs/heartbit-core/latest/heartbit_core/struct.VotingAgent.html)
runs `n` agents in parallel and selects the majority answer (or applies
a custom vote aggregator). Use it for self-consistency: the same
question, multiple temperatures or seeds, take the consensus.

[`DebateAgent`](https://docs.rs/heartbit-core/latest/heartbit_core/struct.DebateAgent.html)
runs multiple agents — typically with opposing system prompts — across
several rounds, feeding each agent the others' previous arguments. Use
it when an adversarial back-and-forth surfaces failure modes a single
agent would miss.

[`MixtureOfAgentsAgent`](https://docs.rs/heartbit-core/latest/heartbit_core/struct.MixtureOfAgentsAgent.html)
runs several "proposer" agents in parallel, then feeds their proposals
to a single "synthesizer" agent that produces the final answer. It's
the workflow shape from the
[Mixture-of-Agents paper](https://arxiv.org/abs/2406.04692) and is
useful when proposers are diverse — different models, different
prompts — and a synthesizer benefits from seeing all of them at once
rather than picking a single winner.

## Choosing the right pattern

A short decision tree:

- Need stage `n+1` to consume stage `n`'s output deterministically?
  `SequentialAgent`.
- Independent fan-out where wall time matters? `ParallelAgent`.
- Refinement until a quality threshold is met? `LoopAgent`.
- Graph-structured work with multiple inputs and conditional joins?
  `DagAgent`.
- Self-consistency over `n` samples? `VotingAgent`.
- Adversarial cross-examination? `DebateAgent`.
- Multiple proposers, single synthesizer? `MixtureOfAgentsAgent`.
- Dispatch decision should itself be LLM-driven, with tools picked at
  runtime? Drop down to the [Orchestrator](../orchestration/README.md).
