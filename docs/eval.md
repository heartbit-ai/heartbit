# Eval Framework

Built-in evaluation framework for testing agent behavior.

## Overview

The eval framework provides:
- **Trajectory scoring** — verify agents call the right tools in the right order
- **Keyword scoring** — check that outputs contain expected keywords
- **Similarity scoring** — cosine similarity to reference outputs

## Quick Start

```rust
use heartbit::{EvalCase, EvalRunner, TrajectoryScorer, KeywordScorer, SimilarityScorer};

let case = EvalCase::new("research-task", "Find info about Rust")
    .expect_tool("websearch")
    .expect_output_contains("Rust")
    .reference_output("Rust is a systems programming language");

let runner = EvalRunner::new(vec![case])
    .scorer(TrajectoryScorer)    // tool call sequence matching
    .scorer(KeywordScorer)       // output keyword checking
    .scorer(SimilarityScorer);   // cosine similarity to reference

let summary = runner.run(agent).await?;
println!("Pass rate: {:.0}%", summary.pass_rate * 100.0);
```

## Eval Cases

An `EvalCase` defines:
- **Name** — identifier for the test case
- **Input** — the task to give the agent
- **Expected tools** — tools the agent should call (trajectory scoring)
- **Expected keywords** — keywords that should appear in the output
- **Reference output** — ideal output for similarity scoring

## Scorers

| Scorer | What it checks |
|--------|---------------|
| `TrajectoryScorer` | Tool call sequence matches expected tools |
| `KeywordScorer` | Output contains expected keywords |
| `SimilarityScorer` | Cosine similarity to reference output |

## Results

`EvalSummary` provides:
- `pass_rate` — fraction of cases that passed
- Per-case `EvalResult` with individual scorer results

## Event Collection

`EventCollector` captures `AgentEvent` variants during eval runs for detailed trajectory analysis.
