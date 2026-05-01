# Eval Framework

## Why eval

Agents are LLM apps; LLMs drift. A prompt that produced the right
trajectory yesterday can return a different answer tomorrow against
the same model — let alone after a model upgrade or a tool change.
The eval framework lets you codify "this prompt + this tool stack
should do X for these inputs" and catch regressions in CI before they
ship.

## EvalRunner and EvalCase

An [`EvalCase`](https://docs.rs/heartbit-core/latest/heartbit_core/eval/struct.EvalCase.html)
captures the input task plus the expectations: required keywords,
forbidden keywords, expected tool trajectory, optional reference
output, and budget caps. An
[`EvalRunner`](https://docs.rs/heartbit-core/latest/heartbit_core/eval/struct.EvalRunner.html)
holds a list of scorers and produces an `EvalResult` per case — either
by running the agent and capturing its output, or by scoring a
pre-collected output via `score_result`. The example below uses the
latter shape, so it runs without any API key:

```rust,no_run
{{#include ../../../crates/heartbit-core/examples/eval.rs}}
```

## Built-in scorers

Heartbit ships seven scorers covering trajectory, output content,
similarity, and operational budgets:

- `TrajectoryScorer`: matches the actual sequence of tool calls
  against `expect_tool` / `expect_tool_at` / `expect_no_tools`
  expectations on the case.
- `KeywordScorer`: counts required `output_contains` and forbidden
  `output_not_contains` keywords in the final response, case-insensitive.
- `SimilarityScorer`: Rouge-1 unigram F1 between the output and the
  case's `reference_output`; default pass threshold is `0.3`.
- `CostScorer`: aggregates `LlmResponse` events from a shared
  collector and asserts total estimated cost stays within
  `expect_max_cost_usd` (or the runner default).
- `LatencyScorer`: asserts cumulative LLM wall-clock latency stays
  under `expect_max_latency_ms`.
- `ToolCallCountScorer`: bounds the number of tool invocations per
  case via `expect_max_tool_calls`.
- `SafetyScorer`: pass rate of guardrail evaluations observed during
  the run via `GuardrailDenied` events.

## Writing custom scorers

Scorers implement the [`EvalScorer`](https://docs.rs/heartbit-core/latest/heartbit_core/eval/trait.EvalScorer.html)
trait — three methods, no async, no allocation contract beyond
returning the per-scorer details vec:

```rust,ignore
use heartbit_core::eval::{EvalCase, EvalScorer};

struct LengthScorer { max_chars: usize }

impl EvalScorer for LengthScorer {
    fn name(&self) -> &str { "length" }
    fn score(&self, _case: &EvalCase, output: &str, _tools: &[String]) -> (f64, Vec<String>) {
        let ok = output.chars().count() <= self.max_chars;
        (if ok { 1.0 } else { 0.0 }, vec![format!("len={}", output.chars().count())])
    }
}
```

Hand it to the runner via `.scorer(LengthScorer { max_chars: 280 })`
alongside the built-ins; first failing scorer fails the case.

## Running evals in CI

The cleanest pattern is a small eval binary in your project that
loads cases from disk (TOML or JSON), runs them through an
`EvalRunner` against your real agent, prints the human-readable
`EvalSummary`, and emits a machine-readable JSON report alongside.
Gate the CI step on `EvalSummary::pass_rate >= threshold` (or on a
zero-regression `EvalComparison` against a stored baseline). For the
end-to-end iteration loop — write a failing case, fix the prompt,
re-run, commit the new baseline — see the
[Eval-driven prompt iteration](../recipes/eval-driven.md) recipe.
