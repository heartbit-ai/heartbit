# Eval-driven prompt iteration

## Goal

Catch prompt regressions automatically. Codify "this prompt + tool
stack should do X for these inputs" as eval cases, run them in CI, and
fail the build if the score drops.

## Solution

Write 5–10 `EvalCase`s that cover your core scenarios plus a few edge
cases. Each case pairs an input prompt with expectations:
`expect_output_contains`, `expect_tool`, `expect_no_tools`,
`reference_output`, and so on. Run them through an `EvalRunner`
configured with the scorers that match your expectations:
`KeywordScorer`, `SimilarityScorer`, `TrajectoryScorer`,
`CostScorer`, `SafetyScorer`.

The runner produces an `EvalSummary` you can serialize to JSON and
gate CI on. For deterministic runs (fast and rate-limit-free), feed
the runner pre-recorded actuals via `score_result` instead of letting
it call the live LLM. The example below does exactly that — no API
key required.

```rust,no_run
{{#include ../../../crates/heartbit-core/examples/eval.rs}}
```

Iterate the system prompt until every case passes, then commit the
case set alongside the prompt. From that point on, any prompt change
that drops a score is caught by CI before merge.

## CI integration

A minimal GitHub Actions workflow that fails the build if the eval
binary exits non-zero:

```yaml
# .github/workflows/eval.yml
name: eval
on: [pull_request]
jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - run: cargo run --release -p heartbit-core --example eval
```

Have the example print a JSON summary to stdout and `std::process::exit(1)`
when any aggregate score drops below your threshold. The Action's
default behaviour — fail on non-zero exit — does the gating for you.

## Notes

- See the [Eval Framework chapter](../eval/README.md) for the full
  list of built-in scorers.
- For deterministic eval runs, use a stub provider that returns
  recorded responses; this keeps CI fast and free of rate limits.
