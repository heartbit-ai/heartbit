//! Score agent outputs against eval cases without an LLM call.
//!
//! Builds an `EvalRunner` with three pure-function scorers, runs three
//! cases against pre-recorded "actual outputs" via `score_result`, and
//! prints the aggregate `EvalSummary`. No API key required.
//!
//! `cargo run -p heartbit-core --example eval`

use heartbit_core::eval::{
    EvalCase, EvalResult, EvalRunner, EvalSummary, KeywordScorer, SimilarityScorer,
    TrajectoryScorer,
};

fn main() {
    let cases = vec![
        EvalCase::new("greeting", "Say hello in one word.")
            .expect_output_contains("hello")
            .expect_no_tools(),
        EvalCase::new("capital", "What is the capital of France?")
            .expect_output_contains("Paris")
            .reference_output("The capital of France is Paris."),
        EvalCase::new("file-read", "Read /tmp/notes.txt and summarise it.")
            .expect_tool("read_file")
            .expect_output_contains("summary"),
    ];

    // Stand in for real agent runs. In a real eval you'd execute the
    // agent and feed `output.result` + collected tool calls instead.
    let actuals: Vec<(&str, Vec<String>)> = vec![
        ("hello", vec![]),
        ("The capital of France is Paris.", vec![]),
        (
            "Here is a summary of the file contents.",
            vec!["read_file".into()],
        ),
    ];

    let runner = EvalRunner::new()
        .scorer(TrajectoryScorer)
        .scorer(KeywordScorer)
        .scorer(SimilarityScorer);

    let results: Vec<EvalResult> = cases
        .iter()
        .zip(actuals)
        .map(|(case, (output, tools))| runner.score_result(case, output, &tools, None))
        .collect();

    let summary = EvalSummary::from_results(&results);
    println!("{summary}");
}
