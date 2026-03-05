//! Evaluation framework example.
//!
//! Defines eval cases with expected tool calls and output keywords, then
//! scores agent behavior using `TrajectoryScorer` and `KeywordScorer`.
//!
//! ```bash
//! export ANTHROPIC_API_KEY="sk-..."
//! cargo run -p heartbit --example eval
//! ```

use std::sync::Arc;

use heartbit::{
    AgentRunner, AnthropicProvider, EvalCase, EvalRunner, EvalSummary, KeywordScorer,
    TrajectoryScorer, build_eval_agent,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key =
        std::env::var("ANTHROPIC_API_KEY").expect("set ANTHROPIC_API_KEY environment variable");
    let provider = Arc::new(AnthropicProvider::new(api_key, "claude-sonnet-4-20250514"));

    // Build an eval-ready agent with event collection for trajectory scoring.
    let builder = AgentRunner::builder(provider)
        .name("eval-agent")
        .system_prompt("You are a helpful assistant. Be concise.")
        .max_turns(3)
        .max_tokens(1024);
    let (agent, collector) = build_eval_agent(builder)?;

    // Define test cases.
    let cases = vec![
        EvalCase::new("greeting", "Say hello")
            .expect_no_tools()
            .expect_output_contains("hello"),
        EvalCase::new("math", "What is 7 * 6? Just the number.")
            .expect_no_tools()
            .expect_output_contains("42"),
    ];

    // Run evaluations and score.
    let runner = EvalRunner::new()
        .scorer(TrajectoryScorer)
        .scorer(KeywordScorer);
    let results = runner.run(&agent, &cases).await;

    // Use the collector for trajectory data on the last case.
    let tool_calls = EvalRunner::collected_tool_calls(&collector);
    eprintln!("[tool calls captured: {tool_calls:?}]");

    let summary = EvalSummary::from_results(&results);
    println!("{summary}");

    Ok(())
}
