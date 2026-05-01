//! Building an Orchestrator with multiple sub-agents.
//!
//! Wires three named sub-agents — researcher, coder, reviewer — and
//! lets the orchestrator's LLM pick which one to dispatch via the
//! `delegate_task` tool.
//!
//! `ANTHROPIC_API_KEY=sk-... cargo run -p heartbit-core --example multi_agent`

use std::sync::Arc;

use heartbit_core::{
    AnthropicProvider, BoxedProvider, Orchestrator, RetryingProvider,
};

#[tokio::main]
async fn main() -> Result<(), heartbit_core::Error> {
    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .expect("set ANTHROPIC_API_KEY environment variable");

    let provider = Arc::new(BoxedProvider::new(RetryingProvider::with_defaults(
        AnthropicProvider::new(&api_key, "claude-sonnet-4-20250514"),
    )));

    let mut orchestrator = Orchestrator::builder(provider)
        .sub_agent(
            "researcher",
            "Investigates a topic and returns a factual summary.",
            "You are a research assistant. Produce a concise factual summary \
             of the requested topic, with no opinions.",
        )
        .sub_agent(
            "coder",
            "Writes or modifies Rust code given clear requirements.",
            "You are a senior Rust engineer. Produce idiomatic, well-tested \
             code that compiles cleanly with `cargo clippy -- -D warnings`.",
        )
        .sub_agent(
            "reviewer",
            "Reviews code or text for issues and suggests improvements.",
            "You are a code reviewer. Identify bugs, style issues, and \
             missing test cases. Be specific and concrete.",
        )
        .max_turns(10)
        .build()?;

    let output = orchestrator
        .run("Summarise the trade-offs of using Rust for an LLM agent runtime.")
        .await?;

    println!("{}", output.result);
    println!(
        "Total tokens: {} in / {} out",
        output.tokens_used.input_tokens, output.tokens_used.output_tokens,
    );
    Ok(())
}
