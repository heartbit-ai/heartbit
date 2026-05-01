//! Sequential workflow: a researcher agent feeds into a writer agent.
//!
//! `cargo run -p heartbit-core --example sequential_agent`

use std::sync::Arc;

use heartbit_core::{
    AgentRunner, AnthropicProvider, BoxedProvider, RetryingProvider, SequentialAgent,
};

#[tokio::main]
async fn main() -> Result<(), heartbit_core::Error> {
    let api_key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY must be set");
    let provider = Arc::new(BoxedProvider::new(RetryingProvider::with_defaults(
        AnthropicProvider::new(&api_key, "claude-sonnet-4-20250514"),
    )));

    let researcher = AgentRunner::builder(provider.clone())
        .system_prompt(
            "Research the topic and produce a concise factual summary (3 bullet points).",
        )
        .build()?;

    let writer = AgentRunner::builder(provider)
        .system_prompt("Rewrite the input as a single engaging paragraph for a general audience.")
        .build()?;

    let workflow = SequentialAgent::builder()
        .agent(researcher)
        .agent(writer)
        .build()?;

    let output = workflow
        .execute("The history of the Rust programming language")
        .await?;
    println!("{}", output.result);
    println!(
        "Total tokens used: {} in / {} out",
        output.tokens_used.input_tokens, output.tokens_used.output_tokens,
    );
    Ok(())
}
