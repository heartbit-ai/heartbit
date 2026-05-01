//! Minimal "hello world" agent.
//!
//! `ANTHROPIC_API_KEY=sk-... cargo run -p heartbit-core --example hello_agent`

use std::sync::Arc;

use heartbit_core::{AgentRunner, AnthropicProvider, BoxedProvider, RetryingProvider};

#[tokio::main]
async fn main() -> Result<(), heartbit_core::Error> {
    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .expect("set ANTHROPIC_API_KEY environment variable");

    let provider = Arc::new(BoxedProvider::new(RetryingProvider::with_defaults(
        AnthropicProvider::new(api_key, "claude-sonnet-4-20250514"),
    )));

    let agent = AgentRunner::builder(provider)
        .system_prompt("You are a helpful assistant. Answer in one short paragraph.")
        .build()?;

    let output = agent.execute("What is Rust?").await?;
    println!("{}", output.result);
    Ok(())
}
