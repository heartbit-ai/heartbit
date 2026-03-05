//! Multi-agent orchestrator example.
//!
//! Creates an orchestrator with two sub-agents (researcher + writer) that
//! collaborate via delegation to complete a task.
//!
//! ```bash
//! export ANTHROPIC_API_KEY="sk-..."
//! cargo run -p heartbit --example multi_agent
//! ```

use std::sync::Arc;

use heartbit::{AnthropicProvider, Orchestrator};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key =
        std::env::var("ANTHROPIC_API_KEY").expect("set ANTHROPIC_API_KEY environment variable");
    let provider = Arc::new(AnthropicProvider::new(api_key, "claude-sonnet-4-20250514"));

    let mut orchestrator = Orchestrator::builder(provider)
        .sub_agent(
            "researcher",
            "Finds facts and data on a topic",
            "You are a research assistant. Find key facts and return them as bullet points.",
        )
        .sub_agent(
            "writer",
            "Writes polished prose from notes",
            "You are a writer. Turn bullet-point notes into a short, polished paragraph.",
        )
        .max_turns(5)
        .max_tokens(4096)
        .build()?;

    let output = orchestrator
        .run("Write a short paragraph about the Rust programming language.")
        .await?;

    println!("{}", output.result);
    eprintln!(
        "[total tokens: {} in / {} out]",
        output.tokens_used.input_tokens, output.tokens_used.output_tokens
    );

    Ok(())
}
