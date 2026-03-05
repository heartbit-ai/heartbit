//! Minimal single-agent example.
//!
//! Sends one task to an Anthropic-backed agent and prints the response.
//!
//! ```bash
//! export ANTHROPIC_API_KEY="sk-..."
//! cargo run -p heartbit --example hello_agent
//! ```

use std::sync::Arc;

use heartbit::{AgentRunner, AnthropicProvider};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key =
        std::env::var("ANTHROPIC_API_KEY").expect("set ANTHROPIC_API_KEY environment variable");
    let provider = Arc::new(AnthropicProvider::new(api_key, "claude-sonnet-4-20250514"));

    let agent = AgentRunner::builder(provider)
        .name("greeter")
        .system_prompt("You are a friendly assistant. Be concise.")
        .max_turns(1)
        .max_tokens(1024)
        .build()?;

    let output = agent.execute("Say hello in three languages.").await?;
    println!("{}", output.result);
    eprintln!(
        "[tokens: {} in / {} out, {} tool calls]",
        output.tokens_used.input_tokens, output.tokens_used.output_tokens, output.tool_calls_made
    );

    Ok(())
}
