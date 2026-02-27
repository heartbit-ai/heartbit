//! Minimal agent example using heartbit as a library.
//!
//! Creates a single `AgentRunner` with an Anthropic provider and built-in tools,
//! then executes a task and prints the output.
//!
//! ```bash
//! export ANTHROPIC_API_KEY="sk-..."
//! cargo run -p heartbit --example simple_agent -- "What is 2 + 2?"
//! ```

use std::sync::Arc;

use heartbit::{AgentRunner, AnthropicProvider, BuiltinToolsConfig, builtin_tools};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Create an LLM provider from an API key.
    let api_key =
        std::env::var("ANTHROPIC_API_KEY").expect("set ANTHROPIC_API_KEY environment variable");
    let provider = Arc::new(AnthropicProvider::new(&api_key, "claude-sonnet-4-20250514"));

    // 2. Create built-in tools (bash, read, write, edit, grep, glob, etc.)
    let tools = builtin_tools(BuiltinToolsConfig::default());

    // 3. Build the agent.
    let runner = AgentRunner::builder(provider)
        .name("assistant")
        .system_prompt("You are a helpful assistant. Be concise.")
        .tools(tools)
        .max_turns(10)
        .max_tokens(4096)
        .build()?;

    // 4. Get task from CLI args or use default.
    let task = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "What is 2 + 2? Just give me the number.".into());

    // 5. Execute and print result.
    let output = runner.execute(&task).await?;
    println!("{}", output.result);
    eprintln!(
        "[tokens: {} in / {} out]",
        output.tokens_used.input_tokens, output.tokens_used.output_tokens
    );

    Ok(())
}
