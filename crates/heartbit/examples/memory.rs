//! Memory-equipped agent example.
//!
//! Creates an `InMemoryStore` and wires it into an agent, giving it tools
//! to store, recall, and manage persistent memories across turns.
//!
//! ```bash
//! export ANTHROPIC_API_KEY="sk-..."
//! cargo run -p heartbit --example memory
//! ```

use std::sync::Arc;

use heartbit::{AgentRunner, AnthropicProvider, InMemoryStore};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key =
        std::env::var("ANTHROPIC_API_KEY").expect("set ANTHROPIC_API_KEY environment variable");
    let provider = Arc::new(AnthropicProvider::new(api_key, "claude-sonnet-4-20250514"));

    let memory = Arc::new(InMemoryStore::new());

    let agent = AgentRunner::builder(provider)
        .name("memory-agent")
        .system_prompt(
            "You are an assistant with memory. Store important facts the user tells you \
             and recall them when asked. Use memory tools proactively.",
        )
        .memory(memory)
        .max_turns(10)
        .max_tokens(4096)
        .build()?;

    let output = agent
        .execute("Remember that my favorite color is blue, then recall all your memories.")
        .await?;

    println!("{}", output.result);

    Ok(())
}
