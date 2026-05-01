//! Implementing the `Tool` trait by hand.
//!
//! Defines a `WordCount` tool, attaches it to an agent, and lets the LLM
//! decide when to call it.
//!
//! `ANTHROPIC_API_KEY=sk-... cargo run -p heartbit-core --example custom_tool`

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use heartbit_core::llm::types::ToolDefinition;
use heartbit_core::{
    AgentRunner, AnthropicProvider, BoxedProvider, RetryingProvider, Tool, ToolOutput,
};

/// A tool that counts the words in a string.
struct WordCount;

impl Tool for WordCount {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "word_count".into(),
            description: "Count the number of whitespace-separated words in a string.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to count words in.",
                    }
                },
                "required": ["text"],
            }),
        }
    }

    fn execute(
        &self,
        input: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, heartbit_core::Error>> + Send + '_>> {
        Box::pin(async move {
            let text = match input.get("text").and_then(|v| v.as_str()) {
                Some(t) => t,
                None => return Ok(ToolOutput::error("missing required field `text`")),
            };
            let count = text.split_whitespace().count();
            Ok(ToolOutput::success(count.to_string()))
        })
    }
}

#[tokio::main]
async fn main() -> Result<(), heartbit_core::Error> {
    let api_key =
        std::env::var("ANTHROPIC_API_KEY").expect("set ANTHROPIC_API_KEY environment variable");

    let provider = Arc::new(BoxedProvider::new(RetryingProvider::with_defaults(
        AnthropicProvider::new(api_key, "claude-sonnet-4-20250514"),
    )));

    let tools: Vec<Arc<dyn Tool>> = vec![Arc::new(WordCount)];

    let agent = AgentRunner::builder(provider)
        .system_prompt("Use the word_count tool to answer questions about word counts.")
        .tools(tools)
        .build()?;

    let output = agent
        .execute("How many words are in: 'The quick brown fox jumps over the lazy dog'?")
        .await?;

    println!("{}", output.result);
    Ok(())
}
