//! LLM-as-Judge guardrail example.
//!
//! Attaches a safety guardrail that uses a cheap judge model to evaluate
//! agent responses against custom criteria before they are returned.
//!
//! ```bash
//! export ANTHROPIC_API_KEY="sk-..."
//! cargo run -p heartbit --example guardrails
//! ```

use std::sync::Arc;

use heartbit::{AgentRunner, AnthropicProvider, BoxedProvider, LlmJudgeGuardrail};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key =
        std::env::var("ANTHROPIC_API_KEY").expect("set ANTHROPIC_API_KEY environment variable");
    let provider = Arc::new(AnthropicProvider::new(&api_key, "claude-sonnet-4-20250514"));

    // Use a cheap model as the safety judge.
    let judge = Arc::new(BoxedProvider::new(AnthropicProvider::new(
        &api_key,
        "claude-haiku-4-5-20251001",
    )));

    let guardrail = LlmJudgeGuardrail::builder(judge)
        .criterion("Response must not contain personal insults")
        .criterion("Response must not include made-up statistics")
        .build()?;

    let agent = AgentRunner::builder(provider)
        .name("safe-agent")
        .system_prompt("You are a helpful assistant. Be concise and factual.")
        .guardrail(Arc::new(guardrail))
        .max_turns(3)
        .max_tokens(2048)
        .build()?;

    let output = agent.execute("Tell me about climate change.").await?;
    println!("{}", output.result);

    Ok(())
}
