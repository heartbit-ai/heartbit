use std::sync::Arc;

use anyhow::{Context, Result, bail};
use tracing_subscriber::EnvFilter;

use heartbit::{AnthropicProvider, Orchestrator};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let api_key =
        std::env::var("ANTHROPIC_API_KEY").context("ANTHROPIC_API_KEY env var required")?;

    let task: String = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    if task.is_empty() {
        bail!("Usage: heartbit <task>");
    }

    let provider = Arc::new(AnthropicProvider::new(api_key, "claude-sonnet-4-20250514"));

    let orchestrator = Orchestrator::builder(provider)
        .sub_agent(
            "researcher",
            "Research specialist who gathers information and facts",
            "You are a research specialist. Your job is to gather relevant information, \
             facts, and data about the topic you're given. Be thorough and cite your \
             reasoning. Focus on accuracy and completeness.",
        )
        .sub_agent(
            "analyst",
            "Analytical thinker who evaluates and synthesizes information",
            "You are an analytical expert. Your job is to evaluate information critically, \
             identify patterns, weigh pros and cons, and provide structured analysis. \
             Be objective and thorough in your reasoning.",
        )
        .sub_agent(
            "writer",
            "Clear communicator who produces well-structured output",
            "You are a writing specialist. Your job is to take information and analysis \
             and produce clear, well-structured, and engaging output. Focus on clarity, \
             organization, and readability.",
        )
        .build();

    let output = orchestrator.run(&task).await?;

    println!("{}", output.result);
    eprintln!(
        "\n---\nTokens used: {} in / {} out | Tool calls: {}",
        output.tokens_used.input_tokens, output.tokens_used.output_tokens, output.tool_calls_made,
    );

    Ok(())
}
