use std::sync::Arc;

use anyhow::{Context, Result, bail};
use tracing_subscriber::EnvFilter;

use heartbit::llm::anthropic::AnthropicProvider;
use heartbit::llm::openrouter::OpenRouterProvider;
use heartbit::{AgentOutput, LlmProvider, Orchestrator};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let task: String = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    if task.is_empty() {
        bail!("Usage: heartbit <task>");
    }

    // Provider selection: check env vars
    let provider_name = std::env::var("HEARTBIT_PROVIDER").unwrap_or_else(|_| {
        if std::env::var("OPENROUTER_API_KEY").is_ok() {
            "openrouter".into()
        } else {
            "anthropic".into()
        }
    });

    match provider_name.as_str() {
        "anthropic" => {
            let api_key = std::env::var("ANTHROPIC_API_KEY")
                .context("ANTHROPIC_API_KEY env var required for anthropic provider")?;
            let model = std::env::var("HEARTBIT_MODEL")
                .unwrap_or_else(|_| "claude-sonnet-4-20250514".into());
            let provider = Arc::new(AnthropicProvider::new(api_key, model));
            let output = run_orchestrator(provider, &task).await?;
            print_output(&output);
        }
        "openrouter" => {
            let api_key = std::env::var("OPENROUTER_API_KEY")
                .context("OPENROUTER_API_KEY env var required for openrouter provider")?;
            let model = std::env::var("HEARTBIT_MODEL")
                .unwrap_or_else(|_| "anthropic/claude-sonnet-4".into());
            let provider = Arc::new(OpenRouterProvider::new(api_key, model));
            let output = run_orchestrator(provider, &task).await?;
            print_output(&output);
        }
        other => bail!("Unknown provider: {other}. Use 'anthropic' or 'openrouter'."),
    }

    Ok(())
}

async fn run_orchestrator<P: LlmProvider + 'static>(
    provider: Arc<P>,
    task: &str,
) -> Result<AgentOutput> {
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

    let output = orchestrator.run(task).await?;
    Ok(output)
}

fn print_output(output: &AgentOutput) {
    println!("{}", output.result);
    eprintln!(
        "\n---\nTokens used: {} in / {} out | Tool calls: {}",
        output.tokens_used.input_tokens, output.tokens_used.output_tokens, output.tool_calls_made,
    );
}
