//! DAG workflow: a planner feeds two parallel workers (research + critique),
//! then a synthesizer combines their outputs.
//!
//! `cargo run -p heartbit-core --example dag_agent`

use std::sync::Arc;

use heartbit_core::{AgentRunner, AnthropicProvider, BoxedProvider, DagAgent, RetryingProvider};

fn make_agent(
    provider: Arc<BoxedProvider>,
    prompt: &str,
) -> Result<AgentRunner<BoxedProvider>, heartbit_core::Error> {
    AgentRunner::builder(provider).system_prompt(prompt).build()
}

#[tokio::main]
async fn main() -> Result<(), heartbit_core::Error> {
    let api_key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY must be set");
    let provider = Arc::new(BoxedProvider::new(RetryingProvider::with_defaults(
        AnthropicProvider::new(&api_key, "claude-sonnet-4-20250514"),
    )));

    let dag = DagAgent::builder()
        .node(
            "planner",
            make_agent(
                provider.clone(),
                "Outline the question into 2 parallel research questions.",
            )?,
        )
        .node(
            "research",
            make_agent(provider.clone(), "Answer the research question concisely.")?,
        )
        .node(
            "critique",
            make_agent(
                provider.clone(),
                "Identify weaknesses in the proposed answer.",
            )?,
        )
        .node(
            "synthesizer",
            make_agent(
                provider,
                "Combine research and critique into a final answer.",
            )?,
        )
        .edge("planner", "research")
        .edge("planner", "critique")
        .edge("research", "synthesizer")
        .edge("critique", "synthesizer")
        .build()?;

    let output = dag
        .execute("Should we use Rust for our agent runtime?")
        .await?;
    println!("{}", output.result);
    Ok(())
}
