//! A small agent with file-reading tools, a turn cap, and an event listener.
//!
//! Demonstrates the parts of `AgentRunnerBuilder` covered in the Agents
//! chapter: `system_prompt`, `tools`, `max_turns`, and `on_event`.
//!
//! `ANTHROPIC_API_KEY=sk-... cargo run -p heartbit-core --example simple_agent`

use std::sync::Arc;

use heartbit_core::tool::builtins::{BuiltinToolsConfig, builtin_tools};
use heartbit_core::{AgentEvent, AgentRunner, AnthropicProvider, BoxedProvider, RetryingProvider};

#[tokio::main]
async fn main() -> Result<(), heartbit_core::Error> {
    let api_key =
        std::env::var("ANTHROPIC_API_KEY").expect("set ANTHROPIC_API_KEY environment variable");

    let provider = Arc::new(BoxedProvider::new(RetryingProvider::with_defaults(
        AnthropicProvider::new(api_key, "claude-sonnet-4-20250514"),
    )));

    // The default builtin set excludes dangerous tools (bash). It includes
    // read/write/edit/patch/list/glob/grep, plus web_fetch/web_search.
    let tools = builtin_tools(BuiltinToolsConfig::default());

    // Print a one-line trace for each lifecycle event.
    let on_event = Arc::new(|event: AgentEvent| match event {
        AgentEvent::TurnStarted { turn, .. } => eprintln!("[turn {turn}]"),
        AgentEvent::ToolCallStarted { tool_name, .. } => eprintln!("  -> {tool_name}"),
        AgentEvent::RunCompleted { total_usage, .. } => {
            eprintln!(
                "[done] {} in / {} out",
                total_usage.input_tokens, total_usage.output_tokens,
            );
        }
        _ => {}
    });

    let agent = AgentRunner::builder(provider)
        .system_prompt("You are a helpful coding assistant.")
        .tools(tools)
        .max_turns(8)
        .on_event(on_event)
        .build()?;

    let output = agent
        .execute(
            "List the files in the current directory and summarise what kind of project this is.",
        )
        .await?;

    println!("{}", output.result);
    Ok(())
}
