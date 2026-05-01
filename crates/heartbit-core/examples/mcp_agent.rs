//! Connecting to an MCP (Model Context Protocol) server.
//!
//! Spawns an MCP server over stdio, discovers its tools, and hands them
//! to an agent alongside the built-in tool set. If no MCP command is
//! supplied — or the connection fails — the agent falls back to the
//! built-ins only, so the example always compiles and runs.
//!
//! ```bash
//! ANTHROPIC_API_KEY=sk-... cargo run -p heartbit-core --example mcp_agent -- \
//!     npx -y @anthropic-ai/mcp-server-filesystem /tmp
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use heartbit_core::tool::builtins::{BuiltinToolsConfig, builtin_tools};
use heartbit_core::{AgentRunner, AnthropicProvider, BoxedProvider, McpClient, RetryingProvider};

#[tokio::main]
async fn main() -> Result<(), heartbit_core::Error> {
    let api_key =
        std::env::var("ANTHROPIC_API_KEY").expect("set ANTHROPIC_API_KEY environment variable");

    let provider = Arc::new(BoxedProvider::new(RetryingProvider::with_defaults(
        AnthropicProvider::new(&api_key, "claude-sonnet-4-20250514"),
    )));

    let mut tools = builtin_tools(BuiltinToolsConfig::default());

    // Anything after `--` is treated as the MCP server command.
    let mcp_args: Vec<String> = std::env::args().skip(1).collect();
    if !mcp_args.is_empty() {
        eprintln!("[connecting to MCP server: {}]", mcp_args.join(" "));
        match McpClient::connect_stdio(&mcp_args[0], &mcp_args[1..], &HashMap::new()).await {
            Ok(client) => {
                let mcp_tools = client.into_tools();
                eprintln!("[discovered {} MCP tools]", mcp_tools.len());
                tools.extend(mcp_tools);
            }
            Err(e) => {
                eprintln!("[MCP connection failed: {e}; falling back to built-ins]");
            }
        }
    }

    let agent = AgentRunner::builder(provider)
        .system_prompt("You are a helpful assistant. Prefer MCP tools when available.")
        .tools(tools)
        .max_turns(10)
        .build()?;

    let task = if mcp_args.is_empty() {
        "List the names of the built-in tools you have access to."
    } else {
        "List the tools you have access to from the connected MCP server."
    };

    let output = agent.execute(task).await?;
    println!("{}", output.result);
    Ok(())
}
