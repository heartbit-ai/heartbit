//! MCP (Model Context Protocol) agent example.
//!
//! Shows how to connect to an MCP tool server via stdio transport and use its
//! tools alongside built-in tools.
//!
//! ```bash
//! export ANTHROPIC_API_KEY="sk-..."
//! cargo run -p heartbit --example mcp_agent -- "List files in /tmp"
//! ```
//!
//! This example assumes an MCP server binary is available. If you don't have
//! one, the agent will work with built-in tools only.

use std::collections::HashMap;
use std::sync::Arc;

use heartbit::{AgentRunner, AnthropicProvider, BuiltinToolsConfig, McpClient, builtin_tools};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key =
        std::env::var("ANTHROPIC_API_KEY").expect("set ANTHROPIC_API_KEY environment variable");
    let provider = Arc::new(AnthropicProvider::new(&api_key, "claude-sonnet-4-20250514"));

    let mut tools = builtin_tools(BuiltinToolsConfig::default());

    // Try to connect to an MCP server via stdio.
    // The server command is taken from args, e.g.:
    //   cargo run --example mcp_agent -- npx -y @anthropic-ai/mcp-server-filesystem /tmp
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
                eprintln!("[MCP connection failed: {e}, continuing with built-in tools only]");
            }
        }
    }

    let runner = AgentRunner::builder(provider)
        .name("mcp-agent")
        .system_prompt("You are a helpful assistant with access to tools. Be concise.")
        .tools(tools)
        .max_turns(10)
        .build()?;

    let task = if mcp_args.is_empty() {
        "What is 2 + 2? Just answer with the number."
    } else {
        "List the available tools you have access to."
    };

    let output = runner.execute(task).await?;
    println!("{}", output.result);

    Ok(())
}
