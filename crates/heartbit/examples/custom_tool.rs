//! Custom tool example: implementing the `Tool` trait for a domain-specific tool.
//!
//! Shows how to create a `PriceLookupTool` and wire it into an agent alongside
//! built-in tools.
//!
//! ```bash
//! export ANTHROPIC_API_KEY="sk-..."
//! cargo run -p heartbit --example custom_tool
//! ```

use std::pin::Pin;
use std::sync::Arc;

use heartbit::{
    AgentRunner, AnthropicProvider, BuiltinToolsConfig, Tool, ToolDefinition, ToolOutput,
    builtin_tools,
};
use serde_json::json;

/// A domain-specific tool that looks up product prices.
struct PriceLookupTool;

impl Tool for PriceLookupTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "price_lookup".into(),
            description: "Look up the price of a product by name. Returns the price in USD.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The product name to look up"
                    }
                },
                "required": ["product"]
            }),
        }
    }

    fn execute<'a>(
        &'a self,
        input: serde_json::Value,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<ToolOutput, heartbit::Error>> + Send + 'a>>
    {
        Box::pin(async move {
            let product = input
                .get("product")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");

            // Simulate a price database lookup.
            let price = match product.to_lowercase().as_str() {
                "widget" => 9.99,
                "gadget" => 24.99,
                "thingamajig" => 14.50,
                _ => return Ok(ToolOutput::error(format!("Product '{product}' not found"))),
            };

            Ok(ToolOutput::success(format!(
                "Product: {product}\nPrice: ${price:.2}"
            )))
        })
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key =
        std::env::var("ANTHROPIC_API_KEY").expect("set ANTHROPIC_API_KEY environment variable");
    let provider = Arc::new(AnthropicProvider::new(&api_key, "claude-sonnet-4-20250514"));

    // Combine built-in tools with our custom tool.
    let mut tools = builtin_tools(BuiltinToolsConfig::default());
    tools.push(Arc::new(PriceLookupTool));

    let runner = AgentRunner::builder(provider)
        .name("quoter")
        .system_prompt(
            "You are a sales assistant. Use the price_lookup tool to find product prices \
             when asked. Be concise and helpful.",
        )
        .tools(tools)
        .max_turns(5)
        .max_total_tokens(50000) // Budget: 50k tokens per execution
        .build()?;

    let output = runner
        .execute("How much does a widget cost? And a gadget?")
        .await?;

    println!("{}", output.result);
    if let Some(cost) = output.estimated_cost_usd {
        eprintln!("[estimated cost: ${cost:.4}]");
    }

    Ok(())
}
