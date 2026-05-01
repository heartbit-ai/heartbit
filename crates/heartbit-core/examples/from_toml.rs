//! Loading agent configuration from a TOML file.
//!
//! `cargo run -p heartbit-core --example from_toml`

use heartbit_core::config::HeartbitConfig;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let toml_text = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "assistant"
description = "A helpful general-purpose assistant."
system_prompt = "You are a helpful assistant."
max_turns = 10
max_tokens = 4096
"#;

    // `from_toml` parses and validates the config in one step.
    let config = HeartbitConfig::from_toml(toml_text)?;

    println!(
        "Loaded provider {} ({}); {} agent(s)",
        config.provider.name,
        config.provider.model,
        config.agents.len()
    );
    for agent in &config.agents {
        println!(
            "  - {} (max_turns={:?}, max_tokens={:?})",
            agent.name, agent.max_turns, agent.max_tokens
        );
    }
    Ok(())
}
