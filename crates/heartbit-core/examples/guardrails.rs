//! Composing several guardrails on a single agent.
//!
//! Wires a secret scanner, a PII redactor, and an LLM judge in front
//! of an `AgentRunner`. First `Deny` wins; `Warn` is logged via the
//! event listener but does not block.
//!
//! `ANTHROPIC_API_KEY=sk-... cargo run -p heartbit-core --example guardrails`

use std::sync::Arc;

use heartbit_core::{
    AgentEvent, AgentRunner, AnthropicProvider, BoxedProvider, Guardrail, LlmJudgeGuardrail,
    PiiAction, PiiGuardrail, RetryingProvider, SecretAction, SecretScannerGuardrail,
};

#[tokio::main]
async fn main() -> Result<(), heartbit_core::Error> {
    let api_key =
        std::env::var("ANTHROPIC_API_KEY").expect("set ANTHROPIC_API_KEY environment variable");

    let provider = Arc::new(BoxedProvider::new(RetryingProvider::with_defaults(
        AnthropicProvider::new(&api_key, "claude-sonnet-4-20250514"),
    )));

    // A judge model can be cheaper than the main agent — wire a separate
    // provider in production. Here we reuse the same one for brevity.
    let judge_provider = provider.clone();

    let secret_scanner = SecretScannerGuardrail::builder()
        .action(SecretAction::Redact)
        .build();

    let pii = PiiGuardrail::all_builtin(PiiAction::Redact);

    let llm_judge = LlmJudgeGuardrail::builder(judge_provider)
        .criterion("response must not give legal advice")
        .criterion("response must stay on the technical topic")
        .build()?;

    let guardrails: Vec<Arc<dyn Guardrail>> =
        vec![Arc::new(secret_scanner), Arc::new(pii), Arc::new(llm_judge)];

    let on_event = Arc::new(|event: AgentEvent| {
        if let AgentEvent::GuardrailDenied { reason, .. } = event {
            eprintln!("[guardrail-denied] {reason}");
        }
    });

    let agent = AgentRunner::builder(provider)
        .system_prompt("You are a helpful technical assistant.")
        .guardrails(guardrails)
        .on_event(on_event)
        .build()?;

    let output = agent
        .execute("Summarise the difference between TCP and UDP in two sentences.")
        .await?;
    println!("{}", output.result);
    Ok(())
}
