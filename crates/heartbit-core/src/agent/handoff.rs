//! Agent handoff runner.
//!
//! Manages conversation transfer between peer agents. When an agent calls the
//! `handoff` tool, the `HandoffRunner` detects the sentinel output and routes
//! the conversation to the target agent with appropriate context transfer.

use std::collections::HashMap;
use std::sync::Arc;

use crate::error::Error;
use crate::llm::LlmProvider;
use crate::llm::types::TokenUsage;
use crate::tool::handoff::{
    HandoffContextMode, HandoffTarget, HandoffTool, parse_handoff_sentinel,
};

use super::{AgentOutput, AgentRunner};

/// Runs agents with handoff support. When an agent calls `handoff`, the runner
/// transfers conversation control to the target agent with context forwarding.
///
/// Unlike the orchestrator (centralized control), handoff is peer-to-peer:
/// each agent can transfer directly to any configured target without going
/// through a coordinator.
pub struct HandoffRunner<P: LlmProvider> {
    agents: HashMap<String, AgentRunner<P>>,
    initial_agent: String,
    max_handoffs: usize,
}

impl<P: LlmProvider> std::fmt::Debug for HandoffRunner<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HandoffRunner")
            .field("initial_agent", &self.initial_agent)
            .field("max_handoffs", &self.max_handoffs)
            .field("agent_count", &self.agents.len())
            .finish()
    }
}

/// Builder for [`HandoffRunner`].
pub struct HandoffRunnerBuilder<P: LlmProvider> {
    agents: HashMap<String, AgentRunner<P>>,
    initial_agent: Option<String>,
    max_handoffs: Option<usize>,
}

impl<P: LlmProvider> HandoffRunner<P> {
    /// Create a new [`HandoffRunnerBuilder`] for configuring a handoff runner.
    pub fn builder() -> HandoffRunnerBuilder<P> {
        HandoffRunnerBuilder {
            agents: HashMap::new(),
            initial_agent: None,
            max_handoffs: None,
        }
    }

    /// Execute the handoff loop starting from the initial agent.
    ///
    /// The initial agent receives the task. If it hands off, the target agent
    /// receives the conversation context and continues. The loop ends when an
    /// agent completes without handing off, or when `max_handoffs` is exceeded.
    pub async fn execute(&self, task: &str) -> Result<AgentOutput, Error> {
        let mut current_agent = self.initial_agent.clone();
        let mut total_usage = TokenUsage::default();
        let mut total_tool_calls = 0usize;
        let mut total_cost: Option<f64> = None;
        let mut effective_task = task.to_string();
        let mut handoff_count = 0;
        // AC3: accumulate every hop's output so a later agent in a multi-hop
        // chain (A→B→C) sees ALL prior agents' contributions in `Full` mode, not
        // just the immediately-preceding one. Previously `Full` mode embedded
        // only the current agent's output, dropping intermediate context.
        let mut transcript = String::new();

        loop {
            let agent = self.agents.get(&current_agent).ok_or_else(|| {
                Error::Agent(format!("handoff target agent '{current_agent}' not found"))
            })?;

            let output = agent
                .execute(&effective_task)
                .await
                .map_err(|e| e.accumulate_usage(total_usage))?;
            output.accumulate_into(&mut total_usage, &mut total_tool_calls, &mut total_cost);

            // Check if the output contains a handoff sentinel
            if let Some((target, context_mode, reason)) = parse_handoff_sentinel(&output.result) {
                handoff_count += 1;
                if handoff_count > self.max_handoffs {
                    // Max handoffs exceeded — return last output with warning
                    let mut final_output = output;
                    final_output.tokens_used = total_usage;
                    final_output.tool_calls_made = total_tool_calls;
                    final_output.estimated_cost_usd = total_cost;
                    final_output.result = format!(
                        "[handoff limit reached after {} handoffs]\n{}",
                        self.max_handoffs, final_output.result
                    );
                    return Ok(final_output);
                }

                // Verify target exists
                if !self.agents.contains_key(&target) {
                    return Err(Error::Agent(format!(
                        "handoff target '{target}' not found. Available: {}",
                        self.agents.keys().cloned().collect::<Vec<_>>().join(", ")
                    )));
                }

                // Append this hop to the running transcript before building the
                // next task, so `Full` mode reflects the whole chain.
                transcript.push_str(&format!(
                    "## Handoff from {current_agent}\nReason: {reason}\n\n{result}\n\n",
                    result = output.result,
                ));

                // Build context for the target agent
                effective_task = match context_mode {
                    HandoffContextMode::Full => {
                        format!(
                            "## Original task\n{task}\n\n\
                             ## Conversation so far\n{transcript}"
                        )
                    }
                    HandoffContextMode::Summary => {
                        format!(
                            "## Handoff from {current_agent}\n\
                             Reason: {reason}\n\n\
                             ## Original task\n{task}"
                        )
                    }
                };

                current_agent = target;
            } else {
                // No handoff — agent completed normally
                let mut final_output = output;
                final_output.tokens_used = total_usage;
                final_output.tool_calls_made = total_tool_calls;
                final_output.estimated_cost_usd = total_cost;
                return Ok(final_output);
            }
        }
    }
}

impl<P: LlmProvider> HandoffRunnerBuilder<P> {
    /// Add an agent with its handoff targets.
    ///
    /// The `handoff_targets` describe which agents this agent can hand off to.
    /// A `HandoffTool` is automatically registered on the agent's tool set.
    pub fn agent(mut self, name: impl Into<String>, runner: AgentRunner<P>) -> Self {
        let name = name.into();
        self.agents.insert(name, runner);
        self
    }

    /// Set the initial agent that receives the task.
    pub fn initial_agent(mut self, name: impl Into<String>) -> Self {
        self.initial_agent = Some(name.into());
        self
    }

    /// Set the maximum number of handoffs allowed (prevents infinite ping-pong).
    pub fn max_handoffs(mut self, max: usize) -> Self {
        self.max_handoffs = Some(max);
        self
    }

    /// Build the [`HandoffRunner`].
    pub fn build(self) -> Result<HandoffRunner<P>, Error> {
        if self.agents.is_empty() {
            return Err(Error::Config(
                "HandoffRunner requires at least one agent".into(),
            ));
        }
        let initial_agent = self
            .initial_agent
            .ok_or_else(|| Error::Config("HandoffRunner requires initial_agent".into()))?;
        if !self.agents.contains_key(&initial_agent) {
            return Err(Error::Config(format!(
                "initial_agent '{initial_agent}' not found in registered agents"
            )));
        }
        let max_handoffs = self
            .max_handoffs
            .ok_or_else(|| Error::Config("HandoffRunner requires max_handoffs".into()))?;
        if max_handoffs == 0 {
            return Err(Error::Config(
                "HandoffRunner max_handoffs must be at least 1".into(),
            ));
        }

        Ok(HandoffRunner {
            agents: self.agents,
            initial_agent,
            max_handoffs,
        })
    }
}

/// Create a `HandoffTool` for an agent, listing its available handoff targets.
///
/// This is a convenience function for building agents with handoff support.
pub fn make_handoff_tool(targets: Vec<HandoffTarget>) -> Arc<dyn crate::tool::Tool> {
    Arc::new(HandoffTool::new(targets))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::test_helpers::{MockProvider, make_agent};
    use crate::tool::handoff::HANDOFF_SENTINEL;

    // -----------------------------------------------------------------------
    // Builder validation tests
    // -----------------------------------------------------------------------

    #[test]
    fn builder_rejects_empty_agents() {
        let result = HandoffRunner::<MockProvider>::builder()
            .initial_agent("triage")
            .max_handoffs(3)
            .build();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("at least one"));
    }

    #[test]
    fn builder_rejects_missing_initial_agent() {
        let provider = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "x", 1, 1,
        )]));
        let result = HandoffRunner::builder()
            .agent("a", make_agent(provider, "a"))
            .max_handoffs(3)
            .build();
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("requires initial_agent")
        );
    }

    #[test]
    fn builder_rejects_nonexistent_initial_agent() {
        let provider = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "x", 1, 1,
        )]));
        let result = HandoffRunner::builder()
            .agent("a", make_agent(provider, "a"))
            .initial_agent("nonexistent")
            .max_handoffs(3)
            .build();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[test]
    fn builder_rejects_zero_max_handoffs() {
        let provider = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "x", 1, 1,
        )]));
        let result = HandoffRunner::builder()
            .agent("a", make_agent(provider, "a"))
            .initial_agent("a")
            .max_handoffs(0)
            .build();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("at least 1"));
    }

    #[test]
    fn builder_rejects_missing_max_handoffs() {
        let provider = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "x", 1, 1,
        )]));
        let result = HandoffRunner::builder()
            .agent("a", make_agent(provider, "a"))
            .initial_agent("a")
            .build();
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("requires max_handoffs")
        );
    }

    #[test]
    fn builder_accepts_valid_config() {
        let provider = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "done", 10, 5,
        )]));
        let result = HandoffRunner::builder()
            .agent("triage", make_agent(provider, "triage"))
            .initial_agent("triage")
            .max_handoffs(5)
            .build();
        assert!(result.is_ok());
    }

    // -----------------------------------------------------------------------
    // Execution tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn execute_no_handoff() {
        let provider = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "Direct answer.",
            100,
            50,
        )]));

        let runner = HandoffRunner::builder()
            .agent("triage", make_agent(provider, "triage"))
            .initial_agent("triage")
            .max_handoffs(5)
            .build()
            .unwrap();

        let output = runner.execute("simple question").await.unwrap();
        assert_eq!(output.result, "Direct answer.");
        assert_eq!(output.tokens_used.input_tokens, 100);
        assert_eq!(output.tokens_used.output_tokens, 50);
    }

    #[tokio::test]
    async fn execute_single_handoff() {
        // Triage agent hands off to billing, billing responds directly
        let triage_provider = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            &format!("{HANDOFF_SENTINEL}billing:summary:User has billing question"),
            50,
            20,
        )]));
        let billing_provider = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "Your bill is $42.",
            80,
            30,
        )]));

        let runner = HandoffRunner::builder()
            .agent("triage", make_agent(triage_provider, "triage"))
            .agent("billing", make_agent(billing_provider, "billing"))
            .initial_agent("triage")
            .max_handoffs(5)
            .build()
            .unwrap();

        let output = runner.execute("How much do I owe?").await.unwrap();
        assert_eq!(output.result, "Your bill is $42.");
        assert_eq!(output.tokens_used.input_tokens, 130);
        assert_eq!(output.tokens_used.output_tokens, 50);
    }

    #[tokio::test]
    async fn full_mode_accumulates_context_across_multi_hop_chain() {
        // AC3: A→B→C in Full mode. When B hands to C, C must see A's
        // contribution too, not just B's. We plant a distinctive marker in A's
        // handoff reason and assert it reaches C's task.
        let a = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            &format!("{HANDOFF_SENTINEL}b:full:ALICE_SECRET_FINDING"),
            10,
            5,
        )]));
        let b = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            &format!("{HANDOFF_SENTINEL}c:full:bob passes to c"),
            10,
            5,
        )]));
        let c = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "final answer from c",
            10,
            5,
        )]));

        let runner = HandoffRunner::builder()
            .agent("a", make_agent(a, "a"))
            .agent("b", make_agent(b, "b"))
            .agent("c", make_agent(c.clone(), "c"))
            .initial_agent("a")
            .max_handoffs(5)
            .build()
            .unwrap();

        let output = runner.execute("do the chain").await.unwrap();
        assert_eq!(output.result, "final answer from c");

        // C's request must carry A's marker (the whole chain), not just B's hop.
        let reqs = c.captured_requests.lock().unwrap();
        let c_sees: String = reqs[0]
            .messages
            .iter()
            .flat_map(|m| m.content.iter())
            .filter_map(|b| match b {
                crate::llm::types::ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(" ");
        assert!(
            c_sees.contains("ALICE_SECRET_FINDING"),
            "agent C did not receive agent A's intermediate context: {c_sees}"
        );
    }

    #[tokio::test]
    async fn execute_max_handoffs_exceeded() {
        // Agent A always hands off to B, B always hands off to A (ping-pong)
        let a_provider = Arc::new(MockProvider::new(vec![
            MockProvider::text_response(&format!("{HANDOFF_SENTINEL}b:summary:need b"), 10, 5),
            MockProvider::text_response(
                &format!("{HANDOFF_SENTINEL}b:summary:need b again"),
                10,
                5,
            ),
        ]));
        let b_provider = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            &format!("{HANDOFF_SENTINEL}a:summary:need a"),
            10,
            5,
        )]));

        let runner = HandoffRunner::builder()
            .agent("a", make_agent(a_provider, "a"))
            .agent("b", make_agent(b_provider, "b"))
            .initial_agent("a")
            .max_handoffs(2)
            .build()
            .unwrap();

        let output = runner.execute("ping pong").await.unwrap();
        assert!(output.result.contains("handoff limit reached"));
    }

    #[tokio::test]
    async fn execute_handoff_to_unknown_agent() {
        let provider = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            &format!("{HANDOFF_SENTINEL}nonexistent:summary:reason"),
            10,
            5,
        )]));

        let runner = HandoffRunner::builder()
            .agent("a", make_agent(provider, "a"))
            .initial_agent("a")
            .max_handoffs(3)
            .build()
            .unwrap();

        let result = runner.execute("test").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[tokio::test]
    async fn execute_full_context_mode() {
        let triage_provider = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            &format!("{HANDOFF_SENTINEL}support:full:Complex issue needs full context"),
            50,
            20,
        )]));
        let support_provider = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "I can see the full context. Fixed!",
            80,
            30,
        )]));

        let runner = HandoffRunner::builder()
            .agent("triage", make_agent(triage_provider, "triage"))
            .agent("support", make_agent(support_provider, "support"))
            .initial_agent("triage")
            .max_handoffs(5)
            .build()
            .unwrap();

        let output = runner.execute("Complex problem").await.unwrap();
        assert_eq!(output.result, "I can see the full context. Fixed!");
    }

    #[test]
    fn debug_impl() {
        let provider = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "x", 1, 1,
        )]));
        let runner = HandoffRunner::builder()
            .agent("a", make_agent(provider, "a"))
            .initial_agent("a")
            .max_handoffs(3)
            .build()
            .unwrap();

        let debug = format!("{runner:?}");
        assert!(debug.contains("HandoffRunner"));
        assert!(debug.contains("initial_agent"));
    }
}
