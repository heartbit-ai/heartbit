//! Mixture-of-Agents (MoA) workflow agent.
//!
//! Orchestrates ensemble reasoning with refinement:
//! 1. N proposer agents run in parallel on the same task
//! 2. Their outputs are collected and formatted as a combined proposal document
//! 3. A synthesizer agent refines the combined proposals into a final output
//! 4. Optionally repeats for multiple layers (rounds of refinement)

use std::sync::Arc;

use tokio::task::JoinSet;

use crate::error::Error;
use crate::llm::LlmProvider;
use crate::llm::types::TokenUsage;

use super::{AgentOutput, AgentRunner};

// ---------------------------------------------------------------------------
// MixtureOfAgentsAgent
// ---------------------------------------------------------------------------

/// Runs N proposer agents in parallel, collects their outputs, and feeds
/// a combined proposal document to a synthesizer agent for refinement.
/// Supports multiple layers where each layer's synthesis feeds the next
/// layer's proposers.
pub struct MixtureOfAgentsAgent<P: LlmProvider + 'static> {
    proposers: Vec<Arc<AgentRunner<P>>>,
    synthesizer: Arc<AgentRunner<P>>,
    layers: usize,
}

impl<P: LlmProvider + 'static> std::fmt::Debug for MixtureOfAgentsAgent<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MixtureOfAgentsAgent")
            .field("proposer_count", &self.proposers.len())
            .field("layers", &self.layers)
            .finish()
    }
}

/// Builder for [`MixtureOfAgentsAgent`].
pub struct MixtureOfAgentsAgentBuilder<P: LlmProvider + 'static> {
    proposers: Vec<Arc<AgentRunner<P>>>,
    synthesizer: Option<Arc<AgentRunner<P>>>,
    layers: Option<usize>,
}

impl<P: LlmProvider + 'static> MixtureOfAgentsAgent<P> {
    /// Create a new [`MixtureOfAgentsAgentBuilder`].
    pub fn builder() -> MixtureOfAgentsAgentBuilder<P> {
        MixtureOfAgentsAgentBuilder {
            proposers: Vec::new(),
            synthesizer: None,
            layers: None,
        }
    }

    /// Execute the mixture-of-agents pipeline.
    ///
    /// For each layer, all proposers run in parallel on the current input,
    /// their outputs are merged into a proposal document, and the synthesizer
    /// produces a refined output. Multi-layer mode feeds each synthesis back
    /// as input to the next layer's proposers.
    pub async fn execute(&self, task: &str) -> Result<AgentOutput, Error> {
        let mut current_input = task.to_string();
        let mut total_usage = TokenUsage::default();
        let mut total_tool_calls = 0usize;
        let mut total_cost: Option<f64> = None;
        let mut last_structured: Option<serde_json::Value> = None;
        let mut last_model_name: Option<String> = None;

        for _ in 0..self.layers {
            // --- Proposer phase: run all proposers in parallel ---
            let mut set = JoinSet::new();
            for proposer in &self.proposers {
                let proposer = Arc::clone(proposer);
                let input = current_input.clone();
                set.spawn(async move {
                    let name = proposer.name().to_string();
                    let result = proposer.execute(&input).await;
                    (name, result)
                });
            }

            let mut proposals: Vec<(String, AgentOutput)> =
                Vec::with_capacity(self.proposers.len());

            while let Some(join_result) = set.join_next().await {
                let (name, agent_result) = join_result
                    .map_err(|e| Error::Agent(format!("proposer task panicked: {e}")))?;
                let output = agent_result.map_err(|e| e.accumulate_usage(total_usage))?;
                output.accumulate_into(&mut total_usage, &mut total_tool_calls, &mut total_cost);
                proposals.push((name, output));
            }

            // Sort by proposer name for deterministic output
            proposals.sort_by(|a, b| a.0.cmp(&b.0));

            let proposal_text = proposals
                .iter()
                .map(|(name, output)| format!("## {name}\n{}", output.result))
                .collect::<Vec<_>>()
                .join("\n\n");

            // --- Synthesizer phase ---
            let synth_output = self
                .synthesizer
                .execute(&proposal_text)
                .await
                .map_err(|e| e.accumulate_usage(total_usage))?;

            synth_output.accumulate_into(&mut total_usage, &mut total_tool_calls, &mut total_cost);

            last_structured = synth_output.structured;
            last_model_name = synth_output.model_name;
            current_input = synth_output.result;
        }

        Ok(AgentOutput {
            result: current_input,
            tool_calls_made: total_tool_calls,
            tokens_used: total_usage,
            structured: last_structured,
            estimated_cost_usd: total_cost,
            model_name: last_model_name,
        })
    }
}

impl<P: LlmProvider + 'static> MixtureOfAgentsAgentBuilder<P> {
    /// Add a proposer agent. Wraps it in `Arc` for concurrent sharing.
    pub fn proposer(mut self, agent: AgentRunner<P>) -> Self {
        self.proposers.push(Arc::new(agent));
        self
    }

    /// Add multiple proposer agents.
    pub fn proposers(mut self, agents: Vec<AgentRunner<P>>) -> Self {
        self.proposers.extend(agents.into_iter().map(Arc::new));
        self
    }

    /// Set the synthesizer agent.
    pub fn synthesizer(mut self, agent: AgentRunner<P>) -> Self {
        self.synthesizer = Some(Arc::new(agent));
        self
    }

    /// Set the number of layers (rounds of proposal + synthesis). Defaults to 1.
    pub fn layers(mut self, n: usize) -> Self {
        self.layers = Some(n);
        self
    }

    /// Build the [`MixtureOfAgentsAgent`].
    pub fn build(self) -> Result<MixtureOfAgentsAgent<P>, Error> {
        if self.proposers.len() < 2 {
            return Err(Error::Config(
                "MixtureOfAgentsAgent requires at least 2 proposers".into(),
            ));
        }
        let synthesizer = self
            .synthesizer
            .ok_or_else(|| Error::Config("MixtureOfAgentsAgent requires a synthesizer".into()))?;
        let layers = self.layers.unwrap_or(1);
        if layers == 0 {
            return Err(Error::Config(
                "MixtureOfAgentsAgent layers must be at least 1".into(),
            ));
        }
        Ok(MixtureOfAgentsAgent {
            proposers: self.proposers,
            synthesizer,
            layers,
        })
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::test_helpers::{MockProvider, make_agent};

    // -----------------------------------------------------------------------
    // Builder validation tests
    // -----------------------------------------------------------------------

    #[test]
    fn builder_rejects_fewer_than_two_proposers() {
        let provider = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "x", 1, 1,
        )]));
        let synth = make_agent(Arc::clone(&provider), "synth");

        // Zero proposers
        let result = MixtureOfAgentsAgent::<MockProvider>::builder()
            .synthesizer(synth)
            .build();
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("at least 2 proposers")
        );

        // One proposer
        let synth2 = make_agent(Arc::clone(&provider), "synth2");
        let p1 = make_agent(provider, "p1");
        let result = MixtureOfAgentsAgent::builder()
            .proposer(p1)
            .synthesizer(synth2)
            .build();
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("at least 2 proposers")
        );
    }

    #[test]
    fn builder_rejects_missing_synthesizer() {
        let p1 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "x", 1, 1,
        )]));
        let p2 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "x", 1, 1,
        )]));

        let result = MixtureOfAgentsAgent::builder()
            .proposer(make_agent(p1, "a"))
            .proposer(make_agent(p2, "b"))
            .build();
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("requires a synthesizer")
        );
    }

    #[test]
    fn builder_rejects_zero_layers() {
        let p1 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "x", 1, 1,
        )]));
        let p2 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "x", 1, 1,
        )]));
        let synth = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "x", 1, 1,
        )]));

        let result = MixtureOfAgentsAgent::builder()
            .proposer(make_agent(p1, "a"))
            .proposer(make_agent(p2, "b"))
            .synthesizer(make_agent(synth, "synth"))
            .layers(0)
            .build();
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("layers must be at least 1")
        );
    }

    #[test]
    fn builder_accepts_valid_config_default_layers() {
        let p1 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "x", 1, 1,
        )]));
        let p2 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "x", 1, 1,
        )]));
        let synth = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "x", 1, 1,
        )]));

        let result = MixtureOfAgentsAgent::builder()
            .proposer(make_agent(p1, "a"))
            .proposer(make_agent(p2, "b"))
            .synthesizer(make_agent(synth, "synth"))
            .build();
        assert!(result.is_ok());
    }

    #[test]
    fn builder_accepts_valid_config_explicit_layers() {
        let p1 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "x", 1, 1,
        )]));
        let p2 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "x", 1, 1,
        )]));
        let synth = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "x", 1, 1,
        )]));

        let result = MixtureOfAgentsAgent::builder()
            .proposer(make_agent(p1, "a"))
            .proposer(make_agent(p2, "b"))
            .synthesizer(make_agent(synth, "synth"))
            .layers(3)
            .build();
        assert!(result.is_ok());
    }

    // -----------------------------------------------------------------------
    // Debug impl test
    // -----------------------------------------------------------------------

    #[test]
    fn debug_impl_shows_proposer_count_and_layers() {
        let p1 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "x", 1, 1,
        )]));
        let p2 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "x", 1, 1,
        )]));
        let p3 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "x", 1, 1,
        )]));
        let synth = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "x", 1, 1,
        )]));

        let moa = MixtureOfAgentsAgent::builder()
            .proposer(make_agent(p1, "a"))
            .proposer(make_agent(p2, "b"))
            .proposer(make_agent(p3, "c"))
            .synthesizer(make_agent(synth, "synth"))
            .layers(2)
            .build()
            .unwrap();

        let debug = format!("{moa:?}");
        assert!(debug.contains("MixtureOfAgentsAgent"));
        assert!(debug.contains("proposer_count: 3"));
        assert!(debug.contains("layers: 2"));
    }

    // -----------------------------------------------------------------------
    // Execution tests
    // -----------------------------------------------------------------------

    #[test]
    fn builder_proposers_bulk_method() {
        let p1 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "x", 1, 1,
        )]));
        let p2 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "x", 1, 1,
        )]));
        let synth = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "x", 1, 1,
        )]));
        let agents = vec![make_agent(p1, "a"), make_agent(p2, "b")];
        let result = MixtureOfAgentsAgent::builder()
            .proposers(agents)
            .synthesizer(make_agent(synth, "synth"))
            .build();
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn single_layer_execution() {
        // 2 proposers + 1 synthesizer
        let p1 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "proposal from alpha",
            100,
            50,
        )]));
        let p2 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "proposal from beta",
            120,
            60,
        )]));
        let synth = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "synthesized result",
            200,
            100,
        )]));

        let moa = MixtureOfAgentsAgent::builder()
            .proposer(make_agent(p1, "alpha"))
            .proposer(make_agent(p2, "beta"))
            .synthesizer(make_agent(synth, "synth"))
            .build()
            .unwrap();

        let output = moa.execute("analyze this").await.unwrap();
        assert_eq!(output.result, "synthesized result");
    }

    #[tokio::test]
    async fn token_usage_accumulated() {
        let p1 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "p1-out", 100, 50,
        )]));
        let p2 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "p2-out", 120, 60,
        )]));
        let synth = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "final", 200, 100,
        )]));

        let moa = MixtureOfAgentsAgent::builder()
            .proposer(make_agent(p1, "a"))
            .proposer(make_agent(p2, "b"))
            .synthesizer(make_agent(synth, "synth"))
            .build()
            .unwrap();

        let output = moa.execute("task").await.unwrap();
        // 100 + 120 + 200 = 420 input tokens
        assert_eq!(output.tokens_used.input_tokens, 420);
        // 50 + 60 + 100 = 210 output tokens
        assert_eq!(output.tokens_used.output_tokens, 210);
    }

    #[tokio::test]
    async fn multi_layer_execution() {
        // Layer 1: 2 proposers + synthesizer
        // Layer 2: same 2 proposers (re-run on synthesis) + synthesizer
        // Each proposer needs responses for both layers (2 responses each)
        let p1 = Arc::new(MockProvider::new(vec![
            MockProvider::text_response("p1-layer1", 10, 5),
            MockProvider::text_response("p1-layer2", 10, 5),
        ]));
        let p2 = Arc::new(MockProvider::new(vec![
            MockProvider::text_response("p2-layer1", 10, 5),
            MockProvider::text_response("p2-layer2", 10, 5),
        ]));
        let synth = Arc::new(MockProvider::new(vec![
            MockProvider::text_response("synth-layer1", 20, 10),
            MockProvider::text_response("synth-layer2-final", 20, 10),
        ]));

        let moa = MixtureOfAgentsAgent::builder()
            .proposer(make_agent(p1, "a"))
            .proposer(make_agent(p2, "b"))
            .synthesizer(make_agent(synth, "synth"))
            .layers(2)
            .build()
            .unwrap();

        let output = moa.execute("task").await.unwrap();
        assert_eq!(output.result, "synth-layer2-final");
        // 2 layers * (10 + 10 proposer + 20 synth) = 2 * 40 = 80 input
        assert_eq!(output.tokens_used.input_tokens, 80);
        // 2 layers * (5 + 5 proposer + 10 synth) = 2 * 20 = 40 output
        assert_eq!(output.tokens_used.output_tokens, 40);
    }

    #[tokio::test]
    async fn proposer_error_carries_partial_usage() {
        let p1 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "ok", 100, 50,
        )]));
        // p2 has no responses -> will error
        let p2 = Arc::new(MockProvider::new(vec![]));
        let synth = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "final", 10, 5,
        )]));

        let moa = MixtureOfAgentsAgent::builder()
            .proposer(make_agent(p1, "good"))
            .proposer(make_agent(p2, "bad"))
            .synthesizer(make_agent(synth, "synth"))
            .build()
            .unwrap();

        let err = moa.execute("task").await.unwrap_err();
        let partial = err.partial_usage();
        // JoinSet ordering is non-deterministic: the successful proposer may
        // or may not finish before the error is collected.
        assert!(
            partial.input_tokens == 0 || partial.input_tokens >= 100,
            "partial usage should be zero or include completed proposer"
        );
    }

    #[tokio::test]
    async fn synthesizer_error_carries_partial_usage_from_proposers() {
        let p1 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "ok1", 100, 50,
        )]));
        let p2 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "ok2", 120, 60,
        )]));
        // Synthesizer has no responses -> will error
        let synth = Arc::new(MockProvider::new(vec![]));

        let moa = MixtureOfAgentsAgent::builder()
            .proposer(make_agent(p1, "a"))
            .proposer(make_agent(p2, "b"))
            .synthesizer(make_agent(synth, "synth"))
            .build()
            .unwrap();

        let err = moa.execute("task").await.unwrap_err();
        let partial = err.partial_usage();
        // Both proposers succeeded: 100 + 120 = 220
        assert!(partial.input_tokens >= 220);
    }

    #[tokio::test]
    async fn synthesizer_receives_sorted_proposal_document() {
        let p1 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "output-c", 10, 5,
        )]));
        let p2 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "output-a", 10, 5,
        )]));
        let p3 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "output-b", 10, 5,
        )]));
        let synth_p = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "final-synthesis",
            10,
            5,
        )]));

        let moa = MixtureOfAgentsAgent::builder()
            .proposer(make_agent(Arc::clone(&p1), "charlie"))
            .proposer(make_agent(Arc::clone(&p2), "alpha"))
            .proposer(make_agent(Arc::clone(&p3), "beta"))
            .synthesizer(make_agent(Arc::clone(&synth_p), "synth"))
            .build()
            .unwrap();

        let output = moa.execute("task").await.unwrap();
        assert_eq!(output.result, "final-synthesis");

        // Inspect what the synthesizer received via captured_requests
        let synth_requests = synth_p.captured_requests.lock().unwrap();
        assert_eq!(synth_requests.len(), 1);
        let synth_input = &synth_requests[0].messages[0];
        let input_text = match &synth_input.content[0] {
            crate::llm::types::ContentBlock::Text { text } => text.as_str(),
            _ => panic!("expected text content"),
        };
        // Proposers should be sorted alphabetically: alpha, beta, charlie
        let alpha_pos = input_text
            .find("## alpha")
            .expect("should contain ## alpha");
        let beta_pos = input_text.find("## beta").expect("should contain ## beta");
        let charlie_pos = input_text
            .find("## charlie")
            .expect("should contain ## charlie");
        assert!(alpha_pos < beta_pos, "alpha should come before beta");
        assert!(beta_pos < charlie_pos, "beta should come before charlie");
        // Each proposer's output should appear
        assert!(input_text.contains("output-a"));
        assert!(input_text.contains("output-b"));
        assert!(input_text.contains("output-c"));
    }
}
