//! Verified best-of-N agent — wires the [`Verifier`] seam into a runnable agent.
//!
//! Runs N candidate agents on the same task (in parallel), then selects the
//! single best candidate output via a [`Verifier`] reward signal
//! ([`select_best`]). Where [`VotingAgent`](super::VotingAgent) picks by majority
//! vote, this picks by *graded* relevance/correctness — the test-time-compute
//! scaling pattern, made concrete so the verifier is a live capability rather
//! than a loose helper.

use std::sync::Arc;

use tokio::task::JoinSet;

use super::verifier::{Verifier, select_best};
use super::{AgentOutput, AgentRunner};
use crate::error::Error;
use crate::llm::LlmProvider;
use crate::llm::types::TokenUsage;

/// The outcome of a verified best-of-N run.
#[derive(Debug, Clone)]
pub struct VerifiedResult {
    /// The winning candidate's full output (with usage/cost accumulated across
    /// ALL candidates, since all were billed).
    pub output: AgentOutput,
    /// Index of the winning candidate.
    pub winner_index: usize,
    /// The winner's verifier score, in `[0, 1]`.
    pub score: f64,
}

/// Runs N candidate agents and returns the verifier-best output.
pub struct VerifiedAgent<P: LlmProvider + 'static> {
    candidates: Vec<Arc<AgentRunner<P>>>,
    verifier: Arc<dyn Verifier>,
}

impl<P: LlmProvider + 'static> std::fmt::Debug for VerifiedAgent<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VerifiedAgent")
            .field("candidate_count", &self.candidates.len())
            .finish()
    }
}

/// Builder for [`VerifiedAgent`].
pub struct VerifiedAgentBuilder<P: LlmProvider + 'static> {
    candidates: Vec<Arc<AgentRunner<P>>>,
    verifier: Option<Arc<dyn Verifier>>,
}

impl<P: LlmProvider + 'static> VerifiedAgent<P> {
    /// Start a builder.
    pub fn builder() -> VerifiedAgentBuilder<P> {
        VerifiedAgentBuilder {
            candidates: Vec::new(),
            verifier: None,
        }
    }

    /// Run all candidates in parallel on `task`, score each with the verifier, and
    /// return the best. Usage/cost are accumulated across all candidates (all are
    /// billed); the returned `output` is the winner's, re-stamped with the totals.
    pub async fn execute(&self, task: &str) -> Result<VerifiedResult, Error> {
        let mut set = JoinSet::new();
        for (idx, candidate) in self.candidates.iter().enumerate() {
            let candidate = Arc::clone(candidate);
            let task = task.to_string();
            set.spawn(async move {
                let result = candidate.execute(&task).await;
                (idx, result)
            });
        }

        let mut outputs: Vec<(usize, AgentOutput)> = Vec::with_capacity(self.candidates.len());
        let mut total_usage = TokenUsage::default();
        while let Some(join_result) = set.join_next().await {
            let (idx, agent_result) = join_result.map_err(|e| {
                Error::Agent(format!("verified agent task panicked: {e}"))
                    .accumulate_usage(total_usage)
            })?;
            let output = agent_result.map_err(|e| e.accumulate_usage(total_usage))?;
            total_usage += output.tokens_used;
            outputs.push((idx, output));
        }
        outputs.sort_by_key(|(idx, _)| *idx);

        // Verifier-best-of-N over the candidate result strings.
        let results: Vec<String> = outputs.iter().map(|(_, o)| o.result.clone()).collect();
        let selected = select_best(task, &results, self.verifier.as_ref()).await?;

        // Accumulate tool calls + cost across all candidates, then return the
        // winner's output stamped with the totals.
        let mut total_tool_calls = 0usize;
        let mut total_cost: Option<f64> = None;
        for (_, o) in &outputs {
            total_tool_calls += o.tool_calls_made;
            if let Some(c) = o.estimated_cost_usd {
                *total_cost.get_or_insert(0.0) += c;
            }
        }

        let winner_index = selected.index;
        let (_, mut winning_output) = outputs.remove(winner_index);
        winning_output.tokens_used = total_usage;
        winning_output.tool_calls_made = total_tool_calls;
        winning_output.estimated_cost_usd = total_cost;

        Ok(VerifiedResult {
            output: winning_output,
            winner_index,
            score: selected.score,
        })
    }
}

impl<P: LlmProvider + 'static> VerifiedAgentBuilder<P> {
    /// Add a candidate agent.
    pub fn candidate(mut self, agent: AgentRunner<P>) -> Self {
        self.candidates.push(Arc::new(agent));
        self
    }

    /// Add multiple candidate agents.
    pub fn candidates(mut self, agents: Vec<AgentRunner<P>>) -> Self {
        self.candidates.extend(agents.into_iter().map(Arc::new));
        self
    }

    /// Set the verifier used to select the best candidate.
    pub fn verifier(mut self, verifier: Arc<dyn Verifier>) -> Self {
        self.verifier = Some(verifier);
        self
    }

    /// Build, validating at least one candidate and a verifier are set.
    pub fn build(self) -> Result<VerifiedAgent<P>, Error> {
        if self.candidates.is_empty() {
            return Err(Error::Agent(
                "VerifiedAgent requires at least one candidate".into(),
            ));
        }
        let verifier = self
            .verifier
            .ok_or_else(|| Error::Agent("VerifiedAgent requires a verifier".into()))?;
        Ok(VerifiedAgent {
            candidates: self.candidates,
            verifier,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::test_helpers::{MockProvider, make_agent};
    use crate::agent::verifier::Verifier;
    use std::future::Future;
    use std::pin::Pin;

    /// Verifier that scores a candidate by whether it contains "GOOD".
    struct KeywordVerifier;
    impl Verifier for KeywordVerifier {
        fn score<'a>(
            &'a self,
            _task: &'a str,
            candidate: &'a str,
        ) -> Pin<Box<dyn Future<Output = Result<f64, Error>> + Send + 'a>> {
            let s = if candidate.contains("GOOD") { 1.0 } else { 0.1 };
            Box::pin(async move { Ok(s) })
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn picks_the_verifier_best_candidate() {
        // Candidate 0 → "meh answer", candidate 1 → "a GOOD answer".
        let p0 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "meh answer",
            5,
            5,
        )]));
        let p1 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "a GOOD answer",
            5,
            5,
        )]));
        let agent = VerifiedAgent::builder()
            .candidate(make_agent(p0, "c0"))
            .candidate(make_agent(p1, "c1"))
            .verifier(Arc::new(KeywordVerifier))
            .build()
            .unwrap();

        let result = agent.execute("the task").await.unwrap();
        assert_eq!(result.winner_index, 1);
        assert_eq!(result.output.result, "a GOOD answer");
        assert!((result.score - 1.0).abs() < 1e-9);
        // Usage accumulated across BOTH candidates (both billed).
        assert_eq!(result.output.tokens_used.input_tokens, 10);
    }

    #[test]
    fn builder_requires_candidate_and_verifier() {
        let provider = Arc::new(MockProvider::new(vec![]));
        // No verifier.
        let r = VerifiedAgent::builder()
            .candidate(make_agent(provider, "c"))
            .build();
        assert!(r.is_err());
        // No candidate.
        let r2 = VerifiedAgent::<MockProvider>::builder()
            .verifier(Arc::new(KeywordVerifier))
            .build();
        assert!(r2.is_err());
    }
}
