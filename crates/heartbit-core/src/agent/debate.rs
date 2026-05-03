//! Debate workflow agent.
//!
//! Orchestrates multi-round debates between N debater agents with a judge
//! agent that synthesizes the final answer from the complete transcript.
//! Debaters run in parallel each round via `tokio::JoinSet`.

use std::sync::Arc;

use tokio::task::JoinSet;

use crate::error::Error;
use crate::llm::LlmProvider;
use crate::llm::types::TokenUsage;

use super::{AgentOutput, AgentRunner};

/// Optional early-stop predicate. Receives the full transcript so far;
/// returns `true` to end the debate before `max_rounds`.
type StopCondition = Box<dyn Fn(&str) -> bool + Send + Sync>;

// ---------------------------------------------------------------------------
// DebateAgent
// ---------------------------------------------------------------------------

/// Orchestrates a multi-round debate between N debater agents, then asks a
/// judge agent to synthesize the final answer from the complete transcript.
///
/// Each round, all debaters run in parallel and receive the full debate
/// history as input. After all rounds (or early stop), the judge produces
/// the final output.
///
/// **Note on transcript growth**: The full transcript is passed to every
/// debater each round, so input tokens grow quadratically with round count
/// and debater count. Keep `max_rounds` small (2–4) or use `should_stop`
/// for early termination.
pub struct DebateAgent<P: LlmProvider + 'static> {
    debaters: Vec<Arc<AgentRunner<P>>>,
    judge: Arc<AgentRunner<P>>,
    max_rounds: usize,
    should_stop: Option<StopCondition>,
}

impl<P: LlmProvider + 'static> std::fmt::Debug for DebateAgent<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DebateAgent")
            .field("debater_count", &self.debaters.len())
            .field("max_rounds", &self.max_rounds)
            .finish()
    }
}

/// Builder for [`DebateAgent`].
pub struct DebateAgentBuilder<P: LlmProvider + 'static> {
    debaters: Vec<AgentRunner<P>>,
    judge: Option<AgentRunner<P>>,
    max_rounds: Option<usize>,
    should_stop: Option<StopCondition>,
}

impl<P: LlmProvider + 'static> DebateAgent<P> {
    /// Create a new [`DebateAgentBuilder`] for constructing a debate agent.
    pub fn builder() -> DebateAgentBuilder<P> {
        DebateAgentBuilder {
            debaters: Vec::new(),
            judge: None,
            max_rounds: None,
            should_stop: None,
        }
    }

    /// Execute the debate.
    ///
    /// Each round, all debaters receive the full transcript and run in
    /// parallel. After `max_rounds` (or early stop), the judge synthesizes
    /// the final answer.
    pub async fn execute(&self, task: &str) -> Result<AgentOutput, Error> {
        let mut total_usage = TokenUsage::default();
        let mut total_tool_calls = 0usize;
        let mut total_cost: Option<f64> = None;

        let mut transcript = format!("# Debate Topic\n{task}\n");

        for round in 1..=self.max_rounds {
            transcript.push_str(&format!("\n### Round {round}\n"));

            // Run all debaters in parallel
            let mut set = JoinSet::new();
            for debater in &self.debaters {
                let debater = Arc::clone(debater);
                let input = transcript.clone();
                set.spawn(async move {
                    let name = debater.name().to_string();
                    let result = debater.execute(&input).await;
                    (name, result)
                });
            }

            let mut round_results: Vec<(String, AgentOutput)> =
                Vec::with_capacity(self.debaters.len());

            while let Some(join_result) = set.join_next().await {
                let (name, agent_result) = join_result
                    .map_err(|e| Error::Agent(format!("debate agent task panicked: {e}")))?;
                let output = agent_result.map_err(|e| e.accumulate_usage(total_usage))?;
                output.accumulate_into(&mut total_usage, &mut total_tool_calls, &mut total_cost);
                round_results.push((name, output));
            }

            // Sort by name for deterministic transcript ordering
            round_results.sort_by(|a, b| a.0.cmp(&b.0));

            for (name, output) in &round_results {
                transcript.push_str(&format!("\n#### {name}\n{}\n", output.result));
            }

            // Check early stop
            if self.should_stop.as_ref().is_some_and(|f| f(&transcript)) {
                break;
            }
        }

        // Judge synthesizes the final answer
        let judge_output = self
            .judge
            .execute(&transcript)
            .await
            .map_err(|e| e.accumulate_usage(total_usage))?;
        judge_output.accumulate_into(&mut total_usage, &mut total_tool_calls, &mut total_cost);

        Ok(AgentOutput {
            result: judge_output.result,
            tool_calls_made: total_tool_calls,
            tokens_used: total_usage,
            structured: judge_output.structured,
            estimated_cost_usd: total_cost,
            model_name: judge_output.model_name,
        })
    }
}

impl<P: LlmProvider + 'static> DebateAgentBuilder<P> {
    /// Add a debater agent.
    pub fn debater(mut self, agent: AgentRunner<P>) -> Self {
        self.debaters.push(agent);
        self
    }

    /// Add multiple debater agents.
    pub fn debaters(mut self, agents: Vec<AgentRunner<P>>) -> Self {
        self.debaters.extend(agents);
        self
    }

    /// Set the judge agent that synthesizes the final answer.
    pub fn judge(mut self, agent: AgentRunner<P>) -> Self {
        self.judge = Some(agent);
        self
    }

    /// Set the maximum number of debate rounds (must be >= 1).
    pub fn max_rounds(mut self, n: usize) -> Self {
        self.max_rounds = Some(n);
        self
    }

    /// Set an optional early-stop predicate. The closure receives the full
    /// transcript so far and returns `true` to end the debate early.
    pub fn should_stop(mut self, f: impl Fn(&str) -> bool + Send + Sync + 'static) -> Self {
        self.should_stop = Some(Box::new(f));
        self
    }

    /// Build the [`DebateAgent`].
    pub fn build(self) -> Result<DebateAgent<P>, Error> {
        if self.debaters.len() < 2 {
            return Err(Error::Config(
                "DebateAgent requires at least 2 debaters".into(),
            ));
        }
        let judge = self
            .judge
            .ok_or_else(|| Error::Config("DebateAgent requires a judge".into()))?;
        let max_rounds = self
            .max_rounds
            .ok_or_else(|| Error::Config("DebateAgent requires max_rounds".into()))?;
        if max_rounds == 0 {
            return Err(Error::Config(
                "DebateAgent max_rounds must be at least 1".into(),
            ));
        }
        Ok(DebateAgent {
            debaters: self.debaters.into_iter().map(Arc::new).collect(),
            judge: Arc::new(judge),
            max_rounds,
            should_stop: self.should_stop,
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
    fn builder_rejects_fewer_than_two_debaters() {
        let p = Arc::new(MockProvider::new(vec![]));
        let judge_p = Arc::new(MockProvider::new(vec![]));
        let result = DebateAgent::builder()
            .debater(make_agent(p, "only-one"))
            .judge(make_agent(judge_p, "judge"))
            .max_rounds(3)
            .build();
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("at least 2 debaters")
        );
    }

    #[test]
    fn builder_rejects_zero_debaters() {
        let judge_p = Arc::new(MockProvider::new(vec![]));
        let result = DebateAgent::<MockProvider>::builder()
            .judge(make_agent(judge_p, "judge"))
            .max_rounds(3)
            .build();
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("at least 2 debaters")
        );
    }

    #[test]
    fn builder_rejects_missing_judge() {
        let p1 = Arc::new(MockProvider::new(vec![]));
        let p2 = Arc::new(MockProvider::new(vec![]));
        let result = DebateAgent::builder()
            .debater(make_agent(p1, "d1"))
            .debater(make_agent(p2, "d2"))
            .max_rounds(3)
            .build();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("requires a judge"));
    }

    #[test]
    fn builder_rejects_missing_max_rounds() {
        let p1 = Arc::new(MockProvider::new(vec![]));
        let p2 = Arc::new(MockProvider::new(vec![]));
        let judge_p = Arc::new(MockProvider::new(vec![]));
        let result = DebateAgent::builder()
            .debater(make_agent(p1, "d1"))
            .debater(make_agent(p2, "d2"))
            .judge(make_agent(judge_p, "judge"))
            .build();
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("requires max_rounds")
        );
    }

    #[test]
    fn builder_rejects_zero_max_rounds() {
        let p1 = Arc::new(MockProvider::new(vec![]));
        let p2 = Arc::new(MockProvider::new(vec![]));
        let judge_p = Arc::new(MockProvider::new(vec![]));
        let result = DebateAgent::builder()
            .debater(make_agent(p1, "d1"))
            .debater(make_agent(p2, "d2"))
            .judge(make_agent(judge_p, "judge"))
            .max_rounds(0)
            .build();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("at least 1"));
    }

    #[test]
    fn builder_accepts_valid_config_without_should_stop() {
        let p1 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "a", 1, 1,
        )]));
        let p2 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "b", 1, 1,
        )]));
        let judge_p = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "j", 1, 1,
        )]));
        let result = DebateAgent::builder()
            .debater(make_agent(p1, "d1"))
            .debater(make_agent(p2, "d2"))
            .judge(make_agent(judge_p, "judge"))
            .max_rounds(3)
            .build();
        assert!(result.is_ok());
    }

    #[test]
    fn builder_accepts_valid_config_with_should_stop() {
        let p1 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "a", 1, 1,
        )]));
        let p2 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "b", 1, 1,
        )]));
        let judge_p = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "j", 1, 1,
        )]));
        let result = DebateAgent::builder()
            .debater(make_agent(p1, "d1"))
            .debater(make_agent(p2, "d2"))
            .judge(make_agent(judge_p, "judge"))
            .max_rounds(3)
            .should_stop(|t| t.contains("CONSENSUS"))
            .build();
        assert!(result.is_ok());
    }

    // -----------------------------------------------------------------------
    // Execution tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn single_round_debate() {
        // 2 debaters + 1 judge, 1 round
        let p1 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "I argue for A",
            100,
            50,
        )]));
        let p2 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "I argue for B",
            200,
            80,
        )]));
        let judge_p = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "After deliberation, A wins",
            150,
            70,
        )]));

        let debate = DebateAgent::builder()
            .debater(make_agent(p1, "debater-a"))
            .debater(make_agent(p2, "debater-b"))
            .judge(make_agent(judge_p, "judge"))
            .max_rounds(1)
            .build()
            .unwrap();

        let output = debate.execute("Which is better?").await.unwrap();
        assert_eq!(output.result, "After deliberation, A wins");
        // 100+200+150 = 450 input, 50+80+70 = 200 output
        assert_eq!(output.tokens_used.input_tokens, 450);
        assert_eq!(output.tokens_used.output_tokens, 200);
    }

    #[tokio::test]
    async fn multi_round_accumulates_usage() {
        // 2 debaters, 2 rounds, then judge
        // Each debater needs 2 responses (one per round)
        let p1 = Arc::new(MockProvider::new(vec![
            MockProvider::text_response("round1-d1", 10, 5),
            MockProvider::text_response("round2-d1", 10, 5),
        ]));
        let p2 = Arc::new(MockProvider::new(vec![
            MockProvider::text_response("round1-d2", 20, 10),
            MockProvider::text_response("round2-d2", 20, 10),
        ]));
        let judge_p = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "final verdict",
            30,
            15,
        )]));

        let debate = DebateAgent::builder()
            .debater(make_agent(p1, "d1"))
            .debater(make_agent(p2, "d2"))
            .judge(make_agent(judge_p, "judge"))
            .max_rounds(2)
            .build()
            .unwrap();

        let output = debate.execute("topic").await.unwrap();
        assert_eq!(output.result, "final verdict");
        // (10+20)*2 + 30 = 90 input, (5+10)*2 + 15 = 45 output
        assert_eq!(output.tokens_used.input_tokens, 90);
        assert_eq!(output.tokens_used.output_tokens, 45);
    }

    #[tokio::test]
    async fn early_stop_via_should_stop() {
        // 2 debaters, max 5 rounds, but stop early when transcript contains "CONSENSUS"
        // Round 1: normal responses
        // Round 2: debater-a says "CONSENSUS reached" -> should stop
        let p1 = Arc::new(MockProvider::new(vec![
            MockProvider::text_response("I disagree", 10, 5),
            MockProvider::text_response("CONSENSUS reached", 10, 5),
        ]));
        let p2 = Arc::new(MockProvider::new(vec![
            MockProvider::text_response("I also disagree", 10, 5),
            MockProvider::text_response("I concur", 10, 5),
        ]));
        let judge_p = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "judge says done",
            10,
            5,
        )]));

        let debate = DebateAgent::builder()
            .debater(make_agent(p1, "debater-a"))
            .debater(make_agent(p2, "debater-b"))
            .judge(make_agent(judge_p, "judge"))
            .max_rounds(5)
            .should_stop(|transcript| transcript.contains("CONSENSUS"))
            .build()
            .unwrap();

        let output = debate.execute("topic").await.unwrap();
        assert_eq!(output.result, "judge says done");
        // 2 rounds * 2 debaters * 10 + judge 10 = 50 input
        assert_eq!(output.tokens_used.input_tokens, 50);
        // 2 rounds * 2 debaters * 5 + judge 5 = 25 output
        assert_eq!(output.tokens_used.output_tokens, 25);
    }

    #[tokio::test]
    async fn error_carries_partial_usage() {
        // Debater 1 succeeds, debater 2 errors in round 1
        let p1 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "ok", 100, 50,
        )]));
        let p2 = Arc::new(MockProvider::new(vec![])); // will error
        let judge_p = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "judge", 10, 5,
        )]));

        let debate = DebateAgent::builder()
            .debater(make_agent(p1, "good"))
            .debater(make_agent(p2, "bad"))
            .judge(make_agent(judge_p, "judge"))
            .max_rounds(1)
            .build()
            .unwrap();

        let err = debate.execute("topic").await.unwrap_err();
        let partial = err.partial_usage();
        // At minimum, partial usage is non-negative; exact value depends on
        // JoinSet completion order (the successful debater may or may not
        // finish before the error is collected).
        assert!(
            partial.input_tokens == 0 || partial.input_tokens >= 100,
            "partial usage should be zero or include completed debater"
        );
    }

    #[tokio::test]
    async fn judge_error_carries_debater_usage() {
        // Both debaters succeed, judge errors
        let p1 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "arg1", 100, 50,
        )]));
        let p2 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "arg2", 200, 80,
        )]));
        let judge_p = Arc::new(MockProvider::new(vec![])); // will error

        let debate = DebateAgent::builder()
            .debater(make_agent(p1, "d1"))
            .debater(make_agent(p2, "d2"))
            .judge(make_agent(judge_p, "judge"))
            .max_rounds(1)
            .build()
            .unwrap();

        let err = debate.execute("topic").await.unwrap_err();
        let partial = err.partial_usage();
        // Both debaters' usage should be in partial
        assert!(partial.input_tokens >= 300);
    }

    // -----------------------------------------------------------------------
    // Debug and transcript tests
    // -----------------------------------------------------------------------

    #[test]
    fn debug_impl() {
        let p1 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "a", 1, 1,
        )]));
        let p2 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "b", 1, 1,
        )]));
        let judge_p = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "j", 1, 1,
        )]));
        let debate = DebateAgent::builder()
            .debater(make_agent(p1, "d1"))
            .debater(make_agent(p2, "d2"))
            .judge(make_agent(judge_p, "judge"))
            .max_rounds(3)
            .build()
            .unwrap();

        let debug = format!("{debate:?}");
        assert!(debug.contains("DebateAgent"));
        assert!(debug.contains("debater_count"));
        assert!(debug.contains("2"));
        assert!(debug.contains("max_rounds"));
        assert!(debug.contains("3"));
    }

    #[tokio::test]
    async fn judge_receives_transcript_with_round_headers_and_names() {
        // Use captured_requests to inspect what the judge actually received.
        let p1 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "position-alpha",
            10,
            5,
        )]));
        let p2 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "position-beta",
            10,
            5,
        )]));
        let judge_p = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "verdict", 10, 5,
        )]));

        let debate = DebateAgent::builder()
            .debater(make_agent(Arc::clone(&p1), "alpha"))
            .debater(make_agent(Arc::clone(&p2), "beta"))
            .judge(make_agent(Arc::clone(&judge_p), "judge"))
            .max_rounds(1)
            .build()
            .unwrap();

        let output = debate.execute("test topic").await.unwrap();
        assert_eq!(output.result, "verdict");

        // Inspect what the judge received via captured_requests
        let judge_requests = judge_p.captured_requests.lock().unwrap();
        assert_eq!(judge_requests.len(), 1);
        let judge_input = &judge_requests[0].messages[0];
        let input_text = match &judge_input.content[0] {
            crate::llm::types::ContentBlock::Text { text } => text.as_str(),
            _ => panic!("expected text content"),
        };
        assert!(
            input_text.contains("# Debate Topic"),
            "should have topic header"
        );
        assert!(
            input_text.contains("test topic"),
            "should have original topic"
        );
        assert!(
            input_text.contains("### Round 1"),
            "should have round header"
        );
        assert!(
            input_text.contains("#### alpha"),
            "should have debater name alpha"
        );
        assert!(
            input_text.contains("#### beta"),
            "should have debater name beta"
        );
        assert!(
            input_text.contains("position-alpha"),
            "should have alpha's argument"
        );
        assert!(
            input_text.contains("position-beta"),
            "should have beta's argument"
        );
    }

    #[test]
    fn builder_debaters_bulk_method() {
        let p1 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "a", 1, 1,
        )]));
        let p2 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "b", 1, 1,
        )]));
        let judge_p = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "j", 1, 1,
        )]));
        let agents = vec![make_agent(p1, "d1"), make_agent(p2, "d2")];
        let result = DebateAgent::builder()
            .debaters(agents)
            .judge(make_agent(judge_p, "judge"))
            .max_rounds(1)
            .build();
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn three_debaters_single_round() {
        let p1 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "arg-1", 10, 5,
        )]));
        let p2 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "arg-2", 20, 10,
        )]));
        let p3 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "arg-3", 30, 15,
        )]));
        let judge_p = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "three-way verdict",
            40,
            20,
        )]));

        let debate = DebateAgent::builder()
            .debater(make_agent(p1, "d1"))
            .debater(make_agent(p2, "d2"))
            .debater(make_agent(p3, "d3"))
            .judge(make_agent(judge_p, "judge"))
            .max_rounds(1)
            .build()
            .unwrap();

        let output = debate.execute("topic").await.unwrap();
        assert_eq!(output.result, "three-way verdict");
        // 10+20+30+40 = 100 input, 5+10+15+20 = 50 output
        assert_eq!(output.tokens_used.input_tokens, 100);
        assert_eq!(output.tokens_used.output_tokens, 50);
    }
}
