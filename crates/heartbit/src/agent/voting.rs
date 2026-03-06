//! Majority-voting workflow agent.
//!
//! Runs N voter agents in parallel on the same task, extracts a vote from each
//! output, and returns the full output of the first voter whose vote matches the
//! majority. Ties are resolved by an optional `tie_breaker` (defaults to first
//! alphabetically).

use std::collections::HashMap;
use std::sync::Arc;

use tokio::task::JoinSet;

use crate::error::Error;
use crate::llm::LlmProvider;
use crate::llm::types::TokenUsage;

use super::{AgentOutput, AgentRunner};

/// Extracts a vote string from an agent's output text.
type VoteExtractor = Box<dyn Fn(&str) -> String + Send + Sync>;

/// Resolves ties when multiple votes share the highest count.
/// Receives the tied vote strings and must return one of them.
type TieBreaker = Box<dyn Fn(&[String]) -> String + Send + Sync>;

/// The result of a voting round, including the winning vote, the full tally,
/// and the output from the first voter that cast the winning vote.
#[derive(Debug)]
pub struct VoteResult {
    /// The vote string that won.
    pub winner: String,
    /// Vote string → number of voters that cast it.
    pub tally: HashMap<String, usize>,
    /// The full `AgentOutput` from the first voter whose vote matched the winner.
    pub output: AgentOutput,
}

/// Orchestrates majority voting across N agents running in parallel.
pub struct VotingAgent<P: LlmProvider + 'static> {
    voters: Vec<Arc<AgentRunner<P>>>,
    vote_extractor: VoteExtractor,
    tie_breaker: TieBreaker,
}

impl<P: LlmProvider + 'static> std::fmt::Debug for VotingAgent<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VotingAgent")
            .field("voter_count", &self.voters.len())
            .finish()
    }
}

/// Builder for [`VotingAgent`].
pub struct VotingAgentBuilder<P: LlmProvider + 'static> {
    voters: Vec<Arc<AgentRunner<P>>>,
    vote_extractor: Option<VoteExtractor>,
    tie_breaker: Option<TieBreaker>,
}

impl<P: LlmProvider + 'static> VotingAgent<P> {
    pub fn builder() -> VotingAgentBuilder<P> {
        VotingAgentBuilder {
            voters: Vec::new(),
            vote_extractor: None,
            tie_breaker: None,
        }
    }

    /// Execute all voters in parallel, tally votes, and return the winning result.
    pub async fn execute(&self, task: &str) -> Result<VoteResult, Error> {
        let mut set = JoinSet::new();

        for (idx, voter) in self.voters.iter().enumerate() {
            let voter = Arc::clone(voter);
            let task = task.to_string();
            set.spawn(async move {
                let result = voter.execute(&task).await;
                (idx, result)
            });
        }

        // Collect results in completion order, tracking accumulated usage for
        // partial-usage-on-error semantics.
        let mut outputs: Vec<(usize, AgentOutput)> = Vec::with_capacity(self.voters.len());
        let mut total_usage = TokenUsage::default();

        while let Some(join_result) = set.join_next().await {
            let (idx, agent_result) = join_result
                .map_err(|e| Error::Agent(format!("voting agent task panicked: {e}")))?;
            let output = agent_result.map_err(|e| e.accumulate_usage(total_usage))?;
            total_usage += output.tokens_used;
            outputs.push((idx, output));
        }

        // Sort by original index for deterministic vote ordering.
        outputs.sort_by_key(|(idx, _)| *idx);

        // Extract votes and build tally.
        let votes: Vec<String> = outputs
            .iter()
            .map(|(_, output)| (self.vote_extractor)(&output.result))
            .collect();

        let mut tally: HashMap<String, usize> = HashMap::new();
        for vote in &votes {
            *tally.entry(vote.clone()).or_insert(0) += 1;
        }

        // Find the maximum vote count.
        let max_count = tally.values().copied().max().unwrap_or(0);

        // Collect all votes tied at the maximum.
        let mut top_votes: Vec<String> = tally
            .iter()
            .filter(|&(_, &count)| count == max_count)
            .map(|(vote, _)| vote.clone())
            .collect();
        top_votes.sort();

        let winner = if top_votes.len() == 1 {
            top_votes.into_iter().next().expect("at least one vote")
        } else {
            (self.tie_breaker)(&top_votes)
        };

        // Find the first voter (by original index) whose vote matches the winner.
        let winner_idx = votes
            .iter()
            .position(|v| *v == winner)
            .expect("winner must be among votes");

        let (_, mut winning_output) = outputs.remove(winner_idx);

        // Accumulate tool_calls and cost from all voters (usage already tracked
        // in `total_usage` during JoinSet collection above).
        let mut total_tool_calls = 0usize;
        let mut total_cost: Option<f64> = None;
        for (_, output) in &outputs {
            total_tool_calls += output.tool_calls_made;
            if let Some(cost) = output.estimated_cost_usd {
                *total_cost.get_or_insert(0.0) += cost;
            }
        }
        total_tool_calls += winning_output.tool_calls_made;
        if let Some(cost) = winning_output.estimated_cost_usd {
            *total_cost.get_or_insert(0.0) += cost;
        }

        winning_output.tokens_used = total_usage;
        winning_output.tool_calls_made = total_tool_calls;
        winning_output.estimated_cost_usd = total_cost;

        Ok(VoteResult {
            winner,
            tally,
            output: winning_output,
        })
    }
}

impl<P: LlmProvider + 'static> VotingAgentBuilder<P> {
    /// Add a voter agent. Wraps it in `Arc` for concurrent sharing.
    pub fn voter(mut self, agent: AgentRunner<P>) -> Self {
        self.voters.push(Arc::new(agent));
        self
    }

    /// Add multiple voter agents.
    pub fn voters(mut self, agents: Vec<AgentRunner<P>>) -> Self {
        self.voters.extend(agents.into_iter().map(Arc::new));
        self
    }

    /// Set the vote extractor function.
    pub fn vote_extractor(mut self, f: impl Fn(&str) -> String + Send + Sync + 'static) -> Self {
        self.vote_extractor = Some(Box::new(f));
        self
    }

    /// Set an optional tie-breaker function.
    pub fn tie_breaker(mut self, f: impl Fn(&[String]) -> String + Send + Sync + 'static) -> Self {
        self.tie_breaker = Some(Box::new(f));
        self
    }

    /// Build the [`VotingAgent`]. Requires at least 2 voters and a vote extractor.
    pub fn build(self) -> Result<VotingAgent<P>, Error> {
        if self.voters.len() < 2 {
            return Err(Error::Config(
                "VotingAgent requires at least 2 voters".into(),
            ));
        }
        let vote_extractor = self
            .vote_extractor
            .ok_or_else(|| Error::Config("VotingAgent requires a vote_extractor".into()))?;
        let tie_breaker = self.tie_breaker.unwrap_or_else(|| {
            Box::new(|votes: &[String]| {
                // Default: first alphabetically (votes are already sorted).
                votes[0].clone()
            })
        });
        Ok(VotingAgent {
            voters: self.voters,
            vote_extractor,
            tie_breaker,
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

    fn yes_no_extractor(output: &str) -> String {
        if output.contains("YES") {
            "YES".to_string()
        } else {
            "NO".to_string()
        }
    }

    // -----------------------------------------------------------------------
    // Builder validation tests
    // -----------------------------------------------------------------------

    #[test]
    fn builder_rejects_fewer_than_two_voters() {
        let provider = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "YES", 10, 5,
        )]));
        let result = VotingAgent::builder()
            .voter(make_agent(provider, "only-one"))
            .vote_extractor(yes_no_extractor)
            .build();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("at least 2"));
    }

    #[test]
    fn builder_rejects_zero_voters() {
        let result = VotingAgent::<MockProvider>::builder()
            .vote_extractor(yes_no_extractor)
            .build();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("at least 2"));
    }

    #[test]
    fn builder_rejects_missing_vote_extractor() {
        let p1 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "YES", 10, 5,
        )]));
        let p2 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "YES", 10, 5,
        )]));
        let result = VotingAgent::builder()
            .voter(make_agent(p1, "a"))
            .voter(make_agent(p2, "b"))
            .build();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("vote_extractor"));
    }

    #[test]
    fn builder_accepts_valid_config_without_tie_breaker() {
        let p1 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "YES", 10, 5,
        )]));
        let p2 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "NO", 10, 5,
        )]));
        let result = VotingAgent::builder()
            .voter(make_agent(p1, "a"))
            .voter(make_agent(p2, "b"))
            .vote_extractor(yes_no_extractor)
            .build();
        assert!(result.is_ok());
    }

    #[test]
    fn builder_accepts_valid_config_with_tie_breaker() {
        let p1 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "YES", 10, 5,
        )]));
        let p2 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "NO", 10, 5,
        )]));
        let result = VotingAgent::builder()
            .voter(make_agent(p1, "a"))
            .voter(make_agent(p2, "b"))
            .vote_extractor(yes_no_extractor)
            .tie_breaker(|votes| votes.last().unwrap().clone())
            .build();
        assert!(result.is_ok());
    }

    // -----------------------------------------------------------------------
    // Execution tests
    // -----------------------------------------------------------------------

    #[test]
    fn builder_voters_bulk_method() {
        let p1 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "YES", 10, 5,
        )]));
        let p2 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "NO", 10, 5,
        )]));
        let agents = vec![make_agent(p1, "a"), make_agent(p2, "b")];
        let result = VotingAgent::builder()
            .voters(agents)
            .vote_extractor(yes_no_extractor)
            .build();
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn unanimous_vote() {
        let p1 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "I vote YES",
            100,
            50,
        )]));
        let p2 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "Definitely YES",
            200,
            80,
        )]));
        let p3 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "YES please",
            150,
            60,
        )]));

        let voting = VotingAgent::builder()
            .voter(make_agent(p1, "v1"))
            .voter(make_agent(p2, "v2"))
            .voter(make_agent(p3, "v3"))
            .vote_extractor(yes_no_extractor)
            .build()
            .unwrap();

        let result = voting.execute("should we?").await.unwrap();
        assert_eq!(result.winner, "YES");
        assert_eq!(result.tally["YES"], 3);
        assert!(!result.tally.contains_key("NO"));
        // Output should be from one of the YES voters
        assert!(result.output.result.contains("YES"));
    }

    #[tokio::test]
    async fn majority_vote_two_of_three() {
        let p1 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "I say YES",
            100,
            50,
        )]));
        let p2 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "NO way", 200, 80,
        )]));
        let p3 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "YES definitely",
            150,
            60,
        )]));

        let voting = VotingAgent::builder()
            .voter(make_agent(p1, "v1"))
            .voter(make_agent(p2, "v2"))
            .voter(make_agent(p3, "v3"))
            .vote_extractor(yes_no_extractor)
            .build()
            .unwrap();

        let result = voting.execute("proceed?").await.unwrap();
        assert_eq!(result.winner, "YES");
        assert_eq!(result.tally["YES"], 2);
        assert_eq!(result.tally["NO"], 1);
    }

    #[tokio::test]
    async fn tie_broken_by_default_alphabetical() {
        // 2 voters: one YES, one NO — tie. Default tie-breaker picks alphabetically first.
        let p1 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "NO thanks",
            100,
            50,
        )]));
        let p2 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "YES sure", 200, 80,
        )]));

        let voting = VotingAgent::builder()
            .voter(make_agent(p1, "v1"))
            .voter(make_agent(p2, "v2"))
            .vote_extractor(yes_no_extractor)
            .build()
            .unwrap();

        let result = voting.execute("tie?").await.unwrap();
        // "NO" < "YES" alphabetically
        assert_eq!(result.winner, "NO");
        assert_eq!(result.tally["YES"], 1);
        assert_eq!(result.tally["NO"], 1);
    }

    #[tokio::test]
    async fn tie_broken_by_custom_tie_breaker() {
        let p1 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "NO thanks",
            100,
            50,
        )]));
        let p2 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "YES sure", 200, 80,
        )]));

        let voting = VotingAgent::builder()
            .voter(make_agent(p1, "v1"))
            .voter(make_agent(p2, "v2"))
            .vote_extractor(yes_no_extractor)
            .tie_breaker(|votes| votes.last().unwrap().clone()) // pick last alphabetically
            .build()
            .unwrap();

        let result = voting.execute("tie?").await.unwrap();
        // Custom tie-breaker picks last alphabetically: "YES"
        assert_eq!(result.winner, "YES");
    }

    #[tokio::test]
    async fn token_usage_accumulated_across_all_voters() {
        let p1 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "YES", 100, 50,
        )]));
        let p2 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "YES", 200, 80,
        )]));
        let p3 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "YES", 150, 60,
        )]));

        let voting = VotingAgent::builder()
            .voter(make_agent(p1, "v1"))
            .voter(make_agent(p2, "v2"))
            .voter(make_agent(p3, "v3"))
            .vote_extractor(yes_no_extractor)
            .build()
            .unwrap();

        let result = voting.execute("go").await.unwrap();
        assert_eq!(result.output.tokens_used.input_tokens, 450);
        assert_eq!(result.output.tokens_used.output_tokens, 190);
    }

    #[tokio::test]
    async fn error_carries_partial_usage() {
        let p1 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "YES", 100, 50,
        )]));
        // Second provider has no responses -> error
        let p2 = Arc::new(MockProvider::new(vec![]));

        let voting = VotingAgent::builder()
            .voter(make_agent(p1, "good"))
            .voter(make_agent(p2, "bad"))
            .vote_extractor(yes_no_extractor)
            .build()
            .unwrap();

        let err = voting.execute("task").await.unwrap_err();
        let partial = err.partial_usage();
        // JoinSet ordering is non-deterministic: the successful voter may
        // or may not finish before the error is collected.
        assert!(
            partial.input_tokens == 0 || partial.input_tokens >= 100,
            "partial usage should be zero or include completed voter"
        );
    }

    #[test]
    fn debug_impl() {
        let p1 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "YES", 10, 5,
        )]));
        let p2 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "NO", 10, 5,
        )]));

        let voting = VotingAgent::builder()
            .voter(make_agent(p1, "a"))
            .voter(make_agent(p2, "b"))
            .vote_extractor(yes_no_extractor)
            .build()
            .unwrap();

        let debug = format!("{voting:?}");
        assert!(debug.contains("VotingAgent"));
        assert!(debug.contains("voter_count"));
        assert!(debug.contains("2"));
    }

    #[tokio::test]
    async fn vote_result_contains_correct_tally() {
        let p1 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "YES agree",
            10,
            5,
        )]));
        let p2 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "NO disagree",
            10,
            5,
        )]));
        let p3 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "YES concur",
            10,
            5,
        )]));
        let p4 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "NO object",
            10,
            5,
        )]));
        let p5 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "YES absolutely",
            10,
            5,
        )]));

        let voting = VotingAgent::builder()
            .voter(make_agent(p1, "v1"))
            .voter(make_agent(p2, "v2"))
            .voter(make_agent(p3, "v3"))
            .voter(make_agent(p4, "v4"))
            .voter(make_agent(p5, "v5"))
            .vote_extractor(yes_no_extractor)
            .build()
            .unwrap();

        let result = voting.execute("vote").await.unwrap();
        assert_eq!(result.winner, "YES");
        assert_eq!(result.tally.len(), 2);
        assert_eq!(result.tally["YES"], 3);
        assert_eq!(result.tally["NO"], 2);
    }
}
