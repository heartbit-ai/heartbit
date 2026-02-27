//! Deterministic workflow agent primitives.
//!
//! These composable agents orchestrate sub-agents without LLM calls:
//! - [`SequentialAgent`]: runs agents in order, piping output as input
//! - [`ParallelAgent`]: runs agents concurrently via `tokio::JoinSet`
//! - [`LoopAgent`]: repeats a single agent until a condition is met

use std::sync::Arc;

use tokio::task::JoinSet;

use crate::error::Error;
use crate::llm::LlmProvider;
use crate::llm::types::TokenUsage;

use super::AgentOutput;
use super::AgentRunner;

/// Termination condition for [`LoopAgent`]. Returns `true` to stop the loop.
type StopCondition = Box<dyn Fn(&str) -> bool + Send + Sync>;

// ---------------------------------------------------------------------------
// SequentialAgent
// ---------------------------------------------------------------------------

/// Runs a list of sub-agents in order. Each agent receives the previous
/// agent's text output as its task input. Returns the final agent's output
/// with accumulated `TokenUsage`.
pub struct SequentialAgent<P: LlmProvider> {
    agents: Vec<AgentRunner<P>>,
}

impl<P: LlmProvider> std::fmt::Debug for SequentialAgent<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SequentialAgent")
            .field("agent_count", &self.agents.len())
            .finish()
    }
}

/// Builder for [`SequentialAgent`].
pub struct SequentialAgentBuilder<P: LlmProvider> {
    agents: Vec<AgentRunner<P>>,
}

impl<P: LlmProvider> SequentialAgent<P> {
    pub fn builder() -> SequentialAgentBuilder<P> {
        SequentialAgentBuilder { agents: Vec::new() }
    }

    /// Execute the sequential pipeline, feeding each agent's output as the
    /// next agent's input.
    pub async fn execute(&self, task: &str) -> Result<AgentOutput, Error> {
        let mut current_input = task.to_string();
        let mut total_usage = TokenUsage::default();
        let mut total_tool_calls = 0usize;
        let mut total_cost: Option<f64> = None;
        let mut last_output: Option<AgentOutput> = None;

        for agent in &self.agents {
            let result = agent.execute(&current_input).await.map_err(|e| {
                // Accumulate partial usage from succeeded agents + error's own partial
                let mut partial = total_usage;
                partial += e.partial_usage();
                e.with_partial_usage(partial)
            })?;
            total_usage += result.tokens_used;
            total_tool_calls += result.tool_calls_made;
            if let Some(cost) = result.estimated_cost_usd {
                *total_cost.get_or_insert(0.0) += cost;
            }
            current_input = result.result.clone();
            last_output = Some(result);
        }

        // Safety: builder guarantees at least one agent
        let mut output = last_output.expect("at least one agent");
        output.tokens_used = total_usage;
        output.tool_calls_made = total_tool_calls;
        output.estimated_cost_usd = total_cost;
        Ok(output)
    }
}

impl<P: LlmProvider> SequentialAgentBuilder<P> {
    /// Add an agent to the sequential pipeline.
    pub fn agent(mut self, agent: AgentRunner<P>) -> Self {
        self.agents.push(agent);
        self
    }

    /// Add multiple agents to the sequential pipeline.
    pub fn agents(mut self, agents: Vec<AgentRunner<P>>) -> Self {
        self.agents.extend(agents);
        self
    }

    /// Build the [`SequentialAgent`]. Requires at least one agent.
    pub fn build(self) -> Result<SequentialAgent<P>, Error> {
        if self.agents.is_empty() {
            return Err(Error::Config(
                "SequentialAgent requires at least one agent".into(),
            ));
        }
        Ok(SequentialAgent {
            agents: self.agents,
        })
    }
}

// ---------------------------------------------------------------------------
// ParallelAgent
// ---------------------------------------------------------------------------

/// Runs multiple sub-agents concurrently via `tokio::JoinSet`. All agents
/// receive the same input task. Returns merged results with accumulated
/// `TokenUsage`.
pub struct ParallelAgent<P: LlmProvider + 'static> {
    agents: Vec<Arc<AgentRunner<P>>>,
}

impl<P: LlmProvider + 'static> std::fmt::Debug for ParallelAgent<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParallelAgent")
            .field("agent_count", &self.agents.len())
            .finish()
    }
}

/// Builder for [`ParallelAgent`].
pub struct ParallelAgentBuilder<P: LlmProvider + 'static> {
    agents: Vec<Arc<AgentRunner<P>>>,
}

impl<P: LlmProvider + 'static> ParallelAgent<P> {
    pub fn builder() -> ParallelAgentBuilder<P> {
        ParallelAgentBuilder { agents: Vec::new() }
    }

    /// Execute all agents concurrently. Fails fast on first error.
    pub async fn execute(&self, task: &str) -> Result<AgentOutput, Error> {
        let mut set = JoinSet::new();

        for agent in &self.agents {
            let agent = Arc::clone(agent);
            let task = task.to_string();
            set.spawn(async move {
                let name = agent.name().to_string();
                let result = agent.execute(&task).await;
                (name, result)
            });
        }

        let mut results: Vec<(String, AgentOutput)> = Vec::with_capacity(self.agents.len());
        let mut total_usage = TokenUsage::default();
        let mut total_tool_calls = 0usize;
        let mut total_cost: Option<f64> = None;

        while let Some(join_result) = set.join_next().await {
            let (name, agent_result) = join_result
                .map_err(|e| Error::Agent(format!("parallel agent task panicked: {e}")))?;
            let output = agent_result.map_err(|e| {
                let mut partial = total_usage;
                partial += e.partial_usage();
                e.with_partial_usage(partial)
            })?;
            total_usage += output.tokens_used;
            total_tool_calls += output.tool_calls_made;
            if let Some(cost) = output.estimated_cost_usd {
                *total_cost.get_or_insert(0.0) += cost;
            }
            results.push((name, output));
        }

        // Sort by agent name for deterministic output ordering
        results.sort_by(|a, b| a.0.cmp(&b.0));

        let merged_text = results
            .iter()
            .map(|(name, output)| format!("## {name}\n{}", output.result))
            .collect::<Vec<_>>()
            .join("\n\n");

        Ok(AgentOutput {
            result: merged_text,
            tool_calls_made: total_tool_calls,
            tokens_used: total_usage,
            structured: None,
            estimated_cost_usd: total_cost,
        })
    }
}

impl<P: LlmProvider + 'static> ParallelAgentBuilder<P> {
    /// Add an agent. Wraps it in `Arc` for concurrent sharing.
    pub fn agent(mut self, agent: AgentRunner<P>) -> Self {
        self.agents.push(Arc::new(agent));
        self
    }

    /// Add multiple agents.
    pub fn agents(mut self, agents: Vec<AgentRunner<P>>) -> Self {
        self.agents.extend(agents.into_iter().map(Arc::new));
        self
    }

    /// Build the [`ParallelAgent`]. Requires at least one agent.
    pub fn build(self) -> Result<ParallelAgent<P>, Error> {
        if self.agents.is_empty() {
            return Err(Error::Config(
                "ParallelAgent requires at least one agent".into(),
            ));
        }
        Ok(ParallelAgent {
            agents: self.agents,
        })
    }
}

// ---------------------------------------------------------------------------
// LoopAgent
// ---------------------------------------------------------------------------

/// Runs a single agent in a loop. Stops when `should_stop` returns `true`
/// on the output text, or when `max_iterations` is reached. Returns the
/// final iteration's output with accumulated `TokenUsage`.
pub struct LoopAgent<P: LlmProvider> {
    agent: AgentRunner<P>,
    max_iterations: usize,
    should_stop: StopCondition,
}

impl<P: LlmProvider> std::fmt::Debug for LoopAgent<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoopAgent")
            .field("max_iterations", &self.max_iterations)
            .finish()
    }
}

/// Builder for [`LoopAgent`].
pub struct LoopAgentBuilder<P: LlmProvider> {
    agent: Option<AgentRunner<P>>,
    max_iterations: Option<usize>,
    should_stop: Option<StopCondition>,
}

impl<P: LlmProvider> LoopAgent<P> {
    pub fn builder() -> LoopAgentBuilder<P> {
        LoopAgentBuilder {
            agent: None,
            max_iterations: None,
            should_stop: None,
        }
    }

    /// Execute the loop, feeding each iteration's output as the next input.
    pub async fn execute(&self, task: &str) -> Result<AgentOutput, Error> {
        let mut current_input = task.to_string();
        let mut total_usage = TokenUsage::default();
        let mut total_tool_calls = 0usize;
        let mut total_cost: Option<f64> = None;
        let mut last_output: Option<AgentOutput> = None;

        for _ in 0..self.max_iterations {
            let result = self.agent.execute(&current_input).await.map_err(|e| {
                let mut partial = total_usage;
                partial += e.partial_usage();
                e.with_partial_usage(partial)
            })?;
            total_usage += result.tokens_used;
            total_tool_calls += result.tool_calls_made;
            if let Some(cost) = result.estimated_cost_usd {
                *total_cost.get_or_insert(0.0) += cost;
            }
            current_input = result.result.clone();
            let should_stop = (self.should_stop)(&result.result);
            last_output = Some(result);
            if should_stop {
                break;
            }
        }

        // Safety: max_iterations >= 1 guarantees at least one iteration
        let mut output = last_output.expect("at least one iteration");
        output.tokens_used = total_usage;
        output.tool_calls_made = total_tool_calls;
        output.estimated_cost_usd = total_cost;
        Ok(output)
    }
}

impl<P: LlmProvider> LoopAgentBuilder<P> {
    /// Set the agent to loop.
    pub fn agent(mut self, agent: AgentRunner<P>) -> Self {
        self.agent = Some(agent);
        self
    }

    /// Set the maximum number of iterations (must be >= 1).
    pub fn max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = Some(n);
        self
    }

    /// Set the termination condition. The closure receives the agent's output
    /// text and returns `true` to stop the loop.
    pub fn should_stop(mut self, f: impl Fn(&str) -> bool + Send + Sync + 'static) -> Self {
        self.should_stop = Some(Box::new(f));
        self
    }

    /// Build the [`LoopAgent`].
    pub fn build(self) -> Result<LoopAgent<P>, Error> {
        let agent = self
            .agent
            .ok_or_else(|| Error::Config("LoopAgent requires an agent".into()))?;
        let max_iterations = self
            .max_iterations
            .ok_or_else(|| Error::Config("LoopAgent requires max_iterations".into()))?;
        if max_iterations == 0 {
            return Err(Error::Config(
                "LoopAgent max_iterations must be at least 1".into(),
            ));
        }
        let should_stop = self
            .should_stop
            .ok_or_else(|| Error::Config("LoopAgent requires a should_stop condition".into()))?;
        Ok(LoopAgent {
            agent,
            max_iterations,
            should_stop,
        })
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::types::{CompletionRequest, CompletionResponse, ContentBlock, StopReason};
    use std::sync::Mutex;

    // -----------------------------------------------------------------------
    // Test helpers
    // -----------------------------------------------------------------------

    struct MockProvider {
        responses: Mutex<Vec<CompletionResponse>>,
    }

    impl MockProvider {
        fn new(responses: Vec<CompletionResponse>) -> Self {
            Self {
                responses: Mutex::new(responses),
            }
        }

        fn text_response(text: &str, input_tokens: u32, output_tokens: u32) -> CompletionResponse {
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: text.to_string(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage {
                    input_tokens,
                    output_tokens,
                    ..Default::default()
                },
                model: None,
            }
        }
    }

    impl LlmProvider for MockProvider {
        async fn complete(&self, _request: CompletionRequest) -> Result<CompletionResponse, Error> {
            let mut responses = self.responses.lock().expect("mock lock poisoned");
            if responses.is_empty() {
                return Err(Error::Agent("no more mock responses".into()));
            }
            Ok(responses.remove(0))
        }

        fn model_name(&self) -> Option<&str> {
            Some("mock-model")
        }
    }

    fn make_agent(provider: Arc<MockProvider>, name: &str) -> AgentRunner<MockProvider> {
        AgentRunner::builder(provider)
            .name(name)
            .system_prompt("test system prompt")
            .max_turns(1)
            .build()
            .expect("failed to build test agent")
    }

    // -----------------------------------------------------------------------
    // SequentialAgent builder tests
    // -----------------------------------------------------------------------

    #[test]
    fn sequential_builder_rejects_empty_agents() {
        let result = SequentialAgent::<MockProvider>::builder().build();
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("at least one agent")
        );
    }

    #[test]
    fn sequential_builder_accepts_one_agent() {
        let provider = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "done", 10, 5,
        )]));
        let agent = make_agent(provider, "a");
        let seq = SequentialAgent::builder().agent(agent).build();
        assert!(seq.is_ok());
    }

    // -----------------------------------------------------------------------
    // SequentialAgent execution tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn sequential_single_agent() {
        let provider = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "hello world",
            100,
            50,
        )]));
        let agent = make_agent(provider, "step1");
        let seq = SequentialAgent::builder().agent(agent).build().unwrap();

        let output = seq.execute("start").await.unwrap();
        assert_eq!(output.result, "hello world");
        assert_eq!(output.tokens_used.input_tokens, 100);
        assert_eq!(output.tokens_used.output_tokens, 50);
    }

    #[tokio::test]
    async fn sequential_chains_output_as_input() {
        // Agent A responds with "step-a-output", Agent B responds with "step-b-output".
        // We verify the second agent runs (and its output is final).
        let provider_a = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "step-a-output",
            100,
            50,
        )]));
        let provider_b = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "step-b-output",
            200,
            80,
        )]));

        let agent_a = make_agent(provider_a, "agent-a");
        let agent_b = make_agent(provider_b, "agent-b");

        let seq = SequentialAgent::builder()
            .agent(agent_a)
            .agent(agent_b)
            .build()
            .unwrap();

        let output = seq.execute("initial task").await.unwrap();
        assert_eq!(output.result, "step-b-output");
        // Usage should be accumulated
        assert_eq!(output.tokens_used.input_tokens, 300);
        assert_eq!(output.tokens_used.output_tokens, 130);
    }

    #[tokio::test]
    async fn sequential_three_agents_accumulates_usage() {
        let p1 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "out1", 10, 5,
        )]));
        let p2 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "out2", 20, 10,
        )]));
        let p3 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "out3", 30, 15,
        )]));

        let seq = SequentialAgent::builder()
            .agent(make_agent(p1, "a"))
            .agent(make_agent(p2, "b"))
            .agent(make_agent(p3, "c"))
            .build()
            .unwrap();

        let output = seq.execute("go").await.unwrap();
        assert_eq!(output.result, "out3");
        assert_eq!(output.tokens_used.input_tokens, 60);
        assert_eq!(output.tokens_used.output_tokens, 30);
    }

    #[tokio::test]
    async fn sequential_error_carries_partial_usage() {
        let p1 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "ok", 100, 50,
        )]));
        // Second provider has no responses -> will error
        let p2 = Arc::new(MockProvider::new(vec![]));

        let seq = SequentialAgent::builder()
            .agent(make_agent(p1, "good"))
            .agent(make_agent(p2, "bad"))
            .build()
            .unwrap();

        let err = seq.execute("task").await.unwrap_err();
        let partial = err.partial_usage();
        // Should include the first agent's usage
        assert!(partial.input_tokens >= 100);
    }

    // -----------------------------------------------------------------------
    // ParallelAgent builder tests
    // -----------------------------------------------------------------------

    #[test]
    fn parallel_builder_rejects_empty_agents() {
        let result = ParallelAgent::<MockProvider>::builder().build();
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("at least one agent")
        );
    }

    #[test]
    fn parallel_builder_accepts_one_agent() {
        let provider = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "ok", 10, 5,
        )]));
        let agent = make_agent(provider, "a");
        let par = ParallelAgent::builder().agent(agent).build();
        assert!(par.is_ok());
    }

    // -----------------------------------------------------------------------
    // ParallelAgent execution tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn parallel_single_agent() {
        let provider = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "result-a", 100, 50,
        )]));
        let agent = make_agent(provider, "agent-a");
        let par = ParallelAgent::builder().agent(agent).build().unwrap();

        let output = par.execute("task").await.unwrap();
        assert!(output.result.contains("agent-a"));
        assert!(output.result.contains("result-a"));
        assert_eq!(output.tokens_used.input_tokens, 100);
        assert_eq!(output.tokens_used.output_tokens, 50);
    }

    #[tokio::test]
    async fn parallel_multiple_agents_accumulates_usage() {
        let p1 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "out-a", 100, 50,
        )]));
        let p2 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "out-b", 200, 80,
        )]));

        let par = ParallelAgent::builder()
            .agent(make_agent(p1, "alpha"))
            .agent(make_agent(p2, "beta"))
            .build()
            .unwrap();

        let output = par.execute("same task").await.unwrap();
        // Both agent outputs should appear
        assert!(output.result.contains("out-a"));
        assert!(output.result.contains("out-b"));
        // Both headers should appear
        assert!(output.result.contains("## alpha"));
        assert!(output.result.contains("## beta"));
        // Usage accumulated
        assert_eq!(output.tokens_used.input_tokens, 300);
        assert_eq!(output.tokens_used.output_tokens, 130);
    }

    #[tokio::test]
    async fn parallel_output_sorted_by_name() {
        let p1 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "out-z", 10, 5,
        )]));
        let p2 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "out-a", 10, 5,
        )]));

        let par = ParallelAgent::builder()
            .agent(make_agent(p1, "zebra"))
            .agent(make_agent(p2, "alpha"))
            .build()
            .unwrap();

        let output = par.execute("task").await.unwrap();
        // "alpha" should come before "zebra" in the output
        let alpha_pos = output.result.find("## alpha").unwrap();
        let zebra_pos = output.result.find("## zebra").unwrap();
        assert!(alpha_pos < zebra_pos);
    }

    #[tokio::test]
    async fn parallel_error_fails_fast() {
        let p1 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "ok", 100, 50,
        )]));
        // Second provider will error
        let p2 = Arc::new(MockProvider::new(vec![]));

        let par = ParallelAgent::builder()
            .agent(make_agent(p1, "good"))
            .agent(make_agent(p2, "bad"))
            .build()
            .unwrap();

        let result = par.execute("task").await;
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // LoopAgent builder tests
    // -----------------------------------------------------------------------

    #[test]
    fn loop_builder_rejects_missing_agent() {
        let result = LoopAgent::<MockProvider>::builder()
            .max_iterations(3)
            .should_stop(|_| true)
            .build();
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("requires an agent")
        );
    }

    #[test]
    fn loop_builder_rejects_missing_max_iterations() {
        let provider = Arc::new(MockProvider::new(vec![]));
        let agent = make_agent(provider, "a");
        let result = LoopAgent::builder()
            .agent(agent)
            .should_stop(|_| true)
            .build();
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("requires max_iterations")
        );
    }

    #[test]
    fn loop_builder_rejects_zero_max_iterations() {
        let provider = Arc::new(MockProvider::new(vec![]));
        let agent = make_agent(provider, "a");
        let result = LoopAgent::builder()
            .agent(agent)
            .max_iterations(0)
            .should_stop(|_| true)
            .build();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("at least 1"));
    }

    #[test]
    fn loop_builder_rejects_missing_should_stop() {
        let provider = Arc::new(MockProvider::new(vec![]));
        let agent = make_agent(provider, "a");
        let result = LoopAgent::builder().agent(agent).max_iterations(3).build();
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("requires a should_stop")
        );
    }

    #[test]
    fn loop_builder_accepts_valid_config() {
        let provider = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "x", 1, 1,
        )]));
        let agent = make_agent(provider, "a");
        let result = LoopAgent::builder()
            .agent(agent)
            .max_iterations(5)
            .should_stop(|_| true)
            .build();
        assert!(result.is_ok());
    }

    // -----------------------------------------------------------------------
    // LoopAgent execution tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn loop_stops_on_condition() {
        // Provide 3 responses: the second one contains "DONE"
        let provider = Arc::new(MockProvider::new(vec![
            MockProvider::text_response("working...", 10, 5),
            MockProvider::text_response("DONE", 10, 5),
            MockProvider::text_response("should not reach", 10, 5),
        ]));
        let agent = make_agent(provider, "worker");

        let loop_agent = LoopAgent::builder()
            .agent(agent)
            .max_iterations(10)
            .should_stop(|text| text.contains("DONE"))
            .build()
            .unwrap();

        let output = loop_agent.execute("start").await.unwrap();
        assert_eq!(output.result, "DONE");
        // Only 2 iterations ran
        assert_eq!(output.tokens_used.input_tokens, 20);
        assert_eq!(output.tokens_used.output_tokens, 10);
    }

    #[tokio::test]
    async fn loop_stops_at_max_iterations() {
        let provider = Arc::new(MockProvider::new(vec![
            MockProvider::text_response("iter1", 10, 5),
            MockProvider::text_response("iter2", 10, 5),
            MockProvider::text_response("iter3", 10, 5),
        ]));
        let agent = make_agent(provider, "worker");

        let loop_agent = LoopAgent::builder()
            .agent(agent)
            .max_iterations(3)
            .should_stop(|_| false) // never stop
            .build()
            .unwrap();

        let output = loop_agent.execute("start").await.unwrap();
        assert_eq!(output.result, "iter3");
        assert_eq!(output.tokens_used.input_tokens, 30);
        assert_eq!(output.tokens_used.output_tokens, 15);
    }

    #[tokio::test]
    async fn loop_single_iteration() {
        let provider = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "once", 50, 25,
        )]));
        let agent = make_agent(provider, "worker");

        let loop_agent = LoopAgent::builder()
            .agent(agent)
            .max_iterations(1)
            .should_stop(|_| false)
            .build()
            .unwrap();

        let output = loop_agent.execute("go").await.unwrap();
        assert_eq!(output.result, "once");
        assert_eq!(output.tokens_used.input_tokens, 50);
    }

    #[tokio::test]
    async fn loop_error_carries_partial_usage() {
        // First response succeeds, second errors
        let provider = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "ok", 100, 50,
        )]));
        let agent = make_agent(provider, "worker");

        let loop_agent = LoopAgent::builder()
            .agent(agent)
            .max_iterations(5)
            .should_stop(|_| false) // never stop, will error on 2nd iteration
            .build()
            .unwrap();

        let err = loop_agent.execute("go").await.unwrap_err();
        let partial = err.partial_usage();
        assert!(partial.input_tokens >= 100);
    }

    // -----------------------------------------------------------------------
    // SequentialAgent builder .agents() method
    // -----------------------------------------------------------------------

    #[test]
    fn sequential_builder_agents_method() {
        let p1 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "a", 1, 1,
        )]));
        let p2 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "b", 1, 1,
        )]));
        let agents = vec![make_agent(p1, "x"), make_agent(p2, "y")];
        let seq = SequentialAgent::builder().agents(agents).build();
        assert!(seq.is_ok());
    }

    // -----------------------------------------------------------------------
    // ParallelAgent builder .agents() method
    // -----------------------------------------------------------------------

    #[test]
    fn parallel_builder_agents_method() {
        let p1 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "a", 1, 1,
        )]));
        let p2 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "b", 1, 1,
        )]));
        let agents = vec![make_agent(p1, "x"), make_agent(p2, "y")];
        let par = ParallelAgent::builder().agents(agents).build();
        assert!(par.is_ok());
    }

    // -----------------------------------------------------------------------
    // AgentRunner::name() getter test
    // -----------------------------------------------------------------------

    #[test]
    fn agent_runner_name_getter() {
        let provider = Arc::new(MockProvider::new(vec![]));
        let agent = make_agent(provider, "test-agent");
        assert_eq!(agent.name(), "test-agent");
    }
}
