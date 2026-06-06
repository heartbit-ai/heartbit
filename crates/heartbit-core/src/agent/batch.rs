//! Batch executor for running the same agent on multiple tasks with controlled concurrency.
//!
//! Unlike [`ParallelAgent`](super::workflow::ParallelAgent) which runs different agents on the
//! same task, `BatchExecutor` runs the **same agent** on different tasks with a concurrency limit
//! via [`tokio::sync::Semaphore`].

use std::sync::Arc;

use tokio::sync::Semaphore;
use tokio::task::JoinSet;

use crate::error::Error;
use crate::llm::LlmProvider;
use crate::llm::types::TokenUsage;

use super::AgentOutput;
use super::AgentRunner;

/// Result of a single batch item execution.
#[derive(Debug)]
pub struct BatchResult {
    /// Index of this item in the original input batch.
    pub index: usize,
    /// The input task that was executed.
    pub input: String,
    /// The execution result (Ok with output, or Err).
    pub result: Result<AgentOutput, Error>,
}

/// Configuration for batch execution.
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum number of concurrent agent executions.
    pub max_concurrency: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_concurrency: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4),
        }
    }
}

/// Executes multiple tasks through an agent with controlled concurrency.
///
/// Unlike `ParallelAgent` which runs different agents on the same task,
/// `BatchExecutor` runs the SAME agent on different tasks with a concurrency limit.
pub struct BatchExecutor<P: LlmProvider + 'static> {
    agent: Arc<AgentRunner<P>>,
    config: BatchConfig,
}

impl<P: LlmProvider + 'static> std::fmt::Debug for BatchExecutor<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchExecutor")
            .field("max_concurrency", &self.config.max_concurrency)
            .finish()
    }
}

/// Builder for [`BatchExecutor`].
pub struct BatchExecutorBuilder<P: LlmProvider + 'static> {
    agent: AgentRunner<P>,
    max_concurrency: Option<usize>,
}

impl<P: LlmProvider + 'static> BatchExecutor<P> {
    /// Create a new builder for `BatchExecutor`.
    pub fn builder(agent: AgentRunner<P>) -> BatchExecutorBuilder<P> {
        BatchExecutorBuilder {
            agent,
            max_concurrency: None,
        }
    }

    /// Execute all tasks with controlled concurrency.
    /// Returns results for ALL tasks (successes and failures).
    /// Results are sorted by input index.
    pub async fn execute(&self, tasks: Vec<String>) -> Vec<BatchResult> {
        if tasks.is_empty() {
            return Vec::new();
        }

        let semaphore = Arc::new(Semaphore::new(self.config.max_concurrency));
        let mut set = JoinSet::new();

        for (index, input) in tasks.into_iter().enumerate() {
            let agent = Arc::clone(&self.agent);
            let sem = Arc::clone(&semaphore);
            set.spawn(async move {
                let _permit = sem.acquire().await.expect("semaphore closed unexpectedly");
                let result = agent.execute(&input).await;
                BatchResult {
                    index,
                    input,
                    result,
                }
            });
        }

        let mut results = Vec::with_capacity(set.len());
        while let Some(join_result) = set.join_next().await {
            match join_result {
                Ok(batch_result) => results.push(batch_result),
                Err(e) => {
                    // JoinSet task panicked — should not happen in normal operation.
                    // We can't recover the index/input, so we skip it.
                    // In practice, agent.execute() should not panic.
                    tracing::error!("batch task panicked: {e}");
                }
            }
        }

        results.sort_by_key(|r| r.index);
        results
    }

    /// Convenience: execute with string slice references.
    pub async fn execute_ref(&self, tasks: &[&str]) -> Vec<BatchResult> {
        let owned: Vec<String> = tasks.iter().map(|s| (*s).to_string()).collect();
        self.execute(owned).await
    }

    /// Returns aggregate token usage across all successful executions.
    pub fn aggregate_usage(results: &[BatchResult]) -> TokenUsage {
        let mut total = TokenUsage::default();
        for r in results {
            if let Ok(output) = &r.result {
                total += output.tokens_used;
            }
        }
        total
    }
}

impl<P: LlmProvider + 'static> BatchExecutorBuilder<P> {
    /// Set the maximum number of concurrent agent executions.
    pub fn max_concurrency(mut self, n: usize) -> Self {
        self.max_concurrency = Some(n);
        self
    }

    /// Build the [`BatchExecutor`].
    pub fn build(self) -> Result<BatchExecutor<P>, Error> {
        let config = match self.max_concurrency {
            Some(n) => {
                if n == 0 {
                    return Err(Error::Config(
                        "BatchExecutor max_concurrency must be at least 1".into(),
                    ));
                }
                BatchConfig { max_concurrency: n }
            }
            None => BatchConfig::default(),
        };
        Ok(BatchExecutor {
            agent: Arc::new(self.agent),
            config,
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
    use crate::llm::types::{CompletionRequest, CompletionResponse, ContentBlock, StopReason};
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A mock provider that tracks concurrency via an atomic counter.
    struct ConcurrencyTrackingProvider {
        /// Current number of concurrent executions.
        current: Arc<AtomicUsize>,
        /// Peak concurrency observed.
        peak: Arc<AtomicUsize>,
        response_text: String,
    }

    impl ConcurrencyTrackingProvider {
        fn new(current: Arc<AtomicUsize>, peak: Arc<AtomicUsize>, response_text: &str) -> Self {
            Self {
                current,
                peak,
                response_text: response_text.to_string(),
            }
        }
    }

    impl LlmProvider for ConcurrencyTrackingProvider {
        async fn complete(&self, _request: CompletionRequest) -> Result<CompletionResponse, Error> {
            let prev = self.current.fetch_add(1, Ordering::SeqCst);
            let concurrent = prev + 1;
            // Update peak
            self.peak.fetch_max(concurrent, Ordering::SeqCst);
            // Simulate some work
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            self.current.fetch_sub(1, Ordering::SeqCst);

            Ok(CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: self.response_text.clone(),
                }],
                stop_reason: StopReason::EndTurn,
                reasoning: None,
                usage: TokenUsage {
                    input_tokens: 10,
                    output_tokens: 5,
                    ..Default::default()
                },
                model: None,
            })
        }

        fn model_name(&self) -> Option<&str> {
            Some("concurrency-mock")
        }
    }

    // -----------------------------------------------------------------------
    // Builder tests
    // -----------------------------------------------------------------------

    #[test]
    fn builder_uses_default_concurrency() {
        let provider = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "ok", 10, 5,
        )]));
        let agent = make_agent(provider, "test");
        let executor = BatchExecutor::builder(agent).build().unwrap();
        assert!(executor.config.max_concurrency >= 1);
    }

    #[test]
    fn builder_accepts_custom_concurrency() {
        let provider = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "ok", 10, 5,
        )]));
        let agent = make_agent(provider, "test");
        let executor = BatchExecutor::builder(agent)
            .max_concurrency(8)
            .build()
            .unwrap();
        assert_eq!(executor.config.max_concurrency, 8);
    }

    #[test]
    fn builder_rejects_zero_concurrency() {
        let provider = Arc::new(MockProvider::new(vec![]));
        let agent = make_agent(provider, "test");
        let result = BatchExecutor::builder(agent).max_concurrency(0).build();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("at least 1"));
    }

    #[test]
    fn debug_impl() {
        let provider = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "ok", 10, 5,
        )]));
        let agent = make_agent(provider, "test");
        let executor = BatchExecutor::builder(agent)
            .max_concurrency(3)
            .build()
            .unwrap();
        let debug = format!("{executor:?}");
        assert!(debug.contains("BatchExecutor"));
        assert!(debug.contains("3"));
    }

    // -----------------------------------------------------------------------
    // Execution tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn empty_batch_returns_empty_vec() {
        let provider = Arc::new(MockProvider::new(vec![]));
        let agent = make_agent(provider, "test");
        let executor = BatchExecutor::builder(agent)
            .max_concurrency(2)
            .build()
            .unwrap();

        let results = executor.execute(vec![]).await;
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn single_task_succeeds() {
        let provider = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "hello", 100, 50,
        )]));
        let agent = make_agent(provider, "test");
        let executor = BatchExecutor::builder(agent)
            .max_concurrency(2)
            .build()
            .unwrap();

        let results = executor.execute(vec!["task1".to_string()]).await;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].index, 0);
        assert_eq!(results[0].input, "task1");
        let output = results[0].result.as_ref().unwrap();
        assert_eq!(output.result, "hello");
        assert_eq!(output.tokens_used.input_tokens, 100);
        assert_eq!(output.tokens_used.output_tokens, 50);
    }

    #[tokio::test]
    async fn multiple_tasks_all_succeed() {
        let provider = Arc::new(MockProvider::new(vec![
            MockProvider::text_response("r1", 10, 5),
            MockProvider::text_response("r2", 20, 10),
            MockProvider::text_response("r3", 30, 15),
            MockProvider::text_response("r4", 40, 20),
            MockProvider::text_response("r5", 50, 25),
        ]));
        let agent = make_agent(provider, "test");
        let executor = BatchExecutor::builder(agent)
            .max_concurrency(5)
            .build()
            .unwrap();

        let tasks: Vec<String> = (1..=5).map(|i| format!("task{i}")).collect();
        let results = executor.execute(tasks).await;

        assert_eq!(results.len(), 5);
        for r in &results {
            assert!(r.result.is_ok(), "task {} failed: {:?}", r.index, r.result);
        }
    }

    #[tokio::test]
    async fn results_ordered_by_index() {
        let provider = Arc::new(MockProvider::new(vec![
            MockProvider::text_response("a", 10, 5),
            MockProvider::text_response("b", 10, 5),
            MockProvider::text_response("c", 10, 5),
        ]));
        let agent = make_agent(provider, "test");
        let executor = BatchExecutor::builder(agent)
            .max_concurrency(3)
            .build()
            .unwrap();

        let tasks = vec!["t0".to_string(), "t1".to_string(), "t2".to_string()];
        let results = executor.execute(tasks).await;

        assert_eq!(results.len(), 3);
        for (i, r) in results.iter().enumerate() {
            assert_eq!(r.index, i);
        }
    }

    #[tokio::test]
    async fn partial_failure_returns_all_results() {
        // Provide only 2 responses for 3 tasks — third will fail
        let provider = Arc::new(MockProvider::new(vec![
            MockProvider::text_response("ok1", 10, 5),
            MockProvider::text_response("ok2", 20, 10),
        ]));
        let agent = make_agent(provider, "test");
        // max_concurrency=1 to get deterministic ordering of mock responses
        let executor = BatchExecutor::builder(agent)
            .max_concurrency(1)
            .build()
            .unwrap();

        let tasks = vec![
            "task0".to_string(),
            "task1".to_string(),
            "task2".to_string(),
        ];
        let results = executor.execute(tasks).await;

        assert_eq!(results.len(), 3);
        // First two succeed, third fails
        assert!(results[0].result.is_ok());
        assert!(results[1].result.is_ok());
        assert!(results[2].result.is_err());
    }

    #[tokio::test]
    async fn concurrency_limit_respected() {
        let current = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));

        let provider = Arc::new(ConcurrencyTrackingProvider::new(
            Arc::clone(&current),
            Arc::clone(&peak),
            "done",
        ));
        let agent = AgentRunner::builder(provider)
            .name("conc-test")
            .system_prompt("test")
            .max_turns(1)
            .build()
            .expect("build agent");

        let executor = BatchExecutor::builder(agent)
            .max_concurrency(2)
            .build()
            .unwrap();

        let tasks: Vec<String> = (0..10).map(|i| format!("task{i}")).collect();
        let results = executor.execute(tasks).await;

        assert_eq!(results.len(), 10);
        // Peak concurrency should not exceed 2
        let observed_peak = peak.load(Ordering::SeqCst);
        assert!(
            observed_peak <= 2,
            "peak concurrency was {observed_peak}, expected <= 2"
        );
    }

    #[tokio::test]
    async fn aggregate_usage_sums_successes() {
        let provider = Arc::new(MockProvider::new(vec![
            MockProvider::text_response("a", 100, 50),
            MockProvider::text_response("b", 200, 80),
        ]));
        let agent = make_agent(provider, "test");
        let executor = BatchExecutor::builder(agent)
            .max_concurrency(1)
            .build()
            .unwrap();

        let results = executor
            .execute(vec!["t1".to_string(), "t2".to_string()])
            .await;

        let usage = BatchExecutor::<MockProvider>::aggregate_usage(&results);
        assert_eq!(usage.input_tokens, 300);
        assert_eq!(usage.output_tokens, 130);
    }

    #[tokio::test]
    async fn aggregate_usage_ignores_failures() {
        let provider = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "ok", 100, 50,
        )]));
        let agent = make_agent(provider, "test");
        let executor = BatchExecutor::builder(agent)
            .max_concurrency(1)
            .build()
            .unwrap();

        // 2 tasks but only 1 response — second fails
        let results = executor
            .execute(vec!["t1".to_string(), "t2".to_string()])
            .await;

        let usage = BatchExecutor::<MockProvider>::aggregate_usage(&results);
        // Only the first task's usage
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 50);
    }

    #[tokio::test]
    async fn execute_ref_convenience() {
        let provider = Arc::new(MockProvider::new(vec![
            MockProvider::text_response("a", 10, 5),
            MockProvider::text_response("b", 10, 5),
        ]));
        let agent = make_agent(provider, "test");
        let executor = BatchExecutor::builder(agent)
            .max_concurrency(2)
            .build()
            .unwrap();

        let results = executor.execute_ref(&["hello", "world"]).await;
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].input, "hello");
        assert_eq!(results[1].input, "world");
    }

    #[test]
    fn aggregate_usage_empty_results() {
        let usage = BatchExecutor::<MockProvider>::aggregate_usage(&[]);
        assert_eq!(usage, TokenUsage::default());
    }
}
