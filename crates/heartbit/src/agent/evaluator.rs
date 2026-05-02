//! Evaluator-Optimizer workflow agent.
//!
//! Iterative refinement: a **generator** produces output, an **evaluator**
//! critiques it, and the generator revises based on feedback until the
//! evaluator accepts or `max_iterations` is reached.

use regex::Regex;

use crate::error::Error;
use crate::llm::LlmProvider;
use crate::llm::types::TokenUsage;

use super::{AgentOutput, AgentRunner};

/// Iterative refinement workflow: a generator produces output, an evaluator
/// critiques it, and the generator revises based on feedback until the
/// evaluator accepts or max iterations reached.
pub struct EvaluatorOptimizerAgent<P: LlmProvider> {
    generator: AgentRunner<P>,
    evaluator: AgentRunner<P>,
    max_iterations: usize,
    accept_pattern: Option<Regex>,
}

impl<P: LlmProvider> std::fmt::Debug for EvaluatorOptimizerAgent<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EvaluatorOptimizerAgent")
            .field("max_iterations", &self.max_iterations)
            .field("has_accept_pattern", &self.accept_pattern.is_some())
            .finish()
    }
}

/// Builder for [`EvaluatorOptimizerAgent`].
pub struct EvaluatorOptimizerAgentBuilder<P: LlmProvider> {
    generator: Option<AgentRunner<P>>,
    evaluator: Option<AgentRunner<P>>,
    max_iterations: Option<usize>,
    accept_pattern: Option<Regex>,
}

impl<P: LlmProvider> EvaluatorOptimizerAgent<P> {
    pub fn builder() -> EvaluatorOptimizerAgentBuilder<P> {
        EvaluatorOptimizerAgentBuilder {
            generator: None,
            evaluator: None,
            max_iterations: None,
            accept_pattern: None,
        }
    }

    /// Execute the evaluator-optimizer loop.
    ///
    /// The generator produces output, the evaluator critiques it, and the
    /// generator revises based on feedback. Stops when the evaluator accepts
    /// (via `accept_pattern` or case-insensitive "ACCEPT") or `max_iterations`
    /// is reached.
    pub async fn execute(&self, task: &str) -> Result<AgentOutput, Error> {
        let mut total_usage = TokenUsage::default();
        let mut total_tool_calls = 0usize;
        let mut total_cost: Option<f64> = None;
        let mut last_gen_output: Option<AgentOutput> = None;
        let mut eval_feedback: Option<String> = None;

        for iteration in 0..self.max_iterations {
            // Build generator input
            let gen_input = if let Some(feedback) = &eval_feedback {
                format!("{task}\n\n## Evaluator Feedback (iteration {iteration}):\n{feedback}")
            } else {
                task.to_string()
            };

            // Run generator
            let gen_result = self
                .generator
                .execute(&gen_input)
                .await
                .map_err(|e| e.accumulate_usage(total_usage))?;
            gen_result.accumulate_into(&mut total_usage, &mut total_tool_calls, &mut total_cost);
            let gen_text = gen_result.result.clone();
            last_gen_output = Some(gen_result);

            // Run evaluator
            let eval_input = format!(
                "## Task:\n{task}\n\n## Generated Output:\n{gen_text}\n\n\
                 Evaluate quality and correctness. If the output is acceptable, \
                 respond with ACCEPT. Otherwise provide specific, actionable \
                 improvement feedback."
            );

            let eval_result = self
                .evaluator
                .execute(&eval_input)
                .await
                .map_err(|e| e.accumulate_usage(total_usage))?;
            eval_result.accumulate_into(&mut total_usage, &mut total_tool_calls, &mut total_cost);
            let eval_text = &eval_result.result;

            // Check acceptance
            let accepted = if let Some(pattern) = &self.accept_pattern {
                pattern.is_match(eval_text)
            } else {
                {
                    static DEFAULT_ACCEPT: std::sync::LazyLock<Regex> =
                        std::sync::LazyLock::new(|| Regex::new(r"(?i)\bACCEPT\b").unwrap());
                    DEFAULT_ACCEPT.is_match(eval_text)
                }
            };

            if accepted {
                let mut output = last_gen_output.expect("generator ran this iteration");
                output.tokens_used = total_usage;
                output.tool_calls_made = total_tool_calls;
                output.estimated_cost_usd = total_cost;
                return Ok(output);
            }

            // Store feedback for next iteration
            eval_feedback = Some(eval_result.result);
        }

        // Max iterations reached — return last generator output
        let mut output = last_gen_output
            .ok_or_else(|| Error::Agent("EvaluatorOptimizerAgent: no iterations ran".into()))?;
        output.tokens_used = total_usage;
        output.tool_calls_made = total_tool_calls;
        output.estimated_cost_usd = total_cost;
        Ok(output)
    }
}

impl<P: LlmProvider> EvaluatorOptimizerAgentBuilder<P> {
    /// Set the generator agent.
    pub fn generator(mut self, agent: AgentRunner<P>) -> Self {
        self.generator = Some(agent);
        self
    }

    /// Set the evaluator agent.
    pub fn evaluator(mut self, agent: AgentRunner<P>) -> Self {
        self.evaluator = Some(agent);
        self
    }

    /// Set the maximum number of iterations (must be >= 1).
    pub fn max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = Some(n);
        self
    }

    /// Set a custom regex pattern for acceptance detection.
    /// If the evaluator's output matches this pattern, the generator's output
    /// is accepted. Returns an error on invalid regex.
    pub fn accept_pattern(mut self, pattern: &str) -> Result<Self, Error> {
        let re = Regex::new(pattern)
            .map_err(|e| Error::Config(format!("invalid accept_pattern regex: {e}")))?;
        self.accept_pattern = Some(re);
        Ok(self)
    }

    /// Build the [`EvaluatorOptimizerAgent`].
    pub fn build(self) -> Result<EvaluatorOptimizerAgent<P>, Error> {
        let generator = self
            .generator
            .ok_or_else(|| Error::Config("EvaluatorOptimizerAgent requires a generator".into()))?;
        let evaluator = self
            .evaluator
            .ok_or_else(|| Error::Config("EvaluatorOptimizerAgent requires an evaluator".into()))?;
        let max_iterations = self.max_iterations.ok_or_else(|| {
            Error::Config("EvaluatorOptimizerAgent requires max_iterations".into())
        })?;
        if max_iterations == 0 {
            return Err(Error::Config(
                "EvaluatorOptimizerAgent max_iterations must be at least 1".into(),
            ));
        }
        Ok(EvaluatorOptimizerAgent {
            generator,
            evaluator,
            max_iterations,
            accept_pattern: self.accept_pattern,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::agent::test_helpers::{MockProvider, make_agent};

    // -----------------------------------------------------------------------
    // Builder validation tests
    // -----------------------------------------------------------------------

    #[test]
    fn builder_rejects_missing_generator() {
        let provider = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "x", 1, 1,
        )]));
        let evaluator = make_agent(provider, "eval");
        let result = EvaluatorOptimizerAgent::<MockProvider>::builder()
            .evaluator(evaluator)
            .max_iterations(3)
            .build();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("generator"));
    }

    #[test]
    fn builder_rejects_missing_evaluator() {
        let provider = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "x", 1, 1,
        )]));
        let generator = make_agent(provider, "gen");
        let result = EvaluatorOptimizerAgent::builder()
            .generator(generator)
            .max_iterations(3)
            .build();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("evaluator"));
    }

    #[test]
    fn builder_rejects_zero_max_iterations() {
        let gen_provider = Arc::new(MockProvider::new(vec![]));
        let eval_provider = Arc::new(MockProvider::new(vec![]));
        let result = EvaluatorOptimizerAgent::builder()
            .generator(make_agent(gen_provider, "gen"))
            .evaluator(make_agent(eval_provider, "eval"))
            .max_iterations(0)
            .build();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("at least 1"));
    }

    #[test]
    fn builder_rejects_missing_max_iterations() {
        let gen_provider = Arc::new(MockProvider::new(vec![]));
        let eval_provider = Arc::new(MockProvider::new(vec![]));
        let result = EvaluatorOptimizerAgent::builder()
            .generator(make_agent(gen_provider, "gen"))
            .evaluator(make_agent(eval_provider, "eval"))
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
    fn builder_accepts_valid_config() {
        let gen_provider = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "x", 1, 1,
        )]));
        let eval_provider = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "ACCEPT", 1, 1,
        )]));
        let result = EvaluatorOptimizerAgent::builder()
            .generator(make_agent(gen_provider, "gen"))
            .evaluator(make_agent(eval_provider, "eval"))
            .max_iterations(3)
            .build();
        assert!(result.is_ok());
    }

    #[test]
    fn invalid_regex_pattern_rejected() {
        let result = EvaluatorOptimizerAgent::<MockProvider>::builder().accept_pattern("[invalid");
        assert!(result.is_err());
        let err = match result {
            Err(e) => e,
            Ok(_) => panic!("expected error"),
        };
        assert!(err.to_string().contains("invalid"));
    }

    #[test]
    fn valid_regex_pattern_accepted() {
        let result =
            EvaluatorOptimizerAgent::<MockProvider>::builder().accept_pattern(r"PASS|APPROVED");
        assert!(result.is_ok());
    }

    // -----------------------------------------------------------------------
    // Execution tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn execute_evaluator_accepts_first_iteration() {
        let gen_provider = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "great output",
            100,
            50,
        )]));
        let eval_provider = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "This is good. ACCEPT",
            80,
            30,
        )]));

        let agent = EvaluatorOptimizerAgent::builder()
            .generator(make_agent(gen_provider, "gen"))
            .evaluator(make_agent(eval_provider, "eval"))
            .max_iterations(5)
            .build()
            .unwrap();

        let output = agent.execute("write something").await.unwrap();
        assert_eq!(output.result, "great output");
        assert_eq!(output.tokens_used.input_tokens, 180);
        assert_eq!(output.tokens_used.output_tokens, 80);
    }

    #[tokio::test]
    async fn execute_feedback_then_accept() {
        let gen_provider = Arc::new(MockProvider::new(vec![
            MockProvider::text_response("draft v1", 100, 50),
            MockProvider::text_response("draft v2 improved", 120, 60),
        ]));
        let eval_provider = Arc::new(MockProvider::new(vec![
            MockProvider::text_response("Needs more detail on section 2.", 80, 30),
            MockProvider::text_response("Much better. ACCEPT", 90, 35),
        ]));

        let agent = EvaluatorOptimizerAgent::builder()
            .generator(make_agent(Arc::clone(&gen_provider), "gen"))
            .evaluator(make_agent(Arc::clone(&eval_provider), "eval"))
            .max_iterations(5)
            .build()
            .unwrap();

        let output = agent.execute("write an essay").await.unwrap();
        assert_eq!(output.result, "draft v2 improved");
        assert_eq!(output.tokens_used.input_tokens, 100 + 80 + 120 + 90);
        assert_eq!(output.tokens_used.output_tokens, 50 + 30 + 60 + 35);
    }

    #[tokio::test]
    async fn execute_max_iterations_returns_last_output() {
        let gen_provider = Arc::new(MockProvider::new(vec![
            MockProvider::text_response("attempt 1", 10, 5),
            MockProvider::text_response("attempt 2", 10, 5),
            MockProvider::text_response("attempt 3", 10, 5),
        ]));
        let eval_provider = Arc::new(MockProvider::new(vec![
            MockProvider::text_response("try harder", 10, 5),
            MockProvider::text_response("still not good", 10, 5),
            MockProvider::text_response("nope", 10, 5),
        ]));

        let agent = EvaluatorOptimizerAgent::builder()
            .generator(make_agent(Arc::clone(&gen_provider), "gen"))
            .evaluator(make_agent(Arc::clone(&eval_provider), "eval"))
            .max_iterations(3)
            .build()
            .unwrap();

        let output = agent.execute("do the thing").await.unwrap();
        assert_eq!(output.result, "attempt 3");
        assert_eq!(output.tokens_used.input_tokens, 60);
        assert_eq!(output.tokens_used.output_tokens, 30);
    }

    #[tokio::test]
    async fn execute_custom_accept_pattern() {
        let gen_provider = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "generated output",
            50,
            25,
        )]));
        let eval_provider = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "Score: APPROVED",
            40,
            20,
        )]));

        let agent = EvaluatorOptimizerAgent::builder()
            .generator(make_agent(gen_provider, "gen"))
            .evaluator(make_agent(eval_provider, "eval"))
            .max_iterations(5)
            .accept_pattern(r"APPROVED")
            .unwrap()
            .build()
            .unwrap();

        let output = agent.execute("task").await.unwrap();
        assert_eq!(output.result, "generated output");
    }

    #[tokio::test]
    async fn execute_case_insensitive_accept() {
        let gen_provider = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "output", 10, 5,
        )]));
        let eval_provider = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "looks good, accept",
            10,
            5,
        )]));

        let agent = EvaluatorOptimizerAgent::builder()
            .generator(make_agent(gen_provider, "gen"))
            .evaluator(make_agent(eval_provider, "eval"))
            .max_iterations(3)
            .build()
            .unwrap();

        let output = agent.execute("task").await.unwrap();
        assert_eq!(output.result, "output");
    }

    #[test]
    fn debug_impl() {
        let gen_provider = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "x", 1, 1,
        )]));
        let eval_provider = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "ACCEPT", 1, 1,
        )]));
        let agent = EvaluatorOptimizerAgent::builder()
            .generator(make_agent(gen_provider, "gen"))
            .evaluator(make_agent(eval_provider, "eval"))
            .max_iterations(3)
            .build()
            .unwrap();

        let debug = format!("{agent:?}");
        assert!(debug.contains("EvaluatorOptimizerAgent"));
        assert!(debug.contains("max_iterations"));
    }
}
