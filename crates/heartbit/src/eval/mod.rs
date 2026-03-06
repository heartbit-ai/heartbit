//! Agent evaluation framework.
//!
//! Provides tools for measuring agent quality through repeatable test cases.
//! Inspired by Google ADK's built-in eval: tool trajectory comparison,
//! response quality scoring, and composable scorers.
//!
//! # Quick start
//!
//! ```rust,ignore
//! use heartbit::eval::{EvalCase, EvalRunner, KeywordScorer, TrajectoryScorer};
//!
//! let cases = vec![
//!     EvalCase::new("greeting", "Say hello")
//!         .expect_output_contains("hello")
//!         .expect_no_tools(),
//!     EvalCase::new("file-read", "Read /tmp/test.txt")
//!         .expect_tool("read_file")
//!         .expect_output_contains("content"),
//! ];
//!
//! let runner = EvalRunner::new()
//!     .scorer(TrajectoryScorer)
//!     .scorer(KeywordScorer);
//!
//! let results = runner.run(&agent, &cases).await;
//! let summary = EvalSummary::from_results(&results);
//! println!("{summary}");
//! ```

use std::sync::Arc;

use serde::Serialize;

use crate::agent::events::AgentEvent;
use crate::error::Error;
use crate::llm::pricing::estimate_cost;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// A single evaluation test case.
#[derive(Debug, Clone, Serialize)]
pub struct EvalCase {
    /// Human-readable name for the test case.
    pub name: String,
    /// The task input to send to the agent.
    pub input: String,
    /// Expected tool calls in order (if `Some`). Empty vec means "expect no tools."
    pub expected_tools: Option<Vec<ExpectedToolCall>>,
    /// Strings that should appear in the agent's output.
    pub output_contains: Vec<String>,
    /// Strings that must NOT appear in the agent's output.
    pub output_not_contains: Vec<String>,
    /// Optional reference output for similarity scoring.
    pub reference_output: Option<String>,
    /// Maximum acceptable cost in USD for this case.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_cost_usd: Option<f64>,
    /// Maximum acceptable total LLM latency in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_latency_ms: Option<u64>,
    /// Maximum acceptable number of tool calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tool_calls: Option<usize>,
}

/// An expected tool call in a trajectory.
#[derive(Debug, Clone, Serialize)]
pub struct ExpectedToolCall {
    /// Tool name (exact match).
    pub name: String,
    /// If `Some(n)`, the tool must be called at position `n` (0-indexed).
    /// If `None`, the tool must appear anywhere in the trajectory.
    pub order: Option<usize>,
}

impl EvalCase {
    /// Create a new eval case with a name and input task.
    pub fn new(name: impl Into<String>, input: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            input: input.into(),
            expected_tools: None,
            output_contains: Vec::new(),
            output_not_contains: Vec::new(),
            reference_output: None,
            max_cost_usd: None,
            max_latency_ms: None,
            max_tool_calls: None,
        }
    }

    /// Expect a specific tool to be called (order-independent).
    pub fn expect_tool(mut self, name: impl Into<String>) -> Self {
        self.expected_tools
            .get_or_insert_with(Vec::new)
            .push(ExpectedToolCall {
                name: name.into(),
                order: None,
            });
        self
    }

    /// Expect a tool at a specific position in the trajectory (0-indexed).
    pub fn expect_tool_at(mut self, name: impl Into<String>, position: usize) -> Self {
        self.expected_tools
            .get_or_insert_with(Vec::new)
            .push(ExpectedToolCall {
                name: name.into(),
                order: Some(position),
            });
        self
    }

    /// Expect no tool calls at all.
    pub fn expect_no_tools(mut self) -> Self {
        self.expected_tools = Some(Vec::new());
        self
    }

    /// Expect the output to contain a string.
    pub fn expect_output_contains(mut self, text: impl Into<String>) -> Self {
        self.output_contains.push(text.into());
        self
    }

    /// Expect the output to NOT contain a string.
    pub fn expect_output_not_contains(mut self, text: impl Into<String>) -> Self {
        self.output_not_contains.push(text.into());
        self
    }

    /// Set a reference output for similarity scoring.
    pub fn reference_output(mut self, text: impl Into<String>) -> Self {
        self.reference_output = Some(text.into());
        self
    }

    /// Set maximum acceptable cost in USD.
    pub fn expect_max_cost_usd(mut self, max: f64) -> Self {
        self.max_cost_usd = Some(max);
        self
    }

    /// Set maximum acceptable total LLM latency in milliseconds.
    pub fn expect_max_latency_ms(mut self, max: u64) -> Self {
        self.max_latency_ms = Some(max);
        self
    }

    /// Set maximum acceptable number of tool calls.
    pub fn expect_max_tool_calls(mut self, max: usize) -> Self {
        self.max_tool_calls = Some(max);
        self
    }
}

/// Result of evaluating a single test case.
#[derive(Debug, Clone, Serialize)]
pub struct EvalResult {
    /// Name of the test case.
    pub case_name: String,
    /// Whether the case passed (all scorers above threshold).
    pub passed: bool,
    /// Per-scorer results.
    pub scores: Vec<ScorerResult>,
    /// Actual tool calls made by the agent (in order).
    pub actual_tools: Vec<String>,
    /// Actual agent output text.
    pub actual_output: String,
    /// Error if the agent failed to execute.
    pub error: Option<String>,
}

/// Result from a single scorer.
#[derive(Debug, Clone, Serialize)]
pub struct ScorerResult {
    /// Scorer name.
    pub scorer: String,
    /// Score value (0.0 to 1.0).
    pub score: f64,
    /// Whether this scorer passed.
    pub passed: bool,
    /// Human-readable details.
    pub details: Vec<String>,
}

/// Aggregate summary of multiple eval results.
#[derive(Debug, Clone, Serialize)]
pub struct EvalSummary {
    /// Total cases evaluated.
    pub total: usize,
    /// Cases that passed all scorers.
    pub passed: usize,
    /// Cases that failed at least one scorer.
    pub failed: usize,
    /// Cases that errored (agent execution failure).
    pub errors: usize,
    /// Average score across all cases and scorers.
    pub avg_score: f64,
    /// Per-scorer average scores.
    pub scorer_averages: Vec<(String, f64)>,
}

impl EvalSummary {
    /// Compute summary statistics from eval results.
    pub fn from_results(results: &[EvalResult]) -> Self {
        let total = results.len();
        let passed = results.iter().filter(|r| r.passed).count();
        let errors = results.iter().filter(|r| r.error.is_some()).count();
        let failed = total - passed - errors;

        // Collect all scores
        let mut all_scores: Vec<f64> = Vec::new();
        let mut scorer_totals: std::collections::HashMap<String, (f64, usize)> =
            std::collections::HashMap::new();

        for result in results {
            for sr in &result.scores {
                all_scores.push(sr.score);
                let entry = scorer_totals.entry(sr.scorer.clone()).or_insert((0.0, 0));
                entry.0 += sr.score;
                entry.1 += 1;
            }
        }

        let avg_score = if all_scores.is_empty() {
            0.0
        } else {
            all_scores.iter().sum::<f64>() / all_scores.len() as f64
        };

        let mut scorer_averages: Vec<(String, f64)> = scorer_totals
            .into_iter()
            .map(|(name, (sum, count))| (name, sum / count as f64))
            .collect();
        scorer_averages.sort_by(|a, b| a.0.cmp(&b.0));

        Self {
            total,
            passed,
            failed,
            errors,
            avg_score,
            scorer_averages,
        }
    }

    /// Overall pass rate as a fraction (0.0 to 1.0).
    pub fn pass_rate(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.passed as f64 / self.total as f64
    }
}

impl std::fmt::Display for EvalSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Eval Summary: {}/{} passed", self.passed, self.total)?;
        writeln!(f, "  Pass rate: {:.1}%", self.pass_rate() * 100.0)?;
        writeln!(f, "  Avg score: {:.3}", self.avg_score)?;
        if self.errors > 0 {
            writeln!(f, "  Errors: {}", self.errors)?;
        }
        for (name, avg) in &self.scorer_averages {
            writeln!(f, "  {name}: {avg:.3}")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// EvalScorer trait
// ---------------------------------------------------------------------------

/// Pluggable scoring function for evaluation.
///
/// Scorers evaluate different aspects of agent behavior:
/// - Tool trajectory correctness
/// - Output content quality
/// - Response similarity to reference
pub trait EvalScorer: Send + Sync {
    /// Scorer name for reporting.
    fn name(&self) -> &str;

    /// Score the agent's execution against the eval case.
    ///
    /// Returns a score between 0.0 (worst) and 1.0 (best).
    /// The `details` vec can include human-readable explanations.
    fn score(&self, case: &EvalCase, output: &str, tool_calls: &[String]) -> (f64, Vec<String>);

    /// Minimum score to pass (default: 1.0 for binary pass/fail).
    fn pass_threshold(&self) -> f64 {
        1.0
    }
}

// ---------------------------------------------------------------------------
// TrajectoryScorer
// ---------------------------------------------------------------------------

/// Scores tool call trajectory against expected tool calls.
///
/// Scoring logic:
/// - If no expected tools are specified (`None`), score is 1.0 (pass).
/// - If expected tools is empty vec, score is 1.0 only if no tools were called.
/// - For ordered expectations: exact position match required.
/// - For unordered expectations: tool must appear anywhere in trajectory.
/// - Score = matched expectations / total expectations.
pub struct TrajectoryScorer;

impl EvalScorer for TrajectoryScorer {
    fn name(&self) -> &str {
        "trajectory"
    }

    fn score(&self, case: &EvalCase, _output: &str, tool_calls: &[String]) -> (f64, Vec<String>) {
        let expected = match &case.expected_tools {
            None => return (1.0, vec!["no trajectory expectations".into()]),
            Some(e) => e,
        };

        // Expect no tools
        if expected.is_empty() {
            return if tool_calls.is_empty() {
                (1.0, vec!["correctly made no tool calls".into()])
            } else {
                (
                    0.0,
                    vec![format!(
                        "expected no tools but got: [{}]",
                        tool_calls.join(", ")
                    )],
                )
            };
        }

        let mut matched = 0usize;
        let mut details = Vec::new();

        for exp in expected {
            if let Some(pos) = exp.order {
                // Ordered match: check exact position
                if tool_calls.get(pos).map(|s| s.as_str()) == Some(&exp.name) {
                    matched += 1;
                    details.push(format!("OK: {} at position {pos}", exp.name));
                } else {
                    let actual = tool_calls.get(pos).map(|s| s.as_str()).unwrap_or("<none>");
                    details.push(format!(
                        "FAIL: expected {} at position {pos}, got {actual}",
                        exp.name
                    ));
                }
            } else {
                // Unordered match: check presence
                if tool_calls.iter().any(|t| t == &exp.name) {
                    matched += 1;
                    details.push(format!("OK: {} found in trajectory", exp.name));
                } else {
                    details.push(format!(
                        "FAIL: {} not found in [{}]",
                        exp.name,
                        tool_calls.join(", ")
                    ));
                }
            }
        }

        let score = matched as f64 / expected.len() as f64;
        (score, details)
    }
}

// ---------------------------------------------------------------------------
// KeywordScorer
// ---------------------------------------------------------------------------

/// Scores output against expected keyword presence/absence.
///
/// Score = (contains_matches + not_contains_matches) / total_expectations.
/// Case-insensitive matching.
pub struct KeywordScorer;

impl EvalScorer for KeywordScorer {
    fn name(&self) -> &str {
        "keyword"
    }

    fn score(&self, case: &EvalCase, output: &str, _tool_calls: &[String]) -> (f64, Vec<String>) {
        let total = case.output_contains.len() + case.output_not_contains.len();
        if total == 0 {
            return (1.0, vec!["no keyword expectations".into()]);
        }

        let lower_output = output.to_lowercase();
        let mut matched = 0usize;
        let mut details = Vec::new();

        for keyword in &case.output_contains {
            if lower_output.contains(&keyword.to_lowercase()) {
                matched += 1;
                details.push(format!("OK: output contains \"{keyword}\""));
            } else {
                details.push(format!("FAIL: output missing \"{keyword}\""));
            }
        }

        for keyword in &case.output_not_contains {
            if !lower_output.contains(&keyword.to_lowercase()) {
                matched += 1;
                details.push(format!("OK: output does not contain \"{keyword}\""));
            } else {
                details.push(format!("FAIL: output contains unwanted \"{keyword}\""));
            }
        }

        let score = matched as f64 / total as f64;
        (score, details)
    }
}

// ---------------------------------------------------------------------------
// SimilarityScorer (Rouge-1 unigram overlap)
// ---------------------------------------------------------------------------

/// Scores output similarity to a reference using unigram overlap (Rouge-1 F1).
///
/// If no `reference_output` is set on the case, returns 1.0 (pass).
/// Uses word-level tokenization with case-insensitive matching.
pub struct SimilarityScorer;

impl EvalScorer for SimilarityScorer {
    fn name(&self) -> &str {
        "similarity"
    }

    fn score(&self, case: &EvalCase, output: &str, _tool_calls: &[String]) -> (f64, Vec<String>) {
        let reference = match &case.reference_output {
            None => return (1.0, vec!["no reference output".into()]),
            Some(r) => r,
        };

        let score = rouge1_f1(output, reference);
        let details = vec![format!("Rouge-1 F1: {score:.3}")];
        (score, details)
    }

    fn pass_threshold(&self) -> f64 {
        0.3 // Lenient by default
    }
}

/// Compute Rouge-1 F1 score between candidate and reference texts.
///
/// Uses whitespace tokenization and case-insensitive matching.
fn rouge1_f1(candidate: &str, reference: &str) -> f64 {
    use std::collections::HashSet;

    let cand_tokens: HashSet<String> = candidate
        .split_whitespace()
        .map(|w| w.to_lowercase())
        .collect();
    let ref_tokens: HashSet<String> = reference
        .split_whitespace()
        .map(|w| w.to_lowercase())
        .collect();

    if cand_tokens.is_empty() || ref_tokens.is_empty() {
        return 0.0;
    }

    let overlap = cand_tokens.intersection(&ref_tokens).count() as f64;
    let precision = overlap / cand_tokens.len() as f64;
    let recall = overlap / ref_tokens.len() as f64;

    if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    }
}

// ---------------------------------------------------------------------------
// EvalRunner
// ---------------------------------------------------------------------------

/// Collects tool call names from `AgentEvent::ToolCallStarted` events.
fn collect_tool_calls(events: &[AgentEvent]) -> Vec<String> {
    events
        .iter()
        .filter_map(|e| match e {
            AgentEvent::ToolCallStarted { tool_name, .. } => Some(tool_name.clone()),
            _ => None,
        })
        .collect()
}

/// Runs evaluation cases against an agent and collects scored results.
///
/// The runner wires an `OnEvent` callback to capture the tool call trajectory,
/// then scores each case using the configured scorers.
pub struct EvalRunner {
    scorers: Vec<Box<dyn EvalScorer>>,
}

impl std::fmt::Debug for EvalRunner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EvalRunner")
            .field(
                "scorers",
                &self.scorers.iter().map(|s| s.name()).collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl Default for EvalRunner {
    fn default() -> Self {
        Self::new()
    }
}

impl EvalRunner {
    /// Create a new eval runner with no scorers.
    pub fn new() -> Self {
        Self {
            scorers: Vec::new(),
        }
    }

    /// Add a scorer to the runner.
    pub fn scorer(mut self, scorer: impl EvalScorer + 'static) -> Self {
        self.scorers.push(Box::new(scorer));
        self
    }

    /// Run all eval cases against an agent, returning results.
    ///
    /// Each case runs the agent independently (fresh execution per case).
    ///
    /// **Limitation:** This method cannot capture tool call trajectory data
    /// because the agent's `OnEvent` callback is set at build time. For
    /// trajectory scoring, build the agent with [`build_eval_agent`] and use
    /// [`score_result`](EvalRunner::score_result) with the collected events.
    pub async fn run<P: crate::llm::LlmProvider>(
        &self,
        agent: &crate::agent::AgentRunner<P>,
        cases: &[EvalCase],
    ) -> Vec<EvalResult> {
        let mut results = Vec::with_capacity(cases.len());
        for case in cases {
            results.push(self.run_case(agent, case).await);
        }
        results
    }

    /// Run a single eval case.
    ///
    /// **Note:** Tool trajectory data is NOT captured here because the agent's
    /// `OnEvent` callback cannot be changed after construction. Trajectory
    /// scoring will vacuously pass (no expectations) or fail (expectations
    /// present but no tools observed). For trajectory scoring, use
    /// [`build_eval_agent`] + [`EvalRunner::score_result`] instead.
    async fn run_case<P: crate::llm::LlmProvider>(
        &self,
        agent: &crate::agent::AgentRunner<P>,
        case: &EvalCase,
    ) -> EvalResult {
        match agent.execute(&case.input).await {
            Ok(output) => {
                // Tool call names are not available from AgentOutput (only count).
                // Pass empty trajectory — keyword/similarity scoring still works.
                self.score_result(case, &output.result, &[], None)
            }
            Err(e) => EvalResult {
                case_name: case.name.clone(),
                passed: false,
                scores: Vec::new(),
                actual_tools: Vec::new(),
                actual_output: String::new(),
                error: Some(e.to_string()),
            },
        }
    }

    /// Score a case result with pre-collected tool calls.
    ///
    /// Use this when you have tool call data from an external source
    /// (e.g., `OnEvent` callback, audit trail, or manual testing).
    pub fn score_result(
        &self,
        case: &EvalCase,
        output: &str,
        tool_calls: &[String],
        error: Option<String>,
    ) -> EvalResult {
        let scores: Vec<ScorerResult> = self
            .scorers
            .iter()
            .map(|scorer| {
                let (score, details) = scorer.score(case, output, tool_calls);
                let passed = score >= scorer.pass_threshold();
                ScorerResult {
                    scorer: scorer.name().to_string(),
                    score,
                    passed,
                    details,
                }
            })
            .collect();

        let passed = error.is_none() && scores.iter().all(|s| s.passed);

        EvalResult {
            case_name: case.name.clone(),
            passed,
            scores,
            actual_tools: tool_calls.to_vec(),
            actual_output: output.to_string(),
            error,
        }
    }

    /// Create an event collector callback for capturing tool call trajectory.
    ///
    /// Wire this into `AgentRunnerBuilder::on_event()` before building the agent.
    /// After execution, call `collected_tool_calls()` on the returned vec.
    pub fn event_collector() -> EventCollector {
        Arc::new(std::sync::Mutex::new(Vec::new()))
    }

    /// Build an `OnEvent` callback that pushes events into the collector.
    pub fn event_callback(collector: &EventCollector) -> Arc<dyn Fn(AgentEvent) + Send + Sync> {
        let collector = Arc::clone(collector);
        Arc::new(move |event| {
            collector.lock().expect("eval collector lock").push(event);
        })
    }

    /// Extract tool call names from a collected event vec.
    pub fn collected_tool_calls(collector: &EventCollector) -> Vec<String> {
        let events = collector.lock().expect("eval collector lock");
        collect_tool_calls(&events)
    }
}

/// Shared event collector for eval tool call trajectory capture.
pub type EventCollector = Arc<std::sync::Mutex<Vec<AgentEvent>>>;

/// Clear all events from a collector.
///
/// Call this between eval cases when reusing a single collector across
/// multiple agent executions. Event-aware scorers (`CostScorer`,
/// `LatencyScorer`, `SafetyScorer`) read accumulated events, so stale
/// events from previous cases will corrupt scores if not cleared.
pub fn clear_events(collector: &EventCollector) {
    collector.lock().expect("clear_events lock").clear();
}

/// Build an eval-ready agent with event collection.
///
/// Returns `(agent, collector)`. After `agent.execute()`, use
/// `EvalRunner::collected_tool_calls(&collector)` to get the trajectory.
///
/// This is a convenience helper — you can also wire the event callback
/// manually via `AgentRunnerBuilder::on_event()`.
pub fn build_eval_agent<P: crate::llm::LlmProvider>(
    builder: crate::agent::AgentRunnerBuilder<P>,
) -> Result<(crate::agent::AgentRunner<P>, EventCollector), Error> {
    let collector = EvalRunner::event_collector();
    let callback = EvalRunner::event_callback(&collector);
    let agent = builder.on_event(callback).build()?;
    Ok((agent, collector))
}

// ---------------------------------------------------------------------------
// CostScorer
// ---------------------------------------------------------------------------

/// Scores agent execution against a cost budget.
///
/// Reads `LlmResponse` events from the event collector, estimates cost using
/// `estimate_cost(model, usage)`, and scores against the budget.
/// Unknown models contribute $0 (documented, not penalized).
///
/// **Important:** The collector accumulates events across calls. Call
/// [`clear_events`] between cases when reusing a single collector.
pub struct CostScorer {
    collector: EventCollector,
    max_cost_usd: f64,
}

impl CostScorer {
    /// Create a cost scorer with a default max cost budget.
    ///
    /// The case's `max_cost_usd` overrides this default when set.
    pub fn new(collector: EventCollector, max_cost_usd: f64) -> Self {
        Self {
            collector,
            max_cost_usd,
        }
    }
}

impl EvalScorer for CostScorer {
    fn name(&self) -> &str {
        "cost"
    }

    fn score(&self, case: &EvalCase, _output: &str, _tool_calls: &[String]) -> (f64, Vec<String>) {
        let max = case.max_cost_usd.unwrap_or(self.max_cost_usd);
        if max <= 0.0 {
            return (0.0, vec!["max cost budget is zero".into()]);
        }
        let events = self.collector.lock().expect("cost collector lock");
        let mut total_cost = 0.0f64;
        let mut details = Vec::new();

        for event in events.iter() {
            if let AgentEvent::LlmResponse { usage, model, .. } = event {
                let model_name = model.as_deref().unwrap_or("unknown");
                match estimate_cost(model_name, usage) {
                    Some(cost) => total_cost += cost,
                    None => {
                        details.push(format!("unknown model \"{model_name}\": $0 contributed"));
                    }
                }
            }
        }

        details.insert(0, format!("total cost: ${total_cost:.6} (max: ${max:.6})"));
        (budget_score(total_cost, max), details)
    }

    fn pass_threshold(&self) -> f64 {
        0.01
    }
}

// ---------------------------------------------------------------------------
// LatencyScorer
// ---------------------------------------------------------------------------

/// Scores agent execution against a latency budget.
///
/// Sums `latency_ms` from `LlmResponse` events and scores against the budget.
///
/// **Important:** The collector accumulates events across calls. Call
/// [`clear_events`] between cases when reusing a single collector.
pub struct LatencyScorer {
    collector: EventCollector,
    max_latency_ms: u64,
}

impl LatencyScorer {
    /// Create a latency scorer with a default max latency in milliseconds.
    ///
    /// The case's `max_latency_ms` overrides this default when set.
    pub fn new(collector: EventCollector, max_latency_ms: u64) -> Self {
        Self {
            collector,
            max_latency_ms,
        }
    }
}

impl EvalScorer for LatencyScorer {
    fn name(&self) -> &str {
        "latency"
    }

    fn score(&self, case: &EvalCase, _output: &str, _tool_calls: &[String]) -> (f64, Vec<String>) {
        let max = case.max_latency_ms.unwrap_or(self.max_latency_ms);
        if max == 0 {
            return (0.0, vec!["max latency budget is zero".into()]);
        }
        let events = self.collector.lock().expect("latency collector lock");
        let total_ms: u64 = events
            .iter()
            .filter_map(|e| match e {
                AgentEvent::LlmResponse { latency_ms, .. } => Some(latency_ms),
                _ => None,
            })
            .sum();

        let details = vec![format!("total latency: {total_ms}ms (max: {max}ms)")];
        (budget_score(total_ms as f64, max as f64), details)
    }

    fn pass_threshold(&self) -> f64 {
        0.01
    }
}

// ---------------------------------------------------------------------------
// ToolCallCountScorer
// ---------------------------------------------------------------------------

/// Scores agent execution against a tool call count budget.
///
/// Uses the `tool_calls` slice length from the scorer arguments.
/// Does not require an `EventCollector`.
pub struct ToolCallCountScorer {
    max_calls: usize,
}

impl ToolCallCountScorer {
    /// Create a tool call count scorer with a default maximum.
    ///
    /// The case's `max_tool_calls` overrides this default when set.
    pub fn new(max_calls: usize) -> Self {
        Self { max_calls }
    }
}

impl EvalScorer for ToolCallCountScorer {
    fn name(&self) -> &str {
        "tool_call_count"
    }

    fn score(&self, case: &EvalCase, _output: &str, tool_calls: &[String]) -> (f64, Vec<String>) {
        let max = case.max_tool_calls.unwrap_or(self.max_calls);
        if max == 0 {
            return (0.0, vec!["max tool call budget is zero".into()]);
        }
        let count = tool_calls.len();
        let details = vec![format!("tool calls: {count} (max: {max})")];
        (budget_score(count as f64, max as f64), details)
    }

    fn pass_threshold(&self) -> f64 {
        0.01
    }
}

// ---------------------------------------------------------------------------
// SafetyScorer
// ---------------------------------------------------------------------------

/// Scores agent execution for guardrail safety.
///
/// Checks for `GuardrailDenied` events in the event collector.
/// Score is 0.0 if any denial occurred, 1.0 otherwise.
/// Warnings pass (only denials fail).
///
/// **Important:** The collector accumulates events across calls. Call
/// [`clear_events`] between cases when reusing a single collector.
pub struct SafetyScorer {
    collector: EventCollector,
}

impl SafetyScorer {
    /// Create a safety scorer that reads from the given event collector.
    pub fn new(collector: EventCollector) -> Self {
        Self { collector }
    }
}

impl EvalScorer for SafetyScorer {
    fn name(&self) -> &str {
        "safety"
    }

    fn score(&self, _case: &EvalCase, _output: &str, _tool_calls: &[String]) -> (f64, Vec<String>) {
        let events = self.collector.lock().expect("safety collector lock");
        let mut denials = Vec::new();

        for event in events.iter() {
            if let AgentEvent::GuardrailDenied {
                hook,
                reason,
                tool_name,
                ..
            } = event
            {
                let tool_info = tool_name
                    .as_deref()
                    .map(|t| format!(" (tool: {t})"))
                    .unwrap_or_default();
                denials.push(format!("denied at {hook}{tool_info}: {reason}"));
            }
        }

        if denials.is_empty() {
            (1.0, vec!["no guardrail denials".into()])
        } else {
            (0.0, denials)
        }
    }

    fn pass_threshold(&self) -> f64 {
        1.0
    }
}

// ---------------------------------------------------------------------------
// EvalComparison (A/B with regression detection)
// ---------------------------------------------------------------------------

/// Tolerance for score comparison. Differences smaller than this are ties.
const REGRESSION_TOLERANCE: f64 = 0.001;

/// Comparison of two eval runs for A/B testing and regression detection.
#[derive(Debug, Clone, Serialize)]
pub struct EvalComparison {
    /// Per-case comparison results.
    pub cases: Vec<CaseComparison>,
}

/// Comparison of a single case between baseline and candidate runs.
#[derive(Debug, Clone, Serialize)]
pub struct CaseComparison {
    /// Test case name.
    pub case_name: String,
    /// Average score across all scorers in the baseline run.
    pub baseline_avg_score: f64,
    /// Average score across all scorers in the candidate run.
    pub candidate_avg_score: f64,
    /// Score delta (candidate - baseline). Negative means regression.
    pub delta: f64,
    /// Whether the candidate regressed on this case.
    pub regressed: bool,
}

impl EvalComparison {
    /// Compare baseline and candidate eval results.
    ///
    /// Matches results by `case_name`. Cases present in only one run are skipped.
    pub fn compare(baseline: &[EvalResult], candidate: &[EvalResult]) -> Self {
        let baseline_map: std::collections::HashMap<&str, &EvalResult> =
            baseline.iter().map(|r| (r.case_name.as_str(), r)).collect();

        let cases: Vec<CaseComparison> = candidate
            .iter()
            .filter_map(|cand_result| {
                let base_result = baseline_map.get(cand_result.case_name.as_str())?;
                let base_avg = avg_score(&base_result.scores);
                let cand_avg = avg_score(&cand_result.scores);
                let delta = cand_avg - base_avg;
                Some(CaseComparison {
                    case_name: cand_result.case_name.clone(),
                    baseline_avg_score: base_avg,
                    candidate_avg_score: cand_avg,
                    delta,
                    regressed: delta < -REGRESSION_TOLERANCE,
                })
            })
            .collect();

        Self { cases }
    }

    /// Number of cases where baseline scored higher.
    pub fn baseline_wins(&self) -> usize {
        self.cases.iter().filter(|c| c.regressed).count()
    }

    /// Number of cases where candidate scored higher.
    pub fn candidate_wins(&self) -> usize {
        self.cases
            .iter()
            .filter(|c| c.delta > REGRESSION_TOLERANCE)
            .count()
    }

    /// Number of cases with equal scores (within tolerance).
    pub fn ties(&self) -> usize {
        self.cases.len() - self.baseline_wins() - self.candidate_wins()
    }

    /// Whether any case regressed from baseline to candidate.
    pub fn has_regressions(&self) -> bool {
        self.cases.iter().any(|c| c.regressed)
    }

    /// Names of cases where candidate scored lower than baseline.
    pub fn regressions(&self) -> Vec<&str> {
        self.cases
            .iter()
            .filter(|c| c.regressed)
            .map(|c| c.case_name.as_str())
            .collect()
    }
}

impl std::fmt::Display for EvalComparison {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "A/B Comparison: {} cases ({} baseline wins, {} candidate wins, {} ties)",
            self.cases.len(),
            self.baseline_wins(),
            self.candidate_wins(),
            self.ties()
        )?;
        for c in &self.cases {
            let marker = if c.regressed { "REGRESSED" } else { "ok" };
            writeln!(
                f,
                "  {}: baseline={:.3} candidate={:.3} delta={:+.3} [{}]",
                c.case_name, c.baseline_avg_score, c.candidate_avg_score, c.delta, marker
            )?;
        }
        let regressions = self.regressions();
        if !regressions.is_empty() {
            writeln!(f, "  Regressions: {}", regressions.join(", "))?;
        }
        Ok(())
    }
}

/// Linear budget score: 1.0 when `actual` is 0, 0.0 when `actual >= max`.
fn budget_score(actual: f64, max: f64) -> f64 {
    (1.0 - actual / max).max(0.0)
}

/// Average score from a slice of scorer results. Returns 0.0 if empty.
fn avg_score(scores: &[ScorerResult]) -> f64 {
    if scores.is_empty() {
        return 0.0;
    }
    scores.iter().map(|s| s.score).sum::<f64>() / scores.len() as f64
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // EvalCase builder tests
    // -----------------------------------------------------------------------

    #[test]
    fn eval_case_new() {
        let case = EvalCase::new("test", "do something");
        assert_eq!(case.name, "test");
        assert_eq!(case.input, "do something");
        assert!(case.expected_tools.is_none());
        assert!(case.output_contains.is_empty());
        assert!(case.output_not_contains.is_empty());
        assert!(case.reference_output.is_none());
    }

    #[test]
    fn eval_case_expect_tool() {
        let case = EvalCase::new("t", "i")
            .expect_tool("bash")
            .expect_tool("read_file");
        let tools = case.expected_tools.as_ref().unwrap();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].name, "bash");
        assert!(tools[0].order.is_none());
        assert_eq!(tools[1].name, "read_file");
    }

    #[test]
    fn eval_case_expect_tool_at() {
        let case = EvalCase::new("t", "i")
            .expect_tool_at("bash", 0)
            .expect_tool_at("read_file", 1);
        let tools = case.expected_tools.as_ref().unwrap();
        assert_eq!(tools[0].order, Some(0));
        assert_eq!(tools[1].order, Some(1));
    }

    #[test]
    fn eval_case_expect_no_tools() {
        let case = EvalCase::new("t", "i").expect_no_tools();
        let tools = case.expected_tools.as_ref().unwrap();
        assert!(tools.is_empty());
    }

    #[test]
    fn eval_case_expect_output() {
        let case = EvalCase::new("t", "i")
            .expect_output_contains("hello")
            .expect_output_not_contains("error");
        assert_eq!(case.output_contains, vec!["hello"]);
        assert_eq!(case.output_not_contains, vec!["error"]);
    }

    #[test]
    fn eval_case_reference_output() {
        let case = EvalCase::new("t", "i").reference_output("expected answer");
        assert_eq!(case.reference_output.as_deref(), Some("expected answer"));
    }

    // -----------------------------------------------------------------------
    // TrajectoryScorer tests
    // -----------------------------------------------------------------------

    #[test]
    fn trajectory_no_expectations_passes() {
        let case = EvalCase::new("t", "i"); // no expected_tools
        let (score, _) = TrajectoryScorer.score(&case, "", &["bash".into()]);
        assert_eq!(score, 1.0);
    }

    #[test]
    fn trajectory_expect_no_tools_with_none() {
        let case = EvalCase::new("t", "i").expect_no_tools();
        let (score, _) = TrajectoryScorer.score(&case, "", &[]);
        assert_eq!(score, 1.0);
    }

    #[test]
    fn trajectory_expect_no_tools_but_got_some() {
        let case = EvalCase::new("t", "i").expect_no_tools();
        let (score, details) = TrajectoryScorer.score(&case, "", &["bash".into()]);
        assert_eq!(score, 0.0);
        assert!(details[0].contains("expected no tools"));
    }

    #[test]
    fn trajectory_unordered_match() {
        let case = EvalCase::new("t", "i")
            .expect_tool("read_file")
            .expect_tool("bash");
        let tools = vec!["bash".into(), "read_file".into()];
        let (score, _) = TrajectoryScorer.score(&case, "", &tools);
        assert_eq!(score, 1.0);
    }

    #[test]
    fn trajectory_unordered_partial_match() {
        let case = EvalCase::new("t", "i")
            .expect_tool("read_file")
            .expect_tool("bash");
        let tools = vec!["bash".into()];
        let (score, _) = TrajectoryScorer.score(&case, "", &tools);
        assert_eq!(score, 0.5);
    }

    #[test]
    fn trajectory_unordered_no_match() {
        let case = EvalCase::new("t", "i").expect_tool("bash");
        let tools: Vec<String> = vec!["read_file".into()];
        let (score, _) = TrajectoryScorer.score(&case, "", &tools);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn trajectory_ordered_exact_match() {
        let case = EvalCase::new("t", "i")
            .expect_tool_at("read_file", 0)
            .expect_tool_at("bash", 1);
        let tools = vec!["read_file".into(), "bash".into()];
        let (score, _) = TrajectoryScorer.score(&case, "", &tools);
        assert_eq!(score, 1.0);
    }

    #[test]
    fn trajectory_ordered_wrong_position() {
        let case = EvalCase::new("t", "i")
            .expect_tool_at("bash", 0)
            .expect_tool_at("read_file", 1);
        let tools = vec!["read_file".into(), "bash".into()]; // swapped
        let (score, details) = TrajectoryScorer.score(&case, "", &tools);
        assert_eq!(score, 0.0);
        assert!(details[0].contains("FAIL"));
    }

    #[test]
    fn trajectory_ordered_position_out_of_bounds() {
        let case = EvalCase::new("t", "i").expect_tool_at("bash", 5);
        let tools = vec!["bash".into()];
        let (score, _) = TrajectoryScorer.score(&case, "", &tools);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn trajectory_mixed_ordered_unordered() {
        let case = EvalCase::new("t", "i")
            .expect_tool_at("read_file", 0) // ordered
            .expect_tool("bash"); // unordered
        let tools = vec!["read_file".into(), "write_file".into(), "bash".into()];
        let (score, _) = TrajectoryScorer.score(&case, "", &tools);
        assert_eq!(score, 1.0);
    }

    // -----------------------------------------------------------------------
    // KeywordScorer tests
    // -----------------------------------------------------------------------

    #[test]
    fn keyword_no_expectations_passes() {
        let case = EvalCase::new("t", "i");
        let (score, _) = KeywordScorer.score(&case, "any output", &[]);
        assert_eq!(score, 1.0);
    }

    #[test]
    fn keyword_contains_match() {
        let case = EvalCase::new("t", "i")
            .expect_output_contains("hello")
            .expect_output_contains("world");
        let (score, _) = KeywordScorer.score(&case, "Hello World", &[]);
        assert_eq!(score, 1.0); // case-insensitive
    }

    #[test]
    fn keyword_contains_partial_match() {
        let case = EvalCase::new("t", "i")
            .expect_output_contains("hello")
            .expect_output_contains("missing");
        let (score, _) = KeywordScorer.score(&case, "hello there", &[]);
        assert_eq!(score, 0.5);
    }

    #[test]
    fn keyword_not_contains_match() {
        let case = EvalCase::new("t", "i")
            .expect_output_not_contains("error")
            .expect_output_not_contains("fail");
        let (score, _) = KeywordScorer.score(&case, "success!", &[]);
        assert_eq!(score, 1.0);
    }

    #[test]
    fn keyword_not_contains_violation() {
        let case = EvalCase::new("t", "i").expect_output_not_contains("error");
        let (score, details) = KeywordScorer.score(&case, "An Error occurred", &[]);
        assert_eq!(score, 0.0); // case-insensitive
        assert!(details[0].contains("FAIL"));
    }

    #[test]
    fn keyword_mixed_contains_and_not_contains() {
        let case = EvalCase::new("t", "i")
            .expect_output_contains("result")
            .expect_output_not_contains("error");
        // Both pass
        let (score, _) = KeywordScorer.score(&case, "the result is 42", &[]);
        assert_eq!(score, 1.0);

        // contains fails, not_contains passes
        let (score, _) = KeywordScorer.score(&case, "no match here", &[]);
        assert_eq!(score, 0.5);
    }

    // -----------------------------------------------------------------------
    // SimilarityScorer tests
    // -----------------------------------------------------------------------

    #[test]
    fn similarity_no_reference_passes() {
        let case = EvalCase::new("t", "i");
        let (score, _) = SimilarityScorer.score(&case, "any output", &[]);
        assert_eq!(score, 1.0);
    }

    #[test]
    fn similarity_identical_text() {
        let case = EvalCase::new("t", "i").reference_output("hello world");
        let (score, _) = SimilarityScorer.score(&case, "hello world", &[]);
        assert_eq!(score, 1.0);
    }

    #[test]
    fn similarity_partial_overlap() {
        let case =
            EvalCase::new("t", "i").reference_output("the quick brown fox jumps over the lazy dog");
        let (score, _) = SimilarityScorer.score(&case, "the quick brown cat", &[]);
        assert!(score > 0.0);
        assert!(score < 1.0);
    }

    #[test]
    fn similarity_no_overlap() {
        let case = EvalCase::new("t", "i").reference_output("alpha beta gamma");
        let (score, _) = SimilarityScorer.score(&case, "one two three", &[]);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn similarity_case_insensitive() {
        let case = EvalCase::new("t", "i").reference_output("Hello World");
        let (score, _) = SimilarityScorer.score(&case, "hello world", &[]);
        assert_eq!(score, 1.0);
    }

    #[test]
    fn similarity_empty_candidate() {
        let case = EvalCase::new("t", "i").reference_output("hello world");
        let (score, _) = SimilarityScorer.score(&case, "", &[]);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn similarity_empty_reference() {
        let case = EvalCase::new("t", "i").reference_output("");
        let (score, _) = SimilarityScorer.score(&case, "hello world", &[]);
        assert_eq!(score, 0.0);
    }

    // -----------------------------------------------------------------------
    // Rouge-1 F1 unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn rouge1_identical() {
        assert_eq!(rouge1_f1("hello world", "hello world"), 1.0);
    }

    #[test]
    fn rouge1_no_overlap() {
        assert_eq!(rouge1_f1("a b c", "x y z"), 0.0);
    }

    #[test]
    fn rouge1_partial() {
        // Candidate: {the, cat} Reference: {the, dog}
        // Overlap: {the} = 1
        // Precision: 1/2, Recall: 1/2
        // F1: 2 * 0.5 * 0.5 / (0.5 + 0.5) = 0.5
        assert_eq!(rouge1_f1("the cat", "the dog"), 0.5);
    }

    #[test]
    fn rouge1_empty_candidate() {
        assert_eq!(rouge1_f1("", "hello"), 0.0);
    }

    #[test]
    fn rouge1_empty_reference() {
        assert_eq!(rouge1_f1("hello", ""), 0.0);
    }

    // -----------------------------------------------------------------------
    // EvalRunner::score_result tests
    // -----------------------------------------------------------------------

    #[test]
    fn score_result_no_scorers() {
        let runner = EvalRunner::new();
        let case = EvalCase::new("t", "i");
        let result = runner.score_result(&case, "output", &[], None);
        assert!(result.passed);
        assert!(result.scores.is_empty());
    }

    #[test]
    fn score_result_all_pass() {
        let runner = EvalRunner::new()
            .scorer(TrajectoryScorer)
            .scorer(KeywordScorer);
        let case = EvalCase::new("t", "i")
            .expect_tool("bash")
            .expect_output_contains("done");
        let result = runner.score_result(&case, "done!", &["bash".into()], None);
        assert!(result.passed);
        assert_eq!(result.scores.len(), 2);
        assert!(result.scores.iter().all(|s| s.passed));
    }

    #[test]
    fn score_result_trajectory_fails() {
        let runner = EvalRunner::new().scorer(TrajectoryScorer);
        let case = EvalCase::new("t", "i").expect_tool("bash");
        let result = runner.score_result(&case, "output", &["read_file".into()], None);
        assert!(!result.passed);
    }

    #[test]
    fn score_result_with_error() {
        let runner = EvalRunner::new().scorer(TrajectoryScorer);
        let case = EvalCase::new("t", "i");
        let result = runner.score_result(&case, "", &[], Some("agent failed".into()));
        assert!(!result.passed);
        assert_eq!(result.error.as_deref(), Some("agent failed"));
    }

    #[test]
    fn score_result_preserves_actual_data() {
        let runner = EvalRunner::new();
        let case = EvalCase::new("test-case", "i");
        let tools = vec!["bash".into(), "read".into()];
        let result = runner.score_result(&case, "my output", &tools, None);
        assert_eq!(result.case_name, "test-case");
        assert_eq!(result.actual_output, "my output");
        assert_eq!(result.actual_tools, vec!["bash", "read"]);
    }

    // -----------------------------------------------------------------------
    // EvalSummary tests
    // -----------------------------------------------------------------------

    #[test]
    fn summary_empty_results() {
        let summary = EvalSummary::from_results(&[]);
        assert_eq!(summary.total, 0);
        assert_eq!(summary.passed, 0);
        assert_eq!(summary.pass_rate(), 0.0);
    }

    #[test]
    fn summary_all_pass() {
        let results = vec![
            EvalResult {
                case_name: "a".into(),
                passed: true,
                scores: vec![ScorerResult {
                    scorer: "trajectory".into(),
                    score: 1.0,
                    passed: true,
                    details: vec![],
                }],
                actual_tools: vec![],
                actual_output: String::new(),
                error: None,
            },
            EvalResult {
                case_name: "b".into(),
                passed: true,
                scores: vec![ScorerResult {
                    scorer: "trajectory".into(),
                    score: 1.0,
                    passed: true,
                    details: vec![],
                }],
                actual_tools: vec![],
                actual_output: String::new(),
                error: None,
            },
        ];
        let summary = EvalSummary::from_results(&results);
        assert_eq!(summary.total, 2);
        assert_eq!(summary.passed, 2);
        assert_eq!(summary.failed, 0);
        assert_eq!(summary.pass_rate(), 1.0);
        assert_eq!(summary.avg_score, 1.0);
    }

    #[test]
    fn summary_mixed_results() {
        let results = vec![
            EvalResult {
                case_name: "pass".into(),
                passed: true,
                scores: vec![ScorerResult {
                    scorer: "keyword".into(),
                    score: 1.0,
                    passed: true,
                    details: vec![],
                }],
                actual_tools: vec![],
                actual_output: String::new(),
                error: None,
            },
            EvalResult {
                case_name: "fail".into(),
                passed: false,
                scores: vec![ScorerResult {
                    scorer: "keyword".into(),
                    score: 0.5,
                    passed: false,
                    details: vec![],
                }],
                actual_tools: vec![],
                actual_output: String::new(),
                error: None,
            },
            EvalResult {
                case_name: "error".into(),
                passed: false,
                scores: vec![],
                actual_tools: vec![],
                actual_output: String::new(),
                error: Some("agent failed".into()),
            },
        ];
        let summary = EvalSummary::from_results(&results);
        assert_eq!(summary.total, 3);
        assert_eq!(summary.passed, 1);
        assert_eq!(summary.failed, 1);
        assert_eq!(summary.errors, 1);
        assert!((summary.pass_rate() - 1.0 / 3.0).abs() < 0.001);
    }

    #[test]
    fn summary_scorer_averages() {
        let results = vec![
            EvalResult {
                case_name: "a".into(),
                passed: true,
                scores: vec![
                    ScorerResult {
                        scorer: "trajectory".into(),
                        score: 1.0,
                        passed: true,
                        details: vec![],
                    },
                    ScorerResult {
                        scorer: "keyword".into(),
                        score: 0.8,
                        passed: true,
                        details: vec![],
                    },
                ],
                actual_tools: vec![],
                actual_output: String::new(),
                error: None,
            },
            EvalResult {
                case_name: "b".into(),
                passed: false,
                scores: vec![
                    ScorerResult {
                        scorer: "trajectory".into(),
                        score: 0.5,
                        passed: false,
                        details: vec![],
                    },
                    ScorerResult {
                        scorer: "keyword".into(),
                        score: 1.0,
                        passed: true,
                        details: vec![],
                    },
                ],
                actual_tools: vec![],
                actual_output: String::new(),
                error: None,
            },
        ];
        let summary = EvalSummary::from_results(&results);
        // trajectory avg: (1.0 + 0.5) / 2 = 0.75
        // keyword avg: (0.8 + 1.0) / 2 = 0.9
        let traj = summary
            .scorer_averages
            .iter()
            .find(|(n, _)| n == "trajectory")
            .unwrap();
        assert!((traj.1 - 0.75).abs() < 0.001);
        let kw = summary
            .scorer_averages
            .iter()
            .find(|(n, _)| n == "keyword")
            .unwrap();
        assert!((kw.1 - 0.9).abs() < 0.001);
    }

    #[test]
    fn summary_display() {
        let results = vec![EvalResult {
            case_name: "a".into(),
            passed: true,
            scores: vec![ScorerResult {
                scorer: "trajectory".into(),
                score: 1.0,
                passed: true,
                details: vec![],
            }],
            actual_tools: vec![],
            actual_output: String::new(),
            error: None,
        }];
        let summary = EvalSummary::from_results(&results);
        let display = format!("{summary}");
        assert!(display.contains("1/1 passed"));
        assert!(display.contains("100.0%"));
    }

    // -----------------------------------------------------------------------
    // collect_tool_calls tests
    // -----------------------------------------------------------------------

    #[test]
    fn collect_tool_calls_extracts_started_events() {
        let events = vec![
            AgentEvent::RunStarted {
                agent: "a".into(),
                task: "t".into(),
            },
            AgentEvent::ToolCallStarted {
                agent: "a".into(),
                tool_name: "bash".into(),
                tool_call_id: "c1".into(),
                input: "{}".into(),
            },
            AgentEvent::ToolCallCompleted {
                agent: "a".into(),
                tool_name: "bash".into(),
                tool_call_id: "c1".into(),
                is_error: false,
                duration_ms: 10,
                output: String::new(),
            },
            AgentEvent::ToolCallStarted {
                agent: "a".into(),
                tool_name: "read_file".into(),
                tool_call_id: "c2".into(),
                input: "{}".into(),
            },
        ];
        let tools = collect_tool_calls(&events);
        assert_eq!(tools, vec!["bash", "read_file"]);
    }

    #[test]
    fn collect_tool_calls_empty_events() {
        let tools = collect_tool_calls(&[]);
        assert!(tools.is_empty());
    }

    // -----------------------------------------------------------------------
    // Event collector tests
    // -----------------------------------------------------------------------

    #[test]
    fn event_collector_and_callback() {
        let collector = EvalRunner::event_collector();
        let callback = EvalRunner::event_callback(&collector);

        callback(AgentEvent::ToolCallStarted {
            agent: "a".into(),
            tool_name: "bash".into(),
            tool_call_id: "c1".into(),
            input: "{}".into(),
        });
        callback(AgentEvent::ToolCallStarted {
            agent: "a".into(),
            tool_name: "read_file".into(),
            tool_call_id: "c2".into(),
            input: "{}".into(),
        });

        let tools = EvalRunner::collected_tool_calls(&collector);
        assert_eq!(tools, vec!["bash", "read_file"]);
    }

    // -----------------------------------------------------------------------
    // Integration: EvalRunner with full scoring
    // -----------------------------------------------------------------------

    #[test]
    fn runner_full_scoring_pass() {
        let runner = EvalRunner::new()
            .scorer(TrajectoryScorer)
            .scorer(KeywordScorer)
            .scorer(SimilarityScorer);

        let case = EvalCase::new("full", "test")
            .expect_tool("bash")
            .expect_output_contains("result")
            .reference_output("the result is 42");

        let result = runner.score_result(&case, "the result is 42", &["bash".into()], None);

        assert!(result.passed);
        assert_eq!(result.scores.len(), 3);
        assert!(result.scores.iter().all(|s| s.passed));
    }

    #[test]
    fn runner_full_scoring_fail() {
        let runner = EvalRunner::new()
            .scorer(TrajectoryScorer)
            .scorer(KeywordScorer);

        let case = EvalCase::new("fail", "test")
            .expect_tool("bash")
            .expect_output_contains("result");

        let result = runner.score_result(&case, "no match here", &["read_file".into()], None);

        assert!(!result.passed);
        // Both trajectory and keyword should fail
        assert!(result.scores.iter().all(|s| !s.passed));
    }

    // -----------------------------------------------------------------------
    // EvalRunner::run with mock agent
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn runner_run_with_mock_agent() {
        use crate::llm::LlmProvider;
        use crate::llm::types::{CompletionRequest, CompletionResponse, ContentBlock, StopReason};
        use std::sync::Mutex;

        struct MockProvider {
            response: Mutex<Option<String>>,
        }

        impl LlmProvider for MockProvider {
            async fn complete(
                &self,
                _request: CompletionRequest,
            ) -> Result<CompletionResponse, crate::error::Error> {
                let text = self
                    .response
                    .lock()
                    .expect("mock")
                    .take()
                    .unwrap_or_default();
                Ok(CompletionResponse {
                    content: vec![ContentBlock::Text { text }],
                    stop_reason: StopReason::EndTurn,
                    usage: Default::default(),
                    model: None,
                })
            }
        }

        let provider = Arc::new(MockProvider {
            response: Mutex::new(Some("hello world".into())),
        });
        let agent = crate::agent::AgentRunner::builder(provider)
            .name("eval-test")
            .system_prompt("test")
            .max_turns(1)
            .build()
            .unwrap();

        let runner = EvalRunner::new().scorer(KeywordScorer);
        let cases = vec![EvalCase::new("greeting", "say hello").expect_output_contains("hello")];

        let results = runner.run(&agent, &cases).await;
        assert_eq!(results.len(), 1);
        assert!(results[0].passed);
        assert_eq!(results[0].actual_output, "hello world");
    }

    // -----------------------------------------------------------------------
    // EvalCase budget builder tests
    // -----------------------------------------------------------------------

    #[test]
    fn eval_case_budget_defaults_none() {
        let case = EvalCase::new("t", "i");
        assert!(case.max_cost_usd.is_none());
        assert!(case.max_latency_ms.is_none());
        assert!(case.max_tool_calls.is_none());
    }

    #[test]
    fn eval_case_budget_builders() {
        let case = EvalCase::new("t", "i")
            .expect_max_cost_usd(0.05)
            .expect_max_latency_ms(5000)
            .expect_max_tool_calls(10);
        assert_eq!(case.max_cost_usd, Some(0.05));
        assert_eq!(case.max_latency_ms, Some(5000));
        assert_eq!(case.max_tool_calls, Some(10));
    }

    // -----------------------------------------------------------------------
    // Serialize tests
    // -----------------------------------------------------------------------

    #[test]
    fn eval_case_serializes_to_json() {
        let case = EvalCase::new("test", "do it")
            .expect_tool("bash")
            .expect_max_cost_usd(0.01);
        let json = serde_json::to_string(&case).unwrap();
        assert!(json.contains("\"name\":\"test\""));
        assert!(json.contains("\"max_cost_usd\":0.01"));
    }

    #[test]
    fn eval_result_serializes_to_json() {
        let result = EvalResult {
            case_name: "a".into(),
            passed: true,
            scores: vec![ScorerResult {
                scorer: "keyword".into(),
                score: 1.0,
                passed: true,
                details: vec!["ok".into()],
            }],
            actual_tools: vec!["bash".into()],
            actual_output: "done".into(),
            error: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"passed\":true"));
        assert!(json.contains("\"scorer\":\"keyword\""));
    }

    #[test]
    fn eval_summary_serializes_to_json() {
        let summary = EvalSummary {
            total: 2,
            passed: 1,
            failed: 1,
            errors: 0,
            avg_score: 0.75,
            scorer_averages: vec![("keyword".into(), 0.9)],
        };
        let json = serde_json::to_string(&summary).unwrap();
        assert!(json.contains("\"total\":2"));
        assert!(json.contains("\"avg_score\":0.75"));
    }

    #[test]
    fn eval_case_omits_none_budget_fields() {
        let case = EvalCase::new("t", "i");
        let json = serde_json::to_string(&case).unwrap();
        assert!(!json.contains("max_cost_usd"));
        assert!(!json.contains("max_latency_ms"));
        assert!(!json.contains("max_tool_calls"));
    }

    // -----------------------------------------------------------------------
    // CostScorer tests
    // -----------------------------------------------------------------------

    fn make_llm_response_event(
        model: Option<&str>,
        input: u32,
        output: u32,
        latency: u64,
    ) -> AgentEvent {
        use crate::llm::types::TokenUsage;
        AgentEvent::LlmResponse {
            agent: "a".into(),
            turn: 1,
            usage: TokenUsage {
                input_tokens: input,
                output_tokens: output,
                ..Default::default()
            },
            stop_reason: crate::llm::types::StopReason::EndTurn,
            tool_call_count: 0,
            text: String::new(),
            latency_ms: latency,
            model: model.map(|s| s.to_string()),
            time_to_first_token_ms: 0,
        }
    }

    #[test]
    fn cost_scorer_under_budget() {
        let collector = EvalRunner::event_collector();
        {
            let mut events = collector.lock().unwrap();
            // claude-sonnet-4-20250514: $3/M input, $15/M output
            events.push(make_llm_response_event(
                Some("claude-sonnet-4-20250514"),
                1000,
                500,
                100,
            ));
        }
        let scorer = CostScorer::new(collector, 1.0);
        let case = EvalCase::new("t", "i");
        let (score, details) = scorer.score(&case, "", &[]);
        assert!(score > 0.95); // tiny cost relative to $1 budget
        assert!(details[0].contains("total cost:"));
    }

    #[test]
    fn cost_scorer_over_budget() {
        let collector = EvalRunner::event_collector();
        {
            let mut events = collector.lock().unwrap();
            // 10M output tokens at $15/M = $150
            events.push(make_llm_response_event(
                Some("claude-sonnet-4-20250514"),
                0,
                10_000_000,
                100,
            ));
        }
        let scorer = CostScorer::new(collector, 0.01);
        let case = EvalCase::new("t", "i");
        let (score, _) = scorer.score(&case, "", &[]);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn cost_scorer_unknown_model() {
        let collector = EvalRunner::event_collector();
        {
            let mut events = collector.lock().unwrap();
            events.push(make_llm_response_event(
                Some("unknown-model-xyz"),
                1000,
                1000,
                100,
            ));
        }
        let scorer = CostScorer::new(collector, 1.0);
        let case = EvalCase::new("t", "i");
        let (score, details) = scorer.score(&case, "", &[]);
        assert_eq!(score, 1.0); // $0 cost
        assert!(details.iter().any(|d| d.contains("unknown model")));
    }

    #[test]
    fn cost_scorer_no_model_field() {
        let collector = EvalRunner::event_collector();
        {
            let mut events = collector.lock().unwrap();
            events.push(make_llm_response_event(None, 1000, 1000, 100));
        }
        let scorer = CostScorer::new(collector, 1.0);
        let case = EvalCase::new("t", "i");
        let (score, _) = scorer.score(&case, "", &[]);
        assert_eq!(score, 1.0); // unknown → $0
    }

    #[test]
    fn cost_scorer_case_override() {
        let collector = EvalRunner::event_collector();
        {
            let mut events = collector.lock().unwrap();
            events.push(make_llm_response_event(
                Some("claude-sonnet-4-20250514"),
                100_000,
                50_000,
                100,
            ));
        }
        let scorer = CostScorer::new(collector, 100.0); // very high default
        let case = EvalCase::new("t", "i").expect_max_cost_usd(0.0001); // tiny override
        let (score, _) = scorer.score(&case, "", &[]);
        assert_eq!(score, 0.0); // should fail with tiny budget
    }

    #[test]
    fn cost_scorer_pass_threshold() {
        let scorer = CostScorer::new(EvalRunner::event_collector(), 1.0);
        assert_eq!(scorer.pass_threshold(), 0.01);
    }

    #[test]
    fn cost_scorer_zero_budget() {
        let scorer = CostScorer::new(EvalRunner::event_collector(), 0.0);
        let case = EvalCase::new("t", "i");
        let (score, details) = scorer.score(&case, "", &[]);
        assert_eq!(score, 0.0);
        assert!(details[0].contains("zero"));
    }

    // -----------------------------------------------------------------------
    // LatencyScorer tests
    // -----------------------------------------------------------------------

    #[test]
    fn latency_scorer_under_budget() {
        let collector = EvalRunner::event_collector();
        {
            let mut events = collector.lock().unwrap();
            events.push(make_llm_response_event(None, 0, 0, 500));
            events.push(make_llm_response_event(None, 0, 0, 300));
        }
        let scorer = LatencyScorer::new(collector, 5000);
        let case = EvalCase::new("t", "i");
        let (score, details) = scorer.score(&case, "", &[]);
        // 800ms / 5000ms = 0.16; score = 0.84
        assert!((score - 0.84).abs() < 0.001);
        assert!(details[0].contains("800ms"));
    }

    #[test]
    fn latency_scorer_over_budget() {
        let collector = EvalRunner::event_collector();
        {
            let mut events = collector.lock().unwrap();
            events.push(make_llm_response_event(None, 0, 0, 10_000));
        }
        let scorer = LatencyScorer::new(collector, 5000);
        let case = EvalCase::new("t", "i");
        let (score, _) = scorer.score(&case, "", &[]);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn latency_scorer_case_override() {
        let collector = EvalRunner::event_collector();
        {
            let mut events = collector.lock().unwrap();
            events.push(make_llm_response_event(None, 0, 0, 500));
        }
        let scorer = LatencyScorer::new(collector, 10_000); // high default
        let case = EvalCase::new("t", "i").expect_max_latency_ms(1000); // override
        let (score, _) = scorer.score(&case, "", &[]);
        // 500 / 1000 = 0.5; score = 0.5
        assert!((score - 0.5).abs() < 0.001);
    }

    #[test]
    fn latency_scorer_no_events() {
        let collector = EvalRunner::event_collector();
        let scorer = LatencyScorer::new(collector, 5000);
        let case = EvalCase::new("t", "i");
        let (score, _) = scorer.score(&case, "", &[]);
        assert_eq!(score, 1.0);
    }

    #[test]
    fn latency_scorer_pass_threshold() {
        let scorer = LatencyScorer::new(EvalRunner::event_collector(), 5000);
        assert_eq!(scorer.pass_threshold(), 0.01);
    }

    #[test]
    fn latency_scorer_zero_budget() {
        let scorer = LatencyScorer::new(EvalRunner::event_collector(), 0);
        let case = EvalCase::new("t", "i");
        let (score, details) = scorer.score(&case, "", &[]);
        assert_eq!(score, 0.0);
        assert!(details[0].contains("zero"));
    }

    // -----------------------------------------------------------------------
    // ToolCallCountScorer tests
    // -----------------------------------------------------------------------

    #[test]
    fn tool_call_count_under_budget() {
        let scorer = ToolCallCountScorer::new(10);
        let case = EvalCase::new("t", "i");
        let tools: Vec<String> = vec!["a".into(), "b".into(), "c".into()];
        let (score, details) = scorer.score(&case, "", &tools);
        // 3/10 = 0.3; score = 0.7
        assert!((score - 0.7).abs() < 0.001);
        assert!(details[0].contains("tool calls: 3"));
    }

    #[test]
    fn tool_call_count_over_budget() {
        let scorer = ToolCallCountScorer::new(2);
        let case = EvalCase::new("t", "i");
        let tools: Vec<String> = vec!["a".into(), "b".into(), "c".into()];
        let (score, _) = scorer.score(&case, "", &tools);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn tool_call_count_zero_calls() {
        let scorer = ToolCallCountScorer::new(10);
        let case = EvalCase::new("t", "i");
        let (score, _) = scorer.score(&case, "", &[]);
        assert_eq!(score, 1.0);
    }

    #[test]
    fn tool_call_count_case_override() {
        let scorer = ToolCallCountScorer::new(100); // high default
        let case = EvalCase::new("t", "i").expect_max_tool_calls(2); // tight override
        let tools: Vec<String> = vec!["a".into(), "b".into(), "c".into()];
        let (score, _) = scorer.score(&case, "", &tools);
        assert_eq!(score, 0.0); // 3 > 2
    }

    #[test]
    fn tool_call_count_pass_threshold() {
        let scorer = ToolCallCountScorer::new(10);
        assert_eq!(scorer.pass_threshold(), 0.01);
    }

    #[test]
    fn tool_call_count_zero_budget() {
        let scorer = ToolCallCountScorer::new(0);
        let case = EvalCase::new("t", "i");
        let (score, details) = scorer.score(&case, "", &[]);
        assert_eq!(score, 0.0);
        assert!(details[0].contains("zero"));
    }

    // -----------------------------------------------------------------------
    // SafetyScorer tests
    // -----------------------------------------------------------------------

    #[test]
    fn safety_scorer_no_denials() {
        let collector = EvalRunner::event_collector();
        {
            let mut events = collector.lock().unwrap();
            events.push(AgentEvent::RunStarted {
                agent: "a".into(),
                task: "t".into(),
            });
        }
        let scorer = SafetyScorer::new(collector);
        let case = EvalCase::new("t", "i");
        let (score, details) = scorer.score(&case, "", &[]);
        assert_eq!(score, 1.0);
        assert!(details[0].contains("no guardrail denials"));
    }

    #[test]
    fn safety_scorer_with_denial() {
        let collector = EvalRunner::event_collector();
        {
            let mut events = collector.lock().unwrap();
            events.push(AgentEvent::GuardrailDenied {
                agent: "a".into(),
                hook: "post_llm".into(),
                reason: "unsafe content".into(),
                tool_name: None,
            });
        }
        let scorer = SafetyScorer::new(collector);
        let case = EvalCase::new("t", "i");
        let (score, details) = scorer.score(&case, "", &[]);
        assert_eq!(score, 0.0);
        assert!(details[0].contains("unsafe content"));
    }

    #[test]
    fn safety_scorer_tool_denial() {
        let collector = EvalRunner::event_collector();
        {
            let mut events = collector.lock().unwrap();
            events.push(AgentEvent::GuardrailDenied {
                agent: "a".into(),
                hook: "pre_tool".into(),
                reason: "blocked".into(),
                tool_name: Some("bash".into()),
            });
        }
        let scorer = SafetyScorer::new(collector);
        let case = EvalCase::new("t", "i");
        let (score, details) = scorer.score(&case, "", &[]);
        assert_eq!(score, 0.0);
        assert!(details[0].contains("(tool: bash)"));
    }

    #[test]
    fn safety_scorer_multiple_denials() {
        let collector = EvalRunner::event_collector();
        {
            let mut events = collector.lock().unwrap();
            events.push(AgentEvent::GuardrailDenied {
                agent: "a".into(),
                hook: "post_llm".into(),
                reason: "reason1".into(),
                tool_name: None,
            });
            events.push(AgentEvent::GuardrailDenied {
                agent: "a".into(),
                hook: "pre_tool".into(),
                reason: "reason2".into(),
                tool_name: Some("bash".into()),
            });
        }
        let scorer = SafetyScorer::new(collector);
        let case = EvalCase::new("t", "i");
        let (score, details) = scorer.score(&case, "", &[]);
        assert_eq!(score, 0.0);
        assert_eq!(details.len(), 2);
    }

    #[test]
    fn safety_scorer_pass_threshold() {
        let scorer = SafetyScorer::new(EvalRunner::event_collector());
        assert_eq!(scorer.pass_threshold(), 1.0);
    }

    // -----------------------------------------------------------------------
    // EvalComparison tests
    // -----------------------------------------------------------------------

    fn make_eval_result(name: &str, scores: Vec<(&str, f64)>) -> EvalResult {
        EvalResult {
            case_name: name.into(),
            passed: true,
            scores: scores
                .into_iter()
                .map(|(scorer, score)| ScorerResult {
                    scorer: scorer.into(),
                    score,
                    passed: score >= 0.5,
                    details: vec![],
                })
                .collect(),
            actual_tools: vec![],
            actual_output: String::new(),
            error: None,
        }
    }

    #[test]
    fn comparison_no_regressions() {
        let baseline = vec![
            make_eval_result("a", vec![("keyword", 0.8)]),
            make_eval_result("b", vec![("keyword", 0.6)]),
        ];
        let candidate = vec![
            make_eval_result("a", vec![("keyword", 0.9)]),
            make_eval_result("b", vec![("keyword", 0.7)]),
        ];
        let cmp = EvalComparison::compare(&baseline, &candidate);
        assert!(!cmp.has_regressions());
        assert_eq!(cmp.candidate_wins(), 2);
        assert_eq!(cmp.baseline_wins(), 0);
        assert_eq!(cmp.ties(), 0);
        assert_eq!(cmp.cases.len(), 2);
    }

    #[test]
    fn comparison_with_regression() {
        let baseline = vec![make_eval_result("a", vec![("keyword", 0.9)])];
        let candidate = vec![make_eval_result("a", vec![("keyword", 0.5)])];
        let cmp = EvalComparison::compare(&baseline, &candidate);
        assert!(cmp.has_regressions());
        assert_eq!(cmp.regressions(), vec!["a"]);
        assert_eq!(cmp.baseline_wins(), 1);
        assert_eq!(cmp.candidate_wins(), 0);
        assert!(cmp.cases[0].regressed);
        assert!((cmp.cases[0].delta - (-0.4)).abs() < 0.001);
    }

    #[test]
    fn comparison_ties() {
        let baseline = vec![make_eval_result("a", vec![("keyword", 0.8)])];
        let candidate = vec![make_eval_result("a", vec![("keyword", 0.8)])];
        let cmp = EvalComparison::compare(&baseline, &candidate);
        assert!(!cmp.has_regressions());
        assert_eq!(cmp.ties(), 1);
    }

    #[test]
    fn comparison_skips_unmatched_cases() {
        let baseline = vec![make_eval_result("a", vec![("keyword", 0.8)])];
        let candidate = vec![make_eval_result("b", vec![("keyword", 0.9)])];
        let cmp = EvalComparison::compare(&baseline, &candidate);
        assert!(cmp.cases.is_empty());
    }

    #[test]
    fn comparison_mixed_results() {
        let baseline = vec![
            make_eval_result("a", vec![("k", 0.8), ("t", 0.6)]),
            make_eval_result("b", vec![("k", 0.5), ("t", 0.9)]),
            make_eval_result("c", vec![("k", 1.0)]),
        ];
        let candidate = vec![
            make_eval_result("a", vec![("k", 0.9), ("t", 0.8)]), // improved
            make_eval_result("b", vec![("k", 0.3), ("t", 0.5)]), // regressed
            make_eval_result("c", vec![("k", 1.0)]),             // tie
        ];
        let cmp = EvalComparison::compare(&baseline, &candidate);
        assert_eq!(cmp.candidate_wins(), 1);
        assert_eq!(cmp.baseline_wins(), 1);
        assert_eq!(cmp.ties(), 1);
        assert_eq!(cmp.regressions(), vec!["b"]);
    }

    #[test]
    fn comparison_display() {
        let baseline = vec![make_eval_result("a", vec![("k", 0.8)])];
        let candidate = vec![make_eval_result("a", vec![("k", 0.6)])];
        let cmp = EvalComparison::compare(&baseline, &candidate);
        let display = format!("{cmp}");
        assert!(display.contains("REGRESSED"));
        assert!(display.contains("Regressions: a"));
    }

    #[test]
    fn comparison_serializes_to_json() {
        let baseline = vec![make_eval_result("a", vec![("k", 0.8)])];
        let candidate = vec![make_eval_result("a", vec![("k", 0.9)])];
        let cmp = EvalComparison::compare(&baseline, &candidate);
        let json = serde_json::to_string(&cmp).unwrap();
        assert!(json.contains("\"case_name\":\"a\""));
        assert!(json.contains("\"regressed\":false"));
        assert_eq!(cmp.candidate_wins(), 1);
    }

    #[test]
    fn comparison_empty_inputs() {
        let cmp = EvalComparison::compare(&[], &[]);
        assert!(cmp.cases.is_empty());
        assert!(!cmp.has_regressions());
    }

    // -----------------------------------------------------------------------
    // avg_score helper tests
    // -----------------------------------------------------------------------

    #[test]
    fn avg_score_empty() {
        assert_eq!(avg_score(&[]), 0.0);
    }

    #[test]
    fn avg_score_single() {
        let scores = vec![ScorerResult {
            scorer: "k".into(),
            score: 0.7,
            passed: true,
            details: vec![],
        }];
        assert!((avg_score(&scores) - 0.7).abs() < 0.001);
    }

    #[test]
    fn avg_score_multiple() {
        let scores = vec![
            ScorerResult {
                scorer: "k".into(),
                score: 0.6,
                passed: true,
                details: vec![],
            },
            ScorerResult {
                scorer: "t".into(),
                score: 0.8,
                passed: true,
                details: vec![],
            },
        ];
        assert!((avg_score(&scores) - 0.7).abs() < 0.001);
    }

    // -----------------------------------------------------------------------
    // Integration: new scorers with EvalRunner
    // -----------------------------------------------------------------------

    #[test]
    fn runner_with_tool_call_count_scorer() {
        let runner = EvalRunner::new().scorer(ToolCallCountScorer::new(5));
        let case = EvalCase::new("t", "i");
        let tools: Vec<String> = vec!["a".into(), "b".into()];
        let result = runner.score_result(&case, "output", &tools, None);
        assert!(result.passed);
        // 2/5 = 0.4; score = 0.6
        assert!((result.scores[0].score - 0.6).abs() < 0.001);
    }

    #[test]
    fn runner_with_safety_scorer() {
        let collector = EvalRunner::event_collector();
        let runner = EvalRunner::new().scorer(SafetyScorer::new(Arc::clone(&collector)));
        let case = EvalCase::new("t", "i");
        let result = runner.score_result(&case, "output", &[], None);
        assert!(result.passed);
        assert_eq!(result.scores[0].score, 1.0);
    }

    // -----------------------------------------------------------------------
    // clear_events tests
    // -----------------------------------------------------------------------

    #[test]
    fn clear_events_resets_collector() {
        let collector = EvalRunner::event_collector();
        {
            let mut events = collector.lock().unwrap();
            events.push(make_llm_response_event(None, 0, 0, 1000));
            events.push(AgentEvent::GuardrailDenied {
                agent: "a".into(),
                hook: "post_llm".into(),
                reason: "bad".into(),
                tool_name: None,
            });
        }
        assert_eq!(collector.lock().unwrap().len(), 2);
        clear_events(&collector);
        assert!(collector.lock().unwrap().is_empty());
    }

    #[test]
    fn clear_events_fixes_accumulation_between_cases() {
        let collector = EvalRunner::event_collector();
        let scorer = LatencyScorer::new(Arc::clone(&collector), 1000);
        let case = EvalCase::new("t", "i");

        // Case 1: 500ms
        {
            collector
                .lock()
                .unwrap()
                .push(make_llm_response_event(None, 0, 0, 500));
        }
        let (score1, _) = scorer.score(&case, "", &[]);
        assert!((score1 - 0.5).abs() < 0.001);

        // Without clearing, case 2 would see 500 + 300 = 800ms
        clear_events(&collector);
        {
            collector
                .lock()
                .unwrap()
                .push(make_llm_response_event(None, 0, 0, 300));
        }
        let (score2, _) = scorer.score(&case, "", &[]);
        // 300/1000 = 0.3; score = 0.7
        assert!((score2 - 0.7).abs() < 0.001);
    }

    // -----------------------------------------------------------------------
    // Regression tolerance tests
    // -----------------------------------------------------------------------

    #[test]
    fn comparison_tiny_delta_is_tie() {
        // Delta of 0.0005 is below REGRESSION_TOLERANCE (0.001), should be a tie
        let baseline = vec![make_eval_result("a", vec![("k", 0.8005)])];
        let candidate = vec![make_eval_result("a", vec![("k", 0.8)])];
        let cmp = EvalComparison::compare(&baseline, &candidate);
        assert!(!cmp.has_regressions());
        assert_eq!(cmp.ties(), 1);
    }

    #[test]
    fn comparison_significant_delta_is_regression() {
        // Delta of -0.01 is above REGRESSION_TOLERANCE, should regress
        let baseline = vec![make_eval_result("a", vec![("k", 0.81)])];
        let candidate = vec![make_eval_result("a", vec![("k", 0.8)])];
        let cmp = EvalComparison::compare(&baseline, &candidate);
        assert!(cmp.has_regressions());
        assert_eq!(cmp.regressions(), vec!["a"]);
    }
}
