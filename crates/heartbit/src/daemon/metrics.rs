use prometheus::{
    Counter, Histogram, HistogramOpts, HistogramVec, IntCounter, IntCounterVec, IntGauge, Opts,
    Registry, TextEncoder,
};

use crate::agent::events::AgentEvent;
use crate::llm::pricing::estimate_cost;

/// Prometheus metrics for the Heartbit daemon.
///
/// Uses a dedicated (non-global) `Registry` so multiple `DaemonMetrics` instances
/// can coexist in tests without conflicting.
pub struct DaemonMetrics {
    registry: Registry,

    // Task lifecycle (prefix: heartbit_daemon_)
    tasks_submitted_total: IntCounter,
    tasks_completed_total: IntCounter,
    tasks_failed_total: IntCounter,
    tasks_cancelled_total: IntCounter,
    tasks_active: IntGauge,
    task_duration_seconds: Histogram,

    // LLM (prefix: heartbit_llm_)
    llm_calls_total: IntCounter,
    llm_call_duration_seconds: Histogram,
    llm_ttft_seconds: Histogram,
    llm_tokens_input_total: IntCounter,
    llm_tokens_output_total: IntCounter,
    llm_tokens_cache_read_total: IntCounter,
    llm_tokens_cache_creation_total: IntCounter,
    llm_cost_usd_total: Counter,

    // Tool (prefix: heartbit_tool_, per-name labels)
    tool_calls_total: IntCounterVec,
    tool_duration_seconds: HistogramVec,
    tool_errors_total: IntCounterVec,

    // Reliability (prefix: heartbit_reliability_)
    retry_attempts_total: IntCounter,
    doom_loops_detected_total: IntCounter,
    context_compactions_total: IntCounter,
    guardrail_denials_total: IntCounterVec,

    // Error classification (prefix: heartbit_errors_)
    errors_total: IntCounterVec,

    // Heartbit pulse (prefix: heartbit_pulse_)
    pulse_runs_total: IntCounter,
    pulse_ok_total: IntCounter,
    pulse_action_total: IntCounter,

    // Task source breakdown (prefix: heartbit_daemon_)
    tasks_by_source_total: IntCounterVec,
}

impl DaemonMetrics {
    /// Create a new `DaemonMetrics` with all instruments registered on a
    /// dedicated `Registry`.
    pub fn new() -> Result<Self, prometheus::Error> {
        let registry = Registry::new();

        // -- Task lifecycle --
        let tasks_submitted_total = IntCounter::new(
            "heartbit_daemon_tasks_submitted_total",
            "Total tasks submitted to the daemon",
        )?;
        let tasks_completed_total = IntCounter::new(
            "heartbit_daemon_tasks_completed_total",
            "Total tasks completed successfully",
        )?;
        let tasks_failed_total = IntCounter::new(
            "heartbit_daemon_tasks_failed_total",
            "Total tasks that failed",
        )?;
        let tasks_cancelled_total = IntCounter::new(
            "heartbit_daemon_tasks_cancelled_total",
            "Total tasks cancelled",
        )?;
        let tasks_active = IntGauge::new(
            "heartbit_daemon_tasks_active",
            "Number of currently active tasks",
        )?;
        let task_duration_seconds = Histogram::with_opts(
            HistogramOpts::new(
                "heartbit_daemon_task_duration_seconds",
                "Task execution duration in seconds",
            )
            .buckets(vec![
                1.0, 5.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0,
            ]),
        )?;

        // -- LLM --
        let llm_calls_total = IntCounter::new("heartbit_llm_calls_total", "Total LLM calls made")?;
        let llm_call_duration_seconds = Histogram::with_opts(
            HistogramOpts::new(
                "heartbit_llm_call_duration_seconds",
                "LLM call duration in seconds",
            )
            .buckets(vec![0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]),
        )?;
        let llm_ttft_seconds = Histogram::with_opts(
            HistogramOpts::new(
                "heartbit_llm_ttft_seconds",
                "Time to first token in seconds",
            )
            .buckets(vec![0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]),
        )?;
        let llm_tokens_input_total = IntCounter::new(
            "heartbit_llm_tokens_input_total",
            "Total LLM input tokens consumed",
        )?;
        let llm_tokens_output_total = IntCounter::new(
            "heartbit_llm_tokens_output_total",
            "Total LLM output tokens generated",
        )?;
        let llm_tokens_cache_read_total = IntCounter::new(
            "heartbit_llm_tokens_cache_read_total",
            "Total tokens read from prompt cache",
        )?;
        let llm_tokens_cache_creation_total = IntCounter::new(
            "heartbit_llm_tokens_cache_creation_total",
            "Total tokens used to create prompt cache entries",
        )?;
        let llm_cost_usd_total = Counter::new(
            "heartbit_llm_cost_usd_total",
            "Estimated total LLM cost in USD",
        )?;

        // -- Tool --
        let tool_calls_total = IntCounterVec::new(
            Opts::new("heartbit_tool_calls_total", "Total tool calls by name"),
            &["tool_name"],
        )?;
        let tool_duration_seconds = HistogramVec::new(
            HistogramOpts::new(
                "heartbit_tool_duration_seconds",
                "Tool execution duration in seconds",
            )
            .buckets(vec![0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0, 15.0, 30.0]),
            &["tool_name"],
        )?;
        let tool_errors_total = IntCounterVec::new(
            Opts::new("heartbit_tool_errors_total", "Total tool errors by name"),
            &["tool_name"],
        )?;

        // -- Reliability --
        let retry_attempts_total = IntCounter::new(
            "heartbit_reliability_retry_attempts_total",
            "Total LLM retry attempts",
        )?;
        let doom_loops_detected_total = IntCounter::new(
            "heartbit_reliability_doom_loops_detected_total",
            "Total doom loops detected",
        )?;
        let context_compactions_total = IntCounter::new(
            "heartbit_reliability_context_compactions_total",
            "Total successful context compactions",
        )?;
        let guardrail_denials_total = IntCounterVec::new(
            Opts::new(
                "heartbit_reliability_guardrail_denials_total",
                "Total guardrail denials by hook",
            ),
            &["hook"],
        )?;

        // -- Error classification --
        let errors_total = IntCounterVec::new(
            Opts::new("heartbit_errors_total", "Total errors by classification"),
            &["error_class"],
        )?;

        // -- Heartbit pulse --
        let pulse_runs_total =
            IntCounter::new("heartbit_pulse_runs_total", "Total heartbit pulse runs")?;
        let pulse_ok_total = IntCounter::new(
            "heartbit_pulse_ok_total",
            "Total heartbit pulse runs with HEARTBIT_OK (idle)",
        )?;
        let pulse_action_total = IntCounter::new(
            "heartbit_pulse_action_total",
            "Total heartbit pulse runs that triggered actions",
        )?;

        // -- Task source breakdown --
        let tasks_by_source_total = IntCounterVec::new(
            Opts::new(
                "heartbit_daemon_tasks_by_source_total",
                "Total tasks processed by source (success + failure)",
            ),
            &["source"],
        )?;

        // Register all collectors
        registry.register(Box::new(tasks_submitted_total.clone()))?;
        registry.register(Box::new(tasks_completed_total.clone()))?;
        registry.register(Box::new(tasks_failed_total.clone()))?;
        registry.register(Box::new(tasks_cancelled_total.clone()))?;
        registry.register(Box::new(tasks_active.clone()))?;
        registry.register(Box::new(task_duration_seconds.clone()))?;

        registry.register(Box::new(llm_calls_total.clone()))?;
        registry.register(Box::new(llm_call_duration_seconds.clone()))?;
        registry.register(Box::new(llm_ttft_seconds.clone()))?;
        registry.register(Box::new(llm_tokens_input_total.clone()))?;
        registry.register(Box::new(llm_tokens_output_total.clone()))?;
        registry.register(Box::new(llm_tokens_cache_read_total.clone()))?;
        registry.register(Box::new(llm_tokens_cache_creation_total.clone()))?;
        registry.register(Box::new(llm_cost_usd_total.clone()))?;

        registry.register(Box::new(tool_calls_total.clone()))?;
        registry.register(Box::new(tool_duration_seconds.clone()))?;
        registry.register(Box::new(tool_errors_total.clone()))?;

        registry.register(Box::new(retry_attempts_total.clone()))?;
        registry.register(Box::new(doom_loops_detected_total.clone()))?;
        registry.register(Box::new(context_compactions_total.clone()))?;
        registry.register(Box::new(guardrail_denials_total.clone()))?;

        registry.register(Box::new(errors_total.clone()))?;

        registry.register(Box::new(pulse_runs_total.clone()))?;
        registry.register(Box::new(pulse_ok_total.clone()))?;
        registry.register(Box::new(pulse_action_total.clone()))?;
        registry.register(Box::new(tasks_by_source_total.clone()))?;

        Ok(Self {
            registry,
            tasks_submitted_total,
            tasks_completed_total,
            tasks_failed_total,
            tasks_cancelled_total,
            tasks_active,
            task_duration_seconds,
            llm_calls_total,
            llm_call_duration_seconds,
            llm_ttft_seconds,
            llm_tokens_input_total,
            llm_tokens_output_total,
            llm_tokens_cache_read_total,
            llm_tokens_cache_creation_total,
            llm_cost_usd_total,
            tool_calls_total,
            tool_duration_seconds,
            tool_errors_total,
            retry_attempts_total,
            doom_loops_detected_total,
            context_compactions_total,
            guardrail_denials_total,
            errors_total,
            pulse_runs_total,
            pulse_ok_total,
            pulse_action_total,
            tasks_by_source_total,
        })
    }

    /// Process an `AgentEvent` and update the relevant metrics.
    pub fn record_event(&self, event: &AgentEvent) {
        match event {
            AgentEvent::LlmResponse {
                usage,
                latency_ms,
                model,
                time_to_first_token_ms,
                ..
            } => {
                self.llm_calls_total.inc();
                self.llm_call_duration_seconds
                    .observe(*latency_ms as f64 / 1000.0);
                if *time_to_first_token_ms > 0 {
                    self.llm_ttft_seconds
                        .observe(*time_to_first_token_ms as f64 / 1000.0);
                }
                self.llm_tokens_input_total
                    .inc_by(u64::from(usage.input_tokens));
                self.llm_tokens_output_total
                    .inc_by(u64::from(usage.output_tokens));
                self.llm_tokens_cache_read_total
                    .inc_by(u64::from(usage.cache_read_input_tokens));
                self.llm_tokens_cache_creation_total
                    .inc_by(u64::from(usage.cache_creation_input_tokens));
                if let Some(model_name) = model
                    && let Some(cost) = estimate_cost(model_name, usage)
                {
                    self.llm_cost_usd_total.inc_by(cost);
                }
            }
            AgentEvent::ToolCallCompleted {
                tool_name,
                is_error,
                duration_ms,
                ..
            } => {
                self.tool_calls_total.with_label_values(&[tool_name]).inc();
                self.tool_duration_seconds
                    .with_label_values(&[tool_name])
                    .observe(*duration_ms as f64 / 1000.0);
                if *is_error {
                    self.tool_errors_total.with_label_values(&[tool_name]).inc();
                }
            }
            AgentEvent::RetryAttempt { .. } => {
                self.retry_attempts_total.inc();
            }
            AgentEvent::DoomLoopDetected { .. } => {
                self.doom_loops_detected_total.inc();
            }
            AgentEvent::AutoCompactionTriggered { success, .. } => {
                if *success {
                    self.context_compactions_total.inc();
                }
            }
            AgentEvent::GuardrailDenied { hook, .. } => {
                self.guardrail_denials_total
                    .with_label_values(&[hook])
                    .inc();
            }
            AgentEvent::RunFailed { .. } => {
                self.errors_total.with_label_values(&["run_failed"]).inc();
            }
            // All other variants are no-ops for metrics.
            _ => {}
        }
    }

    /// Record a task submission.
    pub fn record_task_submitted(&self) {
        self.tasks_submitted_total.inc();
    }

    /// Record a successful task completion with its duration.
    pub fn record_task_completed(&self, duration_secs: f64) {
        self.tasks_completed_total.inc();
        self.task_duration_seconds.observe(duration_secs);
    }

    /// Record a task failure with its duration.
    pub fn record_task_failed(&self, duration_secs: f64) {
        self.tasks_failed_total.inc();
        self.task_duration_seconds.observe(duration_secs);
    }

    /// Record a task cancellation.
    pub fn record_task_cancelled(&self) {
        self.tasks_cancelled_total.inc();
    }

    /// Return a reference to the active tasks gauge for external inc/dec.
    pub fn tasks_active(&self) -> &IntGauge {
        &self.tasks_active
    }

    /// Return a reference to the LLM cost counter for external additions.
    pub fn cost_usd(&self) -> &Counter {
        &self.llm_cost_usd_total
    }

    /// Encode all gathered metrics into Prometheus text exposition format.
    pub fn encode(&self) -> Result<String, prometheus::Error> {
        let encoder = TextEncoder::new();
        let families = self.registry.gather();
        encoder.encode_to_string(&families)
    }

    /// Return a reference to the underlying `Registry`.
    pub fn registry(&self) -> &Registry {
        &self.registry
    }

    /// Record a heartbit pulse run.
    pub fn record_pulse_run(&self) {
        self.pulse_runs_total.inc();
    }

    /// Record a heartbit pulse that returned HEARTBIT_OK (idle).
    pub fn record_pulse_ok(&self) {
        self.pulse_ok_total.inc();
    }

    /// Record a heartbit pulse that triggered actions.
    pub fn record_pulse_action(&self) {
        self.pulse_action_total.inc();
    }

    /// Record a completed task by its source (e.g. "api", "heartbit", "sensor:rss", "telegram", "ws").
    pub fn record_task_by_source(&self, source: &str) {
        self.tasks_by_source_total
            .with_label_values(&[source])
            .inc();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::types::{StopReason, TokenUsage};

    #[test]
    fn new_creates_metrics() {
        let metrics = DaemonMetrics::new();
        assert!(metrics.is_ok());
    }

    #[test]
    fn new_starts_at_zero() {
        let m = DaemonMetrics::new().unwrap();
        let text = m.encode().unwrap();
        // All counters should report 0 (or not appear at all for vec counters without labels).
        // Check that scalar counters are present and at 0.
        assert!(text.contains("heartbit_daemon_tasks_submitted_total 0"));
        assert!(text.contains("heartbit_daemon_tasks_completed_total 0"));
        assert!(text.contains("heartbit_daemon_tasks_failed_total 0"));
        assert!(text.contains("heartbit_daemon_tasks_cancelled_total 0"));
        assert!(text.contains("heartbit_daemon_tasks_active 0"));
        assert!(text.contains("heartbit_llm_calls_total 0"));
        assert!(text.contains("heartbit_llm_tokens_input_total 0"));
        assert!(text.contains("heartbit_llm_tokens_output_total 0"));
        assert!(text.contains("heartbit_llm_tokens_cache_read_total 0"));
        assert!(text.contains("heartbit_llm_tokens_cache_creation_total 0"));
        assert!(text.contains("heartbit_llm_cost_usd_total 0"));
        assert!(text.contains("heartbit_reliability_retry_attempts_total 0"));
        assert!(text.contains("heartbit_reliability_doom_loops_detected_total 0"));
        assert!(text.contains("heartbit_reliability_context_compactions_total 0"));
    }

    #[test]
    fn encode_produces_valid_text() {
        let m = DaemonMetrics::new().unwrap();
        let text = m.encode().unwrap();
        assert!(!text.is_empty());
        assert!(text.contains("# HELP"));
        assert!(text.contains("# TYPE"));
    }

    #[test]
    fn record_llm_response() {
        let m = DaemonMetrics::new().unwrap();
        let event = AgentEvent::LlmResponse {
            agent: "test".into(),
            turn: 1,
            usage: TokenUsage {
                input_tokens: 100,
                output_tokens: 50,
                cache_read_input_tokens: 20,
                cache_creation_input_tokens: 10,
                reasoning_tokens: 0,
            },
            stop_reason: StopReason::EndTurn,
            tool_call_count: 0,
            text: "hello".into(),
            latency_ms: 1500,
            model: Some("claude-sonnet-4-20250514".into()),
            time_to_first_token_ms: 250,
        };
        m.record_event(&event);

        let text = m.encode().unwrap();
        assert!(text.contains("heartbit_llm_calls_total 1"));
        assert!(text.contains("heartbit_llm_tokens_input_total 100"));
        assert!(text.contains("heartbit_llm_tokens_output_total 50"));
        assert!(text.contains("heartbit_llm_tokens_cache_read_total 20"));
        assert!(text.contains("heartbit_llm_tokens_cache_creation_total 10"));
        // Cost should be non-zero for a known model
        assert!(m.cost_usd().get() > 0.0, "cost should be positive");
    }

    #[test]
    fn record_tool_completed_success() {
        let m = DaemonMetrics::new().unwrap();
        let event = AgentEvent::ToolCallCompleted {
            agent: "test".into(),
            tool_name: "web_search".into(),
            tool_call_id: "c1".into(),
            is_error: false,
            duration_ms: 500,
            output: "results".into(),
        };
        m.record_event(&event);

        let text = m.encode().unwrap();
        assert!(
            text.contains(r#"heartbit_tool_calls_total{tool_name="web_search"} 1"#),
            "text: {text}"
        );
        // No error increment
        assert!(
            !text.contains(r#"heartbit_tool_errors_total{tool_name="web_search"}"#),
            "text: {text}"
        );
    }

    #[test]
    fn record_tool_completed_error() {
        let m = DaemonMetrics::new().unwrap();
        let event = AgentEvent::ToolCallCompleted {
            agent: "test".into(),
            tool_name: "bash".into(),
            tool_call_id: "c2".into(),
            is_error: true,
            duration_ms: 100,
            output: "error: not found".into(),
        };
        m.record_event(&event);

        let text = m.encode().unwrap();
        assert!(
            text.contains(r#"heartbit_tool_calls_total{tool_name="bash"} 1"#),
            "text: {text}"
        );
        assert!(
            text.contains(r#"heartbit_tool_errors_total{tool_name="bash"} 1"#),
            "text: {text}"
        );
    }

    #[test]
    fn record_retry_attempt() {
        let m = DaemonMetrics::new().unwrap();
        let event = AgentEvent::RetryAttempt {
            agent: "a".into(),
            attempt: 1,
            max_retries: 3,
            delay_ms: 500,
            error_class: "rate_limited".into(),
        };
        m.record_event(&event);

        let text = m.encode().unwrap();
        assert!(text.contains("heartbit_reliability_retry_attempts_total 1"));
    }

    #[test]
    fn record_doom_loop() {
        let m = DaemonMetrics::new().unwrap();
        let event = AgentEvent::DoomLoopDetected {
            agent: "a".into(),
            turn: 5,
            consecutive_count: 3,
            tool_names: vec!["web_search".into()],
        };
        m.record_event(&event);

        let text = m.encode().unwrap();
        assert!(text.contains("heartbit_reliability_doom_loops_detected_total 1"));
    }

    #[test]
    fn record_compaction_success() {
        let m = DaemonMetrics::new().unwrap();
        let event = AgentEvent::AutoCompactionTriggered {
            agent: "a".into(),
            turn: 2,
            success: true,
            usage: TokenUsage::default(),
        };
        m.record_event(&event);

        let text = m.encode().unwrap();
        assert!(text.contains("heartbit_reliability_context_compactions_total 1"));
    }

    #[test]
    fn record_compaction_failure() {
        let m = DaemonMetrics::new().unwrap();
        let event = AgentEvent::AutoCompactionTriggered {
            agent: "a".into(),
            turn: 2,
            success: false,
            usage: TokenUsage::default(),
        };
        m.record_event(&event);

        let text = m.encode().unwrap();
        // Should NOT increment on failure
        assert!(text.contains("heartbit_reliability_context_compactions_total 0"));
    }

    #[test]
    fn record_guardrail_denied() {
        let m = DaemonMetrics::new().unwrap();
        let event = AgentEvent::GuardrailDenied {
            agent: "a".into(),
            hook: "post_llm".into(),
            reason: "unsafe content".into(),
            tool_name: None,
        };
        m.record_event(&event);

        let text = m.encode().unwrap();
        assert!(
            text.contains(r#"heartbit_reliability_guardrail_denials_total{hook="post_llm"} 1"#),
            "text: {text}"
        );
    }

    #[test]
    fn record_run_failed() {
        let m = DaemonMetrics::new().unwrap();
        let event = AgentEvent::RunFailed {
            agent: "a".into(),
            error: "something broke".into(),
            partial_usage: TokenUsage::default(),
        };
        m.record_event(&event);

        let text = m.encode().unwrap();
        assert!(
            text.contains(r#"heartbit_errors_total{error_class="run_failed"} 1"#),
            "text: {text}"
        );
    }

    #[test]
    fn unrecognized_event_noop() {
        let m = DaemonMetrics::new().unwrap();
        let before = m.encode().unwrap();

        // Fire events that should be no-ops for metrics
        let noop_events = vec![
            AgentEvent::RunStarted {
                agent: "a".into(),
                task: "t".into(),
            },
            AgentEvent::TurnStarted {
                agent: "a".into(),
                turn: 1,
                max_turns: 10,
            },
            AgentEvent::ToolCallStarted {
                agent: "a".into(),
                tool_name: "bash".into(),
                tool_call_id: "c1".into(),
                input: "{}".into(),
            },
            AgentEvent::ApprovalRequested {
                agent: "a".into(),
                turn: 1,
                tool_names: vec!["bash".into()],
            },
            AgentEvent::ApprovalDecision {
                agent: "a".into(),
                turn: 1,
                approved: true,
            },
            AgentEvent::SubAgentsDispatched {
                agent: "o".into(),
                agents: vec!["a".into()],
            },
            AgentEvent::SubAgentCompleted {
                agent: "a".into(),
                success: true,
                usage: TokenUsage::default(),
            },
            AgentEvent::ContextSummarized {
                agent: "a".into(),
                turn: 1,
                usage: TokenUsage::default(),
            },
            AgentEvent::RunCompleted {
                agent: "a".into(),
                total_usage: TokenUsage::default(),
                tool_calls_made: 0,
            },
            AgentEvent::SessionPruned {
                agent: "a".into(),
                turn: 1,
                tool_results_pruned: 0,
                bytes_saved: 0,
                tool_results_total: 0,
            },
        ];
        for event in &noop_events {
            m.record_event(event);
        }

        let after = m.encode().unwrap();
        assert_eq!(before, after, "noop events should not change any metrics");
    }

    #[test]
    fn record_task_lifecycle() {
        let m = DaemonMetrics::new().unwrap();

        // Submit
        m.record_task_submitted();
        let text = m.encode().unwrap();
        assert!(text.contains("heartbit_daemon_tasks_submitted_total 1"));

        // Active
        m.tasks_active().inc();
        let text = m.encode().unwrap();
        assert!(text.contains("heartbit_daemon_tasks_active 1"));

        // Complete
        m.tasks_active().dec();
        m.record_task_completed(12.5);
        let text = m.encode().unwrap();
        assert!(text.contains("heartbit_daemon_tasks_completed_total 1"));
        assert!(text.contains("heartbit_daemon_tasks_active 0"));

        // Submit + fail
        m.record_task_submitted();
        m.tasks_active().inc();
        m.tasks_active().dec();
        m.record_task_failed(3.0);
        let text = m.encode().unwrap();
        assert!(text.contains("heartbit_daemon_tasks_submitted_total 2"));
        assert!(text.contains("heartbit_daemon_tasks_failed_total 1"));

        // Cancel
        m.record_task_cancelled();
        let text = m.encode().unwrap();
        assert!(text.contains("heartbit_daemon_tasks_cancelled_total 1"));
    }

    #[test]
    fn record_task_by_source() {
        let m = DaemonMetrics::new().unwrap();

        m.record_task_by_source("heartbit");
        m.record_task_by_source("heartbit");
        m.record_task_by_source("api");
        m.record_task_by_source("sensor:rss");
        m.record_task_by_source("telegram");
        m.record_task_by_source("ws");

        let text = m.encode().unwrap();
        assert!(
            text.contains(r#"heartbit_daemon_tasks_by_source_total{source="heartbit"} 2"#),
            "text: {text}"
        );
        assert!(
            text.contains(r#"heartbit_daemon_tasks_by_source_total{source="api"} 1"#),
            "text: {text}"
        );
        assert!(
            text.contains(r#"heartbit_daemon_tasks_by_source_total{source="sensor:rss"} 1"#),
            "text: {text}"
        );
        assert!(
            text.contains(r#"heartbit_daemon_tasks_by_source_total{source="telegram"} 1"#),
            "text: {text}"
        );
        assert!(
            text.contains(r#"heartbit_daemon_tasks_by_source_total{source="ws"} 1"#),
            "text: {text}"
        );
    }

    #[test]
    fn record_pulse_metrics() {
        let m = DaemonMetrics::new().unwrap();

        m.record_pulse_run();
        m.record_pulse_run();
        m.record_pulse_ok();
        m.record_pulse_action();

        let text = m.encode().unwrap();
        assert!(
            text.contains("heartbit_pulse_runs_total 2"),
            "expected 2 pulse runs, got: {}",
            text.lines()
                .find(|l| l.contains("pulse_runs"))
                .unwrap_or("not found")
        );
        assert!(text.contains("heartbit_pulse_ok_total 1"));
        assert!(text.contains("heartbit_pulse_action_total 1"));
    }
}
