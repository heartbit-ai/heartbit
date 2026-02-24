use prometheus::{
    CounterVec, Histogram, HistogramOpts, HistogramVec, IntCounter, IntCounterVec, IntGauge, Opts,
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

    // LLM (prefix: heartbit_llm_, per-agent labels)
    llm_calls_total: IntCounterVec,
    llm_call_duration_seconds: HistogramVec,
    llm_ttft_seconds: HistogramVec,
    llm_tokens_input_total: IntCounterVec,
    llm_tokens_output_total: IntCounterVec,
    llm_tokens_cache_read_total: IntCounterVec,
    llm_tokens_cache_creation_total: IntCounterVec,
    llm_cost_usd_total: CounterVec,

    // Tool (prefix: heartbit_tool_, per-agent + per-name labels)
    tool_calls_total: IntCounterVec,
    tool_duration_seconds: HistogramVec,
    tool_errors_total: IntCounterVec,

    // Reliability (prefix: heartbit_reliability_)
    retry_attempts_total: IntCounter,
    doom_loops_detected_total: IntCounter,
    context_compactions_total: IntCounter,
    context_summarizations_total: IntCounter,
    session_prunes_total: IntCounter,
    session_bytes_saved_total: IntCounter,
    guardrail_denials_total: IntCounterVec,

    // Error classification (prefix: heartbit_errors_)
    errors_total: IntCounterVec,

    // Heartbit pulse (prefix: heartbit_pulse_)
    pulse_runs_total: IntCounter,
    pulse_ok_total: IntCounter,
    pulse_action_total: IntCounter,

    // Task source breakdown (prefix: heartbit_daemon_)
    tasks_by_source_total: IntCounterVec,

    // Agent lifecycle (prefix: heartbit_agent_)
    agent_runs_started_total: IntCounterVec,
    agent_turns_total: IntCounterVec,
    agent_runs_completed_total: IntCounterVec,

    // Orchestrator (prefix: heartbit_orchestrator_)
    orchestrator_dispatches_total: IntCounter,
    orchestrator_sub_completions_total: IntCounterVec,

    // Interaction (prefix: heartbit_interaction_)
    approvals_requested_total: IntCounter,
    approvals_decided_total: IntCounterVec,

    // Sensor (prefix: heartbit_sensor_)
    sensor_events_processed_total: IntCounterVec,
    sensor_stories_updated_total: IntCounter,

    // Routing (prefix: heartbit_routing_)
    routing_decisions_total: IntCounterVec,
    routing_escalations_total: IntCounter,

    // Cascade (prefix: heartbit_cascade_)
    cascade_escalations_total: IntCounterVec,
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

        // -- LLM (per-agent) --
        let llm_calls_total = IntCounterVec::new(
            Opts::new("heartbit_llm_calls_total", "Total LLM calls made"),
            &["agent"],
        )?;
        let llm_call_duration_seconds = HistogramVec::new(
            HistogramOpts::new(
                "heartbit_llm_call_duration_seconds",
                "LLM call duration in seconds",
            )
            .buckets(vec![0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]),
            &["agent"],
        )?;
        let llm_ttft_seconds = HistogramVec::new(
            HistogramOpts::new(
                "heartbit_llm_ttft_seconds",
                "Time to first token in seconds",
            )
            .buckets(vec![0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]),
            &["agent"],
        )?;
        let llm_tokens_input_total = IntCounterVec::new(
            Opts::new(
                "heartbit_llm_tokens_input_total",
                "Total LLM input tokens consumed",
            ),
            &["agent"],
        )?;
        let llm_tokens_output_total = IntCounterVec::new(
            Opts::new(
                "heartbit_llm_tokens_output_total",
                "Total LLM output tokens generated",
            ),
            &["agent"],
        )?;
        let llm_tokens_cache_read_total = IntCounterVec::new(
            Opts::new(
                "heartbit_llm_tokens_cache_read_total",
                "Total tokens read from prompt cache",
            ),
            &["agent"],
        )?;
        let llm_tokens_cache_creation_total = IntCounterVec::new(
            Opts::new(
                "heartbit_llm_tokens_cache_creation_total",
                "Total tokens used to create prompt cache entries",
            ),
            &["agent"],
        )?;
        let llm_cost_usd_total = CounterVec::new(
            Opts::new(
                "heartbit_llm_cost_usd_total",
                "Estimated total LLM cost in USD",
            ),
            &["agent"],
        )?;

        // -- Tool (per-agent + per-name) --
        let tool_calls_total = IntCounterVec::new(
            Opts::new("heartbit_tool_calls_total", "Total tool calls by name"),
            &["agent", "tool_name"],
        )?;
        let tool_duration_seconds = HistogramVec::new(
            HistogramOpts::new(
                "heartbit_tool_duration_seconds",
                "Tool execution duration in seconds",
            )
            .buckets(vec![0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0, 15.0, 30.0]),
            &["agent", "tool_name"],
        )?;
        let tool_errors_total = IntCounterVec::new(
            Opts::new("heartbit_tool_errors_total", "Total tool errors by name"),
            &["agent", "tool_name"],
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
        let context_summarizations_total = IntCounter::new(
            "heartbit_reliability_context_summarizations_total",
            "Total context summarizations",
        )?;
        let session_prunes_total = IntCounter::new(
            "heartbit_reliability_session_prunes_total",
            "Total session prunes",
        )?;
        let session_bytes_saved_total = IntCounter::new(
            "heartbit_reliability_session_bytes_saved_total",
            "Total bytes saved by session pruning",
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

        // -- Agent lifecycle --
        let agent_runs_started_total = IntCounterVec::new(
            Opts::new(
                "heartbit_agent_runs_started_total",
                "Total agent runs started",
            ),
            &["agent"],
        )?;
        let agent_turns_total = IntCounterVec::new(
            Opts::new("heartbit_agent_turns_total", "Total agent turns"),
            &["agent"],
        )?;
        let agent_runs_completed_total = IntCounterVec::new(
            Opts::new(
                "heartbit_agent_runs_completed_total",
                "Total agent runs completed",
            ),
            &["agent"],
        )?;

        // -- Orchestrator --
        let orchestrator_dispatches_total = IntCounter::new(
            "heartbit_orchestrator_dispatches_total",
            "Total orchestrator sub-agent dispatches",
        )?;
        let orchestrator_sub_completions_total = IntCounterVec::new(
            Opts::new(
                "heartbit_orchestrator_sub_completions_total",
                "Total orchestrator sub-agent completions",
            ),
            &["success"],
        )?;

        // -- Interaction --
        let approvals_requested_total = IntCounter::new(
            "heartbit_interaction_approvals_requested_total",
            "Total approval requests",
        )?;
        let approvals_decided_total = IntCounterVec::new(
            Opts::new(
                "heartbit_interaction_approvals_decided_total",
                "Total approval decisions",
            ),
            &["approved"],
        )?;

        // -- Sensor --
        let sensor_events_processed_total = IntCounterVec::new(
            Opts::new(
                "heartbit_sensor_events_processed_total",
                "Total sensor events processed",
            ),
            &["sensor", "decision"],
        )?;
        let sensor_stories_updated_total = IntCounter::new(
            "heartbit_sensor_stories_updated_total",
            "Total sensor stories updated",
        )?;

        // -- Routing --
        let routing_decisions_total = IntCounterVec::new(
            Opts::new(
                "heartbit_routing_decisions_total",
                "Total routing decisions",
            ),
            &["decision"],
        )?;
        let routing_escalations_total = IntCounter::new(
            "heartbit_routing_escalations_total",
            "Total routing escalations",
        )?;

        // -- Cascade --
        let cascade_escalations_total = IntCounterVec::new(
            Opts::new(
                "heartbit_cascade_escalations_total",
                "Total model cascade escalations",
            ),
            &["from_tier", "to_tier", "reason"],
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
        registry.register(Box::new(context_summarizations_total.clone()))?;
        registry.register(Box::new(session_prunes_total.clone()))?;
        registry.register(Box::new(session_bytes_saved_total.clone()))?;
        registry.register(Box::new(guardrail_denials_total.clone()))?;

        registry.register(Box::new(errors_total.clone()))?;

        registry.register(Box::new(pulse_runs_total.clone()))?;
        registry.register(Box::new(pulse_ok_total.clone()))?;
        registry.register(Box::new(pulse_action_total.clone()))?;
        registry.register(Box::new(tasks_by_source_total.clone()))?;

        registry.register(Box::new(agent_runs_started_total.clone()))?;
        registry.register(Box::new(agent_turns_total.clone()))?;
        registry.register(Box::new(agent_runs_completed_total.clone()))?;

        registry.register(Box::new(orchestrator_dispatches_total.clone()))?;
        registry.register(Box::new(orchestrator_sub_completions_total.clone()))?;

        registry.register(Box::new(approvals_requested_total.clone()))?;
        registry.register(Box::new(approvals_decided_total.clone()))?;

        registry.register(Box::new(sensor_events_processed_total.clone()))?;
        registry.register(Box::new(sensor_stories_updated_total.clone()))?;

        registry.register(Box::new(routing_decisions_total.clone()))?;
        registry.register(Box::new(routing_escalations_total.clone()))?;

        registry.register(Box::new(cascade_escalations_total.clone()))?;

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
            context_summarizations_total,
            session_prunes_total,
            session_bytes_saved_total,
            guardrail_denials_total,
            errors_total,
            pulse_runs_total,
            pulse_ok_total,
            pulse_action_total,
            tasks_by_source_total,
            agent_runs_started_total,
            agent_turns_total,
            agent_runs_completed_total,
            orchestrator_dispatches_total,
            orchestrator_sub_completions_total,
            approvals_requested_total,
            approvals_decided_total,
            sensor_events_processed_total,
            sensor_stories_updated_total,
            routing_decisions_total,
            routing_escalations_total,
            cascade_escalations_total,
        })
    }

    /// Process an `AgentEvent` and update the relevant metrics.
    pub fn record_event(&self, event: &AgentEvent) {
        match event {
            AgentEvent::RunStarted { agent, .. } => {
                self.agent_runs_started_total
                    .with_label_values(&[agent])
                    .inc();
            }
            AgentEvent::TurnStarted { agent, .. } => {
                self.agent_turns_total.with_label_values(&[agent]).inc();
            }
            AgentEvent::LlmResponse {
                agent,
                usage,
                latency_ms,
                model,
                time_to_first_token_ms,
                ..
            } => {
                self.llm_calls_total.with_label_values(&[agent]).inc();
                self.llm_call_duration_seconds
                    .with_label_values(&[agent])
                    .observe(*latency_ms as f64 / 1000.0);
                if *time_to_first_token_ms > 0 {
                    self.llm_ttft_seconds
                        .with_label_values(&[agent])
                        .observe(*time_to_first_token_ms as f64 / 1000.0);
                }
                self.llm_tokens_input_total
                    .with_label_values(&[agent])
                    .inc_by(u64::from(usage.input_tokens));
                self.llm_tokens_output_total
                    .with_label_values(&[agent])
                    .inc_by(u64::from(usage.output_tokens));
                self.llm_tokens_cache_read_total
                    .with_label_values(&[agent])
                    .inc_by(u64::from(usage.cache_read_input_tokens));
                self.llm_tokens_cache_creation_total
                    .with_label_values(&[agent])
                    .inc_by(u64::from(usage.cache_creation_input_tokens));
                if let Some(model_name) = model
                    && let Some(cost) = estimate_cost(model_name, usage)
                {
                    self.llm_cost_usd_total
                        .with_label_values(&[agent])
                        .inc_by(cost);
                }
            }
            AgentEvent::ToolCallStarted { .. } => {
                // No metric — we track completion + duration which is more useful.
            }
            AgentEvent::ToolCallCompleted {
                agent,
                tool_name,
                is_error,
                duration_ms,
                ..
            } => {
                self.tool_calls_total
                    .with_label_values(&[agent, tool_name])
                    .inc();
                self.tool_duration_seconds
                    .with_label_values(&[agent, tool_name])
                    .observe(*duration_ms as f64 / 1000.0);
                if *is_error {
                    self.tool_errors_total
                        .with_label_values(&[agent, tool_name])
                        .inc();
                }
            }
            AgentEvent::ApprovalRequested { .. } => {
                self.approvals_requested_total.inc();
            }
            AgentEvent::ApprovalDecision { approved, .. } => {
                let label = if *approved { "true" } else { "false" };
                self.approvals_decided_total
                    .with_label_values(&[label])
                    .inc();
            }
            AgentEvent::SubAgentsDispatched { .. } => {
                self.orchestrator_dispatches_total.inc();
            }
            AgentEvent::SubAgentCompleted { success, .. } => {
                let label = if *success { "true" } else { "false" };
                self.orchestrator_sub_completions_total
                    .with_label_values(&[label])
                    .inc();
            }
            AgentEvent::ContextSummarized { .. } => {
                self.context_summarizations_total.inc();
            }
            AgentEvent::RunCompleted { agent, .. } => {
                self.agent_runs_completed_total
                    .with_label_values(&[agent])
                    .inc();
            }
            AgentEvent::GuardrailDenied { hook, .. } => {
                self.guardrail_denials_total
                    .with_label_values(&[hook])
                    .inc();
            }
            AgentEvent::RunFailed { .. } => {
                self.errors_total.with_label_values(&["run_failed"]).inc();
            }
            AgentEvent::RetryAttempt { .. } => {
                self.retry_attempts_total.inc();
            }
            AgentEvent::DoomLoopDetected { .. } => {
                self.doom_loops_detected_total.inc();
            }
            AgentEvent::SessionPruned { bytes_saved, .. } => {
                self.session_prunes_total.inc();
                self.session_bytes_saved_total.inc_by(*bytes_saved as u64);
            }
            AgentEvent::AutoCompactionTriggered { success, .. } => {
                if *success {
                    self.context_compactions_total.inc();
                }
            }
            AgentEvent::SensorEventProcessed {
                sensor_name,
                decision,
                ..
            } => {
                self.sensor_events_processed_total
                    .with_label_values(&[sensor_name, decision])
                    .inc();
            }
            AgentEvent::StoryUpdated { .. } => {
                self.sensor_stories_updated_total.inc();
            }
            AgentEvent::TaskRouted {
                decision,
                escalated,
                ..
            } => {
                self.routing_decisions_total
                    .with_label_values(&[decision])
                    .inc();
                if *escalated {
                    self.routing_escalations_total.inc();
                }
            }
            AgentEvent::ModelEscalated {
                from_tier,
                to_tier,
                reason,
                ..
            } => {
                self.cascade_escalations_total
                    .with_label_values(&[from_tier, to_tier, reason])
                    .inc();
            }
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

    /// Return a reference to the LLM cost counter vec for external additions.
    pub fn cost_usd(&self) -> &CounterVec {
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
        // Scalar counters should be present and at 0.
        assert!(text.contains("heartbit_daemon_tasks_submitted_total 0"));
        assert!(text.contains("heartbit_daemon_tasks_completed_total 0"));
        assert!(text.contains("heartbit_daemon_tasks_failed_total 0"));
        assert!(text.contains("heartbit_daemon_tasks_cancelled_total 0"));
        assert!(text.contains("heartbit_daemon_tasks_active 0"));
        assert!(text.contains("heartbit_reliability_retry_attempts_total 0"));
        assert!(text.contains("heartbit_reliability_doom_loops_detected_total 0"));
        assert!(text.contains("heartbit_reliability_context_compactions_total 0"));
        assert!(text.contains("heartbit_reliability_context_summarizations_total 0"));
        assert!(text.contains("heartbit_reliability_session_prunes_total 0"));
        assert!(text.contains("heartbit_reliability_session_bytes_saved_total 0"));
        assert!(text.contains("heartbit_orchestrator_dispatches_total 0"));
        assert!(text.contains("heartbit_interaction_approvals_requested_total 0"));
        assert!(text.contains("heartbit_sensor_stories_updated_total 0"));
        assert!(text.contains("heartbit_routing_escalations_total 0"));
        assert!(text.contains("heartbit_pulse_runs_total 0"));
        assert!(text.contains("heartbit_pulse_ok_total 0"));
        assert!(text.contains("heartbit_pulse_action_total 0"));
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
        assert!(
            text.contains(r#"heartbit_llm_calls_total{agent="test"} 1"#),
            "text: {text}"
        );
        assert!(
            text.contains(r#"heartbit_llm_tokens_input_total{agent="test"} 100"#),
            "text: {text}"
        );
        assert!(
            text.contains(r#"heartbit_llm_tokens_output_total{agent="test"} 50"#),
            "text: {text}"
        );
        assert!(
            text.contains(r#"heartbit_llm_tokens_cache_read_total{agent="test"} 20"#),
            "text: {text}"
        );
        assert!(
            text.contains(r#"heartbit_llm_tokens_cache_creation_total{agent="test"} 10"#),
            "text: {text}"
        );
        // Cost should be non-zero for a known model
        assert!(
            m.cost_usd().with_label_values(&["test"]).get() > 0.0,
            "cost should be positive"
        );
    }

    #[test]
    fn record_llm_response_agent_label() {
        let m = DaemonMetrics::new().unwrap();
        let make_event = |agent: &str| AgentEvent::LlmResponse {
            agent: agent.into(),
            turn: 1,
            usage: TokenUsage {
                input_tokens: 10,
                output_tokens: 5,
                cache_read_input_tokens: 0,
                cache_creation_input_tokens: 0,
                reasoning_tokens: 0,
            },
            stop_reason: StopReason::EndTurn,
            tool_call_count: 0,
            text: String::new(),
            latency_ms: 100,
            model: None,
            time_to_first_token_ms: 0,
        };
        m.record_event(&make_event("alpha"));
        m.record_event(&make_event("alpha"));
        m.record_event(&make_event("beta"));

        let text = m.encode().unwrap();
        assert!(
            text.contains(r#"heartbit_llm_calls_total{agent="alpha"} 2"#),
            "text: {text}"
        );
        assert!(
            text.contains(r#"heartbit_llm_calls_total{agent="beta"} 1"#),
            "text: {text}"
        );
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
            text.contains(r#"heartbit_tool_calls_total{agent="test",tool_name="web_search"} 1"#),
            "text: {text}"
        );
        // No error increment
        assert!(
            !text.contains(r#"heartbit_tool_errors_total{agent="test",tool_name="web_search"}"#),
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
            text.contains(r#"heartbit_tool_calls_total{agent="test",tool_name="bash"} 1"#),
            "text: {text}"
        );
        assert!(
            text.contains(r#"heartbit_tool_errors_total{agent="test",tool_name="bash"} 1"#),
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
    fn record_run_started() {
        let m = DaemonMetrics::new().unwrap();
        let event = AgentEvent::RunStarted {
            agent: "researcher".into(),
            task: "find info".into(),
        };
        m.record_event(&event);

        let text = m.encode().unwrap();
        assert!(
            text.contains(r#"heartbit_agent_runs_started_total{agent="researcher"} 1"#),
            "text: {text}"
        );
    }

    #[test]
    fn record_turn_started() {
        let m = DaemonMetrics::new().unwrap();
        let event = AgentEvent::TurnStarted {
            agent: "coder".into(),
            turn: 3,
            max_turns: 10,
        };
        m.record_event(&event);
        m.record_event(&event);

        let text = m.encode().unwrap();
        assert!(
            text.contains(r#"heartbit_agent_turns_total{agent="coder"} 2"#),
            "text: {text}"
        );
    }

    #[test]
    fn record_run_completed() {
        let m = DaemonMetrics::new().unwrap();
        let event = AgentEvent::RunCompleted {
            agent: "writer".into(),
            total_usage: TokenUsage::default(),
            tool_calls_made: 5,
        };
        m.record_event(&event);

        let text = m.encode().unwrap();
        assert!(
            text.contains(r#"heartbit_agent_runs_completed_total{agent="writer"} 1"#),
            "text: {text}"
        );
    }

    #[test]
    fn record_sub_agents_dispatched() {
        let m = DaemonMetrics::new().unwrap();
        let event = AgentEvent::SubAgentsDispatched {
            agent: "orchestrator".into(),
            agents: vec!["a".into(), "b".into()],
        };
        m.record_event(&event);
        m.record_event(&event);

        let text = m.encode().unwrap();
        assert!(
            text.contains("heartbit_orchestrator_dispatches_total 2"),
            "text: {text}"
        );
    }

    #[test]
    fn record_sub_agent_completed() {
        let m = DaemonMetrics::new().unwrap();
        let success_event = AgentEvent::SubAgentCompleted {
            agent: "a".into(),
            success: true,
            usage: TokenUsage::default(),
        };
        let failure_event = AgentEvent::SubAgentCompleted {
            agent: "b".into(),
            success: false,
            usage: TokenUsage::default(),
        };
        m.record_event(&success_event);
        m.record_event(&success_event);
        m.record_event(&failure_event);

        let text = m.encode().unwrap();
        assert!(
            text.contains(r#"heartbit_orchestrator_sub_completions_total{success="true"} 2"#),
            "text: {text}"
        );
        assert!(
            text.contains(r#"heartbit_orchestrator_sub_completions_total{success="false"} 1"#),
            "text: {text}"
        );
    }

    #[test]
    fn record_approval_requested() {
        let m = DaemonMetrics::new().unwrap();
        let event = AgentEvent::ApprovalRequested {
            agent: "a".into(),
            turn: 1,
            tool_names: vec!["bash".into()],
        };
        m.record_event(&event);

        let text = m.encode().unwrap();
        assert!(
            text.contains("heartbit_interaction_approvals_requested_total 1"),
            "text: {text}"
        );
    }

    #[test]
    fn record_approval_decision() {
        let m = DaemonMetrics::new().unwrap();
        let approved = AgentEvent::ApprovalDecision {
            agent: "a".into(),
            turn: 1,
            approved: true,
        };
        let denied = AgentEvent::ApprovalDecision {
            agent: "a".into(),
            turn: 2,
            approved: false,
        };
        m.record_event(&approved);
        m.record_event(&approved);
        m.record_event(&denied);

        let text = m.encode().unwrap();
        assert!(
            text.contains(r#"heartbit_interaction_approvals_decided_total{approved="true"} 2"#),
            "text: {text}"
        );
        assert!(
            text.contains(r#"heartbit_interaction_approvals_decided_total{approved="false"} 1"#),
            "text: {text}"
        );
    }

    #[test]
    fn record_context_summarized() {
        let m = DaemonMetrics::new().unwrap();
        let event = AgentEvent::ContextSummarized {
            agent: "a".into(),
            turn: 5,
            usage: TokenUsage::default(),
        };
        m.record_event(&event);

        let text = m.encode().unwrap();
        assert!(
            text.contains("heartbit_reliability_context_summarizations_total 1"),
            "text: {text}"
        );
    }

    #[test]
    fn record_session_pruned() {
        let m = DaemonMetrics::new().unwrap();
        let event = AgentEvent::SessionPruned {
            agent: "a".into(),
            turn: 3,
            tool_results_pruned: 5,
            bytes_saved: 12345,
            tool_results_total: 10,
        };
        m.record_event(&event);

        let text = m.encode().unwrap();
        assert!(
            text.contains("heartbit_reliability_session_prunes_total 1"),
            "text: {text}"
        );
        assert!(
            text.contains("heartbit_reliability_session_bytes_saved_total 12345"),
            "text: {text}"
        );
    }

    #[test]
    fn record_session_pruned_zero_bytes() {
        let m = DaemonMetrics::new().unwrap();
        let event = AgentEvent::SessionPruned {
            agent: "a".into(),
            turn: 1,
            tool_results_pruned: 0,
            bytes_saved: 0,
            tool_results_total: 0,
        };
        m.record_event(&event);

        let text = m.encode().unwrap();
        assert!(
            text.contains("heartbit_reliability_session_prunes_total 1"),
            "text: {text}"
        );
        assert!(
            text.contains("heartbit_reliability_session_bytes_saved_total 0"),
            "text: {text}"
        );
    }

    #[test]
    fn record_sensor_event_processed() {
        let m = DaemonMetrics::new().unwrap();
        let promote = AgentEvent::SensorEventProcessed {
            sensor_name: "tech_rss".into(),
            decision: "promote".into(),
            priority: Some("normal".into()),
            story_id: None,
        };
        let drop = AgentEvent::SensorEventProcessed {
            sensor_name: "work_email".into(),
            decision: "drop".into(),
            priority: None,
            story_id: None,
        };
        m.record_event(&promote);
        m.record_event(&promote);
        m.record_event(&drop);

        let text = m.encode().unwrap();
        assert!(
            text.contains(
                r#"heartbit_sensor_events_processed_total{decision="promote",sensor="tech_rss"} 2"#
            ),
            "text: {text}"
        );
        assert!(
            text.contains(
                r#"heartbit_sensor_events_processed_total{decision="drop",sensor="work_email"} 1"#
            ),
            "text: {text}"
        );
    }

    #[test]
    fn record_story_updated() {
        let m = DaemonMetrics::new().unwrap();
        let event = AgentEvent::StoryUpdated {
            story_id: "s1".into(),
            subject: "Rust news".into(),
            event_count: 3,
            priority: None,
        };
        m.record_event(&event);
        m.record_event(&event);

        let text = m.encode().unwrap();
        assert!(
            text.contains("heartbit_sensor_stories_updated_total 2"),
            "text: {text}"
        );
    }

    #[test]
    fn record_task_routed_single_agent() {
        let m = DaemonMetrics::new().unwrap();
        let event = AgentEvent::TaskRouted {
            decision: "single_agent".into(),
            reason: "low complexity".into(),
            selected_agent: Some("coder".into()),
            complexity_score: 0.15,
            escalated: false,
        };
        m.record_event(&event);

        let text = m.encode().unwrap();
        assert!(
            text.contains(r#"heartbit_routing_decisions_total{decision="single_agent"} 1"#),
            "text: {text}"
        );
        assert!(
            text.contains("heartbit_routing_escalations_total 0"),
            "text: {text}"
        );
    }

    #[test]
    fn record_task_routed_orchestrate_with_escalation() {
        let m = DaemonMetrics::new().unwrap();
        let event = AgentEvent::TaskRouted {
            decision: "orchestrate".into(),
            reason: "high complexity".into(),
            selected_agent: None,
            complexity_score: 0.85,
            escalated: true,
        };
        m.record_event(&event);

        let text = m.encode().unwrap();
        assert!(
            text.contains(r#"heartbit_routing_decisions_total{decision="orchestrate"} 1"#),
            "text: {text}"
        );
        assert!(
            text.contains("heartbit_routing_escalations_total 1"),
            "text: {text}"
        );
    }

    #[test]
    fn tool_call_started_is_noop() {
        let m = DaemonMetrics::new().unwrap();
        let before = m.encode().unwrap();

        let event = AgentEvent::ToolCallStarted {
            agent: "a".into(),
            tool_name: "bash".into(),
            tool_call_id: "c1".into(),
            input: "{}".into(),
        };
        m.record_event(&event);

        let after = m.encode().unwrap();
        assert_eq!(before, after, "ToolCallStarted should be a noop");
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

    #[test]
    fn all_agent_event_variants_handled() {
        // This test ensures that every AgentEvent variant is handled in record_event
        // (no panics, no compilation errors). We fire every variant and verify
        // the metrics output is non-empty.
        let m = DaemonMetrics::new().unwrap();
        let events = vec![
            AgentEvent::RunStarted {
                agent: "a".into(),
                task: "t".into(),
            },
            AgentEvent::TurnStarted {
                agent: "a".into(),
                turn: 1,
                max_turns: 10,
            },
            AgentEvent::LlmResponse {
                agent: "a".into(),
                turn: 1,
                usage: TokenUsage::default(),
                stop_reason: StopReason::EndTurn,
                tool_call_count: 0,
                text: String::new(),
                latency_ms: 100,
                model: None,
                time_to_first_token_ms: 0,
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
                duration_ms: 50,
                output: String::new(),
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
            AgentEvent::GuardrailDenied {
                agent: "a".into(),
                hook: "post_llm".into(),
                reason: "test".into(),
                tool_name: None,
            },
            AgentEvent::RunFailed {
                agent: "a".into(),
                error: "test".into(),
                partial_usage: TokenUsage::default(),
            },
            AgentEvent::RetryAttempt {
                agent: "a".into(),
                attempt: 1,
                max_retries: 3,
                delay_ms: 500,
                error_class: "rate_limited".into(),
            },
            AgentEvent::DoomLoopDetected {
                agent: "a".into(),
                turn: 4,
                consecutive_count: 3,
                tool_names: vec!["bash".into()],
            },
            AgentEvent::SessionPruned {
                agent: "a".into(),
                turn: 3,
                tool_results_pruned: 2,
                bytes_saved: 1000,
                tool_results_total: 4,
            },
            AgentEvent::AutoCompactionTriggered {
                agent: "a".into(),
                turn: 2,
                success: true,
                usage: TokenUsage::default(),
            },
            AgentEvent::SensorEventProcessed {
                sensor_name: "rss".into(),
                decision: "promote".into(),
                priority: None,
                story_id: None,
            },
            AgentEvent::StoryUpdated {
                story_id: "s1".into(),
                subject: "test".into(),
                event_count: 1,
                priority: None,
            },
            AgentEvent::TaskRouted {
                decision: "single_agent".into(),
                reason: "test".into(),
                selected_agent: Some("a".into()),
                complexity_score: 0.1,
                escalated: false,
            },
            AgentEvent::ModelEscalated {
                agent: "a".into(),
                from_tier: "haiku".into(),
                to_tier: "sonnet".into(),
                reason: "gate_rejected".into(),
            },
        ];
        for event in &events {
            m.record_event(event);
        }

        let text = m.encode().unwrap();
        assert!(!text.is_empty());
    }

    #[test]
    fn record_cascade_escalation() {
        let m = DaemonMetrics::new().unwrap();
        let event = AgentEvent::ModelEscalated {
            agent: "a".into(),
            from_tier: "haiku".into(),
            to_tier: "sonnet".into(),
            reason: "gate_rejected".into(),
        };
        m.record_event(&event);
        m.record_event(&event);

        let text = m.encode().unwrap();
        assert!(
            text.contains(
                r#"heartbit_cascade_escalations_total{from_tier="haiku",reason="gate_rejected",to_tier="sonnet"} 2"#
            ),
            "text: {text}"
        );
    }
}
