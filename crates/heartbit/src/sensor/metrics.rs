use prometheus::{
    HistogramOpts, HistogramVec, IntCounterVec, IntGauge, Opts, Registry, TextEncoder,
};

/// Prometheus metrics for the sensor pipeline.
///
/// Uses a dedicated (non-global) `Registry` so multiple `SensorMetrics` instances
/// can coexist in tests without conflicting (same pattern as `DaemonMetrics`).
pub struct SensorMetrics {
    registry: Registry,

    /// Total events received per sensor.
    events_received_total: IntCounterVec,
    /// Total events promoted (forwarded for processing) per sensor.
    events_promoted_total: IntCounterVec,
    /// Total events dropped (discarded by triage) per sensor.
    events_dropped_total: IntCounterVec,
    /// Total events sent to dead-letter per sensor.
    events_dead_letter_total: IntCounterVec,
    /// Triage processing duration in seconds per sensor.
    triage_duration_seconds: HistogramVec,
    /// Number of currently active stories.
    stories_active: IntGauge,
    /// Total model routing decisions by tier.
    model_routing_total: IntCounterVec,
    /// Current token budget used.
    token_budget_used: IntGauge,
    /// Current token budget limit.
    token_budget_limit: IntGauge,
}

impl SensorMetrics {
    /// Create a new `SensorMetrics` with all instruments registered on a
    /// dedicated `Registry`.
    pub fn new() -> Result<Self, prometheus::Error> {
        let registry = Registry::new();

        let events_received_total = IntCounterVec::new(
            Opts::new(
                "heartbit_sensor_events_received_total",
                "Total sensor events received",
            ),
            &["sensor_name"],
        )?;
        let events_promoted_total = IntCounterVec::new(
            Opts::new(
                "heartbit_sensor_events_promoted_total",
                "Total sensor events promoted for processing",
            ),
            &["sensor_name"],
        )?;
        let events_dropped_total = IntCounterVec::new(
            Opts::new(
                "heartbit_sensor_events_dropped_total",
                "Total sensor events dropped by triage",
            ),
            &["sensor_name"],
        )?;
        let events_dead_letter_total = IntCounterVec::new(
            Opts::new(
                "heartbit_sensor_events_dead_letter_total",
                "Total sensor events sent to dead-letter",
            ),
            &["sensor_name"],
        )?;
        let triage_duration_seconds = HistogramVec::new(
            HistogramOpts::new(
                "heartbit_sensor_triage_duration_seconds",
                "Triage processing duration in seconds",
            )
            .buckets(vec![0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]),
            &["sensor_name"],
        )?;
        let stories_active = IntGauge::new(
            "heartbit_sensor_stories_active",
            "Number of currently active stories",
        )?;
        let model_routing_total = IntCounterVec::new(
            Opts::new(
                "heartbit_sensor_model_routing_total",
                "Total model routing decisions by tier",
            ),
            &["tier"],
        )?;
        let token_budget_used = IntGauge::new(
            "heartbit_sensor_token_budget_used",
            "Current token budget used",
        )?;
        let token_budget_limit = IntGauge::new(
            "heartbit_sensor_token_budget_limit",
            "Current token budget limit",
        )?;

        registry.register(Box::new(events_received_total.clone()))?;
        registry.register(Box::new(events_promoted_total.clone()))?;
        registry.register(Box::new(events_dropped_total.clone()))?;
        registry.register(Box::new(events_dead_letter_total.clone()))?;
        registry.register(Box::new(triage_duration_seconds.clone()))?;
        registry.register(Box::new(stories_active.clone()))?;
        registry.register(Box::new(model_routing_total.clone()))?;
        registry.register(Box::new(token_budget_used.clone()))?;
        registry.register(Box::new(token_budget_limit.clone()))?;

        Ok(Self {
            registry,
            events_received_total,
            events_promoted_total,
            events_dropped_total,
            events_dead_letter_total,
            triage_duration_seconds,
            stories_active,
            model_routing_total,
            token_budget_used,
            token_budget_limit,
        })
    }

    /// Record a sensor event received.
    pub fn record_event_received(&self, sensor_name: &str) {
        self.events_received_total
            .with_label_values(&[sensor_name])
            .inc();
    }

    /// Record a duplicate event that was dropped before triage.
    pub fn record_event_dropped(&self, sensor_name: &str) {
        self.events_dropped_total
            .with_label_values(&[sensor_name])
            .inc();
    }

    /// Record a triage decision and its duration.
    ///
    /// `decision` should be one of: `"promote"`, `"drop"`, `"dead_letter"`.
    pub fn record_triage_decision(&self, sensor_name: &str, decision: &str, duration_secs: f64) {
        match decision {
            "promote" => self
                .events_promoted_total
                .with_label_values(&[sensor_name])
                .inc(),
            "drop" => self
                .events_dropped_total
                .with_label_values(&[sensor_name])
                .inc(),
            "dead_letter" => self
                .events_dead_letter_total
                .with_label_values(&[sensor_name])
                .inc(),
            _ => {}
        }
        self.triage_duration_seconds
            .with_label_values(&[sensor_name])
            .observe(duration_secs);
    }

    /// Record a model routing decision.
    pub fn record_model_routing(&self, tier: &str) {
        self.model_routing_total.with_label_values(&[tier]).inc();
    }

    /// Set the number of currently active stories.
    pub fn set_stories_active(&self, count: i64) {
        self.stories_active.set(count);
    }

    /// Set the current token budget usage and limit.
    pub fn set_token_budget(&self, used: i64, limit: i64) {
        self.token_budget_used.set(used);
        self.token_budget_limit.set(limit);
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_creates_metrics() {
        let metrics = SensorMetrics::new();
        assert!(metrics.is_ok());
    }

    #[test]
    fn new_starts_at_zero() {
        let m = SensorMetrics::new().unwrap();
        let text = m.encode().unwrap();
        assert!(text.contains("heartbit_sensor_stories_active 0"));
        assert!(text.contains("heartbit_sensor_token_budget_used 0"));
        assert!(text.contains("heartbit_sensor_token_budget_limit 0"));
    }

    #[test]
    fn record_event_received_increments() {
        let m = SensorMetrics::new().unwrap();
        m.record_event_received("work_email");
        m.record_event_received("work_email");
        m.record_event_received("tech_rss");

        let text = m.encode().unwrap();
        assert!(
            text.contains(r#"heartbit_sensor_events_received_total{sensor_name="work_email"} 2"#),
            "text: {text}"
        );
        assert!(
            text.contains(r#"heartbit_sensor_events_received_total{sensor_name="tech_rss"} 1"#),
            "text: {text}"
        );
    }

    #[test]
    fn record_triage_decision_promote() {
        let m = SensorMetrics::new().unwrap();
        m.record_triage_decision("work_email", "promote", 0.05);

        let text = m.encode().unwrap();
        assert!(
            text.contains(r#"heartbit_sensor_events_promoted_total{sensor_name="work_email"} 1"#),
            "text: {text}"
        );
    }

    #[test]
    fn record_triage_decision_drop() {
        let m = SensorMetrics::new().unwrap();
        m.record_triage_decision("spam_filter", "drop", 0.01);

        let text = m.encode().unwrap();
        assert!(
            text.contains(r#"heartbit_sensor_events_dropped_total{sensor_name="spam_filter"} 1"#),
            "text: {text}"
        );
    }

    #[test]
    fn record_triage_decision_dead_letter() {
        let m = SensorMetrics::new().unwrap();
        m.record_triage_decision("broken_sensor", "dead_letter", 0.5);

        let text = m.encode().unwrap();
        assert!(
            text.contains(
                r#"heartbit_sensor_events_dead_letter_total{sensor_name="broken_sensor"} 1"#
            ),
            "text: {text}"
        );
    }

    #[test]
    fn record_model_routing() {
        let m = SensorMetrics::new().unwrap();
        m.record_model_routing("local");
        m.record_model_routing("local");
        m.record_model_routing("cloud_frontier");

        let text = m.encode().unwrap();
        assert!(
            text.contains(r#"heartbit_sensor_model_routing_total{tier="local"} 2"#),
            "text: {text}"
        );
        assert!(
            text.contains(r#"heartbit_sensor_model_routing_total{tier="cloud_frontier"} 1"#),
            "text: {text}"
        );
    }

    #[test]
    fn set_stories_active_updates_gauge() {
        let m = SensorMetrics::new().unwrap();
        m.set_stories_active(5);

        let text = m.encode().unwrap();
        assert!(
            text.contains("heartbit_sensor_stories_active 5"),
            "text: {text}"
        );

        m.set_stories_active(3);
        let text = m.encode().unwrap();
        assert!(
            text.contains("heartbit_sensor_stories_active 3"),
            "text: {text}"
        );
    }

    #[test]
    fn set_token_budget_updates_gauges() {
        let m = SensorMetrics::new().unwrap();
        m.set_token_budget(5000, 100000);

        let text = m.encode().unwrap();
        assert!(
            text.contains("heartbit_sensor_token_budget_used 5000"),
            "text: {text}"
        );
        assert!(
            text.contains("heartbit_sensor_token_budget_limit 100000"),
            "text: {text}"
        );
    }

    #[test]
    fn encode_produces_valid_text() {
        let m = SensorMetrics::new().unwrap();
        let text = m.encode().unwrap();
        assert!(!text.is_empty());
        assert!(text.contains("# HELP"));
        assert!(text.contains("# TYPE"));
    }

    #[test]
    fn record_event_dropped_increments() {
        let m = SensorMetrics::new().unwrap();
        m.record_event_dropped("work_email");
        m.record_event_dropped("work_email");

        let text = m.encode().unwrap();
        assert!(
            text.contains(r#"heartbit_sensor_events_dropped_total{sensor_name="work_email"} 2"#),
            "text: {text}"
        );
    }

    #[test]
    fn unknown_decision_type_still_records_duration() {
        let m = SensorMetrics::new().unwrap();
        // "unknown" is not promote/drop/dead_letter, so no counter increments
        // but duration should still be recorded
        m.record_triage_decision("test_sensor", "unknown", 0.1);

        let text = m.encode().unwrap();
        // Duration histogram should have one observation
        assert!(
            text.contains("heartbit_sensor_triage_duration_seconds"),
            "text: {text}"
        );
    }
}
