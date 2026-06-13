//! Behavioral monitoring guardrail.
//!
//! Tracks tool call patterns over a sliding window and fires rules when
//! anomalous behavior is detected (frequency spikes, suspicious sequences,
//! denial storms). Stateful — uses `set_turn` for turn context and records
//! calls in `pre_tool` (denied) and `post_tool` (allowed).

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use crate::agent::guardrail::{GuardAction, Guardrail};
use crate::error::Error;
use crate::llm::types::ToolCall;
use crate::tool::ToolOutput;

struct ToolCallRecord {
    tool_name: String,
    turn: usize,
    timestamp: Instant,
    was_denied: bool,
}

/// A behavioral rule evaluated against the sliding window.
pub enum BehaviorRule {
    /// Triggers when more than `max_count` calls to tools matching `tool_pattern`
    /// occur within `window`.
    FrequencyLimit {
        /// Wildcard or substring matched against tool names.
        tool_pattern: String,
        /// Threshold (exclusive); strictly more than this trips the rule.
        max_count: usize,
        /// Sliding-window duration over which `max_count` is evaluated.
        window: Duration,
    },
    /// Triggers when a call matching `first` is followed by a call matching
    /// `then` within `within_turns` turns.
    SuspiciousSequence {
        /// Pattern matching the first tool call in the suspicious sequence.
        first: String,
        /// Pattern matching the follow-up tool call.
        then: String,
        /// Maximum turn gap between `first` and `then`.
        within_turns: usize,
    },
    /// Triggers `Kill` when more than `max_denied` denied calls occur within
    /// `window`.
    DenialSpike {
        /// Threshold (exclusive) of denied-tool-call records within the window.
        max_denied: usize,
        /// Sliding-window duration.
        window: Duration,
    },
}

/// Behavioral monitoring guardrail that tracks tool call patterns over a
/// sliding window and enforces [`BehaviorRule`]s.
pub struct BehavioralMonitorGuardrail {
    window: Mutex<VecDeque<ToolCallRecord>>,
    rules: Vec<BehaviorRule>,
    window_size: usize,
    window_ttl: Duration,
    current_turn: AtomicUsize,
}

impl BehavioralMonitorGuardrail {
    /// Returns a new builder with default settings.
    pub fn builder() -> BehavioralMonitorGuardrailBuilder {
        BehavioralMonitorGuardrailBuilder::new()
    }
}

/// Simple glob-like pattern matching: `*` at end matches any suffix,
/// bare `*` matches everything, otherwise exact match.
fn pattern_matches(pattern: &str, name: &str) -> bool {
    if pattern == "*" {
        true
    } else if let Some(prefix) = pattern.strip_suffix('*') {
        name.starts_with(prefix)
    } else {
        pattern == name
    }
}

impl BehavioralMonitorGuardrail {
    /// Evict entries older than `window_ttl` and trim to `window_size`.
    fn evict(&self, window: &mut VecDeque<ToolCallRecord>) {
        // `checked_sub` guards the case where `window_ttl` exceeds the host's
        // monotonic uptime (fresh boot): `Instant - Duration` would otherwise
        // panic on underflow. When it underflows, no entry can be older than the
        // window, so skip TTL eviction (the size trim below still runs).
        if let Some(cutoff) = Instant::now().checked_sub(self.window_ttl) {
            while window.front().is_some_and(|r| r.timestamp < cutoff) {
                window.pop_front();
            }
        }
        while window.len() > self.window_size {
            window.pop_front();
        }
    }

    /// Evaluate all rules against the current window. Returns the first
    /// firing action, or `Allow` if nothing fires.
    fn evaluate(&self, window: &VecDeque<ToolCallRecord>, current_tool: &str) -> GuardAction {
        let now = Instant::now();
        let turn = self.current_turn.load(Ordering::Relaxed);

        for rule in &self.rules {
            match rule {
                BehaviorRule::FrequencyLimit {
                    tool_pattern,
                    max_count,
                    window: rule_window,
                } => {
                    // `checked_sub` guards underflow when `rule_window` exceeds
                    // host uptime; `None` means every record falls within the
                    // window (cutoff predates all of monotonic time so far).
                    let cutoff = now.checked_sub(*rule_window);
                    let count = window
                        .iter()
                        .filter(|r| {
                            cutoff.is_none_or(|c| r.timestamp >= c)
                                && pattern_matches(tool_pattern, &r.tool_name)
                        })
                        .count();
                    // Also count the current call if it matches
                    let total = if pattern_matches(tool_pattern, current_tool) {
                        count + 1
                    } else {
                        count
                    };
                    if total > *max_count {
                        return GuardAction::deny(format!(
                            "FrequencyLimit: {total} calls to `{tool_pattern}` exceeds limit of {max_count}"
                        ));
                    }
                }
                BehaviorRule::SuspiciousSequence {
                    first,
                    then,
                    within_turns,
                } => {
                    if pattern_matches(then, current_tool) {
                        let found = window.iter().rev().any(|r| {
                            pattern_matches(first, &r.tool_name)
                                && turn.saturating_sub(r.turn) <= *within_turns
                        });
                        if found {
                            return GuardAction::deny(format!(
                                "SuspiciousSequence: `{first}` followed by `{then}` within {within_turns} turns"
                            ));
                        }
                    }
                }
                BehaviorRule::DenialSpike {
                    max_denied,
                    window: rule_window,
                } => {
                    let cutoff = now.checked_sub(*rule_window);
                    let denied_count = window
                        .iter()
                        .filter(|r| r.was_denied && cutoff.is_none_or(|c| r.timestamp >= c))
                        .count();
                    if denied_count > *max_denied {
                        return GuardAction::kill(format!(
                            "DenialSpike: {denied_count} denied calls exceeds limit of {max_denied}"
                        ));
                    }
                }
            }
        }

        GuardAction::Allow
    }
}

impl Guardrail for BehavioralMonitorGuardrail {
    fn name(&self) -> &str {
        "behavioral_monitor"
    }

    fn set_turn(&self, turn: usize) {
        self.current_turn.store(turn, Ordering::Relaxed);
    }

    fn pre_tool(
        &self,
        call: &ToolCall,
    ) -> Pin<Box<dyn Future<Output = Result<GuardAction, Error>> + Send + '_>> {
        let name = call.name.clone();
        Box::pin(async move {
            let mut window = self
                .window
                .lock()
                .map_err(|e| Error::Guardrail(format!("behavioral monitor lock poisoned: {e}")))?;
            self.evict(&mut window);

            let action = self.evaluate(&window, &name);

            if action.is_denied() {
                // Record as denied
                window.push_back(ToolCallRecord {
                    tool_name: name,
                    turn: self.current_turn.load(Ordering::Relaxed),
                    timestamp: Instant::now(),
                    was_denied: true,
                });
            }

            Ok(action)
        })
    }

    fn post_tool(
        &self,
        call: &ToolCall,
        _output: &mut ToolOutput,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>> {
        let name = call.name.clone();
        Box::pin(async move {
            let mut window = self
                .window
                .lock()
                .map_err(|e| Error::Guardrail(format!("behavioral monitor lock poisoned: {e}")))?;

            window.push_back(ToolCallRecord {
                tool_name: name,
                turn: self.current_turn.load(Ordering::Relaxed),
                timestamp: Instant::now(),
                was_denied: false,
            });

            self.evict(&mut window);

            Ok(())
        })
    }
}

/// Builder for [`BehavioralMonitorGuardrail`].
pub struct BehavioralMonitorGuardrailBuilder {
    rules: Vec<BehaviorRule>,
    window_size: usize,
    window_ttl: Duration,
}

impl BehavioralMonitorGuardrailBuilder {
    fn new() -> Self {
        Self {
            rules: Vec::new(),
            window_size: 200,
            window_ttl: Duration::from_secs(30 * 60),
        }
    }

    /// Add a behavior rule.
    pub fn rule(mut self, rule: BehaviorRule) -> Self {
        self.rules.push(rule);
        self
    }

    /// Maximum number of entries kept in the sliding window (default: 200).
    pub fn window_size(mut self, size: usize) -> Self {
        self.window_size = size;
        self
    }

    /// Time-to-live for window entries (default: 30 minutes).
    pub fn window_ttl(mut self, ttl: Duration) -> Self {
        self.window_ttl = ttl;
        self
    }

    /// Build the guardrail.
    pub fn build(self) -> BehavioralMonitorGuardrail {
        BehavioralMonitorGuardrail {
            window: Mutex::new(VecDeque::with_capacity(self.window_size)),
            rules: self.rules,
            window_size: self.window_size,
            window_ttl: self.window_ttl,
            current_turn: AtomicUsize::new(0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn test_call(name: &str) -> ToolCall {
        ToolCall {
            id: "c1".into(),
            name: name.into(),
            input: json!({}),
        }
    }

    #[test]
    fn pattern_matches_exact() {
        assert!(pattern_matches("bash", "bash"));
        assert!(!pattern_matches("bash", "read"));
    }

    #[test]
    fn pattern_matches_wildcard() {
        assert!(pattern_matches("*", "anything"));
        assert!(pattern_matches("*", ""));
    }

    #[test]
    fn pattern_matches_prefix_glob() {
        assert!(pattern_matches("gmail_*", "gmail_send"));
        assert!(pattern_matches("gmail_*", "gmail_"));
        assert!(!pattern_matches("gmail_*", "slack_send"));
    }

    #[tokio::test]
    async fn frequency_limit_triggers() {
        let g = BehavioralMonitorGuardrail::builder()
            .rule(BehaviorRule::FrequencyLimit {
                tool_pattern: "bash".into(),
                max_count: 3,
                window: Duration::from_secs(60),
            })
            .build();

        // Record 3 successful calls via post_tool
        for _ in 0..3 {
            let call = test_call("bash");
            let mut output = ToolOutput::success("ok".to_string());
            g.post_tool(&call, &mut output).await.unwrap();
        }

        // 4th call should be denied (3 in window + 1 current = 4 > 3)
        let action = g.pre_tool(&test_call("bash")).await.unwrap();
        assert!(action.is_denied());
        assert!(!action.is_killed());
    }

    #[tokio::test]
    async fn frequency_limit_allows_under_threshold() {
        let g = BehavioralMonitorGuardrail::builder()
            .rule(BehaviorRule::FrequencyLimit {
                tool_pattern: "bash".into(),
                max_count: 5,
                window: Duration::from_secs(60),
            })
            .build();

        for _ in 0..2 {
            let call = test_call("bash");
            let mut output = ToolOutput::success("ok".to_string());
            g.post_tool(&call, &mut output).await.unwrap();
        }

        let action = g.pre_tool(&test_call("bash")).await.unwrap();
        assert_eq!(action, GuardAction::Allow);
    }

    #[tokio::test]
    async fn suspicious_sequence_detects() {
        let g = BehavioralMonitorGuardrail::builder()
            .rule(BehaviorRule::SuspiciousSequence {
                first: "read_secrets".into(),
                then: "send_email".into(),
                within_turns: 3,
            })
            .build();

        g.set_turn(1);
        let call = test_call("read_secrets");
        let mut output = ToolOutput::success("ok".to_string());
        g.post_tool(&call, &mut output).await.unwrap();

        g.set_turn(2);
        let action = g.pre_tool(&test_call("send_email")).await.unwrap();
        assert!(action.is_denied());
    }

    #[tokio::test]
    async fn suspicious_sequence_outside_turn_window() {
        let g = BehavioralMonitorGuardrail::builder()
            .rule(BehaviorRule::SuspiciousSequence {
                first: "read_secrets".into(),
                then: "send_email".into(),
                within_turns: 2,
            })
            .build();

        g.set_turn(1);
        let call = test_call("read_secrets");
        let mut output = ToolOutput::success("ok".to_string());
        g.post_tool(&call, &mut output).await.unwrap();

        // Turn 10 is more than 2 turns away from turn 1
        g.set_turn(10);
        let action = g.pre_tool(&test_call("send_email")).await.unwrap();
        assert_eq!(action, GuardAction::Allow);
    }

    #[tokio::test]
    async fn denial_spike_triggers_kill() {
        let g = BehavioralMonitorGuardrail::builder()
            .rule(BehaviorRule::DenialSpike {
                max_denied: 2,
                window: Duration::from_secs(60),
            })
            // Also add a rule that will cause denials
            .rule(BehaviorRule::FrequencyLimit {
                tool_pattern: "bash".into(),
                max_count: 0,
                window: Duration::from_secs(60),
            })
            .build();

        // Each pre_tool call to bash will be denied by FrequencyLimit and
        // recorded with was_denied=true
        for _ in 0..3 {
            let _ = g.pre_tool(&test_call("bash")).await.unwrap();
        }

        // Now DenialSpike should fire Kill (3 > 2 denied calls)
        let action = g.pre_tool(&test_call("read")).await.unwrap();
        assert!(action.is_killed());
    }

    #[tokio::test]
    async fn window_ttl_evicts_old_entries() {
        let g = BehavioralMonitorGuardrail::builder()
            .rule(BehaviorRule::FrequencyLimit {
                tool_pattern: "bash".into(),
                max_count: 2,
                window: Duration::from_secs(60),
            })
            .window_ttl(Duration::from_millis(1))
            .build();

        // Record 2 calls
        for _ in 0..2 {
            let call = test_call("bash");
            let mut output = ToolOutput::success("ok".to_string());
            g.post_tool(&call, &mut output).await.unwrap();
        }

        // Wait for TTL expiry
        std::thread::sleep(Duration::from_millis(5));

        // Should allow because old entries are evicted
        let action = g.pre_tool(&test_call("bash")).await.unwrap();
        assert_eq!(action, GuardAction::Allow);
    }

    #[tokio::test]
    async fn huge_window_does_not_panic_on_low_uptime() {
        // Regression: `Instant::now() - window` underflows and panics when the
        // configured window exceeds the host's monotonic uptime (fresh microVM /
        // Firecracker / just-booted CI runner / freshly-rebooted container).
        // An astronomically large window forces the underflow path deterministically
        // regardless of actual uptime.
        let huge = Duration::from_secs(u64::MAX);
        let g = BehavioralMonitorGuardrail::builder()
            .rule(BehaviorRule::FrequencyLimit {
                tool_pattern: "bash".into(),
                max_count: 2,
                window: huge,
            })
            .rule(BehaviorRule::DenialSpike {
                max_denied: 2,
                window: huge,
            })
            .window_ttl(huge)
            .build();

        // post_tool triggers evict() (window_ttl underflow);
        // pre_tool triggers evaluate() FrequencyLimit + DenialSpike (per-rule
        // window underflow). Neither must panic.
        let call = test_call("bash");
        let mut output = ToolOutput::success("ok".to_string());
        g.post_tool(&call, &mut output).await.unwrap();
        let action = g.pre_tool(&test_call("bash")).await.unwrap();
        assert_eq!(action, GuardAction::Allow);
    }

    #[tokio::test]
    async fn set_turn_updates_context() {
        let g = BehavioralMonitorGuardrail::builder().build();
        assert_eq!(g.current_turn.load(Ordering::Relaxed), 0);
        g.set_turn(42);
        assert_eq!(g.current_turn.load(Ordering::Relaxed), 42);
    }

    #[tokio::test]
    async fn clean_traffic_passes() {
        let g = BehavioralMonitorGuardrail::builder()
            .rule(BehaviorRule::FrequencyLimit {
                tool_pattern: "bash".into(),
                max_count: 10,
                window: Duration::from_secs(60),
            })
            .rule(BehaviorRule::SuspiciousSequence {
                first: "read_secrets".into(),
                then: "send_email".into(),
                within_turns: 3,
            })
            .rule(BehaviorRule::DenialSpike {
                max_denied: 5,
                window: Duration::from_secs(60),
            })
            .build();

        // Normal tool calls should all pass
        for tool in &["read", "write", "bash", "list"] {
            let action = g.pre_tool(&test_call(tool)).await.unwrap();
            assert_eq!(action, GuardAction::Allow);
            let mut output = ToolOutput::success("ok".to_string());
            g.post_tool(&test_call(tool), &mut output).await.unwrap();
        }
    }

    #[test]
    fn builder_defaults() {
        let g = BehavioralMonitorGuardrail::builder().build();
        assert_eq!(g.window_size, 200);
        assert_eq!(g.window_ttl, Duration::from_secs(30 * 60));
        assert!(g.rules.is_empty());
    }

    #[tokio::test]
    async fn window_size_limits_entries() {
        let g = BehavioralMonitorGuardrail::builder()
            .window_size(3)
            .rule(BehaviorRule::FrequencyLimit {
                tool_pattern: "*".into(),
                max_count: 5,
                window: Duration::from_secs(60),
            })
            .build();

        // Record 5 calls — window should cap at 3
        for _ in 0..5 {
            let call = test_call("read");
            let mut output = ToolOutput::success("ok".to_string());
            g.post_tool(&call, &mut output).await.unwrap();
        }

        let window = g.window.lock().unwrap();
        assert_eq!(window.len(), 3);
    }

    #[test]
    fn meta_name() {
        let g = BehavioralMonitorGuardrail::builder().build();
        assert_eq!(g.name(), "behavioral_monitor");
    }

    #[tokio::test]
    async fn frequency_limit_with_glob_pattern() {
        let g = BehavioralMonitorGuardrail::builder()
            .rule(BehaviorRule::FrequencyLimit {
                tool_pattern: "gmail_*".into(),
                max_count: 2,
                window: Duration::from_secs(60),
            })
            .build();

        let mut output = ToolOutput::success("ok".to_string());
        g.post_tool(&test_call("gmail_send"), &mut output)
            .await
            .unwrap();
        g.post_tool(&test_call("gmail_draft"), &mut output)
            .await
            .unwrap();

        // 3rd gmail call should be denied (2 in window + 1 current = 3 > 2)
        let action = g.pre_tool(&test_call("gmail_read")).await.unwrap();
        assert!(action.is_denied());
    }

    #[tokio::test]
    async fn non_matching_pattern_allows() {
        let g = BehavioralMonitorGuardrail::builder()
            .rule(BehaviorRule::FrequencyLimit {
                tool_pattern: "gmail_*".into(),
                max_count: 2,
                window: Duration::from_secs(60),
            })
            .build();

        let mut output = ToolOutput::success("ok".to_string());
        g.post_tool(&test_call("gmail_send"), &mut output)
            .await
            .unwrap();

        // Non-gmail should be fine even with gmail entries in window
        let action = g.pre_tool(&test_call("slack_send")).await.unwrap();
        assert_eq!(action, GuardAction::Allow);
    }
}
