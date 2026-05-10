//! Periodic proactive-post scheduler. Fires `DaemonCommand::PersonaPost`
//! per configured persona on the operator's cadence (gated by
//! `active_hours`).
//!
//! See P1.6 spec §8 and Task 9 of the corresponding plan.

use std::sync::Arc;
use std::time::Duration;

use chrono::{Local, Timelike};
use tokio_util::sync::CancellationToken;

use heartbit_core::config::{ActiveHoursConfig, PersonaPostsConfig};

use super::CommandProducer;
use super::types::DaemonCommand;

/// One scheduled poster. Fires a `PersonaPost` command every
/// `interval` (gated by `active_hours` when set) via the producer.
pub struct PersonaPostScheduler {
    persona: String,
    interval: Duration,
    active_hours: Option<ActiveHoursConfig>,
    producer: Arc<dyn CommandProducer>,
    commands_topic: String,
}

impl std::fmt::Debug for PersonaPostScheduler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PersonaPostScheduler")
            .field("persona", &self.persona)
            .field("interval", &self.interval)
            .field("active_hours_set", &self.active_hours.is_some())
            .field("commands_topic", &self.commands_topic)
            .finish()
    }
}

impl PersonaPostScheduler {
    /// Construct from a config entry. The interval is clamped to ≥60s
    /// at construction time (matches the `[[daemon.persona_posts]]`
    /// validation: `post_interval_seconds < 60` is rejected at config
    /// load — this is a defensive fallback).
    pub fn new(
        cfg: &PersonaPostsConfig,
        producer: Arc<dyn CommandProducer>,
        commands_topic: &str,
    ) -> Self {
        Self {
            persona: cfg.persona.clone(),
            interval: Duration::from_secs(cfg.post_interval_seconds.max(60)),
            active_hours: cfg.active_hours.clone(),
            producer,
            commands_topic: commands_topic.into(),
        }
    }

    /// Run the scheduler loop until `cancel` fires.
    ///
    /// Each tick sleeps `interval`, then fires a
    /// `DaemonCommand::PersonaPost` when within the configured
    /// `active_hours` window (or unconditionally when no window is set).
    pub async fn run(self, cancel: CancellationToken) {
        loop {
            tokio::select! {
                _ = cancel.cancelled() => {
                    tracing::info!(persona = %self.persona, "post scheduler shutting down");
                    break;
                }
                _ = tokio::time::sleep(self.interval) => {
                    if !self.is_within_active_hours() {
                        tracing::debug!(
                            persona = %self.persona,
                            "post scheduler tick: outside active hours, skipping"
                        );
                        continue;
                    }
                    let cmd = DaemonCommand::PersonaPost {
                        persona: self.persona.clone(),
                    };
                    let payload = match serde_json::to_vec(&cmd) {
                        Ok(p) => p,
                        Err(e) => {
                            tracing::error!(error = %e, "failed to serialize PersonaPost");
                            continue;
                        }
                    };
                    let key = format!("posts:{}", self.persona);
                    if let Err(e) = self
                        .producer
                        .send_command(&self.commands_topic, &key, &payload)
                        .await
                    {
                        tracing::error!(
                            persona = %self.persona,
                            error = %e,
                            "failed to dispatch PersonaPost"
                        );
                    } else {
                        tracing::debug!(persona = %self.persona, "post scheduler fired");
                    }
                }
            }
        }
    }

    fn is_within_active_hours(&self) -> bool {
        let now = Local::now();
        let current_minutes = now.hour() * 60 + now.minute();
        Self::check_active_hours(&self.active_hours, current_minutes)
    }

    /// Pure logic for active-hours check, extracted for testability.
    /// `current_minutes` is the current local time as minutes since midnight.
    fn check_active_hours(active_hours: &Option<ActiveHoursConfig>, current_minutes: u32) -> bool {
        let Some(hours) = active_hours else {
            return true; // no restriction
        };
        let (start_h, start_m) = match hours.parse_start() {
            Ok(v) => v,
            Err(_) => return true, // malformed = no restriction
        };
        let (end_h, end_m) = match hours.parse_end() {
            Ok(v) => v,
            Err(_) => return true,
        };

        let start_minutes = start_h * 60 + start_m;
        let end_minutes = end_h * 60 + end_m;

        if start_minutes <= end_minutes {
            // Normal range: e.g. 09:00 - 22:00
            current_minutes >= start_minutes && current_minutes < end_minutes
        } else {
            // Overnight range: e.g. 22:00 - 06:00
            current_minutes >= start_minutes || current_minutes < end_minutes
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::ChannelCommandProducer;
    use super::*;

    fn cfg_with_interval(interval: u64) -> PersonaPostsConfig {
        PersonaPostsConfig {
            persona: "heartbit-ghost:x".into(),
            enabled: true,
            post_interval_seconds: interval,
            active_hours: None,
            candidates_per_draft: 3,
            post_history_store: "in_memory".into(),
            post_history_path: None,
            post_history_lookback_days: 30,
            topic_brief: None,
        }
    }

    #[tokio::test(start_paused = true)]
    async fn fires_persona_post_after_interval() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<(String, Vec<u8>)>();
        let producer: Arc<dyn CommandProducer> = Arc::new(ChannelCommandProducer { tx });
        let scheduler =
            PersonaPostScheduler::new(&cfg_with_interval(60), producer, "test.commands");
        let cancel = CancellationToken::new();
        let cancel_for_task = cancel.clone();
        let handle = tokio::spawn(scheduler.run(cancel_for_task));

        tokio::time::advance(Duration::from_secs(61)).await;
        let (msg_key, payload) = rx.recv().await.expect("scheduler should have fired");
        // The message key (routing key) is "posts:<persona>", not the topic.
        assert_eq!(msg_key, "posts:heartbit-ghost:x");
        let parsed: DaemonCommand = serde_json::from_slice(&payload).unwrap();
        assert!(
            matches!(&parsed, DaemonCommand::PersonaPost { persona } if persona == "heartbit-ghost:x")
        );

        cancel.cancel();
        let _ = handle.await;
    }

    #[tokio::test]
    async fn check_active_hours_normal_window() {
        let cfg = ActiveHoursConfig {
            start: "09:00".into(),
            end: "22:00".into(),
        };
        // 8:30 → outside (before start)
        assert!(!PersonaPostScheduler::check_active_hours(
            &Some(cfg.clone()),
            8 * 60 + 30
        ));
        // 09:00 → at start, inclusive → inside
        assert!(PersonaPostScheduler::check_active_hours(
            &Some(cfg.clone()),
            9 * 60
        ));
        // 14:00 → middle → inside
        assert!(PersonaPostScheduler::check_active_hours(
            &Some(cfg.clone()),
            14 * 60
        ));
        // 22:00 → at end, exclusive → outside
        assert!(!PersonaPostScheduler::check_active_hours(
            &Some(cfg.clone()),
            22 * 60
        ));
        // 23:00 → after end → outside
        assert!(!PersonaPostScheduler::check_active_hours(
            &Some(cfg.clone()),
            23 * 60
        ));
        // No active_hours → always allowed
        assert!(PersonaPostScheduler::check_active_hours(&None, 0));
    }

    #[tokio::test(start_paused = true)]
    async fn cancels_cleanly() {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel::<(String, Vec<u8>)>();
        let producer: Arc<dyn CommandProducer> = Arc::new(ChannelCommandProducer { tx });
        let scheduler =
            PersonaPostScheduler::new(&cfg_with_interval(3600), producer, "test.commands");
        let cancel = CancellationToken::new();
        let cancel_for_task = cancel.clone();
        let handle = tokio::spawn(scheduler.run(cancel_for_task));
        cancel.cancel();
        // Should complete promptly.
        tokio::time::timeout(Duration::from_millis(500), handle)
            .await
            .expect("scheduler should exit on cancel")
            .unwrap();
    }
}
