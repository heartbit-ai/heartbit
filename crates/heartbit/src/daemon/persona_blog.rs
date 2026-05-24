//! Periodic blog-pipeline scheduler. Fires `DaemonCommand::PersonaBlog`
//! for the configured persona on the operator's cadence (gated by
//! `active_hours`).
//!
//! Mirrors [`crate::daemon::persona_quote::PersonaQuoteScheduler`] —
//! same jitter / active-hours logic, different command and config
//! source.

use std::sync::Arc;
use std::time::Duration;

use chrono::{Local, Timelike};
use tokio_util::sync::CancellationToken;

use heartbit_core::config::{ActiveHoursConfig, PersonaBlogConfig};

use super::CommandProducer;
use super::types::DaemonCommand;

/// One scheduled blog runner. Fires a `PersonaBlog` command every
/// `interval` (gated by `active_hours` when set) via the producer.
///
/// `jitter_pct` (0..=50) randomizes each tick by `±jitter_pct/100` of
/// `interval` — a 7-day base with 10% jitter fires somewhere in
/// [~6.3d, ~7.7d] per tick. Same anti-clock-locking rationale as the
/// quote scheduler, just on a weekly cadence.
pub struct PersonaBlogScheduler {
    persona: String,
    interval: Duration,
    jitter_pct: u32,
    active_hours: Option<ActiveHoursConfig>,
    producer: Arc<dyn CommandProducer>,
    commands_topic: String,
}

impl std::fmt::Debug for PersonaBlogScheduler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PersonaBlogScheduler")
            .field("persona", &self.persona)
            .field("interval", &self.interval)
            .field("jitter_pct", &self.jitter_pct)
            .field("active_hours_set", &self.active_hours.is_some())
            .field("commands_topic", &self.commands_topic)
            .finish()
    }
}

impl PersonaBlogScheduler {
    /// Construct from a config entry. The interval is clamped to ≥60s
    /// at construction time as a defensive fallback — the `[daemon.persona_blog]`
    /// validator already rejects intervals under 3600s, but the
    /// scheduler still floors at 60s in case it's invoked from tests or
    /// a future code path that bypasses validation.
    pub fn new(
        cfg: &PersonaBlogConfig,
        producer: Arc<dyn CommandProducer>,
        commands_topic: &str,
    ) -> Self {
        Self {
            persona: cfg.persona.clone(),
            interval: Duration::from_secs(cfg.poll_interval_seconds.max(60)),
            // Clamp to a sane band — `0` is "no jitter" (deterministic tests);
            // above `50` produces wild swings that can stretch one tick into
            // days, which defeats the active-hours window.
            jitter_pct: cfg.interval_jitter_pct.min(50),
            active_hours: cfg.active_hours.clone(),
            producer,
            commands_topic: commands_topic.into(),
        }
    }

    /// Compute the next sleep duration, randomized within
    /// `±jitter_pct%` of `interval`. Returns `interval` unchanged when
    /// jitter is 0. Floor is 60s — even with extreme jitter we never
    /// hammer the blog pipeline faster than once a minute.
    fn jittered_interval(&self) -> Duration {
        if self.jitter_pct == 0 {
            return self.interval;
        }
        let base = self.interval.as_secs_f64();
        let pct = self.jitter_pct as f64 / 100.0;
        // Uniform on [-pct, +pct].
        let factor = 1.0 + (rand::random::<f64>() * 2.0 - 1.0) * pct;
        let next = (base * factor).max(60.0);
        Duration::from_secs_f64(next)
    }

    /// Run the scheduler loop until `cancel` fires.
    ///
    /// Each tick sleeps `interval`, then fires a
    /// `DaemonCommand::PersonaBlog` when within the configured
    /// `active_hours` window (or unconditionally when no window is set).
    pub async fn run(self, cancel: CancellationToken) {
        loop {
            // Re-roll each iteration so the cadence drifts over time
            // instead of staying offset-locked to boot time.
            let next = self.jittered_interval();
            tracing::debug!(
                persona = %self.persona,
                next_sleep_secs = next.as_secs(),
                "blog scheduler: sleeping until next tick"
            );
            tokio::select! {
                _ = cancel.cancelled() => {
                    tracing::info!(persona = %self.persona, "blog scheduler shutting down");
                    break;
                }
                _ = tokio::time::sleep(next) => {
                    if !self.is_within_active_hours() {
                        tracing::debug!(
                            persona = %self.persona,
                            "blog scheduler tick: outside active hours, skipping"
                        );
                        continue;
                    }
                    let cmd = DaemonCommand::PersonaBlog {
                        persona: self.persona.clone(),
                    };
                    let payload = match serde_json::to_vec(&cmd) {
                        Ok(p) => p,
                        Err(e) => {
                            tracing::error!(error = %e, "failed to serialize PersonaBlog");
                            continue;
                        }
                    };
                    let key = format!("blog:{}", self.persona);
                    if let Err(e) = self
                        .producer
                        .send_command(&self.commands_topic, &key, &payload)
                        .await
                    {
                        tracing::error!(
                            persona = %self.persona,
                            error = %e,
                            "failed to dispatch PersonaBlog"
                        );
                    } else {
                        tracing::debug!(persona = %self.persona, "blog scheduler fired");
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

    fn cfg_with_interval(interval: u64) -> PersonaBlogConfig {
        PersonaBlogConfig {
            persona: "heartbit-ghost:x".into(),
            enabled: true,
            poll_interval_seconds: interval,
            // Deterministic timing for tests — disable jitter unless the
            // test exercises it explicitly.
            interval_jitter_pct: 0,
            active_hours: None,
            posts_dir: "blog-site/posts".into(),
            out_dir: "blog-site/public".into(),
            seed_lookback_days: 7,
            candidates_per_draft: 2,
            site_url: "https://pascal.heartbit.ai".into(),
            site_title: "pascal.heartbit.ai".into(),
            writer_provider: None,
            deploy_command: None,
            x_announce: None,
            github_readme: None,
        }
    }

    #[tokio::test(start_paused = true)]
    async fn fires_persona_blog_after_interval() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<(String, Vec<u8>)>();
        let producer: Arc<dyn CommandProducer> = Arc::new(ChannelCommandProducer { tx });
        let scheduler =
            PersonaBlogScheduler::new(&cfg_with_interval(60), producer, "test.commands");
        let cancel = CancellationToken::new();
        let cancel_for_task = cancel.clone();
        let handle = tokio::spawn(scheduler.run(cancel_for_task));

        tokio::time::advance(Duration::from_secs(61)).await;
        let (msg_key, payload) = rx.recv().await.expect("scheduler should have fired");
        // The message key (routing key) is "blog:<persona>", not the topic.
        assert_eq!(msg_key, "blog:heartbit-ghost:x");
        let parsed: DaemonCommand = serde_json::from_slice(&payload).unwrap();
        assert!(
            matches!(&parsed, DaemonCommand::PersonaBlog { persona } if persona == "heartbit-ghost:x")
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
        assert!(!PersonaBlogScheduler::check_active_hours(
            &Some(cfg.clone()),
            8 * 60 + 30
        ));
        assert!(PersonaBlogScheduler::check_active_hours(
            &Some(cfg.clone()),
            9 * 60
        ));
        assert!(PersonaBlogScheduler::check_active_hours(
            &Some(cfg.clone()),
            14 * 60
        ));
        assert!(!PersonaBlogScheduler::check_active_hours(
            &Some(cfg.clone()),
            22 * 60
        ));
        assert!(PersonaBlogScheduler::check_active_hours(&None, 0));
    }

    /// Jitter=0 returns the exact base interval — used by tests that need
    /// deterministic timing.
    #[test]
    fn jittered_interval_is_deterministic_when_pct_is_zero() {
        let cfg = cfg_with_interval(604_800);
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel::<(String, Vec<u8>)>();
        let producer: Arc<dyn CommandProducer> = Arc::new(ChannelCommandProducer { tx });
        let scheduler = PersonaBlogScheduler::new(&cfg, producer, "test");
        for _ in 0..20 {
            assert_eq!(scheduler.jittered_interval().as_secs(), 604_800);
        }
    }

    /// Jitter > 50 is clamped at construction time — protects against
    /// pathological config where one tick could stretch into weeks.
    #[test]
    fn jitter_pct_is_clamped_at_50() {
        let mut cfg = cfg_with_interval(3600);
        cfg.interval_jitter_pct = 200;
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel::<(String, Vec<u8>)>();
        let producer: Arc<dyn CommandProducer> = Arc::new(ChannelCommandProducer { tx });
        let scheduler = PersonaBlogScheduler::new(&cfg, producer, "test");
        // 50% of 3600 = 1800; samples in [1800, 5400].
        for _ in 0..20 {
            let s = scheduler.jittered_interval().as_secs();
            assert!((1_800..=5_400).contains(&s), "clamped jitter sample {s}s");
        }
    }

    #[tokio::test(start_paused = true)]
    async fn cancels_cleanly() {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel::<(String, Vec<u8>)>();
        let producer: Arc<dyn CommandProducer> = Arc::new(ChannelCommandProducer { tx });
        let scheduler =
            PersonaBlogScheduler::new(&cfg_with_interval(3600), producer, "test.commands");
        let cancel = CancellationToken::new();
        let cancel_for_task = cancel.clone();
        let handle = tokio::spawn(scheduler.run(cancel_for_task));
        cancel.cancel();
        tokio::time::timeout(Duration::from_millis(500), handle)
            .await
            .expect("scheduler should exit on cancel")
            .unwrap();
    }
}
