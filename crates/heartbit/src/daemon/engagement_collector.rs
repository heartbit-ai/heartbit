//! Periodic engagement-refresh scheduler. Fires
//! `DaemonCommand::EngagementRefresh` per configured persona on a
//! jittered cadence (default 6h ±25%).
//!
//! No `active_hours` gate — engagement collection is cheap and runs
//! around the clock. The handler will skip too-young / too-old tweets
//! at refresh time.
//!
//! Pattern mirrors [`super::persona_post::PersonaPostScheduler`].

use std::sync::Arc;
use std::time::Duration;

use tokio_util::sync::CancellationToken;

use super::CommandProducer;
use super::types::DaemonCommand;

/// Per-persona engagement collector. Fires an `EngagementRefresh`
/// command every `interval` (with `±jitter_pct%` randomization) via
/// the producer. Unlike `PersonaPostScheduler`, there is no
/// `active_hours` gate — engagement metrics refresh happens around
/// the clock and the handler decides eligibility per-tweet.
pub struct EngagementCollectorScheduler {
    persona: String,
    interval: Duration,
    jitter_pct: u32,
    producer: Arc<dyn CommandProducer>,
    commands_topic: String,
}

impl std::fmt::Debug for EngagementCollectorScheduler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EngagementCollectorScheduler")
            .field("persona", &self.persona)
            .field("interval", &self.interval)
            .field("jitter_pct", &self.jitter_pct)
            .field("commands_topic", &self.commands_topic)
            .finish()
    }
}

impl EngagementCollectorScheduler {
    /// Construct a scheduler. `jitter_pct` is clamped to `0..=50` (same
    /// safety band as `PersonaPostScheduler`).
    pub fn new(
        persona: impl Into<String>,
        interval: Duration,
        jitter_pct: u32,
        producer: Arc<dyn CommandProducer>,
        commands_topic: impl Into<String>,
    ) -> Self {
        Self {
            persona: persona.into(),
            interval,
            // Clamp to a sane band — `0` is "no jitter" (deterministic tests);
            // anything above `50` produces wild swings that defeat the cadence.
            jitter_pct: jitter_pct.min(50),
            producer,
            commands_topic: commands_topic.into(),
        }
    }

    /// Compute the next sleep duration, randomized within
    /// `±jitter_pct%` of `interval`. Returns `interval` unchanged when
    /// jitter is 0. Floor is 60s — even with extreme jitter we never
    /// hammer the engagement pipeline faster than once a minute.
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

    /// Run the scheduler loop until `cancel` fires. Each tick sleeps
    /// the (jittered) `interval` then dispatches an
    /// `EngagementRefresh` command to `commands_topic`.
    pub async fn run(self, cancel: CancellationToken) {
        loop {
            // Re-roll each iteration so the cadence drifts over time
            // rather than locking to a fixed offset from boot.
            let next = self.jittered_interval();
            tracing::debug!(
                persona = %self.persona,
                next_sleep_secs = next.as_secs(),
                "engagement collector: sleeping until next refresh"
            );
            tokio::select! {
                _ = cancel.cancelled() => {
                    tracing::info!(persona = %self.persona, "engagement collector shutting down");
                    break;
                }
                _ = tokio::time::sleep(next) => {
                    let cmd = DaemonCommand::EngagementRefresh {
                        persona: self.persona.clone(),
                    };
                    let payload = match serde_json::to_vec(&cmd) {
                        Ok(p) => p,
                        Err(e) => {
                            tracing::error!(error = %e, "failed to serialize EngagementRefresh");
                            continue;
                        }
                    };
                    let key = format!("engagement:{}", self.persona);
                    if let Err(e) = self
                        .producer
                        .send_command(&self.commands_topic, &key, &payload)
                        .await
                    {
                        tracing::error!(
                            persona = %self.persona,
                            error = %e,
                            "failed to dispatch EngagementRefresh"
                        );
                    } else {
                        tracing::debug!(persona = %self.persona, "engagement collector dispatched");
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::ChannelCommandProducer;
    use super::*;

    /// Scheduler fires one `EngagementRefresh` after the configured
    /// interval and the payload round-trips through the wire shape
    /// the consumer expects.
    #[tokio::test(start_paused = true)]
    async fn fires_engagement_refresh_after_interval() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<(String, Vec<u8>)>();
        let producer: Arc<dyn CommandProducer> = Arc::new(ChannelCommandProducer { tx });
        let scheduler = EngagementCollectorScheduler::new(
            "heartbit-ghost:x",
            Duration::from_secs(60),
            0, // deterministic test — no jitter
            producer,
            "test.commands",
        );
        let cancel = CancellationToken::new();
        let cancel_for_task = cancel.clone();
        let handle = tokio::spawn(scheduler.run(cancel_for_task));

        tokio::time::advance(Duration::from_secs(61)).await;
        let (key, payload) = rx.recv().await.expect("scheduler should have fired");
        // Routing key prefix prevents collision with persona-post messages.
        assert_eq!(key, "engagement:heartbit-ghost:x");

        // Wire shape: serde tag is the snake_case variant name.
        let json = std::str::from_utf8(&payload).unwrap();
        assert!(
            json.contains(r#""type":"engagement_refresh""#),
            "json was: {json}"
        );
        let parsed: DaemonCommand = serde_json::from_slice(&payload).unwrap();
        assert!(matches!(
            &parsed,
            DaemonCommand::EngagementRefresh { persona } if persona == "heartbit-ghost:x"
        ));

        cancel.cancel();
        let _ = handle.await;
    }

    /// `jitter_pct > 50` is clamped at construction time. Even with a
    /// pathological `500` input the produced intervals stay in the
    /// band defined by 50%.
    #[test]
    fn jitter_clamps_at_50() {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel::<(String, Vec<u8>)>();
        let producer: Arc<dyn CommandProducer> = Arc::new(ChannelCommandProducer { tx });
        let s = EngagementCollectorScheduler::new(
            "p",
            Duration::from_secs(3600),
            500, // clamped to 50
            producer,
            "test",
        );
        // 50% jitter at 3600s base → [1800, 5400].
        for _ in 0..20 {
            let n = s.jittered_interval().as_secs();
            assert!(
                (1_800..=5_400).contains(&n),
                "clamped jitter sample {n}s out of [1800, 5400]"
            );
        }
    }
}
