//! Cron-driven mention-poll scheduler. Fires one [`DaemonCommand::MentionPoll`]
//! per configured persona on a fixed interval. Mirrors the shape of
//! [`super::heartbit_pulse::HeartbitPulseScheduler`].

use std::sync::Arc;
use std::time::Duration;

use tokio_util::sync::CancellationToken;

use crate::config::PersonaMentionsConfig;

use super::CommandProducer;
use super::types::DaemonCommand;

/// Periodic scheduler that fires [`DaemonCommand::MentionPoll`] for one
/// persona at a fixed interval.
pub struct MentionPollScheduler {
    persona: String,
    user_id: String,
    interval: Duration,
    producer: Arc<dyn CommandProducer>,
    commands_topic: String,
}

impl std::fmt::Debug for MentionPollScheduler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MentionPollScheduler")
            .field("persona", &self.persona)
            .field("user_id", &self.user_id)
            .field("interval", &self.interval)
            .finish()
    }
}

impl MentionPollScheduler {
    /// Construct from a [`PersonaMentionsConfig`].
    pub fn new(
        cfg: &PersonaMentionsConfig,
        producer: Arc<dyn CommandProducer>,
        commands_topic: &str,
    ) -> Result<Self, crate::Error> {
        if cfg.poll_interval_seconds == 0 {
            return Err(crate::Error::Config(
                "persona_mentions.poll_interval_seconds must be > 0".into(),
            ));
        }
        Ok(Self {
            persona: cfg.persona.clone(),
            user_id: cfg.user_id.clone(),
            interval: Duration::from_secs(cfg.poll_interval_seconds),
            producer,
            commands_topic: commands_topic.into(),
        })
    }

    /// Run the poll loop. Fires one [`DaemonCommand::MentionPoll`] per tick
    /// until the cancellation token is triggered.
    pub async fn run(self, cancel: CancellationToken) {
        loop {
            tokio::select! {
                _ = cancel.cancelled() => {
                    tracing::info!(
                        persona = %self.persona,
                        "mention-poll scheduler shutting down"
                    );
                    break;
                }
                _ = tokio::time::sleep(self.interval) => {
                    let cmd = DaemonCommand::MentionPoll {
                        persona: self.persona.clone(),
                        user_id: self.user_id.clone(),
                    };
                    let key = format!("mentions:{}:{}", self.persona, self.user_id);
                    let payload = match serde_json::to_vec(&cmd) {
                        Ok(p) => p,
                        Err(e) => {
                            tracing::error!(
                                persona = %self.persona,
                                error = %e,
                                "failed to serialize MentionPoll command"
                            );
                            continue;
                        }
                    };
                    match self
                        .producer
                        .send_command(&self.commands_topic, &key, &payload)
                        .await
                    {
                        Ok(()) => {
                            tracing::debug!(
                                persona = %self.persona,
                                user_id = %self.user_id,
                                "MentionPoll dispatched"
                            );
                        }
                        Err(e) => {
                            tracing::error!(
                                persona = %self.persona,
                                error = %e,
                                "failed to produce MentionPoll command"
                            );
                        }
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

    type MockProducerHandle = (
        Arc<dyn CommandProducer>,
        tokio::sync::mpsc::UnboundedReceiver<(String, Vec<u8>)>,
    );

    fn mock_producer() -> MockProducerHandle {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        (Arc::new(ChannelCommandProducer { tx }), rx)
    }

    fn test_cfg(interval: u64) -> PersonaMentionsConfig {
        PersonaMentionsConfig {
            persona: "heartbit-ghost".into(),
            enabled: true,
            poll_interval_seconds: interval,
            user_id: "x".into(),
            candidates_per_reply: 5,
            mention_store: "in_memory".into(),
            mention_store_path: None,
            // P1.7 guard defaults for test scaffolding.
            enable_thread_depth_guard: true,
            enable_bot_heuristic_guard: true,
            suspicious_handle_patterns: vec![],
            min_follower_following_ratio: 0.05,
            min_account_age_days: 7,
            bot_heuristic_threshold: 2,
            per_conversation_max_replies: 2,
            daily_token_budget: None,
            budget_store: "in_memory".into(),
            budget_path: None,
        }
    }

    async fn recv_cmd(
        rx: &mut tokio::sync::mpsc::UnboundedReceiver<(String, Vec<u8>)>,
    ) -> (String, Vec<u8>) {
        for _ in 0..100 {
            tokio::task::yield_now().await;
            if let Ok(msg) = rx.try_recv() {
                return msg;
            }
        }
        panic!("timed out waiting for command from mock producer");
    }

    async fn assert_no_cmd(rx: &mut tokio::sync::mpsc::UnboundedReceiver<(String, Vec<u8>)>) {
        for _ in 0..50 {
            tokio::task::yield_now().await;
        }
        assert!(
            rx.try_recv().is_err(),
            "expected no command but one was received"
        );
    }

    #[test]
    fn new_rejects_zero_interval() {
        let (producer, _rx) = mock_producer();
        let cfg = test_cfg(0);
        let err = MentionPollScheduler::new(&cfg, producer, "test.commands").unwrap_err();
        assert!(
            err.to_string()
                .contains("poll_interval_seconds must be > 0")
        );
    }

    #[test]
    fn new_succeeds_with_valid_config() {
        let (producer, _rx) = mock_producer();
        let cfg = test_cfg(60);
        let sched = MentionPollScheduler::new(&cfg, producer, "test.commands").unwrap();
        assert_eq!(sched.persona, "heartbit-ghost");
        assert_eq!(sched.interval, Duration::from_secs(60));
    }

    #[tokio::test(start_paused = true)]
    async fn run_dispatches_mention_poll_on_tick() {
        let (producer, mut rx) = mock_producer();
        let cfg = test_cfg(1);
        let sched = MentionPollScheduler::new(&cfg, producer, "test.commands").unwrap();

        let cancel = CancellationToken::new();
        let cancel2 = cancel.clone();
        tokio::spawn(async move { sched.run(cancel2).await });

        tokio::task::yield_now().await;
        tokio::time::advance(Duration::from_secs(2)).await;

        let (key, payload) = recv_cmd(&mut rx).await;
        assert_eq!(key, "mentions:heartbit-ghost:x");
        let cmd: DaemonCommand = serde_json::from_slice(&payload).unwrap();
        match cmd {
            DaemonCommand::MentionPoll { persona, user_id } => {
                assert_eq!(persona, "heartbit-ghost");
                assert_eq!(user_id, "x");
            }
            other => panic!("unexpected command: {other:?}"),
        }
        cancel.cancel();
    }

    #[tokio::test(start_paused = true)]
    async fn run_fires_multiple_ticks() {
        let (producer, mut rx) = mock_producer();
        let cfg = test_cfg(10);
        let sched = MentionPollScheduler::new(&cfg, producer, "test.commands").unwrap();

        let cancel = CancellationToken::new();
        let cancel2 = cancel.clone();
        tokio::spawn(async move { sched.run(cancel2).await });

        tokio::task::yield_now().await;

        // First tick
        tokio::time::advance(Duration::from_secs(11)).await;
        recv_cmd(&mut rx).await;

        // Second tick
        tokio::time::advance(Duration::from_secs(10)).await;
        recv_cmd(&mut rx).await;

        cancel.cancel();
    }

    #[tokio::test(start_paused = true)]
    async fn run_does_not_fire_before_first_interval() {
        let (producer, mut rx) = mock_producer();
        let cfg = test_cfg(60);
        let sched = MentionPollScheduler::new(&cfg, producer, "test.commands").unwrap();

        let cancel = CancellationToken::new();
        let cancel2 = cancel.clone();
        tokio::spawn(async move { sched.run(cancel2).await });

        tokio::task::yield_now().await;
        // Advance less than one interval
        tokio::time::advance(Duration::from_secs(30)).await;

        assert_no_cmd(&mut rx).await;
        cancel.cancel();
    }

    #[tokio::test(start_paused = true)]
    async fn run_stops_on_cancellation() {
        let (producer, _rx) = mock_producer();
        let cfg = test_cfg(60);
        let sched = MentionPollScheduler::new(&cfg, producer, "test.commands").unwrap();

        let cancel = CancellationToken::new();
        let cancel2 = cancel.clone();
        let handle = tokio::spawn(async move { sched.run(cancel2).await });

        tokio::task::yield_now().await;
        cancel.cancel();
        tokio::time::advance(Duration::from_secs(1)).await;

        tokio::time::timeout(Duration::from_secs(5), handle)
            .await
            .expect("run should exit on cancel")
            .expect("task should not panic");
    }
}
