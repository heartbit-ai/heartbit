use std::str::FromStr;
use std::sync::Arc;

use chrono::Utc;
use cron::Schedule;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::Error;
use crate::config::ScheduleEntry;

use super::CommandProducer;
use super::types::DaemonCommand;

struct ParsedSchedule {
    name: String,
    schedule: Schedule,
    task: String,
}

/// Cron scheduler that produces `SubmitTask` commands via [`CommandProducer`].
pub struct CronScheduler {
    schedules: Vec<ParsedSchedule>,
    producer: Arc<dyn CommandProducer>,
    commands_topic: String,
}

impl std::fmt::Debug for CronScheduler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CronScheduler")
            .field("schedules", &self.schedules.len())
            .field("commands_topic", &self.commands_topic)
            .finish()
    }
}

impl CronScheduler {
    pub fn new(
        entries: &[ScheduleEntry],
        producer: Arc<dyn CommandProducer>,
        commands_topic: &str,
    ) -> Result<Self, Error> {
        let mut schedules = Vec::with_capacity(entries.len());
        for entry in entries {
            if !entry.enabled {
                continue;
            }
            let schedule = Schedule::from_str(&entry.cron).map_err(|e| {
                Error::Daemon(format!(
                    "invalid cron expression '{}' for schedule '{}': {e}",
                    entry.cron, entry.name
                ))
            })?;
            schedules.push(ParsedSchedule {
                name: entry.name.clone(),
                schedule,
                task: entry.task.clone(),
            });
        }
        Ok(Self {
            schedules,
            producer,
            commands_topic: commands_topic.into(),
        })
    }

    /// Run the scheduler loop. Checks every 30 seconds, produces `SubmitTask` when due.
    pub async fn run(self, cancel: CancellationToken) {
        let tick_interval = std::time::Duration::from_secs(30);

        loop {
            tokio::select! {
                _ = cancel.cancelled() => {
                    tracing::info!("cron scheduler shutting down");
                    break;
                }
                _ = tokio::time::sleep(tick_interval) => {
                    let now = Utc::now();
                    let window_start = now - chrono::Duration::seconds(30);

                    for parsed in &self.schedules {
                        // Check if any scheduled time falls within the last 30s window
                        if let Some(next) = parsed.schedule.after(&window_start).next()
                            && next <= now
                        {
                            let id = Uuid::new_v4();
                            let cmd = DaemonCommand::SubmitTask {
                                id,
                                task: parsed.task.clone(),
                                source: format!("cron:{}", parsed.name),
                                story_id: None,
                                trust_level: None,
                            };
                            let payload = match serde_json::to_vec(&cmd) {
                                Ok(p) => p,
                                Err(e) => {
                                    tracing::error!(
                                        schedule = %parsed.name,
                                        error = %e,
                                        "failed to serialize cron command"
                                    );
                                    continue;
                                }
                            };
                            match self
                                .producer
                                .send_command(
                                    &self.commands_topic,
                                    &id.to_string(),
                                    &payload,
                                )
                                .await
                            {
                                Ok(()) => {
                                    tracing::info!(
                                        schedule = %parsed.name,
                                        task_id = %id,
                                        "cron triggered task"
                                    );
                                }
                                Err(e) => {
                                    tracing::error!(
                                        schedule = %parsed.name,
                                        error = %e,
                                        "failed to produce cron task"
                                    );
                                }
                            }
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

    fn mock_producer() -> (
        Arc<dyn CommandProducer>,
        tokio::sync::mpsc::UnboundedReceiver<(String, Vec<u8>)>,
    ) {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        (Arc::new(ChannelCommandProducer { tx }), rx)
    }

    fn mock_producer_only() -> Arc<dyn CommandProducer> {
        mock_producer().0
    }

    #[test]
    fn cron_scheduler_parses_valid_entries() {
        let entries = vec![
            ScheduleEntry {
                name: "daily".into(),
                cron: "0 0 9 * * *".into(),
                task: "Review".into(),
                enabled: true,
            },
            ScheduleEntry {
                name: "hourly".into(),
                cron: "0 0 * * * *".into(),
                task: "Check".into(),
                enabled: true,
            },
        ];

        let scheduler =
            CronScheduler::new(&entries, mock_producer_only(), "test.commands").unwrap();
        assert_eq!(scheduler.schedules.len(), 2);
    }

    #[test]
    fn cron_scheduler_skips_disabled() {
        let entries = vec![
            ScheduleEntry {
                name: "enabled".into(),
                cron: "0 0 9 * * *".into(),
                task: "Review".into(),
                enabled: true,
            },
            ScheduleEntry {
                name: "disabled".into(),
                cron: "0 0 18 * * *".into(),
                task: "Cleanup".into(),
                enabled: false,
            },
        ];

        let scheduler =
            CronScheduler::new(&entries, mock_producer_only(), "test.commands").unwrap();
        assert_eq!(scheduler.schedules.len(), 1);
        assert_eq!(scheduler.schedules[0].name, "enabled");
    }

    #[test]
    fn cron_scheduler_invalid_expression_rejected() {
        let entries = vec![ScheduleEntry {
            name: "bad".into(),
            cron: "not a cron".into(),
            task: "Something".into(),
            enabled: true,
        }];

        let err = CronScheduler::new(&entries, mock_producer_only(), "test.commands").unwrap_err();
        assert!(err.to_string().contains("invalid cron expression"));
    }

    #[test]
    fn cron_scheduler_empty_entries() {
        let scheduler = CronScheduler::new(&[], mock_producer_only(), "test.commands").unwrap();
        assert!(scheduler.schedules.is_empty());
    }

    /// Receive a command from the mock producer via spin-yield loop.
    async fn recv_cmd(
        rx: &mut tokio::sync::mpsc::UnboundedReceiver<(String, Vec<u8>)>,
    ) -> (String, Vec<u8>) {
        for _ in 0..100 {
            tokio::task::yield_now().await;
            if let Ok(msg) = rx.try_recv() {
                return msg;
            }
        }
        panic!("timed out waiting for command on mock producer channel");
    }

    #[tokio::test(start_paused = true)]
    async fn run_fires_task_on_schedule() {
        // Schedule that fires every second (for testing)
        let entries = vec![ScheduleEntry {
            name: "every-second".into(),
            cron: "* * * * * *".into(),
            task: "Run analysis".into(),
            enabled: true,
        }];

        let (producer, mut rx) = mock_producer();
        let scheduler = CronScheduler::new(&entries, producer, "test.commands").unwrap();

        let cancel = CancellationToken::new();
        let cancel2 = cancel.clone();
        tokio::spawn(async move { scheduler.run(cancel2).await });

        // Let spawned task register its sleep
        tokio::task::yield_now().await;
        // Advance past the 30s tick interval
        tokio::time::advance(std::time::Duration::from_secs(31)).await;

        // Should have received at least one command
        let (key, payload) = recv_cmd(&mut rx).await;
        assert!(!key.is_empty());
        let cmd: DaemonCommand = serde_json::from_slice(&payload).unwrap();
        match cmd {
            DaemonCommand::SubmitTask { source, task, .. } => {
                assert_eq!(source, "cron:every-second");
                assert_eq!(task, "Run analysis");
            }
            other => panic!("unexpected command: {other:?}"),
        }

        cancel.cancel();
    }

    #[tokio::test(start_paused = true)]
    async fn run_stops_on_cancellation() {
        let scheduler = CronScheduler::new(&[], mock_producer_only(), "test.commands").unwrap();

        let cancel = CancellationToken::new();
        let cancel2 = cancel.clone();
        let handle = tokio::spawn(async move { scheduler.run(cancel2).await });

        tokio::task::yield_now().await;
        cancel.cancel();
        tokio::time::advance(std::time::Duration::from_secs(1)).await;

        tokio::time::timeout(std::time::Duration::from_secs(5), handle)
            .await
            .expect("run should exit on cancel")
            .expect("task should not panic");
    }
}
