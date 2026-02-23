use std::str::FromStr;

use chrono::Utc;
use cron::Schedule;
use rdkafka::producer::{FutureProducer, FutureRecord};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::Error;
use crate::config::ScheduleEntry;

use super::types::DaemonCommand;

struct ParsedSchedule {
    name: String,
    schedule: Schedule,
    task: String,
}

/// Cron scheduler that produces `SubmitTask` commands to Kafka.
pub struct CronScheduler {
    schedules: Vec<ParsedSchedule>,
    producer: FutureProducer,
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
        producer: FutureProducer,
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
                                .send(
                                    FutureRecord::to(&self.commands_topic)
                                        .key(&id.to_string())
                                        .payload(&payload),
                                    rdkafka::util::Timeout::Never,
                                )
                                .await
                            {
                                Ok(_) => {
                                    tracing::info!(
                                        schedule = %parsed.name,
                                        task_id = %id,
                                        "cron triggered task"
                                    );
                                }
                                Err((e, _)) => {
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
    use super::*;

    #[test]
    fn cron_scheduler_parses_valid_entries() {
        let config = crate::config::KafkaConfig {
            brokers: "localhost:9092".into(),
            consumer_group: "test".into(),
            commands_topic: "test.commands".into(),
            events_topic: "test.events".into(),
            dead_letter_topic: "test.dead-letter".into(),
        };
        let producer = crate::daemon::kafka::create_producer(&config).unwrap();

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

        let scheduler = CronScheduler::new(&entries, producer, "test.commands").unwrap();
        assert_eq!(scheduler.schedules.len(), 2);
    }

    #[test]
    fn cron_scheduler_skips_disabled() {
        let config = crate::config::KafkaConfig {
            brokers: "localhost:9092".into(),
            consumer_group: "test".into(),
            commands_topic: "test.commands".into(),
            events_topic: "test.events".into(),
            dead_letter_topic: "test.dead-letter".into(),
        };
        let producer = crate::daemon::kafka::create_producer(&config).unwrap();

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

        let scheduler = CronScheduler::new(&entries, producer, "test.commands").unwrap();
        assert_eq!(scheduler.schedules.len(), 1);
        assert_eq!(scheduler.schedules[0].name, "enabled");
    }

    #[test]
    fn cron_scheduler_invalid_expression_rejected() {
        let config = crate::config::KafkaConfig {
            brokers: "localhost:9092".into(),
            consumer_group: "test".into(),
            commands_topic: "test.commands".into(),
            events_topic: "test.events".into(),
            dead_letter_topic: "test.dead-letter".into(),
        };
        let producer = crate::daemon::kafka::create_producer(&config).unwrap();

        let entries = vec![ScheduleEntry {
            name: "bad".into(),
            cron: "not a cron".into(),
            task: "Something".into(),
            enabled: true,
        }];

        let err = CronScheduler::new(&entries, producer, "test.commands").unwrap_err();
        assert!(err.to_string().contains("invalid cron expression"));
    }

    #[test]
    fn cron_scheduler_empty_entries() {
        let config = crate::config::KafkaConfig {
            brokers: "localhost:9092".into(),
            consumer_group: "test".into(),
            commands_topic: "test.commands".into(),
            events_topic: "test.events".into(),
            dead_letter_topic: "test.dead-letter".into(),
        };
        let producer = crate::daemon::kafka::create_producer(&config).unwrap();

        let scheduler = CronScheduler::new(&[], producer, "test.commands").unwrap();
        assert!(scheduler.schedules.is_empty());
    }
}
