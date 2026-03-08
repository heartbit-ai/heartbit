use std::sync::Arc;

use heartbit::{CommandProducer, CronScheduler, ScheduleEntry};
use tokio_util::sync::CancellationToken;

/// Start the cron scheduler in a background task.
///
/// Does nothing if `schedules` is empty.
pub fn start_cron(
    schedules: &[ScheduleEntry],
    producer: Arc<dyn CommandProducer>,
    topic: &str,
    cancel: CancellationToken,
) -> anyhow::Result<()> {
    if schedules.is_empty() {
        tracing::info!("no cron schedules configured, skipping cron scheduler");
        return Ok(());
    }

    let cron = CronScheduler::new(schedules, producer, topic)
        .map_err(|e| anyhow::anyhow!("cron scheduler init: {e}"))?;

    tokio::spawn(async move {
        cron.run(cancel).await;
    });

    tracing::info!(schedules = schedules.len(), "cron scheduler started");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn start_cron_empty_schedules_is_noop() {
        // We need a tokio runtime for the function but it returns immediately
        // for empty schedules without spawning anything.
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        rt.block_on(async {
            let cancel = CancellationToken::new();
            // Using a KafkaCommandProducer requires a valid producer, but we
            // never reach that code with empty schedules. We test via the
            // heartbit ChannelCommandProducer pattern — but it's pub(crate).
            // Instead, just verify the empty-schedules fast path.
            // We can't easily construct a CommandProducer here without Kafka,
            // so we use a real producer with dummy config (lazy connection).
            let config = heartbit::KafkaConfig {
                brokers: "localhost:9092".into(),
                consumer_group: "test".into(),
                commands_topic: "test.commands".into(),
                events_topic: "test.events".into(),
                dead_letter_topic: "test.dead-letter".into(),
            };
            let fp = heartbit::daemon::kafka::create_producer(&config).unwrap();
            let producer: Arc<dyn CommandProducer> =
                Arc::new(heartbit::KafkaCommandProducer::new(fp));

            let result = start_cron(&[], producer, "test.commands", cancel);
            assert!(result.is_ok());
        });
    }

    #[test]
    fn start_cron_invalid_expression_returns_error() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        rt.block_on(async {
            let cancel = CancellationToken::new();
            let config = heartbit::KafkaConfig {
                brokers: "localhost:9092".into(),
                consumer_group: "test".into(),
                commands_topic: "test.commands".into(),
                events_topic: "test.events".into(),
                dead_letter_topic: "test.dead-letter".into(),
            };
            let fp = heartbit::daemon::kafka::create_producer(&config).unwrap();
            let producer: Arc<dyn CommandProducer> =
                Arc::new(heartbit::KafkaCommandProducer::new(fp));

            let schedules = vec![heartbit::ScheduleEntry {
                name: "bad".into(),
                cron: "not a cron expression".into(),
                task: "test".into(),
                enabled: true,
            }];

            let result = start_cron(&schedules, producer, "test.commands", cancel);
            assert!(result.is_err());
            assert!(
                result
                    .unwrap_err()
                    .to_string()
                    .contains("cron scheduler init"),
                "error should mention cron scheduler init"
            );
        });
    }
}
