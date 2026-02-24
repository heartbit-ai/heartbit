use std::future::Future;
use std::pin::Pin;

use crate::Error;

pub mod core;
pub mod cron;
pub mod heartbit_pulse;
pub mod kafka;
pub mod metrics;
pub mod store;
pub mod todo;
pub mod types;

pub use self::core::{DaemonCore, DaemonHandle};
pub use cron::CronScheduler;
pub use heartbit_pulse::HeartbitPulseScheduler;
pub use kafka::KafkaCommandProducer;
pub use metrics::DaemonMetrics;
pub use store::{InMemoryTaskStore, PostgresTaskStore, TaskStore};
pub use todo::{FileTodoStore, TodoEntry, TodoList, TodoManageTool};
pub use types::{DaemonCommand, DaemonTask, TaskState, TaskStats};

/// Object-safe async trait for producing daemon commands to a topic.
///
/// Uses `Pin<Box<dyn Future>>` for dyn-compatibility (same pattern as
/// `DynLlmProvider` in `llm/mod.rs`).
pub trait CommandProducer: Send + Sync {
    fn send_command<'a>(
        &'a self,
        topic: &'a str,
        key: &'a str,
        payload: &'a [u8],
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + 'a>>;
}

/// Test mock using `tokio::sync::mpsc` — collects sent commands for assertions.
#[cfg(test)]
pub(crate) struct ChannelCommandProducer {
    pub tx: tokio::sync::mpsc::UnboundedSender<(String, Vec<u8>)>,
}

#[cfg(test)]
impl CommandProducer for ChannelCommandProducer {
    fn send_command<'a>(
        &'a self,
        _topic: &'a str,
        key: &'a str,
        payload: &'a [u8],
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + 'a>> {
        let key = key.to_string();
        let payload = payload.to_vec();
        Box::pin(async move {
            self.tx
                .send((key, payload))
                .map_err(|e| Error::Daemon(e.to_string()))
        })
    }
}
