use std::future::Future;
use std::pin::Pin;

use crate::Error;

pub mod core;
pub mod cron;
pub mod engagement_collector;
pub mod engagement_refresh_handler;
pub mod heartbit_pulse;
pub mod kafka;
pub mod mention_context;
pub mod mention_poll;
pub mod mention_poll_handler;
pub mod metrics;
pub mod notify;
pub mod openai_compat;
pub mod persona_post;
pub mod persona_post_handler;
pub mod persona_quote;
pub mod persona_quote_handler;
pub mod posts_context;
pub mod quotes_context;
pub mod reply_draft_handler;
pub mod runtime_types;
pub mod store;
pub mod todo;
pub mod types;

pub use self::core::{DaemonCore, DaemonHandle};
pub use cron::CronScheduler;
pub use engagement_collector::EngagementCollectorScheduler;
pub use engagement_refresh_handler::{EngagementRefreshDeps, handle_engagement_refresh};
pub use heartbit_pulse::HeartbitPulseScheduler;
pub use kafka::KafkaCommandProducer;
pub use mention_context::{MentionContext, PersonaMentionEntry, ReplySharedContext};
pub use mention_poll::MentionPollScheduler;
pub use mention_poll_handler::{MentionPollDeps, handle_mention_poll};
pub use metrics::DaemonMetrics;
pub use notify::{OnTaskComplete, TaskOutcome, format_notification};
pub use openai_compat::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, ModelListResponse,
    build_done_chunk, build_model_list, build_response, build_role_chunk, build_text_chunk,
    extract_system_prompt, extract_task,
};
pub use persona_post::PersonaPostScheduler;
pub use persona_post_handler::{PersonaPostDeps, handle_persona_post};
pub use persona_quote::PersonaQuoteScheduler;
pub use persona_quote_handler::{PersonaQuoteDeps, handle_persona_quote};
pub use posts_context::{PersonaPostEntry, PostsContext};
pub use quotes_context::{PersonaQuoteEntry, QuotesContext};
pub use reply_draft_handler::{ReplyDraftDeps, handle_reply_draft};
pub use runtime_types::{
    EdgeConditionPattern, EdgeConditionSpec, EdgeTransform, RuntimeAdvancedConfig,
    RuntimeAgentConfig, RuntimeEvalRequest, RuntimeEvalResponse, RuntimeEvalSseEvent,
    RuntimeGuardrailConfig, RuntimeMcpServer, RuntimeMemoryConfig, RuntimeOrchestratorConfig,
    RuntimeProviderConfig, RuntimeProviderType, RuntimeRequest, RuntimeResponse,
    RuntimeScorerConfig, RuntimeSpawnConfig, RuntimeSseEvent, RuntimeSubAgentConfig,
    RuntimeTwitterCredentials, RuntimeWorkflowConfig, RuntimeWorkflowEdge, RuntimeWorkflowNode,
};
#[cfg(feature = "postgres")]
pub use store::PostgresTaskStore;
pub use store::{InMemoryTaskStore, TaskStore};
pub use todo::{FileTodoStore, TodoEntry, TodoList, TodoManageTool};
pub use types::{
    DaemonCommand, DaemonTask, TaskState, TaskStats, UsageGroupBy, UsageQuery, UsageRow,
    UserContext,
};

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
