use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;

use chrono::Utc;
use rdkafka::consumer::StreamConsumer;
use rdkafka::message::Message;
use rdkafka::producer::{FutureProducer, FutureRecord};
use tokio::sync::{Semaphore, broadcast};
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

use crate::Error;
use crate::agent::AgentOutput;
use crate::agent::events::AgentEvent;
use crate::agent::tenant_tracker::TenantTokenTracker;
#[cfg(feature = "postgres")]
use crate::config::DaemonAuditConfig;
use crate::config::{DaemonConfig, IdempotencyConfig};
#[cfg(feature = "postgres")]
use crate::store::PostgresStore;

use super::notify::{OnTaskComplete, TaskOutcome};
use super::store::TaskStore;
use super::types::{DaemonCommand, DaemonTask, TaskState, TaskStats};

/// Error message returned when Kafka-dependent operations are called on an HTTP-only handle.
///
/// Used by CLI handlers to map this specific error to HTTP 503.
pub const KAFKA_REQUIRED: &str = "this operation requires Kafka (daemon is in HTTP-only mode)";

/// Re-insert a task into the store if no row exists for the given id.
///
/// Called from the consumer match arm so that a Kafka redelivery (e.g. after restart)
/// reconstructs the row with the original `idempotency_key`, `user_id`, and `tenant_id`.
/// Insert errors are intentionally swallowed because a concurrent inserter may have
/// already written the row — the consumer cares only about the post-condition.
pub(crate) fn ensure_task_inserted(
    store: &Arc<dyn TaskStore>,
    id: uuid::Uuid,
    task: &str,
    source: &str,
    user_id: Option<&str>,
    tenant_id: Option<&str>,
    idempotency_key: Option<&str>,
) {
    if !matches!(store.get(id), Ok(None)) {
        return;
    }
    let mut daemon_task = match (user_id, tenant_id) {
        (Some(uid), Some(tid)) => DaemonTask::new_with_user(id, task, source, uid, tid),
        _ => DaemonTask::new(id, task, source),
    };
    daemon_task.idempotency_key = idempotency_key.map(String::from);
    let _ = store.insert(daemon_task);
}

/// Cloneable handle for producing commands and reading state.
#[derive(Clone)]
pub struct DaemonHandle {
    producer: Option<FutureProducer>,
    commands_topic: Option<String>,
    store: Arc<dyn TaskStore>,
    event_channels: Arc<parking_lot::RwLock<HashMap<uuid::Uuid, broadcast::Sender<AgentEvent>>>>,
    pub(crate) tenant_tracker: Option<Arc<TenantTokenTracker>>,
}

impl DaemonHandle {
    /// Create an HTTP-only handle (no Kafka producer).
    ///
    /// Task submission and cancellation via Kafka will return errors.
    /// Direct task registration (`register_task`) and reads still work.
    pub fn http_only(store: Arc<dyn TaskStore>) -> Self {
        Self {
            producer: None,
            commands_topic: None,
            store,
            event_channels: Arc::new(parking_lot::RwLock::new(HashMap::new())),
            tenant_tracker: None,
        }
    }

    /// Attach a per-tenant token tracker for admission-gate checks at submit time.
    ///
    /// When set, `submit_task_with_user_idem` and `submit_task_with_user` will call
    /// `tracker.reserve()` before inserting the task. On `Error::TenantOverloaded`
    /// the error propagates to the caller; on success the reservation is dropped
    /// immediately (admission-only — actual per-turn usage is tracked by `AgentRunner`).
    pub fn with_tenant_tracker(mut self, tracker: Arc<TenantTokenTracker>) -> Self {
        self.tenant_tracker = Some(tracker);
        self
    }

    /// Returns the Kafka producer and commands topic, or an error if not configured.
    fn require_kafka(&self) -> Result<(&FutureProducer, &str), Error> {
        let producer = self
            .producer
            .as_ref()
            .ok_or_else(|| Error::Daemon(KAFKA_REQUIRED.into()))?;
        let topic = self
            .commands_topic
            .as_deref()
            .ok_or_else(|| Error::Daemon(KAFKA_REQUIRED.into()))?;
        Ok((producer, topic))
    }

    /// Submit a task: create in store as Pending, produce `SubmitTask` to Kafka.
    ///
    /// Returns an error when no Kafka producer is configured (HTTP-only mode).
    pub async fn submit_task(
        &self,
        task: impl Into<String>,
        source: impl Into<String>,
        story_id: Option<String>,
    ) -> Result<uuid::Uuid, Error> {
        let (producer, commands_topic) = self.require_kafka()?;

        let id = uuid::Uuid::new_v4();
        let task_str = task.into();
        let source_str = source.into();

        let daemon_task = DaemonTask::new(id, &task_str, &source_str);
        self.store.insert(daemon_task)?;

        let cmd = DaemonCommand::SubmitTask {
            id,
            task: task_str,
            source: source_str,
            story_id,
            trust_level: None,
            user_id: None,
            tenant_id: None,
            roles: vec![],
            mcp_auth_tokens: None,
            idempotency_key: None,
        };
        let payload = serde_json::to_vec(&cmd)
            .map_err(|e| Error::Daemon(format!("failed to serialize command: {e}")))?;

        producer
            .send(
                FutureRecord::to(commands_topic)
                    .key(&id.to_string())
                    .payload(&payload),
                rdkafka::util::Timeout::Never,
            )
            .await
            .map_err(|(e, _)| Error::Daemon(format!("failed to produce command: {e}")))?;

        Ok(id)
    }

    /// Submit a task with user context for multi-tenant isolation.
    ///
    /// Like `submit_task`, but attaches user/tenant identity to the command
    /// and creates the task record with user context.
    ///
    /// Returns an error when no Kafka producer is configured (HTTP-only mode).
    pub async fn submit_task_with_user(
        &self,
        task: impl Into<String>,
        source: impl Into<String>,
        story_id: Option<String>,
        user_context: &super::types::UserContext,
    ) -> Result<uuid::Uuid, Error> {
        let (producer, commands_topic) = self.require_kafka()?;

        let id = uuid::Uuid::new_v4();
        let task_str = task.into();
        let source_str = source.into();

        // B5b: per-tenant overload gate — admission-only check.
        if let Some(ref tracker) = self.tenant_tracker {
            let scope = crate::auth::TenantScope::new(&user_context.tenant_id);
            let estimated = task_str.len() / 4 + 4096;
            let _reservation = tracker.reserve(&scope, estimated)?;
        }

        let daemon_task = DaemonTask::new_with_user(
            id,
            &task_str,
            &source_str,
            &user_context.user_id,
            &user_context.tenant_id,
        );
        self.store.insert(daemon_task)?;

        let cmd = DaemonCommand::SubmitTask {
            id,
            task: task_str,
            source: source_str,
            story_id,
            trust_level: None,
            user_id: Some(user_context.user_id.clone()),
            tenant_id: Some(user_context.tenant_id.clone()),
            roles: user_context.roles.clone(),
            mcp_auth_tokens: None,
            idempotency_key: None,
        };
        let payload = serde_json::to_vec(&cmd)
            .map_err(|e| Error::Daemon(format!("failed to serialize command: {e}")))?;

        producer
            .send(
                FutureRecord::to(commands_topic)
                    .key(&id.to_string())
                    .payload(&payload),
                rdkafka::util::Timeout::Never,
            )
            .await
            .map_err(|(e, _)| Error::Daemon(format!("failed to produce command: {e}")))?;

        Ok(id)
    }

    /// Like `submit_task_with_user` but dedups on `idempotency_key`.
    ///
    /// When `idempotency_key` is supplied and an existing task with the same
    /// `(tenant_id, idempotency_key)` pair exists, returns the existing task id
    /// without producing a new Kafka message or creating a duplicate row.
    ///
    /// When `idempotency_key` is `None`, no dedup is applied — every call creates a new task.
    ///
    /// In HTTP-only mode (no Kafka), the row is inserted and `Ok(id)` is returned without
    /// attempting a Kafka publish. The caller is responsible for triggering execution by
    /// another path (e.g. HTTP-direct dispatch). This intentionally diverges from
    /// `submit_task_with_user`, which errors via `require_kafka()` when Kafka is absent —
    /// HTTP-only deployment is a B5b goal.
    pub async fn submit_task_with_user_idem(
        &self,
        task: impl Into<String>,
        source: impl Into<String>,
        story_id: Option<String>,
        user_context: &super::types::UserContext,
        idempotency_key: Option<&str>,
    ) -> Result<uuid::Uuid, Error> {
        let task_str = task.into();
        let source_str = source.into();

        // B5b: per-tenant overload gate — admission-only check.
        // The reservation is dropped immediately after the check; the runner's
        // adjust() handles actual per-turn usage tracking during execution.
        if let Some(ref tracker) = self.tenant_tracker {
            let scope = crate::auth::TenantScope::new(&user_context.tenant_id);
            let estimated = task_str.len() / 4 + 4096;
            let _reservation = tracker.reserve(&scope, estimated)?;
        }

        if let Some(key) = idempotency_key {
            // Lookup-first: return existing task id without producing to Kafka.
            if let Some(existing) = self
                .store
                .find_by_idempotency_key(&user_context.tenant_id, key)?
            {
                tracing::info!(
                    idempotency_key = %key,
                    tenant_id = %user_context.tenant_id,
                    task_id = %existing.id,
                    "idempotency hit; returning existing task id without re-execution"
                );
                return Ok(existing.id);
            }

            let id = uuid::Uuid::new_v4();
            let mut daemon_task = DaemonTask::new_with_user(
                id,
                &task_str,
                &source_str,
                &user_context.user_id,
                &user_context.tenant_id,
            );
            daemon_task.idempotency_key = Some(key.to_string());

            match self.store.insert(daemon_task) {
                Ok(()) => {
                    // Fresh insert: publish to Kafka if configured.
                    self.publish_submit_idem(
                        id,
                        task_str,
                        source_str,
                        story_id,
                        user_context,
                        Some(key.to_string()),
                    )
                    .await?;
                    Ok(id)
                }
                Err(e) if super::store::is_unique_violation(&e) => {
                    tracing::warn!(
                        idempotency_key = %key,
                        tenant_id = %user_context.tenant_id,
                        "idempotency unique-violation fallback; concurrent inserter raced ahead"
                    );
                    // Concurrent inserter raced ahead — resolve to their id.
                    self.store
                        .find_by_idempotency_key(&user_context.tenant_id, key)?
                        .map(|t| t.id)
                        .ok_or_else(|| {
                            Error::Daemon("unique violation but idempotency row not found".into())
                        })
                }
                Err(e) => Err(e),
            }
        } else {
            // No key — insert without dedup, then publish (graceful in HTTP-only mode).
            let id = uuid::Uuid::new_v4();
            let daemon_task = DaemonTask::new_with_user(
                id,
                &task_str,
                &source_str,
                &user_context.user_id,
                &user_context.tenant_id,
            );
            self.store.insert(daemon_task)?;
            self.publish_submit_idem(id, task_str, source_str, story_id, user_context, None)
                .await?;
            Ok(id)
        }
    }

    /// Publish a `SubmitTask` command with idempotency key, skipping gracefully
    /// when no Kafka producer is configured (HTTP-only mode).
    async fn publish_submit_idem(
        &self,
        id: uuid::Uuid,
        task: String,
        source: String,
        story_id: Option<String>,
        user_context: &super::types::UserContext,
        idempotency_key: Option<String>,
    ) -> Result<(), Error> {
        let (producer, commands_topic) = match self.require_kafka() {
            Ok(p) => p,
            // HTTP-only mode: task is already in store; no Kafka publish needed.
            Err(_) => return Ok(()),
        };
        let cmd = DaemonCommand::SubmitTask {
            id,
            task,
            source,
            story_id,
            trust_level: None,
            user_id: Some(user_context.user_id.clone()),
            tenant_id: Some(user_context.tenant_id.clone()),
            roles: user_context.roles.clone(),
            mcp_auth_tokens: None,
            idempotency_key,
        };
        let payload = serde_json::to_vec(&cmd)
            .map_err(|e| Error::Daemon(format!("failed to serialize command: {e}")))?;
        producer
            .send(
                FutureRecord::to(commands_topic)
                    .key(&id.to_string())
                    .payload(&payload),
                rdkafka::util::Timeout::Never,
            )
            .await
            .map_err(|(e, _)| Error::Daemon(format!("failed to produce command: {e}")))?;
        Ok(())
    }

    /// Read task from store.
    pub fn get_task(&self, id: uuid::Uuid) -> Result<Option<DaemonTask>, Error> {
        self.store.get(id)
    }

    /// List tasks from store.
    pub fn list_tasks(
        &self,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<DaemonTask>, usize), Error> {
        self.store.list(limit, offset)
    }

    /// Subscribe to real-time events for a task (for SSE).
    pub fn subscribe_events(&self, id: uuid::Uuid) -> Option<broadcast::Receiver<AgentEvent>> {
        self.event_channels.read().get(&id).map(|tx| tx.subscribe())
    }

    /// Register a task directly in the store (for non-Kafka execution paths like Telegram/WS).
    pub fn register_task(
        &self,
        id: uuid::Uuid,
        task: impl Into<String>,
        source: impl Into<String>,
    ) -> Result<(), Error> {
        let daemon_task = DaemonTask::new(id, task, source);
        self.store.insert(daemon_task)
    }

    /// Register a task with user context for multi-tenant isolation.
    pub fn register_task_with_user(
        &self,
        id: uuid::Uuid,
        task: impl Into<String>,
        source: impl Into<String>,
        user_id: impl Into<String>,
        tenant_id: impl Into<String>,
    ) -> Result<(), Error> {
        let daemon_task = DaemonTask::new_with_user(id, task, source, user_id, tenant_id);
        self.store.insert(daemon_task)
    }

    /// Update a registered task's state (for non-Kafka execution paths).
    pub fn update_task(&self, id: uuid::Uuid, f: &dyn Fn(&mut DaemonTask)) -> Result<(), Error> {
        self.store.update(id, f)
    }

    /// List tasks with optional source/state/tenant filters.
    pub fn list_tasks_filtered(
        &self,
        limit: usize,
        offset: usize,
        source: Option<&str>,
        state: Option<TaskState>,
        tenant_id: Option<&str>,
    ) -> Result<(Vec<DaemonTask>, usize), Error> {
        self.store
            .list_filtered(limit, offset, source, state, tenant_id)
    }

    /// Aggregate stats, optionally scoped to a tenant.
    pub fn stats(&self, tenant_id: Option<&str>) -> Result<TaskStats, Error> {
        self.store.stats(tenant_id)
    }

    /// Query usage statistics with filters and grouping.
    pub fn usage_stats(
        &self,
        query: &super::types::UsageQuery,
    ) -> Result<Vec<super::types::UsageRow>, Error> {
        self.store.usage_stats(query)
    }

    /// Produce a `CancelTask` command.
    ///
    /// Returns an error when no Kafka producer is configured (HTTP-only mode).
    pub async fn cancel_task(&self, id: uuid::Uuid) -> Result<(), Error> {
        let (producer, commands_topic) = self.require_kafka()?;

        let cmd = DaemonCommand::CancelTask { id };
        let payload = serde_json::to_vec(&cmd)
            .map_err(|e| Error::Daemon(format!("failed to serialize command: {e}")))?;

        producer
            .send(
                FutureRecord::to(commands_topic)
                    .key(&id.to_string())
                    .payload(&payload),
                rdkafka::util::Timeout::Never,
            )
            .await
            .map_err(|(e, _)| Error::Daemon(format!("failed to produce cancel: {e}")))?;

        Ok(())
    }
}

/// The daemon event loop. Consumes commands from Kafka, executes tasks.
pub struct DaemonCore {
    consumer: StreamConsumer,
    producer: FutureProducer,
    events_topic: String,
    store: Arc<dyn TaskStore>,
    /// Optional Postgres store for audit log retention pruning.
    /// Set via [`DaemonCore::with_postgres_store`] after construction.
    #[cfg(feature = "postgres")]
    postgres_store: Option<Arc<PostgresStore>>,
    /// Audit retention configuration (TOML-driven; env var fallback in `run`).
    #[cfg(feature = "postgres")]
    audit_config: DaemonAuditConfig,
    /// Idempotency-key TTL sweep configuration.
    idempotency_config: IdempotencyConfig,
    semaphore: Arc<Semaphore>,
    event_channels: Arc<parking_lot::RwLock<HashMap<uuid::Uuid, broadcast::Sender<AgentEvent>>>>,
    task_cancels: Arc<parking_lot::RwLock<HashMap<uuid::Uuid, CancellationToken>>>,
    active_tasks: JoinSet<()>,
    cancel: CancellationToken,
    /// B5b: optional per-tenant token tracker for admission-gate checks in the Kafka consumer loop.
    tenant_tracker: Option<Arc<TenantTokenTracker>>,
    /// Kafka commands topic — used to dispatch follow-up commands
    /// (e.g. `PersonaPost` / `MentionPoll` from the schedulers).
    commands_topic: String,
    /// Optional proactive-posts context. When set, `run()` spawns one
    /// `PersonaPostScheduler` per persona at startup and dispatches
    /// `DaemonCommand::PersonaPost` to `handle_persona_post`.
    posts_context: Option<Arc<crate::daemon::PostsContext>>,
    /// Persona mention context — when set, one `MentionPollScheduler` is spawned per
    /// `persona_mentions` entry, and `MentionPoll` / `ReplyDraft` commands are dispatched
    /// to the real free-function handlers.
    mention_context: Option<Arc<super::mention_context::MentionContext>>,
}

impl DaemonCore {
    pub fn new(
        config: &DaemonConfig,
        consumer: StreamConsumer,
        producer: FutureProducer,
        store: Arc<dyn TaskStore>,
        cancel: CancellationToken,
    ) -> (Self, DaemonHandle) {
        let kafka_config = config
            .kafka
            .as_ref()
            .expect("DaemonCore requires [daemon.kafka] config");
        let event_channels = Arc::new(parking_lot::RwLock::new(HashMap::new()));
        let handle = DaemonHandle {
            producer: Some(producer.clone()),
            commands_topic: Some(kafka_config.commands_topic.clone()),
            store: store.clone(),
            event_channels: event_channels.clone(),
            tenant_tracker: None,
        };
        let core = Self {
            consumer,
            producer,
            events_topic: kafka_config.events_topic.clone(),
            store,
            #[cfg(feature = "postgres")]
            postgres_store: None,
            #[cfg(feature = "postgres")]
            audit_config: config.audit.clone(),
            idempotency_config: config.idempotency.clone(),
            semaphore: Arc::new(Semaphore::new(config.max_concurrent_tasks)),
            event_channels,
            task_cancels: Arc::new(parking_lot::RwLock::new(HashMap::new())),
            active_tasks: JoinSet::new(),
            cancel,
            tenant_tracker: None,
            commands_topic: kafka_config.commands_topic.clone(),
            posts_context: None,
            mention_context: None,
        };
        (core, handle)
    }

    /// Attach a Postgres store for cross-task audit retention. When set together
    /// with a `[daemon.audit].retain_days` (or the `HEARTBIT_AUDIT_RETAIN_DAYS`
    /// env-var fallback), the daemon spawns a background task that calls
    /// `PostgresStore::prune_audit` every `prune_interval_minutes` minutes
    /// (default 60).
    #[cfg(feature = "postgres")]
    pub fn with_postgres_store(mut self, store: Arc<PostgresStore>) -> Self {
        self.postgres_store = Some(store);
        self
    }

    /// Attach a per-tenant token tracker for admission-gate checks in the Kafka consumer loop.
    ///
    /// When set, each `SubmitTask` command from Kafka is checked against the tracker
    /// before work begins. On overload, the command is logged and skipped.
    ///
    /// NOTE: The daemon consumer is currently configured with `enable.auto.commit=true`,
    /// so the early-return on overload does NOT prevent offset commit — the offset
    /// auto-commits after the next poll interval regardless. The gate therefore
    /// functions as an observation/logging point that prevents agent dispatch but does
    /// not redeliver the message. True NACK-on-overload requires switching to manual
    /// commits (`enable.auto.offset.store=false` + `consumer.store_offset()` only on
    /// successful processing). Tracked as a follow-up.
    pub fn with_tenant_tracker(mut self, tracker: Arc<TenantTokenTracker>) -> Self {
        self.tenant_tracker = Some(tracker);
        self
    }

    /// Attach a proactive-posts context. After this is set, `run()`
    /// spawns `PersonaPostScheduler` instances for each entry and
    /// dispatches `PersonaPost` commands to the real handler.
    pub fn with_posts_context(mut self, ctx: Arc<crate::daemon::PostsContext>) -> Self {
        self.posts_context = Some(ctx);
        self
    }

    /// Attach persona mention context. When set, one [`super::mention_poll::MentionPollScheduler`]
    /// is spawned per `persona_mentions` entry at the start of `run()`, and
    /// `DaemonCommand::MentionPoll` / `DaemonCommand::ReplyDraft` commands are dispatched to
    /// the real free-function handlers.
    pub fn with_mention_context(
        mut self,
        ctx: Arc<super::mention_context::MentionContext>,
    ) -> Self {
        self.mention_context = Some(ctx);
        self
    }

    /// Run the Kafka consumer loop. Blocks until cancellation.
    ///
    /// `build_runner` is called for each submitted task. It receives the task ID,
    /// task text, source tag, optional story ID, optional trust level, and an
    /// event callback, and returns a future that produces the agent output.
    ///
    /// `on_complete` is an optional callback fired when any task reaches a
    /// terminal state (Completed, Failed, Cancelled). Used for proactive
    /// notifications (e.g. Telegram).
    pub async fn run<F, Fut>(
        mut self,
        build_runner: F,
        on_complete: Option<Arc<OnTaskComplete>>,
    ) -> Result<(), Error>
    where
        F: Fn(
                uuid::Uuid,
                String,
                String,
                Option<String>,
                Option<crate::config::TrustLevel>,
                Arc<dyn Fn(AgentEvent) + Send + Sync>,
                Option<String>,                                    // user_id
                Option<String>,                                    // tenant_id
                Vec<String>,                                       // roles
                Option<std::collections::HashMap<String, String>>, // mcp_auth_tokens
            ) -> Fut
            + Send
            + Sync
            + 'static,
        Fut: Future<Output = Result<AgentOutput, Error>> + Send + 'static,
    {
        // Spawn background audit retention prune task if Postgres + retain_days are configured.
        // TOML [daemon.audit] config is preferred; HEARTBIT_AUDIT_RETAIN_DAYS env var is fallback.
        // Postgres-only: in-memory deployments need to call prune() explicitly.
        #[cfg(feature = "postgres")]
        {
            let retain_days: Option<u32> = self
                .audit_config
                .retain_days
                .or_else(|| {
                    std::env::var("HEARTBIT_AUDIT_RETAIN_DAYS")
                        .ok()
                        .and_then(|v| v.parse().ok())
                })
                .filter(|&d| d > 0);
            if let Some(days) = retain_days {
                if let Some(pg) = self.postgres_store.as_ref() {
                    let pg = pg.clone();
                    let retain = chrono::Duration::days(i64::from(days));
                    let cancel = self.cancel.clone();
                    let interval_secs = self
                        .audit_config
                        .prune_interval_minutes
                        .filter(|&m| m > 0)
                        .unwrap_or(60)
                        .saturating_mul(60);
                    let interval = std::time::Duration::from_secs(interval_secs);
                    tokio::spawn(async move {
                        let mut tick = tokio::time::interval(interval);
                        tick.tick().await; // consume the immediate first tick
                        loop {
                            tokio::select! {
                                _ = cancel.cancelled() => {
                                    tracing::info!("audit prune task: cancellation received, exiting");
                                    break;
                                }
                                _ = tick.tick() => {
                                    match pg.prune_audit(retain).await {
                                        Ok(n) => {
                                            tracing::info!(removed = n, "audit retention prune completed")
                                        }
                                        Err(e) => {
                                            tracing::warn!(error = %e, "audit retention prune failed")
                                        }
                                    }
                                }
                            }
                        }
                    });
                    tracing::info!(
                        retain_days = days,
                        interval_secs,
                        "audit retention background task spawned"
                    );
                } else {
                    tracing::debug!(
                        "audit retain_days set but no Postgres store attached; \
                         audit retention requires a Postgres store — in-memory trails grow without bound until prune is called explicitly"
                    );
                }
            }
        }

        // B5b: idempotency-key TTL sweep task.
        // Operates through the TaskStore trait — no cfg gate needed.
        if let Some(ttl_hours) = self.idempotency_config.ttl_hours {
            let store = self.store.clone();
            let cancel = self.cancel.clone();
            let interval_min = self.idempotency_config.sweep_interval_minutes.unwrap_or(60);
            let interval = std::time::Duration::from_secs(u64::from(interval_min) * 60);
            tokio::spawn(async move {
                let mut tick = tokio::time::interval(interval);
                tick.tick().await; // skip immediate fire
                loop {
                    tokio::select! {
                        _ = cancel.cancelled() => {
                            tracing::info!("idempotency sweep: cancellation received, exiting");
                            break;
                        }
                        _ = tick.tick() => {
                            let cutoff = chrono::Utc::now()
                                - chrono::Duration::hours(i64::from(ttl_hours));
                            match store.sweep_expired_idempotency_keys(cutoff) {
                                Ok(n) if n > 0 => {
                                    tracing::info!(swept = n, "idempotency keys expired")
                                }
                                Ok(_) => {}
                                Err(e) => {
                                    tracing::warn!(error = %e, "idempotency sweep failed")
                                }
                            }
                        }
                    }
                }
            });
            tracing::info!(
                ttl_hours,
                interval_minutes = interval_min,
                "idempotency key TTL sweep task spawned"
            );
        }

        // --- Spawn PersonaPostScheduler instances per configured persona ---
        if let Some(ctx) = self.posts_context.as_ref() {
            for (persona, entry) in ctx.entries.iter() {
                let cfg = heartbit_core::config::PersonaPostsConfig {
                    persona: persona.clone(),
                    enabled: true,
                    post_interval_seconds: entry.interval.as_secs(),
                    interval_jitter_pct: entry.interval_jitter_pct,
                    active_hours: entry.active_hours.clone(),
                    candidates_per_draft: entry.candidates_per_draft,
                    post_history_store: "in_memory".into(),
                    post_history_path: None,
                    post_history_lookback_days: entry.history_lookback.num_days(),
                    topic_brief: entry.topic_brief.clone(),
                };
                let producer: Arc<dyn crate::daemon::CommandProducer> = Arc::new(
                    crate::daemon::KafkaCommandProducer::new(self.producer.clone()),
                );
                let scheduler =
                    crate::daemon::PersonaPostScheduler::new(&cfg, producer, &self.commands_topic);
                let cancel = self.cancel.clone();
                tokio::spawn(scheduler.run(cancel));
                tracing::info!(persona = %persona, "post scheduler spawned");

                // EngagementCollectorScheduler co-spawned per persona —
                // fires `EngagementRefresh` on a separate jittered cadence.
                let engagement_producer: Arc<dyn crate::daemon::CommandProducer> = Arc::new(
                    crate::daemon::KafkaCommandProducer::new(self.producer.clone()),
                );
                let engagement_scheduler = crate::daemon::EngagementCollectorScheduler::new(
                    persona.clone(),
                    entry.engagement_refresh,
                    entry.engagement_jitter_pct,
                    engagement_producer,
                    self.commands_topic.clone(),
                );
                let engagement_cancel = self.cancel.clone();
                tokio::spawn(engagement_scheduler.run(engagement_cancel));
                tracing::info!(persona = %persona, "engagement collector spawned");
            }
        }

        // Spawn one MentionPollScheduler per enabled persona_mentions entry.
        if let Some(ref mc) = self.mention_context {
            let commands_topic = self.commands_topic.clone();
            let producer: Arc<dyn super::CommandProducer> = Arc::new(
                super::kafka::KafkaCommandProducer::new(self.producer.clone()),
            );
            for entry in &mc.entries {
                let cfg = crate::config::PersonaMentionsConfig {
                    persona: entry.persona.clone(),
                    enabled: true,
                    poll_interval_seconds: entry.poll_interval_seconds,
                    user_id: entry.user_id.clone(),
                    candidates_per_reply: entry.candidates_per_reply,
                    mention_store: String::new(),
                    mention_store_path: None,
                    // P1.7 guard config — populated from entry at dispatch time;
                    // these values in the config struct are only used by
                    // MentionPollScheduler (for the interval), not for guards.
                    enable_thread_depth_guard: entry.enable_thread_depth_guard,
                    enable_bot_heuristic_guard: entry.bot_heuristic.is_some(),
                    suspicious_handle_patterns: vec![],
                    min_follower_following_ratio: 0.05,
                    min_account_age_days: 7,
                    bot_heuristic_threshold: 2,
                    per_conversation_max_replies: entry.per_conversation_max_replies,
                    daily_token_budget: entry.daily_token_budget,
                    budget_store: "in_memory".into(),
                    budget_path: None,
                };
                match super::mention_poll::MentionPollScheduler::new(
                    &cfg,
                    producer.clone(),
                    &commands_topic,
                ) {
                    Ok(scheduler) => {
                        let cancel = self.cancel.clone();
                        tokio::spawn(async move { scheduler.run(cancel).await });
                        tracing::info!(
                            persona = %entry.persona,
                            interval_secs = entry.poll_interval_seconds,
                            "MentionPollScheduler spawned"
                        );
                    }
                    Err(e) => {
                        tracing::warn!(
                            persona = %entry.persona,
                            error = %e,
                            "failed to create MentionPollScheduler, skipping entry"
                        );
                    }
                }
            }
        }

        use futures::StreamExt;

        let build_runner = Arc::new(build_runner);

        let mut stream = self.consumer.stream();

        loop {
            tokio::select! {
                _ = self.cancel.cancelled() => {
                    tracing::info!("daemon core shutting down, draining active tasks");
                    while self.active_tasks.join_next().await.is_some() {}
                    break;
                }
                msg = stream.next() => {
                    let Some(msg_result) = msg else {
                        tracing::warn!("kafka consumer stream ended unexpectedly");
                        break;
                    };
                    let msg = match msg_result {
                        Ok(m) => m,
                        Err(e) => {
                            tracing::error!(error = %e, "kafka consumer error");
                            continue;
                        }
                    };
                    let payload = match msg.payload() {
                        Some(p) => p,
                        None => continue,
                    };
                    let cmd: DaemonCommand = match serde_json::from_slice(payload) {
                        Ok(c) => c,
                        Err(e) => {
                            tracing::error!(error = %e, "failed to deserialize daemon command");
                            continue;
                        }
                    };

                    match cmd {
                        DaemonCommand::SubmitTask { id, task, source, story_id, trust_level, user_id, tenant_id, roles, mcp_auth_tokens, idempotency_key } => {
                            // B5b: per-tenant overload gate in the Kafka consumer loop.
                            // NOTE: the consumer uses enable.auto.commit=true, so returning
                            // early here does not prevent the offset from being committed after
                            // the next poll interval. The gate still provides logging/metrics.
                            // True NACK-on-overload requires switching to manual offset commits.
                            if let Some(ref tracker) = self.tenant_tracker {
                                let tid = tenant_id.as_deref().unwrap_or("");
                                let scope = crate::auth::TenantScope::new(tid);
                                let estimated = task.len() / 4 + 4096;
                                if let Err(e) = tracker.reserve(&scope, estimated) {
                                    tracing::warn!(
                                        error = %e,
                                        tenant_id = ?tenant_id,
                                        task_id = %id,
                                        "submit overloaded; skipping task (auto-commit Kafka mode)"
                                    );
                                    continue;
                                }
                                // Drop the reservation immediately — admission-only check.
                            }

                            // Re-insert task if missing (e.g. after restart with message replay).
                            ensure_task_inserted(
                                &self.store,
                                id,
                                &task,
                                &source,
                                user_id.as_deref(),
                                tenant_id.as_deref(),
                                idempotency_key.as_deref(),
                            );

                            let permit = match self.semaphore.clone().acquire_owned().await {
                                Ok(p) => p,
                                Err(_) => break, // semaphore closed
                            };

                            let (tx, _) = broadcast::channel(1024);
                            self.event_channels.write().insert(id, tx.clone());

                            // Per-task cancellation token
                            let task_cancel = CancellationToken::new();
                            self.task_cancels.write().insert(id, task_cancel.clone());

                            // Build on_event that produces to both Kafka and broadcast.
                            // PERF (P-V2-DAEMON-9): pre-compute the Kafka key
                            // string once per task instead of on every emitted
                            // event — UUID → 36-char String repeated dozens
                            // of times per task otherwise.
                            let event_producer = self.producer.clone();
                            let events_topic = self.events_topic.clone();
                            let kafka_key: Arc<str> = Arc::from(id.to_string());
                            // PERF (P-V2-DAEMON-1): per-task `Vec<u8>` pool
                            // for the JSON serialisation buffer. Without it,
                            // every emitted event paid a fresh `to_vec`
                            // alloc + free; at 50–500 events/task that was
                            // 2.5–100 ms of avoidable allocator traffic on
                            // the per-event Kafka publish path. The buffer
                            // is per-task (no cross-task contention) and
                            // reused via `clear()` + `to_writer()`.
                            let event_buf: Arc<parking_lot::Mutex<Vec<u8>>> =
                                Arc::new(parking_lot::Mutex::new(Vec::with_capacity(4096)));
                            let on_event: Arc<dyn Fn(AgentEvent) + Send + Sync> =
                                Arc::new(move |event: AgentEvent| {
                                    // PERF (P-V2-DAEMON-7): skip the broadcast
                                    // clone+send when no SSE subscribers are
                                    // attached. The event is still produced
                                    // to Kafka so downstream consumers get
                                    // the full event stream.
                                    if tx.receiver_count() > 0 {
                                        let _ = tx.send(event.clone());
                                    }
                                    // Fire-and-forget produce to Kafka. The
                                    // `payload` slice is held only across
                                    // the synchronous queue-into-rdkafka
                                    // call; rdkafka copies the bytes
                                    // internally before returning, so the
                                    // pool buffer is free to be reused on
                                    // the next event.
                                    let mut buf = event_buf.lock();
                                    buf.clear();
                                    if let Err(e) =
                                        serde_json::to_writer(&mut *buf, &event)
                                    {
                                        tracing::error!(
                                            error = %e,
                                            "failed to serialize agent event for kafka"
                                        );
                                        return;
                                    }
                                    drop(event_producer.send(
                                        FutureRecord::to(&events_topic)
                                            .key(kafka_key.as_ref())
                                            .payload(buf.as_slice()),
                                        rdkafka::util::Timeout::Never,
                                    ));
                                });

                            let store = self.store.clone();
                            let channels = self.event_channels.clone();
                            let task_cancels = self.task_cancels.clone();
                            let build_runner = build_runner.clone();
                            let on_complete = on_complete.clone();
                            let outcome_story_id = story_id.clone();
                            let outcome_user_id = user_id.clone();
                            let outcome_tenant_id = tenant_id.clone();

                            self.active_tasks.spawn(async move {
                                store
                                    .update(id, &|t| {
                                        t.state = TaskState::Running;
                                        t.started_at = Some(Utc::now());
                                    })
                                    .ok();

                                let start = std::time::Instant::now();
                                let runner = build_runner(id, task, source.clone(), story_id, trust_level, on_event, user_id, tenant_id, roles, mcp_auth_tokens);
                                tokio::select! {
                                    result = runner => {
                                        let duration_secs = start.elapsed().as_secs_f64();
                                        match result {
                                            Ok(output) => {
                                                let tokens = output.tokens_used;
                                                let cost = output.estimated_cost_usd;
                                                let result_text = output.result.clone();
                                                store
                                                    .update(id, &|t| {
                                                        t.state = TaskState::Completed;
                                                        t.completed_at = Some(Utc::now());
                                                        t.result = Some(result_text.clone());
                                                        t.tokens_used = tokens;
                                                        t.tool_calls_made = output.tool_calls_made;
                                                        t.estimated_cost_usd = cost;
                                                        t.model_name = output.model_name.clone();
                                                    })
                                                    .ok();
                                                if let Some(ref cb) = on_complete {
                                                    cb(TaskOutcome {
                                                        id,
                                                        source: source.clone(),
                                                        state: TaskState::Completed,
                                                        result_summary: Some(result_text),
                                                        error: None,
                                                        duration_secs,
                                                        tokens,
                                                        cost,
                                                        story_id: outcome_story_id.clone(),
                                                        user_id: outcome_user_id.clone(),
                                                        tenant_id: outcome_tenant_id.clone(),
                                                    });
                                                }
                                            }
                                            Err(e) => {
                                                let error_str = e.to_string();
                                                store
                                                    .update(id, &|t| {
                                                        t.state = TaskState::Failed;
                                                        t.completed_at = Some(Utc::now());
                                                        t.error = Some(error_str.clone());
                                                    })
                                                    .ok();
                                                if let Some(ref cb) = on_complete {
                                                    cb(TaskOutcome {
                                                        id,
                                                        source: source.clone(),
                                                        state: TaskState::Failed,
                                                        result_summary: None,
                                                        error: Some(error_str),
                                                        duration_secs,
                                                        tokens: Default::default(),
                                                        cost: None,
                                                        story_id: outcome_story_id.clone(),
                                                        user_id: outcome_user_id.clone(),
                                                        tenant_id: outcome_tenant_id.clone(),
                                                    });
                                                }
                                            }
                                        }
                                    }
                                    _ = task_cancel.cancelled() => {
                                        store
                                            .update(id, &|t| {
                                                t.state = TaskState::Cancelled;
                                                t.completed_at = Some(Utc::now());
                                            })
                                            .ok();
                                        if let Some(ref cb) = on_complete {
                                            cb(TaskOutcome {
                                                id,
                                                source: source.clone(),
                                                state: TaskState::Cancelled,
                                                result_summary: None,
                                                error: None,
                                                duration_secs: start.elapsed().as_secs_f64(),
                                                tokens: Default::default(),
                                                cost: None,
                                                story_id: outcome_story_id.clone(),
                                                user_id: outcome_user_id.clone(),
                                                tenant_id: outcome_tenant_id,
                                            });
                                        }
                                    }
                                }

                                channels.write().remove(&id);
                                task_cancels.write().remove(&id);
                                drop(permit);
                            });
                        }
                        DaemonCommand::CancelTask { id } => {
                            // Cancel the running task if it exists
                            if let Some(token) = self.task_cancels.read().get(&id) {
                                token.cancel();
                            }
                            // If task isn't running, just mark cancelled in store
                            if let Ok(Some(task)) = self.store.get(id)
                                && task.state == TaskState::Pending
                            {
                                self.store
                                    .update(id, &|t| {
                                        t.state = TaskState::Cancelled;
                                        t.completed_at = Some(Utc::now());
                                    })
                                    .ok();
                            }
                            self.event_channels.write().remove(&id);
                        }
                        DaemonCommand::MentionPoll { persona, user_id } => {
                            if let Some(ref mc) = self.mention_context {
                                // Find the matching entry by persona + user_id.
                                let entry = mc.entries.iter().find(|e| {
                                    e.persona == persona && e.user_id == user_id
                                });
                                if let Some(entry) = entry {
                                    let store = entry.store.clone();
                                    let spam_guard = entry.spam_guard.clone();
                                    let exec_ctx = entry.exec_ctx.clone();
                                    let max_results = entry.max_results;
                                    let mentions_tool = mc.mentions_tool.clone();
                                    let producer: Arc<dyn super::CommandProducer> = Arc::new(
                                        super::kafka::KafkaCommandProducer::new(self.producer.clone()),
                                    );
                                    let commands_topic = self.commands_topic.clone();
                                    // Build P1.7 guards before spawn so references into
                                    // owned values are valid inside the async block.
                                    let thread_depth_guard =
                                        heartbit_ghost::reply::ThreadDepthGuard::with_enabled(
                                            entry.enable_thread_depth_guard,
                                        );
                                    let bot_heuristic_guard = entry
                                        .bot_heuristic
                                        .clone()
                                        .map(heartbit_ghost::reply::BotHeuristicGuard::new);
                                    let conversation_depth_guard =
                                        heartbit_ghost::reply::ConversationDepthGuard::new(
                                            entry.per_conversation_max_replies,
                                        );
                                    let daily_budget_guard =
                                        heartbit_ghost::reply::DailyBudgetGuard::new(
                                            entry.daily_token_budget,
                                        );
                                    let budget_tracker = entry.budget_tracker.clone();
                                    let x_enricher = mc.enricher.clone();
                                    let enrichment_cache = mc.enrichment_cache.clone();
                                    tokio::spawn(async move {
                                        let deps =
                                            super::mention_poll_handler::MentionPollDeps {
                                                persona: &persona,
                                                user_id: &user_id,
                                                mentions_tool: mentions_tool.as_ref(),
                                                exec_ctx: &exec_ctx,
                                                store: store.as_ref(),
                                                spam_guard: &spam_guard,
                                                producer: producer.as_ref(),
                                                commands_topic: &commands_topic,
                                                max_results,
                                                thread_depth_guard: &thread_depth_guard,
                                                bot_heuristic: bot_heuristic_guard.as_ref(),
                                                conversation_depth_guard: &conversation_depth_guard,
                                                daily_budget_guard: &daily_budget_guard,
                                                budget_tracker: &*budget_tracker,
                                                enricher: x_enricher.as_deref(),
                                                enrichment_cache: enrichment_cache.as_deref(),
                                            };
                                        if let Err(e) =
                                            super::mention_poll_handler::handle_mention_poll(deps)
                                                .await
                                        {
                                            tracing::error!(
                                                persona,
                                                user_id,
                                                error = %e,
                                                "handle_mention_poll failed"
                                            );
                                        }
                                    });
                                } else {
                                    tracing::warn!(
                                        persona,
                                        user_id,
                                        "MentionPoll received but no matching persona entry in context"
                                    );
                                }
                            } else {
                                tracing::warn!(
                                    persona,
                                    user_id,
                                    "MentionPoll received but no MentionContext configured"
                                );
                            }
                        }
                        DaemonCommand::ReplyDraft {
                            persona,
                            mention,
                            parent,
                            mentioner_context,
                        } => {
                            if let Some(ref mc) = self.mention_context {
                                // Find the matching entry for store + candidates_per_reply.
                                let entry =
                                    mc.entries.iter().find(|e| e.persona == persona);
                                if let Some(entry) = entry {
                                    let store = entry.store.clone();
                                    let candidates_per_reply = entry.candidates_per_reply;
                                    let registry = mc.reply.registry.clone();
                                    let provider = mc.reply.provider.clone();
                                    let delivery = mc.reply.delivery.clone();
                                    let twitter_tool = mc.reply.twitter_tool.clone();
                                    let credentials = mc.reply.credentials.clone();
                                    let corpora_root = mc.reply.corpora_root.clone();
                                    let profiles_root = mc.reply.profiles_root.clone();
                                    let budget_tracker = entry.budget_tracker.clone();
                                    let scam_judge = mc.reply.scam_judge.clone();
                                    tokio::spawn(async move {
                                        let deps = super::reply_draft_handler::ReplyDraftDeps {
                                            registry: &registry,
                                            store: store.as_ref(),
                                            provider,
                                            delivery,
                                            twitter_tool,
                                            credentials,
                                            candidates_per_reply,
                                            corpora_root: &corpora_root,
                                            profiles_root: &profiles_root,
                                            budget_tracker,
                                            scam_judge,
                                        };
                                        if let Err(e) =
                                            super::reply_draft_handler::handle_reply_draft(
                                                deps,
                                                &persona,
                                                mention,
                                                parent,
                                                mentioner_context,
                                            )
                                            .await
                                        {
                                            tracing::error!(
                                                persona,
                                                error = %e,
                                                "handle_reply_draft failed"
                                            );
                                        }
                                    });
                                } else {
                                    tracing::warn!(
                                        persona,
                                        mention_id = %mention.id,
                                        "ReplyDraft received but no matching persona entry in context"
                                    );
                                }
                            } else {
                                tracing::warn!(
                                    persona,
                                    mention_id = %mention.id,
                                    "ReplyDraft received but no MentionContext configured"
                                );
                            }
                        }
                        DaemonCommand::PersonaPost { persona } => {
                            let Some(ctx) = self.posts_context.clone() else {
                                tracing::warn!(
                                    persona = %persona,
                                    "PersonaPost received but no posts_context configured"
                                );
                                continue;
                            };
                            let Some(entry) = ctx.entries.get(&persona) else {
                                tracing::warn!(
                                    persona = %persona,
                                    "PersonaPost for unknown persona"
                                );
                                continue;
                            };
                            let history = entry.history.clone();
                            let topic_brief = entry.topic_brief.clone();
                            let candidates_per_draft = entry.candidates_per_draft;
                            let history_lookback = entry.history_lookback;
                            let operator_user_id = entry.operator_user_id.clone();
                            let registry = ctx.registry.clone();
                            let provider = ctx.provider.clone();
                            let delivery = ctx.delivery.clone();
                            let twitter_thread = ctx.twitter_thread.clone();
                            let credentials = ctx.credentials.clone();
                            let corpora_root = ctx.corpora_root.clone();
                            let profiles_root = ctx.profiles_root.clone();
                            let persona_owned = persona.clone();
                            tokio::spawn(async move {
                                let deps = crate::daemon::PersonaPostDeps {
                                    persona_name: &persona_owned,
                                    registry: &registry,
                                    history: history.as_ref(),
                                    history_lookback,
                                    topic_brief: topic_brief.as_deref(),
                                    operator_user_id: &operator_user_id,
                                    provider,
                                    delivery,
                                    twitter_tool: twitter_thread,
                                    credentials,
                                    candidates_per_draft,
                                    corpora_root: &corpora_root,
                                    profiles_root: &profiles_root,
                                };
                                if let Err(e) =
                                    crate::daemon::handle_persona_post(deps).await
                                {
                                    tracing::error!(
                                        persona = %persona_owned,
                                        error = %e,
                                        "persona post handler failed"
                                    );
                                }
                            });
                        }
                        DaemonCommand::EngagementRefresh { persona } => {
                            let Some(ctx) = self.posts_context.clone() else {
                                tracing::warn!(
                                    persona = %persona,
                                    "EngagementRefresh: no posts_context configured"
                                );
                                continue;
                            };
                            let Some(entry) = ctx.entries.get(&persona) else {
                                tracing::warn!(
                                    persona = %persona,
                                    "EngagementRefresh for unknown persona"
                                );
                                continue;
                            };
                            // The XClient is shared with the mentions/reply
                            // pipeline (one OAuth1 user-context client per
                            // daemon). When mention_context (or its enricher)
                            // is absent there's no client to refresh against.
                            let Some(ref mc) = self.mention_context else {
                                tracing::warn!(
                                    persona = %persona,
                                    "EngagementRefresh dropped: no mention_context. Engagement collection reuses the OAuth1 XClient from [[daemon.persona_mentions]]; add a matching mention entry or drop engagement_top_n=0 to silence this."
                                );
                                continue;
                            };
                            let Some(client_arc) = mc.enricher.clone() else {
                                tracing::warn!(
                                    persona = %persona,
                                    "EngagementRefresh dropped: mention_context has no XClient (X_CONSUMER_KEY/SECRET + X_ACCESS_TOKEN/SECRET likely missing in env). Engagement collection cannot run without OAuth1 user-context credentials."
                                );
                                continue;
                            };
                            let history = entry.history.clone();
                            let engagement = entry.engagement_store.clone();
                            let max_age = entry.engagement_max_age_days;
                            let min_age = entry.engagement_min_age_hours;
                            let persona_owned = persona.clone();
                            tokio::spawn(async move {
                                let deps = crate::daemon::EngagementRefreshDeps {
                                    persona: &persona_owned,
                                    client: client_arc.as_ref(),
                                    history: history.as_ref(),
                                    store: engagement.as_ref(),
                                    max_age_days: max_age,
                                    min_age_hours: min_age,
                                };
                                if let Err(e) =
                                    crate::daemon::handle_engagement_refresh(deps).await
                                {
                                    tracing::error!(
                                        persona = %persona_owned,
                                        error = %e,
                                        "engagement refresh handler failed"
                                    );
                                }
                            });
                        }
                    }
                }
                Some(result) = self.active_tasks.join_next() => {
                    if let Err(e) = result {
                        tracing::error!("task panicked: {e}");
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::daemon::store::InMemoryTaskStore;

    fn test_kafka_config() -> crate::config::KafkaConfig {
        crate::config::KafkaConfig {
            brokers: "localhost:9092".into(),
            consumer_group: "test".into(),
            commands_topic: "test.commands".into(),
            events_topic: "test.events".into(),
            dead_letter_topic: "test.dead-letter".into(),
        }
    }

    fn test_config() -> DaemonConfig {
        crate::config::DaemonConfig {
            kafka: Some(test_kafka_config()),
            bind: "127.0.0.1:0".into(),
            max_concurrent_tasks: 4,
            metrics: None,
            database_url: None,
            auth: None,
            memory: crate::config::DaemonMemoryConfig::default(),
            audit: crate::config::DaemonAuditConfig::default(),
            idempotency: crate::config::IdempotencyConfig::default(),
            persona_mentions: vec![],
            persona_posts: vec![],
        }
    }

    fn test_producer() -> FutureProducer {
        crate::daemon::kafka::create_producer(test_config().kafka.as_ref().unwrap()).unwrap()
    }

    fn test_handle() -> DaemonHandle {
        let store: Arc<dyn TaskStore> = Arc::new(InMemoryTaskStore::new());
        DaemonHandle::http_only(store)
    }

    #[tokio::test]
    async fn daemon_core_new_returns_handle() {
        let config = test_config();
        let kafka = config.kafka.as_ref().unwrap();
        let producer = test_producer();
        let consumer = crate::daemon::kafka::create_commands_consumer(kafka).unwrap();
        let store: Arc<dyn TaskStore> = Arc::new(InMemoryTaskStore::new());
        let cancel = CancellationToken::new();

        let (_core, handle) = DaemonCore::new(&config, consumer, producer, store.clone(), cancel);

        // Handle should be usable — get_task returns None for unknown ID
        assert!(handle.get_task(uuid::Uuid::new_v4()).unwrap().is_none());
    }

    #[test]
    fn handle_get_task_returns_inserted() {
        let handle = test_handle();
        let id = uuid::Uuid::new_v4();
        let task = DaemonTask::new(id, "test", "api");
        handle.store.insert(task).unwrap();

        let fetched = handle.get_task(id).unwrap().unwrap();
        assert_eq!(fetched.id, id);
        assert_eq!(fetched.task, "test");
    }

    #[test]
    fn handle_get_task_not_found() {
        let handle = test_handle();
        assert!(handle.get_task(uuid::Uuid::new_v4()).unwrap().is_none());
    }

    #[test]
    fn handle_list_tasks_returns_stored() {
        let handle = test_handle();
        for i in 0..3 {
            handle
                .store
                .insert(DaemonTask::new(
                    uuid::Uuid::new_v4(),
                    format!("task {i}"),
                    "api",
                ))
                .unwrap();
        }

        let (tasks, total) = handle.list_tasks(10, 0).unwrap();
        assert_eq!(total, 3);
        assert_eq!(tasks.len(), 3);
    }

    #[test]
    fn handle_list_tasks_pagination() {
        let handle = test_handle();
        for i in 0..5 {
            handle
                .store
                .insert(DaemonTask::new(
                    uuid::Uuid::new_v4(),
                    format!("task {i}"),
                    "api",
                ))
                .unwrap();
        }

        let (tasks, total) = handle.list_tasks(2, 1).unwrap();
        assert_eq!(total, 5);
        assert_eq!(tasks.len(), 2);
    }

    #[test]
    fn handle_subscribe_events_none_when_no_channel() {
        let handle = test_handle();
        assert!(handle.subscribe_events(uuid::Uuid::new_v4()).is_none());
    }

    #[test]
    fn handle_subscribe_events_returns_receiver() {
        let handle = test_handle();
        let id = uuid::Uuid::new_v4();

        // Insert a broadcast channel
        let (tx, _) = broadcast::channel(16);
        handle.event_channels.write().insert(id, tx);

        let rx = handle.subscribe_events(id);
        assert!(rx.is_some());
    }

    #[test]
    fn register_task_appears_in_store() {
        let handle = test_handle();
        let id = uuid::Uuid::new_v4();
        handle
            .register_task(id, "do something", "telegram")
            .unwrap();

        let task = handle.get_task(id).unwrap().unwrap();
        assert_eq!(task.id, id);
        assert_eq!(task.task, "do something");
        assert_eq!(task.source, "telegram");
        assert_eq!(task.state, TaskState::Pending);
    }

    #[test]
    fn register_task_duplicate_rejected() {
        let handle = test_handle();
        let id = uuid::Uuid::new_v4();
        handle.register_task(id, "first", "api").unwrap();
        let err = handle.register_task(id, "second", "api").unwrap_err();
        assert!(err.to_string().contains("already exists"));
    }

    #[test]
    fn update_task_modifies_state() {
        let handle = test_handle();
        let id = uuid::Uuid::new_v4();
        handle.register_task(id, "test", "ws").unwrap();

        handle
            .update_task(id, &|t| {
                t.state = TaskState::Running;
                t.started_at = Some(chrono::Utc::now());
            })
            .unwrap();

        let task = handle.get_task(id).unwrap().unwrap();
        assert_eq!(task.state, TaskState::Running);
        assert!(task.started_at.is_some());
    }

    #[test]
    fn update_task_nonexistent_returns_error() {
        let handle = test_handle();
        let err = handle
            .update_task(uuid::Uuid::new_v4(), &|_| {})
            .unwrap_err();
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn handle_list_tasks_filtered_by_source() {
        let handle = test_handle();
        for i in 0..3 {
            let source = if i < 2 { "telegram" } else { "api" };
            handle
                .register_task(uuid::Uuid::new_v4(), format!("task {i}"), source)
                .unwrap();
        }

        let (tasks, total) = handle
            .list_tasks_filtered(10, 0, Some("telegram"), None, None)
            .unwrap();
        assert_eq!(total, 2);
        assert_eq!(tasks.len(), 2);
        assert!(tasks.iter().all(|t| t.source == "telegram"));
    }

    #[test]
    fn handle_list_tasks_filtered_by_state() {
        let handle = test_handle();
        let id1 = uuid::Uuid::new_v4();
        let id2 = uuid::Uuid::new_v4();
        handle.register_task(id1, "a", "api").unwrap();
        handle.register_task(id2, "b", "api").unwrap();
        handle
            .update_task(id1, &|t| t.state = TaskState::Running)
            .unwrap();

        let (tasks, total) = handle
            .list_tasks_filtered(10, 0, None, Some(TaskState::Running), None)
            .unwrap();
        assert_eq!(total, 1);
        assert_eq!(tasks[0].id, id1);
    }

    #[test]
    fn handle_stats_aggregates() {
        let handle = test_handle();
        let id1 = uuid::Uuid::new_v4();
        let id2 = uuid::Uuid::new_v4();
        let id3 = uuid::Uuid::new_v4();
        handle.register_task(id1, "a", "api").unwrap();
        handle.register_task(id2, "b", "telegram").unwrap();
        handle.register_task(id3, "c", "api").unwrap();
        handle
            .update_task(id2, &|t| t.state = TaskState::Running)
            .unwrap();
        handle
            .update_task(id3, &|t| {
                t.state = TaskState::Completed;
                t.tokens_used.input_tokens = 100;
                t.estimated_cost_usd = Some(0.01);
            })
            .unwrap();

        let stats = handle.stats(None).unwrap();
        assert_eq!(stats.total_tasks, 3);
        assert_eq!(stats.active_tasks, 1);
        assert_eq!(stats.tasks_by_source.get("api"), Some(&2));
        assert_eq!(stats.tasks_by_source.get("telegram"), Some(&1));
        assert_eq!(
            stats.tasks_by_state.get(TaskState::Running.as_str()),
            Some(&1)
        );
        assert_eq!(
            stats.tasks_by_state.get(TaskState::Completed.as_str()),
            Some(&1)
        );
        assert_eq!(stats.total_input_tokens, 100);
        assert!((stats.total_estimated_cost_usd - 0.01).abs() < 1e-9);
    }

    #[test]
    fn handle_list_filtered_by_tenant() {
        let handle = test_handle();
        let task1 = DaemonTask::new_with_user(uuid::Uuid::new_v4(), "a", "api", "alice", "acme");
        let task2 = DaemonTask::new_with_user(uuid::Uuid::new_v4(), "b", "api", "bob", "globex");
        let task3 =
            DaemonTask::new_with_user(uuid::Uuid::new_v4(), "c", "telegram", "carol", "acme");
        handle.store.insert(task1).unwrap();
        handle.store.insert(task2).unwrap();
        handle.store.insert(task3).unwrap();

        let (tasks, total) = handle
            .list_tasks_filtered(10, 0, None, None, Some("acme"))
            .unwrap();
        assert_eq!(total, 2);
        assert_eq!(tasks.len(), 2);
        assert!(tasks.iter().all(|t| t.tenant_id.as_deref() == Some("acme")));
    }

    #[test]
    fn handle_stats_filtered_by_tenant() {
        let handle = test_handle();
        let id1 = uuid::Uuid::new_v4();
        let id2 = uuid::Uuid::new_v4();
        let task1 = DaemonTask::new_with_user(id1, "a", "api", "alice", "acme");
        let task2 = DaemonTask::new_with_user(id2, "b", "api", "bob", "globex");
        handle.store.insert(task1).unwrap();
        handle.store.insert(task2).unwrap();
        handle
            .update_task(id1, &|t| t.tokens_used.input_tokens = 50)
            .unwrap();
        handle
            .update_task(id2, &|t| t.tokens_used.input_tokens = 100)
            .unwrap();

        let stats = handle.stats(Some("acme")).unwrap();
        assert_eq!(stats.total_tasks, 1);
        assert_eq!(stats.total_input_tokens, 50);

        let stats = handle.stats(None).unwrap();
        assert_eq!(stats.total_tasks, 2);
        assert_eq!(stats.total_input_tokens, 150);
    }

    #[test]
    fn register_task_with_user_stores_user_context() {
        let handle = test_handle();
        let id = uuid::Uuid::new_v4();
        let task = DaemonTask::new_with_user(id, "check deals", "api", "alice", "acme");
        handle.store.insert(task).unwrap();

        let fetched = handle.get_task(id).unwrap().unwrap();
        assert_eq!(fetched.user_id.as_deref(), Some("alice"));
        assert_eq!(fetched.tenant_id.as_deref(), Some("acme"));
        assert_eq!(fetched.task, "check deals");
    }

    #[test]
    fn register_task_without_user_has_none_context() {
        let handle = test_handle();
        let id = uuid::Uuid::new_v4();
        handle.register_task(id, "basic task", "api").unwrap();

        let task = handle.get_task(id).unwrap().unwrap();
        assert!(task.user_id.is_none());
        assert!(task.tenant_id.is_none());
    }

    #[test]
    fn register_task_with_user_method_stores_context() {
        let handle = test_handle();
        let id = uuid::Uuid::new_v4();
        handle
            .register_task_with_user(id, "user task", "ws", "bob", "globex")
            .unwrap();

        let task = handle.get_task(id).unwrap().unwrap();
        assert_eq!(task.user_id.as_deref(), Some("bob"));
        assert_eq!(task.tenant_id.as_deref(), Some("globex"));
        assert_eq!(task.source, "ws");
    }

    #[tokio::test]
    async fn daemon_core_new_semaphore_matches_config() {
        let mut config = test_config();
        config.max_concurrent_tasks = 2;
        let kafka = config.kafka.as_ref().unwrap();
        let producer = test_producer();
        let consumer = crate::daemon::kafka::create_commands_consumer(kafka).unwrap();
        let store: Arc<dyn TaskStore> = Arc::new(InMemoryTaskStore::new());
        let cancel = CancellationToken::new();

        let (core, _handle) = DaemonCore::new(&config, consumer, producer, store, cancel);

        // Semaphore should have 2 permits
        let p1 = core.semaphore.clone().try_acquire_owned();
        let p2 = core.semaphore.clone().try_acquire_owned();
        let p3 = core.semaphore.clone().try_acquire_owned();
        assert!(p1.is_ok());
        assert!(p2.is_ok());
        assert!(p3.is_err()); // third should fail — only 2 permits
    }

    #[tokio::test]
    async fn http_only_handle_submit_returns_error() {
        let store: Arc<dyn TaskStore> = Arc::new(InMemoryTaskStore::new());
        let handle = DaemonHandle::http_only(store);
        let err = handle.submit_task("test", "api", None).await.unwrap_err();
        assert!(err.to_string().contains(KAFKA_REQUIRED));
    }

    #[tokio::test]
    async fn http_only_handle_cancel_returns_error() {
        let store: Arc<dyn TaskStore> = Arc::new(InMemoryTaskStore::new());
        let handle = DaemonHandle::http_only(store);
        let err = handle.cancel_task(uuid::Uuid::new_v4()).await.unwrap_err();
        assert!(err.to_string().contains(KAFKA_REQUIRED));
    }

    #[test]
    fn http_only_handle_register_and_read_works() {
        let store: Arc<dyn TaskStore> = Arc::new(InMemoryTaskStore::new());
        let handle = DaemonHandle::http_only(store);
        let id = uuid::Uuid::new_v4();
        handle.register_task(id, "test", "execute").unwrap();
        let task = handle.get_task(id).unwrap().unwrap();
        assert_eq!(task.task, "test");
        assert_eq!(task.source, "execute");
    }

    fn test_handle_with_store() -> DaemonHandle {
        let store: Arc<dyn TaskStore> = Arc::new(InMemoryTaskStore::new());
        DaemonHandle::http_only(store)
    }

    fn make_user_ctx(user_id: &str, tenant_id: &str) -> crate::daemon::types::UserContext {
        crate::daemon::types::UserContext {
            user_id: user_id.into(),
            tenant_id: tenant_id.into(),
            roles: vec![],
            raw_token: None,
        }
    }

    #[tokio::test]
    async fn submit_with_idem_returns_same_task_id_on_redelivery() {
        let handle = test_handle_with_store();
        let user_ctx = make_user_ctx("user-1", "tenant-A");

        let id1 = handle
            .submit_task_with_user_idem("hello", "test", None, &user_ctx, Some("idem-xyz"))
            .await
            .expect("first submit");

        let id2 = handle
            .submit_task_with_user_idem("hello again", "test", None, &user_ctx, Some("idem-xyz"))
            .await
            .expect("redelivery");

        assert_eq!(id1, id2, "redelivery must dedup to same task id");
    }

    #[tokio::test]
    async fn submit_with_idem_isolates_per_tenant() {
        let handle = test_handle_with_store();
        let user_a = make_user_ctx("u", "tenant-A");
        let user_b = make_user_ctx("u", "tenant-B");

        let id_a = handle
            .submit_task_with_user_idem("x", "test", None, &user_a, Some("shared-key"))
            .await
            .unwrap();
        let id_b = handle
            .submit_task_with_user_idem("x", "test", None, &user_b, Some("shared-key"))
            .await
            .unwrap();

        assert_ne!(id_a, id_b, "different tenants must NOT collide on same key");
    }

    #[tokio::test]
    async fn submit_without_idem_creates_new_task_each_time() {
        let handle = test_handle_with_store();
        let user_ctx = make_user_ctx("u", "tenant-A");

        let id1 = handle
            .submit_task_with_user_idem("x", "test", None, &user_ctx, None)
            .await
            .unwrap();
        let id2 = handle
            .submit_task_with_user_idem("x", "test", None, &user_ctx, None)
            .await
            .unwrap();

        assert_ne!(id1, id2, "no idem key → no dedup");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn concurrent_submits_same_key_resolve_to_same_id() {
        let handle = std::sync::Arc::new(test_handle_with_store());
        let user_ctx = std::sync::Arc::new(make_user_ctx("u", "tenant-A"));

        let mut joinset = tokio::task::JoinSet::new();
        for _ in 0..10 {
            let h = handle.clone();
            let u = user_ctx.clone();
            joinset.spawn(async move {
                h.submit_task_with_user_idem("payload", "test", None, &u, Some("race-key"))
                    .await
            });
        }

        let mut ids = std::collections::HashSet::new();
        while let Some(res) = joinset.join_next().await {
            ids.insert(res.unwrap().unwrap());
        }
        assert_eq!(
            ids.len(),
            1,
            "all concurrent submits must dedup to one id, got {ids:?}"
        );
    }

    // --- B5b Task 7: per-tenant overload gate tests ---

    #[tokio::test(flavor = "multi_thread")]
    async fn submit_returns_tenant_overloaded_when_tracker_full() {
        let handle = test_handle_with_store();
        let tracker = Arc::new(TenantTokenTracker::new(100));
        let handle = handle.with_tenant_tracker(tracker.clone());

        // Pre-fill the tracker for tenant-A: 99 tokens in-flight, cap=100.
        let scope = crate::auth::TenantScope::new("tenant-A");
        let _hold = tracker.reserve(&scope, 99).unwrap();

        let user_ctx = make_user_ctx("u", "tenant-A");
        // Estimate for "hello" = 5 / 4 + 4096 = 4097, which exceeds remaining cap of 1.
        let err = handle
            .submit_task_with_user_idem("hello", "test", None, &user_ctx, None)
            .await
            .unwrap_err();
        match err {
            crate::Error::TenantOverloaded { tenant_id, .. } => {
                assert_eq!(tenant_id, "tenant-A");
            }
            other => panic!("expected TenantOverloaded, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn submit_succeeds_without_tracker() {
        let handle = test_handle_with_store();
        let user_ctx = make_user_ctx("u", "tenant-A");
        // No tracker; should succeed unconditionally.
        handle
            .submit_task_with_user_idem("x", "test", None, &user_ctx, None)
            .await
            .expect("should succeed without tracker");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn submit_drops_reservation_after_admission_check() {
        let handle = test_handle_with_store();
        let tracker = Arc::new(TenantTokenTracker::new(1_000_000));
        let handle = handle.with_tenant_tracker(tracker.clone());
        let user_ctx = make_user_ctx("u", "tenant-A");

        let _id = handle
            .submit_task_with_user_idem("hello", "test", None, &user_ctx, None)
            .await
            .unwrap();

        // After admission check, the reservation drops immediately.
        // The runner's adjust() handles actual usage tracking.
        // The entry exists with in_flight=0 (tracker creates it via or_default).
        let snap = tracker.snapshot();
        assert_eq!(
            snap.iter()
                .find(|(t, _)| t == "tenant-A")
                .map(|(_, s)| s.in_flight),
            Some(0)
        );
    }

    /// Pins the "no second insert" half of the publish-only-on-fresh-insert contract.
    ///
    /// We can't observe the Kafka publish in `http_only` mode (no producer), and
    /// `DaemonHandle.producer: Option<FutureProducer>` doesn't accept the
    /// `ChannelCommandProducer` mock without a wider refactor. The publish-only-on-fresh-insert
    /// guarantee is held by `submit_task_with_user_idem`'s early return (line ~196):
    /// the lookup-hit path returns BEFORE `publish_submit_idem` is called. This test
    /// asserts the observable consequence — exactly one row exists in the store after
    /// two calls with the same key. Tracked as follow-up: refactor `DaemonHandle.producer`
    /// to `Arc<dyn CommandProducer>` so a channel-backed end-to-end publish test is possible.
    #[tokio::test]
    async fn submit_with_idem_does_not_create_duplicate_row() {
        let handle = test_handle_with_store();
        let user_ctx = make_user_ctx("u", "tenant-A");

        handle
            .submit_task_with_user_idem("hello", "test", None, &user_ctx, Some("k"))
            .await
            .unwrap();
        handle
            .submit_task_with_user_idem("hello again", "test", None, &user_ctx, Some("k"))
            .await
            .unwrap();

        let (rows, total) = handle.store.list(100, 0).unwrap();
        assert_eq!(rows.len(), 1, "redelivery must not insert a duplicate row");
        assert_eq!(total, 1);
        // Idempotency key must persist on the surviving row so subsequent lookups still hit.
        assert_eq!(rows[0].idempotency_key.as_deref(), Some("k"));
    }

    /// I3: the consumer match arm reconstructs a missing row with `idempotency_key`,
    /// `user_id`, and `tenant_id` preserved. Exercised here through the extracted
    /// `ensure_task_inserted` helper so we don't need to drive a full Kafka loop.
    #[test]
    fn ensure_task_inserted_reconstructs_missing_row_with_idempotency_key() {
        let store: Arc<dyn TaskStore> = Arc::new(InMemoryTaskStore::new());
        let id = uuid::Uuid::new_v4();

        ensure_task_inserted(
            &store,
            id,
            "do the thing",
            "kafka",
            Some("alice"),
            Some("acme"),
            Some("k"),
        );

        let task = store.get(id).unwrap().expect("row was inserted");
        assert_eq!(task.task, "do the thing");
        assert_eq!(task.source, "kafka");
        assert_eq!(task.user_id.as_deref(), Some("alice"));
        assert_eq!(task.tenant_id.as_deref(), Some("acme"));
        assert_eq!(task.idempotency_key.as_deref(), Some("k"));
    }

    #[test]
    fn ensure_task_inserted_skips_when_row_already_exists() {
        let store: Arc<dyn TaskStore> = Arc::new(InMemoryTaskStore::new());
        let id = uuid::Uuid::new_v4();
        // Pre-seed an existing row with a DIFFERENT task body to detect overwrite.
        let existing = DaemonTask::new(id, "original", "api");
        store.insert(existing).unwrap();

        ensure_task_inserted(
            &store,
            id,
            "replayed",
            "kafka",
            Some("alice"),
            Some("acme"),
            Some("k"),
        );

        let task = store.get(id).unwrap().unwrap();
        assert_eq!(task.task, "original", "must not overwrite existing row");
        assert!(task.user_id.is_none());
        assert!(task.idempotency_key.is_none());
    }

    #[test]
    fn ensure_task_inserted_without_user_context_uses_anonymous_constructor() {
        let store: Arc<dyn TaskStore> = Arc::new(InMemoryTaskStore::new());
        let id = uuid::Uuid::new_v4();

        ensure_task_inserted(&store, id, "anon", "cron", None, None, Some("k"));

        let task = store.get(id).unwrap().unwrap();
        assert!(task.user_id.is_none());
        assert!(task.tenant_id.is_none());
        assert_eq!(task.idempotency_key.as_deref(), Some("k"));
    }

    // --- P1.5 Task 13: mention context wiring tests ---

    /// `with_mention_context` stores the Arc in `DaemonCore.mention_context`.
    /// Uses an empty-entries context to keep this a pure builder-pattern unit test.
    #[tokio::test]
    async fn daemon_core_with_mention_context_stores_context() {
        use heartbit_core::error::Error as CoreError;
        use heartbit_core::execution_context::{CredentialResolver as CredResolverTrait, Secret};
        use heartbit_core::llm::types::{CompletionRequest, CompletionResponse, ToolDefinition};
        use heartbit_core::llm::{BoxedProvider, LlmProvider};
        use heartbit_core::tool::ToolOutput;
        use heartbit_core::{ExecutionContext, Tool};
        use heartbit_ghost::reply::{ReplyOutcome, ReplyReviewDelivery, ReplyReviewMessage};
        use heartbit_ghost::review::{
            DeliveredReview, DeliveryOutcome, DeliveryReceipt, ReviewDeliveryError,
        };

        use crate::daemon::mention_context::{MentionContext, ReplySharedContext};

        struct NopProvider;
        impl LlmProvider for NopProvider {
            async fn complete(
                &self,
                _: CompletionRequest,
            ) -> Result<CompletionResponse, CoreError> {
                Err(CoreError::Daemon("nop".into()))
            }
        }
        struct NopTool;
        impl Tool for NopTool {
            fn definition(&self) -> ToolDefinition {
                ToolDefinition {
                    name: "nop".into(),
                    description: "nop".into(),
                    input_schema: serde_json::json!({}),
                }
            }
            fn execute(
                &self,
                _: &ExecutionContext,
                _: serde_json::Value,
            ) -> std::pin::Pin<
                Box<dyn std::future::Future<Output = Result<ToolOutput, CoreError>> + Send + '_>,
            > {
                Box::pin(async { Ok(ToolOutput::success(String::new())) })
            }
        }
        struct NopDelivery;
        impl ReplyReviewDelivery for NopDelivery {
            fn deliver<'a>(
                &'a self,
                _: ReplyReviewMessage,
            ) -> std::pin::Pin<
                Box<
                    dyn std::future::Future<Output = Result<DeliveredReview, ReviewDeliveryError>>
                        + Send
                        + 'a,
                >,
            > {
                Box::pin(async {
                    Ok(DeliveredReview {
                        outcome: DeliveryOutcome::Pick(0),
                        receipt: DeliveryReceipt {
                            data: serde_json::Value::Null,
                        },
                    })
                })
            }
            fn report<'a>(
                &'a self,
                _: DeliveryReceipt,
                _: ReplyOutcome,
            ) -> std::pin::Pin<
                Box<dyn std::future::Future<Output = Result<(), ReviewDeliveryError>> + Send + 'a>,
            > {
                Box::pin(async { Ok(()) })
            }
        }
        struct NopCreds;
        impl CredResolverTrait for NopCreds {
            fn resolve(
                &self,
                _: &str,
            ) -> std::pin::Pin<
                Box<dyn std::future::Future<Output = Result<Secret, CoreError>> + Send + '_>,
            > {
                Box::pin(async { Err(CoreError::Daemon("nop".into())) })
            }
        }

        let config = test_config();
        let kafka = config.kafka.as_ref().unwrap();
        let producer = test_producer();
        let consumer = crate::daemon::kafka::create_commands_consumer(kafka).unwrap();
        let store: Arc<dyn TaskStore> = Arc::new(InMemoryTaskStore::new());
        let cancel = CancellationToken::new();

        let (core, _handle) = DaemonCore::new(&config, consumer, producer, store, cancel);
        assert!(core.mention_context.is_none(), "initially None");

        let ctx = Arc::new(MentionContext {
            entries: vec![],
            reply: ReplySharedContext {
                registry: Arc::new(crate::persona::PersonaRegistry::new()),
                provider: Arc::new(BoxedProvider::new(NopProvider)),
                delivery: Arc::new(NopDelivery),
                twitter_tool: Arc::new(NopTool),
                credentials: Arc::new(NopCreds),
                corpora_root: std::path::PathBuf::from("/tmp"),
                profiles_root: std::path::PathBuf::from("/tmp"),
                scam_judge: None,
            },
            mentions_tool: Arc::new(NopTool),
            enricher: None,
            enrichment_cache: None,
        });

        let core = core.with_mention_context(ctx);
        assert!(
            core.mention_context.is_some(),
            "mention_context must be Some after with_mention_context()"
        );
    }
}
