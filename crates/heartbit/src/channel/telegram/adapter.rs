use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use teloxide::prelude::*;
use teloxide::types::{ChatId, ParseMode};

use crate::channel::session::{SessionMessage, SessionRole, SessionStore, format_session_context};
use crate::error::Error;
use crate::memory::{Memory, MemoryQuery};

use super::access::AccessControl;
use super::bridge::TelegramBridge;
use super::config::TelegramConfig;
use super::delivery::{RateLimiter, chunk_message, markdown_to_telegram_html};
use super::keyboard::{CallbackAction, parse_callback_data};
use super::router::{ChatSessionMap, IdleSession};

/// Callback type for running an agent task with bridge callbacks.
///
/// The CLI crate provides this closure to wire `build_orchestrator_from_config`
/// with the Telegram bridge callbacks. Returns the agent's final text output.
pub type RunTask = dyn Fn(RunTaskInput) -> Pin<Box<dyn Future<Output = Result<String, Error>> + Send>>
    + Send
    + Sync;

/// Input for the `RunTask` callback.
pub struct RunTaskInput {
    pub task_text: String,
    pub bridge: Arc<TelegramBridge>,
    /// Pre-existing shared memory store so sub-agent memory tools persist
    /// across tasks. Passed as the raw (un-namespaced) store.
    pub memory: Option<Arc<dyn Memory>>,
    /// User-specific namespace prefix (e.g. `"tg:12345"`). Passed as `story_id`
    /// to `build_orchestrator_from_config` for per-user memory isolation.
    pub user_namespace: Option<String>,
}

/// Callback type for memory consolidation on idle sessions.
pub type ConsolidateSession =
    dyn Fn(i64) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send>> + Send + Sync;

/// The main Telegram adapter that handles the bot lifecycle.
pub struct TelegramAdapter {
    bot: Bot,
    config: TelegramConfig,
    access: AccessControl,
    sessions: ChatSessionMap,
    session_store: Arc<dyn SessionStore>,
    memory: Option<Arc<dyn Memory>>,
    run_task: Arc<RunTask>,
    consolidate: Option<Arc<ConsolidateSession>>,
    max_concurrent: Arc<tokio::sync::Semaphore>,
    /// Active bridges keyed by chat_id, used for resolving inline keyboard callbacks.
    active_bridges: Arc<RwLock<HashMap<i64, Arc<TelegramBridge>>>>,
}

impl TelegramAdapter {
    pub fn new(
        bot: Bot,
        config: TelegramConfig,
        session_store: Arc<dyn SessionStore>,
        memory: Option<Arc<dyn Memory>>,
        run_task: Arc<RunTask>,
        consolidate: Option<Arc<ConsolidateSession>>,
    ) -> Self {
        let access = AccessControl::new(config.dm_policy, config.allowed_users.clone());
        let inactivity = Duration::from_secs(config.inactivity_timeout_seconds);
        let expiry = Duration::from_secs(config.session_expiry_seconds);
        let sessions = ChatSessionMap::new(inactivity, expiry);
        let max_concurrent = Arc::new(tokio::sync::Semaphore::new(config.max_concurrent));

        Self {
            bot,
            config,
            access,
            sessions,
            session_store,
            memory,
            run_task,
            consolidate,
            max_concurrent,
            active_bridges: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Start the Telegram bot with long polling and a background sweep loop.
    ///
    /// This method blocks until the cancellation token is triggered.
    pub async fn run(self: Arc<Self>, cancel: tokio_util::sync::CancellationToken) {
        let adapter = Arc::clone(&self);

        // Spawn sweep loop
        let sweep_adapter = Arc::clone(&self);
        let sweep_cancel = cancel.clone();
        tokio::spawn(async move {
            sweep_adapter.sweep_loop(sweep_cancel).await;
        });

        // Set up teloxide dispatcher with separate branches for messages and callbacks
        let handler = dptree::entry()
            .branch(Update::filter_message().endpoint(handle_message))
            .branch(Update::filter_callback_query().endpoint(handle_callback));

        let bot = self.bot.clone();

        // Use Dispatcher for long-polling
        Dispatcher::builder(bot, handler)
            .dependencies(dptree::deps![adapter])
            .default_handler(|_upd| async {})
            .enable_ctrlc_handler()
            .build()
            .dispatch()
            .await;
    }

    /// Pre-load relevant memories for a user and format them as context.
    ///
    /// Uses `agent_prefix` to match all sub-agent namespaces under this user
    /// (e.g. `"tg:123"` matches `"tg:123:assistant"`, `"tg:123:researcher"`).
    async fn preload_memories(&self, user_id: i64, message: &str) -> String {
        let memory = match &self.memory {
            Some(m) => m,
            None => return String::new(),
        };

        // Skip memory recall for trivial messages (greetings, short acks)
        if self.config.memory_skip_trivial && message.len() < 20 && !message.contains('?') {
            return String::new();
        }

        let query = MemoryQuery {
            text: Some(message.to_string()),
            limit: self.config.memory_recall_limit,
            agent_prefix: Some(format!("tg:{user_id}")),
            ..Default::default()
        };

        match memory.recall(query).await {
            Ok(entries) if !entries.is_empty() => {
                let mut ctx = String::from("## Relevant context from memory\n");
                for entry in &entries {
                    ctx.push_str(&format!("- {}\n", entry.content));
                }
                ctx.push('\n');
                ctx
            }
            _ => String::new(),
        }
    }

    /// Handle an incoming text message from a Telegram DM.
    async fn handle_dm(&self, chat_id: i64, user_id: i64, text: &str) -> Result<(), Error> {
        // Access check
        if !self.access.check_dm(user_id) {
            tracing::debug!(user_id, "telegram DM denied by access policy");
            return Ok(());
        }

        // Acquire concurrency permit
        let _permit = self
            .max_concurrent
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| Error::Telegram("max concurrent sessions reached".into()))?;

        // Get or create session
        let (session_id, _is_new) =
            self.sessions
                .get_or_create(chat_id, user_id, self.session_store.as_ref())?;
        self.sessions.touch(chat_id);

        // Add user message to session
        self.session_store.add_message(
            session_id,
            SessionMessage {
                role: SessionRole::User,
                content: text.to_string(),
                timestamp: chrono::Utc::now(),
            },
        )?;

        // Load session history
        let session = self
            .session_store
            .get(session_id)?
            .ok_or_else(|| Error::Telegram("session disappeared".into()))?;

        // Pre-load memories
        let memory_ctx = self.preload_memories(user_id, text).await;

        // Format context with history (exclude the just-added user message from history)
        let history = if session.messages.len() > 1 {
            &session.messages[..session.messages.len() - 1]
        } else {
            &[]
        };
        let session_ctx = format_session_context(history, text);

        // Combine memory context + session context
        let task_text = if memory_ctx.is_empty() {
            session_ctx
        } else {
            format!("{memory_ctx}{session_ctx}")
        };

        // Create bridge for this chat and register it for callback resolution
        let bridge = Arc::new(TelegramBridge::new(
            self.bot.clone(),
            chat_id,
            self.config.stream_debounce_ms,
            Duration::from_secs(self.config.interaction_timeout_seconds),
        ));
        {
            if let Ok(mut bridges) = self.active_bridges.write() {
                bridges.insert(chat_id, Arc::clone(&bridge));
            }
        }

        // Run agent task — pass the raw shared memory store. User isolation is
        // handled by the orchestrator's story_id namespace (via RunTaskInput). We
        // don't wrap in NamespacedMemory here because the orchestrator already wraps
        // per-agent, and nested NamespacedMemory breaks recall filtering.
        let memory = self.memory.clone();
        let input = RunTaskInput {
            task_text,
            bridge: Arc::clone(&bridge),
            memory,
            user_namespace: Some(format!("tg:{user_id}")),
        };
        let result = (self.run_task)(input).await;

        // Unregister bridge, but only if it's still ours (not replaced by a concurrent request)
        if let Ok(mut bridges) = self.active_bridges.write()
            && let Some(current) = bridges.get(&chat_id)
            && Arc::ptr_eq(current, &bridge)
        {
            bridges.remove(&chat_id);
        }

        // Delete the streaming preview message (plain-text, potentially truncated)
        // and send the final output as formatted HTML.
        bridge.delete_stream_message().await;

        match result {
            Ok(output) => {
                if !output.is_empty() {
                    self.send_html(chat_id, &output).await;
                    let _ = self.session_store.add_message(
                        session_id,
                        SessionMessage {
                            role: SessionRole::Assistant,
                            content: output,
                            timestamp: chrono::Utc::now(),
                        },
                    );
                }
            }
            Err(e) => {
                let error_msg = format!("Error: {e}");
                let _ = self.bot.send_message(ChatId(chat_id), &error_msg).await;
                let _ = self.session_store.add_message(
                    session_id,
                    SessionMessage {
                        role: SessionRole::Assistant,
                        content: error_msg,
                        timestamp: chrono::Utc::now(),
                    },
                );
            }
        }

        bridge.reset_stream().await;
        Ok(())
    }

    /// Handle a callback query from an inline keyboard.
    ///
    /// Looks up the active bridge for the given chat and resolves the
    /// pending approval or question interaction.
    fn handle_callback_for_chat(&self, chat_id: i64, callback_data: &str) -> Result<(), Error> {
        let action = parse_callback_data(callback_data)?;

        let bridge = {
            let bridges = self
                .active_bridges
                .read()
                .map_err(|e| Error::Telegram(format!("lock poisoned: {e}")))?;
            bridges
                .get(&chat_id)
                .cloned()
                .ok_or_else(|| Error::Telegram(format!("no active bridge for chat {chat_id}")))?
        };

        match action {
            CallbackAction::Approval {
                interaction_id,
                decision,
            } => {
                let decision = match decision.as_str() {
                    "allow" => crate::llm::ApprovalDecision::Allow,
                    "always_allow" => crate::llm::ApprovalDecision::AlwaysAllow,
                    _ => crate::llm::ApprovalDecision::Deny,
                };
                bridge.resolve_approval(interaction_id, decision)
            }
            CallbackAction::QuestionAnswer {
                interaction_id,
                question_idx,
                option_idx,
            } => bridge.resolve_question_by_index(interaction_id, question_idx, option_idx),
        }
    }

    /// Send output as HTML-formatted Telegram messages.
    ///
    /// Converts markdown to Telegram HTML, chunks for the 4096-char limit,
    /// and falls back to plain text if Telegram rejects the HTML.
    async fn send_html(&self, chat_id: i64, text: &str) {
        let html = markdown_to_telegram_html(text);
        let chunks = chunk_message(&html);
        let mut limiter = RateLimiter::new(1000);

        for chunk in &chunks {
            let delay = limiter.check();
            if !delay.is_zero() {
                tokio::time::sleep(delay).await;
            }
            let result = self
                .bot
                .send_message(ChatId(chat_id), *chunk)
                .parse_mode(ParseMode::Html)
                .await;
            if result.is_err() {
                // HTML rejected — fall back to plain text for remaining chunks
                self.send_plain(chat_id, text).await;
                return;
            }
            limiter.record_send();
        }
    }

    /// Fallback: send as plain text chunks (no formatting).
    async fn send_plain(&self, chat_id: i64, text: &str) {
        let chunks = chunk_message(text);
        let mut limiter = RateLimiter::new(1000);
        for chunk in chunks {
            let delay = limiter.check();
            if !delay.is_zero() {
                tokio::time::sleep(delay).await;
            }
            let _ = self.bot.send_message(ChatId(chat_id), chunk).await;
            limiter.record_send();
        }
    }

    /// Background loop that sweeps idle and expired sessions.
    async fn sweep_loop(&self, cancel: tokio_util::sync::CancellationToken) {
        let mut interval = tokio::time::interval(Duration::from_secs(60));
        loop {
            tokio::select! {
                _ = interval.tick() => {}
                _ = cancel.cancelled() => {
                    tracing::info!("telegram sweep loop shutting down");
                    return;
                }
            }

            // Consolidate idle sessions
            let idle = self.sessions.collect_idle();
            for IdleSession {
                chat_id,
                session_id: _,
                user_id,
            } in &idle
            {
                tracing::debug!(chat_id, user_id, "consolidating idle telegram session");
                if let Some(consolidate) = &self.consolidate
                    && let Err(e) = consolidate(*user_id).await
                {
                    tracing::warn!(user_id, error = %e, "memory consolidation failed");
                }
            }

            // Clean up expired sessions
            let expired = self.sessions.collect_expired();
            for exp in &expired {
                tracing::info!(
                    chat_id = exp.chat_id,
                    "cleaning up expired telegram session"
                );
                let _ = self.session_store.delete(exp.session_id);
                self.sessions.remove(exp.chat_id);
            }
        }
    }
}

/// Teloxide handler for incoming messages.
async fn handle_message(
    msg: Message,
    adapter: Arc<TelegramAdapter>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Only handle private (DM) messages
    if !msg.chat.is_private() {
        return Ok(());
    }

    let text = match msg.text() {
        Some(t) if !t.is_empty() => t,
        _ => return Ok(()),
    };

    let user_id = msg.from.as_ref().map(|u| u.id.0 as i64).unwrap_or_default();

    if let Err(e) = adapter.handle_dm(msg.chat.id.0, user_id, text).await {
        tracing::error!(
            chat_id = msg.chat.id.0,
            error = %e,
            "telegram DM handler error"
        );
    }
    Ok(())
}

/// Teloxide handler for callback queries (inline keyboard presses).
async fn handle_callback(
    q: CallbackQuery,
    adapter: Arc<TelegramAdapter>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if let Some(data) = &q.data {
        // Extract chat_id from the message that contained the inline keyboard
        let chat_id = q
            .message
            .as_ref()
            .map(|m| m.chat().id.0)
            .unwrap_or_default();

        tracing::debug!(chat_id, data, "telegram callback query received");

        if let Err(e) = adapter.handle_callback_for_chat(chat_id, data) {
            tracing::warn!(chat_id, data, error = %e, "callback resolution failed");
        }

        // Answer the callback to dismiss the loading indicator
        let _ = adapter.bot.answer_callback_query(q.id).await;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::session::InMemorySessionStore;

    fn make_adapter(memory: Option<Arc<dyn Memory>>) -> Arc<TelegramAdapter> {
        make_adapter_with_config(
            memory,
            TelegramConfig {
                dm_policy: super::super::config::DmPolicy::Open,
                max_concurrent: 5,
                ..Default::default()
            },
        )
    }

    fn make_adapter_with_config(
        memory: Option<Arc<dyn Memory>>,
        config: TelegramConfig,
    ) -> Arc<TelegramAdapter> {
        let bot = Bot::new("0:AAAA-test-token");
        let store: Arc<dyn SessionStore> = Arc::new(InMemorySessionStore::new());
        let run_task: Arc<RunTask> =
            Arc::new(|input| Box::pin(async move { Ok(format!("Echo: {}", input.task_text)) }));

        Arc::new(TelegramAdapter::new(
            bot, config, store, memory, run_task, None,
        ))
    }

    #[tokio::test]
    async fn preload_memories_empty_when_no_memory() {
        let adapter = make_adapter(None);
        let ctx = adapter.preload_memories(123, "hello").await;
        assert!(ctx.is_empty());
    }

    #[tokio::test]
    async fn preload_memories_with_store() {
        use crate::memory::in_memory::InMemoryStore;
        use crate::memory::namespaced::NamespacedMemory;
        use crate::memory::{MemoryEntry, MemoryType};

        let store = Arc::new(InMemoryStore::new());

        // Store a memory entry via a sub-agent compound namespace (tg:123:assistant).
        // This simulates what happens when the orchestrator's sub-agent stores a memory
        // with memory_namespace_prefix="tg:123" and agent_name="assistant".
        let ns = NamespacedMemory::new(Arc::clone(&store) as Arc<dyn Memory>, "tg:123:assistant");
        let entry = MemoryEntry {
            id: "test-1".into(),
            agent: String::new(),
            content: "User likes Rust programming".into(),
            category: "general".into(),
            tags: vec!["rust".into()],
            created_at: chrono::Utc::now(),
            last_accessed: chrono::Utc::now(),
            access_count: 0,
            importance: 5,
            memory_type: MemoryType::Episodic,
            keywords: vec!["rust".into(), "programming".into()],
            summary: None,
            strength: 1.0,
            related_ids: Vec::new(),
            source_ids: Vec::new(),
            embedding: None,
        };
        ns.store(entry).await.unwrap();

        // preload_memories uses agent_prefix="tg:123" which matches "tg:123:assistant"
        let adapter = make_adapter(Some(store));
        let ctx = adapter.preload_memories(123, "tell me about Rust").await;
        assert!(ctx.contains("Relevant context from memory"));
        assert!(ctx.contains("Rust programming"));
    }

    #[test]
    fn access_control_integrated() {
        let bot = Bot::new("0:AAAA-test-token");
        let config = TelegramConfig {
            dm_policy: super::super::config::DmPolicy::Allowlist,
            allowed_users: vec![111, 222],
            ..Default::default()
        };
        let store: Arc<dyn SessionStore> = Arc::new(InMemorySessionStore::new());
        let run_task: Arc<RunTask> = Arc::new(|_| Box::pin(async { Ok("ok".into()) }));

        let adapter = TelegramAdapter::new(bot, config, store, None, run_task, None);
        assert!(adapter.access.check_dm(111));
        assert!(!adapter.access.check_dm(333));
    }

    #[test]
    fn session_map_created_with_config_timeouts() {
        let bot = Bot::new("0:AAAA-test-token");
        let config = TelegramConfig {
            inactivity_timeout_seconds: 600,
            session_expiry_seconds: 3600,
            max_concurrent: 10,
            ..Default::default()
        };
        let store: Arc<dyn SessionStore> = Arc::new(InMemorySessionStore::new());
        let run_task: Arc<RunTask> = Arc::new(|_| Box::pin(async { Ok("ok".into()) }));

        let adapter = TelegramAdapter::new(bot, config, store, None, run_task, None);
        assert!(adapter.sessions.is_empty());
    }

    #[tokio::test]
    async fn send_html_handles_short_message() {
        let adapter = make_adapter(None);
        // This will fail to actually send (no real bot), but shouldn't panic
        adapter.send_html(12345, "Hello **world**").await;
    }

    #[test]
    fn handle_callback_for_chat_no_active_bridge() {
        let adapter = make_adapter(None);
        let err = adapter
            .handle_callback_for_chat(999, "a:00000000-0000-0000-0000-000000000000:allow")
            .unwrap_err();
        assert!(err.to_string().contains("no active bridge"));
    }

    #[test]
    fn handle_callback_for_chat_resolves_approval() {
        let adapter = make_adapter(None);
        let chat_id = 42;

        // Create a bridge with a manually-inserted pending approval
        let bridge = Arc::new(TelegramBridge::new(
            Bot::new("0:AAAA-test-token"),
            chat_id,
            500,
            Duration::from_secs(5),
        ));
        let id = uuid::Uuid::new_v4();
        let (tx, rx) = std::sync::mpsc::channel();

        // Use bridge's test helper: inject pending via the internal API
        bridge.inject_pending_approval(id, tx);

        // Register bridge in adapter
        {
            let mut bridges = adapter.active_bridges.write().unwrap();
            bridges.insert(chat_id, Arc::clone(&bridge));
        }

        // Resolve via handle_callback_for_chat
        let callback_data = format!("a:{id}:allow");
        adapter
            .handle_callback_for_chat(chat_id, &callback_data)
            .unwrap();

        let decision = rx.recv().unwrap();
        assert_eq!(decision, crate::llm::ApprovalDecision::Allow);
    }

    #[test]
    fn handle_callback_for_chat_resolves_question_by_index() {
        let adapter = make_adapter(None);
        let chat_id = 42;

        let bridge = Arc::new(TelegramBridge::new(
            Bot::new("0:AAAA-test-token"),
            chat_id,
            500,
            Duration::from_secs(5),
        ));
        let id = uuid::Uuid::new_v4();
        let (tx, rx) = tokio::sync::oneshot::channel();

        // Use bridge's test helper: inject pending question with stored options
        bridge.inject_pending_question(
            id,
            tx,
            vec![vec!["Red".into(), "Green".into(), "Blue".into()]],
        );

        // Register bridge in adapter
        {
            let mut bridges = adapter.active_bridges.write().unwrap();
            bridges.insert(chat_id, Arc::clone(&bridge));
        }

        // Resolve question: question 0, option 2 = "Blue"
        let callback_data = format!("q:{id}:0:2");
        adapter
            .handle_callback_for_chat(chat_id, &callback_data)
            .unwrap();

        let result = rx.blocking_recv().unwrap().unwrap();
        assert_eq!(result.answers, vec![vec!["Blue".to_string()]]);
    }

    #[test]
    fn handle_callback_for_chat_invalid_data() {
        let adapter = make_adapter(None);
        let err = adapter.handle_callback_for_chat(42, "garbage").unwrap_err();
        assert!(err.to_string().contains("unknown callback prefix"));
    }

    #[test]
    fn bridge_unregister_does_not_remove_replacement() {
        let adapter = make_adapter(None);
        let chat_id = 42;

        // Create two bridges for the same chat (simulating concurrent DMs)
        let bridge1 = Arc::new(TelegramBridge::new(
            Bot::new("0:AAAA-test-token"),
            chat_id,
            500,
            Duration::from_secs(5),
        ));
        let bridge2 = Arc::new(TelegramBridge::new(
            Bot::new("0:AAAA-test-token"),
            chat_id,
            500,
            Duration::from_secs(5),
        ));

        // Register bridge1, then bridge2 overwrites it
        {
            let mut bridges = adapter.active_bridges.write().unwrap();
            bridges.insert(chat_id, Arc::clone(&bridge1));
        }
        {
            let mut bridges = adapter.active_bridges.write().unwrap();
            bridges.insert(chat_id, Arc::clone(&bridge2));
        }

        // Simulate bridge1's task completing — it should NOT remove bridge2
        {
            let mut bridges = adapter.active_bridges.write().unwrap();
            if let Some(current) = bridges.get(&chat_id) {
                if Arc::ptr_eq(current, &bridge1) {
                    bridges.remove(&chat_id);
                }
            }
        }

        // bridge2 should still be registered
        let bridges = adapter.active_bridges.read().unwrap();
        assert!(bridges.contains_key(&chat_id));
        assert!(Arc::ptr_eq(bridges.get(&chat_id).unwrap(), &bridge2));
    }

    #[tokio::test]
    async fn preload_memories_skips_trivial_when_enabled() {
        use crate::memory::in_memory::InMemoryStore;
        use crate::memory::namespaced::NamespacedMemory;
        use crate::memory::{MemoryEntry, MemoryType};

        let store = Arc::new(InMemoryStore::new());
        let ns = NamespacedMemory::new(Arc::clone(&store) as Arc<dyn Memory>, "tg:123:assistant");
        let entry = MemoryEntry {
            id: "test-1".into(),
            agent: String::new(),
            content: "User likes Rust".into(),
            category: "general".into(),
            tags: vec!["rust".into()],
            created_at: chrono::Utc::now(),
            last_accessed: chrono::Utc::now(),
            access_count: 0,
            importance: 5,
            memory_type: MemoryType::Episodic,
            keywords: vec!["rust".into()],
            summary: None,
            strength: 1.0,
            related_ids: Vec::new(),
            source_ids: Vec::new(),
            embedding: None,
        };
        ns.store(entry).await.unwrap();

        let config = TelegramConfig {
            dm_policy: super::super::config::DmPolicy::Open,
            max_concurrent: 5,
            memory_skip_trivial: true,
            ..Default::default()
        };
        let adapter = make_adapter_with_config(Some(store), config);

        // Short trivial message (< 20 chars, no '?') → skipped
        let ctx = adapter.preload_memories(123, "hello").await;
        assert!(ctx.is_empty(), "trivial message should skip memory recall");

        // Longer message with question mark → not skipped, recall runs
        let ctx = adapter
            .preload_memories(123, "can you tell me about Rust programming?")
            .await;
        assert!(
            ctx.contains("Relevant context from memory"),
            "non-trivial message should recall memories, got: {ctx:?}"
        );
    }

    #[tokio::test]
    async fn preload_memories_respects_recall_limit() {
        use crate::memory::in_memory::InMemoryStore;
        use crate::memory::namespaced::NamespacedMemory;
        use crate::memory::{MemoryEntry, MemoryType};

        let store = Arc::new(InMemoryStore::new());
        let ns = NamespacedMemory::new(Arc::clone(&store) as Arc<dyn Memory>, "tg:42:assistant");

        // Store 10 entries
        for i in 0..10 {
            let entry = MemoryEntry {
                id: format!("mem-{i}"),
                agent: String::new(),
                content: format!("Memory entry number {i} about programming"),
                category: "general".into(),
                tags: vec!["programming".into()],
                created_at: chrono::Utc::now(),
                last_accessed: chrono::Utc::now(),
                access_count: 0,
                importance: 5,
                memory_type: MemoryType::Episodic,
                keywords: vec!["programming".into()],
                summary: None,
                strength: 1.0,
                related_ids: Vec::new(),
                source_ids: Vec::new(),
                embedding: None,
            };
            ns.store(entry).await.unwrap();
        }

        // With limit=3, should return at most 3 entries
        let config = TelegramConfig {
            dm_policy: super::super::config::DmPolicy::Open,
            max_concurrent: 5,
            memory_recall_limit: 3,
            ..Default::default()
        };
        let adapter = make_adapter_with_config(Some(store), config);

        let ctx = adapter
            .preload_memories(42, "tell me about programming")
            .await;
        let entry_count = ctx.matches("- Memory entry").count();
        assert!(
            entry_count <= 3,
            "expected at most 3 entries but got {entry_count}"
        );
    }
}
