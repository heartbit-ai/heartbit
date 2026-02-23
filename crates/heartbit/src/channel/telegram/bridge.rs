use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use teloxide::prelude::*;
use teloxide::types::{ChatId, InlineKeyboardButton, InlineKeyboardMarkup, MessageId};
use uuid::Uuid;

use crate::agent::events::{AgentEvent, OnEvent};
use crate::error::Error;
use crate::llm::{ApprovalDecision, OnApproval, OnText};
use crate::tool::builtins::{OnQuestion, QuestionRequest, QuestionResponse};

use super::delivery::StreamBuffer;
use super::keyboard::{approval_buttons, question_buttons};

/// Sender half for pending interactions.
enum PendingSender {
    Approval(std::sync::mpsc::Sender<ApprovalDecision>),
    Question(tokio::sync::oneshot::Sender<Result<QuestionResponse, Error>>),
}

/// Telegram-specific interaction bridge for a single chat.
///
/// Creates heartbit callback closures that stream text via `edit_message_text`,
/// show inline keyboards for approvals/questions, and log events via tracing.
pub struct TelegramBridge {
    pending: RwLock<HashMap<Uuid, PendingSender>>,
    /// Stored question option labels keyed by interaction ID.
    /// Used to resolve callback indices back to label text.
    question_options: RwLock<HashMap<Uuid, Vec<Vec<String>>>>,
    bot: Bot,
    chat_id: ChatId,
    stream_buffer: tokio::sync::Mutex<StreamBuffer>,
    timeout: Duration,
}

impl TelegramBridge {
    pub fn new(bot: Bot, chat_id: i64, debounce_ms: u64, timeout: Duration) -> Self {
        Self {
            pending: RwLock::new(HashMap::new()),
            question_options: RwLock::new(HashMap::new()),
            bot,
            chat_id: ChatId(chat_id),
            stream_buffer: tokio::sync::Mutex::new(StreamBuffer::new(chat_id, debounce_ms)),
            timeout,
        }
    }

    /// Create an `OnText` callback that accumulates deltas and does debounced edits.
    pub fn make_on_text(self: &Arc<Self>) -> Arc<OnText> {
        let bridge = Arc::clone(self);
        Arc::new(move |text: &str| {
            let bridge = Arc::clone(&bridge);
            let text = text.to_string();
            // Fire-and-forget: spawn the async edit work
            if let Ok(handle) = tokio::runtime::Handle::try_current() {
                handle.spawn(async move {
                    bridge.handle_text_delta(&text).await;
                });
            }
        })
    }

    /// Create an `OnEvent` callback that logs to tracing (not sent to user).
    pub fn make_on_event(self: &Arc<Self>) -> Arc<OnEvent> {
        Arc::new(|event: AgentEvent| {
            tracing::debug!(?event, "telegram agent event");
        })
    }

    /// Create an `OnApproval` callback that sends an inline keyboard and blocks.
    pub fn make_on_approval(self: &Arc<Self>) -> Arc<OnApproval> {
        let bridge = Arc::clone(self);
        Arc::new(move |tool_calls: &[crate::llm::types::ToolCall]| {
            let interaction_id = Uuid::new_v4();
            let (tx, rx) = std::sync::mpsc::channel();

            // Build tool call summary for the message
            let summary: String = tool_calls
                .iter()
                .map(|tc| format!("• `{}`", tc.name))
                .collect::<Vec<_>>()
                .join("\n");

            // Store pending
            {
                let mut pending = bridge.pending.write().expect("pending lock not poisoned");
                pending.insert(interaction_id, PendingSender::Approval(tx));
            }

            // Send inline keyboard (fire-and-forget async)
            let bot = bridge.bot.clone();
            let chat_id = bridge.chat_id;
            let buttons = approval_buttons(interaction_id);
            if let Ok(handle) = tokio::runtime::Handle::try_current() {
                handle.spawn(async move {
                    let keyboard = InlineKeyboardMarkup::new(vec![
                        buttons
                            .into_iter()
                            .map(|(label, data)| InlineKeyboardButton::callback(label, data))
                            .collect::<Vec<_>>(),
                    ]);
                    let msg = format!("🔧 Tool approval needed:\n{summary}");
                    let _ = bot.send_message(chat_id, msg).reply_markup(keyboard).await;
                });
            }

            // Block waiting for callback
            match rx.recv_timeout(bridge.timeout) {
                Ok(decision) => decision,
                Err(_) => {
                    bridge.cleanup_pending(interaction_id);
                    ApprovalDecision::Deny
                }
            }
        })
    }

    /// Create an `OnQuestion` callback that sends inline keyboard options.
    pub fn make_on_question(self: &Arc<Self>) -> Arc<OnQuestion> {
        let bridge = Arc::clone(self);
        Arc::new(move |request: QuestionRequest| {
            let bridge = Arc::clone(&bridge);
            Box::pin(async move {
                let interaction_id = Uuid::new_v4();
                let (tx, rx) = tokio::sync::oneshot::channel();

                // Build question data for keyboard
                let question_data: Vec<(String, Vec<String>)> = request
                    .questions
                    .iter()
                    .map(|q| {
                        let options: Vec<String> =
                            q.options.iter().map(|o| o.label.clone()).collect();
                        (q.question.clone(), options)
                    })
                    .collect();

                // Store pending + question option labels for index→label resolution
                {
                    let options_per_question: Vec<Vec<String>> =
                        question_data.iter().map(|(_, opts)| opts.clone()).collect();
                    let mut pending = bridge.pending.write().expect("pending lock not poisoned");
                    pending.insert(interaction_id, PendingSender::Question(tx));
                    let mut qo = bridge
                        .question_options
                        .write()
                        .expect("question_options lock not poisoned");
                    qo.insert(interaction_id, options_per_question);
                }

                let buttons = question_buttons(interaction_id, &question_data);

                // Build message text
                let msg_text: String = request
                    .questions
                    .iter()
                    .map(|q| format!("❓ {}", q.question))
                    .collect::<Vec<_>>()
                    .join("\n");

                // Send inline keyboard
                let keyboard = InlineKeyboardMarkup::new(vec![
                    buttons
                        .into_iter()
                        .map(|(label, data)| InlineKeyboardButton::callback(label, data))
                        .collect::<Vec<_>>(),
                ]);
                let _ = bridge
                    .bot
                    .send_message(bridge.chat_id, msg_text)
                    .reply_markup(keyboard)
                    .await;

                // Await with timeout
                match tokio::time::timeout(bridge.timeout, rx).await {
                    Ok(Ok(result)) => result,
                    Ok(Err(_)) => {
                        bridge.cleanup_pending(interaction_id);
                        Err(Error::Telegram("question channel closed".into()))
                    }
                    Err(_) => {
                        bridge.cleanup_pending(interaction_id);
                        Err(Error::Telegram("question timed out".into()))
                    }
                }
            })
        })
    }

    /// Resolve a pending approval interaction.
    pub fn resolve_approval(&self, id: Uuid, decision: ApprovalDecision) -> Result<(), Error> {
        let sender = self.take_pending(id)?;
        match sender {
            PendingSender::Approval(tx) => {
                let _ = tx.send(decision);
                Ok(())
            }
            other => {
                drop(other);
                Err(Error::Telegram(format!(
                    "interaction {id} is not an approval interaction"
                )))
            }
        }
    }

    /// Resolve a pending question interaction.
    pub fn resolve_question(&self, id: Uuid, response: QuestionResponse) -> Result<(), Error> {
        let sender = self.take_pending(id)?;
        match sender {
            PendingSender::Question(tx) => {
                let _ = tx.send(Ok(response));
                Ok(())
            }
            other => {
                drop(other);
                Err(Error::Telegram(format!(
                    "interaction {id} is not a question interaction"
                )))
            }
        }
    }

    /// Resolve a pending question interaction using indices from callback data.
    ///
    /// Looks up stored question option labels to convert `(question_idx, option_idx)`
    /// into the actual label text expected by `QuestionResponse`.
    pub fn resolve_question_by_index(
        &self,
        id: Uuid,
        question_idx: usize,
        option_idx: usize,
    ) -> Result<(), Error> {
        // Look up stored option labels
        let label = {
            let qo = self
                .question_options
                .read()
                .map_err(|e| Error::Telegram(format!("lock poisoned: {e}")))?;
            let options = qo.get(&id).ok_or_else(|| {
                Error::Telegram(format!("no stored question options for interaction {id}"))
            })?;
            let q_opts = options.get(question_idx).ok_or_else(|| {
                Error::Telegram(format!(
                    "question index {question_idx} out of range (have {})",
                    options.len()
                ))
            })?;
            q_opts.get(option_idx).cloned().ok_or_else(|| {
                Error::Telegram(format!(
                    "option index {option_idx} out of range (have {})",
                    q_opts.len()
                ))
            })?
        };

        let response = QuestionResponse {
            answers: vec![vec![label]],
        };
        self.resolve_question(id, response)
    }

    /// Flush any remaining buffered text by sending/editing the message.
    ///
    /// Returns `true` if the stream buffer had content (i.e., streaming was active).
    /// Callers should skip sending the final output separately when this returns `true`.
    pub async fn flush_stream(&self) -> bool {
        let mut buf = self.stream_buffer.lock().await;
        if buf.is_empty() {
            return false;
        }
        let full_text = buf.current_text().to_string();
        // Truncate to Telegram limit (UTF-8 safe)
        let text = if full_text.len() > 4096 {
            let boundary = super::delivery::floor_char_boundary(&full_text, 4096);
            &full_text[..boundary]
        } else {
            &full_text
        };
        match buf.message_id() {
            Some(msg_id) => {
                let _ = self
                    .bot
                    .edit_message_text(self.chat_id, MessageId(msg_id), text)
                    .await;
            }
            None => {
                if let Ok(sent) = self.bot.send_message(self.chat_id, text).await {
                    buf.set_message_id(sent.id.0);
                }
            }
        }
        buf.mark_edited();
        true
    }

    /// Get the current stream message ID (if streaming has started).
    pub async fn stream_message_id(&self) -> Option<i32> {
        let buf = self.stream_buffer.lock().await;
        buf.message_id()
    }

    /// Delete the streamed message. Used before sending formatted chunks.
    pub async fn delete_stream_message(&self) {
        let buf = self.stream_buffer.lock().await;
        if let Some(msg_id) = buf.message_id() {
            let _ = self
                .bot
                .delete_message(self.chat_id, MessageId(msg_id))
                .await;
        }
    }

    /// Reset the stream buffer for a new message cycle.
    pub async fn reset_stream(&self) {
        let mut buf = self.stream_buffer.lock().await;
        buf.reset();
    }

    // --- Internal helpers ---

    async fn handle_text_delta(&self, text: &str) {
        let mut buf = self.stream_buffer.lock().await;
        let should_edit = buf.push(text);
        if !should_edit {
            return;
        }

        let current = buf.current_text().to_string();
        // Truncate to Telegram limit for edit_message_text (UTF-8 safe)
        let truncated = if current.len() > 4096 {
            let boundary = super::delivery::floor_char_boundary(&current, 4096);
            &current[..boundary]
        } else {
            &current
        };

        match buf.message_id() {
            Some(msg_id) => {
                let _ = self
                    .bot
                    .edit_message_text(self.chat_id, MessageId(msg_id), truncated)
                    .await;
            }
            None => {
                if let Ok(sent) = self.bot.send_message(self.chat_id, truncated).await {
                    buf.set_message_id(sent.id.0);
                }
            }
        }
        buf.mark_edited();
    }

    fn take_pending(&self, id: Uuid) -> Result<PendingSender, Error> {
        let mut pending = self
            .pending
            .write()
            .map_err(|e| Error::Telegram(format!("lock poisoned: {e}")))?;
        // Also clean up stored question options
        if let Ok(mut qo) = self.question_options.write() {
            qo.remove(&id);
        }
        pending
            .remove(&id)
            .ok_or_else(|| Error::Telegram(format!("no pending interaction with id {id}")))
    }

    fn cleanup_pending(&self, id: Uuid) {
        if let Ok(mut pending) = self.pending.write() {
            pending.remove(&id);
        }
        if let Ok(mut qo) = self.question_options.write() {
            qo.remove(&id);
        }
    }

    /// Inject a pending approval for testing. Used by adapter tests.
    #[cfg(test)]
    pub(crate) fn inject_pending_approval(
        &self,
        id: Uuid,
        tx: std::sync::mpsc::Sender<ApprovalDecision>,
    ) {
        let mut pending = self.pending.write().expect("pending lock not poisoned");
        pending.insert(id, PendingSender::Approval(tx));
    }

    /// Inject a pending question with stored option labels for testing.
    #[cfg(test)]
    pub(crate) fn inject_pending_question(
        &self,
        id: Uuid,
        tx: tokio::sync::oneshot::Sender<Result<QuestionResponse, Error>>,
        options: Vec<Vec<String>>,
    ) {
        let mut pending = self.pending.write().expect("pending lock not poisoned");
        pending.insert(id, PendingSender::Question(tx));
        let mut qo = self
            .question_options
            .write()
            .expect("question_options lock not poisoned");
        qo.insert(id, options);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: TelegramBridge requires a real Bot instance for full integration testing.
    // These unit tests cover the pending interaction resolve/cleanup logic
    // without making network calls.

    fn make_bridge() -> Arc<TelegramBridge> {
        // Bot::new panics if token is empty, so use a dummy token
        let bot = Bot::new("0:AAAA-test-token");
        Arc::new(TelegramBridge::new(bot, 12345, 500, Duration::from_secs(5)))
    }

    #[test]
    fn resolve_approval_unknown_id() {
        let bridge = make_bridge();
        let err = bridge
            .resolve_approval(Uuid::new_v4(), ApprovalDecision::Allow)
            .unwrap_err();
        assert!(err.to_string().contains("no pending interaction"));
    }

    #[test]
    fn resolve_question_unknown_id() {
        let bridge = make_bridge();
        let err = bridge
            .resolve_question(
                Uuid::new_v4(),
                QuestionResponse {
                    answers: vec![vec!["A".into()]],
                },
            )
            .unwrap_err();
        assert!(err.to_string().contains("no pending interaction"));
    }

    #[test]
    fn resolve_approval_success() {
        let bridge = make_bridge();
        let id = Uuid::new_v4();
        let (tx, rx) = std::sync::mpsc::channel();

        // Manually insert a pending approval
        {
            let mut pending = bridge.pending.write().unwrap();
            pending.insert(id, PendingSender::Approval(tx));
        }

        bridge
            .resolve_approval(id, ApprovalDecision::Allow)
            .unwrap();
        let decision = rx.recv().unwrap();
        assert_eq!(decision, ApprovalDecision::Allow);
    }

    #[test]
    fn resolve_question_success() {
        let bridge = make_bridge();
        let id = Uuid::new_v4();
        let (tx, rx) = tokio::sync::oneshot::channel();

        // Manually insert a pending question
        {
            let mut pending = bridge.pending.write().unwrap();
            pending.insert(id, PendingSender::Question(tx));
        }

        let response = QuestionResponse {
            answers: vec![vec!["Blue".into()]],
        };
        bridge.resolve_question(id, response).unwrap();
        let result = rx.blocking_recv().unwrap();
        let resp = result.unwrap();
        assert_eq!(resp.answers, vec![vec!["Blue".to_string()]]);
    }

    #[test]
    fn resolve_wrong_type_approval_on_question() {
        let bridge = make_bridge();
        let id = Uuid::new_v4();
        let (_tx, _rx) = tokio::sync::oneshot::channel::<Result<QuestionResponse, Error>>();

        {
            let mut pending = bridge.pending.write().unwrap();
            pending.insert(id, PendingSender::Question(_tx));
        }

        let err = bridge
            .resolve_approval(id, ApprovalDecision::Allow)
            .unwrap_err();
        assert!(err.to_string().contains("not an approval interaction"));
    }

    #[test]
    fn resolve_wrong_type_question_on_approval() {
        let bridge = make_bridge();
        let id = Uuid::new_v4();
        let (tx, _rx) = std::sync::mpsc::channel();

        {
            let mut pending = bridge.pending.write().unwrap();
            pending.insert(id, PendingSender::Approval(tx));
        }

        let err = bridge
            .resolve_question(
                id,
                QuestionResponse {
                    answers: vec![vec!["A".into()]],
                },
            )
            .unwrap_err();
        assert!(err.to_string().contains("not a question interaction"));
    }

    #[test]
    fn cleanup_pending_removes_entry() {
        let bridge = make_bridge();
        let id = Uuid::new_v4();
        let (tx, _rx) = std::sync::mpsc::channel();

        {
            let mut pending = bridge.pending.write().unwrap();
            pending.insert(id, PendingSender::Approval(tx));
        }

        bridge.cleanup_pending(id);

        let err = bridge
            .resolve_approval(id, ApprovalDecision::Allow)
            .unwrap_err();
        assert!(err.to_string().contains("no pending interaction"));
    }

    #[test]
    fn cleanup_nonexistent_is_noop() {
        let bridge = make_bridge();
        bridge.cleanup_pending(Uuid::new_v4()); // Should not panic
    }

    #[test]
    fn make_on_event_does_not_panic() {
        let bridge = make_bridge();
        let on_event = bridge.make_on_event();
        // Should not panic
        on_event(AgentEvent::RunStarted {
            agent: "test".into(),
            task: "test task".into(),
        });
    }

    #[test]
    fn resolve_question_by_index_success() {
        let bridge = make_bridge();
        let id = Uuid::new_v4();
        let (tx, rx) = tokio::sync::oneshot::channel();

        // Insert pending question + stored options
        {
            let mut pending = bridge.pending.write().unwrap();
            pending.insert(id, PendingSender::Question(tx));
            let mut qo = bridge.question_options.write().unwrap();
            qo.insert(
                id,
                vec![vec!["Red".into(), "Blue".into()], vec!["Small".into()]],
            );
        }

        bridge.resolve_question_by_index(id, 0, 1).unwrap();
        let result = rx.blocking_recv().unwrap().unwrap();
        assert_eq!(result.answers, vec![vec!["Blue".to_string()]]);
    }

    #[test]
    fn resolve_question_by_index_out_of_range_question() {
        let bridge = make_bridge();
        let id = Uuid::new_v4();
        let (tx, _rx) = tokio::sync::oneshot::channel();

        {
            let mut pending = bridge.pending.write().unwrap();
            pending.insert(id, PendingSender::Question(tx));
            let mut qo = bridge.question_options.write().unwrap();
            qo.insert(id, vec![vec!["A".into()]]);
        }

        let err = bridge.resolve_question_by_index(id, 5, 0).unwrap_err();
        assert!(err.to_string().contains("question index 5 out of range"));
    }

    #[test]
    fn resolve_question_by_index_out_of_range_option() {
        let bridge = make_bridge();
        let id = Uuid::new_v4();
        let (tx, _rx) = tokio::sync::oneshot::channel();

        {
            let mut pending = bridge.pending.write().unwrap();
            pending.insert(id, PendingSender::Question(tx));
            let mut qo = bridge.question_options.write().unwrap();
            qo.insert(id, vec![vec!["A".into()]]);
        }

        let err = bridge.resolve_question_by_index(id, 0, 5).unwrap_err();
        assert!(err.to_string().contains("option index 5 out of range"));
    }

    #[test]
    fn resolve_question_by_index_no_stored_options() {
        let bridge = make_bridge();
        let id = Uuid::new_v4();
        let (tx, _rx) = tokio::sync::oneshot::channel();

        {
            let mut pending = bridge.pending.write().unwrap();
            pending.insert(id, PendingSender::Question(tx));
            // Deliberately don't insert question_options
        }

        let err = bridge.resolve_question_by_index(id, 0, 0).unwrap_err();
        assert!(err.to_string().contains("no stored question options"));
    }

    #[test]
    fn take_pending_cleans_up_question_options() {
        let bridge = make_bridge();
        let id = Uuid::new_v4();
        let (tx, _rx) = tokio::sync::oneshot::channel();

        {
            let mut pending = bridge.pending.write().unwrap();
            pending.insert(id, PendingSender::Question(tx));
            let mut qo = bridge.question_options.write().unwrap();
            qo.insert(id, vec![vec!["A".into()]]);
        }

        // Resolving should clean up both pending and question_options
        bridge
            .resolve_question(
                id,
                QuestionResponse {
                    answers: vec![vec!["A".into()]],
                },
            )
            .unwrap();

        let qo = bridge.question_options.read().unwrap();
        assert!(!qo.contains_key(&id));
    }
}
