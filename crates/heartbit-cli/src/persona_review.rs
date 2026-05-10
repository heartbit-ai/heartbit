//! `TelegramReviewDelivery` — standalone teloxide bot in the CLI process
//! that sends review messages with inline keyboards, awaits a callback,
//! and edits the message in place when the outcome is reported.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use heartbit_ghost::review::{
    DeliveredReview, DeliveryOutcome, DeliveryReceipt, ReportableOutcome, ReviewDelivery,
    ReviewDeliveryError, ReviewMessage, build_report_message, build_review_message,
};
use heartbit_telegram::{CallbackAction, PickChoice, parse_callback_data, persona_pick_buttons};
use teloxide::prelude::*;
use teloxide::types::{ChatId, InlineKeyboardButton, InlineKeyboardMarkup, MessageId};
use tokio::sync::{Mutex as AsyncMutex, oneshot};
use tokio::time::timeout;
use uuid::Uuid;

const DEFAULT_TIMEOUT_SECS: u64 = 3600;

/// Standalone teloxide-backed `ReviewDelivery`.
pub struct TelegramReviewDelivery {
    bot: Bot,
    chat_id: ChatId,
    timeout: Duration,
    /// Pending pick resolvers keyed by interaction_id.
    pending: Arc<AsyncMutex<HashMap<Uuid, oneshot::Sender<DeliveryOutcome>>>>,
}

impl TelegramReviewDelivery {
    /// Construct from environment variables and eagerly spawn the
    /// callback dispatcher.
    ///
    /// Required env:
    /// - `HEARTBIT_TELEGRAM_TOKEN` — bot token
    /// - `HEARTBIT_TELEGRAM_REVIEW_CHAT_ID` — destination chat (`i64`)
    ///
    /// Optional:
    /// - `HEARTBIT_REVIEW_TIMEOUT_SECS` — pick timeout (default 3600 = 1h)
    pub fn from_env() -> Result<Self, ReviewDeliveryError> {
        let token = std::env::var("HEARTBIT_TELEGRAM_TOKEN").map_err(|_| {
            ReviewDeliveryError::Config("HEARTBIT_TELEGRAM_TOKEN env var not set".into())
        })?;
        let chat_id_raw = std::env::var("HEARTBIT_TELEGRAM_REVIEW_CHAT_ID").map_err(|_| {
            ReviewDeliveryError::Config("HEARTBIT_TELEGRAM_REVIEW_CHAT_ID env var not set".into())
        })?;
        let chat_id_num: i64 = chat_id_raw.parse().map_err(|e| {
            ReviewDeliveryError::Config(format!(
                "invalid HEARTBIT_TELEGRAM_REVIEW_CHAT_ID '{chat_id_raw}': {e}"
            ))
        })?;
        let timeout_secs = std::env::var("HEARTBIT_REVIEW_TIMEOUT_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_TIMEOUT_SECS);

        let bot = Bot::new(token);
        let pending: Arc<AsyncMutex<HashMap<Uuid, oneshot::Sender<DeliveryOutcome>>>> =
            Arc::new(AsyncMutex::new(HashMap::new()));

        // Eagerly spawn the callback dispatcher. One dispatcher per
        // TelegramReviewDelivery instance; lives for the duration of the
        // CLI process.
        let dispatcher_bot = bot.clone();
        let dispatcher_pending = pending.clone();
        tokio::spawn(async move {
            let handler =
                Update::filter_callback_query().endpoint(move |q: CallbackQuery, bot: Bot| {
                    let pending = dispatcher_pending.clone();
                    async move {
                        let data = match q.data.as_ref() {
                            Some(d) => d,
                            None => return Ok::<_, teloxide::RequestError>(()),
                        };
                        let action = match parse_callback_data(data) {
                            Ok(a) => a,
                            Err(_) => return Ok(()),
                        };
                        if let CallbackAction::PersonaPick {
                            interaction_id,
                            choice,
                        } = action
                        {
                            let outcome = match choice {
                                PickChoice::Index(i) => DeliveryOutcome::Pick(i),
                                PickChoice::Skip => DeliveryOutcome::Skip,
                            };
                            let mut map = pending.lock().await;
                            if let Some(sender) = map.remove(&interaction_id) {
                                let _ = sender.send(outcome);
                                // Acknowledge the callback (UI feedback).
                                let _ = bot.answer_callback_query(q.id.clone()).await;
                            }
                        }
                        Ok(())
                    }
                });
            Dispatcher::builder(dispatcher_bot, handler)
                .build()
                .dispatch()
                .await;
        });

        Ok(Self {
            bot,
            chat_id: ChatId(chat_id_num),
            timeout: Duration::from_secs(timeout_secs),
            pending,
        })
    }
}

impl ReviewDelivery for TelegramReviewDelivery {
    fn deliver_and_await<'a>(
        &'a self,
        message: &'a ReviewMessage,
    ) -> Pin<Box<dyn Future<Output = Result<DeliveredReview, ReviewDeliveryError>> + Send + 'a>>
    {
        Box::pin(async move {
            let body = build_review_message(message);
            let buttons = persona_pick_buttons(message.interaction_id, message.candidates.len());
            let keyboard = InlineKeyboardMarkup::new(vec![
                buttons
                    .into_iter()
                    .map(|(label, data)| InlineKeyboardButton::callback(label, data))
                    .collect::<Vec<_>>(),
            ]);

            let (tx, rx) = oneshot::channel::<DeliveryOutcome>();
            self.pending.lock().await.insert(message.interaction_id, tx);

            // Dispatcher already running (spawned by from_env).
            let sent = self
                .bot
                .send_message(self.chat_id, body)
                .reply_markup(keyboard)
                .await
                .map_err(|e| ReviewDeliveryError::Transport(format!("send_message: {e}")))?;
            let message_id = sent.id;

            let outcome = match timeout(self.timeout, rx).await {
                Ok(Ok(o)) => o,
                Ok(Err(_)) => DeliveryOutcome::TimedOut, // sender dropped
                Err(_) => {
                    // Timeout — clean up the pending entry.
                    self.pending.lock().await.remove(&message.interaction_id);
                    DeliveryOutcome::TimedOut
                }
            };

            let receipt = DeliveryReceipt {
                data: serde_json::json!({
                    "chat_id": self.chat_id.0,
                    "message_id": message_id.0,
                }),
            };

            Ok(DeliveredReview { outcome, receipt })
        })
    }

    fn report<'a>(
        &'a self,
        receipt: DeliveryReceipt,
        outcome: ReportableOutcome,
    ) -> Pin<Box<dyn Future<Output = Result<(), ReviewDeliveryError>> + Send + 'a>> {
        Box::pin(async move {
            let chat_id_num = receipt
                .data
                .get("chat_id")
                .and_then(|v| v.as_i64())
                .ok_or_else(|| {
                    ReviewDeliveryError::InvalidCallback("receipt missing chat_id".into())
                })?;
            let message_id_num = receipt
                .data
                .get("message_id")
                .and_then(|v| v.as_i64())
                .ok_or_else(|| {
                    ReviewDeliveryError::InvalidCallback("receipt missing message_id".into())
                })?;
            let body = build_report_message(&outcome);
            self.bot
                .edit_message_text(ChatId(chat_id_num), MessageId(message_id_num as i32), body)
                .await
                .map_err(|e| ReviewDeliveryError::Transport(format!("edit_message: {e}")))?;
            Ok(())
        })
    }
}

/// Helper: construct the production `ReviewConfig` from env + CLI args.
#[allow(clippy::too_many_arguments)]
pub async fn review_config_from_env<'a>(
    persona_name: &'a str,
    topic: &'a str,
    candidates_per_draft: usize,
    provider: Arc<heartbit_core::llm::BoxedProvider>,
    corpora_root: &'a std::path::Path,
    profiles_root: &'a std::path::Path,
    on_progress: Option<heartbit_ghost::pipeline::ProgressCallback>,
    mode_addendum: Option<&'static str>,
    researcher_override: Option<heartbit_ghost::pipeline::ResearcherOverride>,
) -> Result<heartbit_ghost::review::ReviewConfig<'a>> {
    let delivery: Arc<dyn ReviewDelivery> =
        Arc::new(TelegramReviewDelivery::from_env().context("construct TelegramReviewDelivery")?);
    let twitter_tool: Arc<dyn heartbit_core::tool::Tool> =
        Arc::new(heartbit_ghost::tools::TwitterThreadTool::new());
    let credentials: Arc<dyn heartbit_core::CredentialResolver> = Arc::new(EnvCredentialResolver);
    Ok(heartbit_ghost::review::ReviewConfig {
        persona_name,
        topic,
        provider,
        corpora_root,
        profiles_root,
        on_progress,
        candidates_per_draft,
        delivery,
        twitter_tool,
        credentials,
        mode_addendum,
        researcher_override,
    })
}

/// Env-only credential resolver — reads `name` from `std::env`, wraps
/// in `Secret`. Error if env var unset.
pub(crate) struct EnvCredentialResolver;

impl heartbit_core::CredentialResolver for EnvCredentialResolver {
    fn resolve(
        &self,
        name: &str,
    ) -> Pin<
        Box<dyn Future<Output = Result<heartbit_core::Secret, heartbit_core::Error>> + Send + '_>,
    > {
        let name = name.to_string();
        Box::pin(async move {
            std::env::var(&name)
                .map(heartbit_core::Secret::new)
                .map_err(|_| heartbit_core::Error::Config(format!("env var '{name}' not set")))
        })
    }
}
