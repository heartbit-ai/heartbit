//! `TelegramReviewDelivery` — standalone teloxide bot in the CLI process
//! that sends review messages with inline keyboards, awaits a callback,
//! and edits the message in place when the outcome is reported.
//!
//! Both `TelegramReviewDelivery` and `TelegramReplyReviewDelivery` share a
//! **single** global callback dispatcher (keyed by `Uuid`), so only one
//! `teloxide::Dispatcher` consumes the bot's update stream — regardless of how
//! many delivery instances are constructed.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, OnceLock};
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

// =============================================================================
// Shared dispatcher infrastructure
// =============================================================================

/// Process-global pending map shared by both `TelegramReviewDelivery` and
/// `TelegramReplyReviewDelivery`. Keyed by `Uuid` generated at deliver time.
///
/// Both delivery types use `DeliveryOutcome` from `heartbit_ghost::review`,
/// so a single map type serves all callers regardless of review variant.
type PendingMap = Arc<AsyncMutex<HashMap<Uuid, oneshot::Sender<DeliveryOutcome>>>>;

static GLOBAL_PENDING: OnceLock<PendingMap> = OnceLock::new();
static DISPATCHER_SPAWNED: OnceLock<()> = OnceLock::new();

/// Return the process-global pending map, initialising it on first call.
fn shared_pending() -> PendingMap {
    GLOBAL_PENDING
        .get_or_init(|| Arc::new(AsyncMutex::new(HashMap::new())))
        .clone()
}

/// Spawn the global callback dispatcher if it hasn't been spawned yet.
///
/// Uses `DISPATCHER_SPAWNED` as a one-shot gate so that constructing multiple
/// delivery instances (e.g. one `TelegramReviewDelivery` + one
/// `TelegramReplyReviewDelivery`) does NOT start a second dispatcher that would
/// silently compete for callback updates.
///
/// Assumption: all delivery instances use the same bot token
/// (`HEARTBIT_TELEGRAM_TOKEN`), so the first bot wins for the dispatcher.
fn ensure_callback_dispatcher(bot: Bot) {
    if DISPATCHER_SPAWNED.set(()).is_err() {
        // Another delivery already spawned the dispatcher.
        return;
    }
    let pending = shared_pending();
    tokio::spawn(async move {
        let handler =
            Update::filter_callback_query().endpoint(move |q: CallbackQuery, bot: Bot| {
                let pending = pending.clone();
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
        Dispatcher::builder(bot, handler).build().dispatch().await;
    });
}

// =============================================================================
// TelegramReviewDelivery
// =============================================================================

/// Standalone teloxide-backed `ReviewDelivery`.
pub struct TelegramReviewDelivery {
    bot: Bot,
    chat_id: ChatId,
    timeout: Duration,
    /// Pending pick resolvers keyed by interaction_id.
    pending: PendingMap,
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

        // Spawn the shared callback dispatcher (no-op if already running).
        ensure_callback_dispatcher(bot.clone());

        Ok(Self {
            bot,
            chat_id: ChatId(chat_id_num),
            timeout: Duration::from_secs(timeout_secs),
            pending: shared_pending(),
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

// =============================================================================
// TelegramReplyReviewDelivery — reply-mode counterpart of TelegramReviewDelivery
// =============================================================================

/// Telegram-backed delivery for the reply pipeline. Renders a
/// parent-quoted message + drafts + [1]…[N] [Skip] inline keyboard,
/// awaits the user's pick, returns the outcome.
///
/// Shares the same process-global pending map and dispatcher as
/// `TelegramReviewDelivery` — only one Telegram dispatcher runs per process.
// Allow dead_code: not yet wired into a CLI subcommand (lifecycle wiring
// deferred to the P1.5c follow-up integration step).
#[allow(dead_code)]
pub struct TelegramReplyReviewDelivery {
    bot: Bot,
    chat_id: ChatId,
    timeout: Duration,
    /// Shared process-global pending resolvers.
    pending: PendingMap,
}

#[allow(dead_code)]
impl TelegramReplyReviewDelivery {
    /// Construct from environment variables and eagerly spawn the
    /// callback dispatcher (shared with `TelegramReviewDelivery`).
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

        // Spawn the shared callback dispatcher (no-op if already running).
        ensure_callback_dispatcher(bot.clone());

        Ok(Self {
            bot,
            chat_id: ChatId(chat_id_num),
            timeout: Duration::from_secs(timeout_secs),
            pending: shared_pending(),
        })
    }

    /// Render the message body for a reply review per spec §9.1.
    fn render_review_body(msg: &heartbit_ghost::reply::ReplyReviewMessage) -> String {
        let mut s = String::new();
        let follower_str = msg
            .mentioner_context
            .as_ref()
            .and_then(|c| c.follower_count)
            .map(|n| n.to_string())
            .unwrap_or_else(|| "?".into());
        s.push_str(&format!(
            "NEW MENTION on your tweet from @{} ({} followers)\n\n",
            msg.mention.author_handle, follower_str,
        ));
        if let Some(p) = &msg.parent {
            let abridged: String = p.text.chars().take(200).collect();
            s.push_str("YOUR TWEET (parent):\n> ");
            s.push_str(&abridged);
            if p.text.chars().count() > 200 {
                s.push('…');
            }
            s.push_str("\n\n");
        }
        s.push_str("THEIR REPLY:\n> ");
        s.push_str(&msg.mention.text);
        s.push_str("\n\n");
        for (i, c) in msg.candidates.iter().enumerate() {
            s.push_str(&format!("DRAFT {}:\n> {}\n\n", i + 1, c));
        }
        s
    }

    /// Render the outcome report that replaces the original Telegram message.
    fn render_outcome_report(outcome: &heartbit_ghost::reply::ReplyOutcome) -> String {
        match outcome {
            heartbit_ghost::reply::ReplyOutcome::Posted {
                chosen_index,
                reply_url,
                ..
            } => {
                format!("Posted draft {}: {}", chosen_index + 1, reply_url)
            }
            heartbit_ghost::reply::ReplyOutcome::Skipped => "Skipped".to_string(),
            heartbit_ghost::reply::ReplyOutcome::TimedOut => {
                "Timed out — no reply sent".to_string()
            }
            heartbit_ghost::reply::ReplyOutcome::GateRejected {
                chosen_index,
                reason,
            } => {
                format!(
                    "Draft {} rejected by publish gate: {}",
                    chosen_index + 1,
                    reason
                )
            }
            heartbit_ghost::reply::ReplyOutcome::PublishFailed {
                chosen_index,
                reason,
            } => {
                format!("Draft {} failed to publish: {}", chosen_index + 1, reason)
            }
            heartbit_ghost::reply::ReplyOutcome::NoReply => {
                "Writer chose 'no_reply' — nothing sent".to_string()
            }
        }
    }
}

impl heartbit_ghost::reply::ReplyReviewDelivery for TelegramReplyReviewDelivery {
    fn deliver<'a>(
        &'a self,
        msg: heartbit_ghost::reply::ReplyReviewMessage,
    ) -> Pin<
        Box<
            dyn Future<
                    Output = Result<
                        heartbit_ghost::review::DeliveredReview,
                        heartbit_ghost::review::ReviewDeliveryError,
                    >,
                > + Send
                + 'a,
        >,
    > {
        Box::pin(async move {
            // Generate an interaction_id internally — the message struct doesn't
            // carry one; the Uuid is a delivery implementation detail.
            let interaction_id = Uuid::new_v4();
            let body = Self::render_review_body(&msg);
            let buttons = persona_pick_buttons(interaction_id, msg.candidates.len());
            let keyboard = InlineKeyboardMarkup::new(vec![
                buttons
                    .into_iter()
                    .map(|(label, data)| InlineKeyboardButton::callback(label, data))
                    .collect::<Vec<_>>(),
            ]);

            let (tx, rx) = oneshot::channel::<DeliveryOutcome>();
            self.pending.lock().await.insert(interaction_id, tx);

            let sent = self
                .bot
                .send_message(self.chat_id, body)
                .reply_markup(keyboard)
                .await
                .map_err(|e| {
                    heartbit_ghost::review::ReviewDeliveryError::Transport(format!(
                        "send_message: {e}"
                    ))
                })?;
            let message_id = sent.id;

            // Use the configured timeout from the message, falling back to
            // self.timeout when the message carries 0.
            let effective_timeout = if msg.interaction_timeout_seconds > 0 {
                Duration::from_secs(msg.interaction_timeout_seconds)
            } else {
                self.timeout
            };

            let outcome = match timeout(effective_timeout, rx).await {
                Ok(Ok(o)) => o,
                Ok(Err(_)) => DeliveryOutcome::TimedOut, // sender dropped
                Err(_) => {
                    // Timeout — clean up the pending entry.
                    self.pending.lock().await.remove(&interaction_id);
                    DeliveryOutcome::TimedOut
                }
            };

            let receipt = heartbit_ghost::review::DeliveryReceipt {
                data: serde_json::json!({
                    "chat_id": self.chat_id.0,
                    "message_id": message_id.0,
                }),
            };

            Ok(heartbit_ghost::review::DeliveredReview { outcome, receipt })
        })
    }

    fn report<'a>(
        &'a self,
        receipt: heartbit_ghost::review::DeliveryReceipt,
        outcome: heartbit_ghost::reply::ReplyOutcome,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<(), heartbit_ghost::review::ReviewDeliveryError>>
                + Send
                + 'a,
        >,
    > {
        Box::pin(async move {
            let chat_id_num = receipt
                .data
                .get("chat_id")
                .and_then(|v| v.as_i64())
                .ok_or_else(|| {
                    heartbit_ghost::review::ReviewDeliveryError::InvalidCallback(
                        "receipt missing chat_id".into(),
                    )
                })?;
            let message_id_num = receipt
                .data
                .get("message_id")
                .and_then(|v| v.as_i64())
                .ok_or_else(|| {
                    heartbit_ghost::review::ReviewDeliveryError::InvalidCallback(
                        "receipt missing message_id".into(),
                    )
                })?;
            let body = Self::render_outcome_report(&outcome);
            self.bot
                .edit_message_text(ChatId(chat_id_num), MessageId(message_id_num as i32), body)
                .await
                .map_err(|e| {
                    heartbit_ghost::review::ReviewDeliveryError::Transport(format!(
                        "edit_message: {e}"
                    ))
                })?;
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
struct EnvCredentialResolver;

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

#[cfg(test)]
mod reply_review_render_tests {
    use super::TelegramReplyReviewDelivery;
    use heartbit_ghost::reply::{Mention, MentionerContext, ReplyOutcome, ReplyReviewMessage};

    fn fixture_msg(parent_text: Option<&str>, candidates: Vec<&str>) -> ReplyReviewMessage {
        ReplyReviewMessage {
            mention: Mention {
                id: "m1".into(),
                text: "hey, is heartbit production ready?".into(),
                author_id: "999".into(),
                author_handle: "curious_dev".into(),
                posted_at: chrono::Utc::now(),
                in_reply_to_tweet_id: None,
            },
            parent: parent_text.map(|t| heartbit_ghost::reply::TweetSnapshot {
                id: "parent1".into(),
                text: t.to_string(),
                posted_at: chrono::Utc::now(),
            }),
            mentioner_context: Some(MentionerContext {
                handle: "curious_dev".into(),
                bio: None,
                recent_tweets: vec![],
                follower_count: Some(1234),
            }),
            candidates: candidates.into_iter().map(String::from).collect(),
            interaction_timeout_seconds: 300,
        }
    }

    #[test]
    fn render_review_body_includes_author_handle_and_follower_count() {
        let msg = fixture_msg(None, vec!["reply one"]);
        let body = TelegramReplyReviewDelivery::render_review_body(&msg);
        assert!(body.contains("@curious_dev"), "should contain handle");
        assert!(body.contains("1234"), "should contain follower count");
        assert!(body.contains("DRAFT 1:"), "should contain draft label");
        assert!(body.contains("reply one"), "should contain candidate text");
    }

    #[test]
    fn render_review_body_includes_parent_section_when_present() {
        let msg = fixture_msg(Some("parent tweet text"), vec!["my reply"]);
        let body = TelegramReplyReviewDelivery::render_review_body(&msg);
        assert!(
            body.contains("YOUR TWEET (parent):"),
            "should show parent section"
        );
        assert!(
            body.contains("parent tweet text"),
            "should show parent text"
        );
    }

    #[test]
    fn render_review_body_omits_parent_section_when_absent() {
        let msg = fixture_msg(None, vec!["my reply"]);
        let body = TelegramReplyReviewDelivery::render_review_body(&msg);
        assert!(
            !body.contains("YOUR TWEET (parent):"),
            "should not show parent section"
        );
    }

    #[test]
    fn render_review_body_abridges_long_parent_at_200_chars() {
        let long_parent = "x".repeat(250);
        let msg = fixture_msg(Some(&long_parent), vec!["reply"]);
        let body = TelegramReplyReviewDelivery::render_review_body(&msg);
        // 200 x's + ellipsis.
        assert!(
            body.contains('…'),
            "should contain ellipsis for long parent"
        );
        // Verify we don't include all 250 chars.
        let x_run: String = "x".repeat(201);
        assert!(
            !body.contains(&x_run),
            "should not include chars beyond 200"
        );
    }

    #[test]
    fn render_review_body_multiple_candidates() {
        let msg = fixture_msg(None, vec!["reply A", "reply B"]);
        let body = TelegramReplyReviewDelivery::render_review_body(&msg);
        assert!(body.contains("DRAFT 1:"), "should label draft 1");
        assert!(body.contains("DRAFT 2:"), "should label draft 2");
        assert!(body.contains("reply A"), "draft 1 text");
        assert!(body.contains("reply B"), "draft 2 text");
    }

    #[test]
    fn render_outcome_report_posted() {
        let o = ReplyOutcome::Posted {
            chosen_index: 0,
            reply_tweet_id: "t1".into(),
            reply_url: "https://x.com/i/web/status/t1".into(),
        };
        let s = TelegramReplyReviewDelivery::render_outcome_report(&o);
        assert!(s.contains("1"), "should mention draft number (1-based)");
        assert!(
            s.contains("https://x.com/i/web/status/t1"),
            "should include url"
        );
    }

    #[test]
    fn render_outcome_report_skipped() {
        let s = TelegramReplyReviewDelivery::render_outcome_report(&ReplyOutcome::Skipped);
        assert!(s.to_lowercase().contains("skip"), "should say skipped");
    }

    #[test]
    fn render_outcome_report_no_reply() {
        let s = TelegramReplyReviewDelivery::render_outcome_report(&ReplyOutcome::NoReply);
        assert!(s.contains("no_reply"), "should mention no_reply");
    }

    #[test]
    fn from_env_fails_without_token_env_var() {
        // Note: `std::env::remove_var` is unsafe in Rust ≥ 1.87 due to
        // thread-safety concerns, so we don't call it in tests. Instead,
        // skip this test if the env var happens to be set.
        if std::env::var("HEARTBIT_TELEGRAM_TOKEN").is_err() {
            let result = TelegramReplyReviewDelivery::from_env();
            let err_msg = match result {
                Err(e) => format!("{e}"),
                Ok(_) => panic!("expected Err when HEARTBIT_TELEGRAM_TOKEN is not set"),
            };
            assert!(
                err_msg.contains("HEARTBIT_TELEGRAM_TOKEN"),
                "error should name the missing env var; got: {err_msg}"
            );
        }
        // If the env var IS set, this test is a no-op (passes trivially).
    }
}
