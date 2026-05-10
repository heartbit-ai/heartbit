//! Reply pipeline — drafts a single short reply to a specific mention,
//! routes to Telegram for review, posts via `twitter_reply` on user pick.
//!
//! See spec §2/§5 for the architecture. The runtime is
//! [`run_reply_pipeline`]; helper types and the [`ReplyReviewDelivery`]
//! trait live here too.

use chrono::{DateTime, Utc};

pub mod bot_guard;
pub mod conversation_guard;
pub mod prompts;
pub mod spam_guard;
pub mod storage;
pub mod thread_guard;
pub use bot_guard::{BotHeuristicConfig, BotHeuristicGuard};
pub use conversation_guard::ConversationDepthGuard;
pub use spam_guard::{SkipReason, SpamGuard, SpamGuardConfig};
pub use storage::{InMemoryMentionStore, JsonlMentionStore, MentionStore, StoreError};
pub use thread_guard::ThreadDepthGuard;

/// A mention of the operator's account fetched from `twitter_mentions`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Mention {
    /// X tweet ID of the mention itself.
    pub id: String,
    /// Plain text of the mention.
    pub text: String,
    /// X user ID of the mentioner.
    pub author_id: String,
    /// Public handle of the mentioner (sans `@`).
    pub author_handle: String,
    /// When the mention was posted.
    pub posted_at: DateTime<Utc>,
    /// Tweet ID this mention is replying to (None when it's a top-level
    /// `@operator …` mention rather than a reply on an operator's tweet).
    pub in_reply_to_tweet_id: Option<String>,
    /// X conversation_id (the root tweet of the thread tree). Used by
    /// the conversation-depth guard (P1.7) to cap reply count per
    /// conversation. `#[serde(default)]` for backward compatibility
    /// with stores written before P1.7.
    #[serde(default)]
    pub conversation_id: Option<String>,
}

/// A small snapshot of a tweet (text + timing). Used as a parent-tweet
/// context for the reply researcher.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TweetSnapshot {
    /// X tweet ID.
    pub id: String,
    /// Plain text of the tweet.
    pub text: String,
    /// When the tweet was posted.
    pub posted_at: DateTime<Utc>,
}

/// Tone-calibration context about the mentioner. None of these are
/// strictly required; the writer degrades gracefully if missing.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct MentionerContext {
    /// Public handle of the mentioner (sans `@`).
    pub handle: String,
    /// Bio text from the mentioner's profile, if available.
    pub bio: Option<String>,
    /// Up to 3 recent tweets, abridged.
    pub recent_tweets: Vec<TweetSnapshot>,
    /// Follower count of the mentioner, if available.
    pub follower_count: Option<u64>,
    /// Following count of the mentioner, if available. Used by the
    /// bot-heuristic guard (P1.7) for the follower/following ratio
    /// signal.
    pub following_count: Option<u64>,
    /// When the mentioner's account was created. Used by the
    /// bot-heuristic guard (P1.7) for the account-age signal.
    pub account_created_at: Option<DateTime<Utc>>,
}

use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;

use heartbit_core::CredentialResolver;
use heartbit_core::Tool;
use heartbit_core::llm::BoxedProvider;
use heartbit_core::llm::types::TokenUsage;
use thiserror::Error;

use crate::pipeline::{PipelineError, ProgressCallback, ResearcherOverride};

/// Configuration for one reply-pipeline run.
#[derive(Clone)]
pub struct ReplyConfig<'a> {
    /// Persona name.
    pub persona_name: &'a str,
    /// LLM provider used for every sub-agent in this run.
    pub provider: Arc<BoxedProvider>,
    /// Root directory containing per-persona corpora.
    pub corpora_root: &'a Path,
    /// Root directory containing per-persona style profiles.
    pub profiles_root: &'a Path,
    /// Optional progress callback.
    pub on_progress: Option<ProgressCallback>,
    /// The mention being replied to.
    pub mention: Mention,
    /// The operator's tweet the mention is replying to (when the mention
    /// is a thread reply rather than a top-level @-mention).
    pub parent: Option<TweetSnapshot>,
    /// Optional bio + recent tweets for tone calibration.
    pub mentioner_context: Option<MentionerContext>,
    /// Number of distinct candidate replies to generate (1..=3).
    /// 1 = no judge; 2 or 3 = judge picks.
    pub candidates_per_reply: usize,
    /// Persona-specific mode addendum.
    pub mode_addendum: Option<&'a str>,
    /// Optional researcher override (same semantics as PipelineConfig).
    pub researcher_override: Option<ResearcherOverride>,
    /// Telegram-or-mock delivery layer for the reply review.
    pub delivery: Arc<dyn ReplyReviewDelivery>,
    /// `twitter_reply` tool — production wires `Arc::new(TwitterReplyTool::new())`;
    /// tests wire a mock.
    pub twitter_tool: Arc<dyn Tool>,
    /// Credential resolver for `twitter_tool`.
    pub credentials: Arc<dyn CredentialResolver>,
}

/// Output of a successful reply-pipeline run.
#[derive(Debug, Clone)]
pub struct ReplyOutput {
    /// X tweet ID of the mention this run was for.
    pub mention_id: String,
    /// One generated reply candidate per slot (`candidates_per_reply`).
    pub candidates: Vec<ReplyCandidateRecord>,
    /// Aggregated token usage across all sub-agents.
    pub usage_summary: TokenUsage,
    /// Final outcome of the run.
    pub outcome: ReplyOutcome,
}

/// One generated reply draft (post-style/fact, pre-publish_gate).
#[derive(Debug, Clone)]
pub struct ReplyCandidateRecord {
    /// The composed reply text.
    pub draft: String,
    /// Voice-match score from the style critic, 0.0..=1.0.
    pub style_match_score: f32,
    /// Fact-check verdict: "verified" | "unverifiable: reason" | "rejected: reason".
    pub fact_check_verdict: String,
}

/// What happened in this reply run.
#[derive(Debug, Clone)]
pub enum ReplyOutcome {
    /// User picked candidate `chosen_index` and the reply was published.
    Posted {
        /// Index of the candidate the user picked.
        chosen_index: usize,
        /// X tweet ID of the published reply.
        reply_tweet_id: String,
        /// Public URL of the published reply.
        reply_url: String,
    },
    /// User pressed Skip.
    Skipped,
    /// Telegram review timed out without a pick.
    TimedOut,
    /// User picked `chosen_index` but `publish_gate` rejected it.
    GateRejected {
        /// Index of the candidate the user picked.
        chosen_index: usize,
        /// Reason from the publish gate.
        reason: String,
    },
    /// User picked `chosen_index` but the X API call failed.
    PublishFailed {
        /// Index of the candidate the user picked.
        chosen_index: usize,
        /// Reason for failure.
        reason: String,
    },
    /// All candidates returned the literal "no_reply" string — the
    /// writer chose not to engage. No Telegram review was sent.
    NoReply,
}

/// Errors raised by `run_reply_pipeline`.
#[derive(Debug, Error)]
pub enum ReplyError {
    /// Underlying pipeline error.
    #[error("pipeline: {0}")]
    Pipeline(#[from] PipelineError),
    /// Telegram delivery error.
    #[error("delivery: {0}")]
    Delivery(#[from] crate::review::ReviewDeliveryError),
    /// Misconfigured `ReplyConfig`.
    #[error("invalid config: {0}")]
    InvalidConfig(String),
}

/// Telegram-or-mock review delivery for reply messages. Mirrors
/// [`crate::review::ReviewDelivery`] but with a parent-quoted message
/// shape. See spec §9 for the full text layout.
///
/// Methods use the project's `Pin<Box<dyn Future>>` desugaring to stay
/// object-safe without the `async-trait` crate (matches `ReviewDelivery`).
pub trait ReplyReviewDelivery: Send + Sync {
    /// Deliver a reply review to the user (Telegram bot or mock).
    fn deliver<'a>(
        &'a self,
        msg: ReplyReviewMessage,
    ) -> Pin<
        Box<
            dyn Future<
                    Output = Result<
                        crate::review::DeliveredReview,
                        crate::review::ReviewDeliveryError,
                    >,
                > + Send
                + 'a,
        >,
    >;

    /// Report final outcome back to the delivery layer (edits the original
    /// message, etc.). Non-fatal; runtime ignores errors.
    fn report<'a>(
        &'a self,
        receipt: crate::review::DeliveryReceipt,
        outcome: ReplyOutcome,
    ) -> Pin<Box<dyn Future<Output = Result<(), crate::review::ReviewDeliveryError>> + Send + 'a>>;
}

/// Message body for a reply review. The Telegram impl renders this as
/// the parent + mention + drafts layout from spec §9.1.
#[derive(Debug, Clone)]
pub struct ReplyReviewMessage {
    /// The mention being replied to.
    pub mention: Mention,
    /// The operator's parent tweet, if applicable.
    pub parent: Option<TweetSnapshot>,
    /// Optional mentioner context for tone calibration.
    pub mentioner_context: Option<MentionerContext>,
    /// One drafted reply per slot.
    pub candidates: Vec<String>,
    /// How long to wait for a user pick (Telegram impl uses this).
    pub interaction_timeout_seconds: u64,
}

/// Execute one reply pipeline. Returns when the user picks (and the
/// reply posts), skips, times out, or all candidates return "no_reply".
pub async fn run_reply_pipeline(cfg: ReplyConfig<'_>) -> Result<ReplyOutput, ReplyError> {
    use crate::agents::{
        fact_check_recipe, judge_recipe, reply_writer_recipe, researcher_recipe,
        style_critic_recipe,
    };
    use heartbit_core::llm::types::TokenUsage;

    // 1. Validate.
    if !(1..=3).contains(&cfg.candidates_per_reply) {
        return Err(ReplyError::InvalidConfig(format!(
            "candidates_per_reply must be in 1..=3 (got {})",
            cfg.candidates_per_reply,
        )));
    }

    let progress = |s: &str| {
        if let Some(cb) = cfg.on_progress.as_ref() {
            cb(s);
        }
    };

    // 2. Load profile snapshot.
    progress("Loading profile snapshot...");
    let store = crate::voice::SnapshotStore::open(cfg.profiles_root, cfg.persona_name)
        .map_err(PipelineError::from)?;
    let snapshot = store
        .load_latest()
        .map_err(PipelineError::from)?
        .ok_or_else(|| PipelineError::NoProfileSnapshot {
            persona: cfg.persona_name.to_string(),
            profiles_dir: cfg.profiles_root.join(cfg.persona_name),
        })?;
    let profile = snapshot.profile;

    // 3. Build the 5 sub-agent runners.
    let (researcher_recipe_used, researcher_tools): (
        heartbit_core::config::AgentConfig,
        Vec<Arc<dyn Tool>>,
    ) = match cfg.researcher_override.as_ref() {
        Some((recipe, tools)) => ((**recipe).clone_config(), tools.clone()),
        None => (
            researcher_recipe(),
            // Replies do NOT need web search by default — context comes from
            // the parent + mentioner_context already passed into the user
            // message. Empty tool set keeps the researcher focused.
            Vec::new(),
        ),
    };
    let researcher = crate::pipeline::runner_from_recipe(
        cfg.provider.clone(),
        researcher_recipe_used,
        researcher_tools,
    )
    .map_err(|source| PipelineError::Builder {
        stage: "researcher".to_string(),
        source,
    })?;
    let writer = crate::pipeline::runner_from_recipe(
        cfg.provider.clone(),
        reply_writer_recipe(),
        Vec::new(),
    )
    .map_err(|source| PipelineError::Builder {
        stage: "reply_writer".to_string(),
        source,
    })?;
    let critic = crate::pipeline::runner_from_recipe(
        cfg.provider.clone(),
        style_critic_recipe(),
        Vec::new(),
    )
    .map_err(|source| PipelineError::Builder {
        stage: "style_critic".to_string(),
        source,
    })?;
    let fact =
        crate::pipeline::runner_from_recipe(cfg.provider.clone(), fact_check_recipe(), Vec::new())
            .map_err(|source| PipelineError::Builder {
                stage: "fact_check".to_string(),
                source,
            })?;
    let judge = if cfg.candidates_per_reply > 1 {
        Some(
            crate::pipeline::runner_from_recipe(cfg.provider.clone(), judge_recipe(), Vec::new())
                .map_err(|source| PipelineError::Builder {
                stage: "judge".to_string(),
                source,
            })?,
        )
    } else {
        None
    };

    let mut total_usage = TokenUsage::default();
    let voice_guidelines = crate::pipeline::render_style_profile_as_english(&profile);

    // 4. Run researcher.
    progress("Researching mention...");
    let research_msg = prompts::build_reply_research_user_message(
        &cfg.mention,
        cfg.parent.as_ref(),
        cfg.mentioner_context.as_ref(),
    );
    let researcher_out =
        researcher
            .execute(&research_msg)
            .await
            .map_err(|source| PipelineError::Agent {
                stage: "researcher".to_string(),
                source,
            })?;
    let digest = researcher_out.result.clone();
    total_usage += researcher_out.tokens_used;

    // 5. Generate N reply candidates in parallel via tokio::JoinSet.
    progress(&format!(
        "Generating {} candidate(s) in parallel...",
        cfg.candidates_per_reply
    ));
    let writer = Arc::new(writer);
    let critic = Arc::new(critic);
    let fact = Arc::new(fact);
    let voice_owned: Arc<str> = voice_guidelines.clone().into();
    let digest_owned: Arc<str> = digest.clone().into();
    let mode_owned: Option<Arc<str>> = cfg.mode_addendum.map(Arc::from);

    let mut joinset: tokio::task::JoinSet<
        Result<(String, f32, String, TokenUsage), heartbit_core::error::Error>,
    > = tokio::task::JoinSet::new();
    for _ in 0..cfg.candidates_per_reply {
        let writer = writer.clone();
        let critic = critic.clone();
        let fact = fact.clone();
        let voice = voice_owned.clone();
        let digest = digest_owned.clone();
        let mode = mode_owned.clone();
        joinset.spawn(async move {
            let writer_msg =
                prompts::build_reply_writer_user_message(&digest, &voice, mode.as_deref());
            let writer_out = writer.execute(&writer_msg).await?;
            let draft = writer_out.result.trim().to_string();
            // Writer-driven no_reply short-circuit.
            if draft.eq_ignore_ascii_case("no_reply") {
                return Ok((
                    draft,
                    0.0_f32,
                    "no_reply".to_string(),
                    writer_out.tokens_used,
                ));
            }
            // Style critic.
            let critic_msg = prompts::build_reply_critic_user_message(&draft, &voice);
            let critic_out = critic.execute(&critic_msg).await?;
            let style_score = parse_style_match_score(&critic_out.result).unwrap_or(0.5);
            // Fact check.
            let fact_msg = prompts::build_reply_fact_user_message(&draft, &digest);
            let fact_out = fact.execute(&fact_msg).await?;
            let fact_verdict = fact_out.result.clone();
            let mut usage = writer_out.tokens_used;
            usage += critic_out.tokens_used;
            usage += fact_out.tokens_used;
            Ok((draft, style_score, fact_verdict, usage))
        });
    }
    let mut survivors: Vec<ReplyCandidateRecord> = Vec::new();
    while let Some(handle) = joinset.join_next().await {
        let (draft, style_score, fact_verdict, usage) = handle
            .map_err(|e| PipelineError::Agent {
                stage: "candidate".to_string(),
                source: heartbit_core::error::Error::Agent(format!("join: {e}")),
            })?
            .map_err(|source| PipelineError::Agent {
                stage: "candidate".to_string(),
                source,
            })?;
        total_usage += usage;
        if !draft.eq_ignore_ascii_case("no_reply") {
            survivors.push(ReplyCandidateRecord {
                draft,
                style_match_score: style_score,
                fact_check_verdict: fact_verdict,
            });
        }
    }

    // 6. If all candidates were no_reply, return early without delivery.
    if survivors.is_empty() {
        return Ok(ReplyOutput {
            mention_id: cfg.mention.id.clone(),
            candidates: Vec::new(),
            usage_summary: total_usage,
            outcome: ReplyOutcome::NoReply,
        });
    }

    // 7. Judge if multiple survivors (skip when 1).
    let chosen_index: usize = if let (Some(judge), true) = (judge.as_ref(), survivors.len() > 1) {
        progress("Judging candidates...");
        let judge_msg = format!(
            "{voice_guidelines}\n\nCandidate replies for the mention from @{}:\n\n{}\n\nReturn your verdict as JSON per the schema.\n",
            cfg.mention.author_handle,
            survivors
                .iter()
                .enumerate()
                .map(|(i, c)| format!("[{i}]\n{}\n", c.draft))
                .collect::<String>(),
        );
        let judge_out = judge
            .execute(&judge_msg)
            .await
            .map_err(|source| PipelineError::Agent {
                stage: "judge".to_string(),
                source,
            })?;
        total_usage += judge_out.tokens_used;
        parse_judge_index(&judge_out.result, survivors.len()).unwrap_or(0)
    } else {
        0
    };

    let chosen_draft = survivors[chosen_index].draft.clone();

    // 8. Publish gate — hard 280-char cap.
    if chosen_draft.chars().count() > 280 {
        return Ok(ReplyOutput {
            mention_id: cfg.mention.id.clone(),
            candidates: survivors,
            usage_summary: total_usage,
            outcome: ReplyOutcome::GateRejected {
                chosen_index,
                reason: format!(
                    "draft exceeds 280 chars (got {})",
                    chosen_draft.chars().count(),
                ),
            },
        });
    }

    // 9. Telegram review delivery.
    progress("Sending review to user...");
    let drafts_for_review: Vec<String> = survivors.iter().map(|c| c.draft.clone()).collect();
    let msg = ReplyReviewMessage {
        mention: cfg.mention.clone(),
        parent: cfg.parent.clone(),
        mentioner_context: cfg.mentioner_context.clone(),
        candidates: drafts_for_review,
        interaction_timeout_seconds: 300,
    };
    let delivered = cfg.delivery.deliver(msg).await?;
    let outcome = match delivered.outcome {
        crate::review::DeliveryOutcome::Pick(idx) if idx < survivors.len() => {
            // 10. twitter_reply tool call.
            progress(&format!("Posting candidate {idx}..."));
            let exec_ctx = heartbit_core::ExecutionContext {
                credentials: Some(cfg.credentials.clone()),
                ..Default::default()
            };
            let tool_input = serde_json::json!({
                "text": survivors[idx].draft,
                "in_reply_to": cfg.mention.id,
            });
            match cfg.twitter_tool.execute(&exec_ctx, tool_input).await {
                Ok(out) if !out.is_error => {
                    let (tweet_id, url) = parse_reply_tool_output(&out.content);
                    ReplyOutcome::Posted {
                        chosen_index: idx,
                        reply_tweet_id: tweet_id,
                        reply_url: url,
                    }
                }
                Ok(out) => ReplyOutcome::PublishFailed {
                    chosen_index: idx,
                    reason: out.content,
                },
                Err(e) => ReplyOutcome::PublishFailed {
                    chosen_index: idx,
                    reason: format!("{e}"),
                },
            }
        }
        crate::review::DeliveryOutcome::Pick(_) => ReplyOutcome::Skipped, // unreachable
        crate::review::DeliveryOutcome::Skip => ReplyOutcome::Skipped,
        crate::review::DeliveryOutcome::TimedOut => ReplyOutcome::TimedOut,
    };

    // 11. Optional report-back to delivery (non-fatal).
    let _ = cfg
        .delivery
        .report(delivered.receipt, outcome.clone())
        .await;

    Ok(ReplyOutput {
        mention_id: cfg.mention.id.clone(),
        candidates: survivors,
        usage_summary: total_usage,
        outcome,
    })
}

// Helpers — pure parse functions ----------------------------------------

fn parse_style_match_score(raw: &str) -> Option<f32> {
    let v: serde_json::Value = serde_json::from_str(raw).ok()?;
    v.get("style_match_score")?.as_f64().map(|x| x as f32)
}

fn parse_judge_index(raw: &str, n: usize) -> Option<usize> {
    let v: serde_json::Value = serde_json::from_str(raw).ok()?;
    let idx = v.get("chosen_index")?.as_u64()? as usize;
    if idx < n { Some(idx) } else { None }
}

fn parse_reply_tool_output(content: &str) -> (String, String) {
    #[derive(serde::Deserialize)]
    struct Parsed {
        tweet_id: String,
        url: String,
    }
    serde_json::from_str::<Parsed>(content)
        .map(|p| (p.tweet_id, p.url))
        .unwrap_or_else(|_| (String::new(), "<unknown>".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mention_round_trips_through_clone() {
        let m = Mention {
            id: "1".into(),
            text: "hi".into(),
            author_id: "12".into(),
            author_handle: "alice".into(),
            posted_at: Utc::now(),
            in_reply_to_tweet_id: Some("99".into()),
            conversation_id: None,
        };
        let copy = m.clone();
        assert_eq!(copy.id, m.id);
        assert_eq!(copy.text, m.text);
        assert_eq!(copy.in_reply_to_tweet_id, m.in_reply_to_tweet_id);
    }

    #[test]
    fn mentioner_context_default_has_empty_handle_and_no_recent_tweets() {
        let m = MentionerContext::default();
        assert!(m.handle.is_empty());
        assert!(m.bio.is_none());
        assert!(m.recent_tweets.is_empty());
        assert!(m.follower_count.is_none());
    }

    #[test]
    fn mentioner_context_default_has_none_for_new_fields() {
        let m = MentionerContext::default();
        assert!(m.following_count.is_none());
        assert!(m.account_created_at.is_none());
    }

    #[test]
    fn reply_outcome_no_reply_is_distinct_from_skipped() {
        let a = ReplyOutcome::Skipped;
        let b = ReplyOutcome::NoReply;
        // No Eq derive (CredentialResolver isn't Eq); rely on debug
        // representation as a stand-in for "these are different variants".
        assert_ne!(format!("{a:?}"), format!("{b:?}"));
    }

    #[test]
    fn reply_error_display_round_trips() {
        let e = ReplyError::InvalidConfig("test".to_string());
        assert!(format!("{e}").contains("invalid config"));
    }

    // -----------------------------------------------------------------------
    // Integration tests for run_reply_pipeline
    // -----------------------------------------------------------------------

    use heartbit_core::ExecutionContext;
    use heartbit_core::error::Error as CoreError;
    use heartbit_core::execution_context::CredentialResolver as CredentialResolverTrait;
    use heartbit_core::execution_context::Secret;
    use heartbit_core::llm::types::ToolDefinition;
    use heartbit_core::llm::types::{
        CompletionRequest, CompletionResponse, ContentBlock, StopReason, TokenUsage,
    };
    use heartbit_core::llm::{BoxedProvider, LlmProvider};
    use heartbit_core::tool::ToolOutput;
    use std::collections::VecDeque;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Mutex;
    use tempfile::TempDir;

    /// MockReplyReviewDelivery — returns a pre-canned outcome. An optional
    /// error mode simulates delivery failure (used to assert delivery is
    /// never called in no_reply / gate-reject tests).
    struct MockReplyReviewDelivery {
        outcome: Option<crate::review::DeliveryOutcome>,
        error_msg: Option<String>,
        reports: Mutex<Vec<ReplyOutcome>>,
    }

    impl MockReplyReviewDelivery {
        fn arc(outcome: crate::review::DeliveryOutcome) -> Arc<MockReplyReviewDelivery> {
            Arc::new(MockReplyReviewDelivery {
                outcome: Some(outcome),
                error_msg: None,
                reports: Mutex::new(Vec::new()),
            })
        }

        fn errored(reason: &str) -> Arc<MockReplyReviewDelivery> {
            Arc::new(MockReplyReviewDelivery {
                outcome: None,
                error_msg: Some(reason.to_string()),
                reports: Mutex::new(Vec::new()),
            })
        }
    }

    impl ReplyReviewDelivery for MockReplyReviewDelivery {
        fn deliver<'a>(
            &'a self,
            _msg: ReplyReviewMessage,
        ) -> Pin<
            Box<
                dyn Future<
                        Output = Result<
                            crate::review::DeliveredReview,
                            crate::review::ReviewDeliveryError,
                        >,
                    > + Send
                    + 'a,
            >,
        > {
            let outcome = self.outcome.clone();
            let error_msg = self.error_msg.clone();
            Box::pin(async move {
                if let Some(msg) = error_msg {
                    return Err(crate::review::ReviewDeliveryError::Transport(msg));
                }
                Ok(crate::review::DeliveredReview {
                    outcome: outcome.expect("either outcome or error_msg must be set"),
                    receipt: crate::review::DeliveryReceipt {
                        data: serde_json::Value::Null,
                    },
                })
            })
        }

        fn report<'a>(
            &'a self,
            _receipt: crate::review::DeliveryReceipt,
            outcome: ReplyOutcome,
        ) -> Pin<Box<dyn Future<Output = Result<(), crate::review::ReviewDeliveryError>> + Send + 'a>>
        {
            self.reports.lock().unwrap().push(outcome);
            Box::pin(async move { Ok(()) })
        }
    }

    /// MockReplyTool — same shape as MockTwitterTool in review/mod.rs.
    struct MockReplyTool {
        canned: Mutex<Option<ToolOutput>>,
        last_input: Mutex<Option<serde_json::Value>>,
    }

    impl MockReplyTool {
        fn success(body: &str) -> Arc<Self> {
            Arc::new(MockReplyTool {
                canned: Mutex::new(Some(ToolOutput::success(body))),
                last_input: Mutex::new(None),
            })
        }

        fn errored(reason: &str) -> Arc<Self> {
            Arc::new(MockReplyTool {
                canned: Mutex::new(Some(ToolOutput::error(reason))),
                last_input: Mutex::new(None),
            })
        }

        fn last_input(&self) -> Option<serde_json::Value> {
            self.last_input.lock().unwrap().clone()
        }
    }

    impl Tool for MockReplyTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "twitter_reply".to_string(),
                description: "mock".to_string(),
                input_schema: serde_json::json!({"type": "object"}),
            }
        }

        fn execute(
            &self,
            _ctx: &ExecutionContext,
            input: serde_json::Value,
        ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, CoreError>> + Send + '_>> {
            *self.last_input.lock().unwrap() = Some(input);
            let canned = self.canned.lock().unwrap().take();
            Box::pin(async move {
                canned.ok_or_else(|| CoreError::Agent("mock reply tool exhausted".into()))
            })
        }
    }

    /// MockProvider — same shape as review/mod.rs::tests::MockProvider.
    struct MockProvider {
        responses: Mutex<VecDeque<String>>,
    }

    impl MockProvider {
        fn arc(responses: Vec<&str>) -> Arc<BoxedProvider> {
            let p = MockProvider {
                responses: Mutex::new(responses.into_iter().map(String::from).collect()),
            };
            Arc::new(BoxedProvider::new(p))
        }
    }

    impl LlmProvider for MockProvider {
        fn complete(
            &self,
            request: CompletionRequest,
        ) -> impl Future<Output = Result<CompletionResponse, CoreError>> + Send {
            let response = self.responses.lock().unwrap().pop_front();
            let has_respond = request
                .tools
                .iter()
                .any(|t| t.name == heartbit_core::llm::types::RESPOND_TOOL_NAME);
            async move {
                let text =
                    response.ok_or_else(|| CoreError::Agent("mock exhausted".to_string()))?;
                let content = if has_respond {
                    let value: serde_json::Value = serde_json::from_str(&text).map_err(|e| {
                        CoreError::Agent(format!("mock: canned response is not valid JSON: {e}"))
                    })?;
                    vec![ContentBlock::ToolUse {
                        id: "respond_1".to_string(),
                        name: "__respond__".to_string(),
                        input: value,
                    }]
                } else {
                    vec![ContentBlock::Text { text }]
                };
                Ok(CompletionResponse {
                    content,
                    usage: TokenUsage::default(),
                    stop_reason: if has_respond {
                        StopReason::ToolUse
                    } else {
                        StopReason::EndTurn
                    },
                    model: None,
                })
            }
        }
    }

    /// Stub credential resolver — never invoked in mock tests.
    struct StubCredentialResolver;

    impl CredentialResolverTrait for StubCredentialResolver {
        fn resolve(
            &self,
            _name: &str,
        ) -> Pin<Box<dyn Future<Output = Result<Secret, CoreError>> + Send + '_>> {
            Box::pin(async move { Ok(Secret::new("stub")) })
        }
    }

    /// Snapshot fixture — mirrors review/mod.rs::tests::seed_snapshot.
    fn seed_snapshot(persona: &str) -> (TempDir, std::path::PathBuf) {
        use crate::voice::{
            BlendEntry, BlendRecipe, EmDashPolicy, EmojiPolicy, Formatting, FragmentFrequency,
            HashtagPolicy, LineBreaks, OpeningPattern, PartialStyleProfile, PeriodsPolicy,
            QuotationMarks, SentenceLengthTarget, SnapshotStore, SpecificityTarget, StyleProfile,
            ThreadRhythm,
        };
        let dir = TempDir::new().unwrap();
        let profile = StyleProfile {
            version: 1,
            sentence_length_target: SentenceLengthTarget::Short,
            sentence_length_distribution: [40, 30, 20, 10],
            fragment_frequency: FragmentFrequency::Common,
            opening_patterns: vec![OpeningPattern::ClaimFirst],
            opening_pattern_weights: vec![1.0],
            formatting: Formatting {
                lowercase: true,
                periods: PeriodsPolicy::Optional,
                em_dashes: EmDashPolicy::Forbidden,
                quotation_marks: QuotationMarks::Double,
                line_breaks: LineBreaks::Single,
            },
            emoji_policy: EmojiPolicy::Never,
            hashtag_policy: HashtagPolicy::Never,
            specificity_target: SpecificityTarget::High,
            voice_traits: vec!["specific".to_string()],
            ai_tells_to_avoid: vec!["delve".to_string()],
            thread_rhythm: ThreadRhythm::Linear,
            thread_max_length: 5,
            thread_opener_must_hook: false,
            topical_obsessions: vec!["AI".to_string()],
            topical_avoidances: vec!["politics".to_string()],
        };
        let recipe = BlendRecipe {
            version: 1,
            blend: vec![BlendEntry {
                writer: "k".to_string(),
                weight: 1.0,
            }],
            overrides: PartialStyleProfile::default(),
        };
        let store = SnapshotStore::open(dir.path(), persona).unwrap();
        store.save_new(profile, &recipe).unwrap();
        let root = dir.path().to_path_buf();
        (dir, root)
    }

    /// Fixture mention.
    fn fixture_mention() -> Mention {
        Mention {
            id: "mention_1".into(),
            text: "how does heartbit compare to rig-rs?".into(),
            author_id: "999".into(),
            author_handle: "grumpy_dev".into(),
            posted_at: chrono::Utc::now(),
            in_reply_to_tweet_id: None,
            conversation_id: None,
        }
    }

    /// Boilerplate builder for ReplyConfig.
    fn mk_reply_cfg<'a>(
        profiles_root: &'a std::path::Path,
        provider: Arc<BoxedProvider>,
        delivery: Arc<dyn ReplyReviewDelivery>,
        twitter_tool: Arc<dyn Tool>,
        candidates_per_reply: usize,
        mention: Mention,
    ) -> ReplyConfig<'a> {
        ReplyConfig {
            persona_name: "x",
            provider,
            corpora_root: profiles_root,
            profiles_root,
            on_progress: None,
            mention,
            parent: None,
            mentioner_context: None,
            candidates_per_reply,
            mode_addendum: None,
            researcher_override: None,
            delivery,
            twitter_tool,
            credentials: Arc::new(StubCredentialResolver),
        }
    }

    // --- Test 1: single candidate, happy path ---

    #[tokio::test]
    async fn run_reply_pipeline_single_candidate_happy_path() {
        let (_dir, profiles_root) = seed_snapshot("x");
        let provider = MockProvider::arc(vec![
            "research digest",
            "concrete short reply",
            r#"{"verdict":"pass","style_match_score":0.92}"#,
            r#"{"verdict":"verified"}"#,
        ]);
        let delivery_concrete =
            MockReplyReviewDelivery::arc(crate::review::DeliveryOutcome::Pick(0));
        let delivery_trait: Arc<dyn ReplyReviewDelivery> = delivery_concrete.clone();
        let twitter_tool = MockReplyTool::success(
            r#"{"tweet_id":"reply123","url":"https://x.com/i/web/status/reply123"}"#,
        );
        let cfg = mk_reply_cfg(
            &profiles_root,
            provider,
            delivery_trait,
            twitter_tool.clone() as Arc<dyn Tool>,
            1,
            fixture_mention(),
        );
        let out = run_reply_pipeline(cfg).await.expect("happy path");
        match out.outcome {
            ReplyOutcome::Posted {
                chosen_index,
                reply_tweet_id,
                reply_url,
            } => {
                assert_eq!(chosen_index, 0);
                assert_eq!(reply_tweet_id, "reply123");
                assert_eq!(reply_url, "https://x.com/i/web/status/reply123");
            }
            other => panic!("expected Posted, got {other:?}"),
        }
        assert_eq!(out.candidates.len(), 1);
        assert_eq!(out.mention_id, "mention_1");

        // Verify delivery.report() was called with the Posted outcome (spec step 12).
        let reports = delivery_concrete.reports.lock().unwrap();
        assert_eq!(reports.len(), 1, "report() should be called exactly once");
        match &reports[0] {
            ReplyOutcome::Posted { chosen_index, .. } => assert_eq!(*chosen_index, 0),
            other => panic!("expected report() to receive Posted, got {other:?}"),
        }
    }

    // --- Test 2: two candidates, judge picks index 1 ---

    #[tokio::test]
    async fn run_reply_pipeline_two_candidates_judge_picks() {
        let (_dir, profiles_root) = seed_snapshot("x");
        // Two parallel candidates + 1 judge = 1 + (writer+critic+fact)×2 + 1 = 8 responses.
        // Identical writer/critic/fact responses so JoinSet ordering doesn't matter.
        let provider = MockProvider::arc(vec![
            "research digest",                               // researcher
            "good reply text",                               // writer (slot 0 or 1)
            r#"{"verdict":"pass","style_match_score":0.8}"#, // critic
            r#"{"verdict":"verified"}"#,                     // fact
            "good reply text",                               // writer (slot 1 or 0)
            r#"{"verdict":"pass","style_match_score":0.8}"#, // critic
            r#"{"verdict":"verified"}"#,                     // fact
            r#"{"chosen_index":1,"reasoning":"second candidate is more specific"}"#, // judge
        ]);
        let delivery = MockReplyReviewDelivery::arc(crate::review::DeliveryOutcome::Pick(1));
        let twitter_tool = MockReplyTool::success(
            r#"{"tweet_id":"reply456","url":"https://x.com/i/web/status/reply456"}"#,
        );
        let cfg = mk_reply_cfg(
            &profiles_root,
            provider,
            delivery as Arc<dyn ReplyReviewDelivery>,
            twitter_tool.clone() as Arc<dyn Tool>,
            2,
            fixture_mention(),
        );
        let out = run_reply_pipeline(cfg).await.expect("two candidates");
        match out.outcome {
            ReplyOutcome::Posted { chosen_index, .. } => {
                assert_eq!(chosen_index, 1);
            }
            other => panic!("expected Posted, got {other:?}"),
        }
        assert_eq!(out.candidates.len(), 2);
    }

    // --- Test 3: writer returns no_reply — NoReply outcome, delivery not called ---

    #[tokio::test]
    async fn run_reply_pipeline_writer_no_reply_returns_no_reply_outcome() {
        let (_dir, profiles_root) = seed_snapshot("x");
        let provider = MockProvider::arc(vec![
            "research digest", // researcher
            "no_reply",        // writer short-circuits; critic+fact not called
        ]);
        // Delivery should NEVER be called — set it to error if it is.
        let delivery = MockReplyReviewDelivery::errored("delivery must not be called");
        let twitter_tool = MockReplyTool::errored("twitter must not be called");
        let cfg = mk_reply_cfg(
            &profiles_root,
            provider,
            delivery as Arc<dyn ReplyReviewDelivery>,
            twitter_tool.clone() as Arc<dyn Tool>,
            1,
            fixture_mention(),
        );
        let out = run_reply_pipeline(cfg).await.expect("no_reply is success");
        assert!(
            matches!(out.outcome, ReplyOutcome::NoReply),
            "expected NoReply, got {:?}",
            out.outcome
        );
        assert!(
            out.candidates.is_empty(),
            "no candidates when writer says no_reply"
        );
        assert!(
            twitter_tool.last_input().is_none(),
            "twitter_tool must not be called"
        );
    }

    // --- Test 4: 281-char draft rejected by publish gate before delivery ---

    #[tokio::test]
    async fn run_reply_pipeline_publish_gate_rejects_281_chars() {
        let (_dir, profiles_root) = seed_snapshot("x");
        let too_long = "x".repeat(281);
        let provider = MockProvider::arc(vec![
            "research digest",
            too_long.as_str(),
            r#"{"verdict":"pass","style_match_score":0.9}"#,
            r#"{"verdict":"verified"}"#,
        ]);
        // Gate rejects BEFORE delivery — errored delivery should never fire.
        let delivery = MockReplyReviewDelivery::errored("delivery must not be called");
        let twitter_tool = MockReplyTool::errored("twitter must not be called");
        let cfg = mk_reply_cfg(
            &profiles_root,
            provider,
            delivery as Arc<dyn ReplyReviewDelivery>,
            twitter_tool.clone() as Arc<dyn Tool>,
            1,
            fixture_mention(),
        );
        let out = run_reply_pipeline(cfg)
            .await
            .expect("gate rejection is success");
        match out.outcome {
            ReplyOutcome::GateRejected {
                chosen_index,
                reason,
            } => {
                assert_eq!(chosen_index, 0);
                assert!(reason.contains("exceeds 280 chars"), "got: {reason}");
            }
            other => panic!("expected GateRejected, got {other:?}"),
        }
        assert!(
            twitter_tool.last_input().is_none(),
            "twitter_tool must not be called on gate rejection"
        );
    }

    // --- Test 5: user presses Skip ---

    #[tokio::test]
    async fn run_reply_pipeline_user_skip_returns_skipped() {
        let (_dir, profiles_root) = seed_snapshot("x");
        let provider = MockProvider::arc(vec![
            "research digest",
            "short reply candidate",
            r#"{"verdict":"pass","style_match_score":0.85}"#,
            r#"{"verdict":"verified"}"#,
        ]);
        let delivery = MockReplyReviewDelivery::arc(crate::review::DeliveryOutcome::Skip);
        let twitter_tool = MockReplyTool::errored("twitter must not be called on skip");
        let cfg = mk_reply_cfg(
            &profiles_root,
            provider,
            delivery as Arc<dyn ReplyReviewDelivery>,
            twitter_tool.clone() as Arc<dyn Tool>,
            1,
            fixture_mention(),
        );
        let out = run_reply_pipeline(cfg).await.expect("skip is success");
        assert!(
            matches!(out.outcome, ReplyOutcome::Skipped),
            "expected Skipped, got {:?}",
            out.outcome
        );
        assert!(
            twitter_tool.last_input().is_none(),
            "twitter_tool must not be called on skip"
        );
    }

    // --- Test 6: Twitter API error returns PublishFailed ---

    #[tokio::test]
    async fn run_reply_pipeline_twitter_api_error_returns_publish_failed() {
        let (_dir, profiles_root) = seed_snapshot("x");
        let provider = MockProvider::arc(vec![
            "research digest",
            "short reply candidate",
            r#"{"verdict":"pass","style_match_score":0.85}"#,
            r#"{"verdict":"verified"}"#,
        ]);
        let delivery = MockReplyReviewDelivery::arc(crate::review::DeliveryOutcome::Pick(0));
        let twitter_tool = MockReplyTool::errored("rate limited (429)");
        let cfg = mk_reply_cfg(
            &profiles_root,
            provider,
            delivery as Arc<dyn ReplyReviewDelivery>,
            twitter_tool.clone() as Arc<dyn Tool>,
            1,
            fixture_mention(),
        );
        let out = run_reply_pipeline(cfg)
            .await
            .expect("publish failure is success");
        match out.outcome {
            ReplyOutcome::PublishFailed {
                chosen_index,
                reason,
            } => {
                assert_eq!(chosen_index, 0);
                assert!(
                    reason.contains("rate limited") || reason.contains("429"),
                    "got: {reason}"
                );
            }
            other => panic!("expected PublishFailed, got {other:?}"),
        }
    }

    #[test]
    fn mention_deserializes_without_conversation_id_field() {
        // Backward compat: old stores wrote Mention without the field.
        let json = r#"{
            "id": "1",
            "text": "hi",
            "author_id": "12",
            "author_handle": "alice",
            "posted_at": "2026-05-08T11:02:00Z",
            "in_reply_to_tweet_id": null
        }"#;
        let m: Mention = serde_json::from_str(json).expect("backward compat");
        assert!(m.conversation_id.is_none());
    }

    #[test]
    fn mention_round_trips_conversation_id() {
        let m = Mention {
            id: "1".into(),
            text: "hi".into(),
            author_id: "12".into(),
            author_handle: "alice".into(),
            posted_at: Utc::now(),
            in_reply_to_tweet_id: Some("99".into()),
            conversation_id: Some("conv-123".into()),
        };
        let s = serde_json::to_string(&m).unwrap();
        let parsed: Mention = serde_json::from_str(&s).unwrap();
        assert_eq!(parsed.conversation_id.as_deref(), Some("conv-123"));
    }
}
