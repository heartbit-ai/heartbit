//! LLM-as-judge filter for low-quality mentions.
//!
//! Runs ONE cheap LLM call per mention to classify whether the content
//! looks like a crypto/financial scam, generic spam, or an ad — things
//! that should never get a human-quality reply.
//!
//! **Why this layer exists**: the structural guards (spam, thread, bot
//! heuristic, conversation, budget) catch handle-pattern and
//! rate-shaped junk, but they're content-blind. A crypto scammer with a
//! clean-looking handle and a normal-looking account profile gets
//! through them. The reply writer can't tell either — it just generates
//! a thoughtful answer to a fake question.
//!
//! **Why LLM, not keyword blocklist**: simple substring matches on
//! "crypto" / "$TOKEN" / "DM me" fire on legitimate technical
//! discussion. An LLM with a tight rubric can distinguish "what's your
//! take on the $TOKEN pump?" (legit discussion) from "yo $TOKEN going
//! to 100x, DM me for entry" (scam).
//!
//! **Where it runs**: in `reply_draft_handler`, after the
//! `was_posted` redelivery check and BEFORE `run_reply_pipeline`. A
//! non-OK verdict marks the mention replied and returns `Skipped`
//! without burning multi-agent pipeline tokens.
//!
//! **Fail-open**: if the judge errors, times out, or returns garbage,
//! the verdict is treated as `Ok`. We err on the side of replying —
//! a flaky guardrail must not silently suppress legitimate engagement.

use std::sync::Arc;
use std::time::Duration;

use heartbit_core::llm::types::{
    CompletionRequest, ContentBlock, Message, ReasoningEffort, ToolChoice,
};
use heartbit_core::llm::{BoxedProvider, LlmProvider};

/// Verdict from the scam judge. Non-OK variants carry a one-line
/// rationale from the model — useful for logs and metrics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScamVerdict {
    /// Legitimate engagement — proceed to the full reply pipeline.
    Ok,
    /// Crypto / financial / "guaranteed returns" / fake-giveaway bait.
    Scam(String),
    /// Bot-style engagement farming, unrelated promotion, hashtag stuffing.
    Spam(String),
    /// Clear product/service promotion, paid placement, affiliate links.
    Ad(String),
}

impl ScamVerdict {
    /// Whether the verdict allows the reply pipeline to run.
    pub fn is_ok(&self) -> bool {
        matches!(self, ScamVerdict::Ok)
    }

    /// Compact label for logs / metrics. Stable across releases.
    pub fn label(&self) -> &'static str {
        match self {
            ScamVerdict::Ok => "ok",
            ScamVerdict::Scam(_) => "scam",
            ScamVerdict::Spam(_) => "spam",
            ScamVerdict::Ad(_) => "ad",
        }
    }

    /// Model-supplied reason, when present.
    pub fn reason(&self) -> Option<&str> {
        match self {
            ScamVerdict::Ok => None,
            ScamVerdict::Scam(r) | ScamVerdict::Spam(r) | ScamVerdict::Ad(r) => Some(r.as_str()),
        }
    }
}

/// Default judge system prompt. Tight rubric, single-line response.
const DEFAULT_SYSTEM_PROMPT: &str = "\
You are a content-quality classifier for an X (Twitter) reply bot. The bot \
replies on behalf of a real person to mentions of their account. Your job is \
to identify mentions that should NOT receive a human-quality reply: \
crypto/financial scams, spam, and generic ads.

You will receive one mention text plus the author's handle. Classify it as \
ONE of:

VERDICT: OK
VERDICT: SCAM: <one-line reason>
VERDICT: SPAM: <one-line reason>
VERDICT: AD: <one-line reason>

Definitions:
- SCAM: crypto pump posts, airdrop bait, \"guaranteed returns\", \"DM for \
  entry\", fake giveaways, financial-advice traps, impersonation. Legitimate \
  technical discussion of crypto/web3 is OK.
- SPAM: engagement farming (\"nice post bro\", \"follow me back\"), unrelated \
  promotion, hashtag stuffing, copy-paste replies.
- AD: clear product/service promotion, paid placement, affiliate links.
- OK: questions, comments, disagreements, technical discussion, jokes, \
  reactions to the parent tweet — anything that resembles a real human \
  engaging with a real human.

When in doubt, classify as OK. We err on the side of replying. Only mark \
non-OK when there is a clear signal.

Reply with exactly ONE line in the format above. No preamble. No analysis.";

/// One-shot LLM classifier for mention quality.
///
/// Construct via `ScamJudge::new(provider)`; reuse across mentions.
/// Calls `provider.complete(...)` with a tight prompt and a low
/// `max_tokens` budget. Fail-open on any error.
pub struct ScamJudge {
    provider: Arc<BoxedProvider>,
    system_prompt: String,
    timeout: Duration,
    max_tokens: u32,
}

impl ScamJudge {
    /// Build a judge that uses `provider` for classification calls.
    /// Defaults: 8s timeout, 96 max output tokens (one-line verdict).
    pub fn new(provider: Arc<BoxedProvider>) -> Self {
        Self {
            provider,
            system_prompt: DEFAULT_SYSTEM_PROMPT.to_string(),
            timeout: Duration::from_secs(8),
            max_tokens: 96,
        }
    }

    /// Override the timeout (default 8s). Use a tight value: this runs
    /// per-mention, so slow judges block the whole pipeline.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Override the max output tokens (default 96).
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Replace the default system prompt. Most callers should leave this
    /// alone; useful for tests and for domain-specific tuning.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    /// Classify one mention. `author_handle` may be empty when
    /// enrichment is disabled or fails — the judge still works on text
    /// alone.
    ///
    /// Returns `ScamVerdict::Ok` on any judge failure (timeout, provider
    /// error, parse failure). Caller logs are the only signal that
    /// classification was skipped.
    pub async fn evaluate(&self, mention_text: &str, author_handle: &str) -> ScamVerdict {
        let user_msg = if author_handle.is_empty() {
            format!("Mention text:\n\n{mention_text}")
        } else {
            format!("Mention by @{author_handle}:\n\n{mention_text}")
        };

        let request = CompletionRequest {
            system: self.system_prompt.clone(),
            messages: vec![Message::user(user_msg)],
            tools: vec![],
            max_tokens: self.max_tokens,
            tool_choice: Some(ToolChoice::Auto),
            reasoning_effort: None::<ReasoningEffort>,
        };

        let call = LlmProvider::complete(self.provider.as_ref(), request);
        let response = match tokio::time::timeout(self.timeout, call).await {
            Ok(Ok(r)) => r,
            Ok(Err(e)) => {
                tracing::warn!(error = %e, "scam judge LLM error — fail-open to Ok");
                return ScamVerdict::Ok;
            }
            Err(_) => {
                tracing::warn!(
                    timeout_secs = self.timeout.as_secs(),
                    "scam judge timed out — fail-open to Ok"
                );
                return ScamVerdict::Ok;
            }
        };

        let text = response
            .content
            .iter()
            .find_map(|b| match b {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .unwrap_or("");

        match parse_verdict(text) {
            Some(v) => v,
            None => {
                tracing::warn!(
                    judge_text = %text.chars().take(120).collect::<String>(),
                    "scam judge response did not parse — fail-open to Ok"
                );
                ScamVerdict::Ok
            }
        }
    }
}

/// Parse a raw judge response into a `ScamVerdict`. Returns `None` when
/// the response doesn't match any expected shape (caller fail-opens).
fn parse_verdict(text: &str) -> Option<ScamVerdict> {
    // Find the first line that starts with "VERDICT:" — the model may
    // emit whitespace or a stray preamble despite the instruction.
    let line = text.lines().find_map(|l| {
        let trimmed = l.trim();
        trimmed
            .strip_prefix("VERDICT:")
            .or_else(|| trimmed.strip_prefix("Verdict:"))
            .map(str::trim)
    })?;

    // line is now everything after `VERDICT:` — e.g. "OK" or
    // "SCAM: crypto pump bait".
    let (kind, reason) = match line.split_once(':') {
        Some((kind, rest)) => (kind.trim().to_uppercase(), rest.trim().to_string()),
        None => (line.trim().to_uppercase(), String::new()),
    };

    match kind.as_str() {
        "OK" => Some(ScamVerdict::Ok),
        "SCAM" => Some(ScamVerdict::Scam(reason)),
        "SPAM" => Some(ScamVerdict::Spam(reason)),
        "AD" => Some(ScamVerdict::Ad(reason)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::future::Future;
    use std::sync::Mutex;

    use heartbit_core::Error as CoreError;
    use heartbit_core::llm::types::{
        CompletionRequest, CompletionResponse, ContentBlock, StopReason, TokenUsage,
    };
    use heartbit_core::llm::{BoxedProvider, LlmProvider};

    use super::*;

    // ── MockProvider: returns canned text responses in FIFO order ──────

    struct MockProvider {
        responses: Mutex<VecDeque<String>>,
        delay: Option<Duration>,
    }

    impl MockProvider {
        fn arc(texts: Vec<&str>) -> Arc<BoxedProvider> {
            Arc::new(BoxedProvider::new(MockProvider {
                responses: Mutex::new(texts.into_iter().map(String::from).collect()),
                delay: None,
            }))
        }

        fn slow(delay: Duration) -> Arc<BoxedProvider> {
            Arc::new(BoxedProvider::new(MockProvider {
                responses: Mutex::new(VecDeque::new()),
                delay: Some(delay),
            }))
        }

        fn failing() -> Arc<BoxedProvider> {
            Arc::new(BoxedProvider::new(MockProvider {
                responses: Mutex::new(VecDeque::new()),
                delay: None,
            }))
        }
    }

    impl LlmProvider for MockProvider {
        fn complete(
            &self,
            _request: CompletionRequest,
        ) -> impl Future<Output = Result<CompletionResponse, CoreError>> + Send {
            let response = self.responses.lock().unwrap().pop_front();
            let delay = self.delay;
            async move {
                if let Some(d) = delay {
                    tokio::time::sleep(d).await;
                }
                let text =
                    response.ok_or_else(|| CoreError::Agent("mock exhausted".to_string()))?;
                Ok(CompletionResponse {
                    content: vec![ContentBlock::Text { text }],
                    usage: TokenUsage::default(),
                    stop_reason: StopReason::EndTurn,
                    model: None,
                    reasoning: None,
                })
            }
        }
    }

    // ── parse_verdict: pure logic, no I/O ─────────────────────────────

    #[test]
    fn parse_verdict_ok() {
        assert_eq!(parse_verdict("VERDICT: OK"), Some(ScamVerdict::Ok));
        assert_eq!(parse_verdict("  VERDICT: OK  "), Some(ScamVerdict::Ok));
        // Stray preamble before the verdict line is tolerated.
        assert_eq!(
            parse_verdict("After careful analysis:\nVERDICT: OK"),
            Some(ScamVerdict::Ok)
        );
        // Case-tolerant on "Verdict:" prefix (some models lowercase it).
        assert_eq!(parse_verdict("Verdict: OK"), Some(ScamVerdict::Ok));
    }

    #[test]
    fn parse_verdict_scam_with_reason() {
        let v = parse_verdict("VERDICT: SCAM: crypto pump bait, suspicious DM ask");
        assert_eq!(
            v,
            Some(ScamVerdict::Scam(
                "crypto pump bait, suspicious DM ask".to_string()
            ))
        );
    }

    #[test]
    fn parse_verdict_spam_and_ad() {
        assert_eq!(
            parse_verdict("VERDICT: SPAM: engagement farming"),
            Some(ScamVerdict::Spam("engagement farming".to_string()))
        );
        assert_eq!(
            parse_verdict("VERDICT: AD: affiliate link promotion"),
            Some(ScamVerdict::Ad("affiliate link promotion".to_string()))
        );
    }

    #[test]
    fn parse_verdict_unknown_kind_returns_none() {
        // Caller treats `None` as fail-open.
        assert_eq!(parse_verdict("VERDICT: MAYBE: idk"), None);
        assert_eq!(parse_verdict("nothing to see here"), None);
        assert_eq!(parse_verdict(""), None);
    }

    #[test]
    fn verdict_label_and_reason() {
        assert_eq!(ScamVerdict::Ok.label(), "ok");
        assert!(ScamVerdict::Ok.is_ok());
        assert_eq!(ScamVerdict::Ok.reason(), None);

        let v = ScamVerdict::Scam("crypto pump".to_string());
        assert_eq!(v.label(), "scam");
        assert!(!v.is_ok());
        assert_eq!(v.reason(), Some("crypto pump"));
    }

    // ── evaluate(): end-to-end with MockProvider ──────────────────────

    #[tokio::test]
    async fn evaluate_returns_ok_on_legit_mention() {
        let judge = ScamJudge::new(MockProvider::arc(vec!["VERDICT: OK"]));
        let v = judge
            .evaluate(
                "great talk on async runtimes — what about io_uring?",
                "real_engineer",
            )
            .await;
        assert_eq!(v, ScamVerdict::Ok);
    }

    #[tokio::test]
    async fn evaluate_classifies_crypto_pump_as_scam() {
        let judge = ScamJudge::new(MockProvider::arc(vec![
            "VERDICT: SCAM: crypto pump with DM ask",
        ]));
        let v = judge
            .evaluate(
                "yo $PUMP going 100x today, DM me for the entry alpha",
                "Crypto_Pump_69",
            )
            .await;
        assert!(matches!(v, ScamVerdict::Scam(_)));
        assert_eq!(v.reason(), Some("crypto pump with DM ask"));
    }

    #[tokio::test]
    async fn evaluate_fail_opens_on_provider_error() {
        // No canned responses — provider errors with "mock exhausted".
        let judge = ScamJudge::new(MockProvider::failing());
        let v = judge.evaluate("anything", "anyone").await;
        assert_eq!(
            v,
            ScamVerdict::Ok,
            "judge failures must fail-open to OK so legit traffic is not silently dropped"
        );
    }

    #[tokio::test]
    async fn evaluate_fail_opens_on_timeout() {
        // Provider sleeps 5s; judge timeout 50ms → fail-open.
        // We use real sleep (not paused time) because heartbit-ghost's
        // tokio doesn't carry the `test-util` feature — the test still
        // completes in ~50ms, fast enough for the suite.
        let judge = ScamJudge::new(MockProvider::slow(Duration::from_secs(5)))
            .with_timeout(Duration::from_millis(50));
        let v = judge.evaluate("anything", "anyone").await;
        assert_eq!(v, ScamVerdict::Ok, "timeout must fail-open to OK");
    }

    #[tokio::test]
    async fn evaluate_fail_opens_on_unparseable_response() {
        // Model returns a non-verdict response.
        let judge = ScamJudge::new(MockProvider::arc(vec![
            "I think this looks fine, no concerns here",
        ]));
        let v = judge.evaluate("anything", "anyone").await;
        assert_eq!(v, ScamVerdict::Ok, "garbage response must fail-open");
    }

    #[tokio::test]
    async fn evaluate_handles_empty_author_handle() {
        // When enrichment is off, author_handle is empty. The judge
        // should still run on text alone (no panic, no special-casing).
        let judge = ScamJudge::new(MockProvider::arc(vec!["VERDICT: SPAM: engagement farming"]));
        let v = judge.evaluate("nice post bro!", "").await;
        assert!(matches!(v, ScamVerdict::Spam(_)));
    }
}
