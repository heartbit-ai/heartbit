//! LLM-based style extractor — turns a [`Corpus`] into a validated
//! [`StyleProfile`] via a single analyst-persona LLM call (umbrella spec
//! §2.3). Sibling of [`crate::voice::style`] (the schema) and
//! [`crate::corpus::Corpus`] (the input).
//!
//! Bodies for [`StyleExtractor`], [`StyleExtractorBuilder`],
//! [`default_system_prompt`], and the pure helpers land in subsequent
//! tasks.

use std::sync::Arc;
use std::time::Duration;

use heartbit_core::llm::types::{CompletionRequest, ContentBlock, Message};
use heartbit_core::llm::{BoxedProvider, LlmProvider};
use thiserror::Error;

use crate::corpus::{Corpus, CorpusEntry};
use crate::voice::error::VoiceError;
use crate::voice::style::StyleProfile;

/// Errors raised by [`StyleExtractor::extract`] (added in Task 3).
#[derive(Debug, Error)]
pub enum ExtractError {
    /// The corpus had zero entries; nothing to analyze.
    #[error("corpus is empty for writer '{0}'")]
    EmptyCorpus(String),

    /// The underlying LLM call failed (network, auth, rate limit, etc.).
    #[error("llm: {0}")]
    Llm(#[source] heartbit_core::Error),

    /// The LLM call exceeded the configured timeout.
    #[error("llm call timed out after {0:?}")]
    Timeout(std::time::Duration),

    /// The LLM produced no text content (e.g., refusal, empty response).
    #[error("llm produced no text response")]
    EmptyResponse,

    /// JSON parse failure. `raw` carries the offending output for debugging.
    #[error("json parse: {source}")]
    JsonParse {
        /// The underlying serde_json error.
        #[source]
        source: serde_json::Error,
        /// The raw LLM output that failed to parse.
        raw: String,
    },

    /// Parsed cleanly but failed [`crate::voice::StyleProfile::validate`]
    /// (sums, ranges, etc.). `raw` carries the offending output; `inner`
    /// is the validation error.
    #[error("validation: {inner}")]
    Validation {
        /// The underlying validation error from `StyleProfile::validate`.
        #[source]
        inner: VoiceError,
        /// The raw LLM output that produced an invalid profile.
        raw: String,
    },
}

/// The default analyst-persona system prompt. Public so callers can
/// inspect it (e.g., for logging) or wrap it before passing back via
/// [`StyleExtractorBuilder::system_prompt`] (added in Task 3).
pub fn default_system_prompt() -> &'static str {
    DEFAULT_SYSTEM_PROMPT
}

const DEFAULT_SYSTEM_PROMPT: &str = r#"You are a writing-style analyst. Your job: read sample posts from one writer and produce a structured fingerprint of their voice as a JSON object.

OUTPUT FORMAT — emit a single JSON object matching exactly this shape (no preamble, no markdown fences, no commentary):

{
  "version": 1,
  "sentence_length_target": "short" | "mixed" | "long",
  "sentence_length_distribution": [<u8>, <u8>, <u8>, <u8>],   // % of posts at lengths [<10, 10–20, 20–40, >40] words. Must sum to 100.
  "fragment_frequency": "rare" | "occasional" | "common",
  "opening_patterns": [<one or more of: "claim_first" | "number_first" | "scene_first" | "question_first" | "aphoristic_first" | "anecdote_first" | "contrarian_first">],
  "opening_pattern_weights": [<f64>, ...],   // parallel to opening_patterns; in [0, 1]; must sum to 1.0
  "formatting": {
    "lowercase": <bool>,
    "periods": "always" | "optional" | "rare",
    "em_dashes": "preferred" | "ok" | "forbidden",
    "quotation_marks": "double" | "single" | "smart",
    "line_breaks": "single" | "double" | "rhythmic"
  },
  "emoji_policy": "never" | "rare_punchline_only" | "occasional" | "frequent",
  "hashtag_policy": "never" | "rare" | "topic_relevant" | "always",
  "specificity_target": "low" | "medium" | "high",
  "voice_traits": [<short snake_case strings>],
  "ai_tells_to_avoid": [<short strings the writer never uses>],
  "thread_rhythm": "linear" | "list_then_payoff" | "punchline_callbacks",
  "thread_max_length": <u32 in 1..=25>,
  "thread_opener_must_hook": <bool>,
  "topical_obsessions": [<short strings>],
  "topical_avoidances": [<short strings>]
}

ANALYSIS GUIDANCE
- Read every post before answering. Look for stable patterns, not one-off quirks.
- Prefer evidence-based claims: if you see 8 short sentences and 2 long ones, "short" is the target with a [60, 30, 10, 0]-ish distribution — not "mixed".
- voice_traits and ai_tells_to_avoid must be observed in the corpus. Do not invent generic AI advice.
- topical_obsessions/avoidances reflect what THIS writer actually posts about (or pointedly doesn't), not generic categories.
- If the writer mostly does standalone posts, set thread_max_length=1 and thread_opener_must_hook=false.

CONSTRAINTS — your JSON must satisfy these or it will be rejected:
- sentence_length_distribution sums to 100
- opening_patterns and opening_pattern_weights have the same length
- opening_pattern_weights sum to 1.0 (within 1e-6)
- thread_max_length is 1..=25
- enum strings match the snake_case vocabulary above exactly

OUTPUT THE JSON OBJECT ONLY. No "Here is the analysis", no code fences, no trailing prose.
"#;

/// Sort `entries` by descending engagement (likes, then reposts, then
/// replies, then `posted_at`), and return references to the top `k`.
///
/// Pure function — no I/O, deterministic. Engagement-less entries sort
/// to the bottom (treated as zero engagement).
pub(crate) fn select_top_k(entries: &[CorpusEntry], k: usize) -> Vec<&CorpusEntry> {
    let mut sorted: Vec<&CorpusEntry> = entries.iter().collect();
    sorted.sort_by(|a, b| {
        let a_eng = a.engagement.unwrap_or_default();
        let b_eng = b.engagement.unwrap_or_default();
        b_eng
            .likes
            .cmp(&a_eng.likes)
            .then(b_eng.reposts.cmp(&a_eng.reposts))
            .then(b_eng.replies.cmp(&a_eng.replies))
            .then_with(|| match (b.posted_at, a.posted_at) {
                (Some(b_at), Some(a_at)) => b_at.cmp(&a_at),
                (Some(_), None) => std::cmp::Ordering::Less,
                (None, Some(_)) => std::cmp::Ordering::Greater,
                (None, None) => std::cmp::Ordering::Equal,
            })
    });
    sorted.truncate(k);
    sorted
}

/// Render the user-message text the LLM sees: a header naming the writer
/// and sample size, then numbered post blocks (with engagement when
/// present), then a closing instruction.
pub(crate) fn render_user_message(writer: &str, samples: &[&CorpusEntry]) -> String {
    let mut out = String::new();
    out.push_str(&format!("Writer: @{writer}\n"));
    out.push_str(&format!(
        "Sample size: {} posts (top by engagement)\n\n",
        samples.len()
    ));
    for (idx, entry) in samples.iter().enumerate() {
        let n = idx + 1;
        match entry.engagement {
            Some(eng) => {
                out.push_str(&format!(
                    "POST {n} ({} likes, {} reposts, {} replies):\n",
                    eng.likes, eng.reposts, eng.replies
                ));
            }
            None => {
                out.push_str(&format!("POST {n} (no engagement data):\n"));
            }
        }
        out.push_str(&entry.post_text);
        out.push_str("\n\n");
    }
    out.push_str("Now produce the JSON object.\n");
    out
}

/// Extracts a structured [`StyleProfile`] from a writer's [`Corpus`] via a
/// single LLM call.
///
/// Build via [`StyleExtractor::builder`].
pub struct StyleExtractor {
    provider: Arc<BoxedProvider>,
    sample_size: usize,
    max_response_tokens: u32,
    timeout: Duration,
    system_prompt: String,
}

impl std::fmt::Debug for StyleExtractor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StyleExtractor")
            .field("sample_size", &self.sample_size)
            .field("max_response_tokens", &self.max_response_tokens)
            .field("timeout", &self.timeout)
            .field("system_prompt_len", &self.system_prompt.len())
            .finish_non_exhaustive()
    }
}

impl StyleExtractor {
    /// Start building an extractor with the supplied LLM provider.
    pub fn builder(provider: Arc<BoxedProvider>) -> StyleExtractorBuilder {
        StyleExtractorBuilder {
            provider,
            sample_size: 50,
            max_response_tokens: 2048,
            timeout: Duration::from_secs(60),
            custom_system_prompt: None,
        }
    }

    /// Extract a [`StyleProfile`] from `corpus`.
    ///
    /// Selects the top `sample_size` entries by engagement, renders them
    /// into the user message, calls the LLM, parses the response as JSON,
    /// and runs [`StyleProfile::validate`].
    pub async fn extract(&self, corpus: &Corpus) -> Result<StyleProfile, ExtractError> {
        if corpus.is_empty() {
            return Err(ExtractError::EmptyCorpus(corpus.writer().to_string()));
        }
        let samples = select_top_k(corpus.entries(), self.sample_size);
        let user_msg = render_user_message(corpus.writer(), &samples);

        let request = CompletionRequest {
            system: self.system_prompt.clone(),
            messages: vec![Message::user(user_msg)],
            tools: vec![],
            max_tokens: self.max_response_tokens,
            tool_choice: None,
            reasoning_effort: None,
        };

        let response = match tokio::time::timeout(
            self.timeout,
            LlmProvider::complete(self.provider.as_ref(), request),
        )
        .await
        {
            Ok(Ok(r)) => r,
            Ok(Err(e)) => return Err(ExtractError::Llm(e)),
            Err(_elapsed) => return Err(ExtractError::Timeout(self.timeout)),
        };

        let text: String = response
            .content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Err(ExtractError::EmptyResponse);
        }
        let unfenced = strip_markdown_fences(trimmed);

        let profile: StyleProfile =
            serde_json::from_str(&unfenced).map_err(|source| ExtractError::JsonParse {
                source,
                raw: unfenced.clone(),
            })?;
        profile
            .validate()
            .map_err(|inner| ExtractError::Validation {
                inner,
                raw: unfenced.clone(),
            })?;
        Ok(profile)
    }
}

/// Builder for [`StyleExtractor`]. Use [`StyleExtractor::builder`] to construct.
pub struct StyleExtractorBuilder {
    provider: Arc<BoxedProvider>,
    sample_size: usize,
    max_response_tokens: u32,
    timeout: Duration,
    custom_system_prompt: Option<String>,
}

impl std::fmt::Debug for StyleExtractorBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StyleExtractorBuilder")
            .field("sample_size", &self.sample_size)
            .field("max_response_tokens", &self.max_response_tokens)
            .field("timeout", &self.timeout)
            .field(
                "has_custom_system_prompt",
                &self.custom_system_prompt.is_some(),
            )
            .finish_non_exhaustive()
    }
}

impl StyleExtractorBuilder {
    /// Override the top-K sample cap. Default: 50.
    pub fn sample_size(mut self, k: usize) -> Self {
        self.sample_size = k;
        self
    }

    /// Override the `max_tokens` budget for the response. Default: 2048.
    pub fn max_response_tokens(mut self, t: u32) -> Self {
        self.max_response_tokens = t;
        self
    }

    /// Override the per-call timeout. Default: 60s.
    pub fn timeout(mut self, d: Duration) -> Self {
        self.timeout = d;
        self
    }

    /// Override the system prompt. Default: [`default_system_prompt`].
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.custom_system_prompt = Some(prompt.into());
        self
    }

    /// Finalize the builder.
    pub fn build(self) -> StyleExtractor {
        let system_prompt = self
            .custom_system_prompt
            .unwrap_or_else(|| default_system_prompt().to_string());
        StyleExtractor {
            provider: self.provider,
            sample_size: self.sample_size,
            max_response_tokens: self.max_response_tokens,
            timeout: self.timeout,
            system_prompt,
        }
    }
}

/// Strip a single ```json ... ``` (or ``` ... ```) markdown fence pair if
/// the LLM ignored the prompt's no-fences instruction. Returns the input
/// unchanged when no matching pair is present.
fn strip_markdown_fences(text: &str) -> String {
    let trimmed = text.trim();
    let after_opening = trimmed
        .strip_prefix("```json\n")
        .or_else(|| trimmed.strip_prefix("```json"))
        .or_else(|| trimmed.strip_prefix("```\n"))
        .or_else(|| trimmed.strip_prefix("```"))
        .unwrap_or(trimmed);
    let body = after_opening.strip_suffix("```").unwrap_or(after_opening);
    body.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_corpus_error_includes_writer_handle() {
        let e = ExtractError::EmptyCorpus("karpathy".to_string());
        let s = format!("{e}");
        assert!(s.contains("karpathy"), "got: {s}");
        assert!(s.starts_with("corpus is empty"), "got: {s}");
    }

    #[test]
    fn jsonparse_error_renders_with_source_message() {
        let bad = serde_json::from_str::<serde_json::Value>("not-json").unwrap_err();
        let e = ExtractError::JsonParse {
            source: bad,
            raw: "not-json".to_string(),
        };
        let s = format!("{e}");
        assert!(s.starts_with("json parse: "), "got: {s}");
    }

    #[test]
    fn validation_error_carries_raw_and_inner() {
        let inner = VoiceError::Validation("weights must sum to 1.0".to_string());
        let raw = r#"{"opening_pattern_weights":[0.5,0.4]}"#.to_string();
        let e = ExtractError::Validation {
            inner,
            raw: raw.clone(),
        };
        let s = format!("{e}");
        assert!(s.contains("validation"), "got: {s}");
        assert!(s.contains("weights must sum to 1.0"), "got: {s}");
        // raw is reachable for debugging
        if let ExtractError::Validation { raw: r, .. } = &e {
            assert_eq!(r, &raw);
        } else {
            panic!("not a Validation variant");
        }
    }

    use crate::corpus::{CorpusEntry, Engagement};
    use chrono::{DateTime, Utc};

    fn entry(id: &str, text: &str, eng: Option<Engagement>, at: Option<&str>) -> CorpusEntry {
        CorpusEntry {
            id: id.to_string(),
            post_text: text.to_string(),
            posted_at: at.map(|s| s.parse::<DateTime<Utc>>().unwrap()),
            engagement: eng,
            tags: Vec::new(),
            embedding: None,
        }
    }

    fn eng(likes: u64, reposts: u64, replies: u64) -> Engagement {
        Engagement {
            likes,
            reposts,
            replies,
        }
    }

    // ---- select_top_k ----------------------------------------------------

    #[test]
    fn select_top_k_empty_returns_empty() {
        let v: Vec<CorpusEntry> = Vec::new();
        let out = select_top_k(&v, 5);
        assert!(out.is_empty());
    }

    #[test]
    fn select_top_k_smaller_than_k_returns_all() {
        let entries = vec![
            entry("1", "a", Some(eng(10, 0, 0)), None),
            entry("2", "b", Some(eng(20, 0, 0)), None),
        ];
        let out = select_top_k(&entries, 5);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn select_top_k_orders_by_likes_desc() {
        let entries = vec![
            entry("low", "a", Some(eng(5, 0, 0)), None),
            entry("high", "b", Some(eng(100, 0, 0)), None),
            entry("mid", "c", Some(eng(50, 0, 0)), None),
        ];
        let out = select_top_k(&entries, 3);
        let ids: Vec<&str> = out.iter().map(|e| e.id.as_str()).collect();
        assert_eq!(ids, vec!["high", "mid", "low"]);
    }

    #[test]
    fn select_top_k_tiebreaks_by_reposts_then_replies() {
        let entries = vec![
            entry("a", "1", Some(eng(10, 1, 5)), None),
            entry("b", "2", Some(eng(10, 5, 1)), None),
            entry("c", "3", Some(eng(10, 5, 9)), None),
        ];
        let out = select_top_k(&entries, 3);
        let ids: Vec<&str> = out.iter().map(|e| e.id.as_str()).collect();
        // likes are equal; reposts: c=5, b=5, a=1 → c/b before a.
        // c and b tie on reposts; replies: c=9, b=1 → c first.
        assert_eq!(ids, vec!["c", "b", "a"]);
    }

    #[test]
    fn select_top_k_tiebreaks_by_posted_at_desc_when_engagement_equal() {
        let entries = vec![
            entry(
                "old",
                "a",
                Some(eng(10, 0, 0)),
                Some("2024-01-01T00:00:00Z"),
            ),
            entry(
                "new",
                "b",
                Some(eng(10, 0, 0)),
                Some("2025-01-01T00:00:00Z"),
            ),
        ];
        let out = select_top_k(&entries, 2);
        let ids: Vec<&str> = out.iter().map(|e| e.id.as_str()).collect();
        assert_eq!(ids, vec!["new", "old"]);
    }

    #[test]
    fn select_top_k_engagementless_entries_sort_to_bottom() {
        let entries = vec![
            entry("eng-low", "a", Some(eng(1, 0, 0)), None),
            entry("no-eng-1", "b", None, None),
            entry("eng-high", "c", Some(eng(100, 0, 0)), None),
        ];
        let out = select_top_k(&entries, 3);
        let ids: Vec<&str> = out.iter().map(|e| e.id.as_str()).collect();
        // Expected: eng-high (100) > eng-low (1) > no-eng-1 (treated as 0)
        assert_eq!(ids, vec!["eng-high", "eng-low", "no-eng-1"]);
    }

    // ---- render_user_message --------------------------------------------

    #[test]
    fn render_user_message_includes_writer_handle() {
        let entries = [entry("1", "hello", None, None)];
        let refs: Vec<&CorpusEntry> = entries.iter().collect();
        let out = render_user_message("karpathy", &refs);
        assert!(out.contains("@karpathy"), "got: {out}");
        assert!(out.contains("Now produce the JSON object."), "got: {out}");
    }

    #[test]
    fn render_user_message_renders_engagement_when_present() {
        let entries = [entry("1", "hot take", Some(eng(1234, 87, 12)), None)];
        let refs: Vec<&CorpusEntry> = entries.iter().collect();
        let out = render_user_message("k", &refs);
        assert!(out.contains("1234 likes"), "got: {out}");
        assert!(out.contains("87 reposts"), "got: {out}");
        assert!(out.contains("12 replies"), "got: {out}");
        assert!(out.contains("hot take"), "got: {out}");
    }

    #[test]
    fn render_user_message_marks_engagementless_entries() {
        let entries = [
            entry("1", "with eng", Some(eng(5, 0, 0)), None),
            entry("2", "without eng", None, None),
        ];
        let refs: Vec<&CorpusEntry> = entries.iter().collect();
        let out = render_user_message("k", &refs);
        assert!(out.contains("POST 1 (5 likes"), "got: {out}");
        assert!(out.contains("POST 2 (no engagement data)"), "got: {out}");
    }

    // ---- default_system_prompt ------------------------------------------

    #[test]
    fn default_system_prompt_contains_load_bearing_vocabulary() {
        let p = default_system_prompt();
        // A non-empty smoke check of the vocabulary the LLM will need.
        assert!(!p.is_empty());
        assert!(p.contains("rare_punchline_only"));
        assert!(p.contains("punchline_callbacks"));
        assert!(p.contains("sentence_length_distribution"));
        assert!(p.contains("OUTPUT THE JSON OBJECT ONLY"));
    }

    use heartbit_core::llm::types::{CompletionResponse, StopReason, TokenUsage};
    use std::future::Future;
    use std::sync::Mutex;

    /// Hand-rolled mock LLM provider for tests. Returns canned text (or
    /// error) on the first call and captures the most-recent request for
    /// assertions. `heartbit_core::Error` is not `Clone`, so the response
    /// is held as a single-shot `Option<Result<...>>` that gets `take()`n.
    struct MockProvider {
        response: Mutex<Option<Result<String, heartbit_core::Error>>>,
        captured_request: Mutex<Option<CompletionRequest>>,
        delay: Option<Duration>,
    }

    impl MockProvider {
        fn ok(text: impl Into<String>) -> Arc<BoxedProvider> {
            let p = MockProvider {
                response: Mutex::new(Some(Ok(text.into()))),
                captured_request: Mutex::new(None),
                delay: None,
            };
            Arc::new(BoxedProvider::new(p))
        }

        fn ok_with_delay(text: impl Into<String>, delay: Duration) -> Arc<BoxedProvider> {
            let p = MockProvider {
                response: Mutex::new(Some(Ok(text.into()))),
                captured_request: Mutex::new(None),
                delay: Some(delay),
            };
            Arc::new(BoxedProvider::new(p))
        }

        fn err(e: heartbit_core::Error) -> Arc<BoxedProvider> {
            let p = MockProvider {
                response: Mutex::new(Some(Err(e))),
                captured_request: Mutex::new(None),
                delay: None,
            };
            Arc::new(BoxedProvider::new(p))
        }

        fn empty_content() -> Arc<BoxedProvider> {
            // Sentinel value: the provider's complete() returns an empty
            // content vec when the response payload is the marker string.
            let p = MockProvider {
                response: Mutex::new(Some(Ok("__EMPTY_CONTENT__".to_string()))),
                captured_request: Mutex::new(None),
                delay: None,
            };
            Arc::new(BoxedProvider::new(p))
        }
    }

    impl LlmProvider for MockProvider {
        fn complete(
            &self,
            request: CompletionRequest,
        ) -> impl Future<Output = Result<CompletionResponse, heartbit_core::Error>> + Send {
            // Capture before any await so assertions can read it.
            *self.captured_request.lock().unwrap() = Some(request);
            let response = self
                .response
                .lock()
                .unwrap()
                .take()
                .unwrap_or_else(|| Err(heartbit_core::Error::Agent("mock used twice".into())));
            let delay = self.delay;
            async move {
                if let Some(d) = delay {
                    tokio::time::sleep(d).await;
                }
                let text = response?;
                let content = if text == "__EMPTY_CONTENT__" {
                    Vec::new()
                } else {
                    vec![ContentBlock::Text { text }]
                };
                Ok(CompletionResponse {
                    content,
                    usage: TokenUsage::default(),
                    stop_reason: StopReason::EndTurn,
                    model: None,
                    reasoning: None,
                })
            }
        }
    }

    /// A valid StyleProfile-shaped JSON string used by happy-path tests.
    const VALID_PROFILE_JSON: &str = r#"{
        "version": 1,
        "sentence_length_target": "short",
        "sentence_length_distribution": [40, 30, 20, 10],
        "fragment_frequency": "common",
        "opening_patterns": ["claim_first", "number_first"],
        "opening_pattern_weights": [0.6, 0.4],
        "formatting": {
            "lowercase": true,
            "periods": "optional",
            "em_dashes": "forbidden",
            "quotation_marks": "double",
            "line_breaks": "single"
        },
        "emoji_policy": "rare_punchline_only",
        "hashtag_policy": "never",
        "specificity_target": "high",
        "voice_traits": ["specific", "no_hedging"],
        "ai_tells_to_avoid": ["delve", "in conclusion"],
        "thread_rhythm": "punchline_callbacks",
        "thread_max_length": 10,
        "thread_opener_must_hook": true,
        "topical_obsessions": ["AI", "engineering"],
        "topical_avoidances": ["politics"]
    }"#;

    fn corpus_with_one_entry(writer: &str) -> Corpus {
        // Build a Corpus by appending JSONL through the public API. We
        // pre-create a TempDir, import a one-line JSONL, and leak the dir
        // so it survives the function return. Reads only after this point.
        let dir = tempfile::TempDir::new().unwrap();
        let mut c = Corpus::open_or_create(dir.path(), writer).unwrap();
        let src = dir.path().join("seed.jsonl");
        std::fs::write(
            &src,
            r#"{"id":"1","post_text":"the bitter lesson keeps winning"}"#,
        )
        .unwrap();
        c.append_from_jsonl(&src).unwrap();
        std::mem::forget(dir);
        c
    }

    fn empty_corpus(writer: &str) -> Corpus {
        let dir = tempfile::TempDir::new().unwrap();
        let c = Corpus::open_or_create(dir.path(), writer).unwrap();
        std::mem::forget(dir);
        c
    }

    // ---- Builder defaults ------------------------------------------------

    #[test]
    fn builder_defaults_match_documented_values() {
        let provider = MockProvider::ok("");
        let extractor = StyleExtractor::builder(provider).build();
        assert_eq!(extractor.sample_size, 50);
        assert_eq!(extractor.max_response_tokens, 2048);
        assert_eq!(extractor.timeout, Duration::from_secs(60));
        assert_eq!(extractor.system_prompt, default_system_prompt());
    }

    // ---- Happy path ------------------------------------------------------

    #[tokio::test]
    async fn extract_returns_validated_profile_on_valid_json() {
        let provider = MockProvider::ok(VALID_PROFILE_JSON);
        let extractor = StyleExtractor::builder(provider).build();
        let corpus = corpus_with_one_entry("karpathy");
        let profile = extractor.extract(&corpus).await.expect("extract ok");
        assert_eq!(profile.version, 1);
        assert_eq!(profile.thread_max_length, 10);
        assert!(profile.formatting.lowercase);
    }

    #[tokio::test]
    async fn extract_strips_markdown_fences_around_json() {
        let fenced = format!("```json\n{VALID_PROFILE_JSON}\n```");
        let provider = MockProvider::ok(fenced);
        let extractor = StyleExtractor::builder(provider).build();
        let corpus = corpus_with_one_entry("karpathy");
        let profile = extractor.extract(&corpus).await.expect("extract ok");
        assert_eq!(profile.version, 1);
    }

    // ---- Error paths -----------------------------------------------------

    #[tokio::test]
    async fn extract_empty_corpus_returns_empty_corpus_error() {
        let provider = MockProvider::ok("unused");
        let extractor = StyleExtractor::builder(provider).build();
        let corpus = empty_corpus("nobody");
        let err = extractor.extract(&corpus).await.unwrap_err();
        match err {
            ExtractError::EmptyCorpus(w) => assert_eq!(w, "nobody"),
            other => panic!("expected EmptyCorpus, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn extract_propagates_llm_provider_error() {
        let provider = MockProvider::err(heartbit_core::Error::Agent("boom".to_string()));
        let extractor = StyleExtractor::builder(provider).build();
        let corpus = corpus_with_one_entry("karpathy");
        let err = extractor.extract(&corpus).await.unwrap_err();
        match err {
            ExtractError::Llm(_) => { /* correct path */ }
            other => panic!("expected Llm, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn extract_returns_timeout_when_provider_exceeds_budget() {
        let provider = MockProvider::ok_with_delay(VALID_PROFILE_JSON, Duration::from_millis(200));
        let extractor = StyleExtractor::builder(provider)
            .timeout(Duration::from_millis(50))
            .build();
        let corpus = corpus_with_one_entry("karpathy");
        let err = extractor.extract(&corpus).await.unwrap_err();
        match err {
            ExtractError::Timeout(d) => assert_eq!(d, Duration::from_millis(50)),
            other => panic!("expected Timeout, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn extract_returns_empty_response_when_no_text_blocks() {
        let provider = MockProvider::empty_content();
        let extractor = StyleExtractor::builder(provider).build();
        let corpus = corpus_with_one_entry("karpathy");
        let err = extractor.extract(&corpus).await.unwrap_err();
        assert!(matches!(err, ExtractError::EmptyResponse), "got: {err:?}");
    }

    #[tokio::test]
    async fn extract_invalid_json_returns_jsonparse_with_raw() {
        let provider = MockProvider::ok("definitely not json");
        let extractor = StyleExtractor::builder(provider).build();
        let corpus = corpus_with_one_entry("karpathy");
        let err = extractor.extract(&corpus).await.unwrap_err();
        match err {
            ExtractError::JsonParse { raw, .. } => {
                assert_eq!(raw, "definitely not json");
            }
            other => panic!("expected JsonParse, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn extract_invalid_profile_returns_validation_with_raw() {
        // Distribution sums to 160 (40*4) — fails StyleProfile::validate.
        let bad = VALID_PROFILE_JSON.replace("[40, 30, 20, 10]", "[40, 40, 40, 40]");
        let provider = MockProvider::ok(bad.clone());
        let extractor = StyleExtractor::builder(provider).build();
        let corpus = corpus_with_one_entry("karpathy");
        let err = extractor.extract(&corpus).await.unwrap_err();
        match err {
            ExtractError::Validation { inner, raw } => {
                let msg = format!("{inner}");
                assert!(msg.contains("sentence_length_distribution"), "msg: {msg}");
                assert!(msg.contains("100"), "msg: {msg}");
                // Raw is the trimmed/unfenced output. Substring is sufficient.
                assert!(raw.contains("[40, 40, 40, 40]"), "raw: {raw}");
            }
            other => panic!("expected Validation, got {other:?}"),
        }
    }

    // ---- Provider sees the right request --------------------------------

    /// Wraps a shared MockProvider so the test can hold one Arc for
    /// inspection while the extractor holds a separate Arc<BoxedProvider>
    /// over an LlmProvider impl that delegates to the inspector.
    struct InspectorWrapper(Arc<MockProvider>);

    impl LlmProvider for InspectorWrapper {
        fn complete(
            &self,
            request: CompletionRequest,
        ) -> impl Future<Output = Result<CompletionResponse, heartbit_core::Error>> + Send {
            let inner = self.0.clone();
            async move { LlmProvider::complete(inner.as_ref(), request).await }
        }
    }

    #[tokio::test]
    async fn extract_passes_system_prompt_and_user_message_to_provider() {
        let inspector = Arc::new(MockProvider {
            response: Mutex::new(Some(Ok(VALID_PROFILE_JSON.to_string()))),
            captured_request: Mutex::new(None),
            delay: None,
        });
        let provider: Arc<BoxedProvider> =
            Arc::new(BoxedProvider::new(InspectorWrapper(inspector.clone())));
        let extractor = StyleExtractor::builder(provider)
            .max_response_tokens(1500)
            .build();
        let corpus = corpus_with_one_entry("karpathy");
        let _ = extractor.extract(&corpus).await.unwrap();

        let captured = inspector.captured_request.lock().unwrap().clone();
        let req = captured.expect("provider was called");
        assert_eq!(req.system, default_system_prompt());
        assert_eq!(req.max_tokens, 1500);
        assert_eq!(req.messages.len(), 1);
        // Message::user(content) produces a user role message with text content.
        let user_text = format!("{:?}", req.messages[0]);
        assert!(user_text.contains("@karpathy"), "got: {user_text}");
    }
}
