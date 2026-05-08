//! LLM-based style extractor — turns a [`Corpus`] into a validated
//! [`StyleProfile`] via a single analyst-persona LLM call (umbrella spec
//! §2.3). Sibling of [`crate::voice::style`] (the schema) and
//! [`crate::corpus::Corpus`] (the input).
//!
//! Bodies for [`StyleExtractor`], [`StyleExtractorBuilder`],
//! [`default_system_prompt`], and the pure helpers land in subsequent
//! tasks.

use thiserror::Error;

use crate::corpus::CorpusEntry;
use crate::voice::error::VoiceError;

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
#[allow(dead_code)] // Used by StyleExtractor::extract (Task 3); silences lib-only build.
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
#[allow(dead_code)] // Used by StyleExtractor::extract (Task 3); silences lib-only build.
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
}
