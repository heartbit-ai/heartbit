# heartbit-ghost P1.2c — LLM style extractor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Single-shot LLM call that turns a `Corpus` into a validated `StyleProfile` (analyst-persona prompt, JSON output, top-K sampling).

**Architecture:** New file `crates/heartbit-ghost/src/voice/extractor.rs` adds `StyleExtractor` + builder + `ExtractError` + a static `default_system_prompt` + pure helpers (`select_top_k`, `render_user_message`). Provider is injected as `Arc<BoxedProvider>` (mirrors `LlmJudgeGuardrail`). Tests use a hand-rolled `MockProvider` (~25 lines).

**Tech Stack:** Rust 2024, `serde_json`, `thiserror`, `tokio::time::timeout`, `chrono` (already a workspace dep), `heartbit_core::llm::{BoxedProvider, LlmProvider, CompletionRequest, CompletionResponse, ContentBlock, Message}`.

---

## File structure

| File | Responsibility |
|------|----------------|
| `crates/heartbit-ghost/src/voice/extractor.rs` | All P1.2c additions: `ExtractError`, `default_system_prompt`, `select_top_k`, `render_user_message`, `StyleExtractor`, `StyleExtractorBuilder`, all 23 tests, `MockProvider` test helper |
| `crates/heartbit-ghost/src/voice/mod.rs` | Add `pub mod extractor;` and re-exports |

4 tasks total: 3 implementation + 1 final acceptance (verification only, no commit).

---

## Task 1: Module scaffolding + `ExtractError`

**Why:** Establish the module + error type so subsequent tasks can build on it. Mirrors P1.2b Task 1's scaffolding pattern.

**Files:**
- Create: `crates/heartbit-ghost/src/voice/extractor.rs`
- Modify: `crates/heartbit-ghost/src/voice/mod.rs` (add `pub mod extractor;` + `pub use extractor::ExtractError;`)

- [ ] **Step 1: Create `crates/heartbit-ghost/src/voice/extractor.rs`**

```rust
//! LLM-based style extractor — turns a [`Corpus`] into a validated
//! [`StyleProfile`] via a single analyst-persona LLM call (umbrella spec
//! §2.3). Sibling of [`crate::voice::style`] (the schema) and
//! [`crate::corpus::Corpus`] (the input).
//!
//! Bodies for [`StyleExtractor`], [`StyleExtractorBuilder`],
//! [`default_system_prompt`], and the pure helpers land in subsequent
//! tasks.

use thiserror::Error;

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
}
```

- [ ] **Step 2: Modify `crates/heartbit-ghost/src/voice/mod.rs`**

The current state (after P1.2a) has:

```rust
pub mod blend;
pub mod error;
pub mod style;

pub use blend::{BlendEntry, BlendRecipe, PartialStyleProfile};
pub use error::VoiceError;
pub use style::{
    EmDashPolicy, EmojiPolicy, Formatting, FragmentFrequency, HashtagPolicy, LineBreaks,
    OpeningPattern, PeriodsPolicy, QuotationMarks, SentenceLengthTarget, SpecificityTarget,
    StyleProfile, ThreadRhythm,
};
```

Add `pub mod extractor;` (alphabetical — between `error` and `style`) and a `pub use extractor::ExtractError;` line. Final state:

```rust
pub mod blend;
pub mod error;
pub mod extractor;
pub mod style;

pub use blend::{BlendEntry, BlendRecipe, PartialStyleProfile};
pub use error::VoiceError;
pub use extractor::ExtractError;
pub use style::{
    EmDashPolicy, EmojiPolicy, Formatting, FragmentFrequency, HashtagPolicy, LineBreaks,
    OpeningPattern, PeriodsPolicy, QuotationMarks, SentenceLengthTarget, SpecificityTarget,
    StyleProfile, ThreadRhythm,
};
```

(Tasks 2/3 will extend the `pub use extractor::...` line with more names. Don't add them now — keep the diff minimal so each task's intent is clear.)

- [ ] **Step 3: Run the tests**

```bash
cargo test -p heartbit-ghost --lib voice::extractor
```

Expected: `3 passed; 0 failed; 0 ignored`.

- [ ] **Step 4: Workspace quality gate**

```bash
cargo fmt -p heartbit-ghost -- --check
cargo clippy -p heartbit-ghost -- -D warnings
```

Both clean.

- [ ] **Step 5: Commit**

```bash
git add crates/heartbit-ghost/src/voice/extractor.rs crates/heartbit-ghost/src/voice/mod.rs
git commit -m "$(cat <<'EOF'
feat(ghost): voice — extractor module scaffolding + ExtractError (P1.2c)

Stub module that compiles, with the error enum its later additions
(Tasks 2 + 3) will return. 6 variants cover every failure path
documented in the spec: EmptyCorpus, Llm (wraps heartbit_core::Error),
Timeout, EmptyResponse, JsonParse + Validation (both carry raw LLM
output for debuggability during prompt iteration).

3 tests on ExtractError: writer-handle in EmptyCorpus, source-message
in JsonParse, raw + inner reachable on Validation.

Refs: docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.2c-llm-style-extractor-design.md §4
EOF
)"
```

---

## Task 2: `default_system_prompt` + `select_top_k` + `render_user_message`

**Why:** All pure logic — no LLM, no async. Lands as 3 free functions plus 10 tests. Locking these down before the orchestration layer (Task 3) means the test suite for `extract` only has to verify wiring, not pure-function behavior.

**Files:**
- Modify: `crates/heartbit-ghost/src/voice/extractor.rs` (append the 3 functions + 10 tests; the existing `ExtractError` block + 3 tests stay)

- [ ] **Step 1: Append imports + `DEFAULT_SYSTEM_PROMPT` constant + `default_system_prompt()` accessor to `voice/extractor.rs`**

Add to the imports block at the top (after the existing `use thiserror::Error;` and `use crate::voice::error::VoiceError;`):

```rust
use crate::corpus::CorpusEntry;
```

Then append (above the `#[cfg(test)] mod tests` block):

```rust
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
```

- [ ] **Step 2: Append `select_top_k` to `voice/extractor.rs` (above tests block)**

```rust
/// Sort `entries` by descending engagement (likes, then reposts, then
/// replies, then `posted_at`), and return references to the top `k`.
///
/// Pure function — no I/O, deterministic. Engagement-less entries sort
/// to the bottom (treated as zero engagement).
pub(crate) fn select_top_k<'a>(
    entries: &'a [CorpusEntry],
    k: usize,
) -> Vec<&'a CorpusEntry> {
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
```

- [ ] **Step 3: Append `render_user_message` to `voice/extractor.rs` (above tests block)**

```rust
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
```

- [ ] **Step 4: Add the 10 pure-logic tests to the existing `#[cfg(test)] mod tests` block**

Inside the existing tests mod (after the 3 ExtractError tests), append:

```rust
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
            entry("old", "a", Some(eng(10, 0, 0)), Some("2024-01-01T00:00:00Z")),
            entry("new", "b", Some(eng(10, 0, 0)), Some("2025-01-01T00:00:00Z")),
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
        let entries = vec![entry("1", "hello", None, None)];
        let refs: Vec<&CorpusEntry> = entries.iter().collect();
        let out = render_user_message("karpathy", &refs);
        assert!(out.contains("@karpathy"), "got: {out}");
        assert!(out.contains("Now produce the JSON object."), "got: {out}");
    }

    #[test]
    fn render_user_message_renders_engagement_when_present() {
        let entries = vec![entry(
            "1",
            "hot take",
            Some(eng(1234, 87, 12)),
            None,
        )];
        let refs: Vec<&CorpusEntry> = entries.iter().collect();
        let out = render_user_message("k", &refs);
        assert!(out.contains("1234 likes"), "got: {out}");
        assert!(out.contains("87 reposts"), "got: {out}");
        assert!(out.contains("12 replies"), "got: {out}");
        assert!(out.contains("hot take"), "got: {out}");
    }

    #[test]
    fn render_user_message_marks_engagementless_entries() {
        let entries = vec![
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
```

- [ ] **Step 5: Run the tests**

```bash
cargo test -p heartbit-ghost --lib voice::extractor
```

Expected: `13 passed` (3 from Task 1 + 10 new).

- [ ] **Step 6: Workspace quality gate**

```bash
cargo fmt -p heartbit-ghost -- --check
cargo clippy -p heartbit-ghost --all-targets -- -D warnings
```

Both clean.

- [ ] **Step 7: Commit**

```bash
git add crates/heartbit-ghost/src/voice/extractor.rs
git commit -m "$(cat <<'EOF'
feat(ghost): voice — extractor pure logic (default prompt, sampling, render) (P1.2c)

Three free helpers that together drive Task 3's extract():

- default_system_prompt(): static analyst-persona system prompt with
  closed-vocab enums embedded inline. Hand-tuned, ~1.5k tokens.
- select_top_k(entries, k): pure deterministic sort by engagement
  (likes desc, then reposts, then replies, then posted_at desc),
  truncate to k. Engagement-less entries sort to the bottom.
- render_user_message(writer, samples): builds the user-message text
  (writer header + numbered POST blocks with engagement when present
  + closing instruction).

10 new tests: 6 select_top_k (empty, smaller-than-k, likes-desc,
reposts-replies tiebreak, posted_at tiebreak, engagementless-bottom),
3 render (writer handle, engagement render, engagementless mark),
1 system prompt vocabulary smoke check.

Refs: docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.2c-llm-style-extractor-design.md §5 §6
EOF
)"
```

---

## Task 3: `StyleExtractor` + `StyleExtractorBuilder` + `extract()`

**Why:** The orchestration layer. Brings everything together: builder defaults, the LLM call wrapped in `tokio::time::timeout`, error mapping, defensive markdown-fence stripping, JSON parse + validate. The largest task.

**Files:**
- Modify: `crates/heartbit-ghost/src/voice/extractor.rs` (append `StyleExtractor`, `StyleExtractorBuilder`, `strip_markdown_fences` private helper, `MockProvider` test helper, 10 tests)
- Modify: `crates/heartbit-ghost/src/voice/mod.rs` (extend the `pub use extractor::...` line with the new names)

- [ ] **Step 1: Update imports at the top of `voice/extractor.rs`**

Add to the existing imports block:

```rust
use std::sync::Arc;
use std::time::Duration;

use heartbit_core::llm::{
    BoxedProvider, CompletionRequest, ContentBlock, LlmProvider, types::Message,
};

use crate::corpus::Corpus;
use crate::voice::style::StyleProfile;
```

(The exact `types::Message` path mirrors `agent/guardrails/llm_judge.rs:14`. If the local re-export form `use heartbit_core::llm::types::Message;` is preferred, that also works — they resolve to the same type. Verify with `cargo check` after editing.)

- [ ] **Step 2: Append `StyleExtractor`, `StyleExtractorBuilder`, and `strip_markdown_fences` to `voice/extractor.rs` (above the `#[cfg(test)] mod tests` block)**

```rust
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
            .field("has_custom_system_prompt", &self.custom_system_prompt.is_some())
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
```

- [ ] **Step 3: Append the `MockProvider` test helper + 10 tests to the `#[cfg(test)] mod tests` block**

Append (after the existing 13 tests from Tasks 1+2):

```rust
    use heartbit_core::llm::types::{
        CompletionResponse, StopReason, TokenUsage, Tool, ToolChoice,
    };
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Mutex;

    /// Hand-rolled mock LLM provider for tests. Returns canned text and
    /// captures the most-recent request for assertions.
    struct MockProvider {
        response: Mutex<Result<String, heartbit_core::Error>>,
        captured_request: Mutex<Option<CompletionRequest>>,
        delay: Option<Duration>,
    }

    impl MockProvider {
        fn ok(text: impl Into<String>) -> Arc<BoxedProvider> {
            let p = MockProvider {
                response: Mutex::new(Ok(text.into())),
                captured_request: Mutex::new(None),
                delay: None,
            };
            Arc::new(BoxedProvider::new(p))
        }

        fn ok_with_delay(text: impl Into<String>, delay: Duration) -> Arc<BoxedProvider> {
            let p = MockProvider {
                response: Mutex::new(Ok(text.into())),
                captured_request: Mutex::new(None),
                delay: Some(delay),
            };
            Arc::new(BoxedProvider::new(p))
        }

        fn err(e: heartbit_core::Error) -> Arc<BoxedProvider> {
            let p = MockProvider {
                response: Mutex::new(Err(e)),
                captured_request: Mutex::new(None),
                delay: None,
            };
            Arc::new(BoxedProvider::new(p))
        }

        fn empty_content() -> Arc<BoxedProvider> {
            // Sentinel value: the responder hands back a CompletionResponse
            // with an empty content vec. Achieved by returning Err for the
            // text and special-casing in complete() — but simpler: return
            // a marker string and the provider produces the empty response.
            // We cheat: use a unique marker the impl recognizes.
            let p = MockProvider {
                response: Mutex::new(Ok("__EMPTY_CONTENT__".to_string())),
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
            let response = self.response.lock().unwrap().clone();
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
        // Use a TempDir-free path: build a Corpus by appending JSONL through
        // the public API. We pre-create a small temp dir on a shared
        // tempfile and import a one-line JSONL.
        use crate::corpus::Corpus;
        let dir = tempfile::TempDir::new().unwrap();
        let mut c = Corpus::open_or_create(dir.path(), writer).unwrap();
        let src = dir.path().join("seed.jsonl");
        std::fs::write(&src, r#"{"id":"1","post_text":"the bitter lesson keeps winning"}"#).unwrap();
        c.append_from_jsonl(&src).unwrap();
        // Drop dir at the end of this fn — but Corpus retains the loaded
        // entries in memory, so the dropped dir doesn't matter for the
        // tests below. (We avoid persisting more, so save() doesn't run.)
        // NOTE: don't mutate the corpus further in tests; reads only.
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
        let bad = VALID_PROFILE_JSON
            .replace("[40, 30, 20, 10]", "[40, 40, 40, 40]");
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

    #[tokio::test]
    async fn extract_passes_system_prompt_and_user_message_to_provider() {
        let provider_arc = MockProvider::ok(VALID_PROFILE_JSON);
        // We need a handle to the underlying MockProvider to read
        // captured_request after the call. The Arc<BoxedProvider> hides
        // the concrete type, so we instead build a fresh MockProvider and
        // a parallel reference for inspection.
        let inspector = Arc::new(MockProvider {
            response: Mutex::new(Ok(VALID_PROFILE_JSON.to_string())),
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
        // The unused arc keeps the type checker happy if we ever switch.
        drop(provider_arc);
    }

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
```

Notes for the implementer:

- `Message::user(content)` is the constructor (`crates/heartbit-core/src/llm/types.rs:62`). Verified.
- `BoxedProvider::new(provider)` wraps any `LlmProvider` (`crates/heartbit-core/src/llm/mod.rs:254`). Verified.
- `corpus_with_one_entry` uses `std::mem::forget(dir)` to leak the `TempDir` so the directory survives the function return. The `Corpus` has loaded all entries into memory at `open_or_create`, so the on-disk dir doesn't matter for read-only tests. The leaked tempdirs are cleaned up at process exit; for ~5 forgotten tempdirs across the whole suite this is fine.

- [ ] **Step 4: Update `crates/heartbit-ghost/src/voice/mod.rs` to extend the extractor re-exports**

Find the existing line:

```rust
pub use extractor::ExtractError;
```

Replace with:

```rust
pub use extractor::{
    ExtractError, StyleExtractor, StyleExtractorBuilder, default_system_prompt,
};
```

(Alphabetical inside the braces — rustfmt may re-sort by case-insensitive order, in which case it becomes `default_system_prompt, ExtractError, StyleExtractor, StyleExtractorBuilder`. Let rustfmt win; don't fight it.)

- [ ] **Step 5: Run the tests**

```bash
cargo test -p heartbit-ghost --lib voice::extractor
```

Expected: `23 passed` (13 from prior tasks + 10 new).

- [ ] **Step 6: Workspace quality gate**

```bash
cargo fmt -p heartbit-ghost -- --check
cargo clippy -p heartbit-ghost --all-targets -- -D warnings
```

Both clean.

- [ ] **Step 7: Commit**

```bash
git add crates/heartbit-ghost/src/voice/extractor.rs crates/heartbit-ghost/src/voice/mod.rs
git commit -m "$(cat <<'EOF'
feat(ghost): voice — StyleExtractor + builder + extract() (P1.2c)

The orchestration layer. extract() is async, single-shot:

- Empty corpus → ExtractError::EmptyCorpus (no provider call)
- select_top_k + render_user_message → CompletionRequest
- tokio::time::timeout wraps provider.complete()
- Markdown fence pair stripped defensively (LLM-ignored-instructions
  is the most common failure mode despite the prompt saying not to)
- serde_json::from_str + StyleProfile::validate; both error variants
  carry the raw LLM output for caller-side debugging

Builder: sample_size (50), max_response_tokens (2048), timeout (60s),
system_prompt (default_system_prompt()).

10 new tests + a hand-rolled MockProvider helper (~40 LOC). Coverage:
builder defaults, happy path, fence-stripped happy path, all 6 error
paths (empty corpus, llm error, timeout, empty response, json parse,
validation), and request-shape verification.

Refs: docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.2c-llm-style-extractor-design.md §3 §7
EOF
)"
```

---

## Task 4: Final acceptance + workspace quality gate

**Why:** Confirm P1.2c meets every acceptance criterion in the spec. Verification only — no commit.

**Files:** none.

- [ ] **Step 1: Workspace quality gate**

```bash
cd /home/pleclech/projects/heartbit
cargo fmt -- --check && cargo clippy --workspace --all-features -- -D warnings && cargo test --workspace --all-features 2>&1 | tail -10
```

Expected: all green. Workspace test count goes from 3849 (post-P1.2b baseline) to ~3872 (~23 new extractor tests: 3 ExtractError display + 6 select_top_k + 3 render + 1 prompt sanity + 1 builder + 2 happy + 6 error paths + 1 request shape).

- [ ] **Step 2: Verify the public surface is reachable**

```bash
cat <<'EOF' > /tmp/heartbit_ghost_extractor_surface_check.rs
fn _check() {
    use heartbit_ghost::voice::{
        ExtractError, StyleExtractor, StyleExtractorBuilder, default_system_prompt,
    };
    let _: fn() -> &'static str = default_system_prompt;
    let _ = ExtractError::EmptyCorpus(String::new());
    // StyleExtractor::builder requires a provider — type-check only.
    let _: fn(_) -> StyleExtractorBuilder = StyleExtractor::builder;
}
EOF
echo "(Surface check is illustrative; the public types are reachable via the workspace cargo check above.)"
rm -f /tmp/heartbit_ghost_extractor_surface_check.rs
```

- [ ] **Step 3: Print final commit log for the branch**

```bash
git log --oneline main..feat/heartbit-ghost-p1.2c
```

Expected: 4 commits — spec doc + 3 task commits (Task 1, 2, 3). No commit for Task 4.

- [ ] **Step 4: No commit for this task**

Task 4 is verification only. The branch is ready for final review + merge.

---

## Acceptance criteria

P1.2c is done when (per spec §10):

- All public types compile cleanly under `cargo check -p heartbit-ghost --all-features`
- `cargo fmt -- --check && cargo clippy --workspace --all-features -- -D warnings && cargo test --workspace --all-features` all green
- ~19 extractor tests pass (actual count expected 23 once all three implementation tasks land — the spec's "~19" was the functional-test count; the per-task breakdown adds 3 ExtractError display tests + 1 prompt sanity test on top, totaling 23)
- `heartbit_ghost::voice::{StyleExtractor, StyleExtractorBuilder, ExtractError, default_system_prompt}` are reachable as public surface
- A test verifies the extractor passes the configured system prompt + the rendered user message + the configured `max_response_tokens` to the provider (covered by `extract_passes_system_prompt_and_user_message_to_provider` in Task 3)

## Out of scope (re-stated)

- Blend algorithm — merging N profiles into one (P1.2d)
- CLI bodies for `profile rebuild` and `profile diff` (P1.2e)
- Runtime conditioning of the writer agent (P1.4)
- Embedding generation (P1.4)
- TOML serialization to disk (P1.2e)
- Auto-retry on parse/validate failure (caller can layer this trivially using the carried `raw` field)
- Few-shot prompt examples (P1.4 if real-world failure rates demand it)
- Live LLM integration tests (mock-only suite; live tests live at the call site)

## Reference

- Spec: `docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.2c-llm-style-extractor-design.md`
- Umbrella heartbit-ghost spec §2.3: `docs/superpowers/specs/2026-05-07-heartbit-ghost-x-agent-design.md`
- LlmJudgeGuardrail (single-shot pattern reference): `crates/heartbit-core/src/agent/guardrails/llm_judge.rs`
- LlmProvider trait + types: `crates/heartbit-core/src/llm/{mod,types}.rs`
- P1.2a (style profile schema): `docs/superpowers/specs/2026-05-07-heartbit-ghost-p1.2a-style-profile-schema-design.md`
- P1.2b (corpus storage): `docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.2b-corpus-storage-design.md`
