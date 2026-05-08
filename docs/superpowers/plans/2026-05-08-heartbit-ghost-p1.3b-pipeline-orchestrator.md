# heartbit-ghost P1.3b — pipeline orchestrator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the 4 P1.3a recipes used in the single-candidate path (researcher, writer, style_critic, fact_check) into a working pipeline. `heartbit persona run <name> --once "<topic>"` prints the final draft to stdout, with progress to stderr.

**Architecture:** New `crates/heartbit-ghost/src/pipeline/` module (5 files: `mod.rs`, `style_render.rs`, `verdicts.rs`, `publish_gate.rs`, `prompts.rs`). Manual orchestration via direct `AgentRunner::execute()` calls — `SequentialAgent` / `LoopAgent` from heartbit-core don't fit the per-stage input construction shape. Style profile rendered as English in the writer's user message at runtime (closes the gap deferred from P1.3a).

**Tech Stack:** Rust 2024, `serde_json` for verdict parsing, `tokio` for async, `thiserror` for errors, `Arc<BoxedProvider>` from P1.2c, `SnapshotStore::load_latest` from P1.2e. No new workspace deps.

---

## File structure

| File | Responsibility |
|------|----------------|
| `crates/heartbit-ghost/src/pipeline/mod.rs` | NEW — `PipelineConfig`, `PipelineOutput`, `PipelineError`, `run_pipeline`, private `runner_from_recipe` helper, `MockProvider` test helper |
| `crates/heartbit-ghost/src/pipeline/style_render.rs` | NEW — `render_style_profile_as_english(&StyleProfile) -> String` + 4 tests |
| `crates/heartbit-ghost/src/pipeline/verdicts.rs` | NEW — `StyleVerdict` + `FactVerdict` + `parse_critic_verdict` + `parse_fact_verdict` + 8 tests |
| `crates/heartbit-ghost/src/pipeline/publish_gate.rs` | NEW — `PublishGateError` + `check_publish_gate` + 6 tests |
| `crates/heartbit-ghost/src/pipeline/prompts.rs` | NEW — `build_writer_user_message` + `build_critic_user_message` + `build_fact_user_message` (tested indirectly via integration tests) |
| `crates/heartbit-ghost/src/lib.rs` | MODIFY — add `pub mod pipeline;` |
| `crates/heartbit-cli/src/persona.rs` | MODIFY — wire `PersonaCommand::Run { name, once }` body + 2 dispatch tests |

4 implementation tasks + 1 final acceptance.

---

## Task 1: Pure helpers (style_render + verdicts + publish_gate + prompts)

**Why:** All 4 helpers are pure data transformations (no I/O, no LLM calls). Implementing them together amortizes the module-scaffolding boilerplate. Each is independently testable; the integration in Task 3 just composes them.

**Files:**
- Create: `crates/heartbit-ghost/src/pipeline/{mod,style_render,verdicts,publish_gate,prompts}.rs`
- Modify: `crates/heartbit-ghost/src/lib.rs` (add `pub mod pipeline;`)

- [ ] **Step 1: Create `crates/heartbit-ghost/src/pipeline/mod.rs` (skeleton only — Task 2 fills in the orchestration types)**

```rust
//! Generation pipeline — wires the P1.3a sub-agent recipes into a working
//! single-candidate path.
//!
//! Public entry: [`run_pipeline`] (added in Task 3 / Task 2 wiring).

pub mod prompts;
pub mod publish_gate;
pub mod style_render;
pub mod verdicts;

pub use publish_gate::{PublishGateError, check_publish_gate};
pub use style_render::render_style_profile_as_english;
pub use verdicts::{FactVerdict, StyleVerdict, parse_critic_verdict, parse_fact_verdict};
```

(`run_pipeline`, `PipelineConfig`, `PipelineOutput`, `PipelineError` land in Task 2.)

- [ ] **Step 2: Create `crates/heartbit-ghost/src/pipeline/style_render.rs`**

```rust
//! Render a [`StyleProfile`] as English voice guidelines for the writer's
//! user message. All 16 non-version fields are surfaced; ~200 tokens.

use crate::voice::{
    EmDashPolicy, EmojiPolicy, Formatting, FragmentFrequency, HashtagPolicy, LineBreaks,
    OpeningPattern, PeriodsPolicy, QuotationMarks, SentenceLengthTarget, SpecificityTarget,
    StyleProfile, ThreadRhythm,
};

/// Render the profile as a structured-English voice-guidelines block.
pub fn render_style_profile_as_english(profile: &StyleProfile) -> String {
    let mut out = String::new();
    out.push_str("Voice guidelines:\n");

    let dist = &profile.sentence_length_distribution;
    out.push_str(&format!(
        "- sentence length: {} ({}% short, {}% medium-short, {}% medium-long, {}% long)\n",
        sentence_length_word(profile.sentence_length_target),
        dist[0], dist[1], dist[2], dist[3]
    ));
    out.push_str(&format!(
        "- fragments: {}\n",
        fragment_frequency_word(profile.fragment_frequency)
    ));

    let openers = profile
        .opening_patterns
        .iter()
        .zip(profile.opening_pattern_weights.iter())
        .map(|(p, w)| format!("{} ({}%)", opening_pattern_word(*p), (w * 100.0).round() as u32))
        .collect::<Vec<_>>()
        .join(", ");
    if openers.is_empty() {
        out.push_str("- opening patterns: (none)\n");
    } else {
        out.push_str(&format!("- opening patterns: {}\n", openers));
    }

    out.push_str(&format!("- formatting: {}\n", render_formatting(&profile.formatting)));
    out.push_str(&format!(
        "- emoji policy: {}\n",
        emoji_policy_word(profile.emoji_policy)
    ));
    out.push_str(&format!(
        "- hashtag policy: {}\n",
        hashtag_policy_word(profile.hashtag_policy)
    ));
    out.push_str(&format!(
        "- specificity target: {}\n",
        specificity_target_word(profile.specificity_target)
    ));
    out.push_str(&format!(
        "- voice traits: {}\n",
        render_string_list(&profile.voice_traits)
    ));
    out.push_str(&format!(
        "- ai tells to avoid: {}\n",
        render_string_list(&profile.ai_tells_to_avoid)
    ));
    out.push_str(&format!(
        "- thread rhythm: {}\n",
        thread_rhythm_word(profile.thread_rhythm)
    ));
    out.push_str(&format!(
        "- thread max length: {} ({})\n",
        profile.thread_max_length,
        if profile.thread_opener_must_hook {
            "opener must hook"
        } else {
            "opener need not hook"
        }
    ));
    out.push_str(&format!(
        "- topical obsessions: {}\n",
        render_string_list(&profile.topical_obsessions)
    ));
    out.push_str(&format!(
        "- topical avoidances: {}\n",
        render_string_list(&profile.topical_avoidances)
    ));

    out
}

fn render_string_list(v: &[String]) -> String {
    if v.is_empty() {
        "(none)".to_string()
    } else {
        v.join(", ")
    }
}

fn render_formatting(f: &Formatting) -> String {
    let mut parts = Vec::new();
    parts.push(if f.lowercase { "lowercase" } else { "sentence case" }.to_string());
    parts.push(format!("{} periods", periods_policy_word(f.periods)));
    parts.push(format!("em-dashes {}", em_dash_policy_word(f.em_dashes)));
    parts.push(format!("{} quotes", quotation_marks_word(f.quotation_marks)));
    parts.push(format!("{} line breaks", line_breaks_word(f.line_breaks)));
    parts.join(", ")
}

fn sentence_length_word(t: SentenceLengthTarget) -> &'static str {
    match t {
        SentenceLengthTarget::Short => "short",
        SentenceLengthTarget::Mixed => "mixed",
        SentenceLengthTarget::Long => "long",
        _ => "unknown",
    }
}

fn fragment_frequency_word(f: FragmentFrequency) -> &'static str {
    match f {
        FragmentFrequency::Rare => "rare",
        FragmentFrequency::Occasional => "occasional",
        FragmentFrequency::Common => "common",
        _ => "unknown",
    }
}

fn opening_pattern_word(p: OpeningPattern) -> &'static str {
    match p {
        OpeningPattern::ClaimFirst => "claim_first",
        OpeningPattern::NumberFirst => "number_first",
        OpeningPattern::SceneFirst => "scene_first",
        OpeningPattern::QuestionFirst => "question_first",
        OpeningPattern::AphoristicFirst => "aphoristic_first",
        OpeningPattern::AnecdoteFirst => "anecdote_first",
        OpeningPattern::ContrarianFirst => "contrarian_first",
        _ => "unknown",
    }
}

fn periods_policy_word(p: PeriodsPolicy) -> &'static str {
    match p {
        PeriodsPolicy::Always => "always",
        PeriodsPolicy::Optional => "optional",
        PeriodsPolicy::Rare => "rare",
        _ => "unknown",
    }
}

fn em_dash_policy_word(e: EmDashPolicy) -> &'static str {
    match e {
        EmDashPolicy::Preferred => "preferred",
        EmDashPolicy::Ok => "ok",
        EmDashPolicy::Forbidden => "forbidden",
        _ => "unknown",
    }
}

fn quotation_marks_word(q: QuotationMarks) -> &'static str {
    match q {
        QuotationMarks::Double => "double",
        QuotationMarks::Single => "single",
        QuotationMarks::Smart => "smart",
        _ => "unknown",
    }
}

fn line_breaks_word(l: LineBreaks) -> &'static str {
    match l {
        LineBreaks::Single => "single",
        LineBreaks::Double => "double",
        LineBreaks::Rhythmic => "rhythmic",
        _ => "unknown",
    }
}

fn emoji_policy_word(e: EmojiPolicy) -> &'static str {
    match e {
        EmojiPolicy::Never => "never",
        EmojiPolicy::RarePunchlineOnly => "rare punchline only",
        EmojiPolicy::Occasional => "occasional",
        EmojiPolicy::Frequent => "frequent",
        _ => "unknown",
    }
}

fn hashtag_policy_word(h: HashtagPolicy) -> &'static str {
    match h {
        HashtagPolicy::Never => "never",
        HashtagPolicy::Rare => "rare",
        HashtagPolicy::TopicRelevant => "topic-relevant",
        HashtagPolicy::Always => "always",
        _ => "unknown",
    }
}

fn specificity_target_word(s: SpecificityTarget) -> &'static str {
    match s {
        SpecificityTarget::Low => "low",
        SpecificityTarget::Medium => "medium",
        SpecificityTarget::High => "high",
        _ => "unknown",
    }
}

fn thread_rhythm_word(t: ThreadRhythm) -> &'static str {
    match t {
        ThreadRhythm::Linear => "linear",
        ThreadRhythm::ListThenPayoff => "list_then_payoff",
        ThreadRhythm::PunchlineCallbacks => "punchline_callbacks",
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voice::StyleProfile;

    fn canonical_profile() -> StyleProfile {
        StyleProfile {
            version: 1,
            sentence_length_target: SentenceLengthTarget::Short,
            sentence_length_distribution: [40, 30, 20, 10],
            fragment_frequency: FragmentFrequency::Common,
            opening_patterns: vec![OpeningPattern::ClaimFirst, OpeningPattern::NumberFirst],
            opening_pattern_weights: vec![0.6, 0.4],
            formatting: Formatting {
                lowercase: true,
                periods: PeriodsPolicy::Optional,
                em_dashes: EmDashPolicy::Forbidden,
                quotation_marks: QuotationMarks::Double,
                line_breaks: LineBreaks::Single,
            },
            emoji_policy: EmojiPolicy::RarePunchlineOnly,
            hashtag_policy: HashtagPolicy::Never,
            specificity_target: SpecificityTarget::High,
            voice_traits: vec!["specific".to_string(), "no_hedging".to_string()],
            ai_tells_to_avoid: vec!["delve".to_string(), "in conclusion".to_string()],
            thread_rhythm: ThreadRhythm::PunchlineCallbacks,
            thread_max_length: 10,
            thread_opener_must_hook: true,
            topical_obsessions: vec!["AI".to_string()],
            topical_avoidances: vec!["politics".to_string()],
        }
    }

    #[test]
    fn render_canonical_profile_includes_all_16_fields() {
        let p = canonical_profile();
        let s = render_style_profile_as_english(&p);
        assert!(s.contains("Voice guidelines:"));
        assert!(s.contains("sentence length: short"));
        assert!(s.contains("40% short, 30% medium-short, 20% medium-long, 10% long"));
        assert!(s.contains("fragments: common"));
        assert!(s.contains("opening patterns: claim_first (60%), number_first (40%)"));
        assert!(s.contains("lowercase"));
        assert!(s.contains("optional periods"));
        assert!(s.contains("em-dashes forbidden"));
        assert!(s.contains("double quotes"));
        assert!(s.contains("single line breaks"));
        assert!(s.contains("emoji policy: rare punchline only"));
        assert!(s.contains("hashtag policy: never"));
        assert!(s.contains("specificity target: high"));
        assert!(s.contains("voice traits: specific, no_hedging"));
        assert!(s.contains("ai tells to avoid: delve, in conclusion"));
        assert!(s.contains("thread rhythm: punchline_callbacks"));
        assert!(s.contains("thread max length: 10 (opener must hook)"));
        assert!(s.contains("topical obsessions: AI"));
        assert!(s.contains("topical avoidances: politics"));
    }

    #[test]
    fn render_empty_string_lists_show_none_marker() {
        let mut p = canonical_profile();
        p.voice_traits.clear();
        p.ai_tells_to_avoid.clear();
        p.topical_obsessions.clear();
        p.topical_avoidances.clear();
        let s = render_style_profile_as_english(&p);
        assert!(s.contains("voice traits: (none)"));
        assert!(s.contains("ai tells to avoid: (none)"));
        assert!(s.contains("topical obsessions: (none)"));
        assert!(s.contains("topical avoidances: (none)"));
    }

    #[test]
    fn render_sentence_case_when_lowercase_false() {
        let mut p = canonical_profile();
        p.formatting.lowercase = false;
        let s = render_style_profile_as_english(&p);
        assert!(
            s.contains("sentence case"),
            "expected 'sentence case' in formatting; got: {s}"
        );
        assert!(
            !s.contains(", lowercase, "),
            "should not contain 'lowercase' when false; got: {s}"
        );
    }

    #[test]
    fn render_thread_opener_need_not_hook_when_false() {
        let mut p = canonical_profile();
        p.thread_opener_must_hook = false;
        let s = render_style_profile_as_english(&p);
        assert!(
            s.contains("opener need not hook"),
            "expected fallback wording; got: {s}"
        );
    }
}
```

- [ ] **Step 3: Create `crates/heartbit-ghost/src/pipeline/verdicts.rs`**

```rust
//! Structured verdict parsing for `style_critic` and `fact_check`.

use serde::Deserialize;

/// Critic verdict — three branches that drive the revise loop.
#[derive(Debug, Clone, PartialEq)]
pub enum StyleVerdict {
    /// Draft is acceptable; ship it.
    Pass {
        /// 0.0..=1.0 voice match score.
        score: f64,
    },
    /// Draft is recoverable; loop back to the writer with this reason.
    Revise {
        /// Short feedback string, fed into the writer's next user message.
        reason: String,
        /// 0.0..=1.0 voice match score.
        score: f64,
    },
    /// Draft is fundamentally off; abort the pipeline.
    Reject {
        /// Short reason explaining the rejection.
        reason: String,
        /// 0.0..=1.0 voice match score.
        score: f64,
    },
}

impl StyleVerdict {
    /// `style_match_score` extractor.
    pub fn score(&self) -> f64 {
        match self {
            StyleVerdict::Pass { score } | StyleVerdict::Revise { score, .. } | StyleVerdict::Reject { score, .. } => *score,
        }
    }
}

/// Fact-check verdict.
#[derive(Debug, Clone, PartialEq)]
pub enum FactVerdict {
    /// Every factual claim is supported by the research digest.
    Verified,
    /// At least one claim is contradicted by or absent from the digest.
    Unverifiable {
        /// Short reason naming the offending claim.
        reason: String,
    },
}

#[derive(Debug, Deserialize)]
struct CriticRaw {
    verdict: String,
    #[serde(default)]
    reason: Option<String>,
    style_match_score: f64,
}

#[derive(Debug, Deserialize)]
struct FactRaw {
    verdict: String,
    #[serde(default)]
    reason: Option<String>,
}

/// Parse the critic's raw output as JSON. Strips a single ```json fence
/// pair if present (defensive — same pattern as P1.2c's
/// `StyleExtractor::strip_markdown_fences`).
pub fn parse_critic_verdict(raw: &str) -> Result<StyleVerdict, serde_json::Error> {
    let unfenced = strip_fence(raw.trim());
    let parsed: CriticRaw = serde_json::from_str(unfenced)?;
    let verdict = match parsed.verdict.as_str() {
        "pass" => StyleVerdict::Pass {
            score: parsed.style_match_score,
        },
        "revise" => StyleVerdict::Revise {
            reason: parsed.reason.unwrap_or_else(|| "unspecified".to_string()),
            score: parsed.style_match_score,
        },
        "reject" => StyleVerdict::Reject {
            reason: parsed.reason.unwrap_or_else(|| "unspecified".to_string()),
            score: parsed.style_match_score,
        },
        other => {
            // Construct a synthetic serde error for the unknown variant.
            // Use the canonical "unknown variant" message via a small helper.
            return Err(serde::de::Error::unknown_variant(other, &["pass", "revise", "reject"]));
        }
    };
    Ok(verdict)
}

/// Parse the fact_check raw output as JSON. Same fence-stripping defense.
pub fn parse_fact_verdict(raw: &str) -> Result<FactVerdict, serde_json::Error> {
    let unfenced = strip_fence(raw.trim());
    let parsed: FactRaw = serde_json::from_str(unfenced)?;
    let verdict = match parsed.verdict.as_str() {
        "verified" => FactVerdict::Verified,
        "unverifiable" => FactVerdict::Unverifiable {
            reason: parsed.reason.unwrap_or_else(|| "unspecified".to_string()),
        },
        other => {
            return Err(serde::de::Error::unknown_variant(other, &["verified", "unverifiable"]));
        }
    };
    Ok(verdict)
}

/// Strip a single ```json … ``` (or ``` … ```) fence pair if present.
fn strip_fence(s: &str) -> &str {
    let body = s
        .strip_prefix("```json\n")
        .or_else(|| s.strip_prefix("```json"))
        .or_else(|| s.strip_prefix("```\n"))
        .or_else(|| s.strip_prefix("```"))
        .unwrap_or(s);
    body.strip_suffix("```")
        .map(str::trim)
        .unwrap_or(body)
        .trim()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_critic_pass() {
        let raw = r#"{"verdict": "pass", "style_match_score": 0.92}"#;
        let v = parse_critic_verdict(raw).unwrap();
        assert_eq!(v, StyleVerdict::Pass { score: 0.92 });
    }

    #[test]
    fn parse_critic_revise_with_reason() {
        let raw = r#"{"verdict": "revise", "reason": "uses em-dashes", "style_match_score": 0.65}"#;
        let v = parse_critic_verdict(raw).unwrap();
        assert_eq!(
            v,
            StyleVerdict::Revise {
                reason: "uses em-dashes".to_string(),
                score: 0.65,
            }
        );
    }

    #[test]
    fn parse_critic_reject_with_reason() {
        let raw = r#"{"verdict": "reject", "reason": "off-topic", "style_match_score": 0.2}"#;
        let v = parse_critic_verdict(raw).unwrap();
        assert_eq!(
            v,
            StyleVerdict::Reject {
                reason: "off-topic".to_string(),
                score: 0.2,
            }
        );
    }

    #[test]
    fn parse_critic_strips_markdown_fence() {
        let raw = "```json\n{\"verdict\": \"pass\", \"style_match_score\": 0.9}\n```";
        let v = parse_critic_verdict(raw).unwrap();
        assert_eq!(v, StyleVerdict::Pass { score: 0.9 });
    }

    #[test]
    fn parse_critic_malformed_returns_err() {
        let raw = "definitely not json";
        let err = parse_critic_verdict(raw).unwrap_err();
        assert!(format!("{err}").contains("expected"), "got: {err}");
    }

    #[test]
    fn parse_fact_verified() {
        let raw = r#"{"verdict": "verified"}"#;
        let v = parse_fact_verdict(raw).unwrap();
        assert_eq!(v, FactVerdict::Verified);
    }

    #[test]
    fn parse_fact_unverifiable_with_reason() {
        let raw = r#"{"verdict": "unverifiable", "reason": "no source for the 47% figure"}"#;
        let v = parse_fact_verdict(raw).unwrap();
        assert_eq!(
            v,
            FactVerdict::Unverifiable {
                reason: "no source for the 47% figure".to_string()
            }
        );
    }

    #[test]
    fn parse_fact_unknown_verdict_returns_err() {
        let raw = r#"{"verdict": "maybe"}"#;
        let err = parse_fact_verdict(raw).unwrap_err();
        assert!(
            format!("{err}").contains("unknown variant") || format!("{err}").contains("maybe"),
            "got: {err}"
        );
    }
}
```

- [ ] **Step 4: Create `crates/heartbit-ghost/src/pipeline/publish_gate.rs`**

```rust
//! Deterministic pre-publish guard. Char count + thread length only.
//! LLM-based content guardrails (PII / brand safety / etc.) are P1.4.

use thiserror::Error;

use crate::voice::StyleProfile;

/// Errors raised by [`check_publish_gate`].
#[derive(Debug, Error)]
pub enum PublishGateError {
    /// One of the tweets exceeds 280 characters.
    #[error("tweet {index} exceeds 280 chars (got {len}); offending text: {text:?}")]
    TweetTooLong {
        /// 0-based tweet index in the thread.
        index: usize,
        /// Character count.
        len: usize,
        /// The offending tweet text.
        text: String,
    },

    /// The thread has more tweets than `profile.thread_max_length`.
    #[error("thread length {actual} exceeds profile.thread_max_length {max}")]
    ThreadTooLong {
        /// Actual tweet count.
        actual: u32,
        /// Profile-imposed maximum.
        max: u32,
    },

    /// The draft is empty or contains only whitespace.
    #[error("draft is empty")]
    EmptyDraft,
}

/// Validate `draft` against the persona's `profile`. Splits the draft on
/// `\n\n` boundaries to identify thread tweets.
pub fn check_publish_gate(draft: &str, profile: &StyleProfile) -> Result<(), PublishGateError> {
    let tweets: Vec<&str> = draft
        .split("\n\n")
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .collect();

    if tweets.is_empty() {
        return Err(PublishGateError::EmptyDraft);
    }

    let max = profile.thread_max_length;
    let actual = tweets.len() as u32;
    if actual > max {
        return Err(PublishGateError::ThreadTooLong { actual, max });
    }

    for (i, tweet) in tweets.iter().enumerate() {
        let len = tweet.chars().count();
        if len > 280 {
            return Err(PublishGateError::TweetTooLong {
                index: i,
                len,
                text: (*tweet).to_string(),
            });
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voice::{
        EmDashPolicy, EmojiPolicy, Formatting, FragmentFrequency, HashtagPolicy, LineBreaks,
        OpeningPattern, PeriodsPolicy, QuotationMarks, SentenceLengthTarget, SpecificityTarget,
        ThreadRhythm,
    };

    fn profile_with_max(thread_max_length: u32) -> StyleProfile {
        StyleProfile {
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
            emoji_policy: EmojiPolicy::RarePunchlineOnly,
            hashtag_policy: HashtagPolicy::Never,
            specificity_target: SpecificityTarget::High,
            voice_traits: vec![],
            ai_tells_to_avoid: vec![],
            thread_rhythm: ThreadRhythm::Linear,
            thread_max_length,
            thread_opener_must_hook: false,
            topical_obsessions: vec![],
            topical_avoidances: vec![],
        }
    }

    #[test]
    fn single_tweet_under_280_passes() {
        let p = profile_with_max(10);
        check_publish_gate("a short post", &p).unwrap();
    }

    #[test]
    fn single_tweet_over_280_rejected() {
        let p = profile_with_max(10);
        let long = "a".repeat(281);
        let err = check_publish_gate(&long, &p).unwrap_err();
        match err {
            PublishGateError::TweetTooLong { index, len, .. } => {
                assert_eq!(index, 0);
                assert_eq!(len, 281);
            }
            other => panic!("expected TweetTooLong, got {other:?}"),
        }
    }

    #[test]
    fn thread_within_limit_passes() {
        let p = profile_with_max(3);
        let thread = "first tweet\n\nsecond tweet\n\nthird tweet";
        check_publish_gate(thread, &p).unwrap();
    }

    #[test]
    fn thread_exceeding_limit_rejected() {
        let p = profile_with_max(2);
        let thread = "one\n\ntwo\n\nthree";
        let err = check_publish_gate(thread, &p).unwrap_err();
        match err {
            PublishGateError::ThreadTooLong { actual, max } => {
                assert_eq!(actual, 3);
                assert_eq!(max, 2);
            }
            other => panic!("expected ThreadTooLong, got {other:?}"),
        }
    }

    #[test]
    fn thread_with_individual_tweet_too_long_rejected() {
        let p = profile_with_max(5);
        let big = "x".repeat(290);
        let thread = format!("ok first\n\n{big}\n\nthird");
        let err = check_publish_gate(&thread, &p).unwrap_err();
        match err {
            PublishGateError::TweetTooLong { index, len, .. } => {
                assert_eq!(index, 1);
                assert_eq!(len, 290);
            }
            other => panic!("expected TweetTooLong, got {other:?}"),
        }
    }

    #[test]
    fn empty_or_whitespace_draft_rejected() {
        let p = profile_with_max(10);
        assert!(matches!(
            check_publish_gate("", &p).unwrap_err(),
            PublishGateError::EmptyDraft
        ));
        assert!(matches!(
            check_publish_gate("   \n\n   \n\n", &p).unwrap_err(),
            PublishGateError::EmptyDraft
        ));
    }
}
```

- [ ] **Step 5: Create `crates/heartbit-ghost/src/pipeline/prompts.rs`**

```rust
//! User-message builders for each pipeline stage. Pure string composition;
//! tested indirectly via the integration tests in `pipeline::tests`.

/// Construct the writer's user message.
///
/// On the first iteration (no `prev_revision`), only includes topic +
/// research digest + voice guidelines. On revision, also includes the
/// previous draft and the critic's feedback.
pub(crate) fn build_writer_user_message(
    topic: &str,
    research_digest: &str,
    voice_guidelines: &str,
    prev_revision: Option<&(String, String)>,
) -> String {
    let mut out = String::new();
    out.push_str(&format!("Topic: {topic}\n\n"));
    out.push_str("Research digest:\n");
    out.push_str(research_digest);
    out.push_str("\n\n");
    out.push_str(voice_guidelines);
    out.push('\n');

    if let Some((prev_draft, critic_reason)) = prev_revision {
        out.push_str("\nPREVIOUS DRAFT:\n");
        out.push_str(prev_draft);
        out.push_str("\n\nSTYLE CRITIC FEEDBACK:\n");
        out.push_str(critic_reason);
        out.push_str(
            "\n\nPlease produce a revised draft addressing the feedback. \
             Output the post text only.\n",
        );
    } else {
        out.push_str("\nProduce one draft. Output the post text only.\n");
    }

    out
}

/// Construct the style_critic's user message.
pub(crate) fn build_critic_user_message(draft: &str, voice_guidelines: &str) -> String {
    format!(
        "Draft to evaluate:\n{draft}\n\n{voice_guidelines}\n\
         Score the draft and return your verdict as JSON per the schema.\n"
    )
}

/// Construct the fact_check's user message.
pub(crate) fn build_fact_user_message(draft: &str, research_digest: &str) -> String {
    format!(
        "Draft to verify:\n{draft}\n\nResearch digest (only source of truth):\n{research_digest}\n\
         Verify and return your verdict as JSON per the schema.\n"
    )
}
```

- [ ] **Step 6: Modify `crates/heartbit-ghost/src/lib.rs` — add `pub mod pipeline;`**

The current `lib.rs` has:

```rust
pub mod agents;
pub mod corpus;
pub mod tools;
pub mod voice;
```

Add `pub mod pipeline;` alphabetically (between `corpus` and `tools`):

```rust
pub mod agents;
pub mod corpus;
pub mod pipeline;
pub mod tools;
pub mod voice;
```

- [ ] **Step 7: Run the tests**

```bash
cargo test -p heartbit-ghost --lib pipeline
```

Expected: `18 passed; 0 failed; 0 ignored` (4 style_render + 8 verdicts + 6 publish_gate + 0 prompts).

- [ ] **Step 8: Workspace quality gate**

```bash
cargo fmt -p heartbit-ghost -- --check
cargo clippy -p heartbit-ghost --all-targets -- -D warnings
```

Both clean.

- [ ] **Step 9: Commit**

```bash
git add crates/heartbit-ghost/src/pipeline/ crates/heartbit-ghost/src/lib.rs
git commit -m "$(cat <<'EOF'
feat(ghost): pipeline — pure helpers (style_render + verdicts + publish_gate + prompts) (P1.3b)

Foundation for the P1.3b orchestrator. Four pure modules with no I/O,
no LLM, no async — fully composable into Task 3's run_pipeline body:

- style_render: render_style_profile_as_english produces a structured
  ~200-token English block from a StyleProfile, surfacing all 16
  non-version fields.
- verdicts: StyleVerdict + FactVerdict + parse_critic_verdict +
  parse_fact_verdict. Defensive markdown-fence stripping (same pattern
  as P1.2c StyleExtractor).
- publish_gate: check_publish_gate validates char count (≤ 280 per
  tweet) + thread length (≤ profile.thread_max_length). LLM-based
  guards (PII/brand safety/etc.) are P1.4.
- prompts: build_writer_user_message + build_critic_user_message +
  build_fact_user_message. Tested indirectly via Task 3 integration.

18 tests: 4 style_render (canonical / empty lists / lowercase=false /
opener_need_not_hook), 8 verdicts (5 critic + 3 fact), 6 publish_gate
(single ok / single too long / thread ok / thread too long /
individual-in-thread too long / empty draft).

Refs: docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.3b-pipeline-orchestrator-design.md §5 §6 §7
EOF
)"
```

---

## Task 2: `PipelineConfig` + `PipelineOutput` + `PipelineError` + `runner_from_recipe`

**Why:** The orchestration scaffolding before Task 3's `run_pipeline` body. Defines the public surface + a private helper that converts a P1.3a `AgentConfig` recipe + tool subset into a runnable `AgentRunner<BoxedProvider>`.

**Files:**
- Modify: `crates/heartbit-ghost/src/pipeline/mod.rs` — append types + helper

- [ ] **Step 1: Replace `crates/heartbit-ghost/src/pipeline/mod.rs` with the full body**

```rust
//! Generation pipeline — wires the P1.3a sub-agent recipes into a working
//! single-candidate path.
//!
//! Public entry: [`run_pipeline`] (added in Task 3).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use heartbit_core::agent::AgentRunner;
use heartbit_core::config::AgentConfig;
use heartbit_core::error::Error as CoreError;
use heartbit_core::llm::types::{ReasoningEffort, TokenUsage};
use heartbit_core::llm::BoxedProvider;
use heartbit_core::tool::Tool;
use thiserror::Error;

use crate::voice::SnapshotError;

pub mod prompts;
pub mod publish_gate;
pub mod style_render;
pub mod verdicts;

pub use publish_gate::{PublishGateError, check_publish_gate};
pub use style_render::render_style_profile_as_english;
pub use verdicts::{FactVerdict, StyleVerdict, parse_critic_verdict, parse_fact_verdict};

/// Configuration for one pipeline run.
pub struct PipelineConfig<'a> {
    /// Persona instance name (used to load the StyleProfile snapshot).
    pub persona_name: &'a str,
    /// Topic / prompt for this run.
    pub topic: &'a str,
    /// LLM provider (shared across all 4 sub-agents).
    pub provider: Arc<BoxedProvider>,
    /// Corpora root (currently unused; reserved for P1.3e few-shot retrieval).
    pub corpora_root: &'a Path,
    /// Profiles root (passed to SnapshotStore::open).
    pub profiles_root: &'a Path,
    /// Optional progress callback. Called with a short status string at each
    /// pipeline stage start.
    pub on_progress: Option<Arc<dyn Fn(&str) + Send + Sync>>,
}

/// Output of a successful pipeline run.
#[derive(Debug, Clone)]
pub struct PipelineOutput {
    /// The final post draft (single tweet or `\n\n`-separated thread).
    pub final_draft: String,
    /// Researcher's digest text.
    pub research_digest: String,
    /// `style_match_score` from the critic on the accepted draft.
    pub style_match_score: f64,
    /// Number of writer iterations until pass (1..=3).
    pub revise_iterations: usize,
    /// Fact-check verdict on the final draft.
    pub fact_check_verdict: FactVerdict,
    /// Accumulated token usage across all 4 agent calls.
    pub usage_summary: TokenUsage,
}

/// Errors raised by [`run_pipeline`].
#[derive(Debug, Error)]
pub enum PipelineError {
    /// No StyleProfile snapshot exists for this persona.
    #[error(
        "no profile snapshot for persona '{persona}' at {}; run `heartbit persona profile rebuild {persona}` first",
        profiles_dir.display()
    )]
    NoProfileSnapshot {
        /// Persona name passed in.
        persona: String,
        /// Resolved profiles directory path.
        profiles_dir: PathBuf,
    },

    /// SnapshotStore I/O / parse failure.
    #[error("snapshot: {0}")]
    Snapshot(#[from] SnapshotError),

    /// AgentRunner construction failed.
    #[error("agent builder for stage '{stage}': {source}")]
    Builder {
        /// Which stage's builder failed.
        stage: String,
        /// Underlying core error.
        #[source]
        source: CoreError,
    },

    /// Agent execution error (network, LLM error, etc.).
    #[error("agent execution at stage '{stage}': {source}")]
    Agent {
        /// Which stage's agent was running.
        stage: String,
        /// Underlying core error.
        #[source]
        source: CoreError,
    },

    /// style_critic returned a malformed verdict.
    #[error("style_critic verdict parse: {source}")]
    CriticParseFailed {
        /// The underlying serde_json error.
        #[source]
        source: serde_json::Error,
        /// The raw critic output that failed to parse.
        raw: String,
    },

    /// fact_check returned a malformed verdict.
    #[error("fact_check verdict parse: {source}")]
    FactCheckParseFailed {
        /// The underlying serde_json error.
        #[source]
        source: serde_json::Error,
        /// The raw fact_check output that failed to parse.
        raw: String,
    },

    /// style_critic returned `Reject` — draft is fundamentally off.
    #[error("style_critic rejected the draft: {reason}")]
    Rejected {
        /// Reason from the critic.
        reason: String,
        /// 0.0..=1.0 score.
        score: f64,
    },

    /// 3 revise iterations exhausted without `Pass`.
    #[error("revise loop exhausted after {iterations} iterations; last reason: {last_reason}")]
    MaxRevisionsExceeded {
        /// Number of iterations attempted.
        iterations: usize,
        /// The final draft produced (failed).
        last_draft: String,
        /// The last critic feedback.
        last_reason: String,
        /// 0.0..=1.0 score from the last iteration.
        last_score: f64,
    },

    /// publish_gate rejected the final draft.
    #[error("publish_gate: {0}")]
    PublishGate(#[from] PublishGateError),
}

/// Build an [`AgentRunner`] from a P1.3a [`AgentConfig`] recipe and a
/// (possibly empty) tool subset.
///
/// Maps `AgentConfig.{name, system_prompt, max_turns, max_tokens,
/// reasoning_effort, response_schema}` onto the corresponding builder
/// methods. The `description` field is metadata-only (not used at
/// runtime). Reasoning effort strings map: "high" → High, "medium" →
/// Medium, "low" → Low; absent or unknown → no `.reasoning_effort()` call.
pub(crate) fn runner_from_recipe(
    provider: Arc<BoxedProvider>,
    recipe: AgentConfig,
    tools: Vec<Arc<dyn Tool>>,
) -> Result<AgentRunner<BoxedProvider>, CoreError> {
    let mut builder = AgentRunner::builder(provider)
        .name(recipe.name)
        .system_prompt(recipe.system_prompt)
        .tools(tools);
    if let Some(n) = recipe.max_turns {
        builder = builder.max_turns(n);
    }
    if let Some(n) = recipe.max_tokens {
        builder = builder.max_tokens(n);
    }
    if let Some(effort) = recipe.reasoning_effort.as_deref() {
        match effort {
            "high" => builder = builder.reasoning_effort(ReasoningEffort::High),
            "medium" => builder = builder.reasoning_effort(ReasoningEffort::Medium),
            "low" => builder = builder.reasoning_effort(ReasoningEffort::Low),
            _ => { /* unknown — leave default */ }
        }
    }
    if let Some(schema) = recipe.response_schema {
        builder = builder.structured_schema(schema);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_profile_snapshot_error_renders_with_persona_and_path() {
        let e = PipelineError::NoProfileSnapshot {
            persona: "x".to_string(),
            profiles_dir: PathBuf::from("/tmp/profiles"),
        };
        let s = format!("{e}");
        assert!(s.contains("'x'"), "got: {s}");
        assert!(s.contains("/tmp/profiles"), "got: {s}");
        assert!(s.contains("profile rebuild x"), "got: {s}");
    }

    #[test]
    fn rejected_error_renders_with_reason() {
        let e = PipelineError::Rejected {
            reason: "off-topic".to_string(),
            score: 0.2,
        };
        let s = format!("{e}");
        assert!(s.contains("off-topic"), "got: {s}");
        assert!(s.contains("rejected"), "got: {s}");
    }

    #[test]
    fn max_revisions_error_renders_with_iterations_and_reason() {
        let e = PipelineError::MaxRevisionsExceeded {
            iterations: 3,
            last_draft: "draft".to_string(),
            last_reason: "still off-voice".to_string(),
            last_score: 0.6,
        };
        let s = format!("{e}");
        assert!(s.contains("3 iterations"), "got: {s}");
        assert!(s.contains("still off-voice"), "got: {s}");
    }
}
```

- [ ] **Step 2: Run the tests**

```bash
cargo test -p heartbit-ghost --lib pipeline
```

Expected: `21 passed; 0 failed; 0 ignored` (18 from Task 1 + 3 new error display tests).

- [ ] **Step 3: Workspace quality gate**

```bash
cargo fmt -p heartbit-ghost -- --check
cargo clippy -p heartbit-ghost --all-targets -- -D warnings
```

Both clean.

- [ ] **Step 4: Commit**

```bash
git add crates/heartbit-ghost/src/pipeline/mod.rs
git commit -m "$(cat <<'EOF'
feat(ghost): pipeline — types + runner_from_recipe helper (P1.3b)

PipelineConfig + PipelineOutput + PipelineError (10 variants) + private
runner_from_recipe(provider, recipe, tools) helper that bridges
AgentConfig to AgentRunner. Maps name/system_prompt/max_turns/max_tokens/
reasoning_effort/response_schema onto AgentRunnerBuilder; description
field is metadata-only (not transferred to runtime).

reasoning_effort string mapping: "high"/"medium"/"low" → enum variants;
absent or unknown → no call (default).

response_schema → AgentRunnerBuilder::structured_schema (note: builder
uses structured_schema internally, AgentConfig uses response_schema —
different names, same field).

3 tests on PipelineError display: NoProfileSnapshot points to fix
command, Rejected/MaxRevisionsExceeded surface reason and counters.

Refs: docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.3b-pipeline-orchestrator-design.md §3
EOF
)"
```

---

## Task 3: `run_pipeline` body + integration tests

**Why:** The orchestration. Wires Task 1's pure helpers + Task 2's scaffolding into the actual pipeline. Includes a `MockProvider` test helper (mirrors P1.2c/P1.2e pattern) and 5 integration tests.

**Files:**
- Modify: `crates/heartbit-ghost/src/pipeline/mod.rs` — append `run_pipeline` body, `MockProvider` test helper, 5 integration tests

- [ ] **Step 1: Append `run_pipeline` to `crates/heartbit-ghost/src/pipeline/mod.rs` (after `runner_from_recipe`, before `#[cfg(test)] mod tests`)**

```rust
/// Execute one generation pipeline run: research → write → critic
/// (with revise loop, max 3) → fact_check → publish_gate → stdout.
pub async fn run_pipeline(cfg: PipelineConfig<'_>) -> Result<PipelineOutput, PipelineError> {
    let progress = |msg: &str| {
        if let Some(cb) = cfg.on_progress.as_ref() {
            cb(msg);
        }
    };

    // 1. Load StyleProfile snapshot.
    progress("Loading profile snapshot...");
    let store = crate::voice::SnapshotStore::open(cfg.profiles_root, cfg.persona_name)?;
    let snapshot = store.load_latest()?.ok_or_else(|| PipelineError::NoProfileSnapshot {
        persona: cfg.persona_name.to_string(),
        profiles_dir: cfg.profiles_root.join(cfg.persona_name),
    })?;
    let profile = snapshot.profile;

    // 2. Build the 4 AgentRunner instances from P1.3a recipes.
    use crate::agents::{
        fact_check_recipe, researcher_recipe, style_critic_recipe, writer_recipe,
    };
    use heartbit_core::tool::builtins::{WebFetchTool, WebSearchTool};

    let researcher_tools: Vec<Arc<dyn Tool>> =
        vec![Arc::new(WebSearchTool::new()), Arc::new(WebFetchTool::new())];
    let researcher = runner_from_recipe(cfg.provider.clone(), researcher_recipe(), researcher_tools)
        .map_err(|e| PipelineError::Builder { stage: "researcher".to_string(), source: e })?;
    let writer = runner_from_recipe(cfg.provider.clone(), writer_recipe(), Vec::new())
        .map_err(|e| PipelineError::Builder { stage: "writer".to_string(), source: e })?;
    let critic = runner_from_recipe(cfg.provider.clone(), style_critic_recipe(), Vec::new())
        .map_err(|e| PipelineError::Builder { stage: "style_critic".to_string(), source: e })?;
    let fact = runner_from_recipe(cfg.provider.clone(), fact_check_recipe(), Vec::new())
        .map_err(|e| PipelineError::Builder { stage: "fact_check".to_string(), source: e })?;

    let mut total_usage = TokenUsage::default();

    // 3. Researcher.
    progress("Researching topic...");
    let researcher_out = researcher.execute(cfg.topic).await.map_err(|e| {
        PipelineError::Agent { stage: "researcher".to_string(), source: e }
    })?;
    let research_digest = researcher_out.result.clone();
    total_usage += researcher_out.tokens_used;

    // 4. Render voice guidelines.
    let voice_guidelines = render_style_profile_as_english(&profile);

    // 5. Revise loop.
    let mut prev_revision: Option<(String, String)> = None;
    let mut final_state: Option<(String, f64, usize)> = None;
    let mut last_score: f64 = 0.0;

    for iter in 1..=3usize {
        progress(&format!("Drafting (iter {iter})..."));
        let writer_msg = prompts::build_writer_user_message(
            cfg.topic,
            &research_digest,
            &voice_guidelines,
            prev_revision.as_ref(),
        );
        let writer_out = writer.execute(&writer_msg).await.map_err(|e| {
            PipelineError::Agent {
                stage: format!("writer (iter {iter})"),
                source: e,
            }
        })?;
        let draft = writer_out.result.clone();
        total_usage += writer_out.tokens_used;

        progress(&format!("Style-checking (iter {iter})..."));
        let critic_msg = prompts::build_critic_user_message(&draft, &voice_guidelines);
        let critic_out = critic.execute(&critic_msg).await.map_err(|e| {
            PipelineError::Agent {
                stage: format!("style_critic (iter {iter})"),
                source: e,
            }
        })?;
        total_usage += critic_out.tokens_used;
        let verdict = parse_critic_verdict(&critic_out.result).map_err(|e| {
            PipelineError::CriticParseFailed { source: e, raw: critic_out.result.clone() }
        })?;
        last_score = verdict.score();

        match verdict {
            StyleVerdict::Pass { score } => {
                final_state = Some((draft, score, iter));
                break;
            }
            StyleVerdict::Reject { reason, score } => {
                return Err(PipelineError::Rejected { reason, score });
            }
            StyleVerdict::Revise { reason, .. } => {
                prev_revision = Some((draft, reason));
                continue;
            }
        }
    }

    let (final_draft, score, iterations) = match final_state {
        Some(v) => v,
        None => {
            let (last_draft, last_reason) = prev_revision.unwrap_or_default();
            return Err(PipelineError::MaxRevisionsExceeded {
                iterations: 3,
                last_draft,
                last_reason,
                last_score,
            });
        }
    };

    // 6. fact_check (non-blocking on Unverifiable).
    progress("Fact-checking...");
    let fact_msg = prompts::build_fact_user_message(&final_draft, &research_digest);
    let fact_out = fact.execute(&fact_msg).await.map_err(|e| {
        PipelineError::Agent { stage: "fact_check".to_string(), source: e }
    })?;
    total_usage += fact_out.tokens_used;
    let fact_verdict = parse_fact_verdict(&fact_out.result).map_err(|e| {
        PipelineError::FactCheckParseFailed { source: e, raw: fact_out.result.clone() }
    })?;
    if let FactVerdict::Unverifiable { ref reason } = fact_verdict {
        progress(&format!("fact_check unverifiable: {reason}"));
    }

    // 7. publish_gate.
    progress("Running publish_gate...");
    check_publish_gate(&final_draft, &profile)?;

    // 8. Print + return.
    println!("{final_draft}");
    progress("Done.");
    Ok(PipelineOutput {
        final_draft,
        research_digest,
        style_match_score: score,
        revise_iterations: iterations,
        fact_check_verdict: fact_verdict,
        usage_summary: total_usage,
    })
}
```

- [ ] **Step 2: Append the `MockProvider` test helper + 5 integration tests inside the existing `#[cfg(test)] mod tests` block**

Append after the 3 PipelineError display tests:

```rust
    use heartbit_core::error::Error as CoreError;
    use heartbit_core::llm::types::{
        CompletionRequest, CompletionResponse, ContentBlock, StopReason, TokenUsage,
    };
    use heartbit_core::llm::{BoxedProvider, LlmProvider};
    use std::future::Future;
    use std::sync::Mutex;
    use tempfile::TempDir;

    /// MockProvider returns a queue of canned responses, one per `complete()` call.
    /// Calls past the queue's end return `Error::Agent("mock exhausted")`.
    struct MockProvider {
        responses: Mutex<std::collections::VecDeque<String>>,
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
            _request: CompletionRequest,
        ) -> impl Future<Output = Result<CompletionResponse, CoreError>> + Send {
            let response = self.responses.lock().unwrap().pop_front();
            async move {
                let text = response.ok_or_else(|| {
                    CoreError::Agent("mock exhausted".to_string())
                })?;
                Ok(CompletionResponse {
                    content: vec![ContentBlock::Text { text }],
                    usage: TokenUsage::default(),
                    stop_reason: StopReason::EndTurn,
                    model: None,
                })
            }
        }
    }

    /// Snapshot fixture — minimal valid StyleProfile + recipe, saved to TempDir.
    fn seed_snapshot(persona: &str) -> (TempDir, std::path::PathBuf) {
        use crate::voice::{
            BlendEntry, BlendRecipe, EmDashPolicy, EmojiPolicy, Formatting,
            FragmentFrequency, HashtagPolicy, LineBreaks, OpeningPattern,
            PartialStyleProfile, PeriodsPolicy, QuotationMarks, SentenceLengthTarget,
            SnapshotStore, SpecificityTarget, StyleProfile, ThreadRhythm,
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
            blend: vec![BlendEntry { writer: "k".to_string(), weight: 1.0 }],
            overrides: PartialStyleProfile::default(),
        };
        let store = SnapshotStore::open(dir.path(), persona).unwrap();
        store.save_new(profile, &recipe).unwrap();
        let root = dir.path().to_path_buf();
        (dir, root)
    }

    #[tokio::test]
    async fn run_pipeline_happy_path_single_iteration() {
        let (_dir, profiles_root) = seed_snapshot("x");
        let provider = MockProvider::arc(vec![
            "Research digest:\n- AI is moving fast",                      // researcher
            "concrete short post",                                         // writer iter 1
            r#"{"verdict": "pass", "style_match_score": 0.92}"#,          // critic iter 1
            r#"{"verdict": "verified"}"#,                                  // fact_check
        ]);
        let corpora = profiles_root.clone();
        let cfg = PipelineConfig {
            persona_name: "x",
            topic: "AI capabilities",
            provider,
            corpora_root: &corpora,
            profiles_root: &profiles_root,
            on_progress: None,
        };
        let out = run_pipeline(cfg).await.expect("happy path");
        assert_eq!(out.final_draft, "concrete short post");
        assert_eq!(out.revise_iterations, 1);
        assert!((out.style_match_score - 0.92).abs() < 1e-9);
        assert_eq!(out.fact_check_verdict, FactVerdict::Verified);
    }

    #[tokio::test]
    async fn run_pipeline_revise_once_then_pass() {
        let (_dir, profiles_root) = seed_snapshot("x");
        let provider = MockProvider::arc(vec![
            "Research digest:\n- topic notes",                                              // researcher
            "first draft with em-dashes — like this",                                       // writer iter 1
            r#"{"verdict": "revise", "reason": "uses em-dashes", "style_match_score": 0.6}"#, // critic iter 1
            "second draft, no em-dashes",                                                   // writer iter 2
            r#"{"verdict": "pass", "style_match_score": 0.91}"#,                            // critic iter 2
            r#"{"verdict": "verified"}"#,                                                   // fact_check
        ]);
        let cfg = PipelineConfig {
            persona_name: "x",
            topic: "Rust async patterns",
            provider,
            corpora_root: &profiles_root,
            profiles_root: &profiles_root,
            on_progress: None,
        };
        let out = run_pipeline(cfg).await.expect("revise then pass");
        assert_eq!(out.final_draft, "second draft, no em-dashes");
        assert_eq!(out.revise_iterations, 2);
    }

    #[tokio::test]
    async fn run_pipeline_max_revisions_exceeded() {
        let (_dir, profiles_root) = seed_snapshot("x");
        let provider = MockProvider::arc(vec![
            "Research digest:\n- notes",                                                  // researcher
            "draft 1",                                                                     // writer iter 1
            r#"{"verdict": "revise", "reason": "off-voice (1)", "style_match_score": 0.5}"#, // critic iter 1
            "draft 2",                                                                     // writer iter 2
            r#"{"verdict": "revise", "reason": "off-voice (2)", "style_match_score": 0.5}"#, // critic iter 2
            "draft 3",                                                                     // writer iter 3
            r#"{"verdict": "revise", "reason": "off-voice (3)", "style_match_score": 0.5}"#, // critic iter 3
        ]);
        let cfg = PipelineConfig {
            persona_name: "x",
            topic: "topic",
            provider,
            corpora_root: &profiles_root,
            profiles_root: &profiles_root,
            on_progress: None,
        };
        let err = run_pipeline(cfg).await.unwrap_err();
        match err {
            PipelineError::MaxRevisionsExceeded { iterations, last_reason, .. } => {
                assert_eq!(iterations, 3);
                assert!(last_reason.contains("(3)"), "last reason should be from iter 3; got: {last_reason}");
            }
            other => panic!("expected MaxRevisionsExceeded, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn run_pipeline_critic_reject_aborts() {
        let (_dir, profiles_root) = seed_snapshot("x");
        let provider = MockProvider::arc(vec![
            "Research digest:\n- notes",                                              // researcher
            "off-topic draft",                                                         // writer iter 1
            r#"{"verdict": "reject", "reason": "off-topic", "style_match_score": 0.1}"#, // critic iter 1
        ]);
        let cfg = PipelineConfig {
            persona_name: "x",
            topic: "topic",
            provider,
            corpora_root: &profiles_root,
            profiles_root: &profiles_root,
            on_progress: None,
        };
        let err = run_pipeline(cfg).await.unwrap_err();
        match err {
            PipelineError::Rejected { reason, score } => {
                assert_eq!(reason, "off-topic");
                assert!((score - 0.1).abs() < 1e-9);
            }
            other => panic!("expected Rejected, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn run_pipeline_no_profile_snapshot_returns_error() {
        // Don't seed a snapshot — the persona dir doesn't exist on disk.
        let dir = TempDir::new().unwrap();
        let provider = MockProvider::arc(vec![]); // never called
        let root = dir.path().to_path_buf();
        let cfg = PipelineConfig {
            persona_name: "x",
            topic: "topic",
            provider,
            corpora_root: &root,
            profiles_root: &root,
            on_progress: None,
        };
        let err = run_pipeline(cfg).await.unwrap_err();
        match err {
            PipelineError::NoProfileSnapshot { persona, .. } => {
                assert_eq!(persona, "x");
            }
            other => panic!("expected NoProfileSnapshot, got {other:?}"),
        }
    }
```

- [ ] **Step 3: Run the tests**

```bash
cargo test -p heartbit-ghost --lib pipeline
```

Expected: `26 passed; 0 failed; 0 ignored` (21 from prior tasks + 5 integration tests).

- [ ] **Step 4: Workspace quality gate**

```bash
cargo fmt -p heartbit-ghost -- --check
cargo clippy -p heartbit-ghost --all-targets -- -D warnings
```

Both clean. Note: integration tests use real `print!` to stdout — that's intentional (the spec requires the final draft to go to stdout). `cargo test` captures stdout per test by default; the prints don't pollute the test runner output.

- [ ] **Step 5: Commit**

```bash
git add crates/heartbit-ghost/src/pipeline/mod.rs
git commit -m "$(cat <<'EOF'
feat(ghost): pipeline — run_pipeline orchestration body (P1.3b)

The orchestration layer. run_pipeline:

1. Loads StyleProfile via SnapshotStore::load_latest (NoProfileSnapshot
   error if absent).
2. Builds 4 AgentRunner instances from P1.3a recipes via the
   runner_from_recipe helper (researcher gets websearch+webfetch tools;
   the LLM-only agents get no tools).
3. Researcher runs first; output is the digest.
4. render_style_profile_as_english(&profile) → voice guidelines for
   the writer's user message.
5. Revise loop (max 3): writer → style_critic → match StyleVerdict.
   Pass exits the loop; Reject aborts; Revise feeds the previous
   draft + reason into the next iteration.
6. fact_check (non-blocking on Unverifiable; logs via on_progress).
7. publish_gate (char count + thread length).
8. Print final draft to stdout; return PipelineOutput with
   accumulated TokenUsage.

5 integration tests (with hand-rolled MockProvider returning canned
text per stage): happy path single iter, revise once then pass,
max revisions exceeded, critic Reject aborts, no profile snapshot.

Refs: docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.3b-pipeline-orchestrator-design.md §4
EOF
)"
```

---

## Task 4: CLI body wiring + dispatch tests

**Why:** Wire `heartbit persona run <name> --once "<topic>"` to call `run_pipeline`. 2 dispatch-level tests for error paths.

**Files:**
- Modify: `crates/heartbit-cli/src/persona.rs` — replace `Run` stub error with the body + 2 tests

- [ ] **Step 1: Modify the dispatch's `Run` arm in `crates/heartbit-cli/src/persona.rs`**

The current dispatch has the `PersonaCommand::Show | Run | Phase | ...` group returning a stub error. Split out `Run` and wire it up.

Find the existing pattern (around lines 156–172):

```rust
        PersonaCommand::Show { name }
        | PersonaCommand::Run { name, .. }
        | PersonaCommand::Phase { name, .. }
        | PersonaCommand::Pause { name }
        | PersonaCommand::Resume { name }
        | PersonaCommand::ExportPreferences { name, .. }
        | PersonaCommand::Audit { name, .. } => {
            if registry.get(&name).is_none() {
                let suffix = registry_suffix(registry);
                return Err(anyhow!("persona '{name}' not found. {suffix}"));
            }
            // P1.0 ships the registration shell; subcommand bodies land in
            // later sub-phases (P1.1–P1.4) alongside the persona's tools,
            // voice modeling, and pipeline.
            Err(anyhow!(
                "persona '{name}': subcommand body is not yet implemented (P1.0 scaffolding stub). The persona is registered; its tools, voice modeling, and pipeline land in later sub-phases."
            ))
        }
```

Replace with:

```rust
        PersonaCommand::Run { name, once } => {
            if registry.get(&name).is_none() {
                return Err(anyhow!(
                    "persona '{name}' not found. {}",
                    registry_suffix(registry)
                ));
            }

            let provider = build_provider_from_env(None)
                .map_err(|e| anyhow!("build llm provider: {e}"))?;
            let corpora_root = heartbit_ghost::corpus::default_corpora_dir()
                .map_err(|e| anyhow!("resolve corpora dir: {e}"))?;
            let profiles_root = heartbit_ghost::voice::default_profiles_dir()
                .map_err(|e| anyhow!("resolve profiles dir: {e}"))?;

            let on_progress: std::sync::Arc<dyn Fn(&str) + Send + Sync> =
                std::sync::Arc::new(|s: &str| eprintln!("> {s}"));

            let cfg = heartbit_ghost::pipeline::PipelineConfig {
                persona_name: &name,
                topic: &once,
                provider,
                corpora_root: &corpora_root,
                profiles_root: &profiles_root,
                on_progress: Some(on_progress),
            };

            let output = heartbit_ghost::pipeline::run_pipeline(cfg)
                .await
                .map_err(|e| anyhow!("pipeline: {e}"))?;

            // run_pipeline already printed final_draft to stdout.
            eprintln!(
                "> ok: revise iterations={}, style match={:.2}, fact check={:?}",
                output.revise_iterations, output.style_match_score, output.fact_check_verdict
            );
            Ok(())
        }
        PersonaCommand::Show { name }
        | PersonaCommand::Phase { name, .. }
        | PersonaCommand::Pause { name }
        | PersonaCommand::Resume { name }
        | PersonaCommand::ExportPreferences { name, .. }
        | PersonaCommand::Audit { name, .. } => {
            if registry.get(&name).is_none() {
                return Err(anyhow!(
                    "persona '{name}' not found. {}",
                    registry_suffix(registry)
                ));
            }
            // Other subcommands land in P1.4.
            Err(anyhow!(
                "persona '{name}': subcommand body is not yet implemented. \
                 Use `heartbit persona run` for one-off generation; other \
                 subcommands land in P1.4."
            ))
        }
```

(The `Run` arm is split out and gets its own body; the remaining 6 stub arms keep the old error message but the wording is updated to reflect that `Run` works.)

- [ ] **Step 2: Append 2 dispatch tests inside the existing `#[cfg(test)] mod tests` block**

Append at the end (after the existing P1.2e and earlier tests):

```rust
    #[tokio::test]
    async fn run_persona_not_found_returns_error() {
        let r = PersonaRegistry::new();
        let cmd = PersonaCommand::Run {
            name: "no-such-persona".to_string(),
            once: "topic".to_string(),
        };
        let result = dispatch(cmd, &r).await;
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("no-such-persona"), "got: {msg}");
    }

    #[tokio::test]
    async fn run_unknown_persona_with_registered_persona_lists_available() {
        let mut r = PersonaRegistry::new();
        heartbit_ghost::register(&mut r);
        let cmd = PersonaCommand::Run {
            name: "no-such-persona".to_string(),
            once: "topic".to_string(),
        };
        let result = dispatch(cmd, &r).await;
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("no-such-persona"), "got: {msg}");
        assert!(
            msg.contains("Available personas: heartbit-ghost:x"),
            "got: {msg}"
        );
        assert!(!msg.contains("No personas registered"), "got: {msg}");
    }
```

- [ ] **Step 3: Run the tests**

```bash
cargo test -p heartbit-cli --lib persona
```

Expected: existing CLI tests + 2 new = passes. The 2 new tests don't actually call `run_pipeline` (they short-circuit at `registry.get(...).is_none()`), so they don't need a real LLM provider.

- [ ] **Step 4: Workspace quality gate**

```bash
cargo fmt -p heartbit-cli -- --check
cargo clippy -p heartbit-cli --all-targets -- -D warnings
```

Both clean.

- [ ] **Step 5: Commit**

```bash
git add crates/heartbit-cli/src/persona.rs
git commit -m "$(cat <<'EOF'
feat(cli): persona run — wire P1.3b pipeline (P1.3b)

Splits PersonaCommand::Run out of the stub error group and wires it
to call heartbit_ghost::pipeline::run_pipeline. Constructs
PipelineConfig from CLI args + env-resolved paths
(default_corpora_dir, default_profiles_dir), passes a stderr-printing
on_progress callback, and prints a one-line summary to stderr after
the pipeline completes.

run_pipeline already prints the final draft to stdout, so the caller
can pipe: heartbit persona run x --once "topic" > draft.txt

The 6 remaining stub-error subcommands (Show / Phase / Pause / Resume
/ ExportPreferences / Audit) keep the not-yet-implemented error but
the wording is updated to mention that `run` works.

2 dispatch tests: run_persona_not_found_returns_error,
run_unknown_persona_with_registered_persona_lists_available
(mirrors the P1.2e Task 4 coverage pattern — verifies registry_suffix
fires the right branch when the registry is populated).

Refs: docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.3b-pipeline-orchestrator-design.md §8
EOF
)"
```

---

## Task 5: Final acceptance + workspace quality gate

**Why:** Confirm P1.3b meets every acceptance criterion. Verification only — no commit.

**Files:** none.

- [ ] **Step 1: Workspace quality gate**

```bash
cd /home/pleclech/projects/heartbit
cargo fmt -- --check && cargo clippy --workspace --all-features -- -D warnings && cargo test --workspace --all-features 2>&1 | tail -10
```

Expected: all green. Workspace test count goes from 3944 (post-P1.3a baseline) to **3972** (28 net new tests):
- Task 1: 18 tests in `pipeline::*` (4 style_render + 8 verdicts + 6 publish_gate)
- Task 2: 3 tests in `pipeline::tests` (PipelineError display)
- Task 3: 5 tests in `pipeline::tests` (run_pipeline integration)
- Task 4: 2 tests in `heartbit-cli persona` (dispatch error paths)

- [ ] **Step 2: Verify the public surface is reachable**

```bash
cat <<'EOF' > /tmp/heartbit_ghost_p1_3b_surface_check.rs
fn _check() {
    use heartbit_ghost::pipeline::{
        PipelineConfig, PipelineError, PipelineOutput,
        FactVerdict, StyleVerdict,
        PublishGateError,
        check_publish_gate, parse_critic_verdict, parse_fact_verdict,
        render_style_profile_as_english, run_pipeline,
    };
    let _ = StyleVerdict::Pass { score: 1.0 };
    let _ = FactVerdict::Verified;
    let _ = PipelineError::NoProfileSnapshot {
        persona: String::new(),
        profiles_dir: std::path::PathBuf::new(),
    };
}
EOF
echo "(Surface check is illustrative; reachability is verified by cargo check above.)"
rm -f /tmp/heartbit_ghost_p1_3b_surface_check.rs
```

- [ ] **Step 3: Print final commit log for the branch**

```bash
git log --oneline main..feat/heartbit-ghost-p1.3b
```

Expected: 5 commits — spec doc + 4 task commits.

- [ ] **Step 4: No commit for this task**

Task 5 is verification only. The branch is ready for final review + merge. P1.3b complete; P1.3c (multi-candidate generation + judge ranking + image_generator) is the next sub-phase.

---

## Acceptance criteria

P1.3b is done when (per spec §12):

- All public types compile cleanly under `cargo check -p heartbit-ghost --all-features` and `cargo check -p heartbit-cli --all-features`
- `cargo fmt -- --check && cargo clippy --workspace --all-features -- -D warnings && cargo test --workspace --all-features` all green
- 28 net new tests pass (18 pure helpers + 3 error display + 5 integration + 2 CLI dispatch)
- `heartbit_ghost::pipeline::{PipelineConfig, PipelineOutput, PipelineError, run_pipeline, render_style_profile_as_english, StyleVerdict, FactVerdict, parse_critic_verdict, parse_fact_verdict, PublishGateError, check_publish_gate}` are reachable as public surface
- `heartbit persona run <name> --once "<topic>"` runs end-to-end against a real profile snapshot + LLM provider, prints progress to stderr and the final draft to stdout
- `heartbit-cli/persona.rs::Run { name, once }` body wires `PipelineConfig` from CLI args + env-resolved paths and calls `run_pipeline`

## Out of scope (re-stated)

- Multi-candidate generation (3-rotation + Levenshtein dedup) → P1.3c
- `judge` recipe usage → P1.3c
- `image_generator` recipe usage → P1.3c+
- `publisher` recipe usage / actual posting → P1.3d
- Telegram delivery → P1.3d
- Pick storage / few-shot exemplar retrieval → P1.3e
- Autonomy phase logic → P1.3d (Phase 0 only) + P1.4
- LLM-based content guardrails (PII, brand safety, harassment, electoral) → P1.4
- Audit log integration → P1.4
- Trigger specs → P1.4
- X "weighted character" counting in publish_gate → P1.4
- Configurable revise-loop max → defer until a real need

## Reference

- Spec: `docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.3b-pipeline-orchestrator-design.md`
- P1.3a (recipes consumed by this phase): `docs/superpowers/plans/2026-05-08-heartbit-ghost-p1.3a-sub-agent-recipes.md`
- P1.2c (MockProvider pattern): `docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.2c-llm-style-extractor-design.md`
- P1.2e (`SnapshotStore::load_latest` + `default_profiles_dir`): `docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.2e-cli-bodies-design.md`
- `AgentRunner` + `AgentRunnerBuilder`: `crates/heartbit-core/src/agent/{runner,builder}.rs`
- `ReasoningEffort` enum: `crates/heartbit-core/src/llm/types.rs:126`
- Existing CLI scaffolding: `crates/heartbit-cli/src/persona.rs::PersonaCommand::Run { name, once }`
