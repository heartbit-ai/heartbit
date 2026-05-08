//! Structured verdict parsing for `style_critic`, `fact_check`, and `judge`.

use serde::Deserialize;
use thiserror::Error;

/// Errors raised by the three verdict parsers. Wraps `serde_json::Error`
/// so it doesn't leak into `heartbit-ghost`'s public API.
#[derive(Debug, Error)]
pub enum VerdictParseError {
    /// `style_critic` returned malformed JSON or an unknown verdict variant.
    #[error("critic verdict parse: {source}")]
    Critic {
        /// Underlying serde error.
        #[source]
        source: serde_json::Error,
        /// The raw critic output that failed to parse, kept for diagnostics.
        raw: String,
    },

    /// `fact_check` returned malformed JSON or an unknown verdict variant.
    #[error("fact_check verdict parse: {source}")]
    Fact {
        /// Underlying serde error.
        #[source]
        source: serde_json::Error,
        /// The raw fact_check output that failed to parse, kept for diagnostics.
        raw: String,
    },

    /// `judge` returned malformed JSON.
    #[error("judge verdict parse: {source}")]
    Judge {
        /// Underlying serde error.
        #[source]
        source: serde_json::Error,
        /// The raw judge output that failed to parse, kept for diagnostics.
        raw: String,
    },

    /// `judge` returned a `chosen_index` outside the valid range `[0, n)`.
    #[error("judge chose index {chosen_index} out of range [0, {n})")]
    JudgeChoiceOutOfRange {
        /// The out-of-range index the judge returned.
        chosen_index: usize,
        /// The number of candidates the judge was given.
        n: usize,
        /// The raw judge output, kept for diagnostics.
        raw: String,
    },
}

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
        /// Short feedback string fed into the writer's next user message.
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
    /// Returns the 0.0..=1.0 voice match score for any verdict variant.
    pub fn score(&self) -> f64 {
        match self {
            StyleVerdict::Pass { score }
            | StyleVerdict::Revise { score, .. }
            | StyleVerdict::Reject { score, .. } => *score,
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

/// Judge verdict — picks one of N candidate drafts.
#[derive(Debug, Clone, PartialEq)]
pub struct JudgeVerdict {
    /// Index into the input candidate slice, validated `0..n` by
    /// [`parse_judge_verdict`].
    pub chosen_index: usize,
    /// Short reasoning string from the judge.
    pub reasoning: String,
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

#[derive(Debug, Deserialize)]
struct JudgeRaw {
    chosen_index: usize,
    reasoning: String,
}

/// Parse the critic's raw output as JSON.
pub fn parse_critic_verdict(raw: &str) -> Result<StyleVerdict, VerdictParseError> {
    let unfenced = strip_fence(raw.trim());
    let parsed: CriticRaw =
        serde_json::from_str(unfenced).map_err(|source| VerdictParseError::Critic {
            source,
            raw: raw.to_string(),
        })?;
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
            let source = serde::de::Error::unknown_variant(other, &["pass", "revise", "reject"]);
            return Err(VerdictParseError::Critic {
                source,
                raw: raw.to_string(),
            });
        }
    };
    Ok(verdict)
}

/// Parse the fact_check raw output as JSON.
pub fn parse_fact_verdict(raw: &str) -> Result<FactVerdict, VerdictParseError> {
    let unfenced = strip_fence(raw.trim());
    let parsed: FactRaw =
        serde_json::from_str(unfenced).map_err(|source| VerdictParseError::Fact {
            source,
            raw: raw.to_string(),
        })?;
    let verdict = match parsed.verdict.as_str() {
        "verified" => FactVerdict::Verified,
        "unverifiable" => FactVerdict::Unverifiable {
            reason: parsed.reason.unwrap_or_else(|| "unspecified".to_string()),
        },
        other => {
            let source = serde::de::Error::unknown_variant(other, &["verified", "unverifiable"]);
            return Err(VerdictParseError::Fact {
                source,
                raw: raw.to_string(),
            });
        }
    };
    Ok(verdict)
}

/// Parse the judge's raw output as JSON. Validates `chosen_index` against
/// `n` (the number of candidates the judge was given).
pub fn parse_judge_verdict(raw: &str, n: usize) -> Result<JudgeVerdict, VerdictParseError> {
    let unfenced = strip_fence(raw.trim());
    let parsed: JudgeRaw =
        serde_json::from_str(unfenced).map_err(|source| VerdictParseError::Judge {
            source,
            raw: raw.to_string(),
        })?;
    if parsed.chosen_index >= n {
        return Err(VerdictParseError::JudgeChoiceOutOfRange {
            chosen_index: parsed.chosen_index,
            n,
            raw: raw.to_string(),
        });
    }
    Ok(JudgeVerdict {
        chosen_index: parsed.chosen_index,
        reasoning: parsed.reasoning,
    })
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

    // ---- StyleVerdict (existing tests, updated for VerdictParseError) ----

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
    fn parse_critic_malformed_returns_critic_variant() {
        let raw = "definitely not json";
        let err = parse_critic_verdict(raw).unwrap_err();
        match err {
            VerdictParseError::Critic { raw: r, .. } => assert_eq!(r, "definitely not json"),
            other => panic!("expected Critic variant, got: {other:?}"),
        }
    }

    // ---- FactVerdict (existing tests, updated for VerdictParseError) ----

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
    fn parse_fact_unknown_verdict_returns_fact_variant() {
        let raw = r#"{"verdict": "maybe"}"#;
        let err = parse_fact_verdict(raw).unwrap_err();
        match err {
            VerdictParseError::Fact { raw: r, .. } => assert_eq!(r, r#"{"verdict": "maybe"}"#),
            other => panic!("expected Fact variant, got: {other:?}"),
        }
    }

    // ---- JudgeVerdict (NEW) ----

    #[test]
    fn parse_judge_happy_path() {
        let raw = r#"{"chosen_index": 1, "reasoning": "candidate 1 has more specific examples"}"#;
        let v = parse_judge_verdict(raw, 3).unwrap();
        assert_eq!(
            v,
            JudgeVerdict {
                chosen_index: 1,
                reasoning: "candidate 1 has more specific examples".to_string(),
            }
        );
    }

    #[test]
    fn parse_judge_strips_markdown_fence() {
        let raw = "```json\n{\"chosen_index\": 0, \"reasoning\": \"first one\"}\n```";
        let v = parse_judge_verdict(raw, 2).unwrap();
        assert_eq!(v.chosen_index, 0);
    }

    #[test]
    fn parse_judge_out_of_range_returns_specific_variant() {
        let raw = r#"{"chosen_index": 5, "reasoning": "anything"}"#;
        let err = parse_judge_verdict(raw, 3).unwrap_err();
        match err {
            VerdictParseError::JudgeChoiceOutOfRange {
                chosen_index, n, ..
            } => {
                assert_eq!(chosen_index, 5);
                assert_eq!(n, 3);
            }
            other => panic!("expected JudgeChoiceOutOfRange, got: {other:?}"),
        }
    }

    #[test]
    fn parse_judge_malformed_returns_judge_variant() {
        let raw = "not json";
        let err = parse_judge_verdict(raw, 3).unwrap_err();
        match err {
            VerdictParseError::Judge { raw: r, .. } => assert_eq!(r, "not json"),
            other => panic!("expected Judge variant, got: {other:?}"),
        }
    }
}
