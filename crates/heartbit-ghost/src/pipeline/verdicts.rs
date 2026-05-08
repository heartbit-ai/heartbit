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
            return Err(serde::de::Error::unknown_variant(
                other,
                &["pass", "revise", "reject"],
            ));
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
            return Err(serde::de::Error::unknown_variant(
                other,
                &["verified", "unverifiable"],
            ));
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
