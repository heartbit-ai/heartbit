//! Verifier-guided test-time compute (best-of-N with a reward/verifier).
//!
//! Scaling inference reliably means generating multiple candidate solutions and
//! selecting the best by a *verifier* (an outcome/process reward signal) rather
//! than trusting a single sample. This module provides the seam:
//!
//! - [`Verifier`] — score a candidate in `[0, 1]` against the task.
//! - [`select_best`] — verifier-best-of-N selection over candidate strings.
//! - [`LlmVerifier`] — a concrete verifier that asks a (typically cheap) judge
//!   model to score a candidate, parsing a `SCORE: <0-100>` line.
//!
//! The existing voting/debate/mixture combinators pick by majority/aggregation;
//! a `Verifier` adds *graded* selection and a place to plug a learned process
//! reward model. Independent of any single agent so it composes with all of them.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::error::Error;
use crate::llm::LlmProvider;
use crate::llm::types::{CompletionRequest, ContentBlock, Message, Role};

/// Scores a candidate solution against the original task. Higher is better;
/// scores are clamped to `[0, 1]` by [`select_best`].
pub trait Verifier: Send + Sync {
    /// Score `candidate` as a solution to `task`. `[0, 1]`, higher = better.
    fn score<'a>(
        &'a self,
        task: &'a str,
        candidate: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<f64, Error>> + Send + 'a>>;
}

/// The winning candidate of a verifier-guided best-of-N selection.
#[derive(Debug, Clone)]
pub struct Selected {
    /// Index of the winning candidate in the input slice.
    pub index: usize,
    /// The winning candidate's verifier score, in `[0, 1]`.
    pub score: f64,
    /// The winning candidate text.
    pub candidate: String,
}

/// Select the highest-scoring candidate via `verifier` (verifier-best-of-N).
///
/// Scores are clamped to `[0, 1]`. Ties keep the earliest (lowest-index)
/// candidate, so a stable verifier yields a deterministic winner. Returns an
/// error if `candidates` is empty or every score errors; a single scoring error
/// is treated as score 0 (that candidate simply can't win) rather than aborting.
pub async fn select_best<V: Verifier + ?Sized>(
    task: &str,
    candidates: &[String],
    verifier: &V,
) -> Result<Selected, Error> {
    if candidates.is_empty() {
        return Err(Error::Agent(
            "select_best: no candidates to choose from".into(),
        ));
    }
    let mut best: Option<Selected> = None;
    let mut any_ok = false;
    for (index, candidate) in candidates.iter().enumerate() {
        let score = match verifier.score(task, candidate).await {
            Ok(s) => {
                any_ok = true;
                s.clamp(0.0, 1.0)
            }
            Err(e) => {
                tracing::warn!(index, error = %e, "verifier scoring failed; treating as 0");
                0.0
            }
        };
        // Strictly-greater keeps the earliest candidate on ties.
        if best.as_ref().is_none_or(|b| score > b.score) {
            best = Some(Selected {
                index,
                score,
                candidate: candidate.clone(),
            });
        }
    }
    match best {
        Some(sel) if any_ok => Ok(sel),
        // No candidate scored successfully — fall back to the first rather than
        // erroring, so a flaky verifier never loses an answer entirely.
        _ => Ok(Selected {
            index: 0,
            score: 0.0,
            candidate: candidates[0].clone(),
        }),
    }
}

const VERIFIER_SYSTEM: &str = "You are a strict solution VERIFIER. Given a TASK and \
a CANDIDATE solution, judge how well the candidate solves the task: correctness, \
completeness, and whether it actually answers what was asked. Reply with exactly \
one line: 'SCORE: N' where N is an integer 0-100 (100 = a perfect solution, 0 = \
wrong or irrelevant). Output only that line.";

/// A [`Verifier`] backed by an LLM judge. Use a cheap, separate model from the
/// one that generated the candidates.
pub struct LlmVerifier<P: LlmProvider> {
    provider: Arc<P>,
    max_tokens: u32,
}

impl<P: LlmProvider> LlmVerifier<P> {
    /// Build an LLM verifier over `provider`.
    pub fn new(provider: Arc<P>) -> Self {
        Self {
            provider,
            max_tokens: 32,
        }
    }
}

/// Parse a `SCORE: N` (0-100) line into a `[0, 1]` score. Tolerant of surrounding
/// text and casing; returns 0.0 when no score is found.
pub(crate) fn parse_score(text: &str) -> f64 {
    for line in text.lines() {
        let upper = line.to_uppercase();
        if let Some(rest) = upper.split("SCORE:").nth(1) {
            let digits: String = rest
                .trim()
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect();
            if let Ok(n) = digits.parse::<u32>() {
                return (n.min(100) as f64) / 100.0;
            }
        }
    }
    0.0
}

impl<P: LlmProvider> Verifier for LlmVerifier<P> {
    fn score<'a>(
        &'a self,
        task: &'a str,
        candidate: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<f64, Error>> + Send + 'a>> {
        Box::pin(async move {
            let user = format!("TASK:\n{task}\n\nCANDIDATE:\n{candidate}");
            let request = CompletionRequest {
                system: VERIFIER_SYSTEM.to_string(),
                messages: vec![Message {
                    role: Role::User,
                    content: vec![ContentBlock::Text { text: user }],
                }],
                tools: Vec::new(),
                max_tokens: self.max_tokens,
                tool_choice: None,
                reasoning_effort: None,
            };
            let response = self.provider.complete(request).await?;
            Ok(parse_score(&response.text()))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::test_helpers::MockProvider;

    struct FixedVerifier(Vec<f64>);
    impl Verifier for FixedVerifier {
        fn score<'a>(
            &'a self,
            _task: &'a str,
            candidate: &'a str,
        ) -> Pin<Box<dyn Future<Output = Result<f64, Error>> + Send + 'a>> {
            // Score = the value parsed from the candidate (e.g. "c2" → index lookup).
            let idx: usize = candidate.trim_start_matches('c').parse().unwrap_or(0);
            let s = self.0.get(idx).copied().unwrap_or(0.0);
            Box::pin(async move { Ok(s) })
        }
    }

    #[tokio::test]
    async fn select_best_picks_highest_score() {
        let candidates = vec!["c0".to_string(), "c1".to_string(), "c2".to_string()];
        let verifier = FixedVerifier(vec![0.2, 0.9, 0.5]);
        let sel = select_best("task", &candidates, &verifier).await.unwrap();
        assert_eq!(sel.index, 1);
        assert!((sel.score - 0.9).abs() < 1e-9);
        assert_eq!(sel.candidate, "c1");
    }

    #[tokio::test]
    async fn select_best_ties_keep_earliest() {
        let candidates = vec!["c0".to_string(), "c1".to_string()];
        let verifier = FixedVerifier(vec![0.7, 0.7]);
        let sel = select_best("t", &candidates, &verifier).await.unwrap();
        assert_eq!(sel.index, 0);
    }

    #[tokio::test]
    async fn select_best_empty_is_error() {
        let verifier = FixedVerifier(vec![]);
        assert!(select_best("t", &[], &verifier).await.is_err());
    }

    #[test]
    fn parse_score_handles_variants() {
        assert!((parse_score("SCORE: 80") - 0.80).abs() < 1e-9);
        assert!((parse_score("blah\nscore: 100 great") - 1.0).abs() < 1e-9);
        assert!((parse_score("SCORE: 250") - 1.0).abs() < 1e-9); // clamped to 100
        assert_eq!(parse_score("no score here"), 0.0);
    }

    #[tokio::test]
    async fn llm_verifier_scores_via_judge() {
        let provider = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "SCORE: 75",
            10,
            2,
        )]));
        let verifier = LlmVerifier::new(provider);
        let s = verifier.score("task", "candidate").await.unwrap();
        assert!((s - 0.75).abs() < 1e-9);
    }
}
