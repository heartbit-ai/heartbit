//! The `intake` workflow recipe: the completion-loop FRONT HALF. Turn a raw
//! feature request into observable ACCEPTANCE CRITERIA (stage 1) plus the
//! load-bearing GAPS the request leaves unspecified (stage 2), classified into
//! safe-to-assume vs ask-the-user. Output is a markdown block the entry agent
//! consumes in-context — the recipe never asks the user itself (flat hierarchy:
//! `OnQuestion` belongs to the entry agent, which owns the user channel).
//!
//! Two ordered LLM leaves, run sequentially (this is the conceptual "pipeline"
//! shape — NOT the [`pipeline`](super::flow::pipeline) combinator, which fans N
//! items through stages and would silently drop a stage failure; here a single
//! request flows through two stages exactly like `deep_research`'s talk-only
//! stages). The gap leaf's reply is parsed defensively: a malformed critic JSON
//! degrades to zero questions plus the raw text as one assumption, and NEVER
//! fails the recipe.

use std::sync::Arc;

use serde_json::json;

use super::flow::agent;
use super::workflow_tool::WorkflowRecipe;
use crate::error::Error;
use crate::tool::builtins::Question;

/// System prompt for stage 1 (criteria extraction). Phrased so the model emits
/// observable, testable behavior — never implementation detail.
const CRITERIA_PROMPT: &str = "\
You are a requirements analyst. From the user's feature request below, produce \
3 to 8 ACCEPTANCE CRITERIA as markdown bullets. Each criterion MUST be phrased \
as OBSERVABLE BEHAVIOR — something a human or a command can verify (\"a human or \
a command can verify X\"), concrete and testable. Do NOT mention implementation \
details (no file names, function names, libraries, or internal design). Output \
ONLY the bulleted criteria, one per line, nothing else.\n\nFeature request:\n";

/// System prompt for stage 2 (gap elicitation — the INVERSE/completeness
/// critic). Asks what the request implies but leaves unspecified, classifies
/// each gap, and emits STRICT JSON whose `questions` match the `question`
/// builtin's input schema exactly.
const GAP_PROMPT: &str = "\
You are a completeness critic. The question is NOT \"is this done?\" but \"what \
does this request IMPLY yet leave UNSPECIFIED?\". Examine the feature request \
and its extracted acceptance criteria below, then surface every gap.\n\n\
Classify each gap:\n\
- ALREADY SPECIFIED in the request (a stated location, language, scope, \
constraint): NOT a gap — never raise it as a question or re-litigate it.\n\
- HIGH-GUESS-RATE (output format / naming / wording): safe to assume — state a \
reasonable default as an ASSUMPTION, do NOT ask.\n\
- LOW-GUESS-RATE (scope / risk / user intent / destructive-vs-additive): asking \
beats guessing — raise it as a QUESTION.\n\
Every option you offer MUST honor every stated constraint — never include a \
choice that contradicts the request.\n\n\
Output STRICT JSON ONLY (no prose, no markdown fences), exactly this shape:\n\
{\"assumptions\": [\"...\"], \"questions\": [{\"question\": \"...\", \"header\": \
\"...\", \"options\": [{\"label\": \"...\", \"description\": \"...\"}], \
\"multiple\": false}]}\n\
Rules: 0 to 4 questions, ONLY for low-guess gaps; each question has a short \
header (<=12 chars) and 2 to 4 options; `multiple` is a boolean. If nothing is \
unspecified, return empty arrays.\n\n";

/// Parsed shape of the gap critic's STRICT JSON reply.
#[derive(serde::Deserialize, Default)]
struct GapReply {
    #[serde(default)]
    assumptions: Vec<String>,
    #[serde(default)]
    questions: Vec<Question>,
}

/// Render the final markdown block the entry agent consumes. `questions` is
/// pretty-printed JSON (or the literal `(none — proceed)` when empty).
fn render(criteria: &str, assumptions: &[String], questions: &[Question]) -> String {
    let criteria = {
        let t = criteria.trim();
        if t.is_empty() {
            "(no criteria extracted)"
        } else {
            t
        }
    };
    let assumptions_block = if assumptions.is_empty() {
        "- (none)".to_string()
    } else {
        assumptions
            .iter()
            .map(|a| format!("- {}", a.trim()))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let questions_block = if questions.is_empty() {
        "(none — proceed)".to_string()
    } else {
        // Pretty-print the typed questions back to JSON — guarantees the shape
        // matches the `question` builtin's input exactly.
        serde_json::to_string_pretty(questions).unwrap_or_else(|_| "(none — proceed)".to_string())
    };
    format!(
        "## Acceptance criteria\n{criteria}\n\n\
         ## Assumptions (proceeding without asking)\n{assumptions_block}\n\n\
         ## Questions for the user (ask via the question tool BEFORE building)\n\
         {questions_block}"
    )
}

/// Build the `intake` recipe: criteria extraction → gap elicitation, returning
/// the markdown brief the entry agent reads in-context.
pub(crate) fn recipe() -> WorkflowRecipe {
    WorkflowRecipe {
        name: "intake".into(),
        description: "Turn a feature request into observable acceptance criteria \
                      + load-bearing gaps before building — the completion-loop \
                      front half."
            .into(),
        args_schema: json!({
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "the user's feature request, verbatim"
                }
            },
            "required": ["request"]
        }),
        run: Arc::new(|ctx, args| {
            Box::pin(async move {
                let request = args
                    .get("request")
                    .and_then(|v| v.as_str())
                    .map(str::trim)
                    .filter(|r| !r.is_empty())
                    .ok_or_else(|| Error::Agent("intake: 'request' is required".into()))?
                    .to_string();

                // Stage 1 — criteria extraction. Cheap classification → "fast"
                // role (degrades to the default provider without a factory).
                let criteria = agent(&ctx, format!("{CRITERIA_PROMPT}{request}"))
                    .label("intake:criteria")
                    .model("fast")
                    .run()
                    .await?
                    .unwrap_or_default();

                // Stage 2 — gap elicitation (completeness critic). Parsed
                // defensively in pure Rust AFTER the leaf: a provider error
                // fails the recipe (like deep_research), but a MALFORMED reply
                // degrades to zero questions + the raw text as one assumption.
                let gap_text = agent(
                    &ctx,
                    format!(
                        "{GAP_PROMPT}Feature request:\n{request}\n\n\
                         Extracted acceptance criteria:\n{criteria}"
                    ),
                )
                .label("intake:gaps")
                .model("fast")
                .run()
                .await?
                .unwrap_or_default();

                let (assumptions, questions) = parse_gaps(&gap_text);
                Ok(render(&criteria, &assumptions, &questions))
            })
        }),
    }
}

/// Parse the gap critic's reply into `(assumptions, questions)`. On unparseable
/// JSON, degrade gracefully: zero questions, and the raw (trimmed) reply
/// surfaced as the single assumption — NEVER an error.
fn parse_gaps(text: &str) -> (Vec<String>, Vec<Question>) {
    let trimmed = text.trim();
    match parse_gap_json(trimmed) {
        Some(reply) => {
            let questions = reply
                .questions
                .into_iter()
                .filter(valid_question)
                .take(4)
                .collect();
            (reply.assumptions, questions)
        }
        None => {
            // Malformed (or empty) — degrade. Surface whatever the critic said
            // (if anything) as an assumption so nothing is silently lost.
            let assumption = if trimmed.is_empty() {
                Vec::new()
            } else {
                vec![trimmed.to_string()]
            };
            (assumption, Vec::new())
        }
    }
}

/// Try to extract the gap reply from `text`, tolerating a model that wraps the
/// JSON in prose or markdown fences by slicing the outermost `{ .. }`.
fn parse_gap_json(text: &str) -> Option<GapReply> {
    if let Ok(reply) = serde_json::from_str::<GapReply>(text) {
        return Some(reply);
    }
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    if end <= start {
        return None;
    }
    serde_json::from_str::<GapReply>(&text[start..=end]).ok()
}

/// A question is only usable if the `question` builtin would accept it: a
/// non-empty prompt and at least 2 options (cap the options at 4).
fn valid_question(q: &Question) -> bool {
    !q.question.trim().is_empty() && q.options.len() >= 2
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::BoxedProvider;
    use crate::agent::flow::WorkflowCtx;
    use crate::llm::LlmProvider;
    use crate::llm::types::{
        CompletionRequest, CompletionResponse, ContentBlock, StopReason, TokenUsage,
    };
    use std::sync::{Arc, Mutex};

    /// Content-routed mock: stages run in a fixed order here, but routing by
    /// PROMPT substring (never call index) keeps the tests robust and mirrors
    /// `deep_research`'s harness. The criteria prompt and the gap prompt have
    /// disjoint marker phrases, so the gap stage (which embeds the criteria
    /// OUTPUT) never mis-routes to the criteria reply.
    struct RoutedProvider {
        criteria_reply: String,
        gap_reply: String,
        captured: Arc<Mutex<Vec<CompletionRequest>>>,
    }

    impl RoutedProvider {
        fn text(t: &str) -> CompletionResponse {
            CompletionResponse {
                content: vec![ContentBlock::Text { text: t.into() }],
                stop_reason: StopReason::EndTurn,
                reasoning: None,
                usage: TokenUsage::default(),
                model: None,
            }
        }
    }

    impl LlmProvider for RoutedProvider {
        async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, Error> {
            let prompt: String = request
                .messages
                .iter()
                .flat_map(|m| m.content.iter())
                .filter_map(|b| match b {
                    ContentBlock::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect();
            self.captured.lock().expect("capture lock").push(request);
            // "completeness critic" is unique to the gap stage; the criteria
            // stage uses "requirements analyst". Order doesn't matter.
            let reply = if prompt.contains("completeness critic") {
                self.gap_reply.as_str()
            } else if prompt.contains("requirements analyst") {
                self.criteria_reply.as_str()
            } else {
                "unexpected prompt"
            };
            Ok(Self::text(reply))
        }
    }

    async fn run_recipe(
        criteria_reply: &str,
        gap_reply: &str,
        args: serde_json::Value,
    ) -> Result<String, Error> {
        let captured = Arc::new(Mutex::new(Vec::new()));
        let provider = Arc::new(BoxedProvider::from_arc(Arc::new(RoutedProvider {
            criteria_reply: criteria_reply.to_string(),
            gap_reply: gap_reply.to_string(),
            captured,
        })));
        let ctx = WorkflowCtx::builder(provider).build().expect("ctx");
        let r = recipe();
        (r.run)(ctx, args).await
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn extracts_observable_acceptance_criteria() {
        let criteria = "- the /health endpoint returns 200\n\
                        - a command can curl /health and see OK\n\
                        - the response body contains a status field";
        let out = run_recipe(
            criteria,
            r#"{"assumptions": [], "questions": []}"#,
            serde_json::json!({"request": "add a /health endpoint"}),
        )
        .await
        .unwrap();
        assert!(
            out.contains("## Acceptance criteria"),
            "missing criteria header: {out}"
        );
        assert!(out.contains("the /health endpoint returns 200"), "{out}");
        assert!(
            out.contains("the response body contains a status field"),
            "{out}"
        );
        // No questions/assumptions → the proceed sentinel.
        assert!(out.contains("(none — proceed)"), "{out}");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn elicits_missing_low_guess_requirements() {
        let gap = r#"{
            "assumptions": ["output is rendered as markdown bullets"],
            "questions": [{
                "question": "Should deleting a record be destructive or soft-delete?",
                "header": "Delete",
                "options": [
                    {"label": "Hard delete", "description": "remove the row permanently"},
                    {"label": "Soft delete", "description": "mark a deleted flag"}
                ],
                "multiple": false
            }]
        }"#;
        let out = run_recipe(
            "- a record can be deleted",
            gap,
            serde_json::json!({"request": "let users delete records"}),
        )
        .await
        .unwrap();
        // The question text surfaces under the questions section.
        assert!(
            out.contains("Should deleting a record be destructive or soft-delete?"),
            "question text missing: {out}"
        );
        // The assumption surfaces under its own section.
        assert!(
            out.contains("## Assumptions (proceeding without asking)"),
            "{out}"
        );
        assert!(
            out.contains("output is rendered as markdown bullets"),
            "assumption missing: {out}"
        );
        // The options come through (pretty-printed JSON).
        assert!(out.contains("Soft delete"), "options missing: {out}");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn malformed_critic_json_degrades_gracefully() {
        let garbage = "I think you should probably ask about auth, maybe? not sure.";
        let out = run_recipe(
            "- the feature works",
            garbage,
            serde_json::json!({"request": "build a thing"}),
        )
        .await
        .unwrap();
        // The recipe still succeeds; no questions are emitted.
        assert!(out.contains("(none — proceed)"), "{out}");
        // The garbage is surfaced (never lost) as an assumption.
        assert!(
            out.contains("I think you should probably ask about auth"),
            "raw critic text must surface as an assumption: {out}"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn empty_request_is_rejected() {
        let err = run_recipe("-", "{}", serde_json::json!({"request": "   "}))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("request"), "{err}");

        let err2 = run_recipe("-", "{}", serde_json::json!({}))
            .await
            .unwrap_err();
        assert!(err2.to_string().contains("request"), "{err2}");
    }

    #[test]
    fn intake_recipe_registered_in_default_registry() {
        let reg = crate::agent::workflow_tool::default_registry();
        assert!(reg.get("intake").is_some(), "intake must be registered");
        let meta = reg.meta();
        assert!(
            meta.iter().any(|(n, _)| n == "intake"),
            "intake must appear in registry meta"
        );
    }

    #[test]
    fn questions_with_too_few_options_are_dropped() {
        // A low-guess question with a single option can't be asked → filtered.
        let (assumptions, questions) = parse_gaps(
            r#"{"assumptions": ["x"], "questions": [
                {"question": "Q?", "header": "H", "options": [{"label": "a", "description": "d"}], "multiple": false}
            ]}"#,
        );
        assert_eq!(assumptions, vec!["x".to_string()]);
        assert!(questions.is_empty(), "1-option question must be dropped");
    }

    #[test]
    fn extra_questions_are_capped_at_four() {
        let one = r#"{"question":"Q?","header":"H","options":[{"label":"a","description":"d"},{"label":"b","description":"e"}],"multiple":false}"#;
        let json =
            format!(r#"{{"assumptions": [], "questions": [{one},{one},{one},{one},{one},{one}]}}"#);
        let (_a, questions) = parse_gaps(&json);
        assert_eq!(questions.len(), 4, "questions cap at 4");
    }

    #[test]
    fn json_wrapped_in_prose_is_recovered() {
        let (assumptions, questions) = parse_gaps(
            "Here is my analysis:\n{\"assumptions\": [\"a1\"], \"questions\": []}\nDone.",
        );
        assert_eq!(assumptions, vec!["a1".to_string()]);
        assert!(questions.is_empty());
    }
}
