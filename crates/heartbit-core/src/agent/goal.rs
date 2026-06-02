//! [`GoalCondition`] — a persistent objective that keeps an agent working across
//! turns until an **independent** judge confirms the objective is met.
//!
//! This is heartbit's equivalent of Claude Code's `/goal`: a completion
//! condition evaluated after the agent would naturally finish a turn. A small,
//! impartial judge (a *separate* provider call — never the working agent grading
//! itself) reads the objective plus what the agent has surfaced and returns
//! met / not-met + a short reason. Not-met re-injects the reason as guidance and
//! the agent continues; met (or the continuation cap is reached) ends the run.
//!
//! ## Why an independent judge (not self-assessment)
//!
//! Agents systematically OVER-REPORT success — an agent asked "are you done?"
//! tends to say yes ("An Illusion of Progress?", arXiv:2504.01382; their WebJudge
//! shows an *independent* judge over the final state agrees with humans ~85% vs.
//! inflated self-reports). And an LLM grading its own output is biased toward it
//! ("Self-Preference Bias in LLM-as-a-Judge", arXiv:2410.21819 — persists even
//! when authorship is hidden). So the judge is a distinct provider call with an
//! impartial-evaluator prompt and no tools, mirroring `/goal`'s separate fast
//! model that "judges only what the agent has surfaced".
//!
//! ## Termination
//!
//! Bounded on both sides: the judge gates the natural-completion exit (no
//! premature stop) while `max_continuations` + the agent's own `max_turns` bound
//! the loop (no infinite loop). On an unparseable/empty judge reply the verdict
//! is treated as NOT met (keep working) — the safe direction against
//! over-reporting — still bounded by the cap.

use std::sync::Arc;

use crate::llm::types::{CompletionRequest, ContentBlock, Message, TokenUsage};
use crate::llm::{BoxedProvider, LlmProvider};

/// Default number of extra continuations granted to reach the goal after the
/// agent first naturally completes.
pub const DEFAULT_MAX_CONTINUATIONS: u32 = 8;

/// Max tokens for a judge reply — the verdict is one short line plus a reason.
const JUDGE_MAX_TOKENS: u32 = 256;

/// Cap on the transcript (tail) shown to the judge, in characters. The judge
/// must see the EVIDENCE (recent tool results), not just the agent's claim, but
/// an unbounded transcript would blow its context — so we keep the most recent
/// tail, where the demonstrating tool output and final answer live.
const MAX_TRANSCRIPT_CHARS: usize = 12_000;

/// Impartial-evaluator system prompt for the goal judge. It deliberately frames
/// the judge as a skeptical external verifier (anti over-report) that decides
/// only from the evidence the agent surfaced.
const JUDGE_SYSTEM_PROMPT: &str = "\
You are an impartial completion judge. You did NOT do the work; you only verify \
it. Given an OBJECTIVE and the WORKING TRANSCRIPT/OUTPUT an agent has produced, \
decide whether the objective is genuinely and verifiably satisfied by the \
evidence shown — not merely claimed. An agent asserting it is done is NOT \
evidence; look for the concrete result the objective requires (e.g. a passing \
test, an exit code, the requested content). Be skeptical: if the evidence is \
absent, ambiguous, or only asserted, the objective is NOT met.\n\n\
Respond with EXACTLY one verdict line, then optionally a brief reason:\n\
  GOAL_MET: YES\n\
or\n\
  GOAL_MET: NO: <one sentence on what concrete evidence is still missing>";

/// A persistent objective evaluated by an independent judge after each
/// natural completion.
#[derive(Clone)]
pub struct GoalCondition {
    objective: String,
    judge: Arc<BoxedProvider>,
    max_continuations: u32,
}

/// The judge's decision plus its reason (the reason becomes the agent's
/// next-turn guidance when the goal is not yet met).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GoalVerdict {
    /// Whether the objective is satisfied by the surfaced evidence.
    pub satisfied: bool,
    /// Short reason — what remains, when not satisfied.
    pub reason: String,
}

impl GoalCondition {
    /// Create a goal from an `objective` string and an INDEPENDENT judge
    /// provider (a separate provider/model from the working agent). Uses
    /// [`DEFAULT_MAX_CONTINUATIONS`].
    pub fn new(objective: impl Into<String>, judge: Arc<BoxedProvider>) -> Self {
        Self {
            objective: objective.into(),
            judge,
            max_continuations: DEFAULT_MAX_CONTINUATIONS,
        }
    }

    /// Override how many extra continuations are granted to reach the goal after
    /// the agent first naturally completes. `0` means "judge once, never
    /// continue" (the judge still gates the exit, but no re-prompt is issued).
    pub fn with_max_continuations(mut self, n: u32) -> Self {
        self.max_continuations = n;
        self
    }

    /// The configured continuation cap.
    pub fn max_continuations(&self) -> u32 {
        self.max_continuations
    }

    /// The objective text (used to recite the goal in continuation guidance).
    pub fn objective(&self) -> &str {
        &self.objective
    }

    /// Build the guidance message injected when the goal is not yet met: the
    /// judge's reason plus a recitation of the objective (goal-recitation keeps
    /// the long-horizon objective in context).
    pub(crate) fn continuation_message(&self, reason: &str) -> String {
        format!(
            "The objective is not yet complete. {reason}\n\nContinue working toward this \
             objective and demonstrate the concrete result it requires: {}",
            self.objective
        )
    }

    /// Ask the independent judge whether the objective is met. `transcript` is
    /// the working conversation rendered to text (including tool results — the
    /// EVIDENCE), of which the most recent [`MAX_TRANSCRIPT_CHARS`] are shown so
    /// the judge grades the demonstrated result, not merely the agent's claim.
    ///
    /// Returns the verdict plus the judge call's token usage (so the caller can
    /// account it against the run's budget). A judge or network error fails
    /// toward NOT-met (keep working) so a flaky judge never prematurely declares
    /// done — bounded by the continuation cap.
    pub(crate) async fn evaluate(&self, transcript: &str) -> (GoalVerdict, TokenUsage) {
        let evidence = tail_chars(transcript, MAX_TRANSCRIPT_CHARS);
        let user = format!(
            "OBJECTIVE:\n{}\n\nWORKING TRANSCRIPT (most recent; tool results are the \
             evidence — `[Tool result: ...]` lines show what actually happened):\n{}\n\n\
             Is the objective satisfied by the evidence above? Reply with the verdict line.",
            self.objective, evidence
        );
        let request = CompletionRequest {
            system: JUDGE_SYSTEM_PROMPT.to_string(),
            messages: vec![Message::user(user)],
            tools: Vec::new(),
            max_tokens: JUDGE_MAX_TOKENS,
            tool_choice: None,
            reasoning_effort: None,
        };
        match self.judge.complete(request).await {
            Ok(response) => (
                parse_goal_verdict(&response_text(&response.content)),
                response.usage,
            ),
            Err(e) => {
                tracing::warn!(error = %e, "goal judge call failed; treating goal as not-met");
                (
                    GoalVerdict {
                        satisfied: false,
                        reason: "judge unavailable; continuing".to_string(),
                    },
                    TokenUsage::default(),
                )
            }
        }
    }
}

/// Keep the last `max` characters of `s` (on a char boundary), prefixed with an
/// elision marker when truncated. The tail holds the most recent tool results
/// and the final answer — the judge's evidence.
fn tail_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let skip = s.chars().count() - max;
    let tail: String = s.chars().skip(skip).collect();
    format!("[... earlier turns omitted ...]\n{tail}")
}

impl std::fmt::Debug for GoalCondition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GoalCondition")
            .field("objective", &self.objective)
            .field("max_continuations", &self.max_continuations)
            .finish_non_exhaustive()
    }
}

/// Concatenate the text content blocks of a completion response.
fn response_text(content: &[ContentBlock]) -> String {
    content
        .iter()
        .filter_map(|b| match b {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Parse a judge reply for a `GOAL_MET:` verdict line. Recognizes
/// `GOAL_MET: YES` (satisfied) and `GOAL_MET: NO[: reason]` (not satisfied).
/// An unrecognized/empty reply is treated as NOT satisfied (the safe direction
/// against over-reporting), with a generic reason.
pub(crate) fn parse_goal_verdict(text: &str) -> GoalVerdict {
    for line in text.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("GOAL_MET:") {
            let rest = rest.trim();
            if rest.eq_ignore_ascii_case("yes") || rest.eq_ignore_ascii_case("yes.") {
                return GoalVerdict {
                    satisfied: true,
                    reason: String::new(),
                };
            }
            if let Some(reason) = rest
                .strip_prefix("NO:")
                .or_else(|| rest.strip_prefix("no:"))
                .or_else(|| rest.strip_prefix("No:"))
            {
                let reason = reason.trim();
                return GoalVerdict {
                    satisfied: false,
                    reason: if reason.is_empty() {
                        "objective not yet demonstrated".to_string()
                    } else {
                        reason.to_string()
                    },
                };
            }
            if rest.eq_ignore_ascii_case("no") || rest.eq_ignore_ascii_case("no.") {
                return GoalVerdict {
                    satisfied: false,
                    reason: "objective not yet demonstrated".to_string(),
                };
            }
            // A GOAL_MET: line that is neither yes nor no — keep scanning.
        }
    }
    GoalVerdict {
        satisfied: false,
        reason: "judge did not return a recognized verdict; continuing".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_yes() {
        let v = parse_goal_verdict("Looks complete.\nGOAL_MET: YES");
        assert!(v.satisfied);
        assert!(v.reason.is_empty());
    }

    #[test]
    fn parses_no_with_reason() {
        let v = parse_goal_verdict("GOAL_MET: NO: tests have not been run yet");
        assert!(!v.satisfied);
        assert_eq!(v.reason, "tests have not been run yet");
    }

    #[test]
    fn parses_bare_no() {
        let v = parse_goal_verdict("GOAL_MET: NO");
        assert!(!v.satisfied);
        assert!(!v.reason.is_empty());
    }

    #[test]
    fn verdict_value_is_case_insensitive() {
        // The `GOAL_MET:` key is matched verbatim (canonical upper form); the
        // YES/NO value is case-insensitive.
        assert!(parse_goal_verdict("GOAL_MET: yes").satisfied);
        assert!(parse_goal_verdict("GOAL_MET: YES").satisfied);
        assert!(!parse_goal_verdict("GOAL_MET: no: x").satisfied);
        assert!(!parse_goal_verdict("GOAL_MET: NO: x").satisfied);
    }

    #[test]
    fn unrecognized_reply_is_not_satisfied() {
        // The safe direction: an agent's "I'm done!" with no verdict must NOT
        // count as met (anti over-report).
        let v = parse_goal_verdict("I have completed everything successfully!");
        assert!(!v.satisfied, "no verdict line must not count as met");
        assert!(!v.reason.is_empty());
    }

    #[test]
    fn empty_reply_is_not_satisfied() {
        assert!(!parse_goal_verdict("").satisfied);
    }

    #[test]
    fn builder_defaults_and_overrides() {
        use crate::agent::test_helpers::MockProvider;
        let judge = Arc::new(BoxedProvider::new(MockProvider::new(vec![])));
        let g = GoalCondition::new("ship it", Arc::clone(&judge));
        assert_eq!(g.max_continuations(), DEFAULT_MAX_CONTINUATIONS);
        assert_eq!(g.objective(), "ship it");
        let g2 = GoalCondition::new("ship it", judge).with_max_continuations(2);
        assert_eq!(g2.max_continuations(), 2);
    }

    #[test]
    fn continuation_message_recites_objective_and_reason() {
        use crate::agent::test_helpers::MockProvider;
        let judge = Arc::new(BoxedProvider::new(MockProvider::new(vec![])));
        let g = GoalCondition::new("all tests pass", judge);
        let msg = g.continuation_message("tests are still failing");
        assert!(msg.contains("tests are still failing"));
        assert!(msg.contains("all tests pass"));
    }

    #[tokio::test]
    async fn evaluate_uses_independent_judge_and_parses_yes() {
        use crate::agent::test_helpers::MockProvider;
        let judge = Arc::new(BoxedProvider::new(MockProvider::new(vec![
            MockProvider::text_response("GOAL_MET: YES", 5, 3),
        ])));
        let g = GoalCondition::new("do the thing", judge);
        let (v, usage) = g.evaluate("I did the thing, here is the result X.").await;
        assert!(v.satisfied);
        // The judge's tokens are returned so the caller can account them.
        assert!(usage.input_tokens + usage.output_tokens > 0);
    }

    #[tokio::test]
    async fn evaluate_judge_error_fails_toward_not_met() {
        use crate::agent::test_helpers::MockProvider;
        // Empty mock → the next complete() errors ("no more mock responses").
        let judge = Arc::new(BoxedProvider::new(MockProvider::new(vec![])));
        let g = GoalCondition::new("do the thing", judge);
        let (v, usage) = g.evaluate("anything").await;
        assert!(!v.satisfied, "a judge error must not declare the goal met");
        assert_eq!(
            usage.input_tokens + usage.output_tokens,
            0,
            "a failed judge call contributes no usage"
        );
    }

    #[test]
    fn transcript_tail_keeps_recent_evidence() {
        let long: String = (0..5000).map(|i| format!("line {i}\n")).collect();
        let t = tail_chars(&long, 100);
        assert!(t.starts_with("[... earlier turns omitted ...]"));
        // The TAIL (recent evidence) is kept, not the head.
        assert!(t.contains("line 4999"));
        assert!(!t.contains("line 0\n"));
        // Short input is returned unchanged.
        assert_eq!(tail_chars("short", 100), "short");
    }

    // ===== End-to-end runner integration (mutation-verifiable) =====

    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::agent::AgentRunner;
    use crate::error::Error;
    use crate::llm::types::{
        CompletionRequest, CompletionResponse, ContentBlock, StopReason, TokenUsage,
    };

    /// A worker provider that natural-completes every turn with a fixed text and
    /// counts how many times it was invoked.
    struct CountingWorker {
        text: String,
        calls: Arc<AtomicUsize>,
    }
    impl LlmProvider for CountingWorker {
        async fn complete(&self, _request: CompletionRequest) -> Result<CompletionResponse, Error> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: self.text.clone(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage {
                    input_tokens: 1,
                    output_tokens: 1,
                    ..Default::default()
                },
                model: None,
            })
        }
        fn model_name(&self) -> Option<&str> {
            Some("worker-mock")
        }
    }

    /// A judge provider that always returns the same verdict line.
    struct FixedJudge(&'static str);
    impl LlmProvider for FixedJudge {
        async fn complete(&self, _request: CompletionRequest) -> Result<CompletionResponse, Error> {
            Ok(CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: self.0.to_string(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage {
                    input_tokens: 1,
                    output_tokens: 1,
                    ..Default::default()
                },
                model: None,
            })
        }
        fn model_name(&self) -> Option<&str> {
            Some("judge-mock")
        }
    }

    fn worker(text: &str) -> (Arc<CountingWorker>, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        let w = Arc::new(CountingWorker {
            text: text.to_string(),
            calls: Arc::clone(&calls),
        });
        (w, calls)
    }

    fn judge(verdict: &'static str) -> Arc<BoxedProvider> {
        Arc::new(BoxedProvider::new(FixedJudge(verdict)))
    }

    /// MUTATION-VERIFIED (always-satisfied): the judge confirming the goal stops
    /// the run after the FIRST natural completion. If the interception ignored
    /// the judge's "yes" and kept looping, worker calls would exceed 1.
    #[tokio::test]
    async fn goal_satisfied_stops_after_one_turn() {
        let (w, calls) = worker("I am done.");
        let runner = AgentRunner::builder(w)
            .name("w")
            .system_prompt("sp")
            .max_turns(10)
            .goal(GoalCondition::new("obj", judge("GOAL_MET: YES")).with_max_continuations(5))
            .build()
            .expect("build");
        let out = runner.execute("task").await.expect("run ok");
        assert_eq!(out.goal_met, Some(true));
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "satisfied → exactly one turn"
        );
    }

    /// MUTATION-VERIFIED (always-unsatisfied): an always-NO judge loops to the
    /// continuation cap and then returns `goal_met = Some(false)` — never
    /// infinite. With cap=2: initial completion + 2 continuations = 3 worker
    /// turns, then cap-exhausted.
    #[tokio::test]
    async fn goal_unsatisfied_loops_to_cap_then_reports_false() {
        let (w, calls) = worker("I am done, everything works!");
        let runner = AgentRunner::builder(w)
            .name("w")
            .system_prompt("sp")
            .max_turns(50)
            .goal(
                GoalCondition::new("obj", judge("GOAL_MET: NO: not yet")).with_max_continuations(2),
            )
            .build()
            .expect("build");
        let out = runner.execute("task").await.expect("run ok");
        assert_eq!(out.goal_met, Some(false), "cap exhausted → goal unmet");
        assert_eq!(
            calls.load(Ordering::SeqCst),
            3,
            "initial completion + 2 continuations"
        );
    }

    /// BEHAVIORAL INDEPENDENCE: the worker asserts it is done ("everything
    /// works!"), but the independent judge says NO — and the run CONTINUES rather
    /// than trusting the worker's self-assessment (anti over-report). Proven by
    /// the worker being invoked more than once.
    #[tokio::test]
    async fn worker_self_claim_does_not_satisfy_an_independent_no_judge() {
        let (w, calls) = worker("Done! Everything is complete and verified.");
        let runner = AgentRunner::builder(w)
            .name("w")
            .system_prompt("sp")
            .max_turns(50)
            .goal(
                GoalCondition::new("obj", judge("GOAL_MET: NO: show the test output"))
                    .with_max_continuations(1),
            )
            .build()
            .expect("build");
        let out = runner.execute("task").await.expect("run ok");
        assert_eq!(out.goal_met, Some(false));
        assert!(
            calls.load(Ordering::SeqCst) > 1,
            "the worker's 'I am done' claim must NOT stop the run when the judge says no"
        );
    }

    /// The goal continuation cap is LAYERED ON max_turns, never resetting the
    /// turn counter: with a low max_turns and an always-NO judge, the run hits
    /// MaxTurnsExceeded (a reported error exit) rather than looping forever.
    #[tokio::test]
    async fn goal_continuations_respect_max_turns() {
        let (w, _calls) = worker("not done yet");
        let runner = AgentRunner::builder(w)
            .name("w")
            .system_prompt("sp")
            .max_turns(2)
            .goal(
                GoalCondition::new("obj", judge("GOAL_MET: NO: keep going"))
                    .with_max_continuations(1000),
            )
            .build()
            .expect("build");
        let result = runner.execute("task").await;
        // execute() wraps the error in Error::WithPartialUsage; unwrap to the source.
        let err = result.expect_err("should hit max_turns, not loop forever");
        let inner = match err {
            Error::WithPartialUsage { source, .. } => *source,
            other => other,
        };
        assert!(
            matches!(inner, Error::MaxTurnsExceeded(2)),
            "goal continuations must be bounded by max_turns, got {inner:?}"
        );
    }

    /// No goal set → `goal_met` is `None` and the run completes in one turn
    /// (goal gating is inert unless a goal is configured).
    #[tokio::test]
    async fn no_goal_leaves_goal_met_none_and_one_turn() {
        let (w, calls) = worker("done");
        let runner = AgentRunner::builder(w)
            .name("w")
            .system_prompt("sp")
            .max_turns(10)
            .build()
            .expect("build");
        let out = runner.execute("task").await.expect("run ok");
        assert_eq!(out.goal_met, None);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    /// CONVERGENCE: a judge that says NO once then YES proves a continuation can
    /// actually REACH the goal (not just always-stop / always-loop). Worker
    /// completes twice; the second judge verdict ends the run met.
    #[tokio::test]
    async fn goal_converges_when_judge_flips_to_yes() {
        use crate::agent::test_helpers::MockProvider;
        let (w, calls) = worker("working on it");
        // Judge: NO on the first completion, YES on the second.
        let judge_p = Arc::new(BoxedProvider::new(MockProvider::new(vec![
            MockProvider::text_response("GOAL_MET: NO: not yet", 1, 1),
            MockProvider::text_response("GOAL_MET: YES", 1, 1),
        ])));
        let runner = AgentRunner::builder(w)
            .name("w")
            .system_prompt("sp")
            .max_turns(10)
            .goal(GoalCondition::new("obj", judge_p).with_max_continuations(5))
            .build()
            .expect("build");
        let out = runner.execute("task").await.expect("run ok");
        assert_eq!(
            out.goal_met,
            Some(true),
            "the second verdict reaches the goal"
        );
        assert_eq!(
            calls.load(Ordering::SeqCst),
            2,
            "1 initial completion (judged NO) + 1 continuation (judged YES)"
        );
    }

    /// `max_continuations = 0`: the judge gates the exit ONCE but never
    /// re-prompts. An unmet goal returns immediately with `goal_met = Some(false)`
    /// and exactly one turn.
    #[tokio::test]
    async fn zero_continuations_judges_once_no_reprompt() {
        let (w, calls) = worker("I am done");
        let runner = AgentRunner::builder(w)
            .name("w")
            .system_prompt("sp")
            .max_turns(10)
            .goal(GoalCondition::new("obj", judge("GOAL_MET: NO: nope")).with_max_continuations(0))
            .build()
            .expect("build");
        let out = runner.execute("task").await.expect("run ok");
        assert_eq!(out.goal_met, Some(false));
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "judged once, no continuation"
        );
    }

    /// A goal and structured output are mutually exclusive at build time (a goal
    /// gates the text completion exit, which structured output bypasses).
    #[test]
    fn goal_with_structured_schema_is_rejected_at_build() {
        let (w, _calls) = worker("x");
        let result = AgentRunner::builder(w)
            .name("w")
            .system_prompt("sp")
            .structured_schema(serde_json::json!({"type": "object"}))
            .goal(GoalCondition::new("obj", judge("GOAL_MET: YES")))
            .build();
        // `AgentRunner` is not `Debug`, so match rather than `unwrap_err`.
        let err = match result {
            Err(e) => e,
            Ok(_) => panic!("goal + structured_schema must be rejected at build"),
        };
        assert!(err.to_string().contains("mutually exclusive"), "got: {err}");
    }
}
