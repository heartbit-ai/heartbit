//! Task-completion judge (capability 15 of the browser-bot spec).
//!
//! "An Illusion of Progress" (Online-Mind2Web, arXiv:2504.01382) found the
//! dominant web-agent failure is declaring success on an unfinished task — the
//! agent fills part of a form, never confirms the final Submit, and reports
//! "done". The fix is an independent **completion judge**: given the original
//! task, the concrete success criteria ("key points"), the action history, and
//! the final page state, a separate (cheap) model decides whether the task was
//! actually accomplished. The paper's WebJudge reaches ~85.7% agreement with
//! human evaluators and is the best automated web-trajectory oracle to date.
//!
//! Crucially, the deep-research synthesis (see the project's SOTA notes) said to
//! build the **thin, DOM-based** version of this — NOT WebJudge's screenshot-
//! ranking pipeline, which exists only because vision agents lack a structured
//! observation. We are a11y-tree/`uid` grounded, so the final *snapshot text*
//! already carries the ground truth (an element's `value="..."`, a success
//! banner, a changed URL). That means **no image bridge is required** for a
//! strong v1 — a major simplification over the original spec, which had this
//! capability blocked on surfacing image tool-results through the MCP layer.
//!
//! This module is the deterministic core: a verdict type, a fail-open parser
//! (mirroring [`crate::LlmJudgeGuardrail`]'s `VERDICT:` convention), and a
//! prompt assembler. The async "call the judge model" shell is layered on later,
//! exactly like [`super::settle::settle`] over [`super::settle::step`].

/// The judge's decision about whether the task was accomplished.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompletionVerdict {
    /// The task's success criteria are all satisfied by the final state.
    Complete,
    /// The task is not finished; the reason names the missing criterion so the
    /// loop can replan toward it (empty string if the judge gave none).
    Incomplete(String),
    /// The judge could not tell from the evidence (treated as not-done by the
    /// loop, but distinct from a confident `Incomplete` for logging/metrics).
    Uncertain(String),
}

impl CompletionVerdict {
    /// Whether this verdict means the task is finished. Only [`Self::Complete`]
    /// is terminal-success; `Incomplete`/`Uncertain` both mean "keep going".
    pub fn is_complete(&self) -> bool {
        matches!(self, CompletionVerdict::Complete)
    }
}

/// Parse a judge model's free-text reply into a [`CompletionVerdict`].
///
/// Looks (case-insensitively, scanning every line so reasoning before the line
/// is fine) for one of:
/// - `VERDICT: COMPLETE`
/// - `VERDICT: INCOMPLETE: <reason>`
/// - `VERDICT: UNCERTAIN: <reason>`
///
/// Returns `None` when no recognizable verdict line is present. Callers
/// **fail-open toward not-done**: a `None` (or `Uncertain`) must never be
/// treated as success — an unparseable judge reply means "we don't know it's
/// done", so the safe action is to keep working, not to stop. (Contrast the
/// safety judge, which fails open toward *allow*; a completion judge that failed
/// toward *done* would reintroduce the very illusion-of-progress bug it exists
/// to catch.)
pub fn parse_completion_verdict(text: &str) -> Option<CompletionVerdict> {
    for line in text.lines() {
        let trimmed = line.trim();
        let Some(rest) = strip_prefix_ci(trimmed, "VERDICT:") else {
            continue;
        };
        let rest = rest.trim();

        if rest.eq_ignore_ascii_case("COMPLETE") {
            return Some(CompletionVerdict::Complete);
        }
        if let Some(reason) = strip_prefix_ci(rest, "INCOMPLETE:") {
            return Some(CompletionVerdict::Incomplete(reason.trim().to_string()));
        }
        if let Some(reason) = strip_prefix_ci(rest, "UNCERTAIN:") {
            return Some(CompletionVerdict::Uncertain(reason.trim().to_string()));
        }
        // A `VERDICT:` line that doesn't match a known shape — keep scanning in
        // case a well-formed one appears later.
    }
    None
}

/// Case-insensitive [`str::strip_prefix`].
///
/// Compares on BYTES so a multibyte char straddling `prefix.len()` cannot panic
/// — `s[..prefix.len()]` (a fixed byte index) would split the codepoint of e.g.
/// `"VERDICTé"`. A match implies the first `prefix.len()` bytes are ASCII (they
/// equal-ignore-case an ASCII prefix), so `s[prefix.len()..]` lands on a char
/// boundary and is safe. Judge replies are untrusted, so this must never panic.
fn strip_prefix_ci<'a>(s: &'a str, prefix: &str) -> Option<&'a str> {
    let (sb, pb) = (s.as_bytes(), prefix.as_bytes());
    if sb.len() >= pb.len() && sb[..pb.len()].eq_ignore_ascii_case(pb) {
        Some(&s[pb.len()..])
    } else {
        None
    }
}

/// Assemble the judge prompt from the task, its success criteria, the action
/// history, and the final (distilled) page snapshot. Deterministic so it is
/// cache-friendly and unit-testable; the caller sends it to a cheap judge model
/// and feeds the reply to [`parse_completion_verdict`].
pub fn build_completion_prompt(
    task: &str,
    key_points: &[String],
    action_history: &[String],
    final_snapshot: &str,
) -> String {
    let mut out = String::with_capacity(
        task.len() + final_snapshot.len() + (key_points.len() + action_history.len()) * 24 + 512,
    );
    out.push_str(
        "You are a strict evaluator deciding whether a web-automation task was \
         actually completed. Judge ONLY from the evidence below; do not assume \
         success.\n\n",
    );

    out.push_str("# Task\n");
    out.push_str(task);
    out.push_str("\n\n");

    out.push_str("# Success criteria (all must hold)\n");
    if key_points.is_empty() {
        out.push_str("(none given — infer the criteria from the task)\n");
    } else {
        for kp in key_points {
            out.push_str("- ");
            out.push_str(kp);
            out.push('\n');
        }
    }
    out.push('\n');

    out.push_str("# Actions taken\n");
    if action_history.is_empty() {
        out.push_str("(no actions recorded)\n");
    } else {
        for (i, a) in action_history.iter().enumerate() {
            out.push_str(&format!("{}. {}\n", i + 1, a));
        }
    }
    out.push('\n');

    out.push_str("# Final page state (accessibility snapshot)\n");
    out.push_str(final_snapshot);
    out.push_str("\n\n");

    out.push_str(
        "Decide whether EVERY success criterion is satisfied by the final state. \
         Reply with exactly one line:\n\
         - VERDICT: COMPLETE\n\
         - VERDICT: INCOMPLETE: <the specific unmet criterion>\n\
         - VERDICT: UNCERTAIN: <what evidence is missing>\n",
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_complete() {
        assert_eq!(
            parse_completion_verdict("VERDICT: COMPLETE"),
            Some(CompletionVerdict::Complete)
        );
        assert!(
            parse_completion_verdict("VERDICT: COMPLETE")
                .unwrap()
                .is_complete()
        );
    }

    #[test]
    fn parses_incomplete_with_reason() {
        assert_eq!(
            parse_completion_verdict("VERDICT: INCOMPLETE: the order was never submitted"),
            Some(CompletionVerdict::Incomplete(
                "the order was never submitted".to_string()
            ))
        );
    }

    #[test]
    fn parses_uncertain_with_reason() {
        assert_eq!(
            parse_completion_verdict("VERDICT: UNCERTAIN: no confirmation page captured"),
            Some(CompletionVerdict::Uncertain(
                "no confirmation page captured".to_string()
            ))
        );
    }

    #[test]
    fn is_case_insensitive_and_skips_reasoning() {
        // Judge models prepend reasoning; the verdict line may be lower/mixed case.
        let reply = "The form shows value=\"\" so the email was never typed.\n\
                     verdict: incomplete: email field is empty";
        assert_eq!(
            parse_completion_verdict(reply),
            Some(CompletionVerdict::Incomplete(
                "email field is empty".to_string()
            ))
        );
    }

    #[test]
    fn none_when_no_verdict_line() {
        // Fail-open-toward-not-done: an unparseable reply is NOT success.
        assert_eq!(parse_completion_verdict("I think it looks fine?"), None);
        assert!(
            !parse_completion_verdict("looks done to me")
                .map(|v| v.is_complete())
                .unwrap_or(false),
            "an unparseable reply must never read as complete"
        );
    }

    #[test]
    fn non_ascii_reply_does_not_panic() {
        // A judge reply with a multibyte char straddling the "VERDICT:" byte
        // length must NOT panic. `s[..prefix.len()]` (a fixed byte index) would
        // split the codepoint of e.g. "VERDICTé" (é spans bytes 7-8, len=9 ≥ 8,
        // byte 8 is mid-char). Untrusted judge output reaches this directly.
        assert_eq!(parse_completion_verdict("VERDICTé stray line"), None);
        // And a well-formed verdict whose reason is non-ASCII parses correctly.
        assert_eq!(
            parse_completion_verdict("VERDICT: INCOMPLETE: café not confirmé"),
            Some(CompletionVerdict::Incomplete(
                "café not confirmé".to_string()
            ))
        );
    }

    #[test]
    fn first_recognized_verdict_wins_after_unrecognized() {
        let reply = "VERDICT: MAYBE\nVERDICT: COMPLETE";
        assert_eq!(
            parse_completion_verdict(reply),
            Some(CompletionVerdict::Complete)
        );
    }

    #[test]
    fn incomplete_with_empty_reason_is_allowed() {
        assert_eq!(
            parse_completion_verdict("VERDICT: INCOMPLETE:"),
            Some(CompletionVerdict::Incomplete(String::new()))
        );
    }

    #[test]
    fn only_complete_is_terminal_success() {
        assert!(CompletionVerdict::Complete.is_complete());
        assert!(!CompletionVerdict::Incomplete("x".into()).is_complete());
        assert!(!CompletionVerdict::Uncertain("x".into()).is_complete());
    }

    #[test]
    fn prompt_includes_task_criteria_actions_and_snapshot() {
        let p = build_completion_prompt(
            "Buy one widget",
            &[
                "cart contains 1 widget".to_string(),
                "order confirmed".to_string(),
            ],
            &[
                "navigate to shop".to_string(),
                "click Add to cart".to_string(),
            ],
            "uid=1_0 RootWebArea \"Cart\"\n  uid=1_1 StaticText \"1 item\"",
        );
        assert!(p.contains("Buy one widget"), "task present");
        assert!(p.contains("cart contains 1 widget"), "key point present");
        assert!(p.contains("click Add to cart"), "action present");
        assert!(
            p.contains("uid=1_1 StaticText \"1 item\""),
            "snapshot present"
        );
        assert!(
            p.contains("VERDICT: COMPLETE"),
            "instructs the verdict format"
        );
        // Deterministic: same inputs → byte-identical prompt (cache-friendly).
        let p2 = build_completion_prompt(
            "Buy one widget",
            &[
                "cart contains 1 widget".to_string(),
                "order confirmed".to_string(),
            ],
            &[
                "navigate to shop".to_string(),
                "click Add to cart".to_string(),
            ],
            "uid=1_0 RootWebArea \"Cart\"\n  uid=1_1 StaticText \"1 item\"",
        );
        assert_eq!(p, p2);
    }

    #[test]
    fn prompt_handles_empty_criteria_and_actions() {
        let p = build_completion_prompt("Do a thing", &[], &[], "uid=1_0 RootWebArea \"X\"");
        assert!(p.contains("infer the criteria"), "empty key points handled");
        assert!(p.contains("no actions recorded"), "empty history handled");
    }
}
