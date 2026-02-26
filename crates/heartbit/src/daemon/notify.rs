use crate::daemon::types::TaskState;
use crate::llm::types::TokenUsage;

/// Fired when a daemon task reaches a terminal state (Completed, Failed, Cancelled).
pub struct TaskOutcome {
    pub id: uuid::Uuid,
    pub source: String,
    pub state: TaskState,
    /// Result text (if any). Truncated to ~500 bytes at display time by
    /// `format_notification()`.
    pub result_summary: Option<String>,
    pub error: Option<String>,
    pub duration_secs: f64,
    pub tokens: TokenUsage,
    pub cost: Option<f64>,
    /// The story ID that grouped this task (if any).
    pub story_id: Option<String>,
}

/// Callback type for task completion notifications.
pub type OnTaskComplete = dyn Fn(TaskOutcome) + Send + Sync;

/// Format a `TaskOutcome` into a compact markdown notification message.
///
/// The output is markdown — the caller is responsible for converting to the
/// appropriate transport format (e.g. `markdown_to_telegram_html()` for Telegram).
pub fn format_notification(outcome: &TaskOutcome) -> String {
    let total_tokens = outcome.tokens.input_tokens + outcome.tokens.output_tokens;

    match outcome.state {
        TaskState::Completed => {
            let mut msg = format!(
                "\u{2705} **{}** ({:.1}s \u{00b7} {} tok)",
                outcome.source,
                outcome.duration_secs,
                format_tokens(total_tokens),
            );
            if let Some(cost) = outcome.cost {
                msg.push_str(&format!(" \u{00b7} ${cost:.4}"));
            }
            if let Some(ref summary) = outcome.result_summary {
                let truncated = truncate_summary(summary, 500);
                msg.push_str("\n\n");
                msg.push_str(truncated);
            }
            msg
        }
        TaskState::Failed => {
            let mut msg = format!(
                "\u{274c} **{}** ({:.1}s)",
                outcome.source, outcome.duration_secs,
            );
            if let Some(ref error) = outcome.error {
                let truncated = truncate_summary(error, 500);
                msg.push_str("\n\n");
                msg.push_str("Error: ");
                msg.push_str(truncated);
            }
            msg
        }
        TaskState::Cancelled => {
            format!("\u{26a0}\u{fe0f} **{}** \u{2014} cancelled", outcome.source,)
        }
        // Pending/Running are not terminal — shouldn't happen, but handle gracefully.
        _ => format!(
            "\u{2139}\u{fe0f} **{}** \u{2014} {}",
            outcome.source,
            format!("{:?}", outcome.state).to_lowercase(),
        ),
    }
}

/// Truncate a string to approximately `max_bytes` at a UTF-8 safe boundary.
fn truncate_summary(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut pos = max_bytes;
    while pos > 0 && !s.is_char_boundary(pos) {
        pos -= 1;
    }
    &s[..pos]
}

/// Format token count with thousands separator.
fn format_tokens(n: u32) -> String {
    if n < 1_000 {
        return n.to_string();
    }
    let s = n.to_string();
    let mut result = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_outcome(state: TaskState) -> TaskOutcome {
        TaskOutcome {
            id: uuid::Uuid::nil(),
            source: "sensor:gmail_inbox".into(),
            state,
            result_summary: None,
            error: None,
            duration_secs: 3.2,
            tokens: TokenUsage {
                input_tokens: 1000,
                output_tokens: 234,
                ..Default::default()
            },
            cost: Some(0.0042),
            story_id: None,
        }
    }

    #[test]
    fn format_completed_with_summary() {
        let mut outcome = make_outcome(TaskState::Completed);
        outcome.result_summary =
            Some("Stored email from john@example.com about project update".into());

        let msg = format_notification(&outcome);
        assert!(msg.contains("\u{2705}"));
        assert!(msg.contains("**sensor:gmail_inbox**"));
        assert!(msg.contains("3.2s"));
        assert!(msg.contains("1,234 tok"));
        assert!(msg.contains("$0.0042"));
        assert!(msg.contains("Stored email from john@example.com"));
    }

    #[test]
    fn format_completed_no_cost() {
        let mut outcome = make_outcome(TaskState::Completed);
        outcome.cost = None;
        outcome.result_summary = Some("Done".into());

        let msg = format_notification(&outcome);
        assert!(msg.contains("\u{2705}"));
        assert!(!msg.contains('$'));
        assert!(msg.contains("Done"));
    }

    #[test]
    fn format_failed_with_error() {
        let mut outcome = make_outcome(TaskState::Failed);
        outcome.error = Some("context overflow".into());

        let msg = format_notification(&outcome);
        assert!(msg.contains("\u{274c}"));
        assert!(msg.contains("**sensor:gmail_inbox**"));
        assert!(msg.contains("Error: context overflow"));
    }

    #[test]
    fn format_cancelled() {
        let outcome = make_outcome(TaskState::Cancelled);

        let msg = format_notification(&outcome);
        assert!(msg.contains("\u{26a0}"));
        assert!(msg.contains("**sensor:gmail_inbox**"));
        assert!(msg.contains("cancelled"));
    }

    #[test]
    fn truncates_long_summary() {
        let long_text = "x".repeat(1000);
        let mut outcome = make_outcome(TaskState::Completed);
        outcome.result_summary = Some(long_text);

        let msg = format_notification(&outcome);
        // The summary in the message should be truncated to ~500 chars
        // The message itself includes the header, so total is longer
        let lines: Vec<&str> = msg.splitn(3, '\n').collect();
        // Third element (index 2) is the summary after the double newline
        assert!(lines.len() >= 3);
        assert!(lines[2].len() <= 500);
    }

    #[test]
    fn markdown_passthrough() {
        let mut outcome = make_outcome(TaskState::Completed);
        outcome.result_summary = Some("Result with **bold** and `code`".into());

        let msg = format_notification(&outcome);
        // Markdown in the summary passes through (caller converts to HTML)
        assert!(msg.contains("**bold**"));
        assert!(msg.contains("`code`"));
    }

    #[test]
    fn truncate_summary_ascii() {
        assert_eq!(truncate_summary("hello world", 5), "hello");
        assert_eq!(truncate_summary("hello", 10), "hello");
        assert_eq!(truncate_summary("", 5), "");
    }

    #[test]
    fn truncate_summary_utf8_safe() {
        // "café" is 5 bytes: c(1) a(1) f(1) é(2)
        let s = "café";
        assert_eq!(s.len(), 5);
        // Truncating at 4 bytes would split 'é', should back up to 3
        assert_eq!(truncate_summary(s, 4), "caf");
        assert_eq!(truncate_summary(s, 5), "café");
        assert_eq!(truncate_summary(s, 3), "caf");
    }

    #[test]
    fn truncate_summary_emoji() {
        let s = "hi🦀bye"; // h(1) i(1) 🦀(4) b(1) y(1) e(1) = 9 bytes
        assert_eq!(truncate_summary(s, 5), "hi"); // mid-emoji, back to 2
        assert_eq!(truncate_summary(s, 6), "hi🦀"); // exact end of emoji
    }

    #[test]
    fn format_tokens_no_separator() {
        assert_eq!(format_tokens(0), "0");
        assert_eq!(format_tokens(999), "999");
    }

    #[test]
    fn format_tokens_with_separator() {
        assert_eq!(format_tokens(1_000), "1,000");
        assert_eq!(format_tokens(1_234), "1,234");
        assert_eq!(format_tokens(12_345), "12,345");
        assert_eq!(format_tokens(123_456), "123,456");
        assert_eq!(format_tokens(1_234_567), "1,234,567");
    }

    #[test]
    fn story_id_roundtrips() {
        let mut outcome = make_outcome(TaskState::Completed);
        outcome.story_id = Some("story-abc-123".into());
        assert_eq!(outcome.story_id.as_deref(), Some("story-abc-123"));
    }

    #[test]
    fn story_id_defaults_to_none() {
        let outcome = make_outcome(TaskState::Completed);
        assert!(outcome.story_id.is_none());
    }
}
