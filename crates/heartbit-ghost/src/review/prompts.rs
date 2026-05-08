//! Render the review message for delivery and the report message for
//! the in-place edit after pick.

use crate::review::delivery::{ReportableOutcome, ReviewMessage};

/// Maximum total characters in the rendered Telegram review message body.
/// Telegram's per-message limit is 4096; we leave headroom for the
/// keyboard footer and emoji decorations.
const MAX_REVIEW_BODY_CHARS: usize = 3500;

/// Per-candidate truncation budget — keeps the message readable when
/// candidates are long. Truncation appends `…` and the rest is hidden.
const PER_CANDIDATE_TRUNCATE_CHARS: usize = 900;

/// Render the review message body for delivery. Output is plain text;
/// the delivery layer handles emoji rendering / Telegram-specific markup.
pub fn build_review_message(message: &ReviewMessage) -> String {
    let mut out = String::with_capacity(MAX_REVIEW_BODY_CHARS);
    out.push_str(&format!(
        "🪶 Draft for {} — {}\n\n",
        message.persona_name, message.topic
    ));
    let emojis = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣"];
    for (i, candidate) in message.candidates.iter().enumerate() {
        let label = emojis.get(i).copied().unwrap_or("•");
        out.push_str(label);
        out.push(' ');
        out.push_str(&truncate_with_ellipsis(
            candidate,
            PER_CANDIDATE_TRUNCATE_CHARS,
        ));
        out.push_str("\n\n");
    }
    out.push_str("Pick one, or Skip");
    // Hard-cap the total body in case truncation didn't keep us under.
    truncate_with_ellipsis(&out, MAX_REVIEW_BODY_CHARS)
}

/// Render the report message that replaces the original Telegram message
/// after the user picks (or skips, or times out).
pub fn build_report_message(outcome: &ReportableOutcome) -> String {
    match outcome {
        ReportableOutcome::Posted {
            chosen_index,
            tweet_url,
        } => format!("✅ Posted draft {} — {}", chosen_index + 1, tweet_url),
        ReportableOutcome::Skipped => "❎ Skipped (no post)".to_string(),
        ReportableOutcome::TimedOut => "⏰ Timed out — no pick".to_string(),
        ReportableOutcome::GateRejected {
            chosen_index,
            reason,
        } => format!(
            "🚫 Draft {} rejected by publish_gate: {}",
            chosen_index + 1,
            reason
        ),
        ReportableOutcome::PublishFailed {
            chosen_index,
            reason,
        } => format!(
            "⚠️ Publish failed for draft {}: {}",
            chosen_index + 1,
            reason
        ),
    }
}

fn truncate_with_ellipsis(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max_chars.saturating_sub(1)).collect();
    out.push('…');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn mk_message(candidates: Vec<&str>) -> ReviewMessage {
        ReviewMessage {
            persona_name: "heartbit-ghost:x".to_string(),
            topic: "agent harness".to_string(),
            candidates: candidates.into_iter().map(String::from).collect(),
            interaction_id: Uuid::new_v4(),
        }
    }

    #[test]
    fn build_review_message_three_candidates_renders_with_emoji_labels() {
        let m = mk_message(vec!["alpha draft", "bravo draft", "charlie draft"]);
        let out = build_review_message(&m);
        assert!(out.contains("🪶 Draft for heartbit-ghost:x — agent harness"));
        assert!(out.contains("1️⃣ alpha draft"));
        assert!(out.contains("2️⃣ bravo draft"));
        assert!(out.contains("3️⃣ charlie draft"));
        assert!(out.ends_with("Pick one, or Skip"));
    }

    #[test]
    fn build_review_message_truncates_long_candidate_with_ellipsis() {
        let long = "x".repeat(2000);
        let m = mk_message(vec![long.as_str()]);
        let out = build_review_message(&m);
        assert!(
            out.chars().count() < 1500,
            "expected per-candidate truncation; got {} chars",
            out.chars().count()
        );
        assert!(out.contains('…'), "expected ellipsis: {out}");
    }

    #[test]
    fn build_review_message_handles_special_chars_passthrough() {
        // No HTML / Markdown escaping in this layer — delivery layer
        // handles its own escaping.
        let m = mk_message(vec!["draft with <html> & \"quotes\""]);
        let out = build_review_message(&m);
        assert!(out.contains("<html>"), "should not strip < >");
        assert!(out.contains("\"quotes\""), "should not strip quotes");
    }

    #[test]
    fn build_report_message_posted_includes_one_based_index_and_url() {
        let o = ReportableOutcome::Posted {
            chosen_index: 1, // 0-based
            tweet_url: "https://twitter.com/i/web/status/12345".to_string(),
        };
        let s = build_report_message(&o);
        assert!(s.contains("Posted draft 2"), "got: {s}"); // 1-based for users
        assert!(s.contains("https://twitter.com"), "got: {s}");
    }
}
