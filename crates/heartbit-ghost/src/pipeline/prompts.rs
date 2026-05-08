//! User-message builders for each pipeline stage. Pure string composition;
//! tested indirectly via the integration tests in `pipeline::tests`.
//!
//! These helpers are wired into the orchestrator body in Task 2/3; until
//! then the dead-code lint would fire crate-wide.
#![allow(dead_code)]

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
