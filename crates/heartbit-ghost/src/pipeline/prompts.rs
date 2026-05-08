//! User-message builders for each pipeline stage. Pure string composition;
//! tested indirectly via the integration tests in `pipeline::tests`.

/// Construct the writer's user message.
///
/// On the first iteration (no `prev_revision`), only includes topic +
/// research digest + voice guidelines. On revision, also includes the
/// previous draft and the critic's feedback.
///
/// When `total_variants > 1`, appends a "you are generating variant X
/// of N" line to encourage diversity across parallel candidate slots.
pub(crate) fn build_writer_user_message(
    topic: &str,
    research_digest: &str,
    voice_guidelines: &str,
    prev_revision: Option<&(String, String)>,
    variant_index: usize,
    total_variants: usize,
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

    if total_variants > 1 {
        out.push_str(&format!(
            "\nYou are generating variant {} of {}. Pursue a distinct angle \
             from the other variants — emphasize different aspects, examples, \
             or framing.\n",
            variant_index + 1,
            total_variants,
        ));
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

/// Construct the judge's user message. Numbered candidate list with
/// voice guidelines and topic context.
pub(crate) fn build_judge_user_message(
    topic: &str,
    voice_guidelines: &str,
    candidates: &[crate::pipeline::CandidateRecord],
) -> String {
    let mut msg = format!("Topic: {topic}\n\n");
    msg.push_str(voice_guidelines);
    msg.push_str("\n\n");
    msg.push_str(&format!(
        "You have {} candidate drafts to choose from. Pick the best one.\n\n",
        candidates.len(),
    ));
    msg.push_str("CANDIDATES\n\n");
    for (i, c) in candidates.iter().enumerate() {
        msg.push_str(&format!("[{i}]\n{}\n\n", c.draft));
    }
    msg.push_str(&format!(
        "Return your verdict as JSON per the schema. The chosen_index must be in [0, {}].\n",
        candidates.len() - 1,
    ));
    msg
}

/// Construct the image_generator's user message.
pub(crate) fn build_image_generator_user_message(
    chosen_draft: &str,
    voice_guidelines: &str,
) -> String {
    format!(
        "Approved draft:\n{chosen_draft}\n\n{voice_guidelines}\n\n\
         Decide whether to attach an image. If no, output the literal \
         string \"no_image\". If yes, call image_generate with a concise \
         visual prompt and return its output.\n"
    )
}
