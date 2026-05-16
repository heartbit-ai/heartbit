//! User-message builders for each blog-pipeline stage. Pure string
//! composition — same shape as `reply/prompts.rs` and `quote/prompts.rs`.
#![allow(dead_code)]

use crate::blog::seed::BlogSeed;

/// Lightweight seed projection — only the fields the prompts need. Lets
/// the prompts module be testable without depending on the full
/// `BlogSeed` runtime type.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BlogSeedInput<'a> {
    pub text: &'a str,
    pub source_url: Option<&'a str>,
    pub rationale: Option<&'a str>,
}

impl<'a> From<&'a BlogSeed> for BlogSeedInput<'a> {
    fn from(seed: &'a BlogSeed) -> Self {
        Self {
            text: &seed.text,
            source_url: Some(&seed.source_url),
            rationale: Some(&seed.rationale),
        }
    }
}

/// Build the blog researcher's user message. The seed gives the topic;
/// the researcher's job is to surface sourced specifics that the writer
/// will weave into the essay.
pub(crate) fn build_blog_research_user_message(seed: &BlogSeedInput<'_>) -> String {
    let mut out = String::new();
    out.push_str("TOPIC SEED\n");
    out.push_str(&format!("Seed text: {}\n", seed.text));
    if let Some(url) = seed.source_url {
        out.push_str(&format!("Originally posted at: {url}\n"));
    }
    if let Some(rationale) = seed.rationale {
        out.push_str(&format!("Why this topic: {rationale}\n"));
    }
    out.push_str("\nResearch this topic. Find 4-6 substantive sources with sourced specifics (numbers, dates, citations, attributions). Output the structured digest per your system prompt. Do NOT compose the essay — the blog_writer composes it next.\n");
    out
}

/// Build the blog writer's user message. Includes the research digest,
/// the topic seed for framing context, and voice guidelines.
pub(crate) fn build_blog_writer_user_message(
    digest: &str,
    seed: &BlogSeedInput<'_>,
    voice_guidelines: &str,
) -> String {
    let mut out = String::new();
    out.push_str("Research digest (sourced facts to anchor the essay):\n");
    out.push_str(digest);
    out.push_str("\n\n");
    out.push_str(&format!("TOPIC SEED: {}\n", seed.text));
    if let Some(url) = seed.source_url {
        out.push_str(&format!(
            "(This topic was derived from a high-engagement X post: {url}. The essay expands on the idea; it does NOT quote or re-tweet the original.)\n"
        ));
    }
    out.push('\n');
    out.push_str(voice_guidelines);
    out.push('\n');
    out.push_str("\nWrite ONE complete essay (800-1500 words) in Markdown. Output the essay text only — no title line, no frontmatter.\n");
    out
}

/// Build the blog style critic's user message.
pub(crate) fn build_blog_critic_user_message(draft: &str, voice_guidelines: &str) -> String {
    format!(
        "Essay draft to evaluate:\n\n{draft}\n\n{voice_guidelines}\n\nScore the draft and return your verdict as JSON per the schema.\n"
    )
}

/// Build the blog fact-check's user message.
pub(crate) fn build_blog_fact_user_message(draft: &str, digest: &str) -> String {
    format!(
        "Essay draft to verify:\n\n{draft}\n\nResearch digest (only source of truth):\n\n{digest}\n\nVerify and return your verdict as JSON per the schema.\n"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_seed_input<'a>() -> BlogSeedInput<'a> {
        BlogSeedInput {
            text: "Every tool response your LLM agent consumes is a potential attack vector.",
            source_url: Some("https://twitter.com/i/web/status/2054484107212538042"),
            rationale: Some("Highest engagement post in the last 7 days (score 4.2)."),
        }
    }

    #[test]
    fn writer_message_includes_seed_and_digest() {
        let seed = fixture_seed_input();
        let s = build_blog_writer_user_message("digest text", &seed, "VOICE GUIDELINES");
        assert!(s.contains("digest text"));
        assert!(s.contains("Every tool response"));
        assert!(s.contains("VOICE GUIDELINES"));
        assert!(s.contains("800-1500 words"));
    }

    #[test]
    fn writer_message_mentions_no_title_line() {
        let seed = fixture_seed_input();
        let s = build_blog_writer_user_message("d", &seed, "v");
        assert!(
            s.contains("no title line") || s.contains("no frontmatter"),
            "writer must be told to skip title/frontmatter (renderer handles them): {s}"
        );
    }

    #[test]
    fn writer_message_clarifies_essay_is_not_a_quote_of_the_source() {
        let seed = fixture_seed_input();
        let s = build_blog_writer_user_message("d", &seed, "v");
        assert!(
            s.contains("does NOT quote or re-tweet"),
            "writer must be told the essay EXPANDS on the seed, not quotes it"
        );
    }

    #[test]
    fn research_message_includes_topic_seed() {
        let seed = fixture_seed_input();
        let s = build_blog_research_user_message(&seed);
        assert!(s.contains("Every tool response"));
        assert!(s.contains("Originally posted at"));
        assert!(s.contains("4-6 substantive sources"));
        assert!(s.contains("Do NOT compose the essay"));
    }

    #[test]
    fn critic_message_includes_draft_and_voice() {
        let s = build_blog_critic_user_message("DRAFT", "VOICE");
        assert!(s.contains("DRAFT"));
        assert!(s.contains("VOICE"));
        assert!(s.contains("JSON"));
    }

    #[test]
    fn fact_message_includes_draft_and_digest() {
        let s = build_blog_fact_user_message("DRAFT", "DIGEST");
        assert!(s.contains("DRAFT"));
        assert!(s.contains("DIGEST"));
        assert!(s.contains("only source of truth"));
    }
}
