# heartbit-ghost P1.3a — sub-agent recipes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the 7 sub-agent recipes (researcher, writer, style_critic, judge, fact_check, image_generator, publisher) and rewrite `XGhostPersona::expand()` to return a populated `PersonaExpansion` with these recipes + 5 tool instances.

**Architecture:** New `crates/heartbit-ghost/src/agents/` module, one file per recipe. Each recipe is `pub fn <name>_recipe() -> AgentConfig` constructed inline from a `&'static str` system prompt. `agents::tools_for_persona()` returns the 5 tool instances. `XGhostPersona::expand()` wires both into `PersonaExpansion`.

**Tech Stack:** Rust 2024, `heartbit_core::config::AgentConfig`, `heartbit_core::persona::PersonaExpansion`, `Arc<dyn Tool>` for tool instances. No new workspace deps.

---

## File structure

| File | Responsibility |
|------|----------------|
| `crates/heartbit-ghost/src/agents/mod.rs` | NEW — module declarations + re-exports + `tools_for_persona()` |
| `crates/heartbit-ghost/src/agents/researcher.rs` | NEW — researcher recipe + 1 shape test + 1 boundary test |
| `crates/heartbit-ghost/src/agents/writer.rs` | NEW — writer recipe + 1 shape test + 1 boundary test |
| `crates/heartbit-ghost/src/agents/style_critic.rs` | NEW — style_critic recipe + 1 shape test |
| `crates/heartbit-ghost/src/agents/judge.rs` | NEW — judge recipe + 1 shape test + 1 boundary test |
| `crates/heartbit-ghost/src/agents/fact_check.rs` | NEW — fact_check recipe + 1 shape test |
| `crates/heartbit-ghost/src/agents/image_generator.rs` | NEW — image_generator recipe + 1 shape test |
| `crates/heartbit-ghost/src/agents/publisher.rs` | NEW — publisher recipe + 1 shape test + 1 X-mention test |
| `crates/heartbit-ghost/src/lib.rs` | MODIFY — add `pub mod agents;`; rewrite `XGhostPersona::expand()`; replace stub test |

3 implementation tasks + 1 final acceptance task.

---

## Task 1: `agents/` scaffolding

**Why:** Land the module skeleton + `tools_for_persona()` first, with empty recipe stubs. Subsequent tasks fill the recipes in. Avoids a single mega-task with 7 recipes + tools + expand all at once.

**Files:**
- Create: `crates/heartbit-ghost/src/agents/mod.rs`
- Create: `crates/heartbit-ghost/src/agents/{researcher,writer,style_critic,judge,fact_check,image_generator,publisher}.rs` (7 stub files)
- Modify: `crates/heartbit-ghost/src/lib.rs` (add `pub mod agents;`)

- [ ] **Step 1: Create `crates/heartbit-ghost/src/agents/mod.rs`**

```rust
//! Sub-agent recipes for the heartbit-ghost X (Twitter) persona.
//!
//! 7 recipes (per umbrella spec §5):
//!
//! - [`researcher`] — websearch + webfetch (reusable beyond Twitter)
//! - [`writer`] — style-conditioned generation, no tools (reusable)
//! - [`style_critic`] — voice match + AI-tell detection, no tools (partial)
//! - [`judge`] — multi-candidate ranking, no tools (reusable)
//! - [`fact_check`] — claim verification, no tools (reusable)
//! - [`image_generator`] — image_generate (reusable)
//! - [`publisher`] — twitter_thread + twitter_reply (Twitter-specific)
//!
//! Each recipe is a `pub fn <name>_recipe() -> AgentConfig` constructed
//! inline from a `&'static str` system prompt constant in the same file.
//!
//! [`tools_for_persona`] returns the 5 tool instances those recipes
//! reference, ready for inclusion in [`heartbit_core::PersonaExpansion::tools`].

use std::sync::Arc;

use heartbit_core::tool::Tool;

pub mod fact_check;
pub mod image_generator;
pub mod judge;
pub mod publisher;
pub mod researcher;
pub mod style_critic;
pub mod writer;

pub use fact_check::fact_check_recipe;
pub use image_generator::image_generator_recipe;
pub use judge::judge_recipe;
pub use publisher::publisher_recipe;
pub use researcher::researcher_recipe;
pub use style_critic::style_critic_recipe;
pub use writer::writer_recipe;

/// Construct the 5 tool instances the persona's 7 recipes reference.
///
/// Returned in declared order: `websearch`, `webfetch`, `image_generate`,
/// `twitter_thread`, `twitter_reply`. (See spec §2 for the `twitter_post`
/// scope cut.)
pub fn tools_for_persona() -> Vec<Arc<dyn Tool>> {
    use crate::tools::{TwitterReplyTool, TwitterThreadTool};
    use heartbit_core::tool::builtins::{ImageGenerateTool, WebFetchTool, WebSearchTool};

    vec![
        Arc::new(WebSearchTool::new()),
        Arc::new(WebFetchTool::new()),
        Arc::new(ImageGenerateTool::new()),
        Arc::new(TwitterThreadTool::new()),
        Arc::new(TwitterReplyTool::new()),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tools_for_persona_returns_five_distinct_tools_in_declared_order() {
        let tools = tools_for_persona();
        assert_eq!(tools.len(), 5);
        let names: Vec<&str> = tools
            .iter()
            .map(|t| t.definition().name.as_str())
            .collect();
        assert_eq!(
            names,
            vec![
                "websearch",
                "webfetch",
                "image_generate",
                "twitter_thread",
                "twitter_reply"
            ]
        );
    }
}
```

- [ ] **Step 2: Create 7 stub recipe files**

Each is a stub that compiles. Task 2 fills in the prompts and bodies. Create:

`crates/heartbit-ghost/src/agents/researcher.rs`:

```rust
//! Researcher sub-agent — websearch + webfetch. Reusable across personas.

use heartbit_core::config::AgentConfig;

/// Construct the researcher [`AgentConfig`].
pub fn researcher_recipe() -> AgentConfig {
    AgentConfig {
        name: "researcher".to_string(),
        ..AgentConfig::default()
    }
}
```

Repeat this exact 9-line shell for each of the 6 other recipes — same content with the `name` and the doc comment changed. The 7 names are: `researcher`, `writer`, `style_critic`, `judge`, `fact_check`, `image_generator`, `publisher`.

(These stubs compile but produce useless agents. Task 2 replaces every `..AgentConfig::default()` with full bodies. The point of Task 1 is to land the module + import structure cleanly.)

- [ ] **Step 3: Modify `crates/heartbit-ghost/src/lib.rs` — add `pub mod agents;`**

Find the existing `pub mod` declarations (currently `pub mod corpus;`, `pub mod tools;`, `pub mod voice;` after P1.2). Add `pub mod agents;` alphabetically (becomes the first one). Final state:

```rust
pub mod agents;
pub mod corpus;
pub mod tools;
pub mod voice;
```

- [ ] **Step 4: Run the tests**

```bash
cargo test -p heartbit-ghost --lib agents
```

Expected: `1 passed; 0 failed; 0 ignored` (the `tools_for_persona_returns_five_distinct_tools_in_declared_order` test in `agents::tests`).

- [ ] **Step 5: Workspace quality gate**

```bash
cargo fmt -p heartbit-ghost -- --check
cargo clippy -p heartbit-ghost -- -D warnings
```

Both clean.

- [ ] **Step 6: Commit**

```bash
git add crates/heartbit-ghost/src/agents/ crates/heartbit-ghost/src/lib.rs
git commit -m "$(cat <<'EOF'
feat(ghost): agents module scaffolding (P1.3a)

7 stub recipe files + agents/mod.rs with tools_for_persona helper that
returns the 5 tool instances the persona's recipes reference (websearch,
webfetch, image_generate, twitter_thread, twitter_reply).

twitter_post is intentionally absent — the heartbit-core TwitterPostTool
requires credentials at construction (incompatible with persona
expansion); a heartbit-ghost-native equivalent is deferred to a
follow-up phase. Publisher uses twitter_thread (length 1) for single
tweets in the meantime.

Stub recipes return AgentConfig with only the name set; Task 2 fills
in the prompts and per-recipe knobs.

1 test: tools_for_persona returns 5 distinct tools in declared order.

Refs: docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.3a-sub-agent-recipes-design.md §2
EOF
)"
```

---

## Task 2: All 7 recipes

**Why:** Each recipe shares a uniform structure (system prompt const + `*_recipe()` function + tests). Implementing them together amortizes the boilerplate. Each file is ~70-100 LOC.

**Files:** Modify all 7 of `crates/heartbit-ghost/src/agents/{researcher,writer,style_critic,judge,fact_check,image_generator,publisher}.rs`.

- [ ] **Step 1: Replace `crates/heartbit-ghost/src/agents/researcher.rs` with the full body**

```rust
//! Researcher sub-agent — websearch + webfetch. Reusable across personas.

use heartbit_core::config::AgentConfig;

/// System prompt for the researcher. Platform-agnostic — does not mention
/// X / Twitter; reusable for future personas (LinkedIn, blog, newsletter).
pub const RESEARCHER_SYSTEM_PROMPT: &str = r#"You are a research analyst. Given a topic or question from the user, find substance: facts, sources, recent developments, and quotable specifics.

PROCESS
1. Use `websearch` to discover relevant articles, posts, papers, and announcements.
2. Use `webfetch` to read the most promising 2-4 sources in full.
3. Synthesize a structured digest. No editorializing — your job is to surface signal, not opinion.

OUTPUT FORMAT (free-form text, no JSON):
- 1-2 sentence framing of the topic.
- 5-8 bullet points of concrete facts, each with a source link inline.
- A short list of "open questions" the topic raises (3-5 items).
- Notable quotes (1-3) attributed by author and link.

Do NOT write the post itself. The writer agent will compose. Do NOT speculate beyond what the sources support."#;

/// Construct the researcher [`AgentConfig`].
pub fn researcher_recipe() -> AgentConfig {
    AgentConfig {
        name: "researcher".to_string(),
        description: "Find substance: facts, sources, quotable specifics on a topic.".to_string(),
        system_prompt: RESEARCHER_SYSTEM_PROMPT.to_string(),
        max_turns: Some(8),
        max_tokens: Some(4096),
        reasoning_effort: Some("medium".to_string()),
        ..AgentConfig::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn researcher_recipe_has_expected_shape() {
        let cfg = researcher_recipe();
        assert_eq!(cfg.name, "researcher");
        assert!(!cfg.description.is_empty());
        assert!(!cfg.system_prompt.is_empty());
        assert_eq!(cfg.max_turns, Some(8));
        assert_eq!(cfg.max_tokens, Some(4096));
        assert_eq!(cfg.reasoning_effort.as_deref(), Some("medium"));
        assert!(cfg.response_schema.is_none(), "researcher produces free-form digest, no schema");
    }

    #[test]
    fn researcher_prompt_is_platform_agnostic() {
        let s = RESEARCHER_SYSTEM_PROMPT.to_lowercase();
        assert!(!s.contains("twitter"), "researcher prompt must not mention twitter; got snippet: {s}");
        assert!(
            !s.contains("(twitter)") && !s.contains("on x ") && !s.contains(" x ("),
            "researcher prompt must not mention X as a platform; got snippet: {s}"
        );
    }
}
```

- [ ] **Step 2: Replace `crates/heartbit-ghost/src/agents/writer.rs` with the full body**

```rust
//! Writer sub-agent — style-conditioned post generation. Reusable across
//! personas (renamed `social_writer` in the umbrella spec; file kept as
//! `writer.rs` for terseness).

use heartbit_core::config::AgentConfig;

/// System prompt for the writer. Style-profile-free — the orchestrator
/// (P1.3b) supplies voice guidelines + topic context + few-shot
/// exemplars in the user message at runtime.
pub const WRITER_SYSTEM_PROMPT: &str = r#"You are a social media writer. Produce one short, engaging post draft per call.

INPUT
The user message contains: a topic or research digest, voice guidelines for the persona, and optionally a few exemplar posts to mirror.

OUTPUT
The post text only. No preamble. No markdown fences. No commentary. No quote characters around the text. If the topic warrants a thread, separate tweets with a blank line.

Honor the voice guidelines exactly. If they say "no em-dashes", use no em-dashes. If they say "lowercase", lowercase everything. If they say "no hedging", make claims, not suggestions."#;

/// Construct the writer [`AgentConfig`].
pub fn writer_recipe() -> AgentConfig {
    AgentConfig {
        name: "writer".to_string(),
        description: "Style-conditioned post generation. One draft per call.".to_string(),
        system_prompt: WRITER_SYSTEM_PROMPT.to_string(),
        max_turns: Some(1),
        max_tokens: Some(1024),
        reasoning_effort: Some("low".to_string()),
        ..AgentConfig::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writer_recipe_has_expected_shape() {
        let cfg = writer_recipe();
        assert_eq!(cfg.name, "writer");
        assert!(!cfg.description.is_empty());
        assert!(!cfg.system_prompt.is_empty());
        assert_eq!(cfg.max_turns, Some(1));
        assert_eq!(cfg.max_tokens, Some(1024));
        assert_eq!(cfg.reasoning_effort.as_deref(), Some("low"));
        assert!(cfg.response_schema.is_none(), "writer produces free-form draft, no schema");
    }

    #[test]
    fn writer_prompt_is_platform_agnostic() {
        let s = WRITER_SYSTEM_PROMPT.to_lowercase();
        assert!(!s.contains("twitter"), "writer prompt must not mention twitter; got snippet: {s}");
        assert!(
            !s.contains("(twitter)") && !s.contains("on x ") && !s.contains(" x ("),
            "writer prompt must not mention X as a platform; got snippet: {s}"
        );
    }
}
```

- [ ] **Step 3: Replace `crates/heartbit-ghost/src/agents/style_critic.rs` with the full body**

```rust
//! Style critic sub-agent — voice match + AI-tell detection. Partially
//! reusable across personas (the schema is generic; the criterion list
//! may be tuned per platform later).

use heartbit_core::config::AgentConfig;

/// System prompt for the style critic.
pub const STYLE_CRITIC_SYSTEM_PROMPT: &str = r#"You score how well a draft post matches a target voice and flag AI-tells.

INPUT
The user message contains: the draft, the voice guidelines for the persona, and optionally specific phrases to avoid.

DECISION
Return a JSON object exactly matching the response schema:
- verdict: "pass" | "revise" | "reject"
- reason: short string explaining the verdict (required for "revise" and "reject")
- style_match_score: float in [0.0, 1.0] — higher means better voice match

VERDICTS
- "pass": draft matches the voice; no AI-tells; ready to ship.
- "revise": draft is recoverable but has issues. Common causes: AI-tell phrase, hedging, off-voice phrasing, wrong sentence-length distribution.
- "reject": draft is fundamentally off (wrong topic, defamatory, off-voice in a way revision won't fix).

Be strict on AI-tells. Phrases like "delve into", "tapestry", "navigate", "it's important to note", "balanced both-sides", "while it's true that", and "as an AI" are immediate revise triggers if voice guidelines list them. Score 0.0-0.4 = severe mismatch; 0.4-0.7 = mismatch; 0.7-0.9 = acceptable; 0.9-1.0 = excellent."#;

/// Construct the style_critic [`AgentConfig`].
pub fn style_critic_recipe() -> AgentConfig {
    AgentConfig {
        name: "style_critic".to_string(),
        description: "Score voice match and flag AI-tells. Returns pass/revise/reject + score."
            .to_string(),
        system_prompt: STYLE_CRITIC_SYSTEM_PROMPT.to_string(),
        max_turns: Some(1),
        max_tokens: Some(512),
        reasoning_effort: Some("medium".to_string()),
        response_schema: Some(serde_json::json!({
            "type": "object",
            "required": ["verdict", "style_match_score"],
            "properties": {
                "verdict": { "type": "string", "enum": ["pass", "revise", "reject"] },
                "reason": { "type": "string" },
                "style_match_score": { "type": "number", "minimum": 0.0, "maximum": 1.0 }
            }
        })),
        ..AgentConfig::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn style_critic_recipe_has_expected_shape() {
        let cfg = style_critic_recipe();
        assert_eq!(cfg.name, "style_critic");
        assert!(!cfg.description.is_empty());
        assert!(!cfg.system_prompt.is_empty());
        assert_eq!(cfg.max_turns, Some(1));
        assert_eq!(cfg.max_tokens, Some(512));
        assert_eq!(cfg.reasoning_effort.as_deref(), Some("medium"));
        assert!(cfg.response_schema.is_some(), "style_critic produces structured verdict");
        let schema = cfg.response_schema.as_ref().unwrap();
        let required = schema.get("required").and_then(|v| v.as_array()).unwrap();
        let required_names: Vec<&str> = required.iter().filter_map(|v| v.as_str()).collect();
        assert!(required_names.contains(&"verdict"));
        assert!(required_names.contains(&"style_match_score"));
    }
}
```

- [ ] **Step 4: Replace `crates/heartbit-ghost/src/agents/judge.rs` with the full body**

```rust
//! Judge sub-agent — multi-candidate ranking. Reusable across personas.

use heartbit_core::config::AgentConfig;

/// System prompt for the judge.
pub const JUDGE_SYSTEM_PROMPT: &str = r#"You rank N candidate drafts and pick the best one.

INPUT
The user message contains: the topic context, the voice guidelines, and N candidate drafts numbered from 0.

DECISION
Return a JSON object exactly matching the response schema:
- chosen_index: integer in [0, N-1] — the index of the best candidate
- reasoning: short string (1-2 sentences) explaining why the chosen candidate beats the others

CRITERIA (in priority order)
1. Voice match — does it sound like the persona's writer would write it?
2. Substance — does it say something concrete and worth reading?
3. AI-tells — fewer is better.
4. Specificity — concrete numbers, names, examples beat vague claims.
5. Engagement — would a thoughtful human reader stop scrolling?

Pick decisively. If two candidates are genuinely tied, prefer the lower index (deterministic tiebreak)."#;

/// Construct the judge [`AgentConfig`].
pub fn judge_recipe() -> AgentConfig {
    AgentConfig {
        name: "judge".to_string(),
        description: "Multi-candidate ranking. Picks the best draft from N options.".to_string(),
        system_prompt: JUDGE_SYSTEM_PROMPT.to_string(),
        max_turns: Some(1),
        max_tokens: Some(512),
        reasoning_effort: Some("medium".to_string()),
        response_schema: Some(serde_json::json!({
            "type": "object",
            "required": ["chosen_index", "reasoning"],
            "properties": {
                "chosen_index": { "type": "integer", "minimum": 0 },
                "reasoning": { "type": "string" }
            }
        })),
        ..AgentConfig::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn judge_recipe_has_expected_shape() {
        let cfg = judge_recipe();
        assert_eq!(cfg.name, "judge");
        assert!(!cfg.description.is_empty());
        assert!(!cfg.system_prompt.is_empty());
        assert_eq!(cfg.max_turns, Some(1));
        assert_eq!(cfg.max_tokens, Some(512));
        assert_eq!(cfg.reasoning_effort.as_deref(), Some("medium"));
        assert!(cfg.response_schema.is_some(), "judge produces structured pick");
    }

    #[test]
    fn judge_prompt_is_platform_agnostic() {
        let s = JUDGE_SYSTEM_PROMPT.to_lowercase();
        assert!(!s.contains("twitter"), "judge prompt must not mention twitter; got: {s}");
        assert!(
            !s.contains("(twitter)") && !s.contains("on x ") && !s.contains(" x ("),
            "judge prompt must not mention X as a platform; got: {s}"
        );
    }
}
```

- [ ] **Step 5: Replace `crates/heartbit-ghost/src/agents/fact_check.rs` with the full body**

```rust
//! Fact check sub-agent — claim verification against research output.
//! Reusable across personas.

use heartbit_core::config::AgentConfig;

/// System prompt for the fact_check agent.
pub const FACT_CHECK_SYSTEM_PROMPT: &str = r#"You verify the factual claims in a draft post against the research digest produced earlier in the pipeline.

INPUT
The user message contains: the draft, the research digest with sources.

DECISION
Return a JSON object exactly matching the response schema:
- verdict: "verified" | "unverifiable"
- reason: required when verdict is "unverifiable" — short string explaining what claim couldn't be verified

PROCESS
1. Extract every factual claim in the draft (numbers, attributions, dates, "X said Y" assertions, "the paper showed Z" references).
2. For each claim, check whether the research digest supports it.
3. If any claim is contradicted by the digest or absent from it, return "unverifiable" with the specific claim called out.
4. If every claim is supported, return "verified".

Do NOT verify against your training data. Only the supplied research digest counts. Aesthetic / stylistic / opinion content is not subject to fact-check (skip it)."#;

/// Construct the fact_check [`AgentConfig`].
pub fn fact_check_recipe() -> AgentConfig {
    AgentConfig {
        name: "fact_check".to_string(),
        description: "Verify factual claims in a draft against the research digest.".to_string(),
        system_prompt: FACT_CHECK_SYSTEM_PROMPT.to_string(),
        max_turns: Some(1),
        max_tokens: Some(1024),
        reasoning_effort: Some("medium".to_string()),
        response_schema: Some(serde_json::json!({
            "type": "object",
            "required": ["verdict"],
            "properties": {
                "verdict": { "type": "string", "enum": ["verified", "unverifiable"] },
                "reason": { "type": "string" }
            }
        })),
        ..AgentConfig::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fact_check_recipe_has_expected_shape() {
        let cfg = fact_check_recipe();
        assert_eq!(cfg.name, "fact_check");
        assert!(!cfg.description.is_empty());
        assert!(!cfg.system_prompt.is_empty());
        assert_eq!(cfg.max_turns, Some(1));
        assert_eq!(cfg.max_tokens, Some(1024));
        assert_eq!(cfg.reasoning_effort.as_deref(), Some("medium"));
        assert!(cfg.response_schema.is_some(), "fact_check produces structured verdict");
    }
}
```

- [ ] **Step 6: Replace `crates/heartbit-ghost/src/agents/image_generator.rs` with the full body**

```rust
//! Image generator sub-agent — optional accompanying image. Uses the
//! existing `image_generate` builtin.

use heartbit_core::config::AgentConfig;

/// System prompt for the image_generator.
pub const IMAGE_GENERATOR_SYSTEM_PROMPT: &str = r#"You produce an image to accompany a social media post when one would meaningfully add to it.

INPUT
The user message contains: the approved draft, the persona's voice guidelines.

DECISION
First decide: does this post benefit from an image at all? If the post is purely textual / aphoristic / a quoted reply, output the literal string "no_image" (lowercase, no quotes, no punctuation) and stop.

If yes, call `image_generate` with a concise visual prompt (one or two sentences). Avoid:
- Real people's likenesses (no "a photo of @karpathy"); use abstracted compositions instead.
- Brand logos.
- Text overlays that duplicate the post text.

Return the image_generate tool's output (URL + alt text) as your final answer."#;

/// Construct the image_generator [`AgentConfig`].
pub fn image_generator_recipe() -> AgentConfig {
    AgentConfig {
        name: "image_generator".to_string(),
        description: "Optionally produce an accompanying image for a draft.".to_string(),
        system_prompt: IMAGE_GENERATOR_SYSTEM_PROMPT.to_string(),
        max_turns: Some(2),
        max_tokens: Some(1024),
        reasoning_effort: Some("low".to_string()),
        ..AgentConfig::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_generator_recipe_has_expected_shape() {
        let cfg = image_generator_recipe();
        assert_eq!(cfg.name, "image_generator");
        assert!(!cfg.description.is_empty());
        assert!(!cfg.system_prompt.is_empty());
        assert_eq!(cfg.max_turns, Some(2));
        assert_eq!(cfg.max_tokens, Some(1024));
        assert_eq!(cfg.reasoning_effort.as_deref(), Some("low"));
        assert!(cfg.response_schema.is_none(), "image_generator output is the tool result, not structured");
    }
}
```

- [ ] **Step 7: Replace `crates/heartbit-ghost/src/agents/publisher.rs` with the full body**

```rust
//! Publisher sub-agent — Twitter-specific final API call.
//! NOT reusable across personas.

use heartbit_core::config::AgentConfig;

/// System prompt for the publisher. The ONLY recipe in heartbit-ghost
/// that mentions X / Twitter — the others are platform-agnostic.
pub const PUBLISHER_SYSTEM_PROMPT: &str = r#"You publish a finalized social post to X (Twitter).

INPUT
The user message contains: the approved post text and an indication of post shape (single, thread, or reply).

TOOL CHOICE
- For any post that's NOT a reply (single tweet OR a chained thread): call `twitter_thread`. Pass a single-element array for one tweet; pass the full sequence for a thread.
- For a reply to an existing tweet: call `twitter_reply` with the target tweet id.

The post text is approved — do not modify, do not paraphrase, do not "improve" it. Pass it through verbatim.

Return the tool's output (the tweet id) without commentary."#;

/// Construct the publisher [`AgentConfig`].
pub fn publisher_recipe() -> AgentConfig {
    AgentConfig {
        name: "publisher".to_string(),
        description: "Twitter-specific publisher. Calls twitter_thread or twitter_reply.".to_string(),
        system_prompt: PUBLISHER_SYSTEM_PROMPT.to_string(),
        max_turns: Some(2),
        max_tokens: Some(512),
        reasoning_effort: Some("low".to_string()),
        ..AgentConfig::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn publisher_recipe_has_expected_shape() {
        let cfg = publisher_recipe();
        assert_eq!(cfg.name, "publisher");
        assert!(!cfg.description.is_empty());
        assert!(!cfg.system_prompt.is_empty());
        assert_eq!(cfg.max_turns, Some(2));
        assert_eq!(cfg.max_tokens, Some(512));
        assert_eq!(cfg.reasoning_effort.as_deref(), Some("low"));
        assert!(cfg.response_schema.is_none(), "publisher output is the tool result (tweet id)");
    }

    #[test]
    fn publisher_prompt_mentions_x_or_twitter() {
        let s = PUBLISHER_SYSTEM_PROMPT.to_lowercase();
        assert!(
            s.contains("twitter") || s.contains("x (twitter)") || s.contains(" x "),
            "publisher prompt must reference X/Twitter; got: {s}"
        );
    }
}
```

- [ ] **Step 8: Run the tests**

```bash
cargo test -p heartbit-ghost --lib agents
```

Expected: `12 passed; 0 failed; 0 ignored`.

Breakdown: 1 from Task 1 (`tools_for_persona`) + 11 from Task 2 (7 shape tests, one per recipe + 3 platform-agnostic boundary tests on researcher/writer/judge + 1 X-mention test on publisher) = 12 total in `agents::*`. The `style_critic` shape test embeds a schema-required sub-assertion (it's not a separate test).

- [ ] **Step 9: Workspace quality gate**

```bash
cargo fmt -p heartbit-ghost -- --check
cargo clippy -p heartbit-ghost --all-targets -- -D warnings
```

Both clean.

- [ ] **Step 10: Commit**

```bash
git add crates/heartbit-ghost/src/agents/
git commit -m "$(cat <<'EOF'
feat(ghost): agents — 7 sub-agent recipes (P1.3a)

All 7 recipes filled in with hand-tuned system prompts + per-recipe
knobs (max_turns, max_tokens, reasoning_effort, response_schema):

- researcher: websearch + webfetch, 8 turns, 4096 tokens, free-form digest
- writer: no tools, 1 turn, 1024 tokens, free-form draft (style-profile
  injection deferred to P1.3b)
- style_critic: no tools, 1 turn, 512 tokens, structured verdict
  (pass/revise/reject + style_match_score)
- judge: no tools, 1 turn, 512 tokens, structured pick (chosen_index +
  reasoning)
- fact_check: no tools, 1 turn, 1024 tokens, structured verdict
  (verified/unverifiable)
- image_generator: image_generate, 2 turns, 1024 tokens, free-form (or
  literal "no_image" sentinel)
- publisher: twitter_thread + twitter_reply via prompt; the only recipe
  that mentions X/Twitter

Reusability boundary enforced by tests: researcher/writer/judge prompts
must NOT contain "twitter", "(twitter)", "on x", or " x (". publisher
prompt MUST contain one of those.

11 new tests: 7 shape tests, 3 platform-agnostic boundary tests,
1 publisher X-mention test.

Refs: docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.3a-sub-agent-recipes-design.md §3
EOF
)"
```

---

## Task 3: `XGhostPersona::expand()` rewrite + integration test

**Why:** The integration. Replace the P1.0 stub returning `PersonaExpansion::default()` with a populated expansion that includes the 7 recipes and 5 tools. Delete the obsolete P1.0 stub test that asserts emptiness.

**Files:** Modify `crates/heartbit-ghost/src/lib.rs`.

- [ ] **Step 1: Replace `XGhostPersona::expand()` body in `crates/heartbit-ghost/src/lib.rs`**

Find the existing `expand()` method (currently around lines 62-66 of `lib.rs`):

```rust
fn expand(&self, _params: &PersonaParams) -> Result<PersonaExpansion, heartbit_core::Error> {
    // P1.0 stub: empty expansion. P1.1+ fills this with the real persona
    // (sub-agent recipes, X tool family, triggers, Telegram review).
    Ok(PersonaExpansion::default())
}
```

Replace with:

```rust
fn expand(&self, _params: &PersonaParams) -> Result<PersonaExpansion, heartbit_core::Error> {
    let agents = vec![
        agents::researcher_recipe(),
        agents::writer_recipe(),
        agents::style_critic_recipe(),
        agents::judge_recipe(),
        agents::fact_check_recipe(),
        agents::image_generator_recipe(),
        agents::publisher_recipe(),
    ];

    let tools = agents::tools_for_persona();

    Ok(PersonaExpansion {
        agents,
        tools,
        // P1.3b populates orchestrator.
        // P1.3d populates review.
        // P1.4 populates triggers.
        ..PersonaExpansion::default()
    })
}
```

- [ ] **Step 2: Replace the obsolete P1.0 stub test in `crates/heartbit-ghost/src/lib.rs`**

Find the existing test:

```rust
#[test]
fn stub_expand_returns_empty_expansion() {
    let p = XGhostPersona::new();
    let params = PersonaParams::default();
    let exp = p.expand(&params).expect("expand returns Ok");
    assert!(exp.agents.is_empty());
    assert!(exp.tools.is_empty());
    assert!(exp.triggers.is_empty());
    assert!(exp.review.is_none());
}
```

Replace with:

```rust
#[test]
fn expand_returns_seven_agents_and_five_tools_in_declared_order() {
    let p = XGhostPersona::new();
    let params = PersonaParams::default();
    let exp = p.expand(&params).expect("expand returns Ok");
    assert_eq!(exp.agents.len(), 7);
    assert_eq!(exp.tools.len(), 5);

    let agent_names: Vec<&str> = exp.agents.iter().map(|a| a.name.as_str()).collect();
    assert_eq!(
        agent_names,
        vec![
            "researcher",
            "writer",
            "style_critic",
            "judge",
            "fact_check",
            "image_generator",
            "publisher",
        ]
    );

    let tool_names: Vec<&str> = exp
        .tools
        .iter()
        .map(|t| t.definition().name.as_str())
        .collect();
    assert_eq!(
        tool_names,
        vec![
            "websearch",
            "webfetch",
            "image_generate",
            "twitter_thread",
            "twitter_reply",
        ]
    );

    // Triggers and review remain default (P1.3d / P1.4).
    assert!(exp.triggers.is_empty());
    assert!(exp.review.is_none());
}
```

(The other P1.0 lib.rs tests — `stub_name_is_stable`, `stub_description_is_non_empty_and_marks_p1_0`, `stub_version_matches_cargo_pkg_version`, `register_adds_persona_to_empty_registry`, `register_twice_is_idempotent` — stay untouched. Only `stub_expand_returns_empty_expansion` is replaced.)

- [ ] **Step 3: Update `XGhostPersona::description()` to remove the "stub" wording**

Find:

```rust
fn description(&self) -> &str {
    "Best-in-class autonomous X (Twitter) agent. Scaffolding stub — Phase 1 P1.0."
}
```

Replace with:

```rust
fn description(&self) -> &str {
    "Best-in-class autonomous X (Twitter) agent. P1.3a: 7 sub-agents wired; pipeline orchestration lands in P1.3b."
}
```

This means `stub_description_is_non_empty_and_marks_p1_0` will fail (it asserts `desc.contains("P1.0") || desc.contains("Scaffolding") || desc.contains("stub")`). Update that test:

Find:

```rust
#[test]
fn stub_description_is_non_empty_and_marks_p1_0() {
    let p = XGhostPersona::new();
    let desc = p.description();
    assert!(!desc.is_empty());
    assert!(desc.contains("P1.0") || desc.contains("Scaffolding") || desc.contains("stub"));
}
```

Replace with:

```rust
#[test]
fn description_is_non_empty_and_marks_current_phase() {
    let p = XGhostPersona::new();
    let desc = p.description();
    assert!(!desc.is_empty());
    assert!(
        desc.contains("P1.3") || desc.contains("sub-agent"),
        "description should reflect the current phase; got: {desc}"
    );
}
```

- [ ] **Step 4: Run the tests**

```bash
cargo test -p heartbit-ghost --lib
```

Expected: full crate test count plus the new `expand_returns_seven_agents_and_five_tools_in_declared_order` test passing. The replaced `description_is_non_empty_and_marks_current_phase` test passes.

The other 4 untouched P1.0 lib.rs tests still pass: `stub_name_is_stable`, `stub_version_matches_cargo_pkg_version`, `register_adds_persona_to_empty_registry`, `register_twice_is_idempotent`.

- [ ] **Step 5: Workspace quality gate**

```bash
cargo fmt -p heartbit-ghost -- --check
cargo clippy -p heartbit-ghost --all-targets -- -D warnings
```

Both clean.

- [ ] **Step 6: Commit**

```bash
git add crates/heartbit-ghost/src/lib.rs
git commit -m "$(cat <<'EOF'
feat(ghost): XGhostPersona::expand() returns populated expansion (P1.3a)

Rewrite the P1.0 stub (which returned PersonaExpansion::default()) to
return 7 agents + 5 tools via agents::*_recipe() + agents::tools_for_persona().

Replaces the now-obsolete stub_expand_returns_empty_expansion test with
expand_returns_seven_agents_and_five_tools_in_declared_order, which
pins the public contract: 7 named agents in declared order + 5 named
tools in declared order. P1.3b's orchestrator can rely on this.

Also updates XGhostPersona::description() to remove the "stub / P1.0"
wording (the corresponding test is updated to match).

Other untouched P1.0 lib.rs tests (stub_name_is_stable, version,
register_*, etc.) remain.

Refs: docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.3a-sub-agent-recipes-design.md §4
EOF
)"
```

---

## Task 4: Final acceptance + workspace quality gate

**Why:** Confirm P1.3a meets every acceptance criterion in the spec. Verification only — no commit.

**Files:** none.

- [ ] **Step 1: Workspace quality gate**

```bash
cd /home/pleclech/projects/heartbit
cargo fmt -- --check && cargo clippy --workspace --all-features -- -D warnings && cargo test --workspace --all-features 2>&1 | tail -10
```

Expected: all green. Workspace test count goes from 3932 (post-P1.2e baseline) to **3944** (12 net new):

- +12 in `agents::*` (Tasks 1 + 2): 1 `tools_for_persona` + 7 shape tests + 3 platform-agnostic boundary tests + 1 publisher X-mention test
- +1 in `lib.rs` for `expand_returns_seven_agents_and_five_tools_in_declared_order` (Task 3)
- −1 in `lib.rs` for the deleted `stub_expand_returns_empty_expansion` (Task 3)
- 0 net change for the in-place rename of `stub_description_is_non_empty_and_marks_p1_0` → `description_is_non_empty_and_marks_current_phase` (Task 3)

Net: +12 tests.

- [ ] **Step 2: Verify the public surface is reachable**

```bash
cat <<'EOF' > /tmp/heartbit_ghost_p1_3a_surface_check.rs
fn _check() {
    use heartbit_ghost::agents::{
        fact_check_recipe, image_generator_recipe, judge_recipe, publisher_recipe,
        researcher_recipe, style_critic_recipe, tools_for_persona, writer_recipe,
    };
    use heartbit_core::config::AgentConfig;
    let _: fn() -> AgentConfig = researcher_recipe;
    let _: fn() -> AgentConfig = writer_recipe;
    let _: fn() -> AgentConfig = style_critic_recipe;
    let _: fn() -> AgentConfig = judge_recipe;
    let _: fn() -> AgentConfig = fact_check_recipe;
    let _: fn() -> AgentConfig = image_generator_recipe;
    let _: fn() -> AgentConfig = publisher_recipe;
    let _ = tools_for_persona();
}
EOF
echo "(Surface check is illustrative; reachability is verified by cargo check above.)"
rm -f /tmp/heartbit_ghost_p1_3a_surface_check.rs
```

- [ ] **Step 3: Print final commit log for the branch**

```bash
git log --oneline main..feat/heartbit-ghost-p1.3
```

Expected: 4-5 commits — spec doc + spec amendment + plan doc + 3 task commits.

- [ ] **Step 4: No commit for this task**

Task 4 is verification only. The branch is ready for final review + merge. P1.3a is complete; P1.3b (pipeline orchestrator) follows.

---

## Acceptance criteria

P1.3a is done when (per spec §8):

- All public types compile cleanly under `cargo check -p heartbit-ghost --all-features`
- `cargo fmt -- --check && cargo clippy --workspace --all-features -- -D warnings && cargo test --workspace --all-features` all green
- 12 net new tests pass; coverage spans every recipe's structural contract (7 shape tests), the reusability boundary on the 3 platform-agnostic recipes (researcher/writer/judge — 3 boundary tests), the Twitter-specific publisher (1 X-mention test), the `tools_for_persona()` order (1 test), and the `expand()` integration (1 test). The spec's earlier "~16 net new" estimate over-counted; the actual breakdown is 13 added − 1 deleted = 12 net
- `heartbit_ghost::agents::{researcher_recipe, writer_recipe, style_critic_recipe, judge_recipe, fact_check_recipe, image_generator_recipe, publisher_recipe, tools_for_persona}` are reachable as public surface
- `XGhostPersona::expand(&PersonaParams::default())` returns a `PersonaExpansion` with 7 agents (in declared order) and 5 tools (in declared order)

## Out of scope (re-stated)

- Pipeline orchestration / agent chaining (P1.3b)
- Style profile injection into writer prompt at runtime (P1.3b)
- Multi-candidate generation (3-rotation + Levenshtein dedup) (P1.3c)
- Telegram review delivery (P1.3d)
- Pick storage / few-shot exemplar retrieval (P1.3e)
- A heartbit-ghost-native `TwitterPostTool` for media-attached single posts (deferred follow-up phase)
- Autonomy phase logic (P1.3d / P1.4)
- Audit log integration (P1.4)
- Trigger specs (P1.4)

## Reference

- Spec: `docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.3a-sub-agent-recipes-design.md`
- Umbrella heartbit-ghost spec: `docs/superpowers/specs/2026-05-07-heartbit-ghost-x-agent-design.md` (§3, §5, §6)
- `Persona` trait + `PersonaExpansion`: `crates/heartbit-core/src/persona/{mod,types}.rs`
- `AgentConfig`: `crates/heartbit-core/src/config/agent.rs`
- Existing X tools: `crates/heartbit-ghost/src/tools/` (P1.1)
- Existing builtin tools: `crates/heartbit-core/src/tool/builtins/`
