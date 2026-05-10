//! `heartbit-rs:x` persona — demonstrates heartbit-core / heartbit-cli
//! features by example. Reuses ghost's pipeline; only the researcher
//! agent and the writer's user-message addendum differ.

use std::sync::Arc;

use heartbit_core::persona::{Persona, PersonaExpansion, PersonaParams, PersonaRegistry};

/// Persona name used by `name() = "heartbit-rs:x"`.
pub const PERSONA_NAME: &str = "heartbit-rs:x";

/// Evangelism-mode addendum surfaced in voice-aware user messages by
/// the pipeline. See spec §6 for the rationale.
pub const MODE_ADDENDUM: &str = r#"EVANGELISM MODE — heartbit-core

You are showing what heartbit-core (a Rust multi-agent framework, published on crates.io and at https://github.com/heartbit-ai/heartbit) does, by example. Your audience is Rust developers and AI engineers evaluating the framework.

THREAD SHAPE
Every thread is structured as: hook → demo → payoff.
- Hook: ONE concrete sentence stating what this feature lets you do (e.g. "Implement two methods on a trait, get a fully-wired tool with retry, guardrails, and telemetry.").
- Demo: a code excerpt taken from the researcher's digest. Paraphrase for tweet-friendliness if needed but do not invent code that wasn't in the digest. Reference the canonical file path inline (e.g., "in `crates/heartbit-core/src/tool/mod.rs`") so curious readers can cross-check on GitHub.
- Payoff: 1-2 tweets on what this enables — concrete benefits, not adjectives.

GROUND TRUTH
- Every claim about heartbit-core MUST trace back to a real file path or type the researcher surfaced. No vague "powerful" / "elegant" / "production-grade" framework adjectives without the corresponding code.
- If you cannot ground a claim, drop the claim.
- The framework is real and public: `cargo add heartbit-core` works, https://github.com/heartbit-ai/heartbit is browseable. If the researcher's digest seems to suggest otherwise, ignore that — the local source is the authority.

NEVER
- Release-note framing ("we shipped X yesterday", "new in v2.0", "just released"). Frame everything time-agnostically — "here's what X does" not "here's what we just added".
- Marketing superlatives without code backing them.
- Code excerpts longer than 8 lines per tweet.
"#;

/// The `heartbit-rs:x` persona type.
pub struct XHeartbitRsPersona {
    /// Persona version string, derived at compile time from the workspace
    /// `Cargo.toml`.
    version: &'static str,
}

impl XHeartbitRsPersona {
    /// Construct a new persona instance.
    pub fn new() -> Self {
        Self {
            version: env!("CARGO_PKG_VERSION"),
        }
    }
}

impl Default for XHeartbitRsPersona {
    fn default() -> Self {
        Self::new()
    }
}

impl Persona for XHeartbitRsPersona {
    fn name(&self) -> &str {
        PERSONA_NAME
    }

    fn description(&self) -> &str {
        "Demonstrates heartbit-core / heartbit-cli features by example. Pure on-demand."
    }

    fn version(&self) -> &str {
        self.version
    }

    fn expand(&self, _params: &PersonaParams) -> Result<PersonaExpansion, heartbit_core::Error> {
        let agents = vec![
            crate::agents::repo_researcher_recipe(), // <— differs from ghost
            crate::agents::writer_recipe(),
            crate::agents::style_critic_recipe(),
            crate::agents::judge_recipe(),
            crate::agents::fact_check_recipe(),
            crate::agents::image_generator_recipe(),
            crate::agents::publisher_recipe(),
        ];
        let tools = crate::agents::tools_for_heartbit_rs(); // <— differs

        let repo_root = match std::env::var("HEARTBIT_REPO_ROOT") {
            Ok(s) => std::path::PathBuf::from(s),
            Err(_) => std::env::current_dir().map_err(|e| {
                heartbit_core::Error::Config(format!(
                    "HEARTBIT_REPO_ROOT not set and current_dir() failed: {e}"
                ))
            })?,
        };

        Ok(PersonaExpansion {
            agents,
            tools,
            mode_addendum: Some(MODE_ADDENDUM),
            topic_context_provider: Some(std::sync::Arc::new(
                crate::posts::HeartbitRsXTopicContext::new(repo_root),
            )),
            ..PersonaExpansion::default()
        })
    }
}

/// Register the heartbit-rs:x persona into the supplied registry.
pub fn register(registry: &mut PersonaRegistry) {
    registry.register(Arc::new(XHeartbitRsPersona::new()));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_is_stable() {
        let p = XHeartbitRsPersona::new();
        assert_eq!(p.name(), "heartbit-rs:x");
        assert_eq!(p.name(), PERSONA_NAME);
    }

    #[test]
    fn description_is_non_empty() {
        let p = XHeartbitRsPersona::new();
        assert!(!p.description().is_empty());
    }

    #[test]
    fn expand_components_match_expected_shape() {
        // Don't call expand() directly — it invokes the env-var path
        // which is unsafe to mutate in tests. Instead build the same
        // pieces expand() composes and assert their shape.

        // 1. Agent slots in declared order (slot 0 differs from ghost).
        let agents = [
            crate::agents::repo_researcher_recipe(),
            crate::agents::writer_recipe(),
            crate::agents::style_critic_recipe(),
            crate::agents::judge_recipe(),
            crate::agents::fact_check_recipe(),
            crate::agents::image_generator_recipe(),
            crate::agents::publisher_recipe(),
        ];
        assert_eq!(agents.len(), 7, "expected 7 sub-agent recipes");
        let names: Vec<&str> = agents.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "repo_researcher", // differs from ghost (was researcher)
                "writer",
                "style_critic",
                "judge",
                "fact_check",
                "image_generator",
                "publisher",
            ]
        );

        // 2. Tool list in declared order — env-free path via _with_root.
        let repo_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf();
        let tools = crate::agents::tools_for_heartbit_rs_with_root(repo_root);
        let tool_names: Vec<String> = tools.iter().map(|t| t.definition().name).collect();
        // websearch + webfetch are deliberately excluded so the
        // researcher cannot default to web lookups and skip repo_inspect.
        assert_eq!(
            tool_names,
            vec![
                "repo_inspect".to_string(),
                "image_generate".to_string(),
                "twitter_thread".to_string(),
                "twitter_reply".to_string(),
            ]
        );

        // 3. The MODE_ADDENDUM constant is non-empty and covers the
        // four required sections.
        assert!(!MODE_ADDENDUM.is_empty());
        assert!(MODE_ADDENDUM.contains("EVANGELISM MODE"));
        assert!(MODE_ADDENDUM.contains("hook → demo → payoff"));
        assert!(MODE_ADDENDUM.contains("GROUND TRUTH"));
        assert!(MODE_ADDENDUM.contains("NEVER"));
    }

    #[test]
    fn expand_populates_topic_context_provider() {
        // Set the workspace root so expand() doesn't depend on cwd.
        let repo_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf();
        unsafe {
            std::env::set_var("HEARTBIT_REPO_ROOT", &repo_root);
        }
        let p = XHeartbitRsPersona::new();
        let exp = p
            .expand(&heartbit_core::PersonaParams::default())
            .expect("expand() should succeed with explicit HEARTBIT_REPO_ROOT");
        assert!(
            exp.topic_context_provider.is_some(),
            "HeartbitRs should populate topic_context_provider"
        );
    }
}
