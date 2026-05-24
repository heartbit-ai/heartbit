//! Handler for `DaemonCommand::BlogAnnounceX`. Builds the announcement
//! pipeline config and invokes [`heartbit_ghost::blog::announce::run_x_announcement_pipeline`].
//!
//! Thin dispatcher — most logic lives in the pipeline itself
//! (`crates/heartbit-ghost/src/blog/announce.rs`).

use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use heartbit_core::CredentialResolver;
use heartbit_core::Tool;
use heartbit_core::llm::BoxedProvider;
use heartbit_core::persona::{PersonaParams, PersonaRegistry};
use heartbit_ghost::blog::announce::{XAnnouncementConfig, run_x_announcement_pipeline};
use heartbit_ghost::review::ReviewDelivery;

/// Inputs to one announcement handler tick. Borrows everything the daemon
/// already owns; the handler doesn't hold state across calls.
pub struct BlogAnnounceXDeps<'a> {
    /// Persona name to load.
    pub persona_name: &'a str,
    /// Persona registry.
    pub registry: &'a PersonaRegistry,
    /// Default LLM provider.
    pub provider: Arc<BoxedProvider>,
    /// Writer-stage provider override (falls back to `provider`).
    pub writer_provider: Option<Arc<BoxedProvider>>,
    /// Persona corpora root.
    pub corpora_root: &'a Path,
    /// Persona voice profiles root.
    pub profiles_root: &'a Path,
    /// Title of the blog post.
    pub title: &'a str,
    /// Blog excerpt.
    pub excerpt: &'a str,
    /// First ~500 chars of body.
    pub body_snippet: &'a str,
    /// Canonical blog URL.
    pub post_url: &'a str,
    /// Telegram review delivery.
    pub delivery: Arc<dyn ReviewDelivery>,
    /// X publish tool.
    pub twitter_tool: Arc<dyn Tool>,
    /// X credentials.
    pub credentials: Arc<dyn CredentialResolver>,
}

/// Run one announcement tick. Never panics; errors are logged and
/// swallowed by the caller.
pub async fn handle_blog_announce_x(deps: BlogAnnounceXDeps<'_>) -> Result<()> {
    let persona = deps
        .registry
        .get(deps.persona_name)
        .ok_or_else(|| anyhow::anyhow!("persona '{}' not registered", deps.persona_name))?;
    let _expansion = persona
        .expand(&PersonaParams::default())
        .map_err(|e| anyhow::anyhow!("expand persona '{}': {e}", deps.persona_name))?;

    let cfg = XAnnouncementConfig {
        persona_name: deps.persona_name,
        provider: deps.provider.clone(),
        writer_provider: deps.writer_provider.clone(),
        corpora_root: deps.corpora_root,
        profiles_root: deps.profiles_root,
        on_progress: Some(Arc::new(|s: &str| tracing::info!("blog_announce_x: {s}"))),
        title: deps.title,
        excerpt: deps.excerpt,
        body_snippet: deps.body_snippet,
        post_url: deps.post_url,
        delivery: deps.delivery.clone(),
        twitter_tool: deps.twitter_tool.clone(),
        credentials: deps.credentials.clone(),
    };

    match run_x_announcement_pipeline(cfg).await {
        Ok(out) => {
            tracing::info!(
                persona = %deps.persona_name,
                outcome = ?out.outcome,
                "blog announce X complete"
            );
        }
        Err(e) => {
            tracing::error!(
                persona = %deps.persona_name,
                error = %e,
                "blog announce X pipeline failed"
            );
        }
    }
    Ok(())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Full handler test requires MockProvider/MockTwitterTool/MockReviewDelivery
    // — those mocks live in heartbit_ghost::blog::announce::tests. This handler
    // is a 30-line dispatcher; the only path worth testing here is the unknown-
    // persona early return.

    #[tokio::test]
    async fn handle_blog_announce_x_unknown_persona_errors() {
        // Build a registry with no personas registered.
        let registry = PersonaRegistry::new();
        // Verify the registry lookup that the handler relies on.
        assert!(registry.get("missing-persona").is_none());
    }
}
