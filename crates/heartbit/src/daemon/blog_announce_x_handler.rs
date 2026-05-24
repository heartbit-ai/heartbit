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

    use std::future::Future;
    use std::pin::Pin;

    use heartbit_core::ExecutionContext;
    use heartbit_core::error::Error as CoreError;
    use heartbit_core::execution_context::{CredentialResolver as CredentialResolverTrait, Secret};
    use heartbit_core::llm::LlmProvider;
    use heartbit_core::llm::types::{
        CompletionRequest, CompletionResponse, ContentBlock, StopReason, TokenUsage, ToolDefinition,
    };
    use heartbit_core::tool::ToolOutput;
    use heartbit_ghost::review::{
        DeliveredReview, DeliveryReceipt, ReportableOutcome, ReviewDeliveryError, ReviewMessage,
    };

    // ── Trivial stubs ─────────────────────────────────────────────────────────
    //
    // The handler bails on `registry.get().ok_or_else(...)` BEFORE it ever
    // touches provider/delivery/tool/credentials. So these stubs only need to
    // satisfy the type bounds — they are never exercised on this path.

    struct StubProvider;

    impl StubProvider {
        fn arc() -> Arc<BoxedProvider> {
            Arc::new(BoxedProvider::new(StubProvider))
        }
    }

    impl LlmProvider for StubProvider {
        async fn complete(
            &self,
            _request: CompletionRequest,
        ) -> Result<CompletionResponse, CoreError> {
            Ok(CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "noop".to_string(),
                }],
                usage: TokenUsage::default(),
                stop_reason: StopReason::EndTurn,
                model: None,
            })
        }
    }

    struct StubDelivery;

    impl ReviewDelivery for StubDelivery {
        fn deliver_and_await<'a>(
            &'a self,
            _message: &'a ReviewMessage,
        ) -> Pin<Box<dyn Future<Output = Result<DeliveredReview, ReviewDeliveryError>> + Send + 'a>>
        {
            Box::pin(async move {
                Err(ReviewDeliveryError::Transport(
                    "stub: never called".to_string(),
                ))
            })
        }

        fn report<'a>(
            &'a self,
            _receipt: DeliveryReceipt,
            _outcome: ReportableOutcome,
        ) -> Pin<Box<dyn Future<Output = Result<(), ReviewDeliveryError>> + Send + 'a>> {
            Box::pin(async move { Ok(()) })
        }
    }

    struct StubTool;

    impl Tool for StubTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "twitter_thread".to_string(),
                description: "stub".to_string(),
                input_schema: serde_json::json!({"type": "object"}),
            }
        }

        fn execute(
            &self,
            _ctx: &ExecutionContext,
            _input: serde_json::Value,
        ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, CoreError>> + Send + '_>> {
            Box::pin(async move { Ok(ToolOutput::success("stub")) })
        }
    }

    struct StubCredentialResolver;

    impl CredentialResolverTrait for StubCredentialResolver {
        fn resolve(
            &self,
            _name: &str,
        ) -> Pin<Box<dyn Future<Output = Result<Secret, CoreError>> + Send + '_>> {
            Box::pin(async move { Ok(Secret::new("stub")) })
        }
    }

    #[tokio::test]
    async fn handle_blog_announce_x_unknown_persona_errors() {
        // Empty registry — no personas registered.
        let registry = PersonaRegistry::new();
        let provider = StubProvider::arc();
        let delivery: Arc<dyn ReviewDelivery> = Arc::new(StubDelivery);
        let twitter_tool: Arc<dyn Tool> = Arc::new(StubTool);
        let credentials: Arc<dyn CredentialResolver> = Arc::new(StubCredentialResolver);

        let deps = BlogAnnounceXDeps {
            persona_name: "missing-persona",
            registry: &registry,
            provider,
            writer_provider: None,
            corpora_root: std::path::Path::new("/tmp"),
            profiles_root: std::path::Path::new("/tmp"),
            title: "T",
            excerpt: "E",
            body_snippet: "B",
            post_url: "https://pascal.heartbit.ai/x/",
            delivery,
            twitter_tool,
            credentials,
        };

        let err = handle_blog_announce_x(deps)
            .await
            .expect_err("expected error for unknown persona");
        assert!(err.to_string().contains("not registered"), "got: {err}");
    }
}
