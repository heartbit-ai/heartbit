//! `TopicContextProvider` — persona-specific pre-fetch strategy that
//! assembles the topic generator's input context. The agent itself is
//! a singleton (no tools); each persona declares HOW to build its
//! context block via this trait.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use heartbit_core::CredentialResolver;

use super::PostHistoryEntry;

/// Dependencies passed to a [`TopicContextProvider`] during pre-fetch.
pub struct TopicContextDeps<'a> {
    /// Credentials for any X API calls the provider needs (own tweets,
    /// mentions). The provider is responsible for building its own
    /// `XClient` from these.
    pub credentials: Arc<dyn CredentialResolver>,
    /// Operator's X user_id (resolved at config load).
    pub operator_user_id: &'a str,
    /// Recent post history (most-recent-first), passed verbatim into
    /// the rendered context so the generator avoids duplicates.
    pub recent_history: Vec<PostHistoryEntry>,
}

/// Builds the persona-specific block of context that goes into the
/// topic generator's user message. Called by `handle_persona_post`
/// once per tick before the generator is invoked.
pub trait TopicContextProvider: Send + Sync {
    /// Returns a multi-line plain-text block. Empty string is allowed
    /// — the generator falls back to the `topic_brief` from config.
    fn build_context<'a>(
        &'a self,
        deps: &'a TopicContextDeps<'a>,
    ) -> Pin<Box<dyn Future<Output = Result<String, anyhow::Error>> + Send + 'a>>;
}

/// X-grounded topic context for `heartbit-ghost:x`. Implementation in Task 4.
pub struct XGhostTopicContext;

/// Repo-grounded topic context for `heartbit-rs:x`. Implementation in Task 5.
pub struct HeartbitRsXTopicContext;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topic_context_deps_can_be_constructed_with_zero_history() {
        struct StubCreds;
        impl CredentialResolver for StubCreds {
            fn resolve(
                &self,
                _name: &str,
            ) -> Pin<
                Box<
                    dyn Future<Output = Result<heartbit_core::Secret, heartbit_core::Error>>
                        + Send
                        + '_,
                >,
            > {
                Box::pin(async { Ok(heartbit_core::Secret::new("x")) })
            }
        }
        let creds: Arc<dyn CredentialResolver> = Arc::new(StubCreds);
        let deps = TopicContextDeps {
            credentials: creds,
            operator_user_id: "12345",
            recent_history: vec![],
        };
        assert!(deps.recent_history.is_empty());
        assert_eq!(deps.operator_user_id, "12345");
    }
}
