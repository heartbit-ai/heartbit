//! Per-request execution context threaded through tool dispatch.
//!
//! Every `Tool::execute` call receives an `&ExecutionContext`. The context
//! carries tenant/user identity, the workspace root, and resolvers for
//! per-tenant secrets and audit sinks. It is constructed at the request
//! boundary (CLI command, Restate workflow activity, daemon dispatch) and
//! threaded through the agent runner unchanged.

use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use crate::error::Error;

/// Per-request context carried into every tool invocation.
#[derive(Clone, Default)]
pub struct ExecutionContext {
    /// Tenant identifier (multi-tenant deployments). `None` outside of multi-tenant flows.
    pub tenant_id: Option<String>,
    /// User identifier on whose behalf the agent runs. `None` outside of authenticated flows.
    pub user_id: Option<String>,
    /// Workspace root for filesystem-aware tools. `None` when no workspace is configured.
    pub workspace: Option<PathBuf>,
    /// Resolver for per-tenant secrets (API keys, OAuth tokens). `None` when no resolver is configured.
    pub credentials: Option<Arc<dyn CredentialResolver>>,
    /// Sink for tool-level audit records. `None` when no audit sink is configured.
    pub audit_sink: Option<Arc<dyn AuditSink>>,
    /// Snapshot of the conversation at tool-dispatch time. `None` outside an
    /// agent run. Lets introspection tools (e.g. the advisor) see the full
    /// history the way Claude Code forwards it to its advisor.
    pub transcript: Option<Arc<Vec<crate::llm::types::Message>>>,
}

impl std::fmt::Debug for ExecutionContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExecutionContext")
            .field("tenant_id", &self.tenant_id)
            .field("user_id", &self.user_id)
            .field("workspace", &self.workspace)
            .field(
                "credentials",
                &self.credentials.as_ref().map(|_| "<resolver>"),
            )
            .field("audit_sink", &self.audit_sink.as_ref().map(|_| "<sink>"))
            .finish()
    }
}

/// Resolves a named secret (API key, token) for the current tenant.
pub trait CredentialResolver: Send + Sync {
    /// Resolve a secret by logical name (e.g. `"X_API_KEY"`).
    fn resolve(
        &self,
        name: &str,
    ) -> Pin<Box<dyn Future<Output = Result<Secret, Error>> + Send + '_>>;
}

/// Receives per-tool audit records emitted by tools that opt in.
pub trait AuditSink: Send + Sync {
    /// Record a structured audit entry. Implementations must not block.
    fn record(
        &self,
        record: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>>;
}

/// A secret value with redacted `Debug`/`Display` formatting.
#[derive(Clone)]
pub struct Secret(String);

impl Secret {
    /// Wrap a secret string. Use `expose()` to read the inner value.
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Read the inner secret string. Caller is responsible for not logging the result.
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Debug for Secret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Secret(<redacted>)")
    }
}

impl std::fmt::Display for Secret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<redacted>")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn execution_context_default_has_no_identity() {
        let ctx = ExecutionContext::default();
        assert!(ctx.tenant_id.is_none());
        assert!(ctx.user_id.is_none());
        assert!(ctx.workspace.is_none());
        assert!(ctx.credentials.is_none());
        assert!(ctx.audit_sink.is_none());
    }

    #[test]
    fn execution_context_clone_preserves_fields() {
        let ctx = ExecutionContext {
            tenant_id: Some("tenant-1".into()),
            user_id: Some("user-2".into()),
            workspace: Some(PathBuf::from("/tmp/ws")),
            credentials: None,
            audit_sink: None,
            transcript: None,
        };
        let cloned = ctx.clone();
        assert_eq!(cloned.tenant_id.as_deref(), Some("tenant-1"));
        assert_eq!(cloned.user_id.as_deref(), Some("user-2"));
        assert_eq!(cloned.workspace, Some(PathBuf::from("/tmp/ws")));
    }

    #[test]
    fn secret_debug_redacts() {
        let s = Secret::new("super-secret-token");
        let debug = format!("{:?}", s);
        assert!(!debug.contains("super-secret-token"));
        assert!(debug.contains("<redacted>"));
    }

    #[test]
    fn secret_display_redacts() {
        let s = Secret::new("super-secret-token");
        let display = format!("{}", s);
        assert!(!display.contains("super-secret-token"));
        assert!(display.contains("<redacted>"));
    }

    #[test]
    fn secret_expose_returns_inner() {
        let s = Secret::new("super-secret-token");
        assert_eq!(s.expose(), "super-secret-token");
    }

    #[test]
    fn execution_context_debug_does_not_leak_resolver_internals() {
        struct DummyResolver;
        impl CredentialResolver for DummyResolver {
            fn resolve(
                &self,
                _name: &str,
            ) -> Pin<Box<dyn Future<Output = Result<Secret, Error>> + Send + '_>> {
                Box::pin(async { Ok(Secret::new("x")) })
            }
        }

        let ctx = ExecutionContext {
            credentials: Some(Arc::new(DummyResolver)),
            ..ExecutionContext::default()
        };
        let debug = format!("{:?}", ctx);
        assert!(debug.contains("<resolver>"));
        assert!(!debug.contains("DummyResolver"));
    }
}
