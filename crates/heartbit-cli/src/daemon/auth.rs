use std::collections::HashSet;
use std::sync::Arc;

use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::{IntoResponse, Json};

use heartbit::{JwtValidator, UserContext};

use super::types::ServiceAuth;

/// Returns `true` if the caller is allowed to access a task with the given `tenant_id`.
///
/// Rules:
/// - Authenticated service (static bearer `ServiceAuth`): can access all tasks.
/// - Authenticated user (JWT `UserContext`): can only access tasks whose `tenant_id` matches.
/// - Truly unauthenticated (neither): cannot access tenant-scoped tasks.
pub(crate) fn task_tenant_allowed(
    task_tenant_id: Option<&str>,
    user_context: Option<&axum::Extension<UserContext>>,
    service_auth: Option<&axum::Extension<ServiceAuth>>,
) -> bool {
    match (user_context, task_tenant_id) {
        // JWT user: must match task tenant
        (Some(axum::Extension(ctx)), Some(tid)) => tid == ctx.tenant_id,
        // Unauthenticated: blocked from tenant-scoped tasks unless service auth present
        (None, Some(_)) => service_auth.is_some(),
        // No tenant on task, or user has no context restriction
        _ => true,
    }
}

/// Merge bearer tokens from config and an optional environment variable into a token set.
///
/// Returns `None` when no tokens are available (auth disabled).
pub(crate) fn resolve_auth_tokens(
    config_tokens: &[String],
    env_token: Option<String>,
) -> Option<Arc<HashSet<String>>> {
    let mut tokens: HashSet<String> = config_tokens.iter().cloned().collect();
    if let Some(key) = env_token.filter(|k| !k.is_empty()) {
        tokens.insert(key);
    }

    if tokens.is_empty() {
        None
    } else {
        Some(Arc::new(tokens))
    }
}

/// Validate a bearer token from the Authorization header.
///
/// Returns `Ok(())` if the token is valid, or an error tuple with status code and message.
pub(crate) fn validate_bearer_token(
    auth_header: Option<&str>,
    tokens: &HashSet<String>,
) -> Result<(), (StatusCode, &'static str)> {
    match auth_header {
        Some(value) if value.starts_with("Bearer ") => {
            let token = &value[7..];
            if token.is_empty() || !heartbit::auth::ct::contains(tokens, token) {
                Err((StatusCode::UNAUTHORIZED, "invalid bearer token"))
            } else {
                Ok(())
            }
        }
        Some(_) => Err((StatusCode::BAD_REQUEST, "expected Bearer authentication")),
        None => Err((StatusCode::UNAUTHORIZED, "missing Authorization header")),
    }
}

/// Auth middleware that validates Bearer tokens on protected routes.
/// State for JWT auth middleware, including whether JWT is required or optional.
#[derive(Clone)]
pub(crate) struct JwtMiddlewareState {
    pub validator: Arc<JwtValidator>,
    /// When true, requests without a JWT are rejected (JWT is sole auth).
    /// When false, requests without a JWT pass through (bearer token auth handles gating).
    pub required: bool,
}

/// Middleware that validates JWTs and injects `UserContext` into request extensions.
///
/// When a valid JWT is present, the extracted `UserContext` is available to handlers
/// via `Option<axum::Extension<UserContext>>`.
///
/// Behavior depends on `required`:
/// - `required = false`: enrich-only mode — requests without JWTs pass through
/// - `required = true`: requests without a valid JWT are rejected with 401
pub(crate) async fn jwt_auth_middleware(
    axum::extract::State(jwt_state): axum::extract::State<JwtMiddlewareState>,
    mut request: axum::http::Request<axum::body::Body>,
    next: Next,
) -> impl IntoResponse {
    // Extract bearer token from Authorization header
    let auth_header = request
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .map(String::from);

    if let Some(ref header) = auth_header
        && let Some(token) = header.strip_prefix("Bearer ")
    {
        match jwt_state.validator.validate(token).await {
            Ok(mut user_ctx) => {
                tracing::debug!(
                    user_id = %user_ctx.user_id,
                    tenant_id = %user_ctx.tenant_id,
                    "JWT validated, user context injected"
                );
                // Preserve the raw token for RFC 8693 token exchange
                user_ctx.raw_token = Some(token.to_string());
                request.extensions_mut().insert(user_ctx);
            }
            Err(e) => {
                tracing::warn!("JWT validation failed: {e}");
                return (
                    StatusCode::UNAUTHORIZED,
                    Json(serde_json::json!({"error": "invalid or expired token"})),
                )
                    .into_response();
            }
        }
    } else if jwt_state.required {
        return (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({"error": "Authorization header with Bearer token required"})),
        )
            .into_response();
    }

    next.run(request).await.into_response()
}

pub(crate) async fn auth_middleware(
    axum::extract::State(tokens): axum::extract::State<Arc<HashSet<String>>>,
    mut request: axum::http::Request<axum::body::Body>,
    next: Next,
) -> impl IntoResponse {
    let raw_header = request.headers().get(axum::http::header::AUTHORIZATION);

    // Distinguish "absent" from "present but non-UTF-8"
    let auth_header = match raw_header {
        Some(v) => match v.to_str() {
            Ok(s) => Some(s),
            Err(_) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({"error": "invalid Authorization header encoding"})),
                )
                    .into_response();
            }
        },
        None => None,
    };

    match validate_bearer_token(auth_header, &tokens) {
        Ok(()) => {
            // Mark as service-authenticated so handlers can distinguish from truly unauthenticated callers.
            request.extensions_mut().insert(ServiceAuth);
            next.run(request).await.into_response()
        }
        Err((status, msg)) => (status, Json(serde_json::json!({"error": msg}))).into_response(),
    }
}

#[cfg(test)]
mod auth_tests {
    use super::*;

    fn make_tokens(keys: &[&str]) -> HashSet<String> {
        keys.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn valid_token_accepted() {
        let tokens = make_tokens(&["secret-key-1", "secret-key-2"]);
        assert!(validate_bearer_token(Some("Bearer secret-key-1"), &tokens).is_ok());
        assert!(validate_bearer_token(Some("Bearer secret-key-2"), &tokens).is_ok());
    }

    #[test]
    fn invalid_token_rejected() {
        let tokens = make_tokens(&["secret-key-1"]);
        let err = validate_bearer_token(Some("Bearer wrong-key"), &tokens).unwrap_err();
        assert_eq!(err.0, StatusCode::UNAUTHORIZED);
        assert_eq!(err.1, "invalid bearer token");
    }

    #[test]
    fn missing_header_rejected() {
        let tokens = make_tokens(&["secret-key-1"]);
        let err = validate_bearer_token(None, &tokens).unwrap_err();
        assert_eq!(err.0, StatusCode::UNAUTHORIZED);
        assert_eq!(err.1, "missing Authorization header");
    }

    #[test]
    fn non_bearer_rejected() {
        let tokens = make_tokens(&["secret-key-1"]);
        let err = validate_bearer_token(Some("Basic dXNlcjpwYXNz"), &tokens).unwrap_err();
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
        assert_eq!(err.1, "expected Bearer authentication");
    }

    #[test]
    fn empty_bearer_rejected() {
        let tokens = make_tokens(&["secret-key-1"]);
        let err = validate_bearer_token(Some("Bearer "), &tokens).unwrap_err();
        assert_eq!(err.0, StatusCode::UNAUTHORIZED);
        assert_eq!(err.1, "invalid bearer token");
    }

    #[test]
    fn multiple_tokens_all_accepted() {
        let tokens = make_tokens(&["key-alpha", "key-beta", "key-gamma"]);
        assert!(validate_bearer_token(Some("Bearer key-alpha"), &tokens).is_ok());
        assert!(validate_bearer_token(Some("Bearer key-beta"), &tokens).is_ok());
        assert!(validate_bearer_token(Some("Bearer key-gamma"), &tokens).is_ok());
    }

    // --- resolve_auth_tokens tests ---

    #[test]
    fn resolve_merges_config_and_env() {
        let result = resolve_auth_tokens(&["config-key".into()], Some("env-key".into()));
        let tokens = result.unwrap();
        assert!(tokens.contains("config-key"));
        assert!(tokens.contains("env-key"));
    }

    #[test]
    fn resolve_env_only() {
        let result = resolve_auth_tokens(&[], Some("env-key".into()));
        let tokens = result.unwrap();
        assert!(tokens.contains("env-key"));
        assert_eq!(tokens.len(), 1);
    }

    #[test]
    fn resolve_config_only() {
        let result = resolve_auth_tokens(&["config-key".into()], None);
        let tokens = result.unwrap();
        assert!(tokens.contains("config-key"));
        assert_eq!(tokens.len(), 1);
    }

    #[test]
    fn resolve_empty_returns_none() {
        assert!(resolve_auth_tokens(&[], None).is_none());
    }

    #[test]
    fn resolve_ignores_empty_env_var() {
        // HEARTBIT_API_KEY="" should not enable auth
        assert!(resolve_auth_tokens(&[], Some(String::new())).is_none());
    }

    #[test]
    fn resolve_deduplicates() {
        let result = resolve_auth_tokens(&["shared-key".into()], Some("shared-key".into()));
        let tokens = result.unwrap();
        assert_eq!(tokens.len(), 1);
    }

    #[test]
    fn validate_bearer_rejects_equal_length_different_token() {
        let mut tokens = HashSet::new();
        tokens.insert("aaaaaaaa".to_string());
        // Same length, different content — must reject.
        let res = validate_bearer_token(Some("Bearer bbbbbbbb"), &tokens);
        assert!(res.is_err());
    }

    #[test]
    fn validate_bearer_accepts_known_token() {
        let mut tokens = HashSet::new();
        tokens.insert("hunter2".to_string());
        let res = validate_bearer_token(Some("Bearer hunter2"), &tokens);
        assert!(res.is_ok());
    }
}
