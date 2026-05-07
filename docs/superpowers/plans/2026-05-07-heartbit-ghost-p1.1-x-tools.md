# heartbit-ghost P1.1 — X Tool Family Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship 5 new X tools (`twitter_thread`, `twitter_reply`, `twitter_search`, `twitter_mentions`, `twitter_user`) in `heartbit-ghost` plus extend `heartbit-core::TwitterPostTool` with optional media + alt-text support. New tools resolve credentials at execute-time via `&ExecutionContext::credentials::CredentialResolver`. ~35-40 new tests via wiremock HTTP stubbing.

**Architecture:** New tools live in `crates/heartbit-ghost/src/tools/` and share an `XClient` infrastructure (HTTP + OAuth1 signing + credential resolution + error mapping). Each tool is one file. The existing `twitter_post` keeps its construction-time `TwitterCredentials` model — its media extension is an in-place edit of `crates/heartbit-core/src/tool/builtins/twitter_post.rs`. P1.1 ships the catalog usable standalone; persona expansion is P1.3.

**Tech Stack:** Rust 2024, Tokio, `reqwest` (HTTP, +multipart for media uploads), `oauth1-request = "0.6"` (OAuth1 signing in heartbit-ghost), `serde_json`, `thiserror`, `wiremock = "0.6"` (dev-dep, HTTP mocking). `heartbit-core` from foundation Phase 0 (Tool trait, ExecutionContext, CredentialResolver, Secret).

**Spec:** `docs/superpowers/specs/2026-05-07-heartbit-ghost-p1.1-x-tools-design.md`

**Branch:** `feat/heartbit-ghost-p1.1`

---

## File Structure

### New files
- `crates/heartbit-ghost/src/tools/mod.rs` — module re-exports
- `crates/heartbit-ghost/src/tools/client.rs` — `XClient`, `XApiError`, `format_error` helper, OAuth1 signing
- `crates/heartbit-ghost/src/tools/user.rs` — `TwitterUserTool`
- `crates/heartbit-ghost/src/tools/search.rs` — `TwitterSearchTool`
- `crates/heartbit-ghost/src/tools/mentions.rs` — `TwitterMentionsTool`
- `crates/heartbit-ghost/src/tools/reply.rs` — `TwitterReplyTool`
- `crates/heartbit-ghost/src/tools/thread.rs` — `TwitterThreadTool`

### Modified files
- `crates/heartbit-ghost/Cargo.toml` — add `reqwest`, `serde_json`, `thiserror`, `oauth1-request`, `serde` to `[dependencies]`; add `wiremock` to `[dev-dependencies]`
- `crates/heartbit-ghost/src/lib.rs` — declare `pub mod tools;`
- `crates/heartbit-core/src/tool/builtins/twitter_post.rs` — extend schema with optional `media_url` + `media_alt_text`; add media-upload code path that runs before the existing tweet POST when media is present
- `CHANGELOG.md` — entry under `[Unreleased]`

### Out of scope (explicit)
- `twitter_dm`, `twitter_schedule`, `twitter_metrics` — P1.4
- Wiring tools into `XGhostPersona::expand()` — P1.3
- Persona-level rate limiting — P1.4
- Per-tenant audit logging via `AuditSink` — P1.3+
- Bearer-token auth for read-only endpoints — deferred
- DRY-ing OAuth1 between heartbit-core's twitter_post and heartbit-ghost's XClient — post-merge cleanup

---

## Task 1: Cargo.toml + scaffolding

**Why:** Add dependencies + create the empty `tools` module so subsequent tasks have a place to land. This task is intentionally tiny — pure plumbing, no logic.

**Files:**
- Modify: `crates/heartbit-ghost/Cargo.toml`
- Create: `crates/heartbit-ghost/src/tools/mod.rs`
- Modify: `crates/heartbit-ghost/src/lib.rs`

- [ ] **Step 1: Add dependencies to `crates/heartbit-ghost/Cargo.toml`**

Read the current file first:

```bash
cat crates/heartbit-ghost/Cargo.toml
```

Replace the `[dependencies]` and `[dev-dependencies]` blocks. The full updated `Cargo.toml`:

```toml
[package]
name = "heartbit-ghost"
version.workspace = true
edition = "2024"
authors.workspace = true
license.workspace = true
repository.workspace = true
homepage.workspace = true
description = "Best-in-class autonomous X (Twitter) agent persona for the Heartbit runtime."
keywords = ["agent", "llm", "twitter", "x", "persona"]

[dependencies]
heartbit-core = { path = "../heartbit-core" }
reqwest = { workspace = true, features = ["json", "multipart"] }
serde = { workspace = true }
serde_json = { workspace = true }
thiserror = { workspace = true }
oauth1-request = "0.6"

[dev-dependencies]
tokio = { workspace = true }
wiremock = "0.6"
```

(The existing `description` and `keywords` stay. Only `[dependencies]` adds new entries; `[dev-dependencies]` adds `wiremock`.)

- [ ] **Step 2: Verify `reqwest`'s `multipart` feature is buildable**

```bash
cargo build -p heartbit-ghost 2>&1 | tail -5
```

Expected: clean build. If `reqwest` doesn't expose the `multipart` feature in this workspace, the workspace `reqwest = ...` dep block in root `Cargo.toml` may need a feature added. Check with:

```bash
grep -A 2 "^reqwest" Cargo.toml
```

If the workspace `reqwest` already includes `multipart` or supports it, no further change. If not, edit root `Cargo.toml` to add `"multipart"` to `reqwest`'s `features` list. Most reqwest versions enable multipart via the feature flag.

- [ ] **Step 3: Create `crates/heartbit-ghost/src/tools/mod.rs`**

```rust
//! X (Twitter) tool family for the heartbit-ghost persona.
//!
//! Each tool implements `heartbit_core::Tool` and resolves credentials at
//! execute-time via `ExecutionContext::credentials`. Tools share `XClient`
//! for HTTP, OAuth1 signing, and error mapping.

pub mod client;

pub use client::{XApiError, XClient, format_error};
```

(Tools 3-7 will append `pub mod user;`, `pub mod search;`, etc. as they land.)

- [ ] **Step 4: Declare the module from `lib.rs`**

In `crates/heartbit-ghost/src/lib.rs`, find the existing module declarations near the top. After `pub use heartbit_core::{...}` imports (or wherever the module declarations sit), add:

```rust
pub mod tools;
```

(The existing `XGhostPersona`, `register`, etc. stay unchanged.)

- [ ] **Step 5: Verify the workspace still builds**

```bash
cargo check -p heartbit-ghost
```

Expected: error about missing `tools/client.rs` — the `mod.rs` declares `pub mod client;` but the file doesn't exist yet. That's expected; Task 2 creates it.

To verify the SCAFFOLDING is correct (not the missing client), temporarily comment out `pub mod client;` in `tools/mod.rs`, re-run `cargo check -p heartbit-ghost`, expect clean. Then uncomment.

Better: skip the cargo check until Task 2 lands a stub `client.rs`. Continue to step 6.

- [ ] **Step 6: Commit (Cargo.toml + empty tools dir)**

We'll wait to commit until Task 2 lands `client.rs` so the crate compiles. **Do not commit at this step** — the commit boundary for this task is rolled into Task 2's commit.

(Skip step 6 — it intentionally has no commit.)

---

## Task 2: `XClient` + `XApiError` + OAuth1 signing + credential resolution

**Why:** All 5 new tools use this shared client. Get it right once and the per-tool work is mostly schemas + endpoint mapping.

**Files:**
- Create: `crates/heartbit-ghost/src/tools/client.rs`

- [ ] **Step 1: Write the failing test (XApiError variants)**

Create `crates/heartbit-ghost/src/tools/client.rs` with the test scaffolding (no impl yet — tests fail to compile):

```rust
//! X HTTP client shared by all heartbit-ghost X tools.
//!
//! - Resolves OAuth1 credentials from `ExecutionContext::credentials` at execute-time
//! - Signs every request via `oauth1-request`
//! - Maps X API responses (200 / 401 / 429 / 4xx / 5xx) to typed `XApiError` variants
//! - `format_error` produces user-friendly tool error messages

use std::sync::Arc;

use heartbit_core::{ExecutionContext, Secret};
use serde::{Serialize, de::DeserializeOwned};
use thiserror::Error;

/// X API base URL — overridden in tests via `XClient::new()`.
const X_API_BASE_URL: &str = "https://api.twitter.com";

/// Strongly typed errors from X API calls.
#[derive(Debug, Error)]
pub enum XApiError {
    /// `ExecutionContext::credentials` is `None`.
    #[error("no credential resolver configured; X tools require ExecutionContext::credentials to be set")]
    MissingResolver,
    /// Resolver returned an error for a specific credential name.
    #[error("credential resolution failed for '{name}': {source}")]
    CredentialResolutionFailed {
        name: String,
        #[source]
        source: heartbit_core::Error,
    },
    /// X API returned 401 Unauthorized.
    #[error("X auth failed (401): {0}")]
    Unauthenticated(String),
    /// X API returned 429 Too Many Requests.
    #[error("X rate limited; retry after {retry_after_secs}s")]
    RateLimited { retry_after_secs: u64 },
    /// X API returned a 4xx or 5xx that isn't 401 or 429.
    #[error("X API error ({status}): {message}")]
    ApiError { status: u16, message: String },
    /// Network error (connect failure, TLS, timeout, etc.).
    #[error("network error: {0}")]
    Network(String),
    /// Response parsing failed (unexpected payload shape).
    #[error("response parse error: {0}")]
    ParseError(String),
}

/// Map `XApiError` to a user-friendly tool error message.
pub fn format_error(err: &XApiError) -> String {
    match err {
        XApiError::MissingResolver => err.to_string(),
        XApiError::CredentialResolutionFailed { .. } => err.to_string(),
        XApiError::Unauthenticated(_) => "X auth failed; check credentials".to_string(),
        XApiError::RateLimited { retry_after_secs } => {
            format!("rate limited; retry after {retry_after_secs}s")
        }
        XApiError::ApiError { status, message } => format!("X API error ({status}): {message}"),
        XApiError::Network(msg) => format!("network error: {msg}"),
        XApiError::ParseError(msg) => format!("X response parse error: {msg}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_error_unauthenticated() {
        let err = XApiError::Unauthenticated("invalid token".into());
        assert_eq!(format_error(&err), "X auth failed; check credentials");
    }

    #[test]
    fn format_error_rate_limited_includes_retry_after() {
        let err = XApiError::RateLimited { retry_after_secs: 30 };
        assert_eq!(format_error(&err), "rate limited; retry after 30s");
    }

    #[test]
    fn format_error_api_error_includes_status_and_message() {
        let err = XApiError::ApiError {
            status: 400,
            message: "bad request".into(),
        };
        assert_eq!(format_error(&err), "X API error (400): bad request");
    }

    #[test]
    fn format_error_missing_resolver_uses_thiserror_message() {
        let err = XApiError::MissingResolver;
        let msg = format_error(&err);
        assert!(msg.contains("no credential resolver"));
        assert!(msg.contains("ExecutionContext::credentials"));
    }
}
```

This file compiles and the 4 tests pass — but the `XClient` struct doesn't exist yet. Step 2 adds it.

- [ ] **Step 2: Run the tests — should pass (4 error-formatting tests)**

```bash
cargo test -p heartbit-ghost --lib client::tests::format_error 2>&1 | tail -5
```

Expected: 4 passed.

- [ ] **Step 3: Add the `XClient` struct + `new()` + `from_context()` skeleton**

Append to `client.rs`:

```rust
/// HTTP client for X API. Resolves OAuth1 credentials from `ExecutionContext`,
/// signs every request, and maps X-specific errors to `XApiError`.
///
/// Construct via `XClient::from_context(ctx).await?` in production. Tests use
/// `XClient::new(base_url, ...)` to point the client at a wiremock server.
pub struct XClient {
    http: reqwest::Client,
    base_url: String,
    consumer_key: Secret,
    consumer_secret: Secret,
    access_token: Secret,
    access_token_secret: Secret,
}

impl XClient {
    /// Construct directly with the 4 credentials and a base URL.
    /// Used by tests pointing at wiremock; production code uses `from_context()`.
    pub fn new(
        base_url: impl Into<String>,
        consumer_key: Secret,
        consumer_secret: Secret,
        access_token: Secret,
        access_token_secret: Secret,
    ) -> Result<Self, XApiError> {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|e| XApiError::Network(format!("failed to build HTTP client: {e}")))?;
        Ok(Self {
            http,
            base_url: base_url.into(),
            consumer_key,
            consumer_secret,
            access_token,
            access_token_secret,
        })
    }

    /// Construct from an `ExecutionContext`, resolving the 4 OAuth1 credentials.
    pub async fn from_context(ctx: &ExecutionContext) -> Result<Self, XApiError> {
        let resolver = ctx.credentials.as_ref().ok_or(XApiError::MissingResolver)?;
        let consumer_key = resolve(resolver, "X_CONSUMER_KEY").await?;
        let consumer_secret = resolve(resolver, "X_CONSUMER_SECRET").await?;
        let access_token = resolve(resolver, "X_ACCESS_TOKEN").await?;
        let access_token_secret = resolve(resolver, "X_ACCESS_TOKEN_SECRET").await?;
        Self::new(
            X_API_BASE_URL,
            consumer_key,
            consumer_secret,
            access_token,
            access_token_secret,
        )
    }
}

/// Helper: resolve a single credential name, mapping the error variant.
async fn resolve(
    resolver: &Arc<dyn heartbit_core::CredentialResolver>,
    name: &str,
) -> Result<Secret, XApiError> {
    resolver
        .resolve(name)
        .await
        .map_err(|e| XApiError::CredentialResolutionFailed {
            name: name.to_string(),
            source: e,
        })
}
```

- [ ] **Step 4: Add tests for `from_context` + missing-resolver path**

In the test mod, add:

```rust
    use std::future::Future;
    use std::pin::Pin;

    /// Test-only resolver: returns the input name as the secret value.
    /// Useful for asserting that `from_context` resolved each name.
    struct EchoResolver;

    impl heartbit_core::CredentialResolver for EchoResolver {
        fn resolve(
            &self,
            name: &str,
        ) -> Pin<Box<dyn Future<Output = Result<Secret, heartbit_core::Error>> + Send + '_>> {
            let name = name.to_string();
            Box::pin(async move { Ok(Secret::new(format!("secret-for-{name}"))) })
        }
    }

    /// Test-only resolver that always errors.
    struct FailingResolver;

    impl heartbit_core::CredentialResolver for FailingResolver {
        fn resolve(
            &self,
            _name: &str,
        ) -> Pin<Box<dyn Future<Output = Result<Secret, heartbit_core::Error>> + Send + '_>> {
            Box::pin(async {
                Err(heartbit_core::Error::Agent("simulated resolver failure".into()))
            })
        }
    }

    #[tokio::test]
    async fn from_context_returns_missing_resolver_when_credentials_none() {
        let ctx = ExecutionContext::default();
        let result = XClient::from_context(&ctx).await;
        assert!(matches!(result, Err(XApiError::MissingResolver)));
    }

    #[tokio::test]
    async fn from_context_resolves_all_four_credentials() {
        let ctx = ExecutionContext {
            credentials: Some(Arc::new(EchoResolver)),
            ..ExecutionContext::default()
        };
        let client = XClient::from_context(&ctx).await.expect("construction succeeds");
        assert_eq!(client.consumer_key.expose(), "secret-for-X_CONSUMER_KEY");
        assert_eq!(client.consumer_secret.expose(), "secret-for-X_CONSUMER_SECRET");
        assert_eq!(client.access_token.expose(), "secret-for-X_ACCESS_TOKEN");
        assert_eq!(client.access_token_secret.expose(), "secret-for-X_ACCESS_TOKEN_SECRET");
    }

    #[tokio::test]
    async fn from_context_propagates_resolver_error() {
        let ctx = ExecutionContext {
            credentials: Some(Arc::new(FailingResolver)),
            ..ExecutionContext::default()
        };
        let result = XClient::from_context(&ctx).await;
        match result {
            Err(XApiError::CredentialResolutionFailed { name, .. }) => {
                // The first resolved name is X_CONSUMER_KEY; failure surfaces there.
                assert_eq!(name, "X_CONSUMER_KEY");
            }
            other => panic!("expected CredentialResolutionFailed, got {:?}", other),
        }
    }
```

- [ ] **Step 5: Run the tests — 7 should pass (4 format_error + 3 from_context)**

```bash
cargo test -p heartbit-ghost --lib client 2>&1 | tail -10
```

Expected: 7 passed.

- [ ] **Step 6: Add `get_json` and `post_json` methods with OAuth1 signing**

Add to the `impl XClient` block:

```rust
    /// GET `<base_url><path>` with optional query params; sign via OAuth1; parse JSON response.
    pub async fn get_json<T: DeserializeOwned>(
        &self,
        path: &str,
        query: &[(&str, &str)],
    ) -> Result<T, XApiError> {
        let url = format!("{}{}", self.base_url, path);
        let auth_header = self.sign("GET", &url, query)?;
        let mut req = self.http.get(&url).header("Authorization", auth_header);
        if !query.is_empty() {
            req = req.query(query);
        }
        let response = req
            .send()
            .await
            .map_err(|e| XApiError::Network(e.to_string()))?;
        Self::parse_response(response).await
    }

    /// POST JSON body to `<base_url><path>`; sign via OAuth1; parse JSON response.
    pub async fn post_json<B: Serialize, T: DeserializeOwned>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<T, XApiError> {
        let url = format!("{}{}", self.base_url, path);
        // OAuth1 signs only oauth_* parameters for POST application/json bodies
        // (the body is NOT included in the signature base string for v2 endpoints).
        let auth_header = self.sign("POST", &url, &[])?;
        let response = self
            .http
            .post(&url)
            .header("Authorization", auth_header)
            .json(body)
            .send()
            .await
            .map_err(|e| XApiError::Network(e.to_string()))?;
        Self::parse_response(response).await
    }

    /// Build the OAuth1 Authorization header for a request.
    fn sign(
        &self,
        method: &str,
        url: &str,
        query: &[(&str, &str)],
    ) -> Result<String, XApiError> {
        // The oauth1-request crate's API may differ slightly; the implementer
        // adapts to whatever the crate exposes in 0.6.x. Below is the contract:
        //   - inputs: method, url, query params, our 4 secrets
        //   - output: a String to use as the value of the Authorization header,
        //     starting with "OAuth ..."
        //
        // If oauth1-request 0.6 does not expose a clean way to do this, fall
        // back to the inline OAuth1 implementation already in
        // crates/heartbit-core/src/tool/builtins/twitter_post.rs (functions
        // `percent_encode` and `build_oauth_header` — copy the pattern).
        oauth1_signing::build_authorization_header(
            method,
            url,
            query,
            self.consumer_key.expose(),
            self.consumer_secret.expose(),
            self.access_token.expose(),
            self.access_token_secret.expose(),
        )
        .map_err(|e| XApiError::Network(format!("OAuth1 signing failed: {e}")))
    }

    /// Parse an HTTP response into either `T` (on 2xx) or a typed `XApiError`.
    async fn parse_response<T: DeserializeOwned>(
        response: reqwest::Response,
    ) -> Result<T, XApiError> {
        let status = response.status();
        if status.is_success() {
            return response
                .json::<T>()
                .await
                .map_err(|e| XApiError::ParseError(e.to_string()));
        }
        let retry_after = response
            .headers()
            .get(reqwest::header::RETRY_AFTER)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok());
        let status_code = status.as_u16();
        let body_text = response.text().await.unwrap_or_default();
        match status_code {
            401 => Err(XApiError::Unauthenticated(body_text)),
            429 => Err(XApiError::RateLimited {
                retry_after_secs: retry_after.unwrap_or(60),
            }),
            _ => {
                let message = extract_x_error_message(&body_text)
                    .unwrap_or_else(|| body_text.clone());
                Err(XApiError::ApiError {
                    status: status_code,
                    message,
                })
            }
        }
    }
}

/// Try to parse X's error JSON shape (`{"errors": [{"message": "..."}]}` or
/// `{"detail": "..."}`); returns `None` if the body isn't recognisable JSON.
fn extract_x_error_message(body: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(body).ok()?;
    if let Some(detail) = value.get("detail").and_then(|d| d.as_str()) {
        return Some(detail.to_string());
    }
    if let Some(errors) = value.get("errors").and_then(|e| e.as_array())
        && let Some(first) = errors.first()
        && let Some(message) = first.get("message").and_then(|m| m.as_str())
    {
        return Some(message.to_string());
    }
    None
}

/// Inline OAuth1 signing module. The implementer chooses between two paths:
/// (a) wrap the `oauth1-request = "0.6"` crate, or
/// (b) port the inline implementation from heartbit-core's twitter_post.rs.
///
/// Both must produce the same `Authorization: OAuth oauth_consumer_key="...", ...`
/// header expected by the X API.
mod oauth1_signing {
    /// Build the `OAuth ...` Authorization header value.
    ///
    /// Returns the header VALUE only (without the leading `Authorization:` name).
    pub fn build_authorization_header(
        method: &str,
        url: &str,
        query: &[(&str, &str)],
        consumer_key: &str,
        consumer_secret: &str,
        access_token: &str,
        access_token_secret: &str,
    ) -> Result<String, String> {
        // Implementer's choice:
        //
        // OPTION A (preferred): use oauth1-request 0.6.x API. Look at the crate's
        // docs (`cargo doc --open -p oauth1-request`) — typical usage is:
        //   - construct a `Token` from the 4 strings
        //   - call a `sign` helper with method/url/query → returns header
        //
        // OPTION B (fallback): port the inline implementation. Copy these helpers
        // from `crates/heartbit-core/src/tool/builtins/twitter_post.rs`:
        //   - `percent_encode`
        //   - `build_oauth_header`
        // Adapt `build_oauth_header` to accept `method` (it currently hardcodes "POST")
        // and to include `query` params in the signature base string when present.
        //
        // OPTION C (emergency): raise if neither works, surface a concrete error.

        // Pick option A or B during implementation. Test the chosen path with the
        // wiremock-based round-trip test in `client::tests::sign_round_trip_via_wiremock`
        // (added in Step 8 below).

        let _ = (method, url, query, consumer_key, consumer_secret, access_token, access_token_secret);
        unimplemented!("implementer picks oauth1-request crate or inline port — see comment above")
    }
}
```

**Implementer note**: the `unimplemented!()` here is a deliberate plan placeholder. The Step 8 wiremock test will fail until the implementer fills in the real signing logic. The plan does not prescribe a specific oauth1-request 0.6 API call because the crate's exact shape may have evolved; the implementer reads `cargo doc -p oauth1-request --open` and picks the right call.

- [ ] **Step 7: Run tests — the 7 from before still pass; signing not yet exercised**

```bash
cargo test -p heartbit-ghost --lib client 2>&1 | tail -5
```

Expected: 7 passed (the format_error + from_context tests don't need OAuth1 signing).

- [ ] **Step 8: Write a wiremock-based round-trip test for the GET path**

Add to the test mod:

```rust
    use wiremock::matchers::{header_exists, method, path as wm_path};
    use wiremock::{Mock, MockServer, ResponseTemplate};
    use serde::Deserialize;

    #[derive(Debug, Deserialize, PartialEq)]
    struct TestResponse {
        ok: bool,
    }

    fn test_client(server_uri: &str) -> XClient {
        XClient::new(
            server_uri,
            Secret::new("ck"),
            Secret::new("cs"),
            Secret::new("at"),
            Secret::new("ats"),
        )
        .expect("client builds")
    }

    #[tokio::test]
    async fn get_json_signs_and_returns_parsed_body() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/test/endpoint"))
            .and(header_exists("authorization"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({"ok": true})),
            )
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let result: TestResponse = client
            .get_json("/test/endpoint", &[])
            .await
            .expect("happy path");
        assert_eq!(result, TestResponse { ok: true });
    }

    #[tokio::test]
    async fn get_json_returns_unauthenticated_on_401() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/auth/test"))
            .respond_with(ResponseTemplate::new(401).set_body_string("invalid token"))
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let result: Result<serde_json::Value, _> = client.get_json("/auth/test", &[]).await;
        assert!(matches!(result, Err(XApiError::Unauthenticated(_))));
    }

    #[tokio::test]
    async fn get_json_returns_rate_limited_with_retry_after() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/rate/test"))
            .respond_with(ResponseTemplate::new(429).insert_header("Retry-After", "30"))
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let result: Result<serde_json::Value, _> = client.get_json("/rate/test", &[]).await;
        match result {
            Err(XApiError::RateLimited { retry_after_secs }) => {
                assert_eq!(retry_after_secs, 30);
            }
            other => panic!("expected RateLimited, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn get_json_extracts_x_error_message_from_4xx_body() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/error/test"))
            .respond_with(
                ResponseTemplate::new(400)
                    .set_body_json(serde_json::json!({"detail": "bad query"})),
            )
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let result: Result<serde_json::Value, _> = client.get_json("/error/test", &[]).await;
        match result {
            Err(XApiError::ApiError { status, message }) => {
                assert_eq!(status, 400);
                assert_eq!(message, "bad query");
            }
            other => panic!("expected ApiError, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn post_json_signs_and_returns_parsed_body() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(wm_path("/post/test"))
            .and(header_exists("authorization"))
            .respond_with(
                ResponseTemplate::new(201).set_body_json(serde_json::json!({"ok": true})),
            )
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let body = serde_json::json!({"text": "hello"});
        let result: TestResponse = client
            .post_json("/post/test", &body)
            .await
            .expect("happy path");
        assert_eq!(result, TestResponse { ok: true });
    }
```

- [ ] **Step 9: Run the new tests — they fail until OAuth1 signing is implemented**

```bash
cargo test -p heartbit-ghost --lib client 2>&1 | tail -10
```

Expected: 5 wiremock tests fail with `unimplemented!()` panic from `oauth1_signing::build_authorization_header`. The 7 pre-existing tests still pass.

- [ ] **Step 10: Implement `oauth1_signing::build_authorization_header`**

Open `cargo doc -p oauth1-request --open` in a browser (or `cargo doc -p oauth1-request --no-deps` and read the generated HTML in `target/doc/oauth1_request/`). Identify the API for building an Authorization header.

If `oauth1-request 0.6` is straightforward, use it. If awkward, **fall back to porting the inline implementation from heartbit-core's twitter_post.rs**:

1. Copy `percent_encode` from `crates/heartbit-core/src/tool/builtins/twitter_post.rs`
2. Copy `build_oauth_header` and adapt to:
   - Accept `method: &str` instead of hardcoded "POST"
   - Include the `query` params in the signature base string (sorted with the oauth_* params before signing)
3. Wire `build_authorization_header` to use these helpers
4. Add `hmac = "0.12"`, `sha1 = "0.10"`, `base64 = "0.21"` to `crates/heartbit-ghost/Cargo.toml` `[dependencies]` (if not already there transitively)

Replace the `unimplemented!()` body. The contract: produce a string starting with `OAuth ` followed by the canonical comma-separated list of OAuth1 parameters (consumer_key, nonce, signature, signature_method, timestamp, token, version).

- [ ] **Step 11: Run all client tests — should now pass**

```bash
cargo test -p heartbit-ghost --lib client 2>&1 | tail -10
```

Expected: 12 passed (7 pre-existing + 5 wiremock).

- [ ] **Step 12: Workspace quality gate**

```bash
cargo fmt -p heartbit-ghost -- --check
cargo clippy -p heartbit-ghost -- -D warnings
cargo test -p heartbit-ghost 2>&1 | tail -3
```

All clean. 12 tests pass.

- [ ] **Step 13: Commit (Task 1 + Task 2 combined)**

```bash
git add crates/heartbit-ghost/Cargo.toml crates/heartbit-ghost/src/
git commit -m "$(cat <<'EOF'
feat(ghost): add XClient + OAuth1 signing + credential resolution (P1.1)

Tasks 1-2 of the heartbit-ghost P1.1 sub-phase. Adds:
- reqwest, serde_json, thiserror, oauth1-request to heartbit-ghost deps
- wiremock dev-dep for HTTP mocking
- crates/heartbit-ghost/src/tools/ module
- XApiError enum (MissingResolver, CredentialResolutionFailed,
  Unauthenticated, RateLimited, ApiError, Network, ParseError)
- XClient with from_context() (resolves 4 OAuth1 creds via
  ExecutionContext::credentials), get_json(), post_json()
- format_error() helper mapping XApiError -> user-facing strings

12 tests: 4 format_error variants + 3 from_context paths +
5 wiremock round-trip tests covering 200 / 401 / 429 / 4xx body
extraction / POST. The 5 X tools (user/search/mentions/reply/thread)
land in subsequent tasks.

Refs: docs/superpowers/specs/2026-05-07-heartbit-ghost-p1.1-x-tools-design.md F-AD-1, F-AD-3, F-AD-5
EOF
)"
```

---

## Task 3: `TwitterUserTool` (read-only — simplest tool)

**Why:** Smallest tool; uses one GET endpoint with no pagination. Validates the XClient + Tool integration.

**Files:**
- Create: `crates/heartbit-ghost/src/tools/user.rs`
- Modify: `crates/heartbit-ghost/src/tools/mod.rs`

- [ ] **Step 1: Create the tool with full Tool trait + tests stub**

Create `crates/heartbit-ghost/src/tools/user.rs`:

```rust
//! `twitter_user` — look up an X user by handle.

use std::future::Future;
use std::pin::Pin;

use heartbit_core::{ExecutionContext, Tool, ToolOutput};
use heartbit_core::llm::types::ToolDefinition;
use serde::{Deserialize, Serialize};

use crate::tools::client::{XApiError, XClient, format_error};

#[derive(Debug, Deserialize)]
struct UserInput {
    /// Handle without leading `@`.
    handle: String,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct UserResponse {
    id: String,
    name: String,
    username: String,
    description: Option<String>,
    public_metrics: Option<UserMetrics>,
    created_at: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct UserMetrics {
    followers_count: Option<u64>,
    following_count: Option<u64>,
    tweet_count: Option<u64>,
}

/// X v2 GET /2/users/by/username/:handle response wrapper.
#[derive(Debug, Deserialize)]
struct UserApiResponse {
    data: UserResponse,
}

/// Look up an X user by handle.
pub struct TwitterUserTool;

impl Default for TwitterUserTool {
    fn default() -> Self {
        Self::new()
    }
}

impl TwitterUserTool {
    /// Create the tool. Credentials are resolved at execute-time via
    /// `ExecutionContext::credentials`.
    pub fn new() -> Self {
        Self
    }
}

impl Tool for TwitterUserTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "twitter_user".into(),
            description: "Look up an X (Twitter) user by handle. Returns id, name, description, and follower/following/tweet counts.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "handle": {
                        "type": "string",
                        "description": "User handle without the leading @, e.g. \"karpathy\""
                    }
                },
                "required": ["handle"]
            }),
        }
    }

    fn execute(
        &self,
        ctx: &ExecutionContext,
        input: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, heartbit_core::Error>> + Send + '_>> {
        let ctx = ctx.clone();
        Box::pin(async move {
            let parsed: UserInput = match serde_json::from_value(input) {
                Ok(v) => v,
                Err(e) => return Ok(ToolOutput::error(format!("invalid input: {e}"))),
            };
            let client = match XClient::from_context(&ctx).await {
                Ok(c) => c,
                Err(e) => return Ok(ToolOutput::error(format_error(&e))),
            };
            match call_x(&client, &parsed).await {
                Ok(out) => {
                    let json = serde_json::to_string(&out)
                        .unwrap_or_else(|_| "<serialize error>".to_string());
                    Ok(ToolOutput::success(json))
                }
                Err(e) => Ok(ToolOutput::error(format_error(&e))),
            }
        })
    }
}

async fn call_x(client: &XClient, input: &UserInput) -> Result<UserResponse, XApiError> {
    let path = format!("/2/users/by/username/{}", input.handle);
    let response: UserApiResponse = client
        .get_json(
            &path,
            &[(
                "user.fields",
                "description,public_metrics,created_at",
            )],
        )
        .await?;
    Ok(response.data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use heartbit_core::{CredentialResolver, Secret};
    use std::sync::Arc;
    use wiremock::matchers::{method, path as wm_path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    /// Resolver that returns canned secrets.
    struct CannedResolver;
    impl CredentialResolver for CannedResolver {
        fn resolve(
            &self,
            name: &str,
        ) -> Pin<Box<dyn Future<Output = Result<Secret, heartbit_core::Error>> + Send + '_>> {
            let n = name.to_string();
            Box::pin(async move { Ok(Secret::new(format!("test-{n}"))) })
        }
    }

    /// Build an ExecutionContext + XClient pointing at the wiremock server.
    /// Returns the tool already constructed; tests call .execute(&ctx, ...).
    async fn ctx_for_server(server_uri: &str) -> (ExecutionContext, TwitterUserTool) {
        // Tools normally resolve credentials at execute-time AND use the production
        // X URL constant. We override by injecting a custom XClient. Since
        // TwitterUserTool's call_x() takes &XClient, we'd need a way to inject the
        // test client. Two options:
        //   (a) refactor call_x() to take &XClient
        //   (b) run the full execute() path but somehow point the tool at wiremock
        //
        // We use (a): the helper function takes &XClient, and tests can call
        // call_x(&test_client, &UserInput { ... }) directly without going through
        // execute(). The execute() integration test (one per tool) uses the
        // ExecutionContext + a custom resolver to verify the full flow once.
        //
        // For the integration test we point ALL X tools at wiremock by overriding
        // the production URL constant via an env var or a builder method on
        // XClient. Simpler: set a custom credential resolver and use a
        // CONFIGURABLE base URL via `XClient::new()` directly inside the test.
        let _ = server_uri;
        let ctx = ExecutionContext {
            credentials: Some(Arc::new(CannedResolver)),
            ..ExecutionContext::default()
        };
        (ctx, TwitterUserTool::new())
    }

    fn test_client(server_uri: &str) -> XClient {
        XClient::new(
            server_uri,
            Secret::new("ck"),
            Secret::new("cs"),
            Secret::new("at"),
            Secret::new("ats"),
        )
        .expect("test client builds")
    }

    #[tokio::test]
    async fn user_lookup_happy_path() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/2/users/by/username/karpathy"))
            .and(query_param("user.fields", "description,public_metrics,created_at"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": {
                    "id": "33836629",
                    "name": "Andrej Karpathy",
                    "username": "karpathy",
                    "description": "AI/ML.",
                    "public_metrics": {
                        "followers_count": 1000000,
                        "following_count": 100,
                        "tweet_count": 5000
                    },
                    "created_at": "2009-04-21T22:00:00.000Z"
                }
            })))
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let input = UserInput { handle: "karpathy".into() };
        let result = call_x(&client, &input).await.expect("happy path");
        assert_eq!(result.id, "33836629");
        assert_eq!(result.username, "karpathy");
        assert_eq!(result.public_metrics.unwrap().followers_count, Some(1000000));
    }

    #[tokio::test]
    async fn user_lookup_returns_unauthenticated_on_401() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/2/users/by/username/anyhandle"))
            .respond_with(ResponseTemplate::new(401))
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let input = UserInput { handle: "anyhandle".into() };
        let result = call_x(&client, &input).await;
        assert!(matches!(result, Err(XApiError::Unauthenticated(_))));
    }

    #[tokio::test]
    async fn user_lookup_returns_rate_limited_on_429() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/2/users/by/username/anyhandle"))
            .respond_with(ResponseTemplate::new(429).insert_header("Retry-After", "60"))
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let input = UserInput { handle: "anyhandle".into() };
        let result = call_x(&client, &input).await;
        match result {
            Err(XApiError::RateLimited { retry_after_secs }) => assert_eq!(retry_after_secs, 60),
            other => panic!("expected RateLimited, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn user_lookup_surfaces_x_error_on_4xx() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/2/users/by/username/nosuchuser"))
            .respond_with(ResponseTemplate::new(404).set_body_json(serde_json::json!({
                "errors": [{"message": "user not found"}]
            })))
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let input = UserInput { handle: "nosuchuser".into() };
        let result = call_x(&client, &input).await;
        match result {
            Err(XApiError::ApiError { status, message }) => {
                assert_eq!(status, 404);
                assert_eq!(message, "user not found");
            }
            other => panic!("expected ApiError, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn execute_returns_error_when_credentials_missing() {
        let tool = TwitterUserTool::new();
        let ctx = ExecutionContext::default();  // no resolver
        let result = tool
            .execute(&ctx, serde_json::json!({"handle": "anything"}))
            .await
            .expect("Tool::execute returns Ok with ToolOutput::error on credential failure");
        assert!(result.is_error);
        assert!(result.content.contains("no credential resolver"));
    }

    #[tokio::test]
    async fn definition_has_stable_name() {
        let tool = TwitterUserTool::new();
        let def = tool.definition();
        assert_eq!(def.name, "twitter_user");
    }
}
```

- [ ] **Step 2: Add the module to `tools/mod.rs`**

Edit `crates/heartbit-ghost/src/tools/mod.rs` — add:

```rust
pub mod user;

pub use user::TwitterUserTool;
```

Right below the existing `pub mod client;` and `pub use client::{...};` lines.

- [ ] **Step 3: Run the tests**

```bash
cargo test -p heartbit-ghost --lib user 2>&1 | tail -10
```

Expected: 6 passed.

- [ ] **Step 4: Workspace quality gate**

```bash
cargo fmt -p heartbit-ghost -- --check
cargo clippy -p heartbit-ghost -- -D warnings
```

All clean.

- [ ] **Step 5: Commit**

```bash
git add crates/heartbit-ghost/src/tools/user.rs crates/heartbit-ghost/src/tools/mod.rs
git commit -m "$(cat <<'EOF'
feat(ghost): TwitterUserTool — look up X users by handle (P1.1)

Read-only tool calling GET /2/users/by/username/:handle with
user.fields=description,public_metrics,created_at. Returns id, name,
username, description, follower/following/tweet counts, created_at.

6 tests: happy path, 401, 429, 404 with X error JSON, missing
credentials path, definition() stability.

Refs: docs/superpowers/specs/2026-05-07-heartbit-ghost-p1.1-x-tools-design.md §1
EOF
)"
```

---

## Task 4: `TwitterSearchTool` (read-only with pagination)

**Why:** Read-only with `since_id` / `next_token` pagination. Establishes the pagination pattern reused by `TwitterMentionsTool`.

**Files:**
- Create: `crates/heartbit-ghost/src/tools/search.rs`
- Modify: `crates/heartbit-ghost/src/tools/mod.rs`

- [ ] **Step 1: Create the tool**

Create `crates/heartbit-ghost/src/tools/search.rs`:

```rust
//! `twitter_search` — search X recent tweets matching a query.

use std::future::Future;
use std::pin::Pin;

use heartbit_core::{ExecutionContext, Tool, ToolOutput};
use heartbit_core::llm::types::ToolDefinition;
use serde::{Deserialize, Serialize};

use crate::tools::client::{XApiError, XClient, format_error};

#[derive(Debug, Deserialize)]
struct SearchInput {
    query: String,
    #[serde(default = "default_max_results")]
    max_results: u32,
    #[serde(default)]
    since_id: Option<String>,
}

fn default_max_results() -> u32 {
    10
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct Tweet {
    id: String,
    text: String,
    author_id: Option<String>,
    created_at: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SearchOutput {
    tweets: Vec<Tweet>,
    next_token: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SearchApiResponse {
    #[serde(default)]
    data: Vec<Tweet>,
    #[serde(default)]
    meta: SearchMeta,
}

#[derive(Debug, Default, Deserialize)]
struct SearchMeta {
    #[serde(default)]
    next_token: Option<String>,
}

pub struct TwitterSearchTool;

impl Default for TwitterSearchTool {
    fn default() -> Self {
        Self::new()
    }
}

impl TwitterSearchTool {
    pub fn new() -> Self {
        Self
    }
}

impl Tool for TwitterSearchTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "twitter_search".into(),
            description: "Search recent X (Twitter) posts. Returns up to `max_results` tweets matching the query. Use `since_id` to paginate forward (newer than the given tweet id).".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "X search query (supports operators like from:user, lang:en, hashtags, etc.). Max 512 chars.",
                        "maxLength": 512
                    },
                    "max_results": {
                        "type": "integer",
                        "minimum": 10,
                        "maximum": 100,
                        "default": 10,
                        "description": "Number of tweets to return (10-100)."
                    },
                    "since_id": {
                        "type": "string",
                        "description": "Optional. Return only tweets newer than this id."
                    }
                },
                "required": ["query"]
            }),
        }
    }

    fn execute(
        &self,
        ctx: &ExecutionContext,
        input: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, heartbit_core::Error>> + Send + '_>> {
        let ctx = ctx.clone();
        Box::pin(async move {
            let parsed: SearchInput = match serde_json::from_value(input) {
                Ok(v) => v,
                Err(e) => return Ok(ToolOutput::error(format!("invalid input: {e}"))),
            };
            let client = match XClient::from_context(&ctx).await {
                Ok(c) => c,
                Err(e) => return Ok(ToolOutput::error(format_error(&e))),
            };
            match call_x(&client, &parsed).await {
                Ok(out) => {
                    let json = serde_json::to_string(&out)
                        .unwrap_or_else(|_| "<serialize error>".to_string());
                    Ok(ToolOutput::success(json))
                }
                Err(e) => Ok(ToolOutput::error(format_error(&e))),
            }
        })
    }
}

async fn call_x(client: &XClient, input: &SearchInput) -> Result<SearchOutput, XApiError> {
    let max_str = input.max_results.to_string();
    let mut query: Vec<(&str, &str)> = vec![
        ("query", &input.query),
        ("max_results", &max_str),
        ("tweet.fields", "author_id,created_at"),
    ];
    if let Some(since) = input.since_id.as_deref() {
        query.push(("since_id", since));
    }
    let response: SearchApiResponse = client.get_json("/2/tweets/search/recent", &query).await?;
    Ok(SearchOutput {
        tweets: response.data,
        next_token: response.meta.next_token,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use heartbit_core::{CredentialResolver, Secret};
    use std::sync::Arc;
    use wiremock::matchers::{method, path as wm_path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    struct CannedResolver;
    impl CredentialResolver for CannedResolver {
        fn resolve(
            &self,
            name: &str,
        ) -> Pin<Box<dyn Future<Output = Result<Secret, heartbit_core::Error>> + Send + '_>> {
            let n = name.to_string();
            Box::pin(async move { Ok(Secret::new(format!("test-{n}"))) })
        }
    }

    fn test_client(server_uri: &str) -> XClient {
        XClient::new(
            server_uri,
            Secret::new("ck"),
            Secret::new("cs"),
            Secret::new("at"),
            Secret::new("ats"),
        )
        .expect("test client builds")
    }

    #[tokio::test]
    async fn search_happy_path() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/2/tweets/search/recent"))
            .and(query_param("query", "rust"))
            .and(query_param("max_results", "10"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [
                    {"id": "1", "text": "rust is great", "author_id": "100", "created_at": "2026-01-01T00:00:00.000Z"}
                ],
                "meta": {"next_token": "next-page-1"}
            })))
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let input = SearchInput { query: "rust".into(), max_results: 10, since_id: None };
        let result = call_x(&client, &input).await.expect("happy path");
        assert_eq!(result.tweets.len(), 1);
        assert_eq!(result.tweets[0].text, "rust is great");
        assert_eq!(result.next_token.as_deref(), Some("next-page-1"));
    }

    #[tokio::test]
    async fn search_with_since_id_passes_param() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/2/tweets/search/recent"))
            .and(query_param("since_id", "12345"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({"data": [], "meta": {}})),
            )
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let input = SearchInput {
            query: "rust".into(),
            max_results: 10,
            since_id: Some("12345".into()),
        };
        let result = call_x(&client, &input).await.expect("with since_id");
        assert!(result.tweets.is_empty());
        assert!(result.next_token.is_none());
    }

    #[tokio::test]
    async fn search_returns_unauthenticated_on_401() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/2/tweets/search/recent"))
            .respond_with(ResponseTemplate::new(401))
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let input = SearchInput { query: "x".into(), max_results: 10, since_id: None };
        let result = call_x(&client, &input).await;
        assert!(matches!(result, Err(XApiError::Unauthenticated(_))));
    }

    #[tokio::test]
    async fn search_returns_rate_limited_on_429() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/2/tweets/search/recent"))
            .respond_with(ResponseTemplate::new(429).insert_header("Retry-After", "15"))
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let input = SearchInput { query: "x".into(), max_results: 10, since_id: None };
        let result = call_x(&client, &input).await;
        match result {
            Err(XApiError::RateLimited { retry_after_secs }) => assert_eq!(retry_after_secs, 15),
            other => panic!("got {:?}", other),
        }
    }

    #[tokio::test]
    async fn execute_no_credentials_returns_clear_error() {
        let tool = TwitterSearchTool::new();
        let ctx = ExecutionContext::default();
        let result = tool
            .execute(&ctx, serde_json::json!({"query": "anything"}))
            .await
            .expect("Tool::execute returns Ok");
        assert!(result.is_error);
        assert!(result.content.contains("no credential resolver"));
    }

    #[tokio::test]
    async fn definition_has_stable_name() {
        let tool = TwitterSearchTool::new();
        assert_eq!(tool.definition().name, "twitter_search");
    }
}
```

- [ ] **Step 2: Register the module**

In `crates/heartbit-ghost/src/tools/mod.rs`, add:

```rust
pub mod search;
pub use search::TwitterSearchTool;
```

- [ ] **Step 3: Run the tests**

```bash
cargo test -p heartbit-ghost --lib search 2>&1 | tail -10
```

Expected: 6 passed.

- [ ] **Step 4: Workspace quality gate**

```bash
cargo fmt -p heartbit-ghost -- --check
cargo clippy -p heartbit-ghost -- -D warnings
```

- [ ] **Step 5: Commit**

```bash
git add crates/heartbit-ghost/src/tools/search.rs crates/heartbit-ghost/src/tools/mod.rs
git commit -m "$(cat <<'EOF'
feat(ghost): TwitterSearchTool — search recent X tweets (P1.1)

GET /2/tweets/search/recent with query, max_results (10-100), and
optional since_id. Returns {tweets: [...], next_token?}. Each tweet
has id, text, author_id, created_at.

6 tests: happy path, since_id pass-through, 401, 429 with retry-after,
missing credentials, definition().

Refs: docs/superpowers/specs/2026-05-07-heartbit-ghost-p1.1-x-tools-design.md §1
EOF
)"
```

---

## Task 5: `TwitterMentionsTool` (read-only — reuses pagination from search)

**Why:** Same pattern as search but for a specific user's mentions. Different endpoint, slightly different output.

**Files:**
- Create: `crates/heartbit-ghost/src/tools/mentions.rs`
- Modify: `crates/heartbit-ghost/src/tools/mod.rs`

- [ ] **Step 1: Create the tool**

Create `crates/heartbit-ghost/src/tools/mentions.rs`:

```rust
//! `twitter_mentions` — fetch mentions of a specific X user.

use std::future::Future;
use std::pin::Pin;

use heartbit_core::{ExecutionContext, Tool, ToolOutput};
use heartbit_core::llm::types::ToolDefinition;
use serde::{Deserialize, Serialize};

use crate::tools::client::{XApiError, XClient, format_error};

#[derive(Debug, Deserialize)]
struct MentionsInput {
    user_id: String,
    #[serde(default = "default_max_results")]
    max_results: u32,
    #[serde(default)]
    since_id: Option<String>,
}

fn default_max_results() -> u32 {
    10
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct Mention {
    id: String,
    text: String,
    author_id: Option<String>,
    created_at: Option<String>,
    in_reply_to_user_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct MentionsOutput {
    mentions: Vec<Mention>,
    next_token: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MentionsApiResponse {
    #[serde(default)]
    data: Vec<Mention>,
    #[serde(default)]
    meta: MentionsMeta,
}

#[derive(Debug, Default, Deserialize)]
struct MentionsMeta {
    #[serde(default)]
    next_token: Option<String>,
}

pub struct TwitterMentionsTool;

impl Default for TwitterMentionsTool {
    fn default() -> Self {
        Self::new()
    }
}

impl TwitterMentionsTool {
    pub fn new() -> Self {
        Self
    }
}

impl Tool for TwitterMentionsTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "twitter_mentions".into(),
            description: "Fetch recent mentions of a specific X user. Returns up to `max_results` mentions; use `since_id` to paginate forward.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "X user id (numeric string), not handle. Get this from twitter_user."
                    },
                    "max_results": {
                        "type": "integer",
                        "minimum": 5,
                        "maximum": 100,
                        "default": 10
                    },
                    "since_id": {
                        "type": "string",
                        "description": "Optional. Return only mentions newer than this id."
                    }
                },
                "required": ["user_id"]
            }),
        }
    }

    fn execute(
        &self,
        ctx: &ExecutionContext,
        input: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, heartbit_core::Error>> + Send + '_>> {
        let ctx = ctx.clone();
        Box::pin(async move {
            let parsed: MentionsInput = match serde_json::from_value(input) {
                Ok(v) => v,
                Err(e) => return Ok(ToolOutput::error(format!("invalid input: {e}"))),
            };
            let client = match XClient::from_context(&ctx).await {
                Ok(c) => c,
                Err(e) => return Ok(ToolOutput::error(format_error(&e))),
            };
            match call_x(&client, &parsed).await {
                Ok(out) => {
                    let json = serde_json::to_string(&out)
                        .unwrap_or_else(|_| "<serialize error>".to_string());
                    Ok(ToolOutput::success(json))
                }
                Err(e) => Ok(ToolOutput::error(format_error(&e))),
            }
        })
    }
}

async fn call_x(client: &XClient, input: &MentionsInput) -> Result<MentionsOutput, XApiError> {
    let path = format!("/2/users/{}/mentions", input.user_id);
    let max_str = input.max_results.to_string();
    let mut query: Vec<(&str, &str)> = vec![
        ("max_results", &max_str),
        ("tweet.fields", "author_id,created_at,in_reply_to_user_id"),
    ];
    if let Some(since) = input.since_id.as_deref() {
        query.push(("since_id", since));
    }
    let response: MentionsApiResponse = client.get_json(&path, &query).await?;
    Ok(MentionsOutput {
        mentions: response.data,
        next_token: response.meta.next_token,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use heartbit_core::Secret;
    use wiremock::matchers::{method, path as wm_path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn test_client(server_uri: &str) -> XClient {
        XClient::new(
            server_uri,
            Secret::new("ck"),
            Secret::new("cs"),
            Secret::new("at"),
            Secret::new("ats"),
        )
        .expect("test client builds")
    }

    #[tokio::test]
    async fn mentions_happy_path() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/2/users/100/mentions"))
            .and(query_param("max_results", "10"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [
                    {
                        "id": "9001",
                        "text": "hey @user",
                        "author_id": "200",
                        "created_at": "2026-01-01T00:00:00.000Z",
                        "in_reply_to_user_id": "100"
                    }
                ],
                "meta": {"next_token": "next-mentions-1"}
            })))
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let input = MentionsInput { user_id: "100".into(), max_results: 10, since_id: None };
        let result = call_x(&client, &input).await.expect("happy path");
        assert_eq!(result.mentions.len(), 1);
        assert_eq!(result.mentions[0].id, "9001");
        assert_eq!(result.mentions[0].in_reply_to_user_id.as_deref(), Some("100"));
    }

    #[tokio::test]
    async fn mentions_with_since_id_passes_param() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/2/users/100/mentions"))
            .and(query_param("since_id", "8000"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({"data": [], "meta": {}})),
            )
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let input = MentionsInput {
            user_id: "100".into(),
            max_results: 10,
            since_id: Some("8000".into()),
        };
        let result = call_x(&client, &input).await.expect("with since_id");
        assert!(result.mentions.is_empty());
    }

    #[tokio::test]
    async fn mentions_returns_unauthenticated_on_401() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/2/users/100/mentions"))
            .respond_with(ResponseTemplate::new(401))
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let input = MentionsInput { user_id: "100".into(), max_results: 10, since_id: None };
        let result = call_x(&client, &input).await;
        assert!(matches!(result, Err(XApiError::Unauthenticated(_))));
    }

    #[tokio::test]
    async fn execute_no_credentials_returns_clear_error() {
        let tool = TwitterMentionsTool::new();
        let ctx = ExecutionContext::default();
        let result = tool
            .execute(&ctx, serde_json::json!({"user_id": "100"}))
            .await
            .expect("Tool::execute returns Ok");
        assert!(result.is_error);
        assert!(result.content.contains("no credential resolver"));
    }

    #[tokio::test]
    async fn definition_has_stable_name() {
        let tool = TwitterMentionsTool::new();
        assert_eq!(tool.definition().name, "twitter_mentions");
    }
}
```

- [ ] **Step 2: Register the module**

In `tools/mod.rs`, add:

```rust
pub mod mentions;
pub use mentions::TwitterMentionsTool;
```

- [ ] **Step 3: Run tests + quality gate + commit**

```bash
cargo test -p heartbit-ghost --lib mentions 2>&1 | tail -5
cargo fmt -p heartbit-ghost -- --check
cargo clippy -p heartbit-ghost -- -D warnings
git add crates/heartbit-ghost/src/tools/mentions.rs crates/heartbit-ghost/src/tools/mod.rs
git commit -m "$(cat <<'EOF'
feat(ghost): TwitterMentionsTool — fetch mentions of a user (P1.1)

GET /2/users/:id/mentions with max_results and optional since_id.
Returns {mentions: [...], next_token?}. Each mention has id, text,
author_id, created_at, in_reply_to_user_id.

5 tests: happy path, since_id pass-through, 401, missing credentials,
definition() stability.

Refs: docs/superpowers/specs/2026-05-07-heartbit-ghost-p1.1-x-tools-design.md §1
EOF
)"
```

Expected: 5 passed.

---

## Task 6: `TwitterReplyTool` (write — single tweet with `in_reply_to`)

**Files:**
- Create: `crates/heartbit-ghost/src/tools/reply.rs`
- Modify: `crates/heartbit-ghost/src/tools/mod.rs`

- [ ] **Step 1: Create the tool**

Create `crates/heartbit-ghost/src/tools/reply.rs`:

```rust
//! `twitter_reply` — reply to an existing tweet.

use std::future::Future;
use std::pin::Pin;

use heartbit_core::{ExecutionContext, Tool, ToolOutput};
use heartbit_core::llm::types::ToolDefinition;
use serde::{Deserialize, Serialize};

use crate::tools::client::{XApiError, XClient, format_error};

const MAX_TWEET_LENGTH: usize = 280;

#[derive(Debug, Deserialize)]
struct ReplyInput {
    text: String,
    in_reply_to: String,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct ReplyOutput {
    tweet_id: String,
    url: String,
}

/// X v2 POST /2/tweets request body for a reply.
#[derive(Debug, Serialize)]
struct ReplyRequest<'a> {
    text: &'a str,
    reply: ReplyTo<'a>,
}

#[derive(Debug, Serialize)]
struct ReplyTo<'a> {
    in_reply_to_tweet_id: &'a str,
}

#[derive(Debug, Deserialize)]
struct ReplyApiResponse {
    data: ReplyApiData,
}

#[derive(Debug, Deserialize)]
struct ReplyApiData {
    id: String,
}

pub struct TwitterReplyTool;

impl Default for TwitterReplyTool {
    fn default() -> Self {
        Self::new()
    }
}

impl TwitterReplyTool {
    pub fn new() -> Self {
        Self
    }
}

impl Tool for TwitterReplyTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "twitter_reply".into(),
            description: "Reply to an X (Twitter) tweet. Posts a new tweet linked to in_reply_to. Max 280 chars.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Reply text. Max 280 characters.",
                        "maxLength": 280
                    },
                    "in_reply_to": {
                        "type": "string",
                        "description": "Tweet id (numeric string) being replied to."
                    }
                },
                "required": ["text", "in_reply_to"]
            }),
        }
    }

    fn execute(
        &self,
        ctx: &ExecutionContext,
        input: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, heartbit_core::Error>> + Send + '_>> {
        let ctx = ctx.clone();
        Box::pin(async move {
            let parsed: ReplyInput = match serde_json::from_value(input) {
                Ok(v) => v,
                Err(e) => return Ok(ToolOutput::error(format!("invalid input: {e}"))),
            };
            let char_count = parsed.text.chars().count();
            if parsed.text.is_empty() {
                return Ok(ToolOutput::error("text must not be empty"));
            }
            if char_count > MAX_TWEET_LENGTH {
                return Ok(ToolOutput::error(format!(
                    "Reply exceeds {MAX_TWEET_LENGTH} characters (got {char_count})."
                )));
            }
            let client = match XClient::from_context(&ctx).await {
                Ok(c) => c,
                Err(e) => return Ok(ToolOutput::error(format_error(&e))),
            };
            match call_x(&client, &parsed).await {
                Ok(out) => {
                    let json = serde_json::to_string(&out)
                        .unwrap_or_else(|_| "<serialize error>".to_string());
                    Ok(ToolOutput::success(json))
                }
                Err(e) => Ok(ToolOutput::error(format_error(&e))),
            }
        })
    }
}

async fn call_x(client: &XClient, input: &ReplyInput) -> Result<ReplyOutput, XApiError> {
    let body = ReplyRequest {
        text: &input.text,
        reply: ReplyTo {
            in_reply_to_tweet_id: &input.in_reply_to,
        },
    };
    let response: ReplyApiResponse = client.post_json("/2/tweets", &body).await?;
    let tweet_id = response.data.id;
    let url = format!("https://twitter.com/i/web/status/{}", tweet_id);
    Ok(ReplyOutput { tweet_id, url })
}

#[cfg(test)]
mod tests {
    use super::*;
    use heartbit_core::Secret;
    use wiremock::matchers::{body_json, method, path as wm_path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn test_client(server_uri: &str) -> XClient {
        XClient::new(
            server_uri,
            Secret::new("ck"),
            Secret::new("cs"),
            Secret::new("at"),
            Secret::new("ats"),
        )
        .expect("test client builds")
    }

    #[tokio::test]
    async fn reply_happy_path() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(wm_path("/2/tweets"))
            .and(body_json(serde_json::json!({
                "text": "thanks!",
                "reply": {"in_reply_to_tweet_id": "9001"}
            })))
            .respond_with(
                ResponseTemplate::new(201).set_body_json(serde_json::json!({"data": {"id": "9999"}})),
            )
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let input = ReplyInput {
            text: "thanks!".into(),
            in_reply_to: "9001".into(),
        };
        let result = call_x(&client, &input).await.expect("happy path");
        assert_eq!(result.tweet_id, "9999");
        assert!(result.url.contains("9999"));
    }

    #[tokio::test]
    async fn reply_returns_unauthenticated_on_401() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(wm_path("/2/tweets"))
            .respond_with(ResponseTemplate::new(401))
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let input = ReplyInput { text: "x".into(), in_reply_to: "1".into() };
        let result = call_x(&client, &input).await;
        assert!(matches!(result, Err(XApiError::Unauthenticated(_))));
    }

    #[tokio::test]
    async fn execute_rejects_text_over_280_chars() {
        let tool = TwitterReplyTool::new();
        let ctx = ExecutionContext::default();
        let too_long = "a".repeat(281);
        let result = tool
            .execute(&ctx, serde_json::json!({"text": too_long, "in_reply_to": "1"}))
            .await
            .expect("Tool::execute returns Ok");
        assert!(result.is_error);
        assert!(result.content.contains("exceeds 280"));
    }

    #[tokio::test]
    async fn execute_rejects_empty_text() {
        let tool = TwitterReplyTool::new();
        let ctx = ExecutionContext::default();
        let result = tool
            .execute(&ctx, serde_json::json!({"text": "", "in_reply_to": "1"}))
            .await
            .expect("Tool::execute returns Ok");
        assert!(result.is_error);
        assert!(result.content.contains("must not be empty"));
    }

    #[tokio::test]
    async fn execute_no_credentials_returns_clear_error() {
        let tool = TwitterReplyTool::new();
        let ctx = ExecutionContext::default();
        let result = tool
            .execute(&ctx, serde_json::json!({"text": "hi", "in_reply_to": "1"}))
            .await
            .expect("Tool::execute returns Ok");
        assert!(result.is_error);
        assert!(result.content.contains("no credential resolver"));
    }

    #[tokio::test]
    async fn definition_has_stable_name() {
        let tool = TwitterReplyTool::new();
        assert_eq!(tool.definition().name, "twitter_reply");
    }
}
```

- [ ] **Step 2: Register + run tests + quality gate + commit**

In `tools/mod.rs`, add:

```rust
pub mod reply;
pub use reply::TwitterReplyTool;
```

```bash
cargo test -p heartbit-ghost --lib reply 2>&1 | tail -5
cargo fmt -p heartbit-ghost -- --check
cargo clippy -p heartbit-ghost -- -D warnings
git add crates/heartbit-ghost/src/tools/reply.rs crates/heartbit-ghost/src/tools/mod.rs
git commit -m "$(cat <<'EOF'
feat(ghost): TwitterReplyTool — reply to an X tweet (P1.1)

POST /2/tweets with reply.in_reply_to_tweet_id. Validates text is
non-empty and <= 280 chars before calling X. Returns {tweet_id, url}.

6 tests: happy path, 401, text >280 rejected, empty text rejected,
missing credentials, definition() stability.

Refs: docs/superpowers/specs/2026-05-07-heartbit-ghost-p1.1-x-tools-design.md §1
EOF
)"
```

Expected: 6 passed.

---

## Task 7: `TwitterThreadTool` (write — chained tweets via `in_reply_to`)

**Files:**
- Create: `crates/heartbit-ghost/src/tools/thread.rs`
- Modify: `crates/heartbit-ghost/src/tools/mod.rs`

- [ ] **Step 1: Create the tool**

Create `crates/heartbit-ghost/src/tools/thread.rs`:

```rust
//! `twitter_thread` — post a thread (sequence of linked tweets).

use std::future::Future;
use std::pin::Pin;

use heartbit_core::{ExecutionContext, Tool, ToolOutput};
use heartbit_core::llm::types::ToolDefinition;
use serde::{Deserialize, Serialize};

use crate::tools::client::{XApiError, XClient, format_error};

const MAX_TWEET_LENGTH: usize = 280;
const MAX_THREAD_LENGTH: usize = 25;

#[derive(Debug, Deserialize)]
struct ThreadInput {
    tweets: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct ThreadOutput {
    thread_root_id: String,
    tweet_ids: Vec<String>,
    urls: Vec<String>,
}

#[derive(Debug, Serialize)]
struct PostRequest<'a> {
    text: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    reply: Option<ReplyTo<'a>>,
}

#[derive(Debug, Serialize)]
struct ReplyTo<'a> {
    in_reply_to_tweet_id: &'a str,
}

#[derive(Debug, Deserialize)]
struct PostApiResponse {
    data: PostApiData,
}

#[derive(Debug, Deserialize)]
struct PostApiData {
    id: String,
}

pub struct TwitterThreadTool;

impl Default for TwitterThreadTool {
    fn default() -> Self {
        Self::new()
    }
}

impl TwitterThreadTool {
    pub fn new() -> Self {
        Self
    }
}

impl Tool for TwitterThreadTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "twitter_thread".into(),
            description: "Post a thread of tweets (1..=25 entries, each <=280 chars). Each tweet is posted in sequence and linked via reply.in_reply_to_tweet_id to the previous one. Fails fast on the first error and returns the tweets posted so far in the error message.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "tweets": {
                        "type": "array",
                        "items": {"type": "string", "maxLength": 280},
                        "minItems": 1,
                        "maxItems": 25,
                        "description": "Ordered list of tweet texts. The first becomes the thread root; each subsequent tweet replies to the previous."
                    }
                },
                "required": ["tweets"]
            }),
        }
    }

    fn execute(
        &self,
        ctx: &ExecutionContext,
        input: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, heartbit_core::Error>> + Send + '_>> {
        let ctx = ctx.clone();
        Box::pin(async move {
            let parsed: ThreadInput = match serde_json::from_value(input) {
                Ok(v) => v,
                Err(e) => return Ok(ToolOutput::error(format!("invalid input: {e}"))),
            };
            if parsed.tweets.is_empty() {
                return Ok(ToolOutput::error("tweets array must contain at least 1 entry"));
            }
            if parsed.tweets.len() > MAX_THREAD_LENGTH {
                return Ok(ToolOutput::error(format!(
                    "thread length {} exceeds maximum {}",
                    parsed.tweets.len(),
                    MAX_THREAD_LENGTH
                )));
            }
            for (i, t) in parsed.tweets.iter().enumerate() {
                if t.is_empty() {
                    return Ok(ToolOutput::error(format!("tweet[{i}] must not be empty")));
                }
                let n = t.chars().count();
                if n > MAX_TWEET_LENGTH {
                    return Ok(ToolOutput::error(format!(
                        "tweet[{i}] exceeds {MAX_TWEET_LENGTH} chars (got {n})"
                    )));
                }
            }
            let client = match XClient::from_context(&ctx).await {
                Ok(c) => c,
                Err(e) => return Ok(ToolOutput::error(format_error(&e))),
            };
            match call_x(&client, &parsed).await {
                Ok(out) => {
                    let json = serde_json::to_string(&out)
                        .unwrap_or_else(|_| "<serialize error>".to_string());
                    Ok(ToolOutput::success(json))
                }
                Err(e) => Ok(ToolOutput::error(format_error(&e))),
            }
        })
    }
}

async fn call_x(client: &XClient, input: &ThreadInput) -> Result<ThreadOutput, XApiError> {
    let mut tweet_ids: Vec<String> = Vec::with_capacity(input.tweets.len());
    let mut prev_id: Option<String> = None;
    for text in &input.tweets {
        let body = PostRequest {
            text,
            reply: prev_id.as_ref().map(|id| ReplyTo {
                in_reply_to_tweet_id: id,
            }),
        };
        let resp: PostApiResponse = client.post_json("/2/tweets", &body).await?;
        prev_id = Some(resp.data.id.clone());
        tweet_ids.push(resp.data.id);
    }
    let urls: Vec<String> = tweet_ids
        .iter()
        .map(|id| format!("https://twitter.com/i/web/status/{id}"))
        .collect();
    let thread_root_id = tweet_ids[0].clone();
    Ok(ThreadOutput {
        thread_root_id,
        tweet_ids,
        urls,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use heartbit_core::Secret;
    use wiremock::matchers::{method, path as wm_path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn test_client(server_uri: &str) -> XClient {
        XClient::new(
            server_uri,
            Secret::new("ck"),
            Secret::new("cs"),
            Secret::new("at"),
            Secret::new("ats"),
        )
        .expect("test client builds")
    }

    #[tokio::test]
    async fn thread_three_tweets_chain_correctly() {
        let server = MockServer::start().await;
        // First post: no reply field
        Mock::given(method("POST"))
            .and(wm_path("/2/tweets"))
            .respond_with(ResponseTemplate::new(201).set_body_json(serde_json::json!({"data": {"id": "1001"}})))
            .up_to_n_times(1)
            .mount(&server)
            .await;
        // Second post: reply to 1001 → returns 1002
        Mock::given(method("POST"))
            .and(wm_path("/2/tweets"))
            .respond_with(ResponseTemplate::new(201).set_body_json(serde_json::json!({"data": {"id": "1002"}})))
            .up_to_n_times(1)
            .mount(&server)
            .await;
        // Third post: reply to 1002 → returns 1003
        Mock::given(method("POST"))
            .and(wm_path("/2/tweets"))
            .respond_with(ResponseTemplate::new(201).set_body_json(serde_json::json!({"data": {"id": "1003"}})))
            .up_to_n_times(1)
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let input = ThreadInput {
            tweets: vec!["one".into(), "two".into(), "three".into()],
        };
        let result = call_x(&client, &input).await.expect("happy path");
        assert_eq!(result.thread_root_id, "1001");
        assert_eq!(result.tweet_ids, vec!["1001", "1002", "1003"]);
        assert_eq!(result.urls.len(), 3);
    }

    #[tokio::test]
    async fn thread_fails_fast_on_mid_thread_error() {
        let server = MockServer::start().await;
        // First post succeeds
        Mock::given(method("POST"))
            .and(wm_path("/2/tweets"))
            .respond_with(ResponseTemplate::new(201).set_body_json(serde_json::json!({"data": {"id": "1001"}})))
            .up_to_n_times(1)
            .mount(&server)
            .await;
        // Second post fails 401
        Mock::given(method("POST"))
            .and(wm_path("/2/tweets"))
            .respond_with(ResponseTemplate::new(401).set_body_string("token expired"))
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let input = ThreadInput { tweets: vec!["one".into(), "two".into()] };
        let result = call_x(&client, &input).await;
        assert!(matches!(result, Err(XApiError::Unauthenticated(_))));
    }

    #[tokio::test]
    async fn execute_rejects_empty_thread() {
        let tool = TwitterThreadTool::new();
        let ctx = ExecutionContext::default();
        let result = tool
            .execute(&ctx, serde_json::json!({"tweets": []}))
            .await
            .expect("Tool::execute returns Ok");
        assert!(result.is_error);
        assert!(result.content.contains("at least 1 entry"));
    }

    #[tokio::test]
    async fn execute_rejects_thread_over_25() {
        let tool = TwitterThreadTool::new();
        let ctx = ExecutionContext::default();
        let many: Vec<&str> = (0..26).map(|_| "x").collect();
        let result = tool
            .execute(&ctx, serde_json::json!({"tweets": many}))
            .await
            .expect("Tool::execute returns Ok");
        assert!(result.is_error);
        assert!(result.content.contains("exceeds maximum 25"));
    }

    #[tokio::test]
    async fn execute_rejects_empty_individual_tweet() {
        let tool = TwitterThreadTool::new();
        let ctx = ExecutionContext::default();
        let result = tool
            .execute(&ctx, serde_json::json!({"tweets": ["ok", ""]}))
            .await
            .expect("Tool::execute returns Ok");
        assert!(result.is_error);
        assert!(result.content.contains("tweet[1]"));
    }

    #[tokio::test]
    async fn execute_rejects_individual_tweet_over_280() {
        let tool = TwitterThreadTool::new();
        let ctx = ExecutionContext::default();
        let too_long = "a".repeat(281);
        let result = tool
            .execute(&ctx, serde_json::json!({"tweets": ["ok", too_long]}))
            .await
            .expect("Tool::execute returns Ok");
        assert!(result.is_error);
        assert!(result.content.contains("tweet[1]"));
        assert!(result.content.contains("280"));
    }

    #[tokio::test]
    async fn definition_has_stable_name() {
        let tool = TwitterThreadTool::new();
        assert_eq!(tool.definition().name, "twitter_thread");
    }
}
```

- [ ] **Step 2: Register + run tests + quality gate + commit**

```rust
// in tools/mod.rs:
pub mod thread;
pub use thread::TwitterThreadTool;
```

```bash
cargo test -p heartbit-ghost --lib thread 2>&1 | tail -5
cargo fmt -p heartbit-ghost -- --check
cargo clippy -p heartbit-ghost -- -D warnings
git add crates/heartbit-ghost/src/tools/thread.rs crates/heartbit-ghost/src/tools/mod.rs
git commit -m "$(cat <<'EOF'
feat(ghost): TwitterThreadTool — post a chained thread (P1.1)

POST /2/tweets ×N. Each tweet (after the first) sets
reply.in_reply_to_tweet_id to the previous tweet's id. Validates
1..=25 entries, each non-empty, each <=280 chars. Fails fast on the
first X error; tweets posted before the failure stay live (X has no
rollback API).

7 tests: 3-tweet happy path with chaining verified, fail-fast on
mid-thread 401, empty thread rejected, >25 entries rejected, empty
individual tweet rejected, individual >280 chars rejected, definition().

Refs: docs/superpowers/specs/2026-05-07-heartbit-ghost-p1.1-x-tools-design.md §1
EOF
)"
```

Expected: 7 passed.

---

## Task 8: `twitter_post` extension — media + alt text (heartbit-core in-place edit)

**Why:** The existing `TwitterPostTool` gains optional `media_url` + `media_alt_text` support. Backward-compatible: text-only callers see no change.

**Files:**
- Modify: `crates/heartbit-core/src/tool/builtins/twitter_post.rs`

- [ ] **Step 1: Read the existing file thoroughly**

```bash
cat crates/heartbit-core/src/tool/builtins/twitter_post.rs
```

You'll see (~422 lines):
- Existing inline OAuth1 implementation (`percent_encode`, `build_oauth_header`)
- `TwitterCredentials` struct with 4 fields
- `TwitterPostTool` with `new()` and `try_new()`
- Existing `Tool::execute` that posts text-only via `POST /2/tweets`
- Existing tests covering 280-char limit, signing, etc.

**Do not touch the existing OAuth1 logic.** The extension reuses it.

- [ ] **Step 2: Extend the input schema**

Find `Tool::definition()` (around line 157). Replace the `input_schema` JSON to add the two new optional fields:

```rust
input_schema: json!({
    "type": "object",
    "properties": {
        "text": {
            "type": "string",
            "description": "The tweet text to post (max 280 characters)"
        },
        "media_url": {
            "type": "string",
            "description": "Optional. Public HTTPS URL of one image to attach (≤5 MB, JPEG/PNG/WebP/GIF)."
        },
        "media_alt_text": {
            "type": "string",
            "description": "Optional. Accessibility description for the image (≤1000 chars). Ignored if media_url is absent.",
            "maxLength": 1000
        }
    },
    "required": ["text"]
}),
```

Also update the `description` field to reflect the new capability:

```rust
description: "Post a tweet to X/Twitter. Maximum 280 characters. Optionally attaches one image via media_url with an accessibility description via media_alt_text.".into(),
```

- [ ] **Step 3: Update `Tool::execute` to handle the optional media path**

In the existing `execute` body (around line 174), after the existing text validation, parse the optional fields:

```rust
let media_url = input.get("media_url").and_then(|v| v.as_str());
let media_alt_text = input.get("media_alt_text").and_then(|v| v.as_str());
```

Then, when `media_url` is present, do the upload BEFORE the tweet POST:

```rust
let media_id_string: Option<String> = if let Some(url) = media_url {
    match self.upload_media(url, media_alt_text).await {
        Ok(id) => Some(id),
        Err(e) => return Ok(ToolOutput::error(format!("media upload failed: {e}"))),
    }
} else {
    None
};
```

When constructing the tweet body that POSTs to `/2/tweets`, attach the media id if present:

```rust
let body = if let Some(ref id) = media_id_string {
    json!({
        "text": text,
        "media": {"media_ids": [id]}
    })
} else {
    json!({"text": text})
};
```

(The existing code constructs `json!({"text": text})`. Replace that with the conditional.)

- [ ] **Step 4: Implement `upload_media` on `TwitterPostTool`**

Add this method to `impl TwitterPostTool`:

```rust
/// Fetch image bytes from `media_url`, upload to X v1.1 media endpoint, and
/// (optionally) attach alt text. Returns the `media_id_string` to be referenced
/// in the tweet POST.
async fn upload_media(
    &self,
    media_url: &str,
    media_alt_text: Option<&str>,
) -> Result<String, Error> {
    // Step A: Fetch the image bytes (HTTP GET, ≤5 MB)
    let bytes = self
        .client
        .get(media_url)
        .send()
        .await
        .map_err(|e| Error::Agent(format!("media fetch failed: {e}")))?;
    let status = bytes.status();
    if !status.is_success() {
        return Err(Error::Agent(format!(
            "media fetch returned status {}",
            status.as_u16()
        )));
    }
    let body = bytes
        .bytes()
        .await
        .map_err(|e| Error::Agent(format!("media body read failed: {e}")))?;
    if body.len() > 5 * 1024 * 1024 {
        return Err(Error::Agent(format!(
            "media exceeds 5 MB limit (got {} bytes)",
            body.len()
        )));
    }

    // Step B: Upload to X v1.1 media endpoint via multipart/form-data
    const MEDIA_UPLOAD_URL: &str = "https://upload.twitter.com/1.1/media/upload.json";
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .map_err(|e| Error::Agent(format!("system time error: {e}")))?
        .as_secs();
    let nonce = format!("{}{}", timestamp, &timestamp.to_string()[..6]); // simple unique nonce

    let auth_header = build_oauth_header(
        MEDIA_UPLOAD_URL,
        &self.credentials.consumer_key,
        &self.credentials.consumer_secret,
        &self.credentials.access_token,
        &self.credentials.access_token_secret,
        &nonce,
        timestamp,
    )?;
    // NOTE: build_oauth_header currently hardcodes "POST" — that's correct for
    // the media upload endpoint. If this bothers a future refactor (e.g., GET
    // signing), see Task 2's notes about parameterising method.

    let form = reqwest::multipart::Form::new()
        .part("media", reqwest::multipart::Part::bytes(body.to_vec()).file_name("media"));
    let response = self
        .client
        .post(MEDIA_UPLOAD_URL)
        .header("Authorization", auth_header)
        .multipart(form)
        .send()
        .await
        .map_err(|e| Error::Agent(format!("media upload failed: {e}")))?;
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        return Err(Error::Agent(format!(
            "media upload returned status {}: {body}",
            status.as_u16()
        )));
    }
    let parsed: serde_json::Value = response
        .json()
        .await
        .map_err(|e| Error::Agent(format!("media upload parse failed: {e}")))?;
    let media_id_string = parsed
        .get("media_id_string")
        .and_then(|v| v.as_str())
        .ok_or_else(|| Error::Agent("media upload returned no media_id_string".into()))?
        .to_string();

    // Step C (optional): attach alt text via metadata/create
    if let Some(alt) = media_alt_text {
        const META_URL: &str = "https://upload.twitter.com/1.1/media/metadata/create.json";
        let meta_timestamp = std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .map_err(|e| Error::Agent(format!("system time error: {e}")))?
            .as_secs();
        let meta_nonce = format!("{}{}", meta_timestamp, &meta_timestamp.to_string()[..6]);
        let meta_auth = build_oauth_header(
            META_URL,
            &self.credentials.consumer_key,
            &self.credentials.consumer_secret,
            &self.credentials.access_token,
            &self.credentials.access_token_secret,
            &meta_nonce,
            meta_timestamp,
        )?;
        let body = json!({
            "media_id": media_id_string,
            "alt_text": {"text": alt}
        });
        let _ = self
            .client
            .post(META_URL)
            .header("Authorization", meta_auth)
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::Agent(format!("alt-text attach failed: {e}")))?;
        // Don't fail if alt-text attach fails — the media itself is already up
        // and the tweet can still post. We log but don't propagate.
    }

    Ok(media_id_string)
}
```

(This method is added to `impl TwitterPostTool`, alongside the existing `new` and `try_new`.)

- [ ] **Step 5: Add tests for the new code paths**

In the existing `#[cfg(test)] mod tests`, add new tests. The existing tests don't go through wiremock; they use the existing `TwitterPostTool::new(test_credentials())` pattern. The media-upload tests need wiremock to stub both the `/1.1/media/upload.json` endpoint and the URL fetch.

To avoid restructuring twitter_post.rs significantly, the new media tests use a custom `TwitterPostTool::new_with_base_urls()` method (test-only) that lets them point at wiremock. Add this method:

```rust
#[cfg(test)]
impl TwitterPostTool {
    /// Test-only constructor that lets tests redirect requests to a wiremock server.
    /// Both `tweet_url` and `media_upload_url` should be the wiremock server URI + path.
    pub(crate) fn new_with_base_urls(
        credentials: TwitterCredentials,
        tweet_url: String,
        media_upload_url: String,
    ) -> Self {
        let client = crate::http::vendor_client_builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .expect("test client builds");
        // For test-only injection, set fields via cfg-only struct fields.
        // The implementer adds (cfg(test)) overrides on the URLs.
        // ...
        let _ = (tweet_url, media_upload_url);
        Self { credentials, client }
    }
}
```

**Implementer note**: cleanest path is to make `X_API_URL` and `MEDIA_UPLOAD_URL`/`META_URL` configurable on the struct (e.g., `tweet_url: String`, `media_upload_url: String`, `media_meta_url: String` fields with default values from constants). This is a small refactor — ~10 lines of struct change + thread the field through `execute` and `upload_media`. Tests pass `server.uri()` paths.

If this refactor feels intrusive, an alternative is to add `#[cfg(test)] static OVERRIDE_MEDIA_URL: OnceLock<String> = OnceLock::new();` and have `upload_media` check it. Less clean but more localised. Either approach is acceptable.

For the new tests, assume URL injection works:

```rust
#[tokio::test]
async fn post_with_media_url_and_alt_text() {
    use wiremock::matchers::{method, path as wm_path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;

    // Stub the media bytes URL (the image)
    Mock::given(method("GET"))
        .and(wm_path("/test-image.png"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_bytes(vec![0u8; 1024])  // 1 KB of zeros
                .insert_header("Content-Type", "image/png"),
        )
        .mount(&server)
        .await;

    // Stub the media upload endpoint (called with auth + multipart)
    Mock::given(method("POST"))
        .and(wm_path("/1.1/media/upload.json"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "media_id_string": "999888777",
                "media_id": 999888777u64
            })),
        )
        .mount(&server)
        .await;

    // Stub the metadata/create endpoint (alt text)
    Mock::given(method("POST"))
        .and(wm_path("/1.1/media/metadata/create.json"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;

    // Stub the tweet POST endpoint
    Mock::given(method("POST"))
        .and(wm_path("/2/tweets"))
        .respond_with(
            ResponseTemplate::new(201).set_body_json(serde_json::json!({
                "data": {"id": "5555"}
            })),
        )
        .mount(&server)
        .await;

    let media_url = format!("{}/test-image.png", server.uri());
    let tool = TwitterPostTool::new_with_base_urls(
        test_credentials(),
        format!("{}/2/tweets", server.uri()),
        format!("{}/1.1/media/upload.json", server.uri()),
    );
    let ctx = crate::ExecutionContext::default();
    let input = json!({
        "text": "look at this",
        "media_url": media_url,
        "media_alt_text": "a square of zeros"
    });
    let result = tool.execute(&ctx, input).await.expect("ok");
    assert!(!result.is_error);
    assert!(result.content.contains("5555"));
}

#[tokio::test]
async fn post_text_only_still_works_without_media_fields() {
    // Existing tests in this file already cover text-only — verify the
    // backward-compat path explicitly with the new schema.
    use wiremock::matchers::{method, path as wm_path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(wm_path("/2/tweets"))
        .respond_with(ResponseTemplate::new(201).set_body_json(serde_json::json!({"data": {"id": "111"}})))
        .mount(&server)
        .await;

    let tool = TwitterPostTool::new_with_base_urls(
        test_credentials(),
        format!("{}/2/tweets", server.uri()),
        format!("{}/1.1/media/upload.json", server.uri()),  // unused
    );
    let ctx = crate::ExecutionContext::default();
    let input = json!({"text": "no media"});
    let result = tool.execute(&ctx, input).await.expect("ok");
    assert!(!result.is_error);
}

#[tokio::test]
async fn post_rejects_oversized_media() {
    use wiremock::matchers::{method, path as wm_path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(wm_path("/big.png"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_bytes(vec![0u8; 6 * 1024 * 1024])  // 6 MB
                .insert_header("Content-Type", "image/png"),
        )
        .mount(&server)
        .await;

    let tool = TwitterPostTool::new_with_base_urls(
        test_credentials(),
        format!("{}/2/tweets", server.uri()),
        format!("{}/1.1/media/upload.json", server.uri()),
    );
    let ctx = crate::ExecutionContext::default();
    let media_url = format!("{}/big.png", server.uri());
    let input = json!({
        "text": "won't fit",
        "media_url": media_url
    });
    let result = tool.execute(&ctx, input).await.expect("Tool::execute returns Ok");
    assert!(result.is_error);
    assert!(result.content.contains("5 MB") || result.content.contains("exceeds"));
}

#[tokio::test]
async fn post_handles_404_on_media_url() {
    use wiremock::matchers::{method, path as wm_path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(wm_path("/missing.png"))
        .respond_with(ResponseTemplate::new(404))
        .mount(&server)
        .await;

    let tool = TwitterPostTool::new_with_base_urls(
        test_credentials(),
        format!("{}/2/tweets", server.uri()),
        format!("{}/1.1/media/upload.json", server.uri()),
    );
    let ctx = crate::ExecutionContext::default();
    let media_url = format!("{}/missing.png", server.uri());
    let input = json!({
        "text": "broken link",
        "media_url": media_url
    });
    let result = tool.execute(&ctx, input).await.expect("Tool::execute returns Ok");
    assert!(result.is_error);
    assert!(result.content.contains("404") || result.content.contains("media fetch"));
}
```

Add `wiremock = "0.6"` to `crates/heartbit-core/Cargo.toml`'s `[dev-dependencies]` if not already present:

```bash
grep -A 5 "^\[dev-dependencies\]" crates/heartbit-core/Cargo.toml
```

If `wiremock` is missing, add it.

- [ ] **Step 6: Run all heartbit-core tests**

```bash
cargo test -p heartbit-core --lib twitter_post 2>&1 | tail -10
```

Expected: existing twitter_post tests still pass + 4 new tests pass. The total goes up by 4.

If existing tests fail because of the schema/struct changes, fix them. The most likely failure: an existing test asserts the schema doesn't have `media_url` field. Update those assertions to reflect the new schema.

- [ ] **Step 7: Workspace quality gate**

```bash
cargo fmt -- --check
cargo clippy --workspace --all-features -- -D warnings
cargo test --workspace --all-features 2>&1 | tail -3
```

All clean. Workspace test count up by ~4 (from heartbit-core) + ~30 (from heartbit-ghost across Tasks 2-7).

- [ ] **Step 8: Commit**

```bash
git add crates/heartbit-core/src/tool/builtins/twitter_post.rs crates/heartbit-core/Cargo.toml
git commit -m "$(cat <<'EOF'
feat(twitter_post): optional media + alt text support (heartbit-ghost P1.1 §1.1)

Extends TwitterPostTool's input schema with optional `media_url` and
`media_alt_text` fields. When media_url is absent, behaviour is
unchanged. When present, the tool:
  1. Fetches the image (HTTPS GET, <=5 MB)
  2. Uploads via POST /1.1/media/upload.json (multipart/form-data)
  3. Optionally attaches alt text via POST /1.1/media/metadata/create.json
  4. Posts the tweet via POST /2/tweets with media.media_ids = [id]

Added test-only constructor `new_with_base_urls` so wiremock tests can
inject endpoint URLs without touching the production constants.

4 new tests via wiremock: text+media+alt happy path; text-only
still works; oversized media (6 MB) rejected; 404 on media URL
surfaces clearly. Existing text-only tests stay green.

Refs: docs/superpowers/specs/2026-05-07-heartbit-ghost-p1.1-x-tools-design.md §1.1
EOF
)"
```

---

## Task 9: Final acceptance + workspace quality gate + CHANGELOG

**Why:** Confirm P1.1 meets every acceptance criterion and document the additions.

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Workspace quality gate**

```bash
cd /home/pleclech/projects/heartbit
cargo fmt -- --check
cargo clippy --workspace --all-features -- -D warnings
cargo test --workspace --all-features 2>&1 | tail -10
```

All four signals green. The test count should be approximately 3729 (P1.0 baseline) + 35–40 (P1.1) = ~3765.

If anything fails, classify and fix.

- [ ] **Step 2: Acceptance criteria walkthrough**

For each criterion in the spec §7, verify:

```bash
# 1. All 5 new tools exist as `pub` types
grep -h "pub struct Twitter" crates/heartbit-ghost/src/tools/*.rs
# Expected: TwitterUserTool, TwitterSearchTool, TwitterMentionsTool, TwitterReplyTool, TwitterThreadTool

# 2. twitter_post extension landed
grep -n "media_url\|media_alt_text" crates/heartbit-core/src/tool/builtins/twitter_post.rs | head -5
# Expected: matches in schema and execute body

# 3. Test count
cargo test --workspace --all-features 2>&1 | grep -oE "[0-9]+ passed" | awk '{s+=$1} END {print s}'
# Expected: ~3765

# 4. heartbit-ghost::tools re-exports
grep "pub use" crates/heartbit-ghost/src/tools/mod.rs
# Expected: 5 lines, one per new tool, plus the client re-exports
```

- [ ] **Step 3: Update CHANGELOG.md**

Open `CHANGELOG.md`. Find the `[Unreleased]` section (if it exists) or add one above the most recent versioned entry.

Add an entry:

```markdown
## [Unreleased] — heartbit-ghost P1.1 (X tool family)

### Added (heartbit-ghost)
- `TwitterUserTool` — `GET /2/users/by/username/:handle`. Returns id, name, description, follower/following/tweet counts.
- `TwitterSearchTool` — `GET /2/tweets/search/recent`. Returns matching tweets + `next_token` for pagination.
- `TwitterMentionsTool` — `GET /2/users/:id/mentions`. Returns mentions + pagination.
- `TwitterReplyTool` — `POST /2/tweets` (with `reply.in_reply_to_tweet_id`). Validates ≤280 chars.
- `TwitterThreadTool` — `POST /2/tweets` ×N, chained via `in_reply_to`. 1..=25 entries; fail-fast.
- Shared `XClient` infrastructure: OAuth1 signing, credential resolution from `ExecutionContext::credentials`, typed `XApiError`, response parsing.
- Stable credential resolver names: `X_CONSUMER_KEY`, `X_CONSUMER_SECRET`, `X_ACCESS_TOKEN`, `X_ACCESS_TOKEN_SECRET`.

### Changed (heartbit-core)
- `TwitterPostTool` now accepts optional `media_url` (HTTPS, ≤5 MB image) and `media_alt_text` (≤1000 chars). Backward-compatible: text-only callers see no change.

### Notes
- New tools use the resolver-based credential model (per-tenant ready); existing `twitter_post` keeps construction-time `TwitterCredentials` for backward compatibility. Persona wiring (P1.3) will switch the persona's `twitter_post` instance to the resolver pattern.
- All new tests use `wiremock` for HTTP stubbing; no live network calls in CI.
```

- [ ] **Step 4: Commit the CHANGELOG**

```bash
git add CHANGELOG.md
git commit -m "$(cat <<'EOF'
docs(changelog): heartbit-ghost P1.1 — X tool family

5 new X tools (user, search, mentions, reply, thread) in heartbit-ghost
plus media + alt-text extension on heartbit-core's existing
TwitterPostTool. Shared XClient with OAuth1 + credential resolution.
~35-40 new tests via wiremock.

Refs: docs/superpowers/specs/2026-05-07-heartbit-ghost-p1.1-x-tools-design.md
EOF
)"
```

- [ ] **Step 5: Print final commit log for the branch**

```bash
git log --oneline main..feat/heartbit-ghost-p1.1
```

Expected: ~9-10 commits — the spec doc + 8 implementation commits + CHANGELOG.

---

## Acceptance criteria

P1.1 is done when (per spec §7):

- 5 new tools (`TwitterUserTool`, `TwitterSearchTool`, `TwitterMentionsTool`, `TwitterReplyTool`, `TwitterThreadTool`) exist as `pub` types in `heartbit_ghost::tools::*`, each implementing `heartbit_core::Tool`.
- `TwitterPostTool` accepts optional `media_url` + `media_alt_text` and posts text+image when both are set.
- ~35–40 new tests across heartbit-ghost + heartbit-core; all passing.
- `cargo fmt -- --check && cargo clippy --workspace --all-features -- -D warnings && cargo test --workspace --all-features` all green.
- CHANGELOG entry under `[Unreleased]` lists the 5 new tools + twitter_post extension.

## Out of scope (re-stated)

- `twitter_dm`, `twitter_schedule`, `twitter_metrics` — P1.4
- Wiring tools into `heartbit_ghost::XGhostPersona::expand()` — P1.3
- Persona-level rate limiting — P1.4
- Per-tenant audit logging via `AuditSink` — P1.3+
- Bearer-token auth for read-only endpoints — deferred
- DRY-ing the OAuth1 logic between heartbit-core's twitter_post and heartbit-ghost's XClient — post-merge cleanup
- Live X API smoke test (manual one-off, requires real credentials) — operator responsibility post-merge

## Reference

- Spec: `docs/superpowers/specs/2026-05-07-heartbit-ghost-p1.1-x-tools-design.md`
- Heartbit-ghost umbrella: `docs/superpowers/specs/2026-05-07-heartbit-ghost-x-agent-design.md`
- Foundation (`ExecutionContext`, `CredentialResolver`, `Secret`): `docs/superpowers/specs/2026-05-07-heartbit-foundation-design.md`
- P1.0 plan: `docs/superpowers/plans/2026-05-07-heartbit-ghost-p1.0-scaffolding.md`
