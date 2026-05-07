//! Live smoke test — calls `TwitterUserTool` against the real X API to
//! verify the OAuth1 path works end-to-end.
//!
//! # Usage
//!
//! ```bash
//! X_CONSUMER_KEY=... X_CONSUMER_SECRET=... \
//! X_ACCESS_TOKEN=... X_ACCESS_TOKEN_SECRET=... \
//!     cargo run -p heartbit-ghost --example smoke_user
//! ```
//!
//! Looks up `@karpathy` by default. Pass a handle as a positional arg to override:
//!
//! ```bash
//! cargo run -p heartbit-ghost --example smoke_user -- jack
//! ```
//!
//! # What this proves
//!
//! - `XClient::from_context` resolves the 4 OAuth1 credentials from
//!   `ExecutionContext::credentials`
//! - The OAuth1 signing produces a valid `Authorization` header that X accepts
//! - `GET /2/users/by/username/:handle` round-trips with `user.fields=...`
//! - The response parses into the `TwitterUserTool`'s output shape
//!
//! Read-only tool — no posts, replies, or DMs are sent.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use heartbit_core::{CredentialResolver, ExecutionContext, Secret, Tool};
use heartbit_ghost::tools::TwitterUserTool;

/// Minimal env-var-backed `CredentialResolver` for examples and ad-hoc testing.
///
/// In production, a `CredentialResolver` impl typically reads from a vault,
/// AWS Secrets Manager, or a tenant-scoped config — not from process env vars.
/// This implementation is suitable for development and one-off smoke tests.
struct EnvResolver;

impl CredentialResolver for EnvResolver {
    fn resolve(
        &self,
        name: &str,
    ) -> Pin<Box<dyn Future<Output = Result<Secret, heartbit_core::Error>> + Send + '_>> {
        let name = name.to_string();
        Box::pin(async move {
            std::env::var(&name).map(Secret::new).map_err(|_| {
                heartbit_core::Error::Agent(format!("env var {name} not set"))
            })
        })
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let handle = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "karpathy".to_string());

    let ctx = ExecutionContext {
        credentials: Some(Arc::new(EnvResolver)),
        ..ExecutionContext::default()
    };

    let tool = TwitterUserTool::new();
    let input = serde_json::json!({ "handle": handle });

    println!("Calling twitter_user for @{handle}...");
    let result = tool.execute(&ctx, input).await?;

    if result.is_error {
        eprintln!("Error: {}", result.content);
        std::process::exit(1);
    }

    println!("Success:");
    let parsed: serde_json::Value = serde_json::from_str(&result.content)?;
    println!("{}", serde_json::to_string_pretty(&parsed)?);
    Ok(())
}
