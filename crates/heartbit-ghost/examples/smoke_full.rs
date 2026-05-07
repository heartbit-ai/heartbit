//! Full live smoke test for heartbit-ghost P1.1 — exercises all 5 X tools
//! against the real X v2 API.
//!
//! `twitter_post` (heartbit-core, extended in P1.1 §1.1) is covered by the
//! 17 unit tests in heartbit-core (13 existing + 4 new wiremock for the
//! media path). This binary focuses on heartbit-ghost's 5 new tools.
//!
//! # Usage
//!
//! ```bash
//! set -a; source .env; set +a
//! cargo run -p heartbit-ghost --example smoke_full
//! ```
//!
//! # What this proves (live)
//!
//! Read-only (no public side-effects):
//! 1. `TwitterUserTool` — looks up `@karpathy`
//! 2. `TwitterSearchTool` — searches for recent tweets `from:karpathy`
//! 3. `TwitterMentionsTool` — fetches mentions of the auth'd user
//!
//! Write (each posts to your real X account; IDs printed for cleanup):
//! 4. `TwitterThreadTool` — posts a 2-tweet chained thread
//! 5. `TwitterReplyTool` — replies to the thread's root tweet
//!
//! # Cleanup
//!
//! At the end, the binary prints all tweet IDs created in steps 4-5.
//! Delete them manually via the X UI or the API. P1.1 doesn't include
//! a `twitter_delete` tool (deferred to P1.4).

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use heartbit_core::{CredentialResolver, ExecutionContext, Secret, Tool};
use heartbit_ghost::tools::{
    TwitterMentionsTool, TwitterReplyTool, TwitterSearchTool, TwitterThreadTool, TwitterUserTool,
};
use serde_json::{Value, json};

/// Env-var-backed CredentialResolver.
struct EnvResolver;

impl CredentialResolver for EnvResolver {
    fn resolve(
        &self,
        name: &str,
    ) -> Pin<Box<dyn Future<Output = Result<Secret, heartbit_core::Error>> + Send + '_>> {
        let name = name.to_string();
        Box::pin(async move {
            std::env::var(&name)
                .map(Secret::new)
                .map_err(|_| heartbit_core::Error::Agent(format!("env var {name} not set")))
        })
    }
}

fn parse_json(content: &str) -> Value {
    serde_json::from_str(content).unwrap_or_else(|_| json!({"_raw": content}))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = ExecutionContext {
        credentials: Some(Arc::new(EnvResolver)),
        ..ExecutionContext::default()
    };

    // The auth'd user's numeric id is the prefix of the access token.
    let auth_user_id = std::env::var("X_ACCESS_TOKEN")?
        .split('-')
        .next()
        .ok_or("malformed X_ACCESS_TOKEN")?
        .to_string();

    let mut created_tweet_ids: Vec<String> = Vec::new();

    // ───── 1. twitter_user ─────────────────────────────────────────────
    println!("══════ 1. twitter_user @karpathy ══════");
    let res = TwitterUserTool::new()
        .execute(&ctx, json!({"handle": "karpathy"}))
        .await?;
    if res.is_error {
        eprintln!("FAIL: {}", res.content);
        std::process::exit(1);
    }
    let user_data = parse_json(&res.content);
    println!(
        "✓ id={}, followers={}",
        user_data["id"].as_str().unwrap_or("?"),
        user_data["public_metrics"]["followers_count"]
            .as_u64()
            .unwrap_or(0)
    );

    // ───── 2. twitter_search ───────────────────────────────────────────
    println!("\n══════ 2. twitter_search from:karpathy ══════");
    let res = TwitterSearchTool::new()
        .execute(&ctx, json!({"query": "from:karpathy", "max_results": 10}))
        .await?;
    if res.is_error {
        eprintln!("FAIL: {}", res.content);
        std::process::exit(1);
    }
    let search_data = parse_json(&res.content);
    let tweet_count = search_data["tweets"].as_array().map_or(0, |a| a.len());
    println!("✓ {tweet_count} recent tweet(s) returned");
    if let Some(first) = search_data["tweets"].get(0) {
        let preview: String = first["text"]
            .as_str()
            .unwrap_or("")
            .chars()
            .take(80)
            .collect();
        println!("  first preview: {preview}...");
    }

    // ───── 3. twitter_mentions ─────────────────────────────────────────
    println!("\n══════ 3. twitter_mentions for auth'd user (id={auth_user_id}) ══════");
    let res = TwitterMentionsTool::new()
        .execute(&ctx, json!({"user_id": auth_user_id, "max_results": 5}))
        .await?;
    if res.is_error {
        eprintln!("FAIL: {}", res.content);
        std::process::exit(1);
    }
    let mentions_data = parse_json(&res.content);
    let mention_count = mentions_data["mentions"].as_array().map_or(0, |a| a.len());
    println!("✓ {mention_count} recent mention(s) returned");

    // ───── 4. twitter_thread (write) ───────────────────────────────────
    println!("\n══════ 4. twitter_thread — 2-tweet chain ══════");
    let res = TwitterThreadTool::new()
        .execute(
            &ctx,
            json!({"tweets": [
                "heartbit-ghost P1.1 smoke test 🧪 — autonomous X agent runtime. github.com/heartbit-ai/heartbit (delete me)",
                "follow-up: validating twitter_thread. (delete me too)"
            ]}),
        )
        .await?;
    if res.is_error {
        eprintln!("FAIL: {}", res.content);
        std::process::exit(1);
    }
    let thread_data = parse_json(&res.content);
    let thread_root_id = thread_data["thread_root_id"]
        .as_str()
        .ok_or("thread returned no thread_root_id")?
        .to_string();
    if let Some(ids) = thread_data["tweet_ids"].as_array() {
        for id in ids {
            if let Some(s) = id.as_str() {
                created_tweet_ids.push(s.to_string());
                println!("✓ posted thread tweet id={s}");
            }
        }
    }

    // ───── 5. twitter_reply (write — replies to the thread's root) ─────
    println!("\n══════ 5. twitter_reply — replying to thread root {thread_root_id} ══════");
    let res = TwitterReplyTool::new()
        .execute(
            &ctx,
            json!({
                "text": "smoke-test reply (delete me) — validates twitter_reply",
                "in_reply_to": thread_root_id,
            }),
        )
        .await?;
    if res.is_error {
        eprintln!("FAIL: {}", res.content);
        std::process::exit(1);
    }
    let reply_data = parse_json(&res.content);
    let reply_id = reply_data["tweet_id"]
        .as_str()
        .ok_or("reply returned no tweet_id")?
        .to_string();
    created_tweet_ids.push(reply_id.clone());
    println!("✓ posted reply id={reply_id}");

    // ───── Summary + cleanup info ──────────────────────────────────────
    println!("\n══════════════════════════════════════════════════════════");
    println!("ALL 5 LIVE TESTS PASSED ✓");
    println!("══════════════════════════════════════════════════════════");
    println!(
        "\nCreated {} tweet(s) on your X account. URLs to delete:\n",
        created_tweet_ids.len()
    );
    for id in &created_tweet_ids {
        println!("  https://twitter.com/i/web/status/{id}");
    }
    println!("\n(P1.1 does not ship a twitter_delete tool — deferred to P1.4.)");

    Ok(())
}
