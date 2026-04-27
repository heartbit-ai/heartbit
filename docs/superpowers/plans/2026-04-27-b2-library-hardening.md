# B2 Library Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the broken test gate, add SSRF defense to `WebFetchTool`, centralize HTTP client construction in `heartbit::http`, add a constant-time auth helper in `heartbit::auth::ct`, swap the bearer compare in heartbit-cli to use it, clear all 26 `cargo clippy --workspace --all-targets` warnings, and tighten CI to enforce all of the above.

**Architecture:** Add two small public modules (`heartbit::http`, `heartbit::auth::ct`). Promote `subtle` from optional/`sensor`-gated to a non-optional dep of `heartbit`. Migrate nine `reqwest::Client::builder()` sites to the new factories — the three sites that take agent-controllable URLs additionally validate via `SafeUrl::parse(...)`. Fix five test-helper struct literals in `heartbit-cli` that block `cargo test --workspace --no-run`. Apply `cargo clippy --fix` plus one manual `MutexGuard` refactor in `tests/sensor_pipeline_e2e.rs`. Tighten the existing CI workflow.

**Tech Stack:** Rust 2024, tokio, reqwest 0.12, `subtle` 2 for constant-time compare, `tokio::net::lookup_host` for DNS, hand-rolled `tokio::net::TcpListener` for the redirect test (no new test deps).

**Spec:** `docs/superpowers/specs/2026-04-27-b2-library-hardening-design.md`

---

## File Map

**Create:**
- `crates/heartbit/src/auth/ct.rs` — constant-time helpers
- `crates/heartbit/src/http.rs` — `IpPolicy`, `SafeUrl`, `safe_client_builder`, `vendor_client_builder`

**Modify (heartbit lib):**
- `crates/heartbit/Cargo.toml` — promote `subtle` to non-optional; drop from `sensor` feature list
- `crates/heartbit/src/lib.rs` — add `pub mod http;`
- `crates/heartbit/src/auth/mod.rs` — add `pub mod ct;`
- `crates/heartbit/src/tool/builtins/webfetch.rs` — migrate + add `with_ip_policy` + SSRF tests
- `crates/heartbit/src/tool/a2a.rs` — migrate prod client + tests
- `crates/heartbit/src/sensor/sources/rss.rs` — migrate (validates URL)
- `crates/heartbit/src/sensor/sources/jmap.rs` — migrate (vendor)
- `crates/heartbit/src/sensor/sources/weather.rs` — migrate (vendor)
- `crates/heartbit/src/tool/builtins/websearch.rs` — migrate (vendor; 2 sites)
- `crates/heartbit/src/tool/builtins/image_generate.rs` — migrate (vendor)
- `crates/heartbit/src/tool/builtins/tts.rs` — migrate (vendor)
- `crates/heartbit/src/tool/builtins/twitter_post.rs` — migrate (vendor)
- `crates/heartbit/src/tool/mcp.rs` — leave inline; add cross-reference comment
- `crates/heartbit/src/agent/tool_filter.rs` — clippy: 5× `.iter().copied().collect()` → `.to_vec()`
- `crates/heartbit/src/agent/guardrails/injection.rs` — clippy: collapsible-if @ 477
- `crates/heartbit/src/agent/mod.rs` — clippy: 1 site @ 2927
- `crates/heartbit/src/channel/telegram/adapter.rs` — clippy: 1 site @ 808
- `crates/heartbit/src/lsp/client.rs` — clippy: 1 site @ 411
- `crates/heartbit/src/daemon/cron.rs` — clippy: 1 site @ 146
- `crates/heartbit/src/daemon/heartbit_pulse.rs` — clippy: 1 site @ 315
- `crates/heartbit/src/daemon/kafka.rs` — clippy: collapsible-if @ 603 + type alias for mock_producer
- `crates/heartbit/src/daemon/types.rs` — clippy: field-reassign-with-default @ 540–541
- `crates/heartbit/src/sensor/triage/structured.rs` — clippy: drop `::default()` @ 172
- `CHANGELOG.md` — add entry for `WebFetchTool` SSRF default-deny

**Modify (heartbit-cli):**
- `crates/heartbit-cli/src/daemon/eval.rs:244,258` — add `twitter_credentials: None,` and `base_url: None,`
- `crates/heartbit-cli/src/daemon/execute.rs:1120,1134,1483` — add the same two fields at three sites
- `crates/heartbit-cli/src/daemon/auth.rs` — bearer compare uses `heartbit::auth::ct::contains`

**Modify (tests):**
- `crates/heartbit/tests/sensor_pipeline_e2e.rs` — auto-fix collapsible-if + drop MutexGuard before `.await` @ 5058–5085

**Modify (CI):**
- `.github/workflows/ci.yml` — `clippy --workspace --all-targets`, explicit `--no-run` step, `--lib` test step

---

## Task 1: Restore the test compile gate

**Why first:** Smallest, highest-impact step. After this, `cargo test --workspace --no-run` is green and every later task can be tested as it goes.

**Files:**
- Modify: `crates/heartbit-cli/src/daemon/eval.rs`
- Modify: `crates/heartbit-cli/src/daemon/execute.rs`

- [ ] **Step 1.1: Verify the gate is currently red.**

Run:
```bash
cargo test --workspace --no-run 2>&1 | grep -E "^error" | head -10
```
Expected: 5 `error[E0063]: missing field` lines mentioning `base_url` and `twitter_credentials`.

- [ ] **Step 1.2: Patch `eval.rs:244` (RuntimeRequest).**

In `crates/heartbit-cli/src/daemon/eval.rs`, inside `make_test_request()`, the `RuntimeRequest { ... }` literal currently ends with `initial_content: vec![],`. Add one line just after it:

```rust
            initial_content: vec![],
            twitter_credentials: None,
        }
    }
```

- [ ] **Step 1.3: Patch `eval.rs:258` (RuntimeProviderConfig).**

Inside the same function, the `provider: RuntimeProviderConfig { ... }` block ends with `prompt_caching: false,`. Add `base_url: None,` immediately after:

```rust
            provider: RuntimeProviderConfig {
                provider_type: RuntimeProviderType::Anthropic,
                api_key: "sk-test".into(),
                model: "claude-sonnet-4-20250514".into(),
                prompt_caching: false,
                base_url: None,
            },
```

- [ ] **Step 1.4: Patch `execute.rs:1120` (RuntimeRequest in `make_test_request`).**

In `crates/heartbit-cli/src/daemon/execute.rs`, inside `mod tests { fn make_test_request() }`, do the same `twitter_credentials: None,` addition right after the `initial_content: vec![],` line (or as the last field before the closing brace if `initial_content` is not yet a field).

- [ ] **Step 1.5: Patch `execute.rs:1134` (RuntimeProviderConfig in `make_test_request`).**

Same fix as Step 1.3 — add `base_url: None,` after `prompt_caching: false,`.

- [ ] **Step 1.6: Patch `execute.rs:1483` (RuntimeProviderConfig literal in `build_workflow_agent_basic` test).**

In the `#[tokio::test] async fn build_workflow_agent_basic()` body, the inline `super::build_provider(&heartbit::RuntimeProviderConfig { ... })` literal needs `base_url: None,` after `prompt_caching: false,`.

- [ ] **Step 1.7: Run the gate.**

Run:
```bash
cargo test --workspace --no-run 2>&1 | tail -5
```
Expected: `Finished` line, exit 0. No `error[E0063]` errors.

- [ ] **Step 1.8: Commit.**

```bash
git add crates/heartbit-cli/src/daemon/eval.rs crates/heartbit-cli/src/daemon/execute.rs
git commit -m "fix(cli): add missing base_url/twitter_credentials to test helpers

The test helpers in daemon/eval.rs and daemon/execute.rs were not updated
when commit a81f84c added required fields to RuntimeRequest and
RuntimeProviderConfig. cargo test --workspace --no-run now compiles."
```

---

## Task 2: Promote `subtle`; add `heartbit::auth::ct`

**Why next:** Provides the bearer-fix primitive (used by Task 6) and unblocks Task 3 from any feature-flag entanglement.

**Files:**
- Modify: `crates/heartbit/Cargo.toml`
- Create: `crates/heartbit/src/auth/ct.rs`
- Modify: `crates/heartbit/src/auth/mod.rs`

- [ ] **Step 2.1: Promote `subtle` to non-optional in `heartbit/Cargo.toml`.**

In `crates/heartbit/Cargo.toml`, find:

```toml
subtle = { workspace = true, optional = true }
```

Replace with:

```toml
subtle = { workspace = true }
```

In the same file, find the `sensor` feature line:

```toml
sensor = ["daemon", "dep:quick-xml", "dep:sha2", "dep:hex", "dep:subtle"]
```

Replace with (drop `"dep:subtle"`):

```toml
sensor = ["daemon", "dep:quick-xml", "dep:sha2", "dep:hex"]
```

- [ ] **Step 2.2: Verify lib still builds without features.**

```bash
cargo build -p heartbit --no-default-features --features core 2>&1 | tail -5
```
Expected: `Finished` line, exit 0.

- [ ] **Step 2.3: Write the failing tests for `auth::ct`.**

Create the new file `crates/heartbit/src/auth/ct.rs` with only the test module (no implementation yet) so we observe the failures:

```rust
//! Constant-time string and set comparison helpers.

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn ct_eq_str_equal_strings() {
        assert!(ct_eq_str("hunter2", "hunter2"));
    }

    #[test]
    fn ct_eq_str_different_same_length() {
        assert!(!ct_eq_str("hunter2", "Hunter2"));
    }

    #[test]
    fn ct_eq_str_different_lengths() {
        assert!(!ct_eq_str("hunter2", "hunter22"));
        assert!(!ct_eq_str("", "x"));
    }

    #[test]
    fn ct_eq_str_empty() {
        assert!(ct_eq_str("", ""));
    }

    #[test]
    fn contains_hits_when_present() {
        let mut s = HashSet::new();
        s.insert("alpha".to_string());
        s.insert("bravo".to_string());
        assert!(contains(&s, "alpha"));
        assert!(contains(&s, "bravo"));
    }

    #[test]
    fn contains_misses_when_absent() {
        let mut s = HashSet::new();
        s.insert("alpha".to_string());
        assert!(!contains(&s, "alphax"));
        assert!(!contains(&s, "alph"));
        assert!(!contains(&s, ""));
    }

    #[test]
    fn contains_empty_set_is_always_false() {
        let s: HashSet<String> = HashSet::new();
        assert!(!contains(&s, "anything"));
    }
}
```

- [ ] **Step 2.4: Wire the module in `auth/mod.rs`.**

Edit `crates/heartbit/src/auth/mod.rs`. Add at the bottom (ungated — pure helper, no feature dependency):

```rust
pub mod ct;
```

- [ ] **Step 2.5: Run the tests; verify they fail.**

```bash
cargo test -p heartbit --lib auth::ct:: 2>&1 | tail -10
```
Expected: compile error — `cannot find function ct_eq_str` and `cannot find function contains` in `auth::ct`.

- [ ] **Step 2.6: Implement `ct_eq_str` and `contains`.**

Replace the contents of `crates/heartbit/src/auth/ct.rs` with the full module:

```rust
//! Constant-time string and set comparison helpers.
//!
//! Use these when comparing secrets (bearer tokens, signing keys, HMACs)
//! to avoid leaking information through timing side-channels. Comparing
//! two equal-length non-equal strings is constant-time. The length
//! difference itself is intentionally not constant-time: token length is
//! not the secret being protected.

use std::collections::HashSet;

use subtle::ConstantTimeEq;

/// Constant-time string equality.
///
/// Returns `false` immediately if lengths differ; this short-circuit is
/// intentional (length is not the secret).
pub fn ct_eq_str(a: &str, b: &str) -> bool {
    a.len() == b.len() && bool::from(a.as_bytes().ct_eq(b.as_bytes()))
}

/// Constant-time membership test against a `HashSet<String>`.
///
/// Iterates every entry on every call; O(n) by design. Use only for sets
/// small enough that the linear scan is acceptable (bearer-token allow-lists,
/// signing-key sets, etc.). For large sets, use a different approach
/// — constant-time HashSet lookup is not something this helper provides.
pub fn contains(set: &HashSet<String>, candidate: &str) -> bool {
    let mut hit = false;
    for known in set {
        // bitwise-or so every iteration runs regardless of an earlier hit
        hit |= ct_eq_str(known, candidate);
    }
    hit
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn ct_eq_str_equal_strings() {
        assert!(ct_eq_str("hunter2", "hunter2"));
    }

    #[test]
    fn ct_eq_str_different_same_length() {
        assert!(!ct_eq_str("hunter2", "Hunter2"));
    }

    #[test]
    fn ct_eq_str_different_lengths() {
        assert!(!ct_eq_str("hunter2", "hunter22"));
        assert!(!ct_eq_str("", "x"));
    }

    #[test]
    fn ct_eq_str_empty() {
        assert!(ct_eq_str("", ""));
    }

    #[test]
    fn contains_hits_when_present() {
        let mut s = HashSet::new();
        s.insert("alpha".to_string());
        s.insert("bravo".to_string());
        assert!(contains(&s, "alpha"));
        assert!(contains(&s, "bravo"));
    }

    #[test]
    fn contains_misses_when_absent() {
        let mut s = HashSet::new();
        s.insert("alpha".to_string());
        assert!(!contains(&s, "alphax"));
        assert!(!contains(&s, "alph"));
        assert!(!contains(&s, ""));
    }

    #[test]
    fn contains_empty_set_is_always_false() {
        let s: HashSet<String> = HashSet::new();
        assert!(!contains(&s, "anything"));
    }
}
```

- [ ] **Step 2.7: Run the tests; verify they pass.**

```bash
cargo test -p heartbit --lib auth::ct:: 2>&1 | tail -5
```
Expected: `7 passed; 0 failed`.

- [ ] **Step 2.8: Run fmt + clippy on the lib.**

```bash
cargo fmt -- --check && cargo clippy -p heartbit --lib -- -D warnings 2>&1 | tail -3
```
Expected: both succeed.

- [ ] **Step 2.9: Commit.**

```bash
git add crates/heartbit/Cargo.toml crates/heartbit/src/auth/mod.rs crates/heartbit/src/auth/ct.rs
git commit -m "feat(auth): add heartbit::auth::ct constant-time helpers

ct_eq_str and contains use subtle::ConstantTimeEq to avoid timing
side-channels when comparing bearer tokens or other secrets against an
allow-list. subtle was previously gated behind the sensor feature; it is
now a non-optional dep of heartbit so this module is always available."
```

---

## Task 3: Add `heartbit::http` (no consumers yet)

**Why now:** Self-contained module with full unit tests; ships before any migration so reviewers can validate the policy in isolation.

**Files:**
- Create: `crates/heartbit/src/http.rs`
- Modify: `crates/heartbit/src/lib.rs`

- [ ] **Step 3.1: Wire the module in `lib.rs`.**

In `crates/heartbit/src/lib.rs`, after the existing `pub mod` lines (around line 90, near `pub mod auth;`), add:

```rust
pub mod http;
```

This is ungated — `reqwest`, `tokio`, and `Url` are all unconditionally available.

- [ ] **Step 3.2: Write the failing tests.**

Create `crates/heartbit/src/http.rs` with **only** the test module first:

```rust
//! HTTP client factories and URL validation primitives.
//!
//! See [`SafeUrl::parse`] and [`safe_client_builder`] for the public API.

#[cfg(test)]
mod tests {
    use super::*;

    // ---- IpPolicy parser ----
    //
    // We test the pure parser (`from_env_value`), not `from_env` itself.
    // Mutating real env vars races with parallel tests in the cargo harness;
    // the parser is the actual logic and is testable without that risk.

    #[test]
    fn ip_policy_unset_is_strict() {
        assert_eq!(IpPolicy::from_env_value(None), IpPolicy::Strict);
    }

    #[test]
    fn ip_policy_one_is_allow() {
        assert_eq!(IpPolicy::from_env_value(Some("1")), IpPolicy::AllowPrivate);
    }

    #[test]
    fn ip_policy_true_case_insensitive_is_allow() {
        assert_eq!(IpPolicy::from_env_value(Some("TRUE")), IpPolicy::AllowPrivate);
        assert_eq!(IpPolicy::from_env_value(Some("True")), IpPolicy::AllowPrivate);
        assert_eq!(IpPolicy::from_env_value(Some("  true  ")), IpPolicy::AllowPrivate);
    }

    #[test]
    fn ip_policy_zero_is_strict() {
        assert_eq!(IpPolicy::from_env_value(Some("0")), IpPolicy::Strict);
        assert_eq!(IpPolicy::from_env_value(Some("false")), IpPolicy::Strict);
    }

    #[test]
    fn ip_policy_garbage_is_strict() {
        assert_eq!(IpPolicy::from_env_value(Some("yesplz")), IpPolicy::Strict);
        assert_eq!(IpPolicy::from_env_value(Some("")), IpPolicy::Strict);
    }

    // ---- SafeUrl::parse — scheme ----

    #[tokio::test]
    async fn safe_url_rejects_non_http_scheme() {
        let err = SafeUrl::parse("file:///etc/passwd", IpPolicy::Strict).await.unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("scheme") && msg.contains("file"), "got: {msg}");
    }

    #[tokio::test]
    async fn safe_url_rejects_invalid_url() {
        let err = SafeUrl::parse("not a url", IpPolicy::Strict).await.unwrap_err();
        assert!(err.to_string().contains("invalid URL"));
    }

    // ---- SafeUrl::parse — literal IP blocklist (Strict) ----

    #[tokio::test]
    async fn safe_url_rejects_loopback_v4() {
        assert!(SafeUrl::parse("http://127.0.0.1/", IpPolicy::Strict).await.is_err());
    }

    #[tokio::test]
    async fn safe_url_rejects_loopback_v6() {
        assert!(SafeUrl::parse("http://[::1]/", IpPolicy::Strict).await.is_err());
    }

    #[tokio::test]
    async fn safe_url_rejects_link_local_v4() {
        // AWS / GCE IMDS
        assert!(SafeUrl::parse("http://169.254.169.254/", IpPolicy::Strict).await.is_err());
    }

    #[tokio::test]
    async fn safe_url_rejects_link_local_v6() {
        assert!(SafeUrl::parse("http://[fe80::1]/", IpPolicy::Strict).await.is_err());
    }

    #[tokio::test]
    async fn safe_url_rejects_rfc1918() {
        for h in ["10.0.0.1", "172.16.0.1", "192.168.1.1"] {
            let r = SafeUrl::parse(&format!("http://{h}/"), IpPolicy::Strict).await;
            assert!(r.is_err(), "{h} should be rejected");
        }
    }

    #[tokio::test]
    async fn safe_url_rejects_cgnat() {
        assert!(SafeUrl::parse("http://100.64.0.1/", IpPolicy::Strict).await.is_err());
        assert!(SafeUrl::parse("http://100.127.255.1/", IpPolicy::Strict).await.is_err());
    }

    #[tokio::test]
    async fn safe_url_rejects_ula() {
        assert!(SafeUrl::parse("http://[fc00::1]/", IpPolicy::Strict).await.is_err());
        assert!(SafeUrl::parse("http://[fd00::1]/", IpPolicy::Strict).await.is_err());
    }

    #[tokio::test]
    async fn safe_url_rejects_multicast() {
        assert!(SafeUrl::parse("http://224.0.0.1/", IpPolicy::Strict).await.is_err());
        assert!(SafeUrl::parse("http://[ff00::1]/", IpPolicy::Strict).await.is_err());
    }

    #[tokio::test]
    async fn safe_url_rejects_unspecified() {
        assert!(SafeUrl::parse("http://0.0.0.0/", IpPolicy::Strict).await.is_err());
        assert!(SafeUrl::parse("http://[::]/", IpPolicy::Strict).await.is_err());
    }

    #[tokio::test]
    async fn safe_url_rejects_broadcast() {
        assert!(SafeUrl::parse("http://255.255.255.255/", IpPolicy::Strict).await.is_err());
    }

    #[tokio::test]
    async fn safe_url_accepts_public_ip() {
        let safe = SafeUrl::parse("http://8.8.8.8/", IpPolicy::Strict).await.unwrap();
        assert_eq!(safe.as_str(), "http://8.8.8.8/");
    }

    // ---- SafeUrl::parse — DNS resolution ----

    #[tokio::test]
    async fn safe_url_rejects_localhost_dns() {
        // "localhost" resolves to 127.0.0.1 / ::1 — must be rejected under Strict.
        assert!(SafeUrl::parse("http://localhost/", IpPolicy::Strict).await.is_err());
    }

    // ---- SafeUrl::parse — AllowPrivate bypass ----

    #[tokio::test]
    async fn safe_url_allow_private_accepts_loopback() {
        let safe = SafeUrl::parse("http://127.0.0.1/", IpPolicy::AllowPrivate).await.unwrap();
        assert_eq!(safe.as_str(), "http://127.0.0.1/");
    }

    #[tokio::test]
    async fn safe_url_allow_private_accepts_localhost() {
        let safe = SafeUrl::parse("http://localhost/", IpPolicy::AllowPrivate).await.unwrap();
        assert_eq!(safe.as_str(), "http://localhost/");
    }

    // ---- Rejection message guidance ----

    #[tokio::test]
    async fn safe_url_rejection_message_mentions_override() {
        let err = SafeUrl::parse("http://127.0.0.1/", IpPolicy::Strict).await.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("HEARTBIT_ALLOW_PRIVATE_IPS"),
            "rejection message should mention the override env var; got: {msg}"
        );
    }

    // ---- Client builders ----

    #[tokio::test]
    async fn safe_client_builder_does_not_follow_redirects() {
        // Spin up a tiny in-process listener that returns 302 → /landed.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            // Accept one request, return a 302.
            if let Ok((mut sock, _)) = listener.accept().await {
                use tokio::io::{AsyncReadExt, AsyncWriteExt};
                let mut buf = [0u8; 1024];
                let _ = sock.read(&mut buf).await;
                let resp = b"HTTP/1.1 302 Found\r\nLocation: /landed\r\nContent-Length: 0\r\n\r\n";
                let _ = sock.write_all(resp).await;
                let _ = sock.shutdown().await;
            }
        });

        let client = safe_client_builder().build().unwrap();
        let resp = client.get(format!("http://{addr}/start")).send().await.unwrap();
        assert_eq!(resp.status().as_u16(), 302, "redirect must NOT be followed");
    }

    #[test]
    fn vendor_client_builder_compiles_and_builds() {
        let _ = vendor_client_builder().build().unwrap();
    }
}
```

- [ ] **Step 3.3: Run the tests; verify they fail to compile.**

```bash
cargo test -p heartbit --lib http:: 2>&1 | tail -10
```
Expected: compile error — `cannot find type IpPolicy` / `cannot find function safe_client_builder` etc.

- [ ] **Step 3.4: Implement the module.**

Replace `crates/heartbit/src/http.rs` contents (preserving the test module at the bottom) with:

```rust
//! HTTP client factories and URL validation primitives.
//!
//! Two factories return preconfigured `reqwest::ClientBuilder`s:
//! - [`safe_client_builder`] — for clients that send to user-controllable URLs.
//!   Caller is expected to validate URLs via [`SafeUrl::parse`] first.
//! - [`vendor_client_builder`] — for clients that send to operator-trusted
//!   vendor APIs (Twitter, OpenAI, etc.). No URL validation is implied.
//!
//! Both builders set `redirect(Policy::none())` so a 302 to a private IP
//! cannot bypass parse-time checks.
//!
//! # Limitation
//!
//! The IP blocklist is parse-time only. An attacker who controls a DNS name
//! can return a public IP at parse time and a private IP at TCP-connect time
//! and bypass this design. Defending against DNS rebind requires a custom
//! `reqwest::dns::Resolve` implementation that filters at connect time and
//! is deferred to a future round.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::str::FromStr;

use reqwest::redirect::Policy;
use reqwest::{ClientBuilder, Url};

use crate::error::Error;

/// Whether to permit requests to non-routable / private IPs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IpPolicy {
    /// Reject loopback, link-local, RFC1918, CGNAT, ULA, multicast,
    /// unspecified, broadcast. Library default.
    Strict,
    /// Allow any IP. Use only for single-tenant deployments where agents
    /// legitimately need internal-network access.
    AllowPrivate,
}

impl Default for IpPolicy {
    fn default() -> Self {
        Self::from_env()
    }
}

impl IpPolicy {
    /// Read the `HEARTBIT_ALLOW_PRIVATE_IPS` environment variable.
    ///
    /// Anything other than `"1"` or `"true"` (case-insensitive, trimmed)
    /// yields `Strict`. Unset → `Strict`.
    pub fn from_env() -> Self {
        Self::from_env_value(std::env::var("HEARTBIT_ALLOW_PRIVATE_IPS").ok().as_deref())
    }

    /// Pure parser — testable without mutating real env vars.
    pub(crate) fn from_env_value(value: Option<&str>) -> Self {
        match value {
            Some(v) => match v.trim().to_ascii_lowercase().as_str() {
                "1" | "true" => Self::AllowPrivate,
                _ => Self::Strict,
            },
            None => Self::Strict,
        }
    }
}

/// A URL that has passed scheme + IP-blocklist validation.
///
/// Construction via [`SafeUrl::parse`] is the only way to satisfy this type;
/// call sites that hold a `SafeUrl` proved they validated it.
#[derive(Debug, Clone)]
pub struct SafeUrl(Url);

impl SafeUrl {
    /// Parse `s`, require an `http://` or `https://` scheme, and reject if the
    /// host is a literal blocked IP, or if any IP that the host name resolves
    /// to is in the blocked set under `policy`.
    ///
    /// DNS resolution uses [`tokio::net::lookup_host`] with the URL's port (or
    /// the scheme's default port). All resolved addresses are checked; if any
    /// resolved address is private, the URL is rejected.
    ///
    /// Under `IpPolicy::AllowPrivate`, no IP check is performed (scheme check
    /// still applies).
    pub async fn parse(s: &str, policy: IpPolicy) -> Result<Self, Error> {
        let url = Url::parse(s).map_err(|e| Error::Agent(format!("invalid URL: {e}")))?;
        let scheme = url.scheme();
        if scheme != "http" && scheme != "https" {
            return Err(Error::Agent(format!(
                "URL scheme {scheme:?} not allowed; only http and https"
            )));
        }
        if matches!(policy, IpPolicy::AllowPrivate) {
            return Ok(Self(url));
        }
        let host = url
            .host_str()
            .ok_or_else(|| Error::Agent("URL has no host".into()))?;
        let port = url.port_or_known_default().unwrap_or(80);

        // Literal IP fast-path.
        if let Ok(ip) = IpAddr::from_str(host) {
            if is_blocked(&ip) {
                return Err(reject(host));
            }
            return Ok(Self(url));
        }

        // DNS path: resolve and check every returned address.
        let addrs = tokio::net::lookup_host((host, port))
            .await
            .map_err(|e| Error::Agent(format!("DNS lookup failed for {host}: {e}")))?;
        let mut any = false;
        for sa in addrs {
            any = true;
            if is_blocked(&sa.ip()) {
                return Err(reject(host));
            }
        }
        if !any {
            return Err(Error::Agent(format!(
                "DNS lookup for {host} returned no addresses"
            )));
        }
        Ok(Self(url))
    }

    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }

    pub fn into_url(self) -> Url {
        self.0
    }
}

fn reject(host: &str) -> Error {
    Error::Agent(format!(
        "URL host {host} resolves to a private/loopback address; \
         refused (set HEARTBIT_ALLOW_PRIVATE_IPS=1 to override)"
    ))
}

fn is_blocked(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => is_blocked_v4(v4),
        IpAddr::V6(v6) => is_blocked_v6(v6),
    }
}

fn is_blocked_v4(ip: &Ipv4Addr) -> bool {
    ip.is_loopback()
        || ip.is_link_local()
        || ip.is_private()
        || ip.is_multicast()
        || ip.is_unspecified()
        || ip.is_broadcast()
        || is_cgnat_v4(ip)
}

fn is_blocked_v6(ip: &Ipv6Addr) -> bool {
    ip.is_loopback()
        || ip.is_multicast()
        || ip.is_unspecified()
        || is_link_local_v6(ip)
        || is_ula_v6(ip)
}

/// CGNAT range (RFC 6598): 100.64.0.0/10. `Ipv4Addr::is_private` does not
/// cover this; we check explicitly.
fn is_cgnat_v4(ip: &Ipv4Addr) -> bool {
    let [a, b, _, _] = ip.octets();
    a == 100 && (64..=127).contains(&b)
}

/// IPv6 link-local: fe80::/10. `Ipv6Addr::is_unicast_link_local` is unstable
/// as of MSRV; we check the prefix directly.
fn is_link_local_v6(ip: &Ipv6Addr) -> bool {
    let s = ip.segments()[0];
    (s & 0xffc0) == 0xfe80
}

/// IPv6 unique local (ULA): fc00::/7. `Ipv6Addr::is_unique_local` is unstable
/// as of MSRV; we check the prefix directly.
fn is_ula_v6(ip: &Ipv6Addr) -> bool {
    let s = ip.segments()[0];
    (s & 0xfe00) == 0xfc00
}

/// `reqwest::ClientBuilder` with `redirect(Policy::none())` baked in.
///
/// Use for clients that send to user-controllable URLs (`webfetch`, `a2a`,
/// `rss`). The caller is responsible for validating the URL via
/// [`SafeUrl::parse`] before issuing the request.
pub fn safe_client_builder() -> ClientBuilder {
    reqwest::Client::builder().redirect(Policy::none())
}

/// `reqwest::ClientBuilder` with `redirect(Policy::none())` baked in.
///
/// Use for clients that send to operator-trusted vendor APIs (Twitter, OpenAI,
/// SerpAPI, etc.). No IP validation is implied — the caller asserts the host
/// is operator-trusted. Redirects are still disabled so a hijacked DNS for the
/// vendor host cannot redirect a vendor call to a private address.
pub fn vendor_client_builder() -> ClientBuilder {
    reqwest::Client::builder().redirect(Policy::none())
}

// (test module from Step 3.2 stays here at the bottom of the file)
```

- [ ] **Step 3.5: Run the tests; verify all pass.**

```bash
cargo test -p heartbit --lib http:: 2>&1 | tail -5
```
Expected: 22+ tests passed; 0 failed.

If `safe_url_rejects_localhost_dns` flakes on systems where `localhost` is configured to resolve to a public IP (rare), document and weaken to "rejected OR resolves to non-public" — but verify locally first.

- [ ] **Step 3.6: Run fmt + clippy.**

```bash
cargo fmt -- --check && cargo clippy -p heartbit --lib --all-targets -- -D warnings 2>&1 | tail -3
```
Expected: both succeed.

- [ ] **Step 3.7: Commit.**

```bash
git add crates/heartbit/src/lib.rs crates/heartbit/src/http.rs
git commit -m "feat(http): add heartbit::http with SafeUrl, IpPolicy, client factories

Centralizes reqwest client construction. safe_client_builder and
vendor_client_builder both disable HTTP redirects so a 302 to a private
IP cannot bypass parse-time checks. SafeUrl::parse validates scheme and
applies a private-IP blocklist (loopback, link-local, RFC1918, CGNAT,
ULA, multicast, unspecified, broadcast) to literal IPs and to all DNS
lookup results. HEARTBIT_ALLOW_PRIVATE_IPS=1 (or the AllowPrivate
policy) bypasses the blocklist for single-tenant / dev deployments.

Limitation: parse-time only. DNS rebind defense deferred."
```

---

## Task 4: Migrate `WebFetchTool`; add SSRF tests; CHANGELOG entry

**Why now:** First real consumer of `crate::http`; validates the design end-to-end before bulk migration.

**Files:**
- Modify: `crates/heartbit/src/tool/builtins/webfetch.rs`
- Modify: `CHANGELOG.md`

- [ ] **Step 4.1: Write the failing SSRF tests.**

In `crates/heartbit/src/tool/builtins/webfetch.rs`, inside the `#[cfg(test)] mod tests { ... }` block (just before the closing `}`), add:

```rust
    #[tokio::test]
    async fn webfetch_rejects_loopback() {
        let tool = WebFetchTool::new();
        let result = tool
            .execute(json!({"url": "http://127.0.0.1/"}))
            .await
            .unwrap();
        assert!(result.is_error, "loopback must be rejected by default");
        assert!(
            result.content.contains("private/loopback"),
            "rejection message should explain why; got: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn webfetch_rejects_imds() {
        let tool = WebFetchTool::new();
        let result = tool
            .execute(json!({"url": "http://169.254.169.254/latest/meta-data/"}))
            .await
            .unwrap();
        assert!(result.is_error, "AWS/GCE IMDS must be rejected");
    }

    #[tokio::test]
    async fn webfetch_rejects_rfc1918() {
        let tool = WebFetchTool::new();
        let result = tool
            .execute(json!({"url": "http://10.0.0.1/"}))
            .await
            .unwrap();
        assert!(result.is_error);
    }

    #[tokio::test]
    async fn webfetch_rejects_localhost_dns() {
        let tool = WebFetchTool::new();
        let result = tool
            .execute(json!({"url": "http://localhost/"}))
            .await
            .unwrap();
        assert!(result.is_error, "localhost (resolves to 127.0.0.1/::1) must be rejected");
    }

    #[tokio::test]
    async fn webfetch_with_allow_private_ips_does_not_reject_loopback() {
        // Use with_ip_policy directly; do NOT mutate global env in tests.
        let tool = WebFetchTool::with_ip_policy(crate::http::IpPolicy::AllowPrivate);
        // The address won't resolve to anything reachable, so the request
        // itself fails — but it should NOT fail with the SSRF rejection.
        let result = tool
            .execute(json!({"url": "http://127.0.0.1:1/"}))
            .await
            .unwrap();
        assert!(result.is_error, "request to closed port should error");
        assert!(
            !result.content.contains("private/loopback"),
            "AllowPrivate should bypass the SSRF rejection; got: {}",
            result.content
        );
    }
```

- [ ] **Step 4.2: Run the new tests; verify they fail.**

```bash
cargo test -p heartbit --lib tool::builtins::webfetch::tests::webfetch_rejects 2>&1 | tail -10
```
Expected: tests don't compile (`with_ip_policy` undefined) or fail at runtime (current code does not reject these).

- [ ] **Step 4.3: Migrate `WebFetchTool` struct + constructors.**

In `crates/heartbit/src/tool/builtins/webfetch.rs`, replace the existing `WebFetchTool` struct and `impl` (lines 15–28) with:

```rust
pub struct WebFetchTool {
    client: reqwest::Client,
    ip_policy: crate::http::IpPolicy,
}

impl WebFetchTool {
    /// Construct with `IpPolicy::default()` — `Strict` unless
    /// `HEARTBIT_ALLOW_PRIVATE_IPS=1` is set in the environment.
    pub fn new() -> Self {
        Self::with_ip_policy(crate::http::IpPolicy::default())
    }

    /// Construct with an explicit IP policy.
    ///
    /// Use `IpPolicy::AllowPrivate` only for single-tenant / dev
    /// deployments where the agent legitimately needs to access internal
    /// services.
    pub fn with_ip_policy(ip_policy: crate::http::IpPolicy) -> Self {
        Self {
            client: crate::http::safe_client_builder()
                .user_agent("heartbit/0.1")
                .build()
                .expect("failed to build reqwest client"),
            ip_policy,
        }
    }
}

impl Default for WebFetchTool {
    fn default() -> Self {
        Self::new()
    }
}
```

- [ ] **Step 4.4: Replace the inline scheme check with `SafeUrl::parse`.**

In the same file, in `impl Tool for WebFetchTool`, replace lines 80–86 (the `// Validate URL scheme ...` block) with:

```rust
            // Validate scheme + private-IP blocklist via crate::http::SafeUrl.
            let safe_url = match crate::http::SafeUrl::parse(url, self.ip_policy).await {
                Ok(u) => u,
                Err(e) => return Ok(ToolOutput::error(e.to_string())),
            };
```

Then change line 90 (the `.get(url)` call) to use the validated URL:

```rust
                .get(safe_url.as_str())
```

- [ ] **Step 4.5: Update existing scheme-rejection tests.**

The existing tests `webfetch_rejects_file_scheme`, `webfetch_rejects_ftp_scheme`, and `rejects_uppercase_ftp_scheme` assert `result.content.contains("http://")`. The new error message format is `URL scheme "file" not allowed; only http and https`. Update each assertion from:

```rust
        assert!(result.content.contains("http://"));
```

to:

```rust
        assert!(
            result.content.contains("scheme") || result.content.contains("invalid URL"),
            "got: {}",
            result.content,
        );
```

(The `invalid URL` branch handles cases where Url::parse rejects before scheme inspection — e.g. uppercase `FTP://` is still parsed by `url` crate, but be defensive against future url-crate changes.)

- [ ] **Step 4.6: Run the full webfetch test suite.**

```bash
cargo test -p heartbit --lib tool::builtins::webfetch:: 2>&1 | tail -10
```
Expected: all tests pass (existing + 5 new SSRF tests).

- [ ] **Step 4.7: Run fmt + clippy.**

```bash
cargo fmt -- --check && cargo clippy -p heartbit --lib --all-targets -- -D warnings 2>&1 | tail -3
```

- [ ] **Step 4.8: Add CHANGELOG entry.**

In `CHANGELOG.md`, add a new section at the top of the file (above the existing first version section):

```markdown
## Unreleased

### Security

- `WebFetchTool` now refuses requests to private/loopback/link-local IPs by
  default (loopback, link-local incl. cloud IMDS at `169.254.169.254`,
  RFC1918, CGNAT, ULA, multicast, unspecified, broadcast). HTTP redirects
  are no longer followed (a 302 to a private IP is surfaced as a 302
  response, not silently followed). Set `HEARTBIT_ALLOW_PRIVATE_IPS=1` or
  pass `IpPolicy::AllowPrivate` via `WebFetchTool::with_ip_policy` for
  single-tenant deployments that legitimately need internal-network access.
```

- [ ] **Step 4.9: Commit.**

```bash
git add crates/heartbit/src/tool/builtins/webfetch.rs CHANGELOG.md
git commit -m "feat(webfetch): default-deny private IPs and HTTP redirects

WebFetchTool now uses crate::http::safe_client_builder (redirect off) and
validates every URL via SafeUrl::parse before issuing the request. The
default IpPolicy is Strict (reads HEARTBIT_ALLOW_PRIVATE_IPS for opt-out);
explicit policy via WebFetchTool::with_ip_policy.

Behavior change: agents that previously fetched http://127.0.0.1/,
http://169.254.169.254/, or other private hosts now receive a clean
ToolOutput::error with a self-documenting message instead of a
successful fetch. CHANGELOG entry added."
```

---

## Task 5: Migrate the eight remaining reqwest sites

**Why now:** Mechanical, low-risk, takes advantage of the validated `crate::http` from Tasks 3–4.

**Files:** Eight production files plus `tool/mcp.rs` for the cross-reference comment.

For each file below, the migration follows the same pattern:
1. Replace `reqwest::Client::builder()` with `crate::http::vendor_client_builder()` or `crate::http::safe_client_builder()` per the table.
2. For sites using `safe_client_builder`, wrap the URL in `SafeUrl::parse(...)` before issuing the request.
3. Run `cargo test -p heartbit --lib <module>::` to confirm no regression.

| Site | File | Builder | Validate URL? |
|---|---|---|---|
| 5.1 | `crates/heartbit/src/sensor/sources/weather.rs:163` | `vendor_client_builder` | no |
| 5.2 | `crates/heartbit/src/sensor/sources/jmap.rs:214` | `vendor_client_builder` | no |
| 5.3 | `crates/heartbit/src/tool/builtins/websearch.rs:34,47` | `vendor_client_builder` (both sites) | no |
| 5.4 | `crates/heartbit/src/tool/builtins/image_generate.rs:20` | `vendor_client_builder` | no |
| 5.5 | `crates/heartbit/src/tool/builtins/tts.rs:30` | `vendor_client_builder` | no |
| 5.6 | `crates/heartbit/src/tool/builtins/twitter_post.rs:53` | `vendor_client_builder` | no |
| 5.7 | `crates/heartbit/src/sensor/sources/rss.rs:49` | `safe_client_builder` | yes — feed URL is operator-config but treated as untrusted |
| 5.8 | `crates/heartbit/src/tool/a2a.rs:433` | `safe_client_builder` | yes — peer URL is agent-supplied |

- [ ] **Step 5.1: Migrate `sensor/sources/weather.rs`.**

In `crates/heartbit/src/sensor/sources/weather.rs:163`, replace:

```rust
            let client = reqwest::Client::builder()
```

with:

```rust
            let client = crate::http::vendor_client_builder()
```

Run `cargo test -p heartbit --lib sensor::sources::weather:: 2>&1 | tail -5` — expected pass.

- [ ] **Step 5.2: Migrate `sensor/sources/jmap.rs`.**

In `crates/heartbit/src/sensor/sources/jmap.rs:214`, same swap:

```rust
            let client = crate::http::vendor_client_builder()
```

Run `cargo test -p heartbit --lib sensor::sources::jmap:: 2>&1 | tail -5` — expected pass.

- [ ] **Step 5.3: Migrate `tool/builtins/websearch.rs`.**

Two construction sites at lines 34 and 47. Replace both `reqwest::Client::builder()` with `crate::http::vendor_client_builder()`. Run:

```bash
cargo test -p heartbit --lib tool::builtins::websearch:: 2>&1 | tail -5
```
Expected pass.

- [ ] **Step 5.4: Migrate `tool/builtins/image_generate.rs:20`.**

```rust
            client: crate::http::vendor_client_builder()
```

Run `cargo test -p heartbit --lib tool::builtins::image_generate:: 2>&1 | tail -5`.

- [ ] **Step 5.5: Migrate `tool/builtins/tts.rs:30`.**

```rust
            client: crate::http::vendor_client_builder()
```

Run `cargo test -p heartbit --lib tool::builtins::tts:: 2>&1 | tail -5`.

- [ ] **Step 5.6: Migrate `tool/builtins/twitter_post.rs:53`.**

```rust
            client: crate::http::vendor_client_builder()
```

Run `cargo test -p heartbit --lib tool::builtins::twitter_post:: 2>&1 | tail -5`.

- [ ] **Step 5.7: Migrate `sensor/sources/rss.rs:49` (with URL validation).**

First read the surrounding ~30 lines to locate the feed URL variable name and the existing return type:

```bash
sed -n '30,90p' crates/heartbit/src/sensor/sources/rss.rs
```

Then in `crates/heartbit/src/sensor/sources/rss.rs:49`, swap `reqwest::Client::builder()` for `crate::http::safe_client_builder()`.

Locate where the feed URL is used to issue the request (look for `client.get(...)` or similar within the same fn body). Insert URL validation before the request, using the existing variable name (likely `url`, `feed_url`, or `src.url`). Use `Error::Sensor(String)` for the failure path (the function already returns `Result<_, Error>`):

```rust
            let safe = crate::http::SafeUrl::parse(&feed_url, crate::http::IpPolicy::default())
                .await
                .map_err(|e| Error::Sensor(format!("rejecting feed URL: {e}")))?;
            let resp = client.get(safe.as_str()).send().await?;
```

If the function uses a different `Error` variant or wraps in `anyhow`, match that pattern instead. The principle is: the existing fn already has a `Result` return; reuse its error type.

Run `cargo test -p heartbit --lib sensor::sources::rss:: 2>&1 | tail -10` — expected pass. If a test exercises a non-public URL via mock server, it must use `127.0.0.1` (which will be *rejected* by `Strict`) — wrap such tests by constructing the source with `IpPolicy::AllowPrivate` if the source struct exposes that knob, or temporarily set `HEARTBIT_ALLOW_PRIVATE_IPS=1` inside the test. If neither is feasible, mark it `#[ignore]` with a comment explaining why.

- [ ] **Step 5.8: Migrate `tool/a2a.rs:433` (with URL validation).**

First read the surrounding ~50 lines to locate the peer URL variable and how the result is plumbed through:

```bash
sed -n '410,475p' crates/heartbit/src/tool/a2a.rs
```

In `crates/heartbit/src/tool/a2a.rs:433`, swap to `crate::http::safe_client_builder()`. Locate the line(s) that issue the request to the peer URL (likely `.post(&url).send()` or similar within the same fn). Insert validation:

```rust
            let safe = match crate::http::SafeUrl::parse(&peer_url, crate::http::IpPolicy::default()).await {
                Ok(u) => u,
                Err(e) => return Ok(ToolOutput::error(format!("peer URL refused: {e}"))),
            };
            // ... use safe.as_str() in the .get() / .post() call
```

Use the existing variable name for the URL (it may be `peer_url`, `target`, `endpoint`, etc.). The error return type matches what's already in the fn: if it returns `Result<ToolOutput, Error>`, use `return Err(Error::A2a(...))` instead of `Ok(ToolOutput::error(...))`. Match the existing pattern.

If the existing tests at lines 868, 883, 911, 940, 1189 use `reqwest::Client::new()` against fake `127.0.0.1:0` listeners, leave them alone for now (test-only pattern, not user-input). The cleanup of those `Client::new()` test patterns is **not** part of this task — see Out of Scope below.

Run `cargo test -p heartbit --lib tool::a2a:: 2>&1 | tail -10` — expected pass. If a tests in this module exercises the production `a2a.rs:433` site against a `127.0.0.1` mock, it will need the same opt-out treatment described in Step 5.7.

- [ ] **Step 5.9: Add the cross-reference comment to `tool/mcp.rs`.**

In `crates/heartbit/src/tool/mcp.rs`, find the two `reqwest::Client::builder()` sites at lines 655 and 1712. Above each, add or extend the existing comment to reference the new module:

At line 655 (token-exchange client), the existing comment is:
```rust
                // Disable redirects — the exchange_url is user-supplied; a redirect to
```
Append at the end of the comment block (before the `.redirect(...)` line):
```rust
                // (See also crate::http::safe_client_builder; this site predates that
                //  module and is intentionally inline pending consolidation.)
```

At line 1712 (MCP client), do the same: extend the existing comment block with:
```rust
            // (See also crate::http::safe_client_builder; this site predates that
            //  module and is intentionally inline pending consolidation.)
```

- [ ] **Step 5.10: Run the full lib test suite.**

```bash
cargo test -p heartbit --lib 2>&1 | tail -5
```
Expected: all tests pass, no regressions.

- [ ] **Step 5.11: Run fmt + clippy on the whole workspace.**

```bash
cargo fmt -- --check && cargo clippy -p heartbit --lib --all-targets -- -D warnings 2>&1 | tail -3
```
Expected: both succeed.

- [ ] **Step 5.12: Commit.**

```bash
git add crates/heartbit/src/sensor/sources/weather.rs \
        crates/heartbit/src/sensor/sources/jmap.rs \
        crates/heartbit/src/tool/builtins/websearch.rs \
        crates/heartbit/src/tool/builtins/image_generate.rs \
        crates/heartbit/src/tool/builtins/tts.rs \
        crates/heartbit/src/tool/builtins/twitter_post.rs \
        crates/heartbit/src/sensor/sources/rss.rs \
        crates/heartbit/src/tool/a2a.rs \
        crates/heartbit/src/tool/mcp.rs
git commit -m "refactor: route 8 reqwest sites through crate::http factories

vendor_client_builder for trusted-vendor APIs (websearch, image_generate,
tts, twitter_post, jmap, weather). safe_client_builder + SafeUrl::parse
for agent-controllable URLs (rss, a2a peer). All inherit the same
redirect(Policy::none()) policy. tool/mcp.rs left inline (pre-existing
hardening from 33d4015) with a cross-reference comment to the new
module."
```

---

## Task 6: Bearer constant-time fix in heartbit-cli

**Why now:** Tiny carry-over fix using the helper added in Task 2.

**Files:**
- Modify: `crates/heartbit-cli/src/daemon/auth.rs`

- [ ] **Step 6.1: Write the failing test.**

In `crates/heartbit-cli/src/daemon/auth.rs`, find the existing `#[cfg(test)] mod tests` block (or create it at the bottom of the file). Add:

```rust
    #[test]
    fn validate_bearer_rejects_equal_length_different_token() {
        use std::collections::HashSet;
        let mut tokens = HashSet::new();
        tokens.insert("aaaaaaaa".to_string());
        // Same length, different content — must reject.
        let res = validate_bearer_token(Some("Bearer bbbbbbbb"), &tokens);
        assert!(res.is_err());
    }

    #[test]
    fn validate_bearer_accepts_known_token() {
        use std::collections::HashSet;
        let mut tokens = HashSet::new();
        tokens.insert("hunter2".to_string());
        let res = validate_bearer_token(Some("Bearer hunter2"), &tokens);
        assert!(res.is_ok());
    }
```

(The second test exists in spirit; add it if missing. If it already exists, skip.)

- [ ] **Step 6.2: Run the tests; verify the new one passes against the current code.**

The new test should already pass — `tokens.contains` is correct *behaviorally*, just timing-variable. Run:

```bash
cargo test -p heartbit-cli --bin heartbit daemon::auth:: 2>&1 | tail -5
```
Expected: passes.

> Note: a true *constant-time* property test (timing) is too flaky to assert in CI. The behavioral assertions plus the use of `subtle::ConstantTimeEq` give us the property by construction.

- [ ] **Step 6.3: Swap the implementation to `heartbit::auth::ct::contains`.**

In `crates/heartbit-cli/src/daemon/auth.rs`, find:

```rust
            if token.is_empty() || !tokens.contains(token) {
```

Replace with:

```rust
            if token.is_empty() || !heartbit::auth::ct::contains(tokens, token) {
```

- [ ] **Step 6.4: Re-run the tests.**

```bash
cargo test -p heartbit-cli --bin heartbit daemon::auth:: 2>&1 | tail -5
```
Expected: passes.

- [ ] **Step 6.5: Run fmt + clippy.**

```bash
cargo fmt -- --check && cargo clippy -p heartbit-cli --all-targets -- -D warnings 2>&1 | tail -3
```

- [ ] **Step 6.6: Commit.**

```bash
git add crates/heartbit-cli/src/daemon/auth.rs
git commit -m "fix(cli): constant-time bearer-token comparison

Replace HashSet::contains (timing-variable on byte comparison) with
heartbit::auth::ct::contains, which uses subtle::ConstantTimeEq. Closes
the timing-side-channel exposure for daemon bearer tokens."
```

---

## Task 7: Clear all 26 clippy warnings

**Why now:** Required before CI can enforce `--all-targets -D warnings` in Task 8.

**Files:** several lib + the parked sensor E2E test file.

- [ ] **Step 7.1: Take a baseline of warnings.**

```bash
cargo clippy --workspace --all-targets 2>&1 | grep "^warning:" | wc -l
```
Expected: ~26.

- [ ] **Step 7.2: Run `clippy --fix` for the auto-fixable ones.**

```bash
cargo clippy --workspace --all-targets --fix --allow-dirty --allow-staged 2>&1 | tail -10
```
Most of the 14 collapsible-if, the 5 `iter().copied().collect()` → `to_vec()`, the field-reassign-with-default, the default-constructed-unit-struct, and the manual-str-repeat / repeat-take cases are auto-fixed.

Run again to confirm:

```bash
cargo clippy --workspace --all-targets 2>&1 | grep "^warning:" | wc -l
```
Expected: dropped to a small handful (the type-complexity warnings and the MutexGuard-across-await are not auto-fixable).

- [ ] **Step 7.3: Add `type` aliases for the type-complexity warnings.**

There are 2 type-complexity warnings. The location of the first one is `crates/heartbit/src/daemon/kafka.rs` (search for `fn mock_producer()`). Above that fn, add:

```rust
type MockProducerHandle = (
    Arc<dyn CommandProducer>,
    tokio::sync::mpsc::UnboundedReceiver<(String, Vec<u8>)>,
);
```

Then change the fn signature from:

```rust
fn mock_producer() -> (
    Arc<dyn CommandProducer>,
    tokio::sync::mpsc::UnboundedReceiver<(String, Vec<u8>)>,
) {
```

to:

```rust
fn mock_producer() -> MockProducerHandle {
```

For the second type-complexity site, run:

```bash
cargo clippy --workspace --all-targets 2>&1 | grep -B1 "very complex type" | head -20
```

Apply the same `type` alias technique at that site. The exact alias depends on the type — extract the inner tuple and give it a descriptive name.

- [ ] **Step 7.4: Fix the MutexGuard-across-await in `tests/sensor_pipeline_e2e.rs:5058`.**

Read the surrounding ~30 lines:

```bash
sed -n '5040,5095p' crates/heartbit/tests/sensor_pipeline_e2e.rs
```

Identify the `let guard = some_mutex.lock().unwrap();` (or `.lock().await` for tokio Mutex — but std::sync::Mutex is the one that triggers the warning) and the `.await` on a later line (around 5085). The fix pattern:

```rust
// BEFORE — guard is held across the await
let guard = state.lock().unwrap();
let value = guard.compute();
some_async_thing(value).await;
drop(guard);

// AFTER — drop or scope the guard before await
let value = {
    let guard = state.lock().unwrap();
    guard.compute()
}; // guard dropped at end of scope
some_async_thing(value).await;
```

Apply the equivalent transformation at the actual site. If the value is more complex, extract via a `clone()` or a small helper inside the lock scope.

- [ ] **Step 7.5: Verify zero warnings.**

```bash
cargo clippy --workspace --all-targets -- -D warnings 2>&1 | tail -5
```
Expected: `Finished` line, exit 0, no warning lines.

- [ ] **Step 7.6: Run the test suite.**

```bash
cargo test --workspace --no-run 2>&1 | tail -3
```
Expected: success.

```bash
cargo test --workspace --lib 2>&1 | tail -3
```
Expected: all lib tests pass.

- [ ] **Step 7.7: Run fmt.**

```bash
cargo fmt -- --check
```
Expected: success.

- [ ] **Step 7.8: Commit.**

```bash
git add -u
git commit -m "chore(clippy): clear all --workspace --all-targets warnings

Auto-fix from cargo clippy --fix for collapsible-if, iter.copied.collect,
default-constructed-unit-struct, field-reassign-with-default, and
manual-str-repeat. Manual changes: type aliases for two
very-complex-type warnings; MutexGuard-across-await refactor in
tests/sensor_pipeline_e2e.rs (drop guard before .await — was a real
correctness issue even though the test is #[ignore]d).

Unblocks the cargo clippy --workspace --all-targets -- -D warnings gate."
```

---

## Task 8: Tighten CI

**Why last:** Now that fmt, clippy, test-compile, and lib-test all pass, the CI gate can enforce the full set without immediately turning red.

**Files:**
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 8.1: Replace the existing CI body.**

Edit `.github/workflows/ci.yml`. Replace the four `name: cargo …` steps (currently `cargo fmt`, `cargo clippy`, `cargo test`) with:

```yaml
      - name: cargo fmt
        run: cargo fmt -- --check

      - name: cargo clippy
        run: cargo clippy --workspace --all-targets -- -D warnings

      - name: cargo test (compile only)
        run: cargo test --workspace --no-run

      - name: cargo test (lib)
        run: cargo test --workspace --lib
```

The first three are gates that must always pass. The fourth runs the lib unit tests (~3308 of them) but skips integration tests that may need infrastructure (Kafka, Postgres, live LLM keys).

- [ ] **Step 8.2: Validate the YAML.**

```bash
python3 -c "import yaml,sys; yaml.safe_load(open('.github/workflows/ci.yml'))" && echo OK
```
Expected: `OK`.

- [ ] **Step 8.3: Run the gate locally end-to-end.**

```bash
cargo fmt -- --check && \
cargo clippy --workspace --all-targets -- -D warnings && \
cargo test --workspace --no-run && \
cargo test --workspace --lib 2>&1 | tail -3
```
Expected: each step succeeds; final `test result: ok. N passed; 0 failed`.

- [ ] **Step 8.4: Commit.**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: enforce clippy --all-targets + explicit test --no-run + --lib

CI now fails fast if:
- formatting drifts
- any clippy warning fires across the entire workspace including tests
- workspace tests fail to compile (the regression that produced this
  patch series)
- lib unit tests regress

Integration tests stay out of CI for now — many need infra (Kafka,
Postgres, live LLM keys) that the runner does not have. The --no-run
step proves they at least compile."
```

---

## Self-Review

Run after Task 8 completes.

- [ ] **Step S.1: Spec coverage.**

Verify each spec section maps to a task above:

- Test gate restoration → Task 1 ✓
- `crate::auth::ct` + `subtle` promotion → Task 2 ✓
- `crate::http` (IpPolicy, SafeUrl, two factories) → Task 3 ✓
- WebFetchTool migration + SSRF tests + CHANGELOG → Task 4 ✓
- 8 reqwest site migrations → Task 5 ✓
- Bearer constant-time fix → Task 6 ✓
- 26 clippy warnings cleared → Task 7 ✓
- CI gate tightened → Task 8 ✓

- [ ] **Step S.2: Run the full exit-criteria battery.**

```bash
cargo fmt -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --no-run
cargo test --workspace --lib
```
All four must exit 0. If any fails, fix and re-commit before declaring done.

- [ ] **Step S.3: Verify CHANGELOG entry exists.**

```bash
grep -A5 "^## Unreleased" CHANGELOG.md | head -10
```
Expected: a "Security" subsection mentioning `WebFetchTool` default-deny.

- [ ] **Step S.4: Verify public API surface.**

```bash
grep -nE "pub fn|pub mod|pub struct|pub enum" crates/heartbit/src/http.rs crates/heartbit/src/auth/ct.rs
```
Expected items:
- `pub mod http` (in lib.rs) ← already verified by clippy passing
- `pub mod ct` (in auth/mod.rs)
- `pub enum IpPolicy { Strict, AllowPrivate }`
- `pub struct SafeUrl`
- `pub fn safe_client_builder`
- `pub fn vendor_client_builder`
- `pub fn ct_eq_str`
- `pub fn contains`
- `pub fn WebFetchTool::with_ip_policy`

If any is missing, the corresponding task is incomplete.

---

## Out of Scope (per spec)

These are NOT part of this plan and should NOT be addressed in this round:

- DNS-rebind defense (parse-time only).
- Triage / un-ignoring of the 78 `#[ignore]`d sensor-E2E tests.
- `heartbit-core` sub-crate extraction.
- Rotation of credentials in the on-disk `.env`.
- Documentation reorganization (CLAUDE.md / AGENTS.md relocation, getting-started docs).
- Cleanup of the 5 `reqwest::Client::new()` test patterns in `tool/a2a.rs:868–1189`. They're test-only and not user-controllable; leave them alone.

If any of these are tempting during execution: stop, note it as a follow-up, and proceed with the plan as written.
