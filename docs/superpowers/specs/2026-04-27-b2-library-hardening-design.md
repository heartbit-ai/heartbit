# B2 — Library Hardening: SSRF Defense, Constant-Time Auth, Restored Gate

**Date:** 2026-04-27
**Status:** Design — pending user approval before implementation plan
**Scope:** `crates/heartbit` (primary), `crates/heartbit-cli` (carry), `.github/workflows/ci.yml`
**Estimated effort:** ~5 engineer-hours, eight independently mergeable steps

## Background

A full project review (2026-04-27) surfaced four classes of issue ranked HIGH:

1. **Test gate broken on `main`.** `cargo test --workspace --no-run` fails to compile because commit `a81f84c` (2026-03-21) added two required fields to `RuntimeRequest` (`twitter_credentials`) and `RuntimeProviderConfig` (`base_url`) without updating the `make_test_request()` helpers in `heartbit-cli/src/daemon/{eval,execute}.rs`. Five struct literals in two files are stale.
2. **`WebFetchTool` ships without SSRF defense.** `crates/heartbit/src/tool/builtins/webfetch.rs` validates only the URL scheme. An agent (or attacker controlling agent input) can fetch `http://169.254.169.254/` (cloud IMDS), loopback, or RFC1918 hosts. The same tool also follows redirects by default, so a public host can 302 to a private one.
3. **Bearer-token comparison is timing-variable.** `heartbit-cli/src/daemon/auth.rs:62` does `tokens.contains(token)` on a `HashSet<String>`, which is hash-based but not constant-time at byte comparison.
4. **Reqwest client construction is duplicated across 10 files.** `tool/mcp.rs` was hardened inline by commit `33d4015` (two construction sites in that one file: MCP itself and token-exchange); the other nine files repeat the same construction without redirect policy. The pattern that produced the original SSRF bypass is still present.

Items #2, #3, #4 share a single root cause: the library has no centralized HTTP-client factory or URL-validation primitive. This spec adds both, migrates the nine outstanding files, fixes #1, and tightens CI so the gate stops rotting.

A separate audit item — DNS-rebind defense — was explicitly de-scoped from this round (see *Out-of-Scope*).

## Goals

1. Restore `cargo test --workspace --no-run` and keep it green via CI enforcement.
2. Make `WebFetchTool` and other agent-input HTTP call sites safe-by-default against parse-time SSRF (private/loopback/link-local IPs and HTTP redirects).
3. Provide a constant-time bearer-token comparison primitive in the library, used by the daemon CLI.
4. Centralize HTTP client construction in `heartbit::http` so future call sites inherit the policy.
5. Clear all 26 `cargo clippy --workspace --all-targets` warnings (including one real `MutexGuard held across .await` correctness bug in `tests/sensor_pipeline_e2e.rs`).
6. Tighten the CI workflow so future regressions are caught at PR time, not six days later.

## Non-Goals

- DNS-rebind defense. The IP blocklist is parse-time only. An attacker who controls a DNS name can return a public IP at parse time and a private IP at connect time and bypass this design. Documented limitation; deferred.
- Triage of the 78 `#[ignore]`d tests in `tests/sensor_pipeline_e2e.rs`. They will be made clippy-clean as a side-effect but their `#[ignore]` status is unchanged.
- Extraction of `heartbit-core` sub-crate. Reviewed item 8; deferred.
- Rotation of credentials in the on-disk `.env` and migration to the vault module. Operator-side action.
- Documentation reorganization (CLAUDE.md / AGENTS.md relocation, CHANGELOG refresh, getting-started docs). Separate round.

## Design

### Architecture

Three concentric rings:

```
┌─────────────────────────────────────────────────────────┐
│  CI gate                                                 │
│    cargo fmt -- --check                                  │
│    cargo clippy --workspace --all-targets -- -D warnings │
│    cargo test --workspace --no-run                       │
│    cargo test --workspace --lib                          │
│   ┌──────────────────────────────────────────────────┐  │
│   │ heartbit::http (NEW, public)                      │  │
│   │   IpPolicy { Strict | AllowPrivate }              │  │
│   │   SafeUrl                                          │  │
│   │   safe_client_builder() / vendor_client_builder() │  │
│   │ heartbit::auth::ct (NEW, public)                  │  │
│   │   ct_eq_str() / contains()                        │  │
│   └──────────────────────────────────────────────────┘  │
│                       ▲                                  │
│   ┌───────────────────┴──────────────────────────────┐  │
│   │ 9 reqwest call sites migrated                     │  │
│   │ + heartbit-cli bearer fix                          │  │
│   │ + 5 test-helper struct literals                    │  │
│   │ + 26 clippy warnings cleared                       │  │
│   └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

**Two new public modules** in `heartbit`. **One promoted dep** (`subtle` from optional/`sensor`-gated → unconditional). **Zero new deps.** **No schema or wire-format changes.** **No breaking changes to existing types.** The only observable behavior change for users: agents requesting private/loopback/link-local IPs from `WebFetchTool` get a `ToolOutput::error(...)` instead of a successful fetch.

### Component: `heartbit::http`

New file `crates/heartbit/src/http.rs`, ~180 LOC + tests.

```rust
/// Whether to permit requests to non-routable / private IPs.
pub enum IpPolicy {
    /// Reject loopback, link-local, RFC1918, CGNAT, ULA, multicast,
    /// unspecified, broadcast. Library default.
    Strict,
    /// Allow any IP. Use only for single-tenant deployments where agents
    /// legitimately need internal-network access.
    AllowPrivate,
}
impl Default for IpPolicy { fn default() -> Self { Self::from_env() } }
impl IpPolicy {
    /// Reads `HEARTBIT_ALLOW_PRIVATE_IPS`. Anything other than "1" / "true"
    /// (case-insensitive) yields `Strict`. Unset → `Strict`.
    pub fn from_env() -> Self;
}

/// A URL that has passed scheme + IP-blocklist validation.
/// Construction is the only way to satisfy the type — call sites that hold
/// a `SafeUrl` proved they validated it.
pub struct SafeUrl(reqwest::Url);

impl SafeUrl {
    /// Parse `s`, require an http:// or https:// scheme, and reject if the
    /// host is a literal blocked IP, or if any IP that the host name
    /// resolves to is in the blocked set under `policy`.
    ///
    /// Resolution uses `tokio::net::lookup_host` with the URL's port (or 80/443).
    /// All resolved addresses are checked; if *any* resolved address is private,
    /// the URL is rejected. This is a parse-time check and does not protect
    /// against DNS rebind; see crate-level rustdoc.
    pub async fn parse(s: &str, policy: IpPolicy) -> Result<Self, Error>;

    pub fn as_str(&self) -> &str;
    pub fn into_url(self) -> reqwest::Url;
}

/// reqwest::ClientBuilder with `redirect(Policy::none())` baked in.
/// Use for clients that send to user-controllable URLs. Caller still wraps
/// the URL with `SafeUrl::parse(...)` before sending.
pub fn safe_client_builder() -> reqwest::ClientBuilder;

/// reqwest::ClientBuilder with `redirect(Policy::none())` baked in.
/// Use for clients that send to operator-trusted vendor APIs. No IP validation
/// is implied — caller asserts the host is safe.
pub fn vendor_client_builder() -> reqwest::ClientBuilder;
```

**Blocklist** applied to literal IP host *and* every IP returned by `tokio::net::lookup_host`:

| Range | What |
|---|---|
| `127.0.0.0/8`, `::1` | loopback |
| `169.254.0.0/16`, `fe80::/10` | link-local (incl. AWS/GCE IMDS `169.254.169.254`) |
| `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16` | RFC1918 |
| `100.64.0.0/10` | CGNAT |
| `fc00::/7` | ULA |
| `224.0.0.0/4`, `ff00::/8` | multicast |
| `0.0.0.0/8`, `::` | unspecified |
| `255.255.255.255` | broadcast |

Implementation: pattern-match on `std::net::IpAddr` using methods already in `std::net::Ipv4Addr` / `std::net::Ipv6Addr` where possible (`is_loopback`, `is_link_local`, `is_private`, `is_multicast`, `is_unspecified`, `is_broadcast`); add explicit checks for CGNAT (`100.64.0.0/10`) and ULA (`fc00::/7`) which `std` does not expose stably as of MSRV.

Rejection error: `Error::Agent(format!("URL host {host} resolves to a private/loopback address; refused (set HEARTBIT_ALLOW_PRIVATE_IPS=1 to override)"))` — reuses the existing `Agent` variant, no new public error variant added.

### Component: `heartbit::auth::ct`

New file `crates/heartbit/src/auth/ct.rs`, ~40 LOC + tests. Re-exported from `crate::auth::mod.rs`.

```rust
use std::collections::HashSet;
use subtle::ConstantTimeEq;

/// Constant-time string equality. Returns false if lengths differ.
/// The length difference is intentionally not constant-time: token length is
/// not the secret. Comparing equal-length non-equal tokens is constant-time.
pub fn ct_eq_str(a: &str, b: &str) -> bool {
    a.len() == b.len() && a.as_bytes().ct_eq(b.as_bytes()).into()
}

/// Constant-time membership test against a HashSet<String>. Iterates the
/// entire set on every call; O(n) by design. Use only for sets small enough
/// that the linear scan is acceptable (bearer-token allow-lists, signing
/// keys, etc.).
pub fn contains(set: &HashSet<String>, candidate: &str) -> bool {
    let mut hit = false;
    for known in set {
        // bitor so every iteration runs regardless of earlier match
        hit |= ct_eq_str(known, candidate);
    }
    hit
}
```

### Component: `WebFetchTool` changes

Add a single field `ip_policy: IpPolicy`. Two constructors:

```rust
impl WebFetchTool {
    /// Constructs with `IpPolicy::default()` — Strict unless
    /// `HEARTBIT_ALLOW_PRIVATE_IPS=1` is set in the environment.
    pub fn new() -> Self;

    pub fn with_ip_policy(policy: IpPolicy) -> Self;
}
```

No builder type. One knob does not justify the surface area. The existing public API of `WebFetchTool::new()` remains backwards-compatible.

In `execute(...)`:

1. After the existing scheme check, call `let safe = SafeUrl::parse(url, self.ip_policy).await?` — but wrap the `Err` in `Ok(ToolOutput::error(e.to_string()))` to keep the agent loop alive (matches the pattern already in webfetch for file://, ftp://, oversize body).
2. Pass `safe.as_str()` to `self.client.get(...)`.

The client itself is built via `crate::http::safe_client_builder()` instead of `reqwest::Client::builder()`.

### Component: heartbit-cli bearer fix

In `crates/heartbit-cli/src/daemon/auth.rs:62`, replace:

```rust
if token.is_empty() || !tokens.contains(token) {
```

with:

```rust
if token.is_empty() || !heartbit::auth::ct::contains(tokens, token) {
```

Empty-token rejection stays first; the length-zero short-circuit is fine because empty is not a secret. No other changes to error responses or status codes.

### Component: 5 test-helper struct literals

Add explicit `twitter_credentials: None,` and `base_url: None,` to:

- `crates/heartbit-cli/src/daemon/eval.rs:244` — `RuntimeRequest`
- `crates/heartbit-cli/src/daemon/eval.rs:258` — `RuntimeProviderConfig`
- `crates/heartbit-cli/src/daemon/execute.rs:1120` — `RuntimeRequest`
- `crates/heartbit-cli/src/daemon/execute.rs:1134` — `RuntimeProviderConfig`
- `crates/heartbit-cli/src/daemon/execute.rs:1483` — `RuntimeProviderConfig`

No `Default` derive added. Matches the explicit-field style already used by the round-trip tests in `crates/heartbit/src/daemon/runtime_types.rs`.

### Component: 26 clippy warnings

Categorize as found:

| Category | Count | Sites | Fix |
|---|---|---|---|
| collapsible-if | 14 | mostly tests, a few in `daemon/kafka.rs:603`, `lsp/client.rs:411`, etc. | combine `if A { if B { ... } }` → `if A && B { ... }` (or `if let .. && let ..`) |
| `iter().copied().collect()` → `to_vec()` | 5 | `agent/tool_filter.rs:273,299,325,340,351` | mechanical rewrite |
| type-complexity | 2 | `daemon/kafka.rs` mock_producer, one workflow types tuple | introduce `type` aliases |
| field-reassign-with-default | 1 | `daemon/types.rs:540-541` | use struct-update syntax |
| default-constructed-unit-struct | 1 | `sensor/triage/structured.rs:172` | drop `::default()` |
| manual-str-repeat | 1 | tests | use `"x".repeat(n)` |
| repeat-take | 1 | tests | use `repeat_n` or shorter form |
| MutexGuard-across-await | **1** | `tests/sensor_pipeline_e2e.rs:5058–5085` | **real bug — drop guard before `.await`** |

Approach: run `cargo clippy --fix --workspace --all-targets --allow-dirty --allow-staged` first, manually verify each diff, then hand-fix the remaining ones (especially the MutexGuard refactor).

### Component: CI workflow

Modify `.github/workflows/ci.yml` `cargo clippy` and `cargo test` steps to:

```yaml
      - name: cargo fmt
        run: cargo fmt -- --check

      - name: cargo clippy
        run: cargo clippy --workspace --all-targets -- -D warnings

      - name: cargo test --no-run
        run: cargo test --workspace --no-run

      - name: cargo test --lib
        run: cargo test --workspace --lib
```

Rationale for `--lib` instead of full `--workspace`:
- 78 `#[ignore]`d sensor-E2E tests are already skipped by default; not the issue.
- Integration tests in the workspace may rely on infrastructure (Postgres, Kafka, live LLM keys) that the CI runner does not have. We don't want the gate to fail for environment reasons.
- `--no-run` already proves the integration tests *compile*. That's the regression we actually had.
- Lib tests run completely in-process and are the fast, reliable signal.

Future round can add `cargo test --workspace --bins --tests` once integration tests are infrastructure-independent.

## Per-Site Migration Table

| # | File | URL source | Builder | Wraps in `SafeUrl`? |
|---|---|---|---|---|
| 1 | `tool/builtins/webfetch.rs:22` | agent input | `safe_client_builder` | yes |
| 2 | `tool/a2a.rs:433` | peer agent (agent input) | `safe_client_builder` | yes |
| 3 | `tool/a2a.rs:868,883,911,940,1189` | tests, `Client::new()` | `vendor_client_builder` | n/a (tests) |
| 4 | `sensor/sources/rss.rs:49` | operator-config feed URL | `safe_client_builder` | yes — operator-config still passes through agent surface area |
| 5 | `sensor/sources/jmap.rs:214` | operator-config mail server | `vendor_client_builder` | no — mail server is operator-trusted |
| 6 | `sensor/sources/weather.rs:163` | hardcoded weather API | `vendor_client_builder` | no |
| 7 | `tool/builtins/websearch.rs:34,47` | hardcoded SerpAPI / DDG | `vendor_client_builder` | no |
| 8 | `tool/builtins/image_generate.rs:20` | hardcoded image API | `vendor_client_builder` | no |
| 9 | `tool/builtins/tts.rs:30` | hardcoded TTS vendor | `vendor_client_builder` | no |
| 10 | `tool/builtins/twitter_post.rs:53` | hardcoded api.twitter.com | `vendor_client_builder` | no |
| — | `tool/mcp.rs:655,1712` | already hardened (commit `33d4015`) | leave inline | leave inline. Add comment cross-referencing `crate::http`. Optional follow-up to migrate. |

## Error Handling

- `SafeUrl::parse` returns `Result<SafeUrl, Error>` using the existing `Error::Agent(String)` variant. No new public error variant.
- `WebFetchTool` and `A2A` callers wrap rejection in `Ok(ToolOutput::error(...))` so the agent loop continues. Matches the existing webfetch pattern for file://, ftp://, and oversize body errors.
- Sensor sources (rss, jmap, weather) propagate `Result` to the sensor manager, which already logs and continues. No new handling.
- Bearer-token failure path is unchanged. Only the comparison primitive changes.

## Testing

| What | Location | New tests |
|---|---|---|
| `IpPolicy::from_env` | `http` unit | env unset → Strict; env=`1` → AllowPrivate; env=`true` → AllowPrivate; env=`0` → Strict; env=garbage → Strict |
| `SafeUrl::parse` blocklist | `http` unit | one test per blocked range × IPv4 + key IPv6 cases; one positive test (`https://8.8.8.8`); one DNS test (`http://localhost` → reject); one rejection-message smoke test |
| `safe_client_builder` redirect | `http` unit | hand-rolled `tokio::net::TcpListener` returns 302 → assert `reqwest::Error::is_redirect()` is false but the response is the 302 itself (i.e., not followed) |
| `ct_eq_str` / `ct::contains` | `auth::ct` unit | equal / not-equal-same-length / different-length / `contains` hit / `contains` miss / empty-set rejection |
| `WebFetchTool` SSRF | `tool/builtins/webfetch` unit | `webfetch_rejects_loopback`; `webfetch_rejects_imds`; `webfetch_rejects_rfc1918`; `webfetch_rejects_localhost_dns`; `webfetch_with_allow_private_ips_accepts_loopback`; existing scheme tests retained |
| Bearer ct | `heartbit-cli/src/daemon/auth.rs` unit | retain existing tests; add `validate_bearer_rejects_equal_length_different_token` |
| Test gate restored | meta | CI assertion: `cargo test --workspace --no-run` exits 0 |

No new fixtures or fakes beyond the in-process `tokio::net::TcpListener` for the redirect test (~30 LOC). The codebase has no existing HTTP mocking dep (no `wiremock`, `httpmock`, or `mockito`); hand-rolled is consistent with the existing style.

## Sequencing

Each step ships as an independent commit; a single PR or 2–3 small ones, at the implementer's discretion.

| # | Step | Effort | Net change |
|---|---|---|---|
| 1 | Patch the 5 struct literals | ~10 min | `cargo test --workspace --no-run` is green |
| 2 | Add `crate::auth::ct` + tests; promote `subtle` to non-optional | ~30 min | New public module; no consumers yet |
| 3 | Add `crate::http` + tests | ~90 min | New public module; no consumers yet |
| 4 | Migrate `WebFetchTool`; add SSRF tests | ~30 min | First real consumer of `crate::http`. The default-deny behavior change for private IPs is significant — implementer adds a single CHANGELOG entry for it (this is a normal commit-hygiene entry, not the broader CHANGELOG refresh that Non-Goals defers). |
| 5 | Migrate the other 8 reqwest sites | ~60 min | Mechanical |
| 6 | Bearer fix in heartbit-cli | ~10 min | One call swap; one new test |
| 7 | Clear 26 clippy warnings | ~90 min | `clippy --fix` plus manual MutexGuard refactor |
| 8 | Tighten CI workflow | ~15 min | `.github/workflows/ci.yml` |

Total: ~5 engineer-hours.

## Risks

- **DNS-rebind unaddressed.** Documented in `crate::http` rustdoc with a pointer to the future hardening (B3 round). If a high-value tenant requires this, it can be backfilled by adding a custom `reqwest::dns::Resolve` implementation that filters at connect time.
- **Sensor sources to `safe_client_builder`.** RSS feeds may be operator-config but are treated as agent-controllable for blast-radius reasons. Operators with legitimate need for internal feeds use `HEARTBIT_ALLOW_PRIVATE_IPS=1`. Documented.
- **`subtle` becoming a non-optional dep.** Minor SemVer impact; we are pre-1.0 (versioned `2026.x.x`). Not a real concern.
- **Auto-fix on `tests/sensor_pipeline_e2e.rs`.** That file has 78 `#[ignore]`d tests; auto-fix touches parked code. Acceptable: we are not changing logic, only making it lint-clean. The MutexGuard refactor is a real correctness fix and is worth doing even in parked tests.

## Out-of-Scope

- DNS-rebind defense (deferred to a future B3).
- Triage of the 78 `#[ignore]`d sensor-E2E tests.
- `heartbit-core` sub-crate extraction.
- `.env` rotation and vault adoption (operator-side action).
- Documentation reorganization (CLAUDE.md / AGENTS.md relocation, CHANGELOG refresh, getting-started docs).

## Exit Criteria

1. `cargo fmt -- --check` exits 0.
2. `cargo clippy --workspace --all-targets -- -D warnings` exits 0.
3. `cargo test --workspace --no-run` exits 0.
4. `cargo test --workspace --lib` exits 0.
5. `.github/workflows/ci.yml` runs all four steps.
6. New unit tests in `crate::http`, `crate::auth::ct`, and `WebFetchTool` SSRF section all pass.
7. The nine migrated reqwest call sites use `heartbit::http::*` factories; only `tool/mcp.rs` retains its inline construction (with cross-reference comment).

## Public API Additions Summary

- `pub mod heartbit::http`
  - `pub enum IpPolicy { Strict, AllowPrivate }`
  - `pub struct SafeUrl(reqwest::Url)`
  - `pub fn safe_client_builder() -> reqwest::ClientBuilder`
  - `pub fn vendor_client_builder() -> reqwest::ClientBuilder`
- `pub mod heartbit::auth::ct`
  - `pub fn ct_eq_str(a: &str, b: &str) -> bool`
  - `pub fn contains(set: &HashSet<String>, candidate: &str) -> bool`
- `WebFetchTool::with_ip_policy(IpPolicy) -> Self`

No breaking changes. No removals.
