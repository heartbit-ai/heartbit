# `heartbit-ghost` P1.1 — X Tool Family Design

**Date:** 2026-05-07
**Status:** design (awaiting review)
**Depends on:** P1.0 (heartbit-ghost crate scaffolding) — already merged to `main` as of `5ef2284`.
**Implements:** [`2026-05-07-heartbit-ghost-x-agent-design.md`](2026-05-07-heartbit-ghost-x-agent-design.md) §4 (X tool family catalog) + §9 P1.1 sub-phase.

## Mission

Ship the X (Twitter) tool family for `heartbit-ghost`. Five new tools (`twitter_thread`, `twitter_reply`, `twitter_search`, `twitter_mentions`, `twitter_user`) plus a media + alt-text extension to the existing `heartbit-core::TwitterPostTool`. Tools are usable standalone via the existing builtin pattern (`AgentRunnerBuilder::tools(...)`); they are NOT yet wired into the X persona's `expand()` — that's P1.3.

The new tools resolve per-tenant credentials at execute-time via `&ExecutionContext::credentials::CredentialResolver` (the Phase 0 surface). They share an `XClient` infrastructure for HTTP, OAuth1 signing, and error mapping. Tests use `wiremock` for HTTP stubbing.

## Architecture decisions

### AD-1 · Credential resolution: resolver-only for new tools

Each new tool resolves credentials at execute-time:

```rust
let resolver = ctx.credentials.as_ref()
    .ok_or_else(|| Error::Agent("no credential resolver in execution context".into()))?;
let consumer_key    = resolver.resolve("X_CONSUMER_KEY").await?;
let consumer_secret = resolver.resolve("X_CONSUMER_SECRET").await?;
let access_token    = resolver.resolve("X_ACCESS_TOKEN").await?;
let access_secret   = resolver.resolve("X_ACCESS_TOKEN_SECRET").await?;
```

The 4 credential names are stable and documented. The resolver returns `heartbit_core::Secret` per credential.

The existing `heartbit_core::TwitterPostTool` is **not** migrated to this pattern in P1.1. It keeps its construction-time `TwitterCredentials` parameter for backward compatibility with `BuiltinToolsConfig::twitter_credentials` users. The persona's wiring of `twitter_post` (in P1.3) can switch to a resolver-based instance at that time.

**Rationale.** Per-tenant deployments require execute-time resolution (separate tool instances per tenant is unscalable). Migrating the existing tool's constructor in this phase would break downstream callers and inflate scope. Two patterns coexist; consolidation is post-merge cleanup.

### AD-2 · Tool placement: heartbit-ghost owns the new tools

```
crates/heartbit-ghost/src/
  lib.rs                  (existing — XGhostPersona stub + register, unchanged in P1.1)
  tools/
    mod.rs                (module re-exports)
    client.rs             (XClient: HTTP + OAuth1 + credential resolution + error mapping)
    thread.rs             (TwitterThreadTool)
    reply.rs              (TwitterReplyTool)
    search.rs             (TwitterSearchTool)
    mentions.rs           (TwitterMentionsTool)
    user.rs               (TwitterUserTool)
```

The existing `twitter_post` stays at `crates/heartbit-core/src/tool/builtins/twitter_post.rs`. The media + alt-text extension is an in-place edit of that file.

**Rationale.** New tools are X-persona-scoped; placing them in heartbit-ghost keeps heartbit-core's builtin set generic. The existing twitter_post is a generic builtin (anyone can use it without depending on heartbit-ghost) and stays where it is.

### AD-3 · HTTP mocking: `wiremock = "0.6"`

Added as `[dev-dependencies] wiremock = "0.6"` to `crates/heartbit-ghost/Cargo.toml`. All HTTP-touching tests use wiremock to stub X API responses (200/401/429/4xx/5xx) without hitting the live network.

**Rationale.** Async-first, expressive request matching, idiomatic for modern async Rust HTTP testing. Worth the dep weight for ~30–40 tests.

### AD-4 · Rate limiting: none in P1.1

No rate limiting at the tool layer. Each tool surfaces 429 as `ToolOutput::error("rate limited; retry after Ns")` parsed from the `Retry-After` header. The persona-level token bucket (12 posts/day, 1/hour burst per the heartbit-ghost spec §8.1) lives at the orchestration layer in P1.4.

**Rationale.** YAGNI for tool-layer rate limiting. The right place for global per-account budget is the persona orchestrator, not the leaf tool.

### AD-5 · OAuth1 signing: `oauth1-request` crate

Added as `[dependencies] oauth1-request = "0.6"` to `crates/heartbit-ghost/Cargo.toml`. The `XClient` uses it to sign every X API request.

The existing `heartbit_core::TwitterPostTool` retains its inline OAuth1 implementation. We do **not** extract it to a shared utility — that would touch twitter_post and is out of AD-1 scope. Mild duplication is accepted; consolidation is post-merge cleanup.

**Rationale.** A small, well-tested dep for a security-sensitive cryptographic operation beats hand-rolling. ~50KB compressed; acceptable footprint for a persona crate that will grow.

## 1. Tool catalog

All tools take `&ExecutionContext` (Phase 0 trait) and resolve credentials per AD-1. Output is JSON-encoded `ToolOutput::success(...)` on the happy path; well-formatted `ToolOutput::error(...)` on auth/rate/4xx/5xx failure.

Tool names are stable. Schemas use JSON Schema (the framework's existing convention).

| Tool | X API endpoint | Input schema | Output (`success` content) |
|---|---|---|---|
| `twitter_post` (extended) | `POST /2/tweets` (+`POST /1.1/media/upload.json` when media is present) | `{text: string (≤280), media_url?: string (URL), media_alt_text?: string (≤1000)}` | `{tweet_id: string, url: string}` |
| `twitter_thread` | `POST /2/tweets` × N (each linked via `in_reply_to_tweet_id`) | `{tweets: [string] (1..=25 entries, each ≤280)}` | `{thread_root_id: string, tweet_ids: [string], urls: [string]}` |
| `twitter_reply` | `POST /2/tweets` (with `reply.in_reply_to_tweet_id`) | `{text: string (≤280), in_reply_to: string (tweet id)}` | `{tweet_id: string, url: string}` |
| `twitter_search` | `GET /2/tweets/search/recent` | `{query: string (≤512), max_results?: int (10..=100, default 10), since_id?: string}` | `{tweets: [{id, text, author_id, created_at}], next_token?: string}` |
| `twitter_mentions` | `GET /2/users/:id/mentions` | `{user_id: string, max_results?: int (5..=100, default 10), since_id?: string}` | `{mentions: [{id, text, author_id, created_at, in_reply_to_user_id?}], next_token?: string}` |
| `twitter_user` | `GET /2/users/by/username/:handle` | `{handle: string (no leading `@`)}` | `{id, name, handle, description, follower_count, following_count, tweet_count, created_at}` |

**Out of scope for P1.1** (per spec §4 — deferred to P1.4): `twitter_dm`, `twitter_schedule`, `twitter_metrics`.

### 1.1 `twitter_post` extension (heartbit-core in-place edit)

The existing `TwitterPostTool` schema gains two optional fields:

```json
{
  "text": "string, ≤280 chars (required)",
  "media_url": "string, public URL of one image (optional)",
  "media_alt_text": "string, ≤1000 chars accessibility description (optional)"
}
```

Behaviour:
- If `media_url` is absent → existing path (post text-only via `POST /2/tweets`)
- If `media_url` is present:
  1. Fetch the image from the URL (HTTP GET; ≤5 MB; reject if HTTPS-only verification fails)
  2. Upload bytes via `POST /1.1/media/upload.json` (multipart form-data) → receive `media_id_string`
  3. If `media_alt_text` is set, attach it via `POST /1.1/media/metadata/create.json`
  4. Post the tweet via `POST /2/tweets` with `media.media_ids = [media_id_string]`

The OAuth1 signing logic for the media-upload path stays inline (it's an extension of the existing inline OAuth1 logic in `twitter_post.rs`). No shared XClient is consumed here — twitter_post stays self-contained per AD-1 / AD-5.

**Backward compatibility:** existing callers passing `{text}` only see no behaviour change. The new fields are `Option<String>` with `#[serde(default)]`.

## 2. Shared infrastructure: `tools/client.rs`

```rust
pub struct XClient {
    http: reqwest::Client,
    consumer_key: Secret,
    consumer_secret: Secret,
    access_token: Secret,
    access_token_secret: Secret,
}

impl XClient {
    /// Construct from an ExecutionContext, resolving the 4 OAuth1 credentials.
    pub async fn from_context(ctx: &ExecutionContext) -> Result<Self, Error>;

    /// GET with OAuth1 signing + JSON parsing. Query params injected into URL.
    pub async fn get_json<T: DeserializeOwned>(
        &self,
        path: &str,
        query: &[(&str, &str)],
    ) -> Result<T, XApiError>;

    /// POST with OAuth1 signing + JSON body + JSON response parsing.
    pub async fn post_json<B: Serialize, T: DeserializeOwned>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<T, XApiError>;
}

/// Strongly typed X API error categories. Each variant carries diagnostics.
#[derive(Debug, thiserror::Error)]
pub enum XApiError {
    #[error("missing credential resolver in ExecutionContext")]
    MissingResolver,
    #[error("credential resolution failed for '{name}': {source}")]
    CredentialResolutionFailed { name: String, source: heartbit_core::Error },
    #[error("X auth failed (401): {0}")]
    Unauthenticated(String),
    #[error("X rate limited; retry after {retry_after_secs}s")]
    RateLimited { retry_after_secs: u64 },
    #[error("X API error ({status}): {message}")]
    ApiError { status: u16, message: String },
    #[error("network error: {0}")]
    Network(String),
}
```

Each tool's `execute()` becomes:

```rust
fn execute(&self, ctx: &ExecutionContext, input: Value) -> Pin<Box<dyn Future<...>>> {
    Box::pin(async move {
        let parsed: ThreadInput = serde_json::from_value(input)
            .map_err(|e| Error::Agent(format!("invalid input: {e}")))?;
        let client = XClient::from_context(ctx).await
            .map_err(|e| Error::Agent(e.to_string()))?;
        match self.call_x(&client, parsed).await {
            Ok(out) => Ok(ToolOutput::success(serde_json::to_string(&out).unwrap())),
            Err(e) => Ok(ToolOutput::error(format_error(&e))),
        }
    })
}
```

The `format_error` helper maps `XApiError` variants to user-friendly messages:
- `Unauthenticated` → `"X auth failed; check credentials"`
- `RateLimited { retry_after_secs }` → `"rate limited; retry after {N}s"`
- `ApiError { status, message }` → `"X API error ({status}): {message}"`

## 3. Credential resolver contract

The new tools resolve these 4 names from `CredentialResolver`:

| Name | Type (X API term) |
|---|---|
| `X_CONSUMER_KEY` | App-level OAuth1 consumer key |
| `X_CONSUMER_SECRET` | App-level OAuth1 consumer secret |
| `X_ACCESS_TOKEN` | User-context OAuth1 access token |
| `X_ACCESS_TOKEN_SECRET` | User-context OAuth1 access token secret |

The names match the existing `TwitterCredentials` field semantics. A future env-var-backed `CredentialResolver` (deployment-side concern, not shipped here) would map `X_CONSUMER_KEY` → `std::env::var("X_CONSUMER_KEY")`.

If the resolver is `None` on the context, all new tools return a clear error: `"no credential resolver configured; X tools require ExecutionContext::credentials to be set"`. This makes the failure mode unambiguous for users who construct an `AgentRunner` without wiring a resolver.

## 4. File structure

```
crates/heartbit-ghost/
  Cargo.toml                          (add reqwest, oauth1-request, serde, serde_json, thiserror; wiremock dev)
  src/
    lib.rs                            (existing — XGhostPersona stub + register, unchanged)
    tools/
      mod.rs                          (pub use {thread::TwitterThreadTool, …, client::XClient, client::XApiError})
      client.rs                       (~200 lines)
      thread.rs                       (~120 lines)
      reply.rs                        (~80 lines)
      search.rs                       (~120 lines)
      mentions.rs                     (~120 lines)
      user.rs                         (~100 lines)

crates/heartbit-core/src/tool/builtins/
  twitter_post.rs                     (in-place: extend schema + add media-upload code path)
```

`heartbit-ghost/Cargo.toml` adds:
```toml
[dependencies]
reqwest = { workspace = true, features = ["json", "multipart"] }
serde = { workspace = true }
serde_json = { workspace = true }
thiserror = { workspace = true }
oauth1-request = "0.6"
async-trait = { workspace = true }    # if needed for trait helpers; verify at impl time

[dev-dependencies]
wiremock = "0.6"
tokio = { workspace = true }          # already there
```

## 5. Test plan

### 5.1 Per-tool unit tests (`#[cfg(test)] mod tests` in each tool's file)

Every new tool has at minimum:
- **Happy path** — wiremock stubs the X endpoint with a 200 + canonical payload; `tool.execute(&ctx, input).await` yields `ToolOutput::success(...)` with the expected JSON content
- **Auth failure** — wiremock returns 401; tool yields `ToolOutput::error("X auth failed; check credentials")` (or the equivalent `format_error` output)
- **Rate limited** — wiremock returns 429 with `Retry-After: 30`; tool yields `ToolOutput::error("rate limited; retry after 30s")`
- **X 4xx error** — wiremock returns 400 + an X error-shape JSON; tool surfaces the X message verbatim under `"X API error (400): <message>"`
- **No credentials** — `ExecutionContext` constructed with `credentials: None`; tool yields the clear "no credential resolver configured" error
- **Schema rejection** — the framework's `validate_tool_input` rejects malformed input *before* `execute` runs; existing infrastructure covers this. One smoke test per tool to confirm schemas are well-formed JSON Schema (e.g., serialize the tool's `definition()` and feed an invalid input through `validate_tool_input`)

Approximate test count per tool: 5–6. Total across 5 tools: ~25–30.

### 5.2 `client.rs` tests

- **OAuth1 signature** — sign a fixed request, assert the resulting `Authorization` header matches the X-spec example values (or just round-trips through `oauth1-request` cleanly)
- **`from_context` resolves all 4 credentials** — mock resolver, verify all 4 names are resolved
- **`from_context` errors clearly when resolver is missing**
- **Error mapping** — feed each XApiError variant through `format_error`, assert the string

Approximate test count: 4–6.

### 5.3 `twitter_post` extension tests (in `heartbit-core/src/tool/builtins/twitter_post.rs`)

- **Existing text-only path still works** — the existing tests stay green
- **With `media_url`** — wiremock stubs both `/1.1/media/upload.json` and `/2/tweets`; tool produces `tweet_id` after both calls succeed
- **With `media_url` + `media_alt_text`** — wiremock additionally stubs `/1.1/media/metadata/create.json`
- **Media too large** — mock returns a 5-MB+ response from the URL fetch; tool errors clearly
- **Media URL fetch fails (404)** — tool errors clearly without polluting the X account

Approximate new test count: 4–5.

### 5.4 Total

Approximate **35–40 new tests**. Workspace test count goes from 3729 → ~3765.

## 6. Implementation phases (within P1.1)

9 tasks. Each independently committed. Estimated total: 1–2 days of subagent-driven execution.

| # | Task | Files | Deps |
|---|---|---|---|
| 1 | Cargo.toml + scaffolding | `heartbit-ghost/Cargo.toml`, `tools/mod.rs` | none |
| 2 | `XClient` + OAuth1 + credential resolution + error mapping | `tools/client.rs` | task 1 |
| 3 | `TwitterUserTool` (simplest read-only) | `tools/user.rs` | task 2 |
| 4 | `TwitterSearchTool` (read; pagination basics) | `tools/search.rs` | task 2 |
| 5 | `TwitterMentionsTool` (read; reuses pagination from search) | `tools/mentions.rs` | task 4 |
| 6 | `TwitterReplyTool` (write) | `tools/reply.rs` | task 2 |
| 7 | `TwitterThreadTool` (write; iterates posts with `in_reply_to`) | `tools/thread.rs` | task 6 |
| 8 | `twitter_post` extension (heartbit-core) | `heartbit-core/src/tool/builtins/twitter_post.rs` | task 1 (only for cargo invariants — not strictly dependent on heartbit-ghost work) |
| 9 | Final acceptance + workspace quality gate + CHANGELOG | `CHANGELOG.md`, smoke tests | tasks 1–8 |

**Critical-path order:** 1 → 2 → 3 → 4 → 5 → 6 → 7. Task 8 (twitter_post extension) can run in parallel with any of 3–7 since it touches a different crate. Task 9 is final.

## 7. Acceptance criteria

P1.1 is done when:

- All 5 new tools exist as `pub` types in `heartbit_ghost::tools::*`, each implementing `heartbit_core::Tool`
- `twitter_post` accepts optional `media_url` + `media_alt_text` and posts text+image when both are set
- ~35–40 new tests across heartbit-ghost (per §5.1, §5.2) and heartbit-core (§5.3); all passing
- `cargo fmt -- --check && cargo clippy --workspace --all-features -- -D warnings && cargo test --workspace --all-features` all green
- A smoke test (manual, not automated): with a real `OPENROUTER_API_KEY` and real X credentials in env, an `AgentRunner` configured with `TwitterUserTool` can ask the LLM "look up @karpathy on X" and get a structured response back. Confirms end-to-end credential resolution path works against the live X API. (This requires real credentials and is one-time validation, not a CI test.)
- CHANGELOG entry under `[Unreleased]`: lists the 5 new tools + twitter_post extension

## 8. Out of scope (explicit)

- `twitter_dm`, `twitter_schedule`, `twitter_metrics` — P1.4 sub-phase
- Wiring tools into `heartbit_ghost::XGhostPersona::expand()` — P1.3 sub-phase (this just builds the catalog)
- Persona-level rate limiting (12 posts/day, 1/hour burst) — P1.4
- Per-tenant audit logging via `AuditSink` — wired when persona uses it (P1.3+)
- Bearer-token (app-only) auth for read-only endpoints — could simplify search/user; OAuth1 works for everything; deferred
- DRY-ing the OAuth1 logic between heartbit-core's `twitter_post` and heartbit-ghost's `XClient` — post-merge cleanup
- Auth/credential resolution against vault, AWS Secrets Manager, etc. — that's a `CredentialResolver` impl, shipped separately
- Twitter v1.1 endpoints other than media-upload paths — v2 only for new tools
- Streaming tool output (`twitter_search` could stream paginated results) — single-page returns only in P1.1; pagination is via explicit `since_id` / `next_token` fields

## 9. Open questions (deferred to implementation)

These are minor decisions the implementer can make during the plan; not blocking the spec:

- Whether `XClient` is `Clone` (would let tools share an instance per turn instead of re-resolving credentials each call). Probably no — credential resolution happens once per `from_context` call which is once per `execute()` call which is once per turn.
- Whether `twitter_thread` halts immediately on the first failure or attempts to roll back posted tweets. Fail-fast is simpler and matches X's actual behavior (no rollback API exists). Document the failure mode in the tool description.
- Whether `twitter_search` accepts X-specific advanced search operators (`from:`, `to:`, `lang:en`, etc.) verbatim or wraps them. Pass-through is simpler; users who want advanced operators can include them in `query`.
- Exact `media_url` validation policy in `twitter_post` extension — accept `https://` only? size limit (5 MB)? content-type sniffing? Implementer should add a small validation layer; specifics can be tuned in PR review.

## 10. Risks

- **OAuth1 signing edge cases** — `oauth1-request` crate may not handle every X-specific parameter ordering. Mitigation: per-tool wiremock test that asserts the produced `Authorization` header matches the X spec; fallback to inline implementation if the crate misbehaves.
- **X API rate-limit error format** — X's 429 responses sometimes return `Retry-After` in seconds, sometimes in HTTP-date format. The shared error mapper must handle both.
- **Media upload response parsing** — X v1.1 media endpoint returns `media_id` (number) and `media_id_string` (string). Always use the string form; large media IDs overflow JSON number precision in some clients.
- **Dependency churn** — adding `oauth1-request` and `wiremock` introduces new transitive deps. Mitigation: `cargo tree -p heartbit-ghost` audit at the end of P1.1 to confirm reasonable footprint.

## 11. Reference

- Heartbit-ghost umbrella spec: `docs/superpowers/specs/2026-05-07-heartbit-ghost-x-agent-design.md`
- P1.0 spec: implicit in §9 of the umbrella spec; plan at `docs/superpowers/plans/2026-05-07-heartbit-ghost-p1.0-scaffolding.md`
- Foundation spec: `docs/superpowers/specs/2026-05-07-heartbit-foundation-design.md` (`ExecutionContext`, `CredentialResolver`, `Secret` definitions)
- X API docs: https://docs.x.com/x-api/posts/post-creation, https://developer.x.com/en/docs/twitter-api
