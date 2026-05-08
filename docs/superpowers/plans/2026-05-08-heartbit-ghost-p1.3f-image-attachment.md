# heartbit-ghost P1.3f — Image attachment to thread head — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate an image to accompany the chosen draft and attach it to the **first tweet** of the thread when posting via `twitter_thread`. Skip cleanly when `image_generator` returns `"no_image"`. Subsequent tweets stay text-only.

**Architecture:** Schema-additive change to `TwitterThreadTool` (new optional `head_image_b64` field). New `XClient::upload_image_chunked` does INIT → APPEND → FINALIZE chunked upload at `POST https://api.x.com/2/media/upload`. `run_review_pipeline` runs `image_generator` AFTER publish_gate (cost-conscious — no image work on rejected drafts), parses the `[IMAGE:base64:...]` marker, and threads the base64 string through to the existing `twitter_tool.execute`.

**Tech Stack:** Rust 2024, `reqwest` (already configured with `multipart` feature), `base64` (workspace dep), `wiremock` (heartbit-ghost dev-dep) for HTTP fixture tests, OAuth 1.0a (existing `XClient::sign` reused — multipart bodies aren't signed per RFC 5849 §3.4.1.3.1, so empty `query` slice works). **No new workspace deps.**

---

## File structure

| File | Action | Responsibility |
|---|---|---|
| `crates/heartbit-ghost/src/tools/client.rs` | MODIFY | Add `XClient::upload_image_chunked(bytes, mime_type) -> Result<String, XApiError>` (`pub(crate)`) + private `post_multipart` helper. INIT/APPEND/FINALIZE single-segment flow. 3 wiremock-backed unit tests. |
| `crates/heartbit-ghost/src/tools/thread.rs` | MODIFY | `ThreadInput` gains `head_image_b64: Option<String>` (`#[serde(default)]`). New `PostRequest.media: Option<Media>` field. `Media { media_ids: Vec<String> }`. Update tool's `input_schema` JSON. Private `sniff_mime` (magic-byte check) + `decode_and_validate` (base64 + 5 MiB cap). `call_x` decodes → uploads → attaches `media_ids` to first PostRequest only. 4 new unit tests. |
| `crates/heartbit-ghost/src/review/mod.rs` | MODIFY | New `pub(crate) fn extract_image_marker(raw: &str) -> Option<String>`. `ReviewOutput` gains `head_image_attached: bool`. `run_review_pipeline` body: after publish_gate succeeds and BEFORE `twitter_tool.execute`, build image_generator runner via `runner_from_recipe`, execute on chosen draft, extract marker, pass `head_image_b64` JSON field to twitter_tool. Failure non-blocking. 3 helper unit tests + 2 new integration tests. |

3 implementation tasks + 1 final acceptance.

---

## Task 1: `XClient::upload_image_chunked` + 3 wiremock tests

**Why:** Foundation. The chunked upload is the only really new networking primitive; everything else in P1.3f is plumbing on top. Standalone testable via `wiremock`.

**Files:**
- Modify: `crates/heartbit-ghost/src/tools/client.rs`

- [ ] **Step 1: Append `upload_image_chunked` (and a private `post_multipart` helper) to `XClient` impl block in `crates/heartbit-ghost/src/tools/client.rs`**

Inside `impl XClient` block (after `post_json`, before `sign`):

```rust
    /// Upload an image to X via the v2 chunked media upload endpoint.
    /// Returns the `media_id` to attach to a subsequent `POST /2/tweets`.
    ///
    /// Implements INIT → APPEND → FINALIZE for a single segment. One
    /// segment is sufficient for any image we'd reasonably attach
    /// (X's per-image limit is 5 MiB, validated by the caller before
    /// invoking this method).
    pub(crate) async fn upload_image_chunked(
        &self,
        bytes: &[u8],
        mime_type: &str,
    ) -> Result<String, XApiError> {
        // 1. INIT — declare intent + size, get a media_id back.
        let init_url = format!("{}/2/media/upload", self.base_url);
        let total_bytes = bytes.len().to_string();
        let init_form = reqwest::multipart::Form::new()
            .text("command", "INIT")
            .text("media_type", mime_type.to_string())
            .text("total_bytes", total_bytes)
            .text("media_category", "tweet_image");
        let init_resp: InitResponse =
            self.post_multipart(&init_url, init_form).await?;
        let media_id = init_resp.data.id;

        // 2. APPEND — single segment.
        let append_url = format!("{}/2/media/upload", self.base_url);
        let part = reqwest::multipart::Part::bytes(bytes.to_vec())
            .file_name("image")
            .mime_str(mime_type)
            .map_err(|e| XApiError::Network(format!("invalid mime_type '{mime_type}': {e}")))?;
        let append_form = reqwest::multipart::Form::new()
            .text("command", "APPEND")
            .text("media_id", media_id.clone())
            .text("segment_index", "0")
            .part("media", part);
        // APPEND returns no significant body; just check status.
        self.post_multipart_no_body(&append_url, append_form).await?;

        // 3. FINALIZE — completes synchronously for tweet_image.
        let finalize_url = format!("{}/2/media/upload", self.base_url);
        let finalize_form = reqwest::multipart::Form::new()
            .text("command", "FINALIZE")
            .text("media_id", media_id.clone());
        let _finalize_resp: FinalizeResponse =
            self.post_multipart(&finalize_url, finalize_form).await?;

        Ok(media_id)
    }

    /// POST a multipart form, sign via OAuth1 (URL + method only — multipart
    /// bodies aren't included in the OAuth signature base string per
    /// RFC 5849 §3.4.1.3.1), parse the JSON response.
    async fn post_multipart<T: DeserializeOwned>(
        &self,
        url: &str,
        form: reqwest::multipart::Form,
    ) -> Result<T, XApiError> {
        let auth_header = self.sign("POST", url, &[])?;
        let response = self
            .http
            .post(url)
            .header("Authorization", auth_header)
            .multipart(form)
            .send()
            .await
            .map_err(|e| XApiError::Network(e.to_string()))?;
        Self::parse_response(response).await
    }

    /// Like `post_multipart` but discards the response body — used for
    /// APPEND which returns 2xx with no body fields we need.
    async fn post_multipart_no_body(
        &self,
        url: &str,
        form: reqwest::multipart::Form,
    ) -> Result<(), XApiError> {
        let auth_header = self.sign("POST", url, &[])?;
        let response = self
            .http
            .post(url)
            .header("Authorization", auth_header)
            .multipart(form)
            .send()
            .await
            .map_err(|e| XApiError::Network(e.to_string()))?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            let message = extract_x_error_message(&body)
                .unwrap_or_else(|| body.chars().take(200).collect());
            return Err(XApiError::ApiError {
                status: status.as_u16(),
                message,
            });
        }
        Ok(())
    }
```

Add response types near the existing `PostApiResponse` / `PostApiData` (file currently has these for tweet creation):

```rust
#[derive(Debug, Deserialize)]
struct InitResponse {
    data: InitData,
}

#[derive(Debug, Deserialize)]
struct InitData {
    id: String,
    #[serde(default)]
    #[allow(dead_code)]
    media_key: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    expires_after_secs: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct FinalizeResponse {
    #[serde(default)]
    #[allow(dead_code)]
    data: serde_json::Value,
}
```

(`#[allow(dead_code)]` on `media_key` / `expires_after_secs` / `FinalizeResponse.data` is justified — we don't read them, but they're kept on the struct for documentation and future-proofing. Clippy `dead_code` enforces if we don't tag.)

- [ ] **Step 2: Add 3 wiremock-backed unit tests inside the existing `#[cfg(test)] mod tests` in `crates/heartbit-ghost/src/tools/client.rs`**

Append after the existing tests:

```rust
    use wiremock::matchers::{body_string_contains, header_exists, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    /// Helper: build an XClient pointed at a wiremock MockServer.
    /// Reuses the existing `XClient::new(...)` constructor pattern.
    /// (If the existing tests already have a `mk_client(server)` helper,
    /// reuse that and skip this duplicate.)
    fn mk_client_for(base_url: &str) -> XClient {
        XClient::new(
            base_url,
            Secret::new("ck"),
            Secret::new("cs"),
            Secret::new("at"),
            Secret::new("ats"),
        )
    }

    #[tokio::test]
    async fn upload_image_chunked_happy_path_returns_media_id() {
        let server = MockServer::start().await;
        let media_id = "1234567890";
        // INIT
        Mock::given(method("POST"))
            .and(path("/2/media/upload"))
            .and(body_string_contains("INIT"))
            .and(header_exists("authorization"))
            .respond_with(ResponseTemplate::new(202).set_body_json(serde_json::json!({
                "data": {"id": media_id, "media_key": "mk_x", "expires_after_secs": 86400}
            })))
            .expect(1)
            .mount(&server)
            .await;
        // APPEND
        Mock::given(method("POST"))
            .and(path("/2/media/upload"))
            .and(body_string_contains("APPEND"))
            .respond_with(ResponseTemplate::new(204))
            .expect(1)
            .mount(&server)
            .await;
        // FINALIZE
        Mock::given(method("POST"))
            .and(path("/2/media/upload"))
            .and(body_string_contains("FINALIZE"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": {"id": media_id, "processing_info": {"state": "succeeded"}}
            })))
            .expect(1)
            .mount(&server)
            .await;

        let client = mk_client_for(&server.uri());
        let bytes = b"\x89PNG\r\n\x1a\nfake_png_payload";
        let id = client
            .upload_image_chunked(bytes, "image/png")
            .await
            .expect("happy path");
        assert_eq!(id, media_id);
    }

    #[tokio::test]
    async fn upload_image_chunked_init_4xx_surfaces_api_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/2/media/upload"))
            .and(body_string_contains("INIT"))
            .respond_with(ResponseTemplate::new(401).set_body_string(
                r#"{"errors":[{"message":"Unauthorized"}]}"#,
            ))
            .mount(&server)
            .await;
        let client = mk_client_for(&server.uri());
        let bytes = b"\x89PNG\r\n";
        let err = client
            .upload_image_chunked(bytes, "image/png")
            .await
            .unwrap_err();
        match err {
            XApiError::Unauthenticated(msg) => {
                assert!(msg.contains("Unauthorized"), "got: {msg}");
            }
            other => panic!("expected Unauthenticated, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn upload_image_chunked_append_5xx_surfaces_api_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/2/media/upload"))
            .and(body_string_contains("INIT"))
            .respond_with(ResponseTemplate::new(202).set_body_json(serde_json::json!({
                "data": {"id": "id_x"}
            })))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/2/media/upload"))
            .and(body_string_contains("APPEND"))
            .respond_with(ResponseTemplate::new(503).set_body_string("upstream down"))
            .mount(&server)
            .await;
        let client = mk_client_for(&server.uri());
        let err = client
            .upload_image_chunked(b"\x89PNG", "image/png")
            .await
            .unwrap_err();
        match err {
            XApiError::ApiError { status, .. } => assert_eq!(status, 503),
            other => panic!("expected ApiError(503), got: {other:?}"),
        }
    }
```

> Note on the `Unauthenticated` test arm: the existing `parse_response` in `client.rs` distinguishes 401 (→ `Unauthenticated`) from other 4xx (→ `ApiError`). If your local `parse_response` is different, adjust the test's expected variant. The first test's match needs to align with `parse_response`'s actual classification.

- [ ] **Step 3: Run the new tests**

```bash
cd /home/pleclech/projects/heartbit
cargo test -p heartbit-ghost --lib tools::client::tests::upload_image_chunked 2>&1 | tail -8
```

Expected: 3 passed, 0 failed.

- [ ] **Step 4: Workspace gate (heartbit-ghost only)**

```bash
cargo fmt -p heartbit-ghost -- --check
cargo clippy -p heartbit-ghost --all-targets -- -D warnings
cargo test -p heartbit-ghost --lib tools::client 2>&1 | tail -3
```

All clean.

- [ ] **Step 5: Commit**

```bash
cd /home/pleclech/projects/heartbit
git add crates/heartbit-ghost/src/tools/client.rs
git commit -m "$(cat <<'EOF'
feat(ghost): tools/client — XClient::upload_image_chunked (P1.3f)

Add INIT → APPEND → FINALIZE chunked image upload to X via the v2
endpoint POST https://api.x.com/2/media/upload (v1.1 single-shot
deprecated 2025-03-31). Single-segment flow — sufficient for
any image we'd reasonably attach to a tweet (5 MiB cap, validated
by caller).

OAuth 1.0a User Context — reuses existing `XClient::sign` with an
empty `query` slice. Per RFC 5849 §3.4.1.3.1, multipart bodies
aren't included in the OAuth signature base string, so the existing
sign helper works without modification.

New private helpers:
- `post_multipart<T>` — generic POST + sign + parse JSON response
- `post_multipart_no_body` — APPEND uses this (returns 204 with no
  meaningful body)

3 wiremock-backed tests cover happy-path media_id round-trip, INIT
4xx (Unauthenticated branch), APPEND 5xx (ApiError branch).

Refs: docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.3f-image-attachment-design.md §3.1, §4
EOF
)"
```

---

## Task 2: `TwitterThreadTool` head_image_b64 + MIME sniff + decode + attach

**Why:** Schema-additive — the new optional `head_image_b64` field doesn't break any existing caller. MIME sniffing via magic-byte check (no new dep). 5 MiB cap rejected before INIT to avoid wasted upload bandwidth. `call_x` decodes → uploads → attaches `media_ids` to the first PostRequest's `media` field.

**Files:**
- Modify: `crates/heartbit-ghost/src/tools/thread.rs`

- [ ] **Step 1: Update `ThreadInput` and add `Media` / update `PostRequest` in `crates/heartbit-ghost/src/tools/thread.rs`**

Find the existing `ThreadInput` struct and replace:

```rust
#[derive(Debug, Deserialize)]
struct ThreadInput {
    tweets: Vec<String>,
    /// Optional base64-encoded image bytes to attach to the FIRST tweet.
    /// MIME type is sniffed from the bytes (PNG/JPEG/WebP/GIF).
    /// When `None`, posts text-only (existing behavior preserved).
    #[serde(default)]
    head_image_b64: Option<String>,
}
```

Find the existing `PostRequest` struct and replace:

```rust
#[derive(Debug, Serialize)]
struct PostRequest<'a> {
    text: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    reply: Option<ReplyTo<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    media: Option<Media>,
}

#[derive(Debug, Serialize)]
struct Media {
    media_ids: Vec<String>,
}
```

- [ ] **Step 2: Add MIME-sniff and decode-validate helpers near the top of `thread.rs` (after the use statements, before `MAX_TWEET_LENGTH`)**

```rust
const MAX_HEAD_IMAGE_BYTES: usize = 5 * 1024 * 1024; // 5 MiB

/// Sniff MIME type from the first bytes of an image. Returns one of
/// `"image/png"`, `"image/jpeg"`, `"image/gif"`, `"image/webp"`. Falls
/// back to `"image/png"` when no magic-bytes match (Gemini's image
/// preview model returns PNG by default, and X infers from media_type).
fn sniff_mime(bytes: &[u8]) -> &'static str {
    if bytes.len() >= 8 && &bytes[..8] == b"\x89PNG\r\n\x1a\n" {
        return "image/png";
    }
    if bytes.len() >= 3 && &bytes[..3] == b"\xFF\xD8\xFF" {
        return "image/jpeg";
    }
    if bytes.len() >= 6 && (&bytes[..6] == b"GIF87a" || &bytes[..6] == b"GIF89a") {
        return "image/gif";
    }
    if bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        return "image/webp";
    }
    "image/png"
}

/// Decode the caller's base64 string and validate size.
/// Returns `(bytes, sniffed_mime)` on success, or an error message
/// suitable for `ToolOutput::error`.
fn decode_and_validate_head_image(b64: &str) -> Result<(Vec<u8>, &'static str), String> {
    use base64::Engine;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64.trim())
        .map_err(|e| format!("invalid head_image_b64: {e}"))?;
    if bytes.len() > MAX_HEAD_IMAGE_BYTES {
        return Err(format!(
            "head image exceeds 5 MiB (got {} bytes)",
            bytes.len()
        ));
    }
    if bytes.is_empty() {
        return Err("head_image_b64 decoded to zero bytes".to_string());
    }
    let mime = sniff_mime(&bytes);
    Ok((bytes, mime))
}
```

- [ ] **Step 3: Update `call_x` to handle `head_image_b64` and attach `media_ids` to the first tweet only**

The current `call_x` signature is `async fn call_x(client: &XClient, input: &ThreadInput) -> Result<ThreadOutput, XApiError>`. Replace the body:

```rust
async fn call_x(client: &XClient, input: &ThreadInput) -> Result<ThreadOutput, XApiError> {
    // 1. If head_image_b64 is set, decode → validate → upload → media_id.
    let mut head_media_ids: Option<Vec<String>> = None;
    if let Some(b64) = input.head_image_b64.as_ref() {
        let (bytes, mime) = decode_and_validate_head_image(b64)
            .map_err(|msg| XApiError::ApiError { status: 0, message: msg })?;
        let media_id = client.upload_image_chunked(&bytes, mime).await?;
        head_media_ids = Some(vec![media_id]);
    }

    // 2. Post tweets sequentially. First tweet gets the optional media;
    // subsequent tweets are text-only and reply to the previous.
    let mut tweet_ids: Vec<String> = Vec::with_capacity(input.tweets.len());
    let mut prev_id: Option<String> = None;
    for (idx, text) in input.tweets.iter().enumerate() {
        let media = if idx == 0 {
            head_media_ids.take().map(|media_ids| Media { media_ids })
        } else {
            None
        };
        let body = PostRequest {
            text,
            reply: prev_id.as_ref().map(|id| ReplyTo {
                in_reply_to_tweet_id: id,
            }),
            media,
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
```

> The existing `call_x` body had only the per-tweet posting loop with `text` + `reply`. The new body adds the head-upload step before the loop and inserts `media` on the first iteration.

The decode error path uses `XApiError::ApiError { status: 0, message }` — `status: 0` is a sentinel meaning "validation failed before any HTTP call was made." Caller-facing error text comes through unchanged via `format_error`. Alternative: add a new `XApiError::Validation(String)` variant. Stick with the sentinel for minimal churn; the plan can revisit if `format_error` mishandles it.

Actually verify `format_error`'s behavior on `status: 0`. Open `crates/heartbit-ghost/src/tools/client.rs` and check the `format_error` function. If it produces something nonsensical for `status: 0` (e.g., "X API error (0): ..."), prefer adding a `Validation(String)` variant to `XApiError`:

```rust
// In client.rs's XApiError enum, add:
    /// Caller-side validation failed before any HTTP call (e.g., invalid
    /// base64, image too large).
    #[error("validation: {0}")]
    Validation(String),
```

And in `format_error`, add an arm that returns the inner message as-is:

```rust
        XApiError::Validation(msg) => msg.clone(),
```

Then change the decode error path in `call_x` to:

```rust
    .map_err(XApiError::Validation)?;
```

This is the cleaner option. Use it if you can do the change in this same Task 2 (file already touched). The plan endorses it as an inline tightening of XApiError.

- [ ] **Step 4: Update the tool's `input_schema` JSON in `TwitterThreadTool::definition`**

Find the existing `definition()` impl and update the `input_schema` to include `head_image_b64`:

```rust
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "tweets": {
                        "type": "array",
                        "items": {"type": "string", "maxLength": 280},
                        "minItems": 1,
                        "maxItems": 25,
                        "description": "Ordered list of tweet texts. The first becomes the thread root; each subsequent tweet replies to the previous."
                    },
                    "head_image_b64": {
                        "type": "string",
                        "description": "Optional base64-encoded image to attach to the first tweet (PNG/JPEG/WebP/GIF, ≤5 MiB). When omitted, posts text-only."
                    }
                },
                "required": ["tweets"]
            }),
```

- [ ] **Step 5: Add 4 new unit tests inside the existing `#[cfg(test)] mod tests` in `crates/heartbit-ghost/src/tools/thread.rs`**

Append after the existing tests:

```rust
    use base64::Engine;

    #[test]
    fn thread_input_deserializes_with_head_image_b64() {
        let json = r#"{"tweets": ["hello"], "head_image_b64": "iVBORw0KGgo="}"#;
        let parsed: ThreadInput = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.tweets, vec!["hello".to_string()]);
        assert_eq!(parsed.head_image_b64.as_deref(), Some("iVBORw0KGgo="));
    }

    #[test]
    fn thread_input_deserializes_without_head_image_b64() {
        // Existing P1.3d callers don't pass the field; verify back-compat.
        let json = r#"{"tweets": ["hello"]}"#;
        let parsed: ThreadInput = serde_json::from_str(json).unwrap();
        assert!(parsed.head_image_b64.is_none());
    }

    #[test]
    fn sniff_mime_recognizes_known_magic_bytes() {
        assert_eq!(sniff_mime(b"\x89PNG\r\n\x1a\nfoo"), "image/png");
        assert_eq!(sniff_mime(b"\xFF\xD8\xFFfoo"), "image/jpeg");
        assert_eq!(sniff_mime(b"GIF87afoo"), "image/gif");
        assert_eq!(sniff_mime(b"GIF89afoo"), "image/gif");
        // WebP: bytes 0..4 = RIFF, bytes 8..12 = WEBP.
        let mut webp = b"RIFF\x00\x00\x00\x00WEBP".to_vec();
        webp.push(0xAA);
        assert_eq!(sniff_mime(&webp), "image/webp");
        // Unknown → png fallback.
        assert_eq!(sniff_mime(b"definitely not an image"), "image/png");
        // Too short → png fallback (won't match any magic).
        assert_eq!(sniff_mime(b"abc"), "image/png");
    }

    #[test]
    fn decode_and_validate_head_image_rejects_oversize() {
        // Build a 6 MiB raw byte vec, encode to base64.
        let big = vec![0u8; 6 * 1024 * 1024];
        let b64 = base64::engine::general_purpose::STANDARD.encode(&big);
        let err = decode_and_validate_head_image(&b64).unwrap_err();
        assert!(err.contains("exceeds 5 MiB"), "got: {err}");
    }
```

- [ ] **Step 6: Run the new tests + the existing thread tests**

```bash
cd /home/pleclech/projects/heartbit
cargo test -p heartbit-ghost --lib tools::thread 2>&1 | tail -10
```

Expected: existing thread tests + 4 new = all pass.

- [ ] **Step 7: Workspace gate**

```bash
cargo fmt -p heartbit-ghost -- --check
cargo clippy -p heartbit-ghost --all-targets -- -D warnings
```

Both clean. If clippy complains about an unused `Media` import in tests, that's because we don't construct `Media` directly in tests (only via deserialization). Suppress via `#[allow(dead_code)]` only if necessary; otherwise leave clean.

- [ ] **Step 8: Commit**

```bash
cd /home/pleclech/projects/heartbit
git add crates/heartbit-ghost/src/tools/thread.rs crates/heartbit-ghost/src/tools/client.rs
git commit -m "$(cat <<'EOF'
feat(ghost): tools/thread — head_image_b64 attaches image to first tweet (P1.3f)

Schema-additive: ThreadInput gains optional `head_image_b64: Option<String>`
field. When present:

1. base64-decode + validate (≤5 MiB cap, non-empty)
2. MIME sniff via magic-byte check (PNG / JPEG / WebP / GIF; png fallback)
3. XClient::upload_image_chunked → media_id
4. First PostRequest gets `media: Some(Media { media_ids: vec![media_id] })`
5. Subsequent tweets in the thread stay text-only

Existing callers (P1.3d run_review_pipeline today, P1.3c direct mode)
that don't pass the field continue to work — `#[serde(default)]` on the
new field, `#[serde(skip_serializing_if = "Option::is_none")]` on
PostRequest.media.

XApiError gains a `Validation(String)` variant for caller-side errors
(invalid base64, oversize image) — kept distinct from API status codes.
format_error returns the inner message as-is.

input_schema JSON updated to advertise the new field to LLMs.

4 new unit tests: deserialize with/without field, MIME sniffing across
all 4 supported types + fallback, oversize rejection.

Refs: docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.3f-image-attachment-design.md §3.2
EOF
)"
```

---

## Task 3: `run_review_pipeline` integration + `extract_image_marker` + `head_image_attached`

**Why:** Wire `image_generator` into the review-mode pipeline. Run AFTER publish_gate (cost-conscious — no image work on rejected drafts). Parse the marker. Pass the base64 string through to `twitter_tool` as a JSON field. Failure non-blocking.

**Files:**
- Modify: `crates/heartbit-ghost/src/review/mod.rs`

- [ ] **Step 1: Add `extract_image_marker` helper**

Append to `review/mod.rs` (place after `parse_twitter_thread_output`, before `run_review_pipeline`):

```rust
/// Extract the base64 image data from `image_generator`'s output.
///
/// `ImageGenerateTool` emits matched output with the prefix
/// `[IMAGE:base64:` and a closing `]`. Returns `None` when:
/// - the marker is absent
/// - the recipe returned the literal `"no_image"` (case-insensitive)
/// - input is empty / whitespace
///
/// Base64 alphabet excludes `]`, so `find(']')` after the prefix is
/// safe.
pub(crate) fn extract_image_marker(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.eq_ignore_ascii_case("no_image") {
        return None;
    }
    let prefix = "[IMAGE:base64:";
    let start = trimmed.find(prefix)?;
    let after_prefix = &trimmed[start + prefix.len()..];
    let end = after_prefix.find(']')?;
    let b64 = &after_prefix[..end];
    if b64.is_empty() {
        return None;
    }
    Some(b64.to_string())
}
```

- [ ] **Step 2: Add `head_image_attached: bool` field to `ReviewOutput`**

Find the existing `pub struct ReviewOutput` and add the field:

```rust
/// Output of a successful review-mode run.
#[derive(Debug, Clone)]
pub struct ReviewOutput {
    pub candidates: Vec<CandidateRecord>,
    pub research_digest: String,
    pub usage_summary: TokenUsage,
    pub outcome: ReviewOutcome,
    /// Whether `image_generator` produced an image that was attached to
    /// the head tweet. `false` when the recipe returned `"no_image"`,
    /// when the call failed, when the marker was absent, or when the
    /// outcome was Skipped/TimedOut/GateRejected/PublishFailed (i.e.,
    /// no post happened).
    pub head_image_attached: bool,
}
```

- [ ] **Step 3: Update `run_review_pipeline` body — run image_generator after publish_gate, pass through to twitter_tool**

Find the `Pick(chosen_index)` arm of the outcome match in `run_review_pipeline`. Inside that arm, after the `Ok(())` branch of `check_publish_gate`, BEFORE the `let tweets = parse_thread_tweets(...)` line, insert:

```rust
                Ok(()) => {
                    // 10b. NEW (P1.3f): run image_generator on the
                    // chosen draft. Failure is non-blocking — text-only
                    // post on any error.
                    progress("Generating optional image...");
                    let head_image_b64: Option<String> = {
                        let recipe = crate::agents::image_generator_recipe();
                        let image_tools: Vec<Arc<dyn Tool>> = vec![
                            Arc::new(heartbit_core::tool::builtins::ImageGenerateTool::new()),
                        ];
                        match crate::pipeline::runner_from_recipe(
                            cfg.provider.clone(), recipe, image_tools,
                        ) {
                            Ok(image_runner) => {
                                let voice = crate::pipeline::render_style_profile_as_english(&profile);
                                let msg = format!(
                                    "Approved draft:\n{}\n\n{}\n\n\
                                     Decide whether to attach an image. \
                                     If no, output the literal string \"no_image\". \
                                     If yes, call image_generate with a concise visual prompt and return the result.",
                                    chosen.draft, voice,
                                );
                                match image_runner.execute(&msg).await {
                                    Ok(out) => {
                                        total_usage += out.tokens_used;
                                        extract_image_marker(&out.result)
                                    }
                                    Err(e) => {
                                        progress(&format!("image_generator failed (non-blocking): {e}"));
                                        None
                                    }
                                }
                            }
                            Err(e) => {
                                progress(&format!("image_generator builder failed (non-blocking): {e}"));
                                None
                            }
                        }
                    };
                    let head_image_attached = head_image_b64.is_some();

                    // 10c. Post via twitter_tool (with optional head image).
                    progress(&format!("Posting candidate {chosen_index}..."));
                    let tweets = parse_thread_tweets(&chosen.draft);
                    let exec_ctx = heartbit_core::ExecutionContext {
                        credentials: Some(cfg.credentials.clone()),
                        ..Default::default()
                    };
                    let mut input = serde_json::json!({"tweets": tweets});
                    if let Some(b64) = head_image_b64.as_ref() {
                        input["head_image_b64"] = serde_json::Value::String(b64.clone());
                    }
                    let post_outcome = match cfg.twitter_tool.execute(&exec_ctx, input).await {
                        Err(e) => {
                            let reason = format!("{e}");
                            progress(&format!("twitter_tool errored: {reason}"));
                            (
                                ReviewOutcome::PublishFailed { chosen_index, reason: reason.clone() },
                                ReportableOutcome::PublishFailed { chosen_index, reason },
                                false, // head_image_attached
                            )
                        }
                        Ok(tool_out) if tool_out.is_error => {
                            let reason = tool_out.content.clone();
                            progress(&format!("twitter_tool returned is_error=true: {reason}"));
                            (
                                ReviewOutcome::PublishFailed { chosen_index, reason: reason.clone() },
                                ReportableOutcome::PublishFailed { chosen_index, reason },
                                false,
                            )
                        }
                        Ok(tool_out) => {
                            let (tweet_ids, tweet_url) = parse_twitter_thread_output(&tool_out.content);
                            (
                                ReviewOutcome::Posted {
                                    chosen_index,
                                    tweet_url: tweet_url.clone(),
                                    tweet_ids,
                                },
                                ReportableOutcome::Posted { chosen_index, tweet_url },
                                head_image_attached, // true iff image actually generated AND post succeeded
                            )
                        }
                    };
                    (post_outcome.0, post_outcome.1, post_outcome.2)
                }
```

> The existing arm returns a 2-tuple `(ReviewOutcome, ReportableOutcome)`. P1.3f extends this to a 3-tuple `(ReviewOutcome, ReportableOutcome, bool)` where the bool is `head_image_attached`. Refactor the surrounding match arms (`Skip`, `TimedOut`, `GateRejected`) to also return 3-tuples with `false` as the third element.

Specifically, find the existing match block:

```rust
let (outcome, report) = match delivered.outcome {
    DeliveryOutcome::Skip => { /* ... */ (ReviewOutcome::Skipped, ReportableOutcome::Skipped) }
    DeliveryOutcome::TimedOut => { /* ... */ (ReviewOutcome::TimedOut, ReportableOutcome::TimedOut) }
    DeliveryOutcome::Pick(chosen_index) => { /* ... */ }
};
```

Change to:

```rust
let (outcome, report, head_image_attached) = match delivered.outcome {
    DeliveryOutcome::Skip => {
        progress("User skipped.");
        (ReviewOutcome::Skipped, ReportableOutcome::Skipped, false)
    }
    DeliveryOutcome::TimedOut => {
        progress("Review timed out.");
        (ReviewOutcome::TimedOut, ReportableOutcome::TimedOut, false)
    }
    DeliveryOutcome::Pick(chosen_index) => {
        if chosen_index >= candidates.len() {
            return Err(ReviewError::InvalidConfig(/* ...existing... */));
        }
        let chosen = &candidates[chosen_index];
        match crate::pipeline::check_publish_gate(&chosen.draft, &profile) {
            Err(gate_err) => {
                let reason = format!("{gate_err}");
                progress(&format!("publish_gate rejected pick: {reason}"));
                (
                    ReviewOutcome::GateRejected { chosen_index, reason: reason.clone() },
                    ReportableOutcome::GateRejected { chosen_index, reason },
                    false,
                )
            }
            Ok(()) => {
                // ...the new image_generator + post block from above...
            }
        }
    }
};
```

And update the `Ok(ReviewOutput { ... })` construction at the bottom of `run_review_pipeline` to set the new field:

```rust
Ok(ReviewOutput {
    candidates,
    research_digest,
    usage_summary: total_usage,
    outcome,
    head_image_attached,
})
```

- [ ] **Step 4: Add 3 unit tests for `extract_image_marker`**

Inside the existing `#[cfg(test)] mod tests` block in `review/mod.rs`, append:

```rust
    #[test]
    fn extract_image_marker_happy_path() {
        let raw = "preamble [IMAGE:base64:iVBORw0KGgo=] suffix";
        let out = extract_image_marker(raw);
        assert_eq!(out.as_deref(), Some("iVBORw0KGgo="));
    }

    #[test]
    fn extract_image_marker_no_image_returns_none() {
        assert!(extract_image_marker("no_image").is_none());
        assert!(extract_image_marker("NO_IMAGE").is_none());
        assert!(extract_image_marker("  no_image  ").is_none());
    }

    #[test]
    fn extract_image_marker_absent_marker_returns_none() {
        assert!(extract_image_marker("just some text").is_none());
        assert!(extract_image_marker("").is_none());
        // Marker prefix without closing bracket → None.
        assert!(extract_image_marker("[IMAGE:base64:abc").is_none());
        // Empty body between brackets → None.
        assert!(extract_image_marker("[IMAGE:base64:]").is_none());
    }
```

- [ ] **Step 5: Add 2 integration tests covering the new image-generator flow**

Inside the existing test mod, after the existing P1.3d integration tests, append:

```rust
    #[tokio::test]
    async fn run_review_pipeline_pick_with_image_generator_attaches_head_image() {
        let (_dir, profiles_root) = seed_snapshot("x");
        // Response queue: researcher, writer, critic, fact_check, then
        // image_generator (returns base64 marker), then... twitter_tool
        // (mocked separately, doesn't consume from the LLM queue).
        let provider = MockProvider::arc(vec![
            "Research digest:\n- AI is moving fast",                   // researcher
            "concrete short post",                                      // writer iter 1
            r#"{"verdict": "pass", "style_match_score": 0.92}"#,       // critic iter 1
            r#"{"verdict": "verified"}"#,                               // fact_check
            "[IMAGE:base64:iVBORw0KGgo=]",                              // image_generator
        ]);
        let delivery = MockReviewDelivery::arc(DeliveryOutcome::Pick(0));
        let twitter_tool = MockTwitterTool::success(
            r#"{"thread_root_id":"123","tweet_ids":["123"],"urls":["https://twitter.com/i/web/status/123"]}"#,
        );
        let cfg = mk_review_cfg(&profiles_root, provider, delivery, twitter_tool.clone());
        let out = run_review_pipeline(cfg).await.expect("happy path with image");
        assert!(matches!(out.outcome, ReviewOutcome::Posted { .. }));
        assert!(
            out.head_image_attached,
            "image_generator returned a marker; head_image_attached must be true"
        );
        // Assert the twitter_tool received head_image_b64 in its input.
        let last_input = twitter_tool
            .last_input()
            .expect("twitter_tool was called");
        assert_eq!(
            last_input.get("head_image_b64").and_then(|v| v.as_str()),
            Some("iVBORw0KGgo="),
            "twitter_tool input did not carry head_image_b64; got: {last_input}"
        );
    }

    #[tokio::test]
    async fn run_review_pipeline_pick_image_generator_no_image_posts_text_only() {
        let (_dir, profiles_root) = seed_snapshot("x");
        let provider = MockProvider::arc(vec![
            "Research digest:\n- notes",
            "draft text",
            r#"{"verdict": "pass", "style_match_score": 0.9}"#,
            r#"{"verdict": "verified"}"#,
            "no_image",                                                 // image_generator opts out
        ]);
        let delivery = MockReviewDelivery::arc(DeliveryOutcome::Pick(0));
        let twitter_tool = MockTwitterTool::success(
            r#"{"thread_root_id":"456","tweet_ids":["456"],"urls":["https://twitter.com/i/web/status/456"]}"#,
        );
        let cfg = mk_review_cfg(&profiles_root, provider, delivery, twitter_tool.clone());
        let out = run_review_pipeline(cfg).await.expect("no_image path");
        assert!(matches!(out.outcome, ReviewOutcome::Posted { .. }));
        assert!(
            !out.head_image_attached,
            "image_generator returned no_image; head_image_attached must be false"
        );
        let last_input = twitter_tool.last_input().expect("twitter_tool was called");
        assert!(
            last_input.get("head_image_b64").is_none(),
            "no_image path must NOT pass head_image_b64; got: {last_input}"
        );
    }
```

> **Note on `MockTwitterTool::last_input()`:** P1.3d's `MockTwitterTool` already captures `last_input` in a `Mutex<Option<serde_json::Value>>` but the field had no public getter (Task 3's code review flagged this as a dead-store). Add a public method `pub fn last_input(&self) -> Option<serde_json::Value>` on `MockTwitterTool`:
>
> ```rust
>     impl MockTwitterTool {
>         pub fn last_input(&self) -> Option<serde_json::Value> {
>             self.last_input.lock().unwrap().clone()
>         }
>     }
> ```
>
> If the existing struct's `last_input` field is `pub(super)` or private, change to allow read access from inside the test mod (it's already in the same module, so private works as long as we add the helper method).
>
> Also: `MockTwitterTool::success(...)` returns `Arc<dyn Tool>`. To call `.last_input()` we need access to the concrete type. Refactor `success` and `errored` to return `Arc<MockTwitterTool>` and let tests upcast via `Arc::clone` for `ReviewConfig.twitter_tool`:
>
> ```rust
>     impl MockTwitterTool {
>         pub fn success(body: &str) -> Arc<Self> {
>             Arc::new(MockTwitterTool {
>                 canned: Mutex::new(Some(ToolOutput::success(body))),
>                 last_input: Mutex::new(None),
>             })
>         }
>         // ...same for errored...
>     }
> ```
>
> Then test call sites: `let twitter_tool = MockTwitterTool::success(...);  let cfg = mk_review_cfg(..., twitter_tool.clone() as Arc<dyn Tool>);` — the `Arc::clone` produces another reference to the same instance; one goes into `ReviewConfig.twitter_tool` (`Arc<dyn Tool>` via auto-coerce or explicit `as` cast), the other stays in the test for inspecting `last_input()`.
>
> P1.3d's existing tests pass `MockTwitterTool::success(...)` directly into `mk_review_cfg`. Update those existing tests to use the new pattern (`twitter_tool.clone() as Arc<dyn Tool>` — minor change, ~5 sites). This is necessary mechanical churn for P1.3f.

- [ ] **Step 6: Update existing P1.3d integration tests for the new `head_image_attached` field**

The existing integration tests don't assert on `head_image_attached`. They construct `ReviewConfig` and call `run_review_pipeline`, then assert on `out.outcome`. Adding the field to `ReviewOutput` is a compile-time additive change — existing tests don't break unless they construct `ReviewOutput` literals (they don't).

However, the queue-of-canned-LLM-responses pattern means existing tests that reach the Pick branch and call `image_generator` will pop an unexpected response. Specifically:
- `run_review_pipeline_pick_index_0_posts_to_twitter` — needs an extra `"no_image"` response at the end of its provider queue (after fact_check, before twitter_tool). The twitter_tool is mocked separately so it doesn't consume from the LLM queue, but `image_generator` is a real AgentRunner that DOES consume from the queue.
- Same for `run_review_pipeline_pick_publish_gate_rejects_long_draft` — but this test never reaches image_generator (gate rejects first), so no queue update needed.
- Same for `run_review_pipeline_pick_twitter_api_error_returns_publish_failed` — this DOES reach image_generator before the failed twitter_tool call. Add `"no_image"` to its queue.
- The `Skip`, `TimedOut`, `all_candidates_fail`, and `invalid_config` tests don't reach image_generator. No queue update.
- `run_review_pipeline_pick_with_image_generator_attaches_head_image` (new in Step 5) explicitly puts `"[IMAGE:base64:iVBORw0KGgo=]"` in the queue.
- `run_review_pipeline_pick_image_generator_no_image_posts_text_only` (new) puts `"no_image"` in the queue.

Concretely, search the existing `run_review_pipeline_pick_index_0_posts_to_twitter` test and add `"no_image",` as the 5th entry of the `MockProvider::arc(vec![...])` (after `r#"{"verdict": "verified"}"#`). Same surgery for `run_review_pipeline_pick_twitter_api_error_returns_publish_failed`.

Updated test fixtures look like:

```rust
        let provider = MockProvider::arc(vec![
            "Research digest:\n- AI is moving fast",
            "concrete short post",
            r#"{"verdict": "pass", "style_match_score": 0.92}"#,
            r#"{"verdict": "verified"}"#,
            "no_image",                              // <-- NEW for P1.3f
        ]);
```

- [ ] **Step 7: Run all review tests**

```bash
cd /home/pleclech/projects/heartbit
cargo test -p heartbit-ghost --lib review 2>&1 | tail -10
```

Expected: 20 (P1.3d) + 3 (helper) + 2 (integration) = **25 passed**.

Plus pipeline + keyboard tests stay green:

```bash
cargo test -p heartbit-ghost --lib pipeline:: 2>&1 | tail -3
cargo test -p heartbit-telegram --lib keyboard 2>&1 | tail -3
```

- [ ] **Step 8: Workspace gate**

```bash
cargo fmt -- --check
cargo clippy --workspace --all-features -- -D warnings 2>&1 | tail -3
cargo test --workspace --all-features 2>&1 | grep -E "test result:" | awk '{ p+=$4; f+=$6 } END { print "Total: " p " passed, " f " failed" }'
```

Workspace count: 4016 (post-P1.3d) → ~4028 (+12 net).

- [ ] **Step 9: Commit**

```bash
cd /home/pleclech/projects/heartbit
git add crates/heartbit-ghost/src/review/mod.rs
git commit -m "$(cat <<'EOF'
feat(ghost): review — wire image_generator + extract_image_marker (P1.3f)

run_review_pipeline now runs image_generator after publish_gate
succeeds and BEFORE twitter_tool.execute. Failure non-blocking —
on error or "no_image", posts text-only.

- New `extract_image_marker(raw) -> Option<String>` helper:
  parses ImageGenerateTool's `[IMAGE:base64:<data>]` marker.
  Returns None for empty/whitespace input, "no_image" (case-insensitive),
  absent marker, or empty body between brackets.
- New `ReviewOutput.head_image_attached: bool` field. Set true iff
  image_generator returned a non-empty base64 marker AND the post
  ultimately succeeded.
- run_review_pipeline body: in the Pick → Ok(gate) branch,
  build image_generator AgentRunner via runner_from_recipe (with
  ImageGenerateTool), execute on chosen draft (system prompt:
  "Decide whether to attach an image..."), extract marker, pass
  head_image_b64 into twitter_tool's input JSON.
- The 3-tuple match shape (outcome, report, head_image_attached)
  replaces P1.3d's 2-tuple. Skip/TimedOut/GateRejected/PublishFailed
  arms always set head_image_attached = false.

Test infrastructure: MockTwitterTool gains `pub fn last_input(&self) -> Option<Value>`
(P1.3d's dead-store from Task 3 review now used). success/errored
factories return Arc<Self> so tests can both pass it as Arc<dyn Tool>
to the config AND inspect last_input. Existing P1.3d tests updated
mechanically (add `as Arc<dyn Tool>` at call sites; add `"no_image"`
to provider queues for tests that reach image_generator).

5 new tests: 3 helper (happy / no_image / absent-marker) + 2 integration
(Pick→image attached; Pick→no_image text-only). Plus 2 existing
P1.3d tests get a "no_image" response added to their queues.

Workspace 4016 → ~4028 (+12 net new across P1.3f).

Refs: docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.3f-image-attachment-design.md §3.3, §3.4
EOF
)"
```

---

## Task 4: Final acceptance + workspace quality gate + final review

**Why:** Confirm P1.3f meets every acceptance criterion. Verification only — no commit.

**Files:** none.

- [ ] **Step 1: Workspace quality gate**

```bash
cd /home/pleclech/projects/heartbit
cargo fmt -- --check && cargo clippy --workspace --all-features -- -D warnings && cargo test --workspace --all-features 2>&1 | tail -10
```

Expected: all green. Workspace test count: 4016 (post-P1.3d baseline) → ~4028 (+12 net new across all tasks).

- [ ] **Step 2: Verify the public surface is reachable**

```bash
cd /home/pleclech/projects/heartbit
cat > /tmp/p1_3f_surface_check.rs <<'EOF'
fn _check() {
    // ReviewOutput gains head_image_attached.
    use heartbit_ghost::review::{ReviewConfig, ReviewOutput, ReviewOutcome};
    // No new public-public types in review (extract_image_marker is
    // pub(crate); upload_image_chunked is pub(crate)). The only
    // user-visible change beyond `--review` behaviour is
    // ReviewOutput.head_image_attached.
    let _ = |o: ReviewOutput| -> bool { o.head_image_attached };
}
EOF
echo "(surface check is illustrative; cargo check covers it)"
rm -f /tmp/p1_3f_surface_check.rs
```

- [ ] **Step 3: Print final commit log for the branch**

```bash
git log --oneline main..feat/heartbit-ghost-p1.3f
```

Expected: 5 commits — spec + plan + 3 task commits.

- [ ] **Step 4: No commit for this task**

Task 4 is verification only. The branch is ready for final review + merge.

---

## Acceptance criteria

P1.3f is done when (per spec §8):

1. All public types compile cleanly under `cargo check -p heartbit-ghost --all-features` and `cargo check -p heartbit-cli --all-features`.
2. `cargo fmt -- --check && cargo clippy --workspace --all-features -- -D warnings && cargo test --workspace --all-features` all green.
3. ~12 net new tests pass: 3 client (chunked upload happy + INIT 4xx + APPEND 5xx) + 4 thread (input deserialize ± field, MIME sniff, oversize) + 3 review-mod helper (extract_image_marker variants) + 2 integration (image-attached / no_image).
4. New (semi-)public surface: `pub(crate) fn extract_image_marker`, `pub(crate) async fn upload_image_chunked`. Public additive change: `ReviewOutput.head_image_attached: bool`. Schema-additive change: `TwitterThreadTool` input gains `head_image_b64: Option<String>` field.
5. Live verification: `heartbit persona run heartbit-ghost:x --once "<topic>" --review` posts a thread with an image attached to the head tweet (when image_generator decides to attach) or a text-only thread (when it returns "no_image"). Telegram report message format unchanged.

## Out of scope (re-stated from spec §9)

- Per-tweet media (parallel-array schema for non-head tweet attachments) → P1.4
- Video / animated GIF upload (we accept GIF MIME but only because magic-byte sniffing is uniform; static-only flow today) → P1.4
- Multiple images on the head tweet (X allows up to 4) → P1.4 if needed
- Alt-text metadata via media metadata endpoint → P1.4
- Image generation in P1.3c direct mode actually attaching to tweets (direct mode prints to stdout, doesn't post to X — moot) → no change needed
- Reply-trigger flows posting images → P1.4

## Reference

- Spec: `docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.3f-image-attachment-design.md`
- P1.3d spec/plan (predecessor): `docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.3d-telegram-review-design.md`
- ImageGenerateTool: `crates/heartbit-core/src/tool/builtins/image_generate.rs` — emits `[IMAGE:base64:...]` markers via OpenRouter Gemini
- TwitterThreadTool today: `crates/heartbit-ghost/src/tools/thread.rs` — POST `/2/tweets` per tweet with `reply.in_reply_to_tweet_id` chaining
- XClient OAuth1.0a + post_json: `crates/heartbit-ghost/src/tools/client.rs`
- X v2 chunked media upload spec: `https://docs.x.com/x-api/media/quickstart/media-upload-chunked`
- POST /2/tweets media field: `https://docs.x.com/x-api/posts/manage-tweets/quickstart`
- v1.1 deprecation announcement (2025-03-31): `https://devcommunity.x.com/t/questions-about-https-api-twitter-com-2-media-upload-availability-and-functionality-on-free-tier/240226`
