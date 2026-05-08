# heartbit-ghost P1.3f — Image attachment to thread-head tweet

**Status:** approved 2026-05-08
**Predecessor:** P1.3d (Telegram review delivery + twitter_thread direct posting) merged to `main` at `c17ea3f`
**Branch:** `feat/heartbit-ghost-p1.3f` (created off `main`)
**Successor:** P1.3e — Persistent pick storage + reflection memory (originally next; this brings P1.4's image work forward)
**Brainstorming:** done inline in conversation (see "deep research" exchange leading to Option A — head-only attachment).

---

## 1. Goal

Generate an image to accompany the chosen draft and attach it to the **first tweet** of the thread when posting via `twitter_thread`. Skip cleanly when the `image_generator` recipe decides `"no_image"`. Subsequent tweets in the thread remain text-only.

After P1.3f, `heartbit persona run heartbit-ghost:x --once "<topic>" --review` produces a Telegram review → user picks → `image_generator` runs on the chosen draft → on success the head tweet posts with the image attached; on `"no_image"` the thread posts text-only as today.

This brings P1.4's deferred "image attachment to actual tweets" work forward, behind a clean schema-additive change to `TwitterThreadTool`.

## 2. Architecture

```
run_review_pipeline (review mode):
  ...candidate generation, dedup, Telegram review, pick handling...
  ↓
  publish_gate(chosen) ✓
  ↓
  image_generator agent on chosen draft       (NEW — P1.3a recipe; was unused in review mode)
  ↓
  parse output (literal "no_image" → None; else extract [IMAGE:base64:...] marker → Some(bytes))
  ↓
  twitter_tool.execute({"tweets": [...], "head_image_b64": <Option<String>>})
        │
        └─ TwitterThreadTool internal flow:
             ├─ if head_image_b64 is Some:
             │     ├─ base64-decode → Vec<u8>
             │     ├─ MIME sniff (PNG / JPEG / WebP / GIF) — first 16 bytes
             │     ├─ XClient::upload_image_chunked(bytes, mime_type) → media_id
             │     └─ first PostRequest gets `media: { media_ids: [media_id] }`
             └─ subsequent tweets in thread: text-only (in_reply_to chain, no media)
```

**Where the image data lives:** `ImageGenerateTool` already returns base64 in a marker `[IMAGE:base64:<data>]` (see `crates/heartbit-core/src/tool/builtins/image_generate.rs:9`). `run_review_pipeline` extracts the marker and passes the raw base64 string through. `TwitterThreadTool` does the decode + upload inside its `execute`, keeping the upload concern colocated with the post concern.

**Why head-only (not per-tweet):** `image_generator` produces ONE image; the natural attachment point is the first tweet (highest visibility, sets the visual context for the thread). Per-tweet media adds a parallel-array schema and an N×LLM cost to image_generator that doesn't pay off until P1.4 wants richer review previews. Defer.

## 3. Public API

### 3.1 `XClient` extensions (`crates/heartbit-ghost/src/tools/client.rs`)

```rust
impl XClient {
    /// Upload an image to X via the v2 chunked media upload endpoint.
    /// Returns the `media_id` to attach to a subsequent `POST /2/tweets`.
    ///
    /// Implements INIT → APPEND → FINALIZE for a single segment (one
    /// segment is fine for any image we'd reasonably attach; tweets cap
    /// at 5 MiB per image). For images >5 MiB we error early.
    ///
    /// # Errors
    /// - `XApiError::ApiError` on non-200 from any of the three commands
    /// - `XApiError::Network` on connect / TLS / timeout
    /// - `XApiError::ParseError` if the INIT response doesn't carry `data.id`
    pub(crate) async fn upload_image_chunked(
        &self,
        bytes: &[u8],
        mime_type: &str,  // "image/png" | "image/jpeg" | "image/webp" | "image/gif"
    ) -> Result<String, XApiError>;
}
```

Uses OAuth 1.0a User Context (already in use). Multipart helper added inline in `client.rs`. No new public API surface — `pub(crate)` only; tests live in same file.

### 3.2 `TwitterThreadTool` schema change (`crates/heartbit-ghost/src/tools/thread.rs`)

```rust
#[derive(Debug, Deserialize)]
struct ThreadInput {
    tweets: Vec<String>,
    /// Optional base64-encoded image bytes to attach to the FIRST tweet.
    /// MIME type is sniffed from the bytes (PNG/JPEG/WebP/GIF).
    /// When None, posts text-only (existing P1.3d behavior).
    #[serde(default)]
    head_image_b64: Option<String>,
}
```

`PostRequest` for the first tweet conditionally includes:

```rust
#[derive(Debug, Serialize)]
struct Media {
    media_ids: Vec<String>,
}

#[derive(Debug, Serialize)]
struct PostRequest<'a> {
    text: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    reply: Option<ReplyTo<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    media: Option<Media>,
}
```

`ToolDefinition.input_schema` JSON gains:
```json
"head_image_b64": {
    "type": "string",
    "description": "Optional base64-encoded image to attach to the first tweet (PNG/JPEG/WebP/GIF, ≤5 MiB)"
}
```

**Backward-compatible:** existing callers (which don't pass `head_image_b64`) continue to work.

### 3.3 `run_review_pipeline` integration (`crates/heartbit-ghost/src/review/mod.rs`)

After publish_gate passes, before twitter_tool.execute:

```rust
// Run image_generator (NEW — P1.3d skipped this).
progress("Generating optional image...");
let image_recipe = crate::agents::image_generator_recipe();
let image_tool: Vec<Arc<dyn Tool>> =
    vec![Arc::new(heartbit_core::tool::builtins::ImageGenerateTool::new())];
let image_runner = crate::pipeline::runner_from_recipe(
    cfg.provider.clone(), image_recipe, image_tool,
).map_err(|e| ReviewError::Pipeline(PipelineError::Builder {
    stage: "image_generator".to_string(), source: e,
}))?;

let image_msg = format!(
    "Approved draft:\n{}\n\n{}\n\nDecide whether to attach an image. \
     If no, output the literal string \"no_image\". \
     If yes, call image_generate with a concise visual prompt and return the result.",
    chosen.draft, voice_guidelines,
);
let head_image_b64: Option<String> = match image_runner.execute(&image_msg).await {
    Ok(out) => {
        total_usage += out.tokens_used;
        extract_image_marker(&out.result)
    }
    Err(e) => {
        progress(&format!("image_generator failed (non-blocking): {e}"));
        None
    }
};

// 10b. Post via twitter_tool (with optional head image).
let tweets = parse_thread_tweets(&chosen.draft);
let exec_ctx = heartbit_core::ExecutionContext {
    credentials: Some(cfg.credentials.clone()),
    ..Default::default()
};
let mut input = serde_json::json!({"tweets": tweets});
if let Some(b64) = head_image_b64.as_ref() {
    input["head_image_b64"] = serde_json::Value::String(b64.clone());
}
match cfg.twitter_tool.execute(&exec_ctx, input).await { /* ...same as P1.3d... */ }
```

New helper:

```rust
/// Extract the base64 image data from `image_generator`'s output.
/// Returns `None` for "no_image" (case-insensitive), empty input,
/// or absent marker. The marker prefix is `[IMAGE:base64:` and ends at `]`.
pub(crate) fn extract_image_marker(raw: &str) -> Option<String>;
```

### 3.4 `ReviewOutput` extension

```rust
pub struct ReviewOutput {
    // existing fields unchanged...
    /// Image attached to the head tweet, if `image_generator` produced
    /// one. `None` when the recipe returned `"no_image"`, when the call
    /// failed, or when the marker was absent.
    pub head_image_attached: bool,
}
```

Just a bool — the actual base64 isn't useful in the output (already posted to X). Bool tells callers whether image generation contributed to the final post.

`ReviewOutcome::Posted` does NOT change — the tweet_url still points to the head tweet, which now has the image attached on the X side.

## 4. Upload contract details

Per X v2 chunked media upload (researched 2026-05-08; v1.1 deprecated 2025-03-31):

**INIT:** `POST https://api.x.com/2/media/upload` with multipart form:
- `command=INIT`
- `media_type=<image/png|image/jpeg|image/webp|image/gif>`
- `total_bytes=<int>`
- `media_category=tweet_image` (or `tweet_gif` for animated GIF)

Response: `{"data": {"id": "<media_id>", "media_key": "<key>", "expires_after_secs": 86400}}`

**APPEND:** `POST https://api.x.com/2/media/upload` with multipart form:
- `command=APPEND`
- `media_id=<from INIT>`
- `segment_index=0`
- `media=<binary bytes>`

Response: status check (no required body fields for our flow).

**FINALIZE:** `POST https://api.x.com/2/media/upload` with multipart form:
- `command=FINALIZE`
- `media_id=<from INIT>`

Response: `{"data": {"id": "<media_id>", "processing_info": {...}}}` — for images, `processing_info.state` is `succeeded` synchronously; no STATUS poll needed.

**MIME sniffing** (no separate library — first-bytes magic-number check):
- PNG: starts with `\x89PNG\r\n\x1a\n` (8 bytes)
- JPEG: starts with `\xFF\xD8\xFF`
- GIF: starts with `GIF87a` or `GIF89a`
- WebP: bytes 0–3 = `RIFF`, bytes 8–11 = `WEBP`

Defaults to `image/png` if unknown (Gemini's image preview model returns PNG).

**Size limit:** 5 MiB hard cap before INIT (we error early to avoid wasting an uploaded segment that would FINALIZE-fail).

## 5. Error handling

`TwitterThreadTool` errors fall into three groups, all routed through the existing `ToolOutput::error(...)`:

| Cause | Surfaced as |
|---|---|
| Invalid base64 in `head_image_b64` | `ToolOutput::error("invalid head_image_b64: <reason>")` — fail-fast before any X API call |
| Image >5 MiB | `ToolOutput::error("head image exceeds 5 MiB (got <N> bytes)")` |
| Upload INIT/APPEND/FINALIZE returns 4xx/5xx | `format_error(&XApiError)` — same path as tweet creation today |
| Tweet creation succeeds for tweet 0 with media but fails for tweet 1 (text-only) | Same as P1.3d — partial thread posted; `is_error=true` with first error |

No new error types in `XApiError`. The chunked upload errors slot into `XApiError::ApiError { status, message }` via the existing `client.post_json`-equivalent path (a new `post_multipart` helper).

`run_review_pipeline`'s image_generator failure is **non-blocking** (matches `image_generator` integration in P1.3c direct mode): logs via `progress`, sets `head_image_b64 = None`, posts text-only.

## 6. Testing

| File | New tests | Coverage |
|---|---|---|
| `crates/heartbit-ghost/src/tools/client.rs` | +3 unit | INIT/APPEND/FINALIZE URL construction, OAuth signature on multipart, response parsing for `data.id` |
| `crates/heartbit-ghost/src/tools/thread.rs` | +4 unit | (1) ThreadInput deserializes with/without `head_image_b64`; (2) MIME sniffing — PNG / JPEG / WebP / GIF / unknown→png; (3) >5 MiB rejected before upload; (4) full happy-path with mock XClient: upload → media_id → first tweet POST has `media.media_ids`, subsequent tweets don't |
| `crates/heartbit-ghost/src/review/mod.rs` | +3 unit | `extract_image_marker`: happy path / no marker → None / "no_image" → None |
| `crates/heartbit-ghost/src/review/mod.rs` | +2 integration | (1) full happy path: image_generator returns base64 → twitter_tool receives `head_image_b64` and posts with media; (2) image_generator returns `no_image` → twitter_tool receives no `head_image_b64` and posts text-only |

Total: ~12 net new tests. Workspace 4016 → ~4028.

**Mock test infrastructure:** `MockTwitterTool` from P1.3d's review tests gains a way to inspect the input it received (the spec already noted `last_input` was captured but never read — this is the test that finally uses it). Tests assert `last_input["head_image_b64"]` is present/absent as expected.

The chunked-upload XClient tests use a mock HTTP server (the existing test pattern in `client.rs` for the post-tweet flow — `MockServer` from `wiremock` if already a dev-dep, else verify via `MockHttp` plumbing).

**Live verification:** same setup as P1.3d. After merge: re-run `heartbit persona run heartbit-ghost:x --once "<topic>" --review`. Pick a candidate. Tweet should appear on X with an image attached to the first tweet. The Telegram report message format is unchanged — only the actual posted content differs.

## 7. ADs (architecture decisions)

| AD | Decision | Reason |
|---|---|---|
| AD-1 | Head-only attachment (single image, first tweet) | image_generator produces one image; per-tweet media is unused complexity until P1.4 needs richer review previews |
| AD-2 | Schema-additive `head_image_b64: Option<String>` field on existing `TwitterThreadTool` (not a new tool) | Backward-compatible; one tool to maintain; image attachment is naturally part of "post a thread" |
| AD-3 | Decode + upload inside `TwitterThreadTool::execute` (not in caller) | Colocates the upload concern with the post concern; caller passes opaque base64 string |
| AD-4 | MIME sniff via magic-number check (no new dep) | The 4 supported MIME types have unambiguous magic bytes; rolling our own avoids `infer` / `mime_guess` dep additions |
| AD-5 | 5 MiB hard cap, error before INIT | X's actual limit; failing early avoids wasted upload bandwidth |
| AD-6 | Chunked upload (INIT/APPEND/FINALIZE), single segment | v1.1 single-shot deprecated 2025-03-31; chunked works for any size; one-segment uploads are trivially fast for ~1-3 MiB images |
| AD-7 | image_generator failure is non-blocking | Matches P1.3c direct mode; image is enhancement, not a hard requirement; never block a post over an image generation issue |
| AD-8 | New `pub(crate) fn extract_image_marker` in review mod | Shared between review-mode integration and tests; not a public API |
| AD-9 | `ReviewOutput.head_image_attached: bool` (just a bool, not the URL) | Image is on X already; bool tells caller whether image_generator contributed; the X CDN URL would require an extra API call |
| AD-10 | OAuth 1.0a (not OAuth 2.0 with `media.write` scope) | Existing `XClient` does OAuth 1.0a; X v2 media upload supports both; no auth migration needed |
| AD-11 | Always run image_generator after publish_gate (not before) | Saves an LLM call when gate rejects the draft; matches the cost-conscious design pattern from P1.3c |
| AD-12 | `head_image_b64` plumbed as JSON string (not bytes) through Tool input | `Tool::execute` takes `serde_json::Value`; bytes-as-base64-string is the standard JSON convention |

## 8. Acceptance criteria

P1.3f is done when:

1. All public types compile under `cargo check -p heartbit-ghost --all-features` and `cargo check -p heartbit-cli --all-features`.
2. `cargo fmt -- --check && cargo clippy --workspace --all-features -- -D warnings && cargo test --workspace --all-features` all green.
3. ~12 net new tests pass (3 client + 4 thread + 3 mod helpers + 2 integration).
4. New (semi-)public surface: `pub(crate) fn extract_image_marker`, `pub(crate) fn upload_image_chunked` (both internal to crates), plus the additive `head_image_b64` field on `TwitterThreadTool`'s input schema and the new `head_image_attached: bool` field on `ReviewOutput`.
5. `heartbit persona run heartbit-ghost:x --once "<topic>" --review`:
   - User picks a candidate
   - Pipeline runs `image_generator` after publish_gate; outcome non-blocking
   - On image generation success: head tweet posts with image attached (visible on X)
   - On image generation `"no_image"` or failure: thread posts text-only (existing P1.3d behavior)
   - Telegram report unchanged

## 9. Out of scope (deferred)

- Per-tweet media (parallel-array schema for attaching images to non-head tweets) → P1.4 if needed
- Video / animated GIF upload (we'll accept GIF MIME but only because magic-byte sniffing is uniform; the `tweet_image` category covers static images; animated GIF needs `tweet_gif` category and STATUS polling) → P1.4
- Multiple images on the head tweet (X allows up to 4) → P1.4 if needed; image_generator only produces one today
- Alt-text metadata via `POST /1.1/media/metadata/create` (or v2 equivalent) → P1.4
- Image generation in P1.3c direct mode posting an image (today direct mode generates but doesn't attach — moot since direct mode prints to stdout, doesn't post to X) → no change needed
- Reply-trigger flows posting images → P1.4

## 10. Reference

- P1.3d spec (predecessor): `docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.3d-telegram-review-design.md`
- ImageGenerateTool: `crates/heartbit-core/src/tool/builtins/image_generate.rs` — outputs `[IMAGE:base64:...]`-marked content via OpenRouter Gemini
- TwitterThreadTool today: `crates/heartbit-ghost/src/tools/thread.rs` — POST `/2/tweets` per tweet with `reply.in_reply_to_tweet_id` chaining
- XClient: `crates/heartbit-ghost/src/tools/client.rs` — OAuth 1.0a User Context, `post_json` helper
- X v2 chunked media upload: https://docs.x.com/x-api/media/quickstart/media-upload-chunked
- POST /2/tweets media field: https://docs.x.com/x-api/posts/manage-tweets/quickstart
- v1.1 deprecation announcement (2025-03-31): https://devcommunity.x.com/t/questions-about-https-api-twitter-com-2-media-upload-availability-and-functionality-on-free-tier/240226
- image_generator recipe (P1.3a): `crates/heartbit-ghost/src/agents/image_generator.rs`
