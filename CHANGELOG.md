# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

Eight commits since `v2026.607.1`, five of which change heartbit-core runtime
behavior (orchestrator, runner, set_scope, bash, builder prompts). Most are
hardening driven by live CRM-session traces (6a25ca5e, 6a25d21b, 6a25eb4d,
6a265efd).

### Added

- **Gate observability + hard advisor escalation + clarify question-quality**
  (`a2ff809`) — the 9 deterministic gates now emit
  `AgentEvent::GateFired { gate, reason }` so fired gates are auditable in the
  trace; after repeated consecutive failed builds, edit/write/patch are denied
  until the advisor is consulted (advisor-gated, resets per request); the
  clarify rule and intake gap prompt forbid re-asking facts the user already
  specified and require offered options to honor every stated constraint.
- **Orchestration nudge — refuse-once same-agent parallel fan-out** (`05ea2f6`)
  — `delegate_task` batches that pile 3+ tasks onto the SAME agent are refused
  once with actionable guidance (do it yourself / one sequential task); an
  identical retry dispatches, and both paths emit `GateFired` so retry-through
  is measurable. Entry prompt + parallel schema now state the real predicate:
  parallel only when there is no shared write target and no cross-task
  dependency.
- **CI: auto-publish heartbit-core to crates.io on tag** (`f3734f8`) — release
  workflow gains a `publish-crate` job: no-op until `CARGO_REGISTRY_TOKEN` is
  set, idempotent on already-published versions, scoped to heartbit-core only.

### Fixed

- **Sub-agents no longer stream through the parent `on_text`** (`fa20786`,
  core + TUI) — N parallel sub-agents sharing one un-attributed `on_text`
  merged token streams character-by-character. Restores the documented
  contract: sub-agents use `complete()`, only the orchestrator's own synthesis
  streams; sub-agent work stays visible via attributed tool-call events. TUI:
  a delegation auto-activates the roster and per-agent badges
  (`saw_delegation`), and dispatch notices count duplicates ("worker ×4").
- **Filesystem boundary stated in the workspace prompt** (`699d01b`) — the
  workspace hint now says file tools can ONLY access paths inside the
  workspace, outside paths (/tmp, home, /etc) are rejected, and
  temporary/scratch work belongs in an in-workspace `./scratch` subdirectory;
  stops the /tmp probe-and-thrash loop.
- **Bash cwd visibility, doom-loop hard-stop, write/scope clarity**
  (`9ecd8d6`) — bash surfaces its persistent cwd when it drifts from the
  workspace root; the doom-loop soft warning is now backed by a hard abort
  (`Error::DoomLoopAborted`) after 2 more ignored turns; `set_scope` resolves
  relative roots against the workspace; "File unchanged" results state
  emphatically that the write SUCCEEDED; new exit-127 (command-not-found)
  repair-hint class.
- **"Temporary directory" guidance reconciled with the workspace jail**
  (`5c7f319`) — three places told three different stories (one of them
  unsatisfiable: "/tmp, never inside the repo" while the file tools reject
  /tmp). All guidance now points at one destination: a gitignored `./scratch`
  subdirectory inside the workspace. The `set_scope` guard was inverted:
  outside scopes are refused once with the scratch redirect; relocating INTO
  the workspace is never refused. `resolve_path` rejections now carry the
  destination.
- **TUI: `/model` takes effect on the next message** (`bfe4782`) — the
  long-lived agent thread captured the model at spawn, so `/model` persisted
  but never applied until process restart. Model changes now respawn the agent
  (input-channel recreation, lazy respawn on the next message). Bonus: startup
  notice when `HEARTBIT_MODEL` env overrides the configured model.

### Notes

- The workspace version is still `2026.607.1` (already tagged and published —
  immutable); it must be bumped before the next tag so the release workflow's
  idempotency guard cannot silently no-op.

## [2026.607.1] - 2026-06-07 — request-intent harness, completion-loop gates, deep research + advisor, TUI streaming

Covers the 60 commits on `feat/tui-streaming-markdown` since `v2026.507.4`. The
headline is a **deterministic request-fidelity harness**: the framework now infers
the response mode a request calls for and enforces it, compensating for mid-tier
models' tendency to charge ahead on hedged/underspecified requests. All changes are
additive; the enforced quality gate (fmt + clippy `-D warnings` + tests) is green at
5206 tests / 0 failures. heartbit-core has no intra-workspace path dependencies and
is independently publishable.

### Added — request-intent router & mode contracts (`agent/router.rs`)

- `RequestRouter` with a 3-layer design: Layer 0 deterministic markers
  (force × completeness, FR conditional morphology), Layer 1 `fast`-role LLM
  classifier for the ambiguous residue, Layer 2 safe-default on low confidence.
- `RequestMode` (`Answer` / `Execute` / `Study` / `Clarify`), `RoutedMode`,
  `RouteSource`; `AgentEvent::RequestRouted { mode, source, confidence }`.
- Mode contracts enforced by tool masking (`ToolProfile::ReadOnly`) + an
  execution-deny backstop: STUDY/ANSWER are read-only and must end in a proposal;
  CLARIFY asks before mutating. User overrides: go-tokens force EXECUTE,
  `/mode answer|study|clarify|execute|auto` pins live; the model can never
  self-promote.
- Builder wiring: `AgentRunnerBuilder::request_router`,
  `OrchestratorBuilder::entry_request_router`. CLI `chat` inherits it
  (`HEARTBIT_REQUEST_ROUTER=0` to opt out).

### Added — completion-loop harness (judge-gated work to a finish)

- Runtime goal: `GoalSlot` + the `set_goal` tool install/replace acceptance
  criteria mid-run; an independent judge (`GoalCondition::with_per_criterion`)
  gates every natural stop; `OrchestratorBuilder::entry_goal_judge`, `/goal`.
- Scope guard: `ScopeGuard` + the `set_scope` tool bound the blast radius
  (`SetScopeTool::with_workspace` refuses an outside→inside workspace relocation).
- `TodoItem.acceptance` per-loop done-conditions, recited at the context tail.
- Front-half `intake` recipe (acceptance-criteria extraction + gap elicitation).
- The structured `question` tool/`OnQuestion` channel is now reachable from the
  TUI (options modal, single/multi-select).

### Added — deterministic stop/dispatch gates (`agent/runner.rs`)

- Ask-gate (prose question battery → the structured `question` tool), act-gate
  (announced-then-stalled → execute or ask), plan-gate (building without a plan
  artifact → blocked pre-execution), repair hints (stale-API / type-mismatch /
  ownership classes + `cargo add` nudge), advisor escalation after consecutive
  failed builds, and the `DelegationNudge` (entry agent reminded to use the squad).
- Context-overflow defense in depth: grep/glob skip build dirs + byte caps, a
  default-on per-result ingestion cap (window-clamped), estimate-aware proactive
  compaction, bounded summarization, and deterministic reactive recovery.

### Added — workflows, deep research & advisor

- `deep_research` workflow recipe (plan → tooled angles → verify → cited synthesis)
  with a tolerant angle parser; `run_workflow` gains resume from a content-addressed
  journal and a `budget` arg; worktree isolation for flow leaves.
- Advisor mode — a frontier-model reviewer over the full transcript
  (`SessionHandoffTool` for purpose-tailored session bridges).
- Model roles wired end-to-end (per-call model via host `ProviderFactory`;
  `deep_research` plans on `fast`).
- MCP request timeout via `HEARTBIT_MCP_TIMEOUT_SECS`; absolute paths inside the
  workspace accepted; more actionable `form_squad` errors.

### Added — TUI (`heartbit-tui`)

- Live streaming Markdown; a splash screen; `/research`, `/workflows`, `/stats`
  (styled card), `/model advisor`, the bare `/mode` picker; "advised by <model>" in
  the status line; YOLO default mode; persistent per-directory prompt history.

### Notes

- The "heartbit-ghost P1.1 — X (Twitter) tool family" section under `2026.507.4`
  below is the detail of the X-tool family that shipped as part of that release;
  retained for history.

## [2026.507.4] - 2026-05-26 — heartbit-ghost P1.x integration + heartbit-core boundary cleanup

Covers everything since `v2026.507.3` (commit `ca994e6`, published 2026-05-07
but never tagged in git — tagged retroactively today). The release pulls in
heartbit-ghost's P1.1-P1.7 work that landed on `main` over the past three
weeks plus a deliberate `heartbit-core` boundary cleanup that makes the crate
shippable as a SOTA Rust agentic framework via `default-features = false`.

### Added (heartbit-core)

- **`ghost-domain-config` cargo feature** (default on) — gates the
  `heartbit-ghost` domain leaks (X persona configs, `ImageSource` enum,
  `TwitterPostTool`/`TwitterCredentials`) so SOTA framework users can
  depend on `heartbit-core` with `default-features = false` and get a
  pure agent-framework surface (~95 tests / ~1500 LoC stay behind the
  flag). Mirrors the pattern langchain-rust uses for vendor backends.
- **`Tool::redact_for_history` trait method** + `AgentOutput.tool_call_results`
  — generic fix for multimodal-blob context bloat (P1.3g). Lets tools
  swap large base64/binary payloads for a short SHA-256 placeholder in
  conversation history without losing the original tool result.
- **`TopicContextProvider` trait** + 2 persona impls (`HeartbitRsXTopicContext`,
  `XGhostTopicContext`). Pluggable seam for per-persona topic-context wiring.
- **`PersonaExpansion::mode_addendum` field** — optional system-prompt
  addendum for sub-mode scoping (e.g. a persona that posts generally vs.
  one focused on a specific topic cluster).
- Persona-specific config types under `ghost-domain-config` (gated):
  `PersonaPostsConfig`, `PersonaQuotesConfig`, `PersonaBlogConfig` +
  `XAnnounceConfig` + `GithubReadmeConfig`, `PersonaMentionsConfig`,
  `ImageSource` enum (`Online` / `Ai` / `None`).
- `TwitterPostTool` optional `media_url` + `media_alt_text` (gated under
  `ghost-domain-config`). Backward-compatible text-only path unchanged.

### Changed (heartbit-core)

- **`#![deny(missing_docs)]` is now genuinely enforced.** Three inner-attr
  escape hatches (`#![allow(missing_docs)]` in `config/daemon.rs`,
  `tool/builtins/mod.rs`, `tool/builtins/twitter_post.rs`) silently
  neutralized the gate across ~40 files in the crate. Dropping them
  surfaced ~410 missing-docs items, all filled with one-line rustdoc.
  Zero remaining module-level allows in the crate.
- **Cascade `__respond__` escalation fix** — when a non-final tier returns
  plain text where a `__respond__` tool call was expected (structured
  output mode), the cascade now correctly escalates to the next tier
  instead of accepting the unparseable response.
- `OpenverseImageSearchTool` and `Persona*Config` types are no longer
  unconditionally re-exported at the crate root; they live under the
  `ghost-domain-config` feature.
- `PersonaPostsConfig` gains `writer_provider` (per-persona engagement-voice
  override), `interval_jitter_pct` (anti-bot cadence), `image_source`
  toggle (online/ai/none).

### Removed (heartbit-core)

- `OpenverseImageSearchTool` (the CC0/public-domain Openverse image-search
  builtin added earlier) **moved to `heartbit-ghost`** — its sole consumer
  is the ghost X-post review pipeline. SOTA Rust frameworks (rig, swiftide,
  autoagents) keep domain-specific tools out of core; this matches that
  precedent.
- "Evangelism framing" example phrasing removed from
  `PersonaExpansion::mode_addendum` doc — replaced with a domain-neutral
  sub-mode-scoping description. The trait itself is unchanged; only the
  docstring leaked the heartbit-ghost use case.
- Empty `TriggerSpec {}` and `ReviewSpec {}` placeholder enums in
  `persona/types.rs` were kept (they're referenced by `PersonaExpansion`)
  but now carry explicit placeholder rustdoc.

### Breaking changes (heartbit-core)

- `Tool::execute(&self, ctx: &ExecutionContext, input: Value)` — was
  shipped in `v2026.507.3` (commit `93ba0e5`). External `Tool`
  implementors need to add the `&ExecutionContext` parameter. The
  `ExecutionContext` provides `credentials: Option<Arc<dyn CredentialResolver>>`
  and `audit: Option<Arc<dyn AuditSink>>` per-request.
- **`AgentOutput`, `PersonaExpansion`, and `DaemonConfig` are now
  `#[non_exhaustive]`.** Construction via struct literal from outside
  `heartbit-core` is no longer permitted. Internal callers (heartbit,
  heartbit-cli, heartbit-ghost) are already migrated. External
  consumers should use `Type::default()` then mutate the fields they
  care about, or — for `AgentOutput` specifically — read it from
  `runner.execute(...)` rather than construct manually. The structs
  now all `#[derive(Default)]`, making this idiomatic. Trade-off
  this release for future-proofing: subsequent field additions to
  these types are non-breaking.
- The cargo feature gate is itself additive — internal callers depend
  on `heartbit-core` with default features and see zero surface change.
  SOTA users opting out via `default-features = false` get a NARROWER
  surface (no `Persona*Config`, no `TwitterPostTool`, no `ImageSource`),
  which is the entire point of the release.

### heartbit-ghost (companion changes — not on crates.io)

The persona-specific crate gained: blog feature (Cloudflare Pages
deploy via `deploy_command` hook, GitHub README auto-update, weekly
X-derived blog seed selection, full LLM essay pipeline with strict
sourcing chain), quote/reply pipelines with bot-signature guards,
`PersonaMentionsConfig` + mention-poll pipeline, `MentionPoll` handler,
`PersonaQuotesConfig` + quote-tweet scheduler, X-announce thread for
blog publishes, OpenverseImageSearchTool now lives here. Not published
to crates.io.

### Notes

- **Internal callers see zero surface change**: `heartbit` (umbrella) and
  `heartbit-cli` depend on `heartbit-core` with default features on, so
  every prior API path keeps working. Operator TOML format (`daemon-dev.toml`)
  is unchanged — `cargo run -- daemon --validate-config` passes.
- **SOTA-user verification**: `cargo build --package heartbit-core --no-default-features`
  builds clean; `cargo test --lib --no-default-features` produces 2353
  passing tests (vs 2448 with default features — the 95-test delta is
  the `ghost-domain-config`-gated material).
- The `v2026.507.3` git tag was created retroactively on commit `ca994e6`
  (matches the crates.io publish on 2026-05-07).

### heartbit-ghost P1.1 — X (Twitter) tool family (shipped within 2026.507.4)


The P1.1 increment ships the X (Twitter) tool family on top of the Phase 0
`ExecutionContext` / `CredentialResolver` foundation. Five new tools live
in `heartbit-ghost`, plus a backward-compatible media + alt-text extension
on the existing `heartbit-core` `TwitterPostTool`. All HTTP interaction is
covered by `wiremock`-stubbed tests; no live network calls in CI.

### Added (heartbit-ghost P1.1)

- `TwitterUserTool` — `GET /2/users/by/username/:handle`. Returns id, name,
  description, follower/following/tweet counts.
- `TwitterSearchTool` — `GET /2/tweets/search/recent`. Returns matching
  tweets + `next_token` for pagination.
- `TwitterMentionsTool` — `GET /2/users/:id/mentions`. Returns mentions +
  pagination.
- `TwitterReplyTool` — `POST /2/tweets` (with `reply.in_reply_to_tweet_id`).
  Validates ≤280 chars.
- `TwitterThreadTool` — `POST /2/tweets` ×N, chained via `in_reply_to`.
  1..=25 entries; fail-fast on first error (X has no rollback API; tweets
  posted before the failure stay live).
- Shared `XClient` infrastructure: OAuth1 signing, credential resolution
  from `ExecutionContext::credentials`, typed `XApiError`, response parsing.
- Stable credential resolver names: `X_CONSUMER_KEY`, `X_CONSUMER_SECRET`,
  `X_ACCESS_TOKEN`, `X_ACCESS_TOKEN_SECRET`.

### Changed (heartbit-core)

- `TwitterPostTool` now accepts optional `media_url` (HTTPS, ≤5 MB image)
  and `media_alt_text` (≤1000 chars). Backward-compatible: text-only
  callers see no change. Test-only constructor `new_with_base_urls` added
  for wiremock injection.

### Notes (heartbit-ghost P1.1)

- New tools use the resolver-based credential model (per-tenant ready);
  existing `twitter_post` keeps construction-time `TwitterCredentials` for
  backward compatibility. Persona wiring (P1.3) will switch the persona's
  `twitter_post` instance to the resolver pattern.
- All new tests use `wiremock` for HTTP stubbing; no live network calls
  in CI.
- `twitter_thread` errors do not include partial-thread state in P1.1
  (e.g., the list of tweets posted before the failure). Operators should
  inspect the X timeline. A future API-shape extension could surface this
  for orchestrators that need it.

## [2026.507.3] - 2026-05-07 — heartbit Foundation Phase 0

The Phase 0 foundation lands the cross-cutting plumbing that the persona
work (heartbit-ghost, etc.) requires before any concrete persona crate can
be wired in. Nine tasks, executed on `feat/heartbit-foundation-phase-0`:
`ExecutionContext` threading through the `Tool` trait, the
`PersonaRegistry` and `[[persona]]` config block (empty registry — concrete
personas land in Phase 1), and the `heartbit persona` CLI surface.

### Breaking

- **`Tool::execute` signature change.** `Tool::execute` now takes
  `&heartbit_core::ExecutionContext` as its first argument, before the
  existing `serde_json::Value` input. All in-tree tools have been migrated.
  External consumers implementing `impl Tool` for their own types must add
  `_ctx: &heartbit_core::ExecutionContext` as the first parameter of their
  `execute()` method. The `#[heartbit_tool]` proc-macro emits the new
  signature automatically — no caller changes needed for macro-generated
  tools.

### Added

- **`heartbit_core::ExecutionContext`** — request-scoped value type carrying
  per-request `tenant_id`, `user_id`, `workspace`, a `credentials` resolver,
  and an `audit_sink`. Constructed once per agent turn and passed through
  to every tool's `execute()` call so multi-tenant secrets and audit
  logging hang off a single object. `AgentRunnerBuilder::audit_user_context`
  (and friends on `OrchestratorBuilder`) populate `tenant_id` / `user_id`;
  the daemon's per-task `build_runner` closure threads request identity
  through unchanged.
- **`heartbit_core::CredentialResolver`** trait + `Secret` newtype — the
  contract for tenant-scoped secret resolution. Phase 0 ships the trait
  shape; concrete env-glob and KMS resolvers land with their consumers.
- **`heartbit_core::AuditSink`** trait — per-tool audit logging hook.
  Phase 0 ships the trait; the existing `audit_trail` plumbing on the
  agent runner remains the active sink.
- **`heartbit_core::PersonaRegistry`** + `Persona` trait — registry surface
  for persona recipes. The registry is **empty in this release**;
  concrete persona implementations (heartbit-ghost X agent, etc.) land in
  Phase 1.
- **`[[persona]]` block on `HeartbitConfig`** — declarative persona
  instances in `heartbit.toml` / `daemon.toml`. Validation is **lexical
  only** in Phase 0: `recipe` must parse as `<crate>:<name>` and per-file
  instance names must be unique. Registry lookup is deferred to daemon
  startup once persona crates load. Tightened `overrides: toml::Table`
  (was `toml::Value`) so a misuse like `overrides = "string"` fails at
  deserialize-time.
- **`heartbit persona <subcommand>`** CLI surface — functional shells
  (`list`, `show`, `run`, `corpus {add, list}`, `profile {rebuild, diff}`,
  `phase`, `pause`, `resume`, `export-preferences`, `audit`) operating
  against the empty registry. Wires up cleanly so Phase 1 persona crates
  light up the surface without further CLI work.

### Phase 1 follow-ups

Four production sites currently pass `&ExecutionContext::default()` (or
hardcode `tenant_id: None, user_id: None`) and are tagged with
`TODO(phase-1):` markers. They have no functional impact in Phase 0 (no
persona uses tenant-scoped secrets yet) but must be addressed when
concrete personas land:

- `crates/heartbit-core/src/tool/mcp_server.rs` — MCP server tool
  dispatch; the JSON-RPC envelope doesn't carry tenant identity. Phase 1
  derives identity from the MCP session / `clientInfo`.
- `crates/heartbit-sensors/src/sources/mcp.rs` (poll loop) — sensor
  MCP fan-out has no per-tenant request identity to thread; Phase 1
  sensor-owner identity work plumbs it through.
- `crates/heartbit-sensors/src/sources/mcp.rs` (enrich loop) — same
  rationale as the poll loop.
- `crates/heartbit/src/workflow/agent_workflow.rs` — Restate workflow
  caller hardcodes `tenant_id: None, user_id: None` on `ToolCallRequest`
  because `AgentTask` doesn't yet carry caller identity. Activity-side
  (`workflow/agent_service.rs`) already constructs `ExecutionContext`
  from the request fields, so Phase 1 only needs to extend `AgentTask`
  and thread the values from the caller.

The canonical table lives in
`docs/superpowers/specs/2026-05-07-heartbit-foundation-design.md` under
"Phase 1 follow-ups (`TODO(phase-1)` markers)"; this list mirrors it.
A workspace-wide `grep -rn "TODO(phase-1)" crates/` finds the full set.

## [2026.507.2] - 2026-05-07

### Performance — perf cycle 2 (full-workspace sweep + BM25 opt-in)

A second-pass perf audit at `tasks/performance-audit-v2-2026-05-07.md`
(51 perf findings + 14 bench gaps) drove this release. Cycle 2 lands
five atomic commits: daemon and edge-crate `parking_lot` adoption (T2
echo from cycle 1's heartbit-core sweep), per-event JSON buffer pool
on the daemon Kafka publish path, the long-missing agent-ReAct-turn
benchmark, and the audit's deferred headline `P-MEM-2` BM25 opt-in
inverted index.

#### New benches

- **`agent_react_turn`** ([`b458016`](https://github.com/heartbit-ai/heartbit/commit/b458016)) — `AgentRunner::execute()` against a
  mock provider returning a canned `EndTurn` response. The audit's
  "single most embarrassing" gap is now closed: the per-execute hot
  path is **measured at 1.88 µs**, providing ground truth for any
  future P-RUNNER work. Behind the existing `bench-internals` feature.

#### Selective-query speedup (Phase 8 opt-in)

| Bench | Substring (default) | `exact_words: true` | Speedup |
|---|---|---|---|
| `memory_recall/.../selective@10k` | 3.43 ms | **0.73 ms** | **4.7×** |
| `memory_recall/.../selective@1k`  | 295 µs | **65 µs** | **4.6×** |

Broad queries (where every entry contains every query token) incur
the full BM25 cost on both paths — `exact_words` is opt-in for the
selective-query regime.

### Added

- **`MemoryQuery::exact_words: bool`** ([`653c076`](https://github.com/heartbit-ai/heartbit/commit/653c076)) — opt-in flag
  for the BM25 inverted-index fast path on `InMemoryStore`. When
  `true`, recall reads from a sibling `inverted: HashMap<token,
  HashSet<entry_id>>` map (built at store time, maintained in
  lock-step with `entries` and `tokens` on every store / update /
  forget / prune / cap-eviction). Default is `false`; the legacy
  substring `word.contains(token)` semantics are preserved unchanged.
  Trade-off: `exact_words: true` does not match query tokens that
  are *prefixes* of indexed words (`"perf"` won't match
  `"performance"`); opt in only when the caller controls the query
  vocabulary. Field doc-comment calls this out explicitly.
- **`crates/heartbit-core/benches/agent_react_turn.rs`** ([`b458016`](https://github.com/heartbit-ai/heartbit/commit/b458016)) —
  end-to-end criterion bench for `AgentRunner::execute()`. Mocked
  provider lives in `__bench::BenchMockProvider` (always returns the
  same canned response, no draining unlike the test-only
  `MockProvider`).

### Changed

- **Daemon `parking_lot::RwLock` adoption** ([`e3f7fe4`](https://github.com/heartbit-ai/heartbit/commit/e3f7fe4)) — `event_channels`
  + `task_cancels` (`daemon/core.rs`), in-memory task store
  `tasks` + `order` (`daemon/store.rs`), todo cache
  (`daemon/todo.rs`), and JWKS cache (`auth/jwt.rs`). Drops 14
  `expect("...lock poisoned")` chains; `parking_lot` returns guards
  directly. `parking_lot = { workspace = true }` added to the
  `heartbit` umbrella's runtime deps.
- **Daemon per-event JSON buffer pool** ([`193bbd6`](https://github.com/heartbit-ai/heartbit/commit/193bbd6)) — the `on_event`
  closure now uses a per-task `Mutex<Vec<u8>>` and
  `serde_json::to_writer` instead of allocating a fresh `Vec` on
  every emitted agent event. Eliminates 50–500 alloc/free cycles
  per task (P-V2-DAEMON-1, audit's headline daemon Critical).
- **Daemon Kafka-key reuse + broadcast gating** ([`e3f7fe4`](https://github.com/heartbit-ai/heartbit/commit/e3f7fe4)) — pre-compute
  the per-task `Arc<str>` Kafka key once in the consumer loop
  instead of `id.to_string()` per event (P-V2-DAEMON-9). Skip the
  broadcast `event.clone()` when no SSE subscribers are attached
  (P-V2-DAEMON-7); Kafka publish is unaffected.
- **Sensors / Telegram `parking_lot` adoption** ([`5001d92`](https://github.com/heartbit-ai/heartbit/commit/5001d92)) —
  `StoryCorrelator` Mutex (sensor triage hot path,
  P-V2-EDGE-4); Telegram `bridge.rs` double `pending` +
  `question_options` RwLocks (every callback,
  P-V2-EDGE-7-Critical); `ChatSessionMap` in
  `telegram/router.rs`. `parking_lot` added to the
  `heartbit-sensors` and `heartbit-telegram` deps.

### Notes

- Quality gates throughout: 3669 workspace tests passing,
  `cargo fmt -- --check` clean, `cargo clippy --workspace
  --all-targets --features heartbit-core/bench-internals
  -- -D warnings` clean.
- No security regressions: F-* findings from the security cycle
  remain intact. F-AUTH-5 nonce binding on telegram `pending`,
  F-MEM-3 capacity eviction (now also de-indexes from inverted
  index on victim removal), JWT cache TTL, sensor correlation
  semantics — all preserved.
- Cycle 2 followups still tracked at `tasks/performance-audit-v2
  -2026-05-07.md`: P-V2-DAEMON-10 rolling stats aggregates,
  P-V2-DAEMON-11 idempotency secondary index, P-V2-EDGE-2 sensor
  entity-list `Arc<[String]>`, and the remaining missing benches
  (MCP roundtrip, daemon Kafka dispatch, channel WebSocket
  bridge).

## [2026.507.1] - 2026-05-07

### Performance — Phase 2c (memory subsystem stepping stone)

Continues the perf cycle from `tasks/performance-audit-heartbit-core
-2026-05-06.md`. Adds the `tokens` side cache to `InMemoryStore` —
the recall hot path no longer pays the per-entry `to_lowercase()` +
`split_whitespace()` cost; that work moves to store/update time and
is reused on every recall.

### Changed

- **`InMemoryStore::tokens` side cache** ([`a528e4c`](https://github.com/heartbit-ai/heartbit/commit/a528e4c)) — sibling
  `parking_lot::RwLock<HashMap<String, EntryTokens>>` map maintained
  in lock-step with `entries`. `EntryTokens` carries
  `lower_content: String`, `content_words: Vec<String>`, and
  `lower_keywords: Vec<String>`, populated on every `store` / `update`
  and removed on `forget` / `prune` / cap-eviction. The recall hot
  path reads from the cache and feeds the new
  `bm25::bm25_score_pre(&[String], &[String], ...)` directly,
  eliminating up to ~140k transient `String` allocations per recall
  at N=10k. Lock order is `entries` → `tokens` everywhere; both are
  `parking_lot::RwLock` and never held across `.await`.

### Added

- **`bm25::bm25_score_pre(&[String], &[String], &[String], f64, f64,
  f64)`** — public pre-tokenised variant of `bm25_score`; accepts
  already-lowercased content words and keywords. The legacy
  `bm25_score(&str, &[String], ...)` now delegates to this new
  variant after performing the lowercase + tokenise itself, so all
  existing callers behave identically.

### Bench deltas (cumulative since the audit baseline)

| Bench | Baseline | This release | Cumulative delta |
|---|---|---|---|
| `memory_recall/text_query_top10@10k` | 19.8 ms | **12.69 ms** | **−36%** (−5% vs 506.2) |
| `memory_recall/text_query_top10@1k` | 1.78 ms | **~0.5 ms** | **−72%** |
| `memory_recall/agent_filter_top10@10k` | 3.20 ms | **2.02 ms** | **−37%** (−9% vs 506.2) |
| `memory_recall/agent_filter_top10@1k` | 308 µs | **187 µs** | **−39%** (−10% vs 506.2) |
| `sse_parse/feed_16kb_one_shot` | 11.3 µs | **7.0 µs** | **−38.5%** (unchanged from 506.2) |

### Notes

- The headline `P-MEM-2` BM25 inverted index (audit-predicted
  <2 ms text@10k) remains deferred. Reaching it requires switching
  from substring-match (`word.contains(token)`) to exact-word match
  so a `HashMap<token, Vec<entry_id>>` index can short-circuit the
  candidate set — a semantic change beyond this cycle's scope.
  Preserving substring semantics would need a trigram/suffix index
  (much larger storage) or an opt-in fallback path.
- Quality gates: 3669 workspace tests green (incl. 214 memory tests).
  `cargo fmt -- --check` clean; `cargo clippy --workspace
  --all-targets --features heartbit-core/bench-internals
  -- -D warnings` clean.
- No security regressions: F-* findings from the security cycle
  remain intact.

## [2026.506.2] - 2026-05-06

### Performance — heartbit-core deep perf audit + first remediation cycle

A 113-finding static perf audit landed at
`tasks/performance-audit-heartbit-core-2026-05-06.md` (7 sub-reports
under `tasks/perf-audit-*.md`). This release lands the first
remediation cycle: 5 atomic perf commits, validated by criterion
benches scaffolded under `crates/heartbit-core/benches/`.

#### Bench deltas (this release vs the audit baseline)

| Bench | Baseline | This release | Delta |
|---|---|---|---|
| `sse_parse/feed_16kb_one_shot` | 11.3 µs | **7.0 µs** | **−38.5%** (620 → 980 MiB/s) |
| `sse_parse/feed_4kb_chunks` | 10.1 µs | **7.5 µs** | −27.5% |
| `memory_recall/text_query_top10@10k` | 19.8 ms | **13.25 ms** | **−37%** |
| `memory_recall/agent_filter_top10@10k` | 3.20 ms | **2.22 ms** | −28% |
| `memory_recall/agent_filter_top10@1k` | 308 µs | **207 µs** | −32% |

#### Phase 1 — drop-in workspace sweeps (`891d80b`)

Frequency-multiplied micro-wins across the hot paths. Static-only,
~10+ fixes:
- **T1 — `LazyLock<Regex>`** for `redact_idp_body` (3 patterns,
  ~500–800 µs / IdP-error call), `sanitize_html_for_agent` (3 HTML
  patterns, ~100–200 µs / fetch), and `list.rs` `DEFAULT_IGNORES`
  (~200–500 µs / list call).
- **T2 — `parking_lot::RwLock` / `Mutex`** on every hot non-await
  lock: `InMemoryAuditTrail`, `permission_rules` on `AgentRunner`,
  `TenantTokenTracker`, `ResponseCache`, `ActionBudgetGuardrail.counts`,
  `CircuitTracker.circuits`, `InMemorySessionStore.sessions`,
  `McpServer.sessions`, `FileTracker.records`,
  `ReflectionTracker.accumulated`. Drops the `expect("...lock
  poisoned")` chains; ~2× faster uncontended reads, no fairness
  pile-ups under heavy concurrent traffic.
- **P-TOOL-5 / P-TOOL-14** patch.rs short-circuit ladder for
  `fuzzy_lines_match` (exact → trim_end → trim_both → unicode-
  normalise) with an ASCII fast-path that skips the unicode pass
  entirely when both sides are pure ASCII. ~50–200 µs / patch call.
- **P-TOOL-10** bash cwd marker uses a process-startup `LazyLock<
  String>` UUID base + `AtomicU64` counter instead of a fresh
  `Uuid::new_v4()` per spawn. F-FS-8 forge resistance preserved
  (attacker still can't observe the per-process random base).
  ~5–50 µs / bash call.
- **P-CROSS-7** `audit::strip_content_owned(Value) -> Value`
  consumes the payload by ownership and walks the tree without
  cloning preserved scalars. The runner uses `mem::take(&mut record.
  payload)` + the owned variant. ~1 ms / record on 100 KB payloads.
  F-AUTH-3 allow-list semantics identical.

#### Phase 2a — memory `parking_lot::RwLock` swap (`0828700`)

Drop-in lock swap on `InMemoryStore`. Six call sites cleaned up; no
single-threaded delta on the bench (the lock isn't the bottleneck
single-threaded), but ~2–3× faster acquisition under contention.

#### Phase 2b — defer `MemoryEntry` clone after limit (`8a47256`)

The recall pipeline now collects candidate entries as
`Vec<&MemoryEntry>` and carries references through filter / BM25 /
sort / truncate / graph-expansion / re-sort, only cloning the
surviving top-K **after** the limit has been applied. At N=10k
that's ~5 MB of `MemoryEntry` allocation eliminated per recall.

Bonus refactors in the same commit:
- **Filter ordering**: cheap field comparisons before the expensive
  `to_lowercase()` text scan.
- **Composite-cache via pair-and-sort**: `effective_strength` runs
  once per entry instead of twice per `sort_by` comparison
  (~280k redundant `exp()` calls eliminated at N=10k).
- **`HashMap<&str, f64>`** for `bm25_map` / `relevance_map` — keys
  are slices into entries, zero String-key allocations during
  scoring.
- **Graph expansion** scores related entries inline against the
  shared `avgdl` / `max_bm25`, eliminating duplicate `new_bm25_map`.

#### Phase 3 — SSE parser zero-copy (`74d315d`)

Eliminates two per-event allocation hotspots in
`crates/heartbit-core/src/llm/anthropic.rs` `SseParser`:
- **P-LLM-2 — zero-copy line scan**: `feed()` `mem::take`s the
  buffer once at the top of the chunk and processes each line as a
  `&str` slice via the new `next_line_boundary` free helper, instead
  of allocating a fresh `String` per line via `next_line().to_string()`.
- **P-LLM-14 — single `data` buffer**: `data_lines: Vec<String>`
  collapsed to `data: String` with an explicit `\n` separator
  inserted before each non-first `data:` line. `emit_event()` no
  longer joins per dispatch and skips `mem::take` entirely on empty
  events. Cap accounting (F-LLM-3) still applies.

A `bench-internals` cargo feature gates a `__bench` module that
exposes a thin wrapper over `SseParser` for the `sse_parse`
criterion bench. Downstream consumers cannot accidentally enable it.

#### Audit infrastructure (`78b911d`, `eedd435`)

- `crates/heartbit-core/benches/` — three criterion benches
  (`memory_recall`, `guardrail_pii`, `sse_parse`), `criterion`
  added as a dev-dependency only.
- `tasks/performance-audit-heartbit-core-2026-05-06.md` — master
  report with 113 findings ranked by `(execution frequency) ×
  (per-call overhead)`, security-regression cross-check against
  the 78-finding security audit, recommended 3-PR phasing.

### Fixed

- **gh#9 — `KeywordRoutingStrategy` Tier 1 short-circuit + step /
  delegation patterns** (`9c17a8c`):
  - **#A** — Tier 1 now picks the best-covering agent from
    `domain_signals` instead of hardcoding `agent_index: 0`. Helper
    `best_covering_agent(task_domains, agents)` returns the index
    with the most overlap; ties break on registration order; falls
    back to 0 when no agent matches any detected domain (preserves
    the empty-domain semantics).
  - **#A.2** — `DELEGATION_PHRASES` extended with `"ask the "`,
    `"have the "`, `"use the "`, `"tell the "`, `"instruct the "`,
    `"let the "`, `"get the "`. False-positive cost bounded
    (a 0.30 score boost only ever moves a task into Tier 2
    capability matching, never into unsolicited orchestration).
  - **#B** — new `is_numbered_step_marker(word)` recognises
    `1.`, `(1)`, `1)`, and the same patterns followed by `;` /
    `,` / `:` so the gh#9 repro `(1); (2); (3); ...` now reports
    `step_markers >= 4`.

### Notes

- Recommended Phase 2c (BM25 inverted index, est. drop the residual
  13.25 ms text@10k recall toward <2 ms) is deferred to a follow-up
  release. Tracked in the audit report; would require a side cache
  maintained on `store` / `forget` / `update` with careful handling
  of substring-match semantics in the BM25 inner loop.
- Quality gates throughout: 3669 workspace tests passing
  (2397 heartbit-core + 454 umbrella + 65 CLI + others). `cargo fmt
  -- --check` clean; `cargo clippy --workspace --all-targets
  --features heartbit-core/bench-internals -- -D warnings` clean.
- No security regressions: the 30+ "obvious wins" rejected during
  the audit (cross-tenant caching, lifted DoS caps, removed nonce
  markers, `==` on auth tokens, etc.) remain rejected. F-* findings
  from the security cycle are all intact.

## [2026.506.1] - 2026-05-06

### Security — heartbit-core Deep Audit

A 78-finding deep security audit (`tasks/security-audit-heartbit-core-2026-05-06.md`)
was conducted against `heartbit-core` across all attack surfaces (LLM,
MCP, A2A, filesystem, network, auth, agent, memory). This release lands
remediation for all Critical, High, Medium, and Low/Info findings.

### Changed — Breaking

- **`AuditMode::default()` is now `MetadataOnly`** (was `Full`). Privacy-
  by-default for regulated deployments — audit records keep tool names,
  timing, token counts, verdicts, hooks, model labels, but not tool
  inputs / outputs / response text. Operators wanting full content
  capture must opt in with `config.audit_mode = AuditMode::Full;`. The
  recursive `strip_content` allow-list (F-AUTH-3) ensures nested fields
  are also redacted. (F-AUTH-6)
- **5 prior High BREAKING changes** to tool/sandbox/auth APIs landed in
  the same window — see commit `987f9b6` for the trait signature
  changes (tenant scoping on stores, etc.).

### Fixed — Critical (6)

- 6 Critical findings in `heartbit-core` closed in commit `3c7fc8b`,
  including DNS rebinding mitigation refinements, command injection
  hardening, and tenant scope enforcement on shared stores.

### Fixed — High (17)

- 12 non-breaking + 5 breaking High findings closed in `8502fa8` and
  `987f9b6` — covering sandbox path policy, redirect handling on LLM
  providers, tool-name repair Levenshtein bypass, MCP token cache
  isolation, and pre-tool guardrail ordering.

### Fixed — Medium (8)

- 8 Medium hardening findings closed in `187e19c` — reqwest
  `redirect::Policy::none()`, `https_only(true)`, `connect_timeout`,
  `no_proxy()`, SSE bounded buffer, MCP stdio line cap.

### Fixed — Network (1)

- **F-NET-2: DNS rebinding.** Custom `SafeDnsResolver` re-applies the IP
  blocklist at connect time (not just parse time), wired into both
  `safe_client_builder` and `vendor_client_builder`. (`52c0b58`)

### Fixed — Low/Info (18)

- **Filesystem (5):** `MAX_WALK_DEPTH=8` on skill discovery (F-FS-7);
  nonce-bearing `__HEARTBIT_CWD_<uuid>__` marker (F-FS-8); default
  protected paths `/etc`, `/root`, `~/.ssh`, `~/.aws`, `~/.config/gcloud`
  (F-FS-9); `is_protected` normalizes parent-of relationships (F-FS-11);
  patch.rs defense-in-depth (F-FS-12).
- **MCP / A2A (5):** `sanitize_log_field` (F-MCP-6); `TokenCacheKey`
  4-tuple struct prevents resource collision (F-MCP-8); sampling
  capability removed from advertisement (F-MCP-9); strict JSON-RPC id
  verification per spec 2.0 (F-MCP-13); `redact_idp_body` scrubs JWTs /
  bearer tokens / `*_token` fields from IdP error logs (F-MCP-16).
- **Memory (2):** `SafeDnsResolver`-equipped client + redirect policy on
  embedding provider (F-MEM-4); tenant filter uses
  `author_tenant_id.as_deref` comparison (F-MEM-5).
- **Network (2):** generic User-Agent, no version leak (F-NET-5);
  `sanitize_html_for_agent` strips `<script>`, `<style>`, `<iframe>`,
  `<object>`, `<embed>`, and `on*` attributes (F-NET-7).
- **Channel/Auth (1):** `PendingEntry` binds nonce to `session_id`,
  `resolve_input_for_session` enforces both — no cross-session input
  injection (F-AUTH-5).
- **Agent (3):** `PERMISSIONS_FILE_MAX_BYTES=1MB` +
  `PERMISSIONS_MAX_RULES=10000` DoS bounds (F-AGENT-13); multilingual
  E.164 + NANP phone regex for PII recall (F-AGENT-15);
  `HeuristicGate` refusal patterns trimmed to 4 high-precision
  phrases (F-LLM-7).

### Notes

- `rustls-webpki` bumped to `0.103.13` to patch RUSTSEC-2026-0049/
  0098/0099/0104. `rsa` (Marvin) and `quinn-proto` confirmed not in
  `heartbit-core`'s dependency tree via `cargo tree --invert`.
- Reqwest 0.12 redirect strip list confirmed by reading
  `redirect.rs:239-251` upstream — only `AUTHORIZATION`, `COOKIE`,
  `cookie2`, `PROXY_AUTHORIZATION`, `WWW_AUTHENTICATE` are stripped on
  cross-host redirect; custom auth headers (`x-api-key`,
  `x-goog-api-key`) require the explicit `redirect::Policy::none()`
  applied here.
- 2330 `heartbit-core` + 454 umbrella+CLI tests green.
  `cargo fmt -- --check && cargo clippy --tests -- -D warnings` clean.

## [2026.505.1] - 2026-05-05

### Fixed — Runtime Usability Audit

- **`AgentRunner::builder()` minimal chain now works.** The builder used to
  initialise `name` to `String::new()` while `build()` rejected empty names —
  every minimal chain (including `examples/hello_agent.rs` and the runner
  doctest) crashed at runtime with `agent name must not be empty`. The
  default is now `"agent"`; explicit `.name("")` still errors. (#4)
- **`MemoryQuery::reinforce` makes recall-time strength reinforcement
  opt-out.** `InMemoryStore::recall` previously reinforced `strength` by
  `+0.2` on every read, defeating decay-based pruning workflows that needed
  to observe the literal stored value. Default is `true` (preserves prior
  behaviour); set `reinforce: false` for a pure read. `Memory::store`,
  `Memory::recall`, and `MemoryEntry::strength` rustdoc now document the
  contract. (#5)
- **`EvalRunner::with_event_collector` clears events between cases.**
  `EvalRunner::run` never cleared the shared `EventCollector`, so
  `CostScorer` / `LatencyScorer` / `SafetyScorer` accumulated events across
  cases. Attach the collector via the new builder method to isolate per-case
  scoring. (#6)
- **`ConsolidationPipeline` skips are visible.** The pipeline silently
  dropped clusters when the per-cluster summary tripped the hardcoded
  `max_tokens = 512`. New `ConsolidationResult` struct + `run_detailed()`
  method surface `clusters_skipped`; new `with_summary_max_tokens(u32)`
  builder lets callers raise the cap; skipped clusters now emit
  `tracing::warn!`. The legacy `run()` 3-tuple shape is preserved. (#8)

### Changed — Breaking

- **`Guardrail::post_llm` now takes `&mut CompletionResponse`.** This is
  what makes `PiiGuardrail::Redact` actually redact LLM response text
  instead of silently degrading to `Warn`. Trait rustdoc explains that
  mutations must run synchronously inside the method body — the future's
  lifetime is tied to `&self`, not to `response`. Built-in guardrails and
  tests are updated; downstream guardrail implementors must change
  `&CompletionResponse` to `&mut CompletionResponse` in their `post_llm`
  impls. (#7)

## [2026.503.1] - 2026-05-03

> **Note:** `2026.306.7` was published earlier the same day but was misnumbered
> against the project's `YEAR.MMDD.patch` CalVer convention (`306` looks like
> `MM=03 DD=06`, but the release date was 2026-05-03). That artifact was yanked
> from crates.io. `2026.503.1` is the canonical first publish — same code,
> correct version.

### Added — B5b Failure-Mode Hardening

- **Idempotency keys.** `DaemonCommand::SubmitTask` and `POST /v1/tasks`
  accept an `idempotency_key` (`Option<String>` field / `Idempotency-Key`
  HTTP header). Scoped to `(tenant_id, idempotency_key)` via a partial
  unique index on `daemon_tasks`. No TTL by default; configurable via
  `[daemon.idempotency]` (`ttl_hours`, `sweep_interval_minutes`) — when
  `ttl_hours` is set, a background sweep task nulls expired keys.
  Duplicate requests return the existing task id without re-executing.
- **Per-tenant token cap.** `TenantTokenTracker` with `Arc`-owning RAII
  reservation tracks in-flight tokens per tenant. Configurable cap via
  `orchestrator.max_tokens_in_flight_per_tenant`. Submissions estimated to
  exceed the cap return `Error::TenantOverloaded` (HTTP 503 +
  `Retry-After: 5`). The in-flight counter is reconciled per turn using
  actual token usage; on task completion the runner releases its cumulative
  actual tokens back to the tenant's budget.
- **Per-(tenant, provider) circuit breaker.** `CircuitBreakerProvider`
  wraps any `LlmProvider`. Composes outside `RetryingProvider`
  (`CircuitBreaker<Retrying<Provider>>`). State machine: Closed → Open
  (after N consecutive retry-exhausted failures) → HalfOpen → Closed/Open.
  Open circuits fail fast with `Error::CircuitOpen`; no retries fire while
  open. Each `(tenant, provider)` pair has its own independent circuit.
  Configurable via `[provider.circuit]` (`failure_threshold`,
  `initial_open_duration_seconds`, `max_open_duration_seconds`,
  `backoff_multiplier`). The failure classifier trips on `ServerError`,
  `RateLimited`, and `Network` errors; `AuthError`, `InvalidRequest`, and
  `ContextOverflow` do not trip the circuit.
- Failure-mode hardening recipe in the user docs
  (`book/src/recipes/failure-modes.md`).

### Changed — B5b Failure-Mode Hardening

- `daemon_tasks.tenant_id` tightened to `NOT NULL DEFAULT ''` (matches the
  B4 `audit_log` pattern). Existing rows are backfilled to the
  empty-string single-tenant sentinel on migration. The migration is
  idempotent (`ADD COLUMN IF NOT EXISTS` / `UPDATE … WHERE … IS NULL`).

### Added — B4 Multi-Tenant Hardening

- `heartbit_core::auth::TenantScope` — owned `(tenant_id: String, user_id: Option<String>)`
  type required by `Memory` and `AuditTrail`. Empty-string tenant is the
  single-tenant sentinel; `new("")` and `with_user("")` normalize via code,
  not just prose. `From<&UserContext>` builds a scope from the daemon's JWT
  user context. `from_audit_fields(Option<&str>, Option<&str>)` is the
  helper for code paths that already split tenant/user into separate fields.
- `heartbit_core::sandbox::CorePathPolicy` + `CorePathPolicyBuilder` — path
  allowlist + glob denylist shared across filesystem-touching builtins.
  Canonicalize-first symlink defense. Always available (not Linux-gated).
- `heartbit_core::sandbox::SandboxPolicy` (moved from the umbrella; gated
  `target_os = "linux"` + `feature = "sandbox"`) composes a `CorePathPolicy`.
  `from_path_policy(Arc<CorePathPolicy>)` derives Landlock read/write paths
  from the policy's allowed dirs so bash subprocesses get kernel-level
  enforcement out of the box.
- `with_path_policy(Arc<CorePathPolicy>)` builder method on `BashTool`,
  `PatchTool`, `EditTool`, `WriteTool`, `ReadTool`. The policy's
  `check_path` is called before any I/O so denied paths return a sandbox
  error without ever touching the filesystem.
- `AgentRunnerBuilder::max_tool_calls_per_turn(u32)` — caps dispatched
  tool-use blocks per LLM turn. Distinct from the existing
  `max_tools_per_turn` (which limits *tool definitions*, a pre-filter on the
  LLM's tool list). Excess returns `Error::Agent` wrapped in
  `Error::WithPartialUsage`. Zero rejected at build time. Available via
  TOML `orchestrator.max_tool_calls_per_turn` (with per-agent override) and
  `HEARTBIT_MAX_TOOL_CALLS_PER_TURN` env var.
- `AuditTrail::entries_since(&TenantScope, since, limit)` — windowed scoped
  read.
- `AuditTrail::prune(retain)` — retention DELETE. `PostgresAuditTrail`
  implementation runs `DELETE FROM audit_log WHERE created_at < $1` against
  the new `idx_audit_created_at` index.
- `[sandbox]` config section: `allowed_dirs: Vec<PathBuf>`,
  `deny_globs: Vec<String>`. CLI builds a shared `CorePathPolicy` and threads
  it into all five filesystem builtins (and bash's `SandboxPolicy` on Linux).
- `[daemon.audit]` config section: `retain_days`, `prune_interval_minutes`.
  Daemon spawns a cancellation-aware background prune task. `retain_days = 0`
  and `prune_interval_minutes = 0` rejected at config-load time. Falls back
  to `HEARTBIT_AUDIT_RETAIN_DAYS` env var when TOML doesn't set it.
- Postgres schema: `memories.author_tenant_id TEXT NOT NULL DEFAULT ''`,
  `memories.author_user_id TEXT`, `audit_log.tenant_id TEXT NOT NULL DEFAULT ''`,
  `audit_log.user_id TEXT`. Indexes — single-column on `tenant_id` and
  `created_at` for the most common scoped/retention paths
  (`idx_audit_tenant`, `idx_audit_created_at`, `idx_memories_author_tenant`)
  plus composite indexes `(tenant_id, created_at DESC)` on `audit_log` and
  `(author_tenant_id, agent, created_at DESC)` on `memories` for the
  scoped-recall query shapes. All `ADD COLUMN IF NOT EXISTS` — idempotent,
  safe to re-run.
- Multi-tenant recipe chapter in the user docs (`book/src/recipes/multi-tenant.md`).

### Changed — B4 Multi-Tenant Hardening (breaking; pre-release)

- `Memory` trait: every method now takes `&TenantScope` as the first
  parameter (after `&self`). Migrate single-tenant call sites by passing
  `&TenantScope::default()`. Daemon-mode call sites build the scope from
  the request's `UserContext` via the `From<&UserContext>` impl.
  `InMemoryStore`, `NamespacedMemory`, `EmbeddingMemory`, and
  `PostgresMemoryStore` all updated. `NamespacedMemory` keeps its
  `agent_prefix` for intra-tenant namespacing — orthogonal to the new
  cross-tenant `TenantScope`.
- `AuditTrail::entries()` (no args) renamed to `entries_unscoped(limit)`.
  Callers must explicitly opt into cross-tenant visibility — `grep
  entries_unscoped` now finds every site that crosses the tenant boundary.
- `AuditTrail::entries_for_tenant(Option<&str>)` replaced by
  `entries(&TenantScope, limit)` — typed instead of stringly.
- The umbrella's `crates/heartbit/src/sandbox.rs` becomes a re-export of
  `heartbit_core::sandbox::*`. Existing `heartbit::sandbox::SandboxPolicy`
  imports continue to work. `heartbit::CorePathPolicy` and
  `heartbit::CorePathPolicyBuilder` are now flat-reachable from the umbrella
  on all platforms (previously the entire `sandbox` module was Linux+sandbox-gated).
- `landlock` dep migrated from the umbrella's optional deps to
  heartbit-core (under `[target.'cfg(target_os = "linux")'.dependencies]`).
  The umbrella's `sandbox = ["dep:landlock", "heartbit-core/sandbox"]`
  feature is now `sandbox = ["heartbit-core/sandbox"]`.

### Migration notes

heartbit-core is not yet on crates.io, so the `Memory` and `AuditTrail`
trait changes are pre-release breaking — any external consumer
(heartbit-cloud, downstream forks) needs to update call sites.

Before upgrading the Postgres schema on a multi-tenant deployment, run:

```sql
SELECT count(*) FROM memories WHERE author_tenant_id IS NULL;
SELECT count(*) FROM audit_log WHERE tenant_id IS NULL;
```

Non-zero on a multi-tenant installation means rows were written without a
scope — the migration backfills them with the empty-string sentinel,
making them invisible to multi-tenant queries.

### Refactor

- Workspace restructured: `heartbit-core` extracted as the official Rust
  agentic framework. The `heartbit` crate becomes a thin umbrella that
  re-exports `heartbit-core` (`pub use heartbit_core::*;`) and adds platform
  integrations (Postgres, Telegram/Discord/Slack adapters, Restate
  workflows, fastembed local embeddings, vault, JWT validator, daemon
  mode). **No breaking public API changes** — every existing import
  (`use heartbit::AgentRunner;` etc.) continues to compile via the
  umbrella's glob re-export. A handful of items previously `pub(crate)`
  in the umbrella are now `pub` in `heartbit-core` (transitional surface
  to keep umbrella-side code that consumes them compiling); these may
  return to `pub(crate)` in a future cleanup round. Library users should
  target `heartbit-core` directly; runtime/platform users keep using
  `heartbit`. `heartbit-cli` gains a small `daemon` feature that
  forwards to `heartbit/daemon` so the CLI can be built without daemon
  mode (the default `heartbit-cli` ships with `features = ["full"]`, so
  this is a no-op for default builds). Documentation rewritten:
  top-level `README.md` now leads with the framework
  (`cargo add heartbit-core` + quickstart), platform content moves to
  `crates/heartbit-cli/README.md` and `docs/platform.md`, and
  `crates/heartbit-core/README.md` is the docs.rs landing page.

### Security

- `WebFetchTool` now refuses requests to private/loopback/link-local IPs by
  default (loopback, link-local incl. cloud IMDS at `169.254.169.254`,
  RFC1918, CGNAT, ULA, multicast, unspecified, broadcast). HTTP redirects
  are no longer followed (a 302 to a private IP is surfaced as a 302
  response, not silently followed). Set `HEARTBIT_ALLOW_PRIVATE_IPS=1` or
  pass `IpPolicy::AllowPrivate` via `WebFetchTool::with_ip_policy` for
  single-tenant deployments that legitimately need internal-network access.

## [2026.228.3] - 2026-02-28

### Fixed

- **Cross-tenant token cache isolation** — `TokenExchangeAuthProvider` cache now keyed by `(tenant_id, user_id)` tuple instead of `user_id` alone, preventing user "alice" in tenant "acme" from receiving cached tokens belonging to user "alice" in tenant "globex". Token TTL capped at 3600 seconds (1 hour max).
- **Delegation chain forwarding to sub-agents** — `DelegateTaskTool`, `FormSquadTool`, and `OrchestratorBuilder::build()` now propagate `audit_delegation_chain` to sub-agent runners. Previously, delegation chain was set on the orchestrator but lost when spawning sub-agents.
- **Unauthenticated cross-tenant task access** — `handle_get`, `handle_cancel`, and `handle_events` now reject unauthenticated callers accessing tenant-scoped tasks. Previously, tasks with `tenant_id` set were accessible to any caller without a JWT.
- **WebSocket session tenant isolation** — All session operations (`SESSION_CREATE`, `SESSION_LIST`, `SESSION_DELETE`, `CHAT_HISTORY`, `CHAT_SEND`) now enforce tenant boundaries. `UserContext` from JWT middleware is threaded through the WebSocket upgrade → connection → dispatch chain. Sessions are created with `user_id`/`tenant_id` when user context is present.
- **Subject token wiring for RFC 8693** — `UserContext.raw_token` carries the original JWT. HTTP submit handler populates shared `user_tokens` map (keyed by `"{tenant_id}:{user_id}"`) for `TokenExchangeAuthProvider` consumption.
- **Interactive WS sessions now pass user_id/tenant_id** to `run_interactive_task` → `build_orchestrator_from_config`, enabling per-user memory namespacing and audit context for WebSocket-initiated agent runs.
- **WebSocket task registration with tenant context** — WS-initiated tasks now include `user_id`/`tenant_id` in the task store when user context is available. Previously, WS tasks were registered without tenant context, making them accessible to unauthenticated callers via `GET /tasks/{id}`.
- **Telegram consolidation prune scoped to user namespace** — Telegram idle-session memory pruning now passes user-scoped prefix (`tg:{user_id}`) to `Memory::prune()`. Previously, pruning was global and could delete weak entries from other users' namespaces.

### Changed

- **Namespace-scoped `Memory::prune()`** — `prune()` gains `agent_prefix: Option<&str>` parameter. When set, only entries whose `agent` field starts with the prefix are pruned. `NamespacedMemory::prune()` now always scopes to its own namespace, preventing cross-user memory deletion in multi-tenant setups. Previously, pruning via a `NamespacedMemory` would delete weak entries from ALL namespaces.
- **Session struct gains `user_id`/`tenant_id`** — `Session` struct, `SessionStore` trait (`create_with_user`, `list_for_tenant` methods), `InMemorySessionStore`, and `PostgresSessionStore` all updated for multi-tenant session isolation. `PostgresSessionStore::list_for_tenant()` uses SQL `WHERE tenant_id = $1` pushdown instead of in-memory filtering. PostgreSQL migration adds `user_id`/`tenant_id` columns with backward-compatible ALTER TABLE. Serde `#[serde(default)]` ensures backward compatibility with existing data.
- 10 new tests: 4 prune namespace isolation, 5 session multi-tenant isolation and backward compat, 1 register_task_with_user.

## [2026.228.2] - 2026-02-28

### Added

- **`[daemon.auth.token_exchange]` config** — RFC 8693 Token Exchange configuration (`exchange_url`, `client_id`, `client_secret`, `agent_token`, `scopes`) for per-user MCP auth delegation. Config validation rejects empty required fields.
- **Per-task MCP tool loading** — when `token_exchange` is configured and user context is present, each daemon task creates fresh MCP connections with a user-scoped delegated token instead of shared static auth.
- **`audit_delegation_chain`** on `AgentRunner`, `Orchestrator`, `SubAgentConfig` — records which agent(s) are in the delegation path when acting on behalf of a user. Populated automatically in multi-tenant mode.
- **Tenant-scoped store queries** — `TaskStore::list_filtered()` and `stats()` accept `tenant_id` parameter, pushing filter to store level. Fixes pagination counts and prevents cross-tenant data in stats.
- 11 new tests: 6 tenant-filtered store/core, 5 token exchange config validation.

## [2026.228.1] - 2026-02-28

### Added

- **Multi-tenant daemon** — single daemon instance serves multiple users with per-request tenant isolation. JWT-authenticated API ensures tasks, memory, and workspaces are scoped per user/tenant.
- **JWT/JWKS authentication** (`auth/jwt.rs`): `JwksClient` fetches and caches JWKS keys (5-minute TTL, auto-refetch on key rotation). `JwtValidator` verifies RS256 tokens, extracts `UserContext` (user_id, tenant_id, roles) from configurable claim names.
- **`UserContext` struct** (`daemon/types.rs`): carries `user_id`, `tenant_id`, and `roles` through every request. Extracted from JWT claims by auth middleware, injected into request extensions.
- **`[daemon.auth]` config section** (`config.rs`): `bearer_tokens` (static API keys with rotation support), `jwks_url` (JWKS endpoint), `issuer`/`audience` (JWT validation), `user_id_claim`/`tenant_id_claim`/`roles_claim` (configurable claim names for different IdPs).
- **Per-user memory namespacing** — daemon wraps memory store with `NamespacedMemory` using `tenant:{tid}:user:{uid}` prefix. Users cannot access each other's memories. Institutional memory remains shared via `shared_memory_read` tool.
- **Dynamic MCP token injection** (`tool/mcp.rs`): `AuthProvider` trait with `auth_header_for(user_id, tenant_id)` enables per-request authentication. `StaticAuthProvider` for backward-compatible static headers. `TokenExchangeAuthProvider` implements RFC 8693 token exchange against an IdP (e.g. xavyo-idp) to obtain user-scoped MCP tokens with in-memory caching.
- **Per-user workspace isolation** — workspace root becomes `{base}/{tenant_id}/{user_id}/` in multi-tenant mode. Path traversal prevention already enforced.
- **Audit trail enrichment** (`agent/audit.rs`): `AuditRecord` gains `user_id`, `tenant_id`, and `delegation_chain` fields. `AuditTrail` trait gains `entries_for_tenant()` for tenant-scoped queries.
- **A2A Agent Card** — daemon serves `GET /.well-known/agent.json` for agent discovery. Card includes agent name, description, skills (from config agents), auth schemes (bearer/JWT), and endpoint URL.
- `Error::Auth` variant for authentication-specific errors (distinct from infrastructure errors).
- `PostgresTaskStore` gains `user_id` and `tenant_id` columns with ALTER TABLE migration for existing databases.
- 32 new tests for JWT validation, claim extraction, cross-namespace isolation, and token exchange hardening.

### Changed

- All JWT/auth errors now use `Error::Auth` (previously used `Error::Agent`, making 401 vs 502 indistinguishable).
- `NamespacedMemory::recall()` always forces own namespace — ignores caller-supplied agent parameter to prevent cross-namespace reads via prompt injection.
- `TokenExchangeAuthProvider` hardened: token cache with TTL (30s early expiry), 10-second HTTP timeout, error body truncated to 512 bytes, respects `token_type` from response.
- `JwtValidator::validate()` rejects empty tokens and tokens exceeding 16 KiB.
- `extract_string_claim()` rejects null, boolean, object, and array claim values (accepts only string and number).

## [2026.227.1] - 2026-02-27

### Added

- **Local embedding provider** (`LocalEmbeddingProvider`): offline ONNX-based text embeddings via fastembed. No API keys, no network, zero cost per query. Supports 9 models (all-MiniLM-L6-v2 default, BGE variants, nomic variants, plus quantized `-q` suffixes). Feature-gated behind `local-embedding`.
- **Eval framework** (`eval/mod.rs`): built-in agent behavior testing with `EvalCase`, `EvalRunner`, and pluggable scorers (`TrajectoryScorer`, `KeywordScorer`, `SimilarityScorer`). Concurrent evaluation with per-case and aggregate scoring.
- **Workflow agents** (`agent/workflow.rs`): deterministic orchestration without LLM cost — `SequentialAgent` (chains output→input), `ParallelAgent` (concurrent via `JoinSet`), `LoopAgent` (repeat until condition).
- **Audit trail** (`agent/audit.rs`): `AuditTrail` trait with `InMemoryAuditTrail` and `PostgresAuditTrail` for logging agent decisions, tool calls, and guardrail outcomes.
- **Injection classifier guardrail** (`guardrails/injection.rs`): detect prompt injection attempts with warn or deny mode.
- **PII guardrail** (`guardrails/pii.rs`): detect PII (email, phone, SSN, credit card) with redact, warn, or deny actions.
- **Tool policy guardrail** (`guardrails/tool_policy.rs`): declarative per-tool allow/deny rules with input constraints (patterns, max length).
- **LLM-as-judge guardrail** (`guardrails/llm_judge.rs`): safety evaluation via a cheap judge model with criteria-based prompts. Fail-open on timeout.
- **Guardrail composition** (`guardrails/compose.rs`): `ConditionalGuardrail`, `GuardrailChain`, and `WarnToDeny` escalation.
- **`GuardrailMeta` trait**: optional guardrail identification for debugging and audit.
- **`GuardrailsConfig`** in config: top-level `[guardrails]` section with injection, PII, tool policy, and LLM judge sub-configs. Per-agent `guardrails` override.
- `cache_dir` field on `EmbeddingConfig` for local provider model cache directory.
- `local-embedding` feature flag on `heartbit` and `heartbit-cli` crates.
- `RoutingStrategy` trait and `KeywordRoutingStrategy` for pluggable task routing.
- `TrustLevel` enum moved to `config.rs` (always available, not sensor-gated).
- `SensorModality` re-exported from config (always available).
- Examples: `simple_agent.rs`, `mcp_agent.rs`, `custom_tool.rs`.
- `llms.md`: LLM-friendly project context file (llmstxt.org pattern).
- `install.md`: comprehensive installation guide with troubleshooting.

### Changed

- **Feature-gated modules**: `daemon`, `sensor`, `workflow` modules now require their respective feature flags. Previously always compiled.
- **Feature-gated re-exports**: `PostgresMemoryStore`, `PostgresSessionStore`, `PostgresTaskStore`, `PostgresStore`, `PostgresAuditTrail` gated behind `postgres`; `A2aClient` behind `a2a`; `SensorSecurityGuardrail` behind `sensor`; sensor re-exports behind `sensor`; `LocalEmbeddingProvider` behind `local-embedding`.
- Agent events expanded from 13 to 18 variants (added `GuardrailWarned`, `LlmRetry`, `ModelEscalated`, `ToolDeselected`, `ReflectionTriggered`).
- Guardrail re-exports expanded: all 8 guardrails, `GuardrailMode`, `PiiAction`, `PiiDetector`, `InputConstraint`, `ToolRule`, `WarnToDeny` now re-exported from crate root.
- Config re-exports expanded: `GuardrailsConfig`, `InjectionConfig`, `PiiConfig`, `ToolPolicyConfig`, `InputConstraintConfig`, `ToolPolicyRuleConfig`, `SensorModality`, `TrustLevel`.
- README comprehensively updated: expanded guardrails (2→8 with table), memory section (embedding providers, hybrid retrieval, confidentiality), feature flags section, workflow agents section, eval framework section, audit trail section, environment variables (12→27), config example (cascade, routing, dispatch_mode, session_prune), test count (2374→2665+).

## [2026.226.2] - 2026-02-26

### Fixed

- Repetitive pulse notifications dumping entire todo list (including completed items) every 30 minutes.
- Added `snoozed_until` field to `TodoEntry` for suppressing items from pulse.
- `format_for_pulse_prompt()` filters terminal/snoozed entries.
- Snooze action added to `TodoManageTool` (default 24h, validates hours > 0).
- Kafka event serialization: log-and-skip instead of sending empty payload.
- `subscribe_events`: log on lock poison instead of silent `None`.
- Validate Kafka `consumer_group`, `commands_topic`, `events_topic` for non-empty.

## [2026.226.1] - 2026-02-26

### Added

- Institutional memory: daemon task results auto-persist to shared `"institutional"` namespace, enabling cross-context knowledge flow from sensor pipeline to Telegram chat agents.
- Telegram dual recall: `preload_memories` now queries both user-private and institutional namespaces concurrently via `tokio::join!`.
- `institutional_recall_limit` config field for Telegram (default: 3).
- `story_id` field on `TaskOutcome` for story-scoped provenance tracking.

### Changed

- README rewritten with architecture diagrams, contributor guide, and disclaimer.
- README updated to highlight Telegram and Google Workspace integrations.
- README updated with Telegram community link.

## [2026.2.26] - 2026-02-26

### Added

- Multi-agent runtime with orchestrator and sub-agents (flat hierarchy, parallel dispatch via `tokio::JoinSet`).
- Three execution paths: standalone (in-process), Restate (durable with replay), and daemon (Kafka-backed with HTTP API).
- 14 built-in tools: bash, read, write, edit, patch, glob, grep, list, webfetch, websearch, todowrite, todoread, skill, question.
- MCP Streamable HTTP client (protocol `2025-03-26`) with automatic tool discovery and optional authentication.
- LLM providers: Anthropic and OpenRouter with SSE streaming.
- Retry provider with exponential backoff on 429/5xx errors.
- Cascading provider: tries cheapest model first, escalates on gate rejection or error.
- Prompt caching for Anthropic (cache reads at 10% input rate, writes at 125%).
- Structured output via synthetic `__respond__` tool with JSON Schema validation.
- Context management strategies: unlimited, sliding window, and LLM-generated summarization.
- Memory system with in-memory and PostgreSQL backends (store, recall, update, forget, consolidate).
- Composite recall scoring (recency, importance, relevance, strength) with Ebbinghaus decay.
- Knowledge base with paragraph-aware chunking, keyword search, and file/glob/URL loaders.
- Guardrails: pre/post LLM and tool hooks with allow/deny actions.
- Human-in-the-loop approval for tool execution (`--approve` flag).
- Sensor pipeline with 6 sources, triage, deduplication, and story grouping.
- Telegram bot integration with DM support, streaming responses, and multimodal input (photos, voice, documents).
- Daemon mode: Kafka consumer loop, Axum HTTP API, SSE event streaming, cron scheduler, heartbeat pulse.
- Cross-agent coordination via shared blackboard and memory tools.
- Dynamic task routing based on complexity heuristics.
- Agent workspace with path traversal prevention.
- Cost tracking with per-model pricing for Claude 4, 3.5, and 3 generations.
- 13 structured agent event variants with JSON stderr output (`--verbose`).
- OpenTelemetry tracing via OTLP exporter.
- Interactive chat mode with multi-turn REPL.
- Docker support with multi-stage build.
- Doom loop detection and auto-compaction on context overflow.
- Tool output truncation with UTF-8 safe boundaries.
- Tool name repair via Levenshtein distance matching.
