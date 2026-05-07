# heartbit Foundation — Phase 0 Design

**Date:** 2026-05-07
**Status:** design (awaiting review)
**Implements first:** yes — prerequisite for `2026-05-07-heartbit-ghost-x-agent-design.md`

## Why this exists separately

The `heartbit-ghost` project (best-in-class autonomous X agent with voice modeling and A/B feedback) requires three structural changes to `heartbit-core` and `heartbit-cli`. These changes are trait-touching, ripple across ~30 in-tree tool implementations, and benefit from a focused, reviewable release on their own — *before* any feature work consumes them. This document specifies that release.

After Phase 0 ships, `heartbit-ghost` becomes a pure consumer: it adds a recipe to a registry, ships tools that use a context arg, and wires Telegram review on top of existing infrastructure. Zero further core changes.

## Architecture decisions

### F-AD-1 · `Tool::execute` takes `&ExecutionContext` (hard break)

**Today.**
```rust
pub trait Tool: Send + Sync {
    fn definition(&self) -> ToolDefinition;
    fn execute(&self, input: serde_json::Value)
        -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>>;
}
```

**After Phase 0.**
```rust
pub trait Tool: Send + Sync {
    fn definition(&self) -> ToolDefinition;
    fn execute(&self, ctx: &ExecutionContext, input: serde_json::Value)
        -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>>;
}

#[derive(Debug, Clone, Default)]
pub struct ExecutionContext {
    pub tenant_id: Option<String>,
    pub user_id: Option<String>,
    pub workspace: Option<PathBuf>,
    pub credentials: Option<Arc<dyn CredentialResolver>>,
    pub audit_sink: Option<Arc<dyn AuditSink>>,
}

pub trait CredentialResolver: Send + Sync {
    fn resolve(&self, name: &str) -> Pin<Box<dyn Future<Output = Result<Secret, Error>> + Send + '_>>;
}
```

**Migration approach: hard break.** No backward-compat bridge method, no deprecation cycle. All in-tree tools migrate to the new signature in the same release. External `impl Tool` consumers receive a compile error and update.

**Rationale.**
- The bridge approach (default `execute` calling `execute_with_context` with empty context, or vice versa) creates two source-of-truth paths for the same behavior. Subtle context-dropping bugs become possible (a tool calls the no-context method internally and silently loses tenant info).
- The trait surface is small. ~30 in-tree implementations migrate in one PR set; one-line change for tools that don't use context (just add the unused `_ctx: &ExecutionContext` parameter).
- Released as a minor version bump with a clear migration note: "If you implement `Tool`, add `&ExecutionContext` as the first arg to `execute`."

**Context population.**
- `AgentRunner` constructs an `ExecutionContext` per turn and passes it into every tool call. Default empty context if no upstream provider populates it.
- Restate workflow path constructs context at the activity boundary from the workflow's invocation params.
- `DaemonCore::dispatch_command` populates `tenant_id` / `user_id` from the existing audit data already in `DaemonCommand::SubmitTask` and threads through to the agent runner.
- Tests construct `ExecutionContext::default()` — same ergonomics as before for unit tests.

**`CredentialResolver` is a trait, not a concrete type.** This phase ships only the trait. A no-op default implementation lives in `heartbit-core`. Concrete implementations (env-var, vault-backed, daemon tenant-scoped) ship in their respective crates.

### F-AD-2 · Persona registry trait in `heartbit-core`

**Goal.** A small, generic abstraction so future personas (`heartbit-ghost`, eventual `heartbit-coder`, etc.) plug in identically. Registry is **empty** in this phase — the trait surface ships, no concrete personas yet.

**Surface.**
```rust
pub trait Persona: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn version(&self) -> &str;
    fn expand(&self, params: &PersonaParams) -> Result<PersonaExpansion, Error>;
}

pub struct PersonaParams {
    pub overrides: toml::Value,        // user-supplied overrides for this instance
    pub credentials_env: Option<String>, // glob like "X_*"
    pub authorship_mode: AuthorshipMode,
}

pub struct PersonaExpansion {
    pub agents: Vec<AgentConfig>,       // sub-agents the persona requires
    pub orchestrator: OrchestratorConfig,
    pub tools: Vec<Arc<dyn Tool>>,      // tool instances the persona contributes
    pub triggers: Vec<TriggerSpec>,     // cron / sensor / mention-poll / manual hooks
    pub review: Option<ReviewSpec>,     // A/B review channel config (telegram, etc.)
}

pub struct PersonaRegistry { /* private fields */ }

impl PersonaRegistry {
    pub fn new() -> Self;
    pub fn register(&mut self, persona: Arc<dyn Persona>);
    pub fn get(&self, name: &str) -> Option<Arc<dyn Persona>>;
    pub fn list(&self) -> Vec<&str>;
}
```

`AuthorshipMode`, `TriggerSpec`, `ReviewSpec` are enums/structs with `non_exhaustive` annotations. Their concrete shapes are spelled out in the heartbit-ghost design; in Phase 0 they exist as enums with the variants required by the registered consumers (which is none yet — variants are added when ghost lands).

**Phase 0 includes only:**
- The `Persona` trait + `PersonaRegistry` struct
- `PersonaParams` and `PersonaExpansion` skeletons
- `AuthorshipMode` enum with the variants the ghost design will need (`HumanAssisted`, `AutonomousDisclosed`, `AutonomousUndisclosed`) so the ghost design's TOML schema parses cleanly when it lands
- `TriggerSpec` / `ReviewSpec` are deliberately empty enums (`#[non_exhaustive] enum TriggerSpec {}`) in Phase 0 — variants land with their consumers

### F-AD-3 · `[[persona]]` config schema

**Goal.** `HeartbitConfig` parses a `[[persona]]` section so `daemon.toml` files referencing `heartbit-ghost:x` are valid TOML once the ghost crate ships, without needing another core change.

**Schema.**
```toml
[[persona]]
name = "x"                                # tenant-scoped instance name
recipe = "heartbit-ghost:x"               # registry key — "<crate>:<recipe>"
credentials_env = "X_*"
authorship_mode = "autonomous_undisclosed"
phase = "calibration"                     # calibration | supervised | autonomous | sentinel

# Persona-specific overrides — opaque to core, passed through to expand()
[persona.x.style]
# arbitrary keys; persona expand() interprets them

[persona.x.cadence]
post_times = ["09:30+/-15m"]
```

**Validation in Phase 0.**
- Parse the section into a `PersonaConfig` struct
- Validate `name` is a unique identifier (no duplicates within a config)
- Validate `recipe` parses as `<crate>:<name>` (lexical only — no registry lookup)
- Validate `phase` is a known variant
- **Do not** look up the recipe in the registry. Registry is empty; lookup happens at daemon startup once the ghost crate is added.

**Error messages.**
- Unknown recipe in daemon startup: `error: persona "x" references recipe "heartbit-ghost:x" which is not registered. Available recipes: <empty>. Did you forget to add heartbit-ghost as a daemon dependency?`

### F-AD-4 · `heartbit persona <subcommand>` CLI surface

**Goal.** Wire the CLI subcommand surface against the (empty) registry so the user-facing API is stable from Phase 0. Each subcommand returns a clean, intent-revealing error against the empty registry.

**Subcommands shipped as functional shells:**

| Subcommand | Phase 0 behavior |
|---|---|
| `heartbit persona list` | Prints "No personas registered." (zero rows from empty registry) |
| `heartbit persona show <name>` | `error: persona "x" not found. <list-equivalent error>` |
| `heartbit persona run <name> --once <prompt>` | Same not-found error |
| `heartbit persona corpus add <writer> <path>` | `error: corpus management requires a registered persona; none registered.` |
| `heartbit persona corpus list <name>` | Same |
| `heartbit persona profile rebuild <name>` | Same |
| `heartbit persona profile diff <name> <v1> <v2>` | Same |
| `heartbit persona phase <name> --set <phase>` | Same |
| `heartbit persona pause <name>` | Same |
| `heartbit persona resume <name>` | Same |
| `heartbit persona export-preferences <name>` | Same |
| `heartbit persona audit <name> --since <duration>` | Same |

**Wiring.** Each subcommand uses `clap` with full arg parsing. Help text (`--help`) is complete and accurate from Phase 0. The unimplemented body returns `Err(...)` with the canonical not-found message.

**Daemon-only vs standalone.**
- `list`, `show`, `run --once` work standalone (load `[[persona]]` from a config file passed via `-c`)
- `pause`, `resume`, `audit`, `phase`, `corpus *`, `profile *`, `export-preferences` require a running daemon — they call into the daemon's HTTP API. In Phase 0, those subcommands check for the daemon endpoint and return a clear error if it's unreachable; the API endpoint itself can be a 501 stub until ghost lands.

## Implementation order

The PR sequence within Phase 0:

1. **`heartbit-core`: add `ExecutionContext` + `CredentialResolver` trait + `AuditSink` trait**
   - New types, no consumers yet
2. **`heartbit-core`: change `Tool::execute` signature** (the hard break)
   - One PR; touches every `impl Tool for *` in the workspace
   - Group by module (filesystem builtins, web builtins, memory tools, blackboard tools, mcp adapters, orchestrator tools, knowledge tools, handoff, a2a, daemon todo)
3. **`heartbit-core`: thread context through `AgentRunner`**
   - `AgentRunner::run` constructs context per turn; passes to `select_tools_for_turn` then to each tool call
4. **`heartbit/workflow`: thread context through Restate path**
   - `tool_call` activity constructs context from invocation params
5. **`heartbit/daemon`: populate context in `dispatch_command`**
   - From existing audit fields on `DaemonCommand::SubmitTask`
6. **`heartbit-core`: add `Persona` trait + `PersonaRegistry`**
   - Empty registry; trait surface only
7. **`heartbit-core`: add `[[persona]]` config schema**
   - Lexical validation; no registry lookup
8. **`heartbit-cli`: add `persona` subcommand surface**
   - Full clap parsing, not-found errors, daemon-API stubs

Each step is a separate commit (or small PR) within one Phase 0 release.

## Tests

- **`ExecutionContext` propagation:** an integration test where a custom test tool asserts on the context it receives — verifying tenant_id, workspace, credentials propagate from `AgentRunner::run` and from the daemon's `dispatch_command`
- **Empty registry:** `PersonaRegistry::new()` then `list()` returns empty; `get(...)` returns None; CLI subcommands return canonical not-found errors
- **Schema validation:** valid `[[persona]]` parses; duplicate names rejected; malformed recipe key rejected; unknown phase rejected
- **All existing tool tests pass** after the trait migration (regression net)
- **CLI help output** for `heartbit persona --help` and each subcommand is complete (smoke test that `--help` exits 0 with non-empty output)

## Risks

- **Trait change ripples to every `impl Tool`.** Mitigation: do it in a single atomic PR per crate; CI's existing `cargo fmt && cargo clippy -- -D warnings && cargo test` catches signature drift.
- **External implementors break.** Mitigation: release notes explicitly call out the trait change as the headline; provide a one-line example diff in the migration guide; the change is a `_ctx: &ExecutionContext` parameter most tools ignore.
- **`AuthorshipMode` / `TriggerSpec` / `ReviewSpec` are added without consumers.** Mitigation: `non_exhaustive`; consumers (`heartbit-ghost`) add variants when they land. Phase 0 ships the minimum needed to parse a daemon.toml that mentions a ghost persona.

## Out of scope (explicit)

- The `heartbit-ghost` crate itself (Phase 1; separate design doc)
- Any concrete persona, recipe, or tool addition
- Voice modeling, corpus, blend, profile schema
- A/B feedback loop, Telegram review wiring
- Sensor / cron / mention-poll handlers
- CLI debt sweep beyond the new `persona` subcommand surface (user chose narrow scope)
- Harness mechanics refactor (`BuiltinToolsConfig`, dead `ToolRisk`, schemars-driven schemas)
- Rate-limiting middleware, streaming output, lifecycle hooks, cancellation
- Training / fine-tuning infrastructure

## Acceptance criteria

Phase 0 is done when:

- `cargo fmt -- --check && cargo clippy -- -D warnings && cargo test` green across the workspace
- All in-tree `impl Tool for *` use the new signature; the old signature is gone (no bridge code)
- `PersonaRegistry::new()` is exported from `heartbit-core` and works
- A daemon.toml with a `[[persona]]` referencing `heartbit-ghost:x` parses, validates lexically, and fails with a clear "recipe not registered" error at startup
- `heartbit persona --help` lists all subcommands; each subcommand has working `--help`; each returns canonical not-found errors

## Next: Phase 1

Once Phase 0 ships, the implementation plan from the heartbit-ghost design (`2026-05-07-heartbit-ghost-x-agent-design.md`) takes over. That spec assumes Phase 0 is complete and references its primitives directly.
