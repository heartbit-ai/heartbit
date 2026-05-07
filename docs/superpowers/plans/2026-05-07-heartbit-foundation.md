# heartbit Foundation (Phase 0) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the trait-touching prerequisites for `heartbit-ghost`: hard-break of `Tool::execute` to take `&ExecutionContext`, new `Persona` trait + `PersonaRegistry` in `heartbit-core`, `[[persona]]` TOML config schema, and the `heartbit persona` CLI subcommand surface (functional shells against an empty registry).

**Architecture:** New context arg threads from request entry (CLI / Restate / daemon) → `AgentRunner` → every tool's `execute()`. Persona registry is a small in-memory abstraction that future persona crates register into at startup. Empty in this release; concrete personas land in Phase 1 (heartbit-ghost).

**Tech Stack:** Rust 2021, Tokio, `clap` for CLI, `serde`/`toml` for config, `tracing`, existing workspace.

**Spec:** `docs/superpowers/specs/2026-05-07-heartbit-foundation-design.md`

---

## File Structure

### New files
- `crates/heartbit-core/src/execution_context.rs` — `ExecutionContext` struct, `CredentialResolver` and `AuditSink` traits, `Secret` newtype
- `crates/heartbit-core/src/persona/mod.rs` — `Persona` trait, `PersonaRegistry`
- `crates/heartbit-core/src/persona/types.rs` — `PersonaParams`, `PersonaExpansion`, `AuthorshipMode`, empty `TriggerSpec`/`ReviewSpec` enums
- `crates/heartbit-core/src/config/persona.rs` — `PersonaConfig` (the `[[persona]]` block)
- `crates/heartbit-cli/src/persona.rs` — `persona` subcommand module (functional shells)

### Modified files
- `crates/heartbit-core/src/lib.rs` — module declarations + re-exports
- `crates/heartbit-core/src/tool/mod.rs` — `Tool` trait signature change
- `crates/heartbit-core/src/config/mod.rs` — add `pub personas: Vec<PersonaConfig>` to `HeartbitConfig`
- All `impl Tool for *` sites in the workspace (~35 sites; full list in Task 2)
- `crates/heartbit-core/src/agent/runner.rs` — thread `ExecutionContext` through `execute_tools_parallel`
- `crates/heartbit/src/workflow/agent_service.rs` — Restate `tool_call` activity constructs context
- `crates/heartbit/src/daemon/core.rs` — `dispatch_command` populates context from existing audit fields
- `crates/heartbit-cli/src/main.rs` — register the new `persona` subcommand

---

## Task 1: Foundation types — `ExecutionContext`, `CredentialResolver`, `AuditSink`

**Why:** Add the new types as a self-contained module with no consumers yet. This decouples the type-introduction commit from the trait-signature commit, keeping each diff small and reviewable.

**Files:**
- Create: `crates/heartbit-core/src/execution_context.rs`
- Modify: `crates/heartbit-core/src/lib.rs`
- Test: `crates/heartbit-core/src/execution_context.rs` (in-file `#[cfg(test)] mod tests`)

- [ ] **Step 1: Write the failing tests for `ExecutionContext::default()` and `Secret` redaction**

Create `crates/heartbit-core/src/execution_context.rs` with:

```rust
//! Per-request execution context threaded through tool dispatch.
//!
//! Every `Tool::execute` call receives an `&ExecutionContext`. The context
//! carries tenant/user identity, the workspace root, and resolvers for
//! per-tenant secrets and audit sinks. It is constructed at the request
//! boundary (CLI command, Restate workflow activity, daemon dispatch) and
//! threaded through the agent runner unchanged.

use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use crate::error::Error;

/// Per-request context carried into every tool invocation.
#[derive(Clone, Default)]
pub struct ExecutionContext {
    /// Tenant identifier (multi-tenant deployments). `None` outside of multi-tenant flows.
    pub tenant_id: Option<String>,
    /// User identifier on whose behalf the agent runs. `None` outside of authenticated flows.
    pub user_id: Option<String>,
    /// Workspace root for filesystem-aware tools. `None` when no workspace is configured.
    pub workspace: Option<PathBuf>,
    /// Resolver for per-tenant secrets (API keys, OAuth tokens). `None` when no resolver is configured.
    pub credentials: Option<Arc<dyn CredentialResolver>>,
    /// Sink for tool-level audit records. `None` when no audit sink is configured.
    pub audit_sink: Option<Arc<dyn AuditSink>>,
}

impl std::fmt::Debug for ExecutionContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExecutionContext")
            .field("tenant_id", &self.tenant_id)
            .field("user_id", &self.user_id)
            .field("workspace", &self.workspace)
            .field("credentials", &self.credentials.as_ref().map(|_| "<resolver>"))
            .field("audit_sink", &self.audit_sink.as_ref().map(|_| "<sink>"))
            .finish()
    }
}

/// Resolves a named secret (API key, token) for the current tenant.
pub trait CredentialResolver: Send + Sync {
    /// Resolve a secret by logical name (e.g. `"X_API_KEY"`).
    fn resolve(
        &self,
        name: &str,
    ) -> Pin<Box<dyn Future<Output = Result<Secret, Error>> + Send + '_>>;
}

/// Receives per-tool audit records emitted by tools that opt in.
pub trait AuditSink: Send + Sync {
    /// Record a structured audit entry. Implementations must not block.
    fn record(
        &self,
        record: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>>;
}

/// A secret value with redacted `Debug`/`Display` formatting.
#[derive(Clone)]
pub struct Secret(String);

impl Secret {
    /// Wrap a secret string. Use `expose()` to read the inner value.
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Read the inner secret string. Caller is responsible for not logging the result.
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Debug for Secret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Secret(<redacted>)")
    }
}

impl std::fmt::Display for Secret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<redacted>")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn execution_context_default_has_no_identity() {
        let ctx = ExecutionContext::default();
        assert!(ctx.tenant_id.is_none());
        assert!(ctx.user_id.is_none());
        assert!(ctx.workspace.is_none());
        assert!(ctx.credentials.is_none());
        assert!(ctx.audit_sink.is_none());
    }

    #[test]
    fn execution_context_clone_preserves_fields() {
        let ctx = ExecutionContext {
            tenant_id: Some("tenant-1".into()),
            user_id: Some("user-2".into()),
            workspace: Some(PathBuf::from("/tmp/ws")),
            credentials: None,
            audit_sink: None,
        };
        let cloned = ctx.clone();
        assert_eq!(cloned.tenant_id.as_deref(), Some("tenant-1"));
        assert_eq!(cloned.user_id.as_deref(), Some("user-2"));
        assert_eq!(cloned.workspace, Some(PathBuf::from("/tmp/ws")));
    }

    #[test]
    fn secret_debug_redacts() {
        let s = Secret::new("super-secret-token");
        let debug = format!("{:?}", s);
        assert!(!debug.contains("super-secret-token"));
        assert!(debug.contains("<redacted>"));
    }

    #[test]
    fn secret_display_redacts() {
        let s = Secret::new("super-secret-token");
        let display = format!("{}", s);
        assert!(!display.contains("super-secret-token"));
        assert!(display.contains("<redacted>"));
    }

    #[test]
    fn secret_expose_returns_inner() {
        let s = Secret::new("super-secret-token");
        assert_eq!(s.expose(), "super-secret-token");
    }

    #[test]
    fn execution_context_debug_does_not_leak_resolver_internals() {
        struct DummyResolver;
        impl CredentialResolver for DummyResolver {
            fn resolve(
                &self,
                _name: &str,
            ) -> Pin<Box<dyn Future<Output = Result<Secret, Error>> + Send + '_>> {
                Box::pin(async { Ok(Secret::new("x")) })
            }
        }

        let ctx = ExecutionContext {
            credentials: Some(Arc::new(DummyResolver)),
            ..ExecutionContext::default()
        };
        let debug = format!("{:?}", ctx);
        assert!(debug.contains("<resolver>"));
        assert!(!debug.contains("DummyResolver"));
    }
}
```

- [ ] **Step 2: Run tests — should fail with module-not-found**

```bash
cargo test -p heartbit-core --lib execution_context
```

Expected: `error[E0583]: file not found for module \`execution_context\`` (because lib.rs doesn't declare it yet).

- [ ] **Step 3: Declare and re-export the module from `lib.rs`**

In `crates/heartbit-core/src/lib.rs`, find the module declarations near the top (after `pub mod error;`) and add:

```rust
pub mod execution_context;
```

In the re-exports section (alongside other `pub use`), add:

```rust
pub use execution_context::{AuditSink, CredentialResolver, ExecutionContext, Secret};
```

- [ ] **Step 4: Run tests — all 6 should pass**

```bash
cargo test -p heartbit-core --lib execution_context
```

Expected: `test result: ok. 6 passed; 0 failed`.

- [ ] **Step 5: Workspace-wide quality gate**

```bash
cargo fmt -- --check && cargo clippy -p heartbit-core -- -D warnings
```

Expected: no errors, no warnings.

- [ ] **Step 6: Commit**

```bash
git add crates/heartbit-core/src/execution_context.rs crates/heartbit-core/src/lib.rs
git commit -m "feat(core): add ExecutionContext, CredentialResolver, AuditSink, Secret

Foundation types for the upcoming Tool trait migration. No consumers
yet; types are exported for use by downstream crates and the trait
change in the next commit.

Refs: docs/superpowers/specs/2026-05-07-heartbit-foundation-design.md F-AD-1"
```

---

## Task 2: Hard-break of `Tool::execute` signature + atomic migration of all impls

**Why:** The trait change must be atomic. Once the signature changes, every `impl Tool for *` in the workspace must compile. The spec mandates a hard break (no bridge method). All ~35 impl sites migrate in one PR / commit.

**Files (signature change):**
- Modify: `crates/heartbit-core/src/tool/mod.rs`

**Files (impl migration — apply the same trivial diff to each):**

Each migration adds an unused `_ctx: &ExecutionContext` first parameter to `execute()`. No tool in this phase reads from the context — that's a Phase 1 (heartbit-ghost) consumer concern. The diff per file is:

```rust
// Before
fn execute(&self, input: serde_json::Value)
    -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>>

// After
fn execute(&self, _ctx: &heartbit_core::ExecutionContext, input: serde_json::Value)
    -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>>
```

(Inside `heartbit-core` itself, the import path is `crate::ExecutionContext` instead of `heartbit_core::ExecutionContext`.)

**Files to migrate** (every `impl Tool for *` in the workspace):

`heartbit-core` builtins (`crates/heartbit-core/src/tool/builtins/`):
- `bash.rs` — `BashTool`
- `edit.rs` — `EditTool`
- `glob.rs` — `GlobTool`
- `grep.rs` — `GrepTool`
- `image_generate.rs` — `ImageGenerateTool`
- `list.rs` — `ListTool`
- `patch.rs` — `PatchTool`
- `question.rs` — `QuestionTool`
- `read.rs` — `ReadTool`
- `skill.rs` — `SkillTool`
- `todo.rs` — `TodoWriteTool`, `TodoReadTool` (2 impls)
- `tts.rs` — `TtsTool`
- `twitter_post.rs` — `TwitterPostTool`
- `webfetch.rs` — `WebFetchTool`
- `websearch.rs` — `WebSearchTool`
- `write.rs` — `WriteTool`

`heartbit-core` other tool impls:
- `tool/handoff.rs` — `HandoffTool`
- `tool/a2a.rs` — `A2aTool` (cfg-gated `feature = "a2a"`)
- `tool/mcp.rs` — `McpTool`, `McpResourceTool`, `McpPromptTool` (3 impls)
- `tool/mcp_server.rs` — `EchoTool`, `FailTool` (2 test impls in `#[cfg(test)]`)
- `agent/orchestrator.rs` — `DelegateTaskTool`, `FormSquadTool`, `SpawnAgentTool` (3 impls)
- `agent/blackboard_tools.rs` — `BlackboardReadTool`, `BlackboardWriteTool`, `BlackboardListTool` (3 impls)
- `memory/tools.rs` — `MemoryStoreTool`, `MemoryRecallTool`, `MemoryUpdateTool`, `MemoryForgetTool`, `MemoryConsolidateTool` (5 impls)
- `memory/shared_tools.rs` — `SharedMemoryReadTool`, `SharedMemoryWriteTool` (2 impls)
- `knowledge/tools.rs` — `KnowledgeSearchTool`
- `agent/mod.rs` test impls in `#[cfg(test)] mod tests`: `MockTool`, `SlowTool`, `StrictTool`, `BigTool`, `TrackingTool` (5 test impls)
- `agent/runner.rs` test impl in `#[cfg(test)]`: `NoopTool` (1 test impl)
- `examples/custom_tool.rs` — `WordCount`

Outside `heartbit-core`:
- `crates/heartbit/src/daemon/todo.rs` — `TodoManageTool`
- `crates/heartbit/src/workflow/agent_service.rs` — `MockTool` (test)
- `crates/heartbit/examples/custom_tool.rs` — `PriceLookupTool`

**Test:** existing test suite is the regression net; no new tests in this task. Tool-call sites that go through `AgentRunner::execute_tools_parallel` are addressed in Task 3.

- [ ] **Step 1: Change the trait signature in `crates/heartbit-core/src/tool/mod.rs`**

Find the current trait definition (around line 107):

```rust
pub trait Tool: Send + Sync {
    fn definition(&self) -> ToolDefinition;

    fn execute(
        &self,
        input: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>>;
}
```

Replace with:

```rust
pub trait Tool: Send + Sync {
    fn definition(&self) -> ToolDefinition;

    fn execute(
        &self,
        ctx: &crate::ExecutionContext,
        input: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>>;
}
```

Also update the docstring example earlier in the file (the `EchoTool` example block) so the documented signature matches the new trait. The example should show:

```rust
fn execute(
    &self,
    _ctx: &heartbit_core::ExecutionContext,
    input: serde_json::Value,
) -> Pin<Box<dyn Future<Output = Result<ToolOutput, heartbit_core::Error>> + Send + '_>> {
    Box::pin(async move {
        let text = input.get("text").and_then(|v| v.as_str()).unwrap_or("");
        Ok(ToolOutput::success(text.to_string()))
    })
}
```

- [ ] **Step 2: Run a workspace check — should explode with ~35 errors**

```bash
cargo check --workspace 2>&1 | grep "error\[E" | head -50
```

Expected: many errors of the form `error[E0046]: not all trait items implemented, missing: \`execute\`` or `error[E0050]: method \`execute\` has 1 parameter but the declaration in trait \`Tool::execute\` has 2`. This confirms the breakage is caught and tells you which files to update.

- [ ] **Step 3: Migrate each builtin impl in `heartbit-core`**

For each file in the "heartbit-core builtins" list above, update the `impl Tool for X { fn execute(&self, input: ...) -> ... }` to `fn execute(&self, _ctx: &crate::ExecutionContext, input: ...) -> ...`.

Files in this batch:
- `crates/heartbit-core/src/tool/builtins/bash.rs`
- `crates/heartbit-core/src/tool/builtins/edit.rs`
- `crates/heartbit-core/src/tool/builtins/glob.rs`
- `crates/heartbit-core/src/tool/builtins/grep.rs`
- `crates/heartbit-core/src/tool/builtins/image_generate.rs`
- `crates/heartbit-core/src/tool/builtins/list.rs`
- `crates/heartbit-core/src/tool/builtins/patch.rs`
- `crates/heartbit-core/src/tool/builtins/question.rs`
- `crates/heartbit-core/src/tool/builtins/read.rs`
- `crates/heartbit-core/src/tool/builtins/skill.rs`
- `crates/heartbit-core/src/tool/builtins/todo.rs` (two impls)
- `crates/heartbit-core/src/tool/builtins/tts.rs`
- `crates/heartbit-core/src/tool/builtins/twitter_post.rs`
- `crates/heartbit-core/src/tool/builtins/webfetch.rs`
- `crates/heartbit-core/src/tool/builtins/websearch.rs`
- `crates/heartbit-core/src/tool/builtins/write.rs`

For any tool that **internally** calls another tool's `execute(input)`, update the call site to `execute(_ctx, input)` (most builtins don't; only check sites if cargo check reports errors).

- [ ] **Step 4: Migrate non-builtin Tool impls in `heartbit-core`**

Apply the same diff to:
- `crates/heartbit-core/src/tool/handoff.rs` (`HandoffTool`)
- `crates/heartbit-core/src/tool/a2a.rs` (`A2aTool`, cfg-gated)
- `crates/heartbit-core/src/tool/mcp.rs` (`McpTool`, `McpResourceTool`, `McpPromptTool`)
- `crates/heartbit-core/src/tool/mcp_server.rs` (`EchoTool`, `FailTool` in test mod)
- `crates/heartbit-core/src/agent/orchestrator.rs` (`DelegateTaskTool`, `FormSquadTool`, `SpawnAgentTool`)
- `crates/heartbit-core/src/agent/blackboard_tools.rs` (`BlackboardReadTool`, `BlackboardWriteTool`, `BlackboardListTool`)
- `crates/heartbit-core/src/memory/tools.rs` (5 memory tools)
- `crates/heartbit-core/src/memory/shared_tools.rs` (2 shared memory tools)
- `crates/heartbit-core/src/knowledge/tools.rs` (`KnowledgeSearchTool`)
- `crates/heartbit-core/src/agent/mod.rs` test impls (5)
- `crates/heartbit-core/src/agent/runner.rs` test impl (1)
- `crates/heartbit-core/examples/custom_tool.rs` (`WordCount`)

If any test impl calls another tool's `execute()` internally, update those call sites too.

- [ ] **Step 5: Migrate Tool impls outside `heartbit-core`**

Apply the same diff to:
- `crates/heartbit/src/daemon/todo.rs` (`TodoManageTool`) — uses `heartbit_core::ExecutionContext`
- `crates/heartbit/src/workflow/agent_service.rs` (`MockTool` in test mod)
- `crates/heartbit/examples/custom_tool.rs` (`PriceLookupTool`)

- [ ] **Step 6: Workspace check — should now compile**

```bash
cargo check --workspace
```

Expected: zero errors. If errors remain, they are call-site errors (places calling `tool.execute(input)` with one arg) — those will be fixed in Task 3 (`AgentRunner` and friends). For Task 2's commit boundary, all `impl Tool for *` sites must be migrated and the trait must compile; call-site fixes follow.

If `cargo check` is still failing after all `impl` sites are migrated, the failures should be exclusively `expected 2 arguments, found 1` errors at call sites. Those are addressed in Task 3.

- [ ] **Step 7: Run the heartbit-core test suite — many tests will fail**

```bash
cargo test -p heartbit-core --lib 2>&1 | tail -20
```

Expected: many failures of the form `error: this method takes 2 arguments but 1 argument was supplied` — these are the tests calling `tool.execute(json!({}))` directly. Those tests will be fixed in Task 3 alongside the `AgentRunner` change.

For now, just confirm the failures are *call-site* failures, not logic regressions. If any test is failing for a reason other than the new arg count, stop and investigate.

- [ ] **Step 8: Commit (intentionally on a non-green workspace)**

```bash
git add -A crates/
git commit -m "refactor(core)!: Tool::execute takes &ExecutionContext (hard break)

BREAKING: every \`impl Tool\` must add \`_ctx: &ExecutionContext\` as the
first arg of \`execute()\`. All ~35 in-tree impls are migrated in this
commit. Call sites in AgentRunner / Restate / daemon / tests are
updated in the following commits.

Refs: docs/superpowers/specs/2026-05-07-heartbit-foundation-design.md F-AD-1"
```

(The workspace will be green by the end of Task 5; intermediate commits ship a partial tree by design — this is a coordinated trait migration.)

---

## Task 3: Thread `ExecutionContext` through `AgentRunner`

**Why:** Now that the `Tool` trait demands a context, every call site must construct one and pass it. `AgentRunner` is the primary call site (loop in `execute_tools_parallel`). It will construct an `ExecutionContext` per turn from the runner's existing audit fields and thread it into every tool dispatch.

**Files:**
- Modify: `crates/heartbit-core/src/agent/runner.rs`
- Test: same file's `#[cfg(test)] mod tests`

- [ ] **Step 1: Write the failing test that asserts context propagates to a tool**

Add to `crates/heartbit-core/src/agent/runner.rs` inside the existing test module:

```rust
#[tokio::test]
async fn execution_context_propagates_to_tool() {
    use std::sync::Mutex;
    use crate::ExecutionContext;
    use crate::llm::types::ToolDefinition;
    use crate::tool::{Tool, ToolOutput};

    struct CtxCapturingTool {
        captured_tenant: Arc<Mutex<Option<String>>>,
    }

    impl Tool for CtxCapturingTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "ctx_capture".into(),
                description: "Captures the tenant_id from ExecutionContext.".into(),
                input_schema: serde_json::json!({"type": "object"}),
            }
        }

        fn execute(
            &self,
            ctx: &ExecutionContext,
            _input: serde_json::Value,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<ToolOutput, crate::error::Error>> + Send + '_>,
        > {
            let captured = self.captured_tenant.clone();
            let tenant = ctx.tenant_id.clone();
            Box::pin(async move {
                *captured.lock().unwrap() = tenant;
                Ok(ToolOutput::success("ok"))
            })
        }
    }

    let captured = Arc::new(Mutex::new(None));
    let tool = Arc::new(CtxCapturingTool {
        captured_tenant: captured.clone(),
    });

    // Build a runner that has audit_tenant_id set, with no LLM (no actual run);
    // call execute_tools_parallel directly via the test hook used by other tests.
    // Use the same scaffolding pattern as the existing `execute_tools_parallel`
    // test in this file (see the `MockTool` test for reference).
    let runner = AgentRunnerBuilder::new()
        .name("test")
        .system_prompt("test")
        .max_turns(1)
        .tools(vec![tool as Arc<dyn Tool>])
        .audit_tenant_id("test-tenant")
        .build_for_test();   // helper introduced in Step 3

    let calls = vec![crate::llm::types::ToolCall {
        id: "c1".into(),
        name: "ctx_capture".into(),
        input: serde_json::json!({}),
    }];
    let _results = runner.execute_tools_parallel(&calls, 0).await;

    assert_eq!(
        captured.lock().unwrap().as_deref(),
        Some("test-tenant"),
        "tool did not receive the tenant_id from ExecutionContext"
    );
}
```

(If the `AgentRunnerBuilder` does not currently expose `audit_tenant_id` as a builder method, find the equivalent in `runner.rs` — it sets `self.audit_tenant_id` from a config field. Use whatever method exists; if none exists publicly, set the field directly via a `pub(crate)` test-only constructor or extend the builder.)

- [ ] **Step 2: Run the test — should fail to compile (`build_for_test` does not exist; `execute_tools_parallel` may be private)**

```bash
cargo test -p heartbit-core --lib execution_context_propagates_to_tool
```

Expected: compile errors.

- [ ] **Step 3: Add the test scaffolding to `AgentRunner`**

In `crates/heartbit-core/src/agent/runner.rs`, find the `AgentRunnerBuilder` and ensure:
- A `pub(crate) fn build_for_test(self) -> AgentRunner` method exists that builds without a provider (use a no-op provider stub already present in tests, or introduce a minimal one). If a similar test helper exists with a different name, use that name in the test instead.
- The `execute_tools_parallel` method must be reachable from tests. If currently `pub(super) async fn`, that's already test-reachable. No change needed.

- [ ] **Step 4: Update `execute_tools_parallel` to construct and pass `ExecutionContext`**

Find the function (around line 2021). Around the loop body where each tool is dispatched, before the `t.execute(input)` call, construct the context:

```rust
// Construct per-turn ExecutionContext from runner's audit fields.
let exec_ctx = crate::ExecutionContext {
    tenant_id: self.audit_tenant_id.clone(),
    user_id: self.audit_user_id.clone(),
    workspace: self.workspace.clone(),  // if AgentRunner has a workspace field; otherwise None
    credentials: self.credential_resolver.clone(),  // if present; otherwise None
    audit_sink: None,
};
```

Then change every `t.execute(input)` call inside this method to `t.execute(&exec_ctx, input)`.

If `AgentRunner` does not currently have a `workspace` or `credential_resolver` field, leave those as `None` for now — the persona registry (Task 6) will not consume them in Phase 0.

- [ ] **Step 5: Run the new test — should pass**

```bash
cargo test -p heartbit-core --lib execution_context_propagates_to_tool
```

Expected: `test result: ok. 1 passed`.

- [ ] **Step 6: Run the full heartbit-core test suite — should be green**

```bash
cargo test -p heartbit-core --lib 2>&1 | tail -10
```

Expected: all previously passing tests still pass; only call-site failures remain in dependent crates (`heartbit`, `heartbit-cli`).

If any test fails inside `heartbit-core` because it called `tool.execute(input)` with one arg, fix it: pass `&ExecutionContext::default()` as the first arg. Re-run.

- [ ] **Step 7: Commit**

```bash
git add crates/heartbit-core/src/agent/runner.rs
git commit -m "feat(core): thread ExecutionContext through AgentRunner

execute_tools_parallel constructs an ExecutionContext per turn from
the runner's existing audit fields (audit_tenant_id, audit_user_id,
workspace) and passes it into every tool's execute() call.

Refs: docs/superpowers/specs/2026-05-07-heartbit-foundation-design.md F-AD-1"
```

---

## Task 4: Thread `ExecutionContext` through the Restate workflow path

**Why:** Restate's `tool_call` activity also dispatches to `Tool::execute`. The activity boundary must construct an `ExecutionContext` from the workflow's invocation params (which already carry audit / tenant info).

**Files:**
- Modify: `crates/heartbit/src/workflow/agent_service.rs`
- Test: same file's `#[cfg(test)]` block

- [ ] **Step 1: Write the failing test asserting Restate `tool_call` constructs context**

Add to `crates/heartbit/src/workflow/agent_service.rs` test module (where `MockTool` lives):

```rust
#[tokio::test]
async fn tool_call_activity_constructs_context_from_invocation_params() {
    use heartbit_core::ExecutionContext;
    use std::sync::Mutex;

    struct CtxCapture {
        captured_tenant: Arc<Mutex<Option<String>>>,
    }

    impl heartbit_core::Tool for CtxCapture {
        fn definition(&self) -> heartbit_core::llm::types::ToolDefinition {
            heartbit_core::llm::types::ToolDefinition {
                name: "ctx_capture".into(),
                description: "captures tenant".into(),
                input_schema: serde_json::json!({"type":"object"}),
            }
        }
        fn execute(
            &self,
            ctx: &ExecutionContext,
            _input: serde_json::Value,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<heartbit_core::ToolOutput, heartbit_core::Error>> + Send + '_>> {
            let cap = self.captured_tenant.clone();
            let t = ctx.tenant_id.clone();
            Box::pin(async move {
                *cap.lock().unwrap() = t;
                Ok(heartbit_core::ToolOutput::success("ok"))
            })
        }
    }

    // Use the existing helper in this file (or write a small one) that
    // invokes the `tool_call` activity body directly, bypassing the Restate
    // ctx. The test asserts that when invocation params carry a tenant_id,
    // the captured value matches.
    // Example shape (exact API depends on the existing test scaffolding in
    // this file — match it):

    let captured = Arc::new(Mutex::new(None));
    let tool = Arc::new(CtxCapture { captured_tenant: captured.clone() });
    let tools: std::collections::HashMap<String, Arc<dyn heartbit_core::Tool>> =
        [("ctx_capture".to_string(), tool as Arc<dyn heartbit_core::Tool>)].into();

    let params = ToolCallParams {
        tool_name: "ctx_capture".into(),
        input: serde_json::json!({}),
        tenant_id: Some("restate-tenant".into()),
        user_id: Some("restate-user".into()),
    };

    // tool_call_inner is a small extracted helper introduced in this task —
    // the part of the Restate activity body that doesn't need the Restate ctx.
    let _result = tool_call_inner(&tools, params).await.unwrap();

    assert_eq!(captured.lock().unwrap().as_deref(), Some("restate-tenant"));
}
```

(`ToolCallParams` is the existing struct passed into the Restate activity. If it does not currently have `tenant_id` / `user_id` fields, add them in this task with `#[serde(default)]` for backward compat. If the activity already has access to these via Restate's invocation context, route them through `ToolCallParams` for testability.)

- [ ] **Step 2: Run test — should fail (helper missing or fields missing)**

```bash
cargo test -p heartbit --lib tool_call_activity_constructs_context_from_invocation_params
```

- [ ] **Step 3: Update `ToolCallParams` if needed**

If `ToolCallParams` lacks `tenant_id` / `user_id` fields, add them:

```rust
#[derive(serde::Serialize, serde::Deserialize)]
pub struct ToolCallParams {
    pub tool_name: String,
    pub input: serde_json::Value,
    #[serde(default)]
    pub tenant_id: Option<String>,
    #[serde(default)]
    pub user_id: Option<String>,
}
```

Existing serialised payloads remain compatible because of `#[serde(default)]`.

- [ ] **Step 4: Extract and update `tool_call_inner`**

Find the existing `tool_call` activity body in `agent_service.rs`. Extract the tool-dispatch logic into a private async helper:

```rust
async fn tool_call_inner(
    tools: &std::collections::HashMap<String, Arc<dyn heartbit_core::Tool>>,
    params: ToolCallParams,
) -> Result<heartbit_core::ToolOutput, heartbit_core::Error> {
    let tool = tools.get(&params.tool_name).ok_or_else(|| {
        heartbit_core::Error::Other(format!("tool '{}' not found", params.tool_name))
    })?;
    let ctx = heartbit_core::ExecutionContext {
        tenant_id: params.tenant_id.clone(),
        user_id: params.user_id.clone(),
        workspace: None,
        credentials: None,
        audit_sink: None,
    };
    tool.execute(&ctx, params.input).await
}
```

Then update the Restate activity to call `tool_call_inner` with the extracted params:

```rust
async fn tool_call(&self, ctx: Context, params: ToolCallParams) -> Result<...> {
    // ... existing pre-call audit / event emission ...
    tool_call_inner(&self.tools, params).await
        .map_err(|e| /* existing error mapping */)
}
```

(Match the existing error mapping and audit hooks present in the file.)

- [ ] **Step 5: Run test — should pass**

```bash
cargo test -p heartbit --lib tool_call_activity_constructs_context_from_invocation_params
```

- [ ] **Step 6: Workspace test — heartbit crate**

```bash
cargo test -p heartbit --lib 2>&1 | tail -10
```

Expected: green except for any remaining call sites broken by Task 2 (which Task 5 finishes).

- [ ] **Step 7: Commit**

```bash
git add crates/heartbit/src/workflow/agent_service.rs
git commit -m "feat(workflow): construct ExecutionContext in Restate tool_call activity

ToolCallParams gains optional tenant_id/user_id fields (serde default
for backward compat). The activity body extracts tool_call_inner that
constructs an ExecutionContext and calls Tool::execute with it.

Refs: docs/superpowers/specs/2026-05-07-heartbit-foundation-design.md F-AD-1"
```

---

## Task 5: Populate `ExecutionContext` in `DaemonCore::dispatch_command`

**Why:** The daemon path is the third entry point that dispatches tasks to agents. It already has `audit_user_id` / `audit_tenant_id` on `DaemonCommand::SubmitTask`. Thread these into the `AgentRunner` it constructs so `execute_tools_parallel` (Task 3) can use them.

**Files:**
- Modify: `crates/heartbit/src/daemon/core.rs`
- Test: same file's `#[cfg(test)]` block

- [ ] **Step 1: Identify the `AgentRunner` construction site**

Read `crates/heartbit/src/daemon/core.rs:485` onward. Find where `AgentRunner` (or `Orchestrator`) is built per command in `dispatch_command`. Look for `AgentRunnerBuilder::new()` or the equivalent.

- [ ] **Step 2: Write the failing test asserting tenant propagation through dispatch**

Add to `crates/heartbit/src/daemon/core.rs` test module:

```rust
#[tokio::test]
async fn dispatch_command_propagates_tenant_to_agent_runner() {
    // Construct a DaemonCore with a mock provider and a single CtxCapturingTool.
    // Submit a DaemonCommand::SubmitTask with tenant_id = "daemon-tenant".
    // Assert the tool's execute() saw tenant_id = "daemon-tenant".
    //
    // Use the existing pattern from other tests in this file for constructing
    // a test DaemonCore (in-process channel mode, mock provider).
    // ...
}
```

(Match the existing test scaffolding in `daemon/core.rs` — the file already has tests that construct test DaemonCore instances.)

- [ ] **Step 3: Run test — should fail**

```bash
cargo test -p heartbit --lib dispatch_command_propagates_tenant_to_agent_runner
```

- [ ] **Step 4: Update `dispatch_command` to propagate audit fields to the runner**

Find where `AgentRunner` is built per task in `dispatch_command`. After the existing audit fields are set (e.g., `.audit_user_id(...)`, `.audit_tenant_id(...)`), confirm they exist; if not, add them. The `AgentRunner` already reads these for context construction (Task 3), so the daemon's job is just to pass them through.

If the daemon already passes audit fields to the runner, this task may be a no-op verification. The test from Step 2 still validates end-to-end propagation.

If the runner builder lacks `audit_tenant_id` / `audit_user_id` setters, add them — they must mirror the runner fields used in Task 3.

- [ ] **Step 5: Run test — should pass**

```bash
cargo test -p heartbit --lib dispatch_command_propagates_tenant_to_agent_runner
```

- [ ] **Step 6: Workspace test — full crate**

```bash
cargo test --workspace 2>&1 | tail -10
```

Expected: workspace green at this point (Tasks 2–5 collectively complete the trait migration).

If any tests still fail, they are call sites that pass `tool.execute(input)` with the old signature. Fix each by passing `&ExecutionContext::default()` as the first arg.

- [ ] **Step 7: Commit**

```bash
git add crates/heartbit/src/daemon/core.rs
git commit -m "feat(daemon): propagate tenant_id/user_id to AgentRunner via dispatch

DaemonCore::dispatch_command threads SubmitTask audit fields into the
AgentRunner so execute_tools_parallel (heartbit-core) populates the
ExecutionContext correctly under multi-tenant.

Workspace is green again after this commit.

Refs: docs/superpowers/specs/2026-05-07-heartbit-foundation-design.md F-AD-1"
```

---

## Task 6: `Persona` trait + `PersonaRegistry`

**Why:** Provide the small, generic abstraction that future persona crates (`heartbit-ghost`, etc.) register into. Empty in this release.

**Files:**
- Create: `crates/heartbit-core/src/persona/mod.rs`
- Create: `crates/heartbit-core/src/persona/types.rs`
- Modify: `crates/heartbit-core/src/lib.rs`

- [ ] **Step 1: Write the failing tests for `PersonaRegistry`**

Create `crates/heartbit-core/src/persona/mod.rs`:

```rust
//! Persona registry: a small abstraction so concrete persona crates
//! (e.g. `heartbit-ghost`) plug in identically. Empty in Phase 0;
//! concrete personas land in Phase 1.

pub mod types;

use std::collections::HashMap;
use std::sync::Arc;

pub use types::{
    AuthorshipMode, PersonaExpansion, PersonaParams, ReviewSpec, TriggerSpec,
};

use crate::error::Error;

/// A persona is a recipe that expands into agent configurations, tools,
/// triggers, and a review spec. Implementations live in dedicated crates
/// (e.g. `heartbit-ghost`) and register themselves into a `PersonaRegistry`
/// at startup.
pub trait Persona: Send + Sync {
    /// Stable persona identifier. Convention: `<crate_short_name>:<recipe>`,
    /// e.g. `"heartbit-ghost:x"`.
    fn name(&self) -> &str;

    /// One-line human-readable description.
    fn description(&self) -> &str;

    /// Persona version (semver-ish). Useful for audit logs.
    fn version(&self) -> &str;

    /// Expand the persona into runtime artifacts using the per-instance params.
    fn expand(&self, params: &PersonaParams) -> Result<PersonaExpansion, Error>;
}

/// In-memory registry of personas by name. Constructed at startup; concrete
/// personas (from dependent crates) call `register()` during initialization.
pub struct PersonaRegistry {
    personas: HashMap<String, Arc<dyn Persona>>,
}

impl Default for PersonaRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl PersonaRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            personas: HashMap::new(),
        }
    }

    /// Register a persona. The persona's `name()` is used as the key.
    /// Registering the same name twice replaces the previous entry (last-write-wins).
    pub fn register(&mut self, persona: Arc<dyn Persona>) {
        self.personas.insert(persona.name().to_string(), persona);
    }

    /// Look up a persona by name.
    pub fn get(&self, name: &str) -> Option<Arc<dyn Persona>> {
        self.personas.get(name).cloned()
    }

    /// List the names of all registered personas.
    pub fn list(&self) -> Vec<&str> {
        self.personas.keys().map(|k| k.as_str()).collect()
    }

    /// Number of registered personas.
    pub fn len(&self) -> usize {
        self.personas.len()
    }

    /// True if no personas are registered.
    pub fn is_empty(&self) -> bool {
        self.personas.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyPersona;

    impl Persona for DummyPersona {
        fn name(&self) -> &str {
            "dummy:p"
        }
        fn description(&self) -> &str {
            "test persona"
        }
        fn version(&self) -> &str {
            "0.1.0"
        }
        fn expand(&self, _params: &PersonaParams) -> Result<PersonaExpansion, Error> {
            Ok(PersonaExpansion::default())
        }
    }

    #[test]
    fn registry_starts_empty() {
        let r = PersonaRegistry::new();
        assert!(r.is_empty());
        assert_eq!(r.len(), 0);
        assert!(r.list().is_empty());
        assert!(r.get("anything").is_none());
    }

    #[test]
    fn register_and_get_round_trip() {
        let mut r = PersonaRegistry::new();
        r.register(Arc::new(DummyPersona));
        assert_eq!(r.len(), 1);
        assert!(r.get("dummy:p").is_some());
        assert_eq!(r.list(), vec!["dummy:p"]);
    }

    #[test]
    fn register_same_name_replaces() {
        let mut r = PersonaRegistry::new();
        r.register(Arc::new(DummyPersona));
        r.register(Arc::new(DummyPersona));
        assert_eq!(r.len(), 1);
    }
}
```

- [ ] **Step 2: Create `persona/types.rs`**

Create `crates/heartbit-core/src/persona/types.rs`:

```rust
//! Persona-related value types. Most variants are intentionally empty in
//! Phase 0: concrete persona crates extend them as they land.

use crate::config::AgentConfig;
use crate::config::OrchestratorConfig;
use crate::tool::Tool;
use std::sync::Arc;

/// Per-instance parameters supplied at expansion time. Constructed from the
/// `[[persona]]` block in `HeartbitConfig`.
#[derive(Debug, Clone, Default)]
pub struct PersonaParams {
    /// Persona-specific overrides (free-form TOML; interpreted by `expand()`).
    pub overrides: toml::Value,
    /// Glob for environment-variable credential lookup, e.g. `"X_*"`.
    pub credentials_env: Option<String>,
    /// Authorship mode for posts/communications produced by the persona.
    pub authorship_mode: AuthorshipMode,
}

/// What a persona expands into at startup.
#[derive(Default)]
pub struct PersonaExpansion {
    /// Sub-agents the persona requires.
    pub agents: Vec<AgentConfig>,
    /// Orchestrator config for the persona's pipeline.
    pub orchestrator: OrchestratorConfig,
    /// Tool instances contributed by the persona.
    pub tools: Vec<Arc<dyn Tool>>,
    /// Trigger specs (cron / sensors / mention polling / manual). Empty in Phase 0.
    pub triggers: Vec<TriggerSpec>,
    /// Optional review channel spec. None in Phase 0.
    pub review: Option<ReviewSpec>,
}

impl std::fmt::Debug for PersonaExpansion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PersonaExpansion")
            .field("agents", &self.agents.len())
            .field("tools", &self.tools.len())
            .field("triggers", &self.triggers.len())
            .field("review", &self.review.is_some())
            .finish()
    }
}

/// Authorship mode declared per persona instance.
///
/// Operators are responsible for ensuring the chosen mode is permitted under
/// the target platform's terms of service and applicable regulation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthorshipMode {
    /// A human approves every action before it goes out.
    #[default]
    HumanAssisted,
    /// Fully autonomous; the agent's bot status is publicly disclosed
    /// (account labelled, ToS-compliant).
    AutonomousDisclosed,
    /// Fully autonomous; the agent's bot status is not disclosed. Operator
    /// owns regulatory compliance.
    AutonomousUndisclosed,
}

/// Trigger spec — concrete variants land with their consumers (Phase 1).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum TriggerSpec {}

/// Review channel spec — concrete variants land with their consumers (Phase 1).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ReviewSpec {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn authorship_mode_default_is_human_assisted() {
        assert_eq!(AuthorshipMode::default(), AuthorshipMode::HumanAssisted);
    }

    #[test]
    fn authorship_mode_serde_round_trip() {
        let mode = AuthorshipMode::AutonomousUndisclosed;
        let json = serde_json::to_string(&mode).unwrap();
        assert_eq!(json, "\"autonomous_undisclosed\"");
        let parsed: AuthorshipMode = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, mode);
    }

    #[test]
    fn persona_params_default_authorship_is_human_assisted() {
        let p = PersonaParams::default();
        assert_eq!(p.authorship_mode, AuthorshipMode::HumanAssisted);
    }

    #[test]
    fn persona_expansion_default_is_empty() {
        let e = PersonaExpansion::default();
        assert!(e.agents.is_empty());
        assert!(e.tools.is_empty());
        assert!(e.triggers.is_empty());
        assert!(e.review.is_none());
    }
}
```

- [ ] **Step 3: Wire the new module into `lib.rs`**

In `crates/heartbit-core/src/lib.rs`, alongside other module declarations:

```rust
pub mod persona;
```

In the re-exports section:

```rust
pub use persona::{
    AuthorshipMode, Persona, PersonaExpansion, PersonaParams, PersonaRegistry,
    ReviewSpec, TriggerSpec,
};
```

- [ ] **Step 4: Run the tests — all 7 pass**

```bash
cargo test -p heartbit-core --lib persona
```

Expected: `test result: ok. 7 passed`.

- [ ] **Step 5: Workspace quality gate**

```bash
cargo fmt -- --check && cargo clippy -p heartbit-core -- -D warnings
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add crates/heartbit-core/src/persona/ crates/heartbit-core/src/lib.rs
git commit -m "feat(core): Persona trait + PersonaRegistry (empty in Phase 0)

Adds the small abstraction for persona recipes. Concrete personas
(heartbit-ghost, future heartbit-coder) will register into a shared
PersonaRegistry at startup. AuthorshipMode is the only non-empty
enum in this commit; TriggerSpec/ReviewSpec are non_exhaustive
empty enums until consumers add variants in Phase 1.

Refs: docs/superpowers/specs/2026-05-07-heartbit-foundation-design.md F-AD-2"
```

---

## Task 7: `[[persona]]` config schema in `HeartbitConfig`

**Why:** Make `daemon.toml` files referencing `recipe = "heartbit-ghost:x"` parse and validate lexically once Phase 1 ships, without further core changes.

**Files:**
- Create: `crates/heartbit-core/src/config/persona.rs`
- Modify: `crates/heartbit-core/src/config/mod.rs` (add `personas: Vec<PersonaConfig>` to `HeartbitConfig`, plus validation)

- [ ] **Step 1: Write the failing tests**

Create `crates/heartbit-core/src/config/persona.rs`:

```rust
//! `[[persona]]` config block for declaring persona instances in
//! `heartbit.toml` / `daemon.toml`.

use serde::Deserialize;

use crate::error::Error;
use crate::persona::AuthorshipMode;

/// Autonomy phase progression for a persona instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PersonaPhase {
    /// 100% candidates routed to review.
    Calibration,
    /// 80% review / 20% auto-publish (high-confidence).
    Supervised,
    /// 10% review (sampled) / 90% auto-publish.
    Autonomous,
    /// Only flagged candidates routed to review.
    Sentinel,
}

impl Default for PersonaPhase {
    fn default() -> Self {
        Self::Calibration
    }
}

/// One `[[persona]]` block.
#[derive(Debug, Clone, Deserialize)]
pub struct PersonaConfig {
    /// Local instance name (must be unique within the config file).
    pub name: String,
    /// Recipe key in the form `<crate_short>:<recipe>`, e.g. `"heartbit-ghost:x"`.
    pub recipe: String,
    /// Glob for env-var credential lookup, e.g. `"X_*"`.
    #[serde(default)]
    pub credentials_env: Option<String>,
    /// Authorship mode (default `human_assisted`).
    #[serde(default)]
    pub authorship_mode: AuthorshipMode,
    /// Initial autonomy phase.
    #[serde(default)]
    pub phase: PersonaPhase,
    /// Persona-specific overrides (free-form; interpreted by the recipe's `expand()`).
    #[serde(default)]
    pub overrides: toml::Value,
}

impl PersonaConfig {
    /// Lexical validation: recipe key parses, name is non-empty.
    /// Does not consult the registry.
    pub fn validate(&self) -> Result<(), Error> {
        if self.name.trim().is_empty() {
            return Err(Error::Config("persona name must be non-empty".into()));
        }
        if !self.recipe.contains(':') {
            return Err(Error::Config(format!(
                "persona '{}' recipe '{}' must be of the form '<crate>:<name>'",
                self.name, self.recipe
            )));
        }
        let (lhs, rhs) = self.recipe.split_once(':').unwrap();
        if lhs.trim().is_empty() || rhs.trim().is_empty() {
            return Err(Error::Config(format!(
                "persona '{}' recipe '{}' has empty crate or name component",
                self.name, self.recipe
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(toml_text: &str) -> Result<PersonaConfig, toml::de::Error> {
        toml::from_str::<PersonaConfig>(toml_text)
    }

    #[test]
    fn parses_minimal_persona() {
        let c: PersonaConfig = parse(
            r#"
            name = "x"
            recipe = "heartbit-ghost:x"
            "#,
        )
        .expect("parses");
        assert_eq!(c.name, "x");
        assert_eq!(c.recipe, "heartbit-ghost:x");
        assert_eq!(c.authorship_mode, AuthorshipMode::HumanAssisted);
        assert_eq!(c.phase, PersonaPhase::Calibration);
    }

    #[test]
    fn parses_full_persona() {
        let c: PersonaConfig = parse(
            r#"
            name = "x"
            recipe = "heartbit-ghost:x"
            credentials_env = "X_*"
            authorship_mode = "autonomous_undisclosed"
            phase = "supervised"
            "#,
        )
        .expect("parses");
        assert_eq!(c.credentials_env.as_deref(), Some("X_*"));
        assert_eq!(c.authorship_mode, AuthorshipMode::AutonomousUndisclosed);
        assert_eq!(c.phase, PersonaPhase::Supervised);
    }

    #[test]
    fn validate_rejects_empty_name() {
        let c = PersonaConfig {
            name: "".into(),
            recipe: "heartbit-ghost:x".into(),
            credentials_env: None,
            authorship_mode: AuthorshipMode::default(),
            phase: PersonaPhase::default(),
            overrides: toml::Value::Table(toml::map::Map::new()),
        };
        let err = c.validate().unwrap_err();
        assert!(matches!(err, Error::Config(s) if s.contains("non-empty")));
    }

    #[test]
    fn validate_rejects_recipe_without_colon() {
        let c = PersonaConfig {
            name: "x".into(),
            recipe: "heartbit-ghost-x".into(),
            credentials_env: None,
            authorship_mode: AuthorshipMode::default(),
            phase: PersonaPhase::default(),
            overrides: toml::Value::Table(toml::map::Map::new()),
        };
        let err = c.validate().unwrap_err();
        assert!(matches!(err, Error::Config(s) if s.contains("'<crate>:<name>'")));
    }

    #[test]
    fn validate_rejects_empty_lhs_or_rhs() {
        for bad in [":x", "heartbit-ghost:", ":"] {
            let c = PersonaConfig {
                name: "x".into(),
                recipe: bad.into(),
                credentials_env: None,
                authorship_mode: AuthorshipMode::default(),
                phase: PersonaPhase::default(),
                overrides: toml::Value::Table(toml::map::Map::new()),
            };
            let err = c.validate().unwrap_err();
            assert!(matches!(err, Error::Config(_)), "expected Config error for recipe {:?}", bad);
        }
    }

    #[test]
    fn rejects_unknown_phase() {
        let result = parse(
            r#"
            name = "x"
            recipe = "heartbit-ghost:x"
            phase = "ludicrous"
            "#,
        );
        assert!(result.is_err());
    }
}
```

- [ ] **Step 2: Run the tests — should fail (module not declared)**

```bash
cargo test -p heartbit-core --lib config::persona
```

- [ ] **Step 3: Wire the module + add `personas` to `HeartbitConfig`**

In `crates/heartbit-core/src/config/mod.rs`:

1. Add `pub mod persona;` near the top of the file alongside other config modules.
2. Re-export `pub use persona::{PersonaConfig, PersonaPhase};` in the same module's `pub use` block (or wherever existing re-exports live).
3. Add a field to `HeartbitConfig` (around line 169–209):

```rust
/// Persona instances declared in this config (Phase 0: parsed and
/// lexically validated; the registry lookup happens at daemon startup
/// once persona crates are loaded).
#[serde(default, rename = "persona")]
pub personas: Vec<PersonaConfig>,
```

Use `rename = "persona"` so TOML readers write `[[persona]]` (singular block name) — matching the spec.

4. In `HeartbitConfig::validate()` (the existing validation method that runs after parse — find it near the `from_toml` impl), add:

```rust
// Persona blocks: lexical validation + duplicate-name check.
let mut seen_persona_names = std::collections::HashSet::new();
for persona in &self.personas {
    persona.validate()?;
    if !seen_persona_names.insert(persona.name.clone()) {
        return Err(Error::Config(format!(
            "duplicate persona name: '{}'",
            persona.name
        )));
    }
}
```

(If `HeartbitConfig` does not have a discrete `validate()` method but instead does inline checks in `from_toml`, add the validation block there.)

- [ ] **Step 4: Add a duplicate-name test in `config/mod.rs`**

In `crates/heartbit-core/src/config/mod.rs` test module, add:

```rust
#[test]
fn rejects_duplicate_persona_names() {
    let toml_text = r#"
        [[provider]]
        name = "anthropic"
        model = "claude-sonnet-4-20250514"

        [[persona]]
        name = "x"
        recipe = "heartbit-ghost:x"

        [[persona]]
        name = "x"
        recipe = "heartbit-ghost:x"
    "#;
    let err = HeartbitConfig::from_toml(toml_text).unwrap_err();
    let msg = format!("{:?}", err);
    assert!(msg.contains("duplicate persona name"), "got: {}", msg);
}

#[test]
fn parses_persona_block_round_trip() {
    let toml_text = r#"
        [provider]
        name = "anthropic"
        model = "claude-sonnet-4-20250514"

        [[persona]]
        name = "x"
        recipe = "heartbit-ghost:x"
        authorship_mode = "autonomous_undisclosed"
        phase = "calibration"
    "#;
    let config = HeartbitConfig::from_toml(toml_text).expect("parses");
    assert_eq!(config.personas.len(), 1);
    assert_eq!(config.personas[0].name, "x");
    assert_eq!(config.personas[0].recipe, "heartbit-ghost:x");
}
```

(Match the actual `from_toml` API and provider block syntax used elsewhere in the file's existing tests.)

- [ ] **Step 5: Run all config tests — should pass**

```bash
cargo test -p heartbit-core --lib config
```

Expected: all green, including 6 new persona-config tests + 2 new HeartbitConfig tests.

- [ ] **Step 6: Workspace quality gate**

```bash
cargo fmt -- --check && cargo clippy -p heartbit-core -- -D warnings
```

- [ ] **Step 7: Commit**

```bash
git add crates/heartbit-core/src/config/
git commit -m "feat(config): [[persona]] block in HeartbitConfig (lexical validation only)

PersonaConfig accepts name, recipe, credentials_env, authorship_mode,
phase, overrides. Validation: non-empty name, recipe in '<crate>:<name>'
form, no duplicate names. Registry lookup happens at daemon startup;
not in this commit.

Refs: docs/superpowers/specs/2026-05-07-heartbit-foundation-design.md F-AD-3"
```

---

## Task 8: `heartbit persona <subcommand>` CLI surface (functional shells)

**Why:** Wire the user-facing CLI surface against the empty registry. Each subcommand parses fully via `clap`, has accurate `--help`, and returns a canonical not-found error. When Phase 1 (heartbit-ghost) registers personas, these subcommands light up with no further CLI changes.

**Files:**
- Create: `crates/heartbit-cli/src/persona.rs`
- Modify: `crates/heartbit-cli/src/main.rs` (add `Persona` variant to the `Commands` enum + dispatch)

- [ ] **Step 1: Write the persona module with full clap parsing**

Create `crates/heartbit-cli/src/persona.rs`:

```rust
//! `heartbit persona <sub>` subcommand surface.
//!
//! Functional shells against the (empty in Phase 0) `PersonaRegistry`. Each
//! subcommand returns a canonical "no personas registered" error. Once
//! persona crates (e.g. `heartbit-ghost`) register their recipes, these
//! subcommands light up without any CLI changes.

use anyhow::{Result, anyhow};
use clap::Subcommand;

use heartbit_core::PersonaRegistry;

#[derive(Debug, Subcommand)]
pub enum PersonaCommand {
    /// List registered personas.
    List,

    /// Show the configured persona instance (expanded TOML).
    Show {
        /// Persona instance name as declared in `[[persona]]`.
        name: String,
    },

    /// Run the persona once with a one-off prompt; print result to stdout.
    Run {
        /// Persona instance name.
        name: String,
        /// Run a single dry-run prompt without posting.
        #[arg(long, value_name = "PROMPT")]
        once: String,
    },

    /// Manage the persona's reference corpus.
    Corpus {
        #[command(subcommand)]
        sub: CorpusCommand,
    },

    /// Manage the persona's blended style profile.
    Profile {
        #[command(subcommand)]
        sub: ProfileCommand,
    },

    /// Set the persona's autonomy phase.
    Phase {
        /// Persona instance name.
        name: String,
        /// Phase: calibration | supervised | autonomous | sentinel.
        #[arg(long)]
        set: String,
    },

    /// Halt this persona on a running daemon.
    Pause {
        /// Persona instance name.
        name: String,
    },

    /// Resume this persona on a running daemon.
    Resume {
        /// Persona instance name.
        name: String,
    },

    /// Export the user-preference dataset for external training (L3).
    ExportPreferences {
        /// Persona instance name.
        name: String,
        /// Output format. Default: jsonl.
        #[arg(long, default_value = "jsonl")]
        format: String,
    },

    /// Show recent posts with their full audit trail.
    Audit {
        /// Persona instance name.
        name: String,
        /// Time window, e.g. `24h`, `7d`.
        #[arg(long, default_value = "24h")]
        since: String,
    },
}

#[derive(Debug, Subcommand)]
pub enum CorpusCommand {
    /// Add an exemplar corpus from a JSONL file.
    Add {
        /// Writer handle (without `@`), e.g. `karpathy`.
        writer: String,
        /// Path to a JSONL file of posts.
        path: std::path::PathBuf,
    },
    /// List the corpus sources for a persona.
    List {
        /// Persona instance name.
        name: String,
    },
}

#[derive(Debug, Subcommand)]
pub enum ProfileCommand {
    /// Recompute the blended style profile from the current corpus.
    Rebuild {
        /// Persona instance name.
        name: String,
    },
    /// Diff two profile versions.
    Diff {
        /// Persona instance name.
        name: String,
        /// First version, e.g. `v3`.
        v1: String,
        /// Second version, e.g. `v4`.
        v2: String,
    },
}

const NO_PERSONAS_REGISTERED: &str =
    "No personas registered. (heartbit-ghost or another persona crate must be linked into this build.)";

/// Dispatch a `persona` subcommand against the (Phase 0: empty) registry.
pub async fn run(cmd: PersonaCommand) -> Result<()> {
    let registry = PersonaRegistry::new();
    dispatch(cmd, &registry).await
}

async fn dispatch(cmd: PersonaCommand, registry: &PersonaRegistry) -> Result<()> {
    match cmd {
        PersonaCommand::List => {
            let names = registry.list();
            if names.is_empty() {
                println!("No personas registered.");
            } else {
                for name in names {
                    println!("{name}");
                }
            }
            Ok(())
        }
        PersonaCommand::Show { name }
        | PersonaCommand::Run { name, .. }
        | PersonaCommand::Phase { name, .. }
        | PersonaCommand::Pause { name }
        | PersonaCommand::Resume { name }
        | PersonaCommand::ExportPreferences { name, .. }
        | PersonaCommand::Audit { name, .. } => {
            if registry.get(&name).is_none() {
                return Err(anyhow!("persona '{name}' not found. {NO_PERSONAS_REGISTERED}"));
            }
            // Bodies for non-empty registry land in Phase 1 alongside concrete persona crates.
            Err(anyhow!(
                "persona subcommand bodies are not implemented in Phase 0; this CLI shell ships with the foundation release."
            ))
        }
        PersonaCommand::Corpus { sub } => match sub {
            CorpusCommand::Add { .. } | CorpusCommand::List { .. } => {
                Err(anyhow!("corpus management requires a registered persona. {NO_PERSONAS_REGISTERED}"))
            }
        },
        PersonaCommand::Profile { sub } => match sub {
            ProfileCommand::Rebuild { .. } | ProfileCommand::Diff { .. } => {
                Err(anyhow!("profile management requires a registered persona. {NO_PERSONAS_REGISTERED}"))
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn list_against_empty_registry_prints_message() {
        // The println! goes to stdout; we just assert dispatch returns Ok.
        let r = PersonaRegistry::new();
        let result = dispatch(PersonaCommand::List, &r).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn show_against_empty_registry_returns_error() {
        let r = PersonaRegistry::new();
        let result = dispatch(
            PersonaCommand::Show {
                name: "x".into(),
            },
            &r,
        )
        .await;
        let err = result.unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("persona 'x' not found"));
        assert!(msg.contains("No personas registered"));
    }

    #[tokio::test]
    async fn corpus_add_against_empty_registry_returns_error() {
        let r = PersonaRegistry::new();
        let result = dispatch(
            PersonaCommand::Corpus {
                sub: CorpusCommand::Add {
                    writer: "karpathy".into(),
                    path: std::path::PathBuf::from("/tmp/x.jsonl"),
                },
            },
            &r,
        )
        .await;
        let err = result.unwrap_err();
        assert!(format!("{err}").contains("No personas registered"));
    }

    #[tokio::test]
    async fn profile_rebuild_against_empty_registry_returns_error() {
        let r = PersonaRegistry::new();
        let result = dispatch(
            PersonaCommand::Profile {
                sub: ProfileCommand::Rebuild { name: "x".into() },
            },
            &r,
        )
        .await;
        let err = result.unwrap_err();
        assert!(format!("{err}").contains("No personas registered"));
    }
}
```

- [ ] **Step 2: Wire `Persona` variant into the top-level `Commands` enum and dispatch**

In `crates/heartbit-cli/src/main.rs`:

1. Add the module declaration near the top (after other `mod` declarations):

```rust
mod persona;
```

2. Find the top-level `Commands` enum (the one used by clap's main subcommand router). Add a new variant:

```rust
/// Manage personas (list, run, configure, audit).
Persona {
    #[command(subcommand)]
    sub: persona::PersonaCommand,
},
```

3. In the `match cli.command { ... }` dispatch (or however main routes subcommands), add an arm:

```rust
Commands::Persona { sub } => persona::run(sub).await?,
```

(Match the existing pattern used by other subcommands like `serve`, `submit`. If main returns `anyhow::Result<()>`, the `?` works directly; otherwise adapt the error mapping.)

- [ ] **Step 3: Run the persona module tests — all 4 pass**

```bash
cargo test -p heartbit-cli --lib persona
```

Expected: 4 passing.

- [ ] **Step 4: Manual smoke test — `--help` is complete**

```bash
cargo run -p heartbit-cli -- persona --help
```

Expected: lists all subcommands (list, show, run, corpus, profile, phase, pause, resume, export-preferences, audit) with one-line descriptions.

```bash
cargo run -p heartbit-cli -- persona list
```

Expected: prints `No personas registered.` and exits 0.

```bash
cargo run -p heartbit-cli -- persona show x
```

Expected: prints an error message containing `persona 'x' not found` and `No personas registered` and exits non-zero.

```bash
cargo run -p heartbit-cli -- persona corpus add karpathy /tmp/x.jsonl
```

Expected: prints an error message containing `corpus management requires a registered persona` and `No personas registered`.

- [ ] **Step 5: Workspace quality gate**

```bash
cargo fmt -- --check && cargo clippy -p heartbit-cli -- -D warnings
```

- [ ] **Step 6: Commit**

```bash
git add crates/heartbit-cli/src/persona.rs crates/heartbit-cli/src/main.rs
git commit -m "feat(cli): heartbit persona subcommand surface (functional shells)

Adds the heartbit persona <sub> command tree wired against the (Phase 0:
empty) PersonaRegistry. Each subcommand parses fully via clap and
returns a canonical 'No personas registered' error. Phase 1
(heartbit-ghost) lights up the bodies without further CLI changes.

Subcommands: list, show, run, corpus add/list, profile rebuild/diff,
phase --set, pause, resume, export-preferences, audit.

Refs: docs/superpowers/specs/2026-05-07-heartbit-foundation-design.md F-AD-4"
```

---

## Task 9: Final quality gate + acceptance verification

**Why:** Confirm the foundation release meets every acceptance criterion in the spec.

**Files:** none (verification only).

- [ ] **Step 1: Workspace quality gate**

```bash
cd /home/pleclech/projects/heartbit
cargo fmt -- --check && cargo clippy --workspace -- -D warnings && cargo test --workspace 2>&1 | tail -20
```

Expected: zero formatting issues, zero clippy warnings, all tests green across the workspace.

If any failure surfaces, classify it:
- Formatting: run `cargo fmt`, re-run check.
- Clippy: fix the lint or, if it's a pre-existing lint not caused by this work, document why and add to a follow-up.
- Test failure: classify as call-site bug (Tasks 2–5 should have fixed all of these), pre-existing flake, or actual regression. Real regressions block the release.

- [ ] **Step 2: Manual smoke against acceptance criteria**

For each acceptance criterion in the foundation spec (§Acceptance criteria), verify:

```bash
# 1. New trait signature is in place; no bridge code.
grep -n "fn execute" crates/heartbit-core/src/tool/mod.rs
# Expected: only one execute() signature, taking ctx and input.

# 2. PersonaRegistry is exported.
cargo doc --no-deps -p heartbit-core 2>&1 | grep -E "PersonaRegistry|Persona "
# Or: confirm it appears in lib.rs `pub use persona::...`.

# 3. daemon.toml referencing heartbit-ghost:x parses lexically.
cat > /tmp/test-persona.toml <<EOF
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[persona]]
name = "x"
recipe = "heartbit-ghost:x"
phase = "calibration"
EOF

# Use the existing config validation entry point (e.g. `heartbit run -c <file> --validate`
# if it exists, or write a one-off test). Since the CLI doesn't have a validate-only
# flag, this is verified by Task 7's tests.

# 4. CLI persona surface lights up.
cargo run -p heartbit-cli -- persona --help
cargo run -p heartbit-cli -- persona list
cargo run -p heartbit-cli -- persona show x
```

- [ ] **Step 3: Update CHANGELOG.md (if present)**

Add an entry to `CHANGELOG.md` under a new release section:

```markdown
### Breaking
- `Tool::execute` now takes `&ExecutionContext` as its first arg. All in-tree
  tools migrated. External `impl Tool` consumers must add `_ctx:
  &heartbit_core::ExecutionContext` as the first parameter of their
  `execute()` method.

### Added
- `heartbit_core::ExecutionContext` carries per-request tenant_id, user_id,
  workspace, credential resolver, audit sink.
- `heartbit_core::CredentialResolver` and `heartbit_core::AuditSink` traits
  for tenant-scoped secret resolution and audit logging.
- `heartbit_core::PersonaRegistry` and `Persona` trait for persona-recipe
  registration.
- `[[persona]]` block in `HeartbitConfig` (lexical validation; registry
  lookup happens at daemon startup).
- `heartbit persona <sub>` CLI subcommand surface.
```

- [ ] **Step 4: Final commit**

```bash
git add CHANGELOG.md
git commit -m "docs: changelog for foundation release (Phase 0 / heartbit-ghost prereq)"
```

- [ ] **Step 5: Verify the full commit log of this work**

```bash
git log --oneline | head -10
```

Expected: 8 task commits (one per Task 1, 2, 3, 4, 5, 6, 7, 8) plus the changelog commit, in clear sequence.

---

## Out of scope (per spec)

- The `heartbit-ghost` crate itself (Phase 1 — separate plan)
- Any concrete persona, recipe, or tool addition
- Voice modeling, corpus, blend, profile schema
- A/B feedback loop, Telegram review wiring
- Sensor / cron / mention-poll handlers
- CLI debt sweep beyond the new `persona` subcommand surface
- Harness mechanics refactor (`BuiltinToolsConfig`, dead `ToolRisk`, schemars-driven schemas)
- Rate-limiting middleware, streaming output, lifecycle hooks, cancellation
- Training / fine-tuning infrastructure

## Reference

- Spec: `docs/superpowers/specs/2026-05-07-heartbit-foundation-design.md`
- Phase 1 follow-up: `docs/superpowers/specs/2026-05-07-heartbit-ghost-x-agent-design.md`
