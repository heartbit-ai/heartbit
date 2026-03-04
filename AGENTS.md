# AGENTS.md — Heartbit Agent Guidance

This file follows the AGENTS.md convention for project-specific agent guidance (analogous to CLAUDE.md for Claude, or system prompts for other agents). AI agents cloning or working in this repository should read this file before taking any action.

## Project Identity

**Heartbit** is a multi-agent enterprise runtime written in Rust. It is not a wrapper around another framework — all agent loop logic, tool execution, memory management, and LLM integration are implemented from scratch.

- **Language**: Rust (stable toolchain)
- **Crates**: `heartbit` (lib), `heartbit-cli` (bin)
- **Tests**: `cargo test` — all must pass before any commit

## Mandatory Quality Gate

Before committing any change, all three checks must pass:

```bash
cargo fmt -- --check && cargo clippy -- -D warnings && cargo test
```

No warnings are allowed. No `.unwrap()` in library code (`crates/heartbit/src/`).

## Architecture Rules

### Flat Agent Hierarchy
- Orchestrator spawns sub-agents. Sub-agents **never spawn** further sub-agents.
- Do not add recursive spawning.

### Three Execution Paths
| Path | Entry point | Key types |
|------|------------|-----------|
| Standalone | `AgentRunner` | `tokio::JoinSet` for parallel tools |
| Durable | Restate SDK 0.8 | `AgentWorkflow`, `OrchestratorWorkflow` |
| Daemon | Kafka + Axum | `DaemonCore`, `DaemonHandle` |

New features must consider all three paths or explicitly scope to one.

### Error Handling
- **Library code** (`crates/heartbit/`): use `thiserror`, `?` operator, `Result<_, Error>`
- **CLI code** (`crates/heartbit-cli/`): use `anyhow`
- Never use `.unwrap()` in library code. `expect()` is allowed only for provably infallible operations.

### Memory and Ownership
- Prefer `&str` / `impl Into<String>` over owned `String` in function parameters
- Prefer borrowing over cloning
- `Arc<dyn Trait>` for shared ownership of trait objects

## Code Style

- Iterators over explicit loops
- `Vec::with_capacity` when size is known
- `pub(crate)` for internal APIs
- Builder pattern for complex configuration structs
- No premature abstraction: three similar lines beat one unused helper

## Where Things Live

| Task | Files |
|------|-------|
| Agent ReAct loop | `crates/heartbit/src/agent/mod.rs` |
| Tool trait | `crates/heartbit/src/tool/mod.rs` |
| Built-in tools (14) | `crates/heartbit/src/tool/builtins/` |
| MCP client | `crates/heartbit/src/tool/mcp.rs` |
| A2A client | `crates/heartbit/src/tool/a2a.rs` |
| LLM providers | `crates/heartbit/src/llm/` |
| Memory system | `crates/heartbit/src/memory/` |
| Guardrails | `crates/heartbit/src/agent/guardrails/` |
| Daemon core | `crates/heartbit/src/daemon/core.rs` |
| Daemon types | `crates/heartbit/src/daemon/types.rs` |
| JWT validation | `crates/heartbit/src/auth/jwt.rs` |
| Config | `crates/heartbit/src/config.rs` |
| CLI entry | `crates/heartbit-cli/src/main.rs` |
| Daemon HTTP | `crates/heartbit-cli/src/daemon.rs` |

## Protocol Implementation Status

### MCP Client (2025-11-25)
- **Implemented**: `tools/list`, `tools/call` over Streamable HTTP and stdio transports
- **Not implemented**: Resources protocol, Prompts protocol, Sampling (server-initiated LLM calls)
- Client declares `capabilities: {}` (empty — no client-side capabilities offered)

### A2A (0.2.x)
- **Implemented**: Agent card (`/.well-known/agent.json`), 8-state task lifecycle
- **Task states**: `pending`, `working`, `completed`, `failed`, `canceled`, `input_required`, `auth_required`, `rejected`
- **`input_required`** wired: WS path transitions to this state on `AgentEvent::ApprovalRequested`; returns to `working` on `ApprovalDecision`
- **`auth_required` / `rejected`**: defined, not yet automatically set (future wiring)
- **Not implemented**: Push notifications (webhook delivery)

### Auth Standards
- **RFC 8693** (Token Exchange): implemented in `TokenExchangeAuthProvider` for per-user MCP delegation
- **RFC 9449** (DPoP): not implemented — Bearer tokens only

## Testing Conventions

- Tests live in the same file as implementation (`#[cfg(test)] mod tests`)
- Every public function needs at least one test
- Use `#[tokio::test]` for async tests
- Mock providers: `MockLlmProvider` in `crates/heartbit/src/llm/mock.rs`
- No test should rely on network access or external services

## Multi-Tenant Daemon

When working on daemon code, respect the isolation boundaries:

- **Memory**: always namespace with `NamespacedMemory::new(store, "tenant:{tid}:user:{uid}")`
- **Workspace**: always scope to `{base}/{tenant_id}/{user_id}/`; use `validate_path_component()` on both
- **Tasks**: filter by `tenant_id` when a `UserContext` is present
- **MCP auth**: use `TokenExchangeAuthProvider` for per-user token exchange (RFC 8693)
- **Audit**: include `user_id`, `tenant_id`, `delegation_chain` in all audit records

**Never share a user's memory, workspace, or delegated tokens with another user.**

## What Not To Build (Yet)

Unless explicitly requested: no NATS, event sourcing, DAG scheduler, gRPC, Redis, Prometheus.

