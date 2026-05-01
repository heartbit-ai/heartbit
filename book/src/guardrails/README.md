# Guardrails

Guardrails sit between the agent loop and the outside world, watching
LLM input and output and every tool call as it happens. They are how
you stop secrets from leaking into responses, redact PII before it
hits a log, refuse prompt-injection attempts, cap the number of
expensive tool calls, enforce policy on which tools may run with which
arguments, and let an LLM judge a sibling LLM's output before it
reaches the user. They run on the standalone path — the same code
that powers the durable and daemon execution paths — and a denial
turns into structured feedback the model sees on the next turn rather
than a panic.

## The Guardrail trait

Every guardrail implements
[`heartbit_core::agent::guardrails`](https://docs.rs/heartbit-core/latest/heartbit_core/agent/guardrails/index.html)
through the four hooks of the `Guardrail` trait: `pre_llm` to inspect
or mutate the request before it leaves, `post_llm` to inspect the
response that came back, `pre_tool` to gate a specific tool call, and
`post_tool` to inspect or rewrite the tool's output. Each method has
a default no-op, so you only override what you need.

Hooks return a `GuardAction`: `Allow` lets the operation through,
`Warn { reason }` allows it but emits an event for monitoring,
`Deny { reason }` blocks it, and `Kill { reason }` terminates the run
immediately. The runner walks the guardrail list in order on each
hook and the first `Deny` wins — `Warn`s never short-circuit. All
hooks are async; they return `Pin<Box<dyn Future>>` for
dyn-compatibility, so the runner can hold them as
`Vec<Arc<dyn Guardrail>>`.

## Built-in guardrails

The framework ships nine production-ready guardrails covering the
common safety surfaces.

### LLM judge
`LlmJudgeGuardrail` runs a second, typically cheaper LLM against the
main agent's response (or, optionally, its tool inputs) and asks it
to vote `SAFE` / `UNSAFE` / `WARN` against a list of criteria. Fail-
open on timeout or judge error keeps the agent loop running through
transient judge issues.

### Secret scanner
`SecretScannerGuardrail` matches AWS keys, generic API keys, bearer
tokens, JWTs, private keys, and database connection strings against
LLM responses and tool outputs. The default action is `Redact`; the
alternative is `Deny`.

### PII guardrail
`PiiGuardrail::all_builtin(PiiAction::Redact)` covers email addresses,
phone numbers, SSNs, and credit card numbers (Luhn-validated) out of
the box. Add custom regex detectors for tenant-specific identifiers.

### Content fence
`ContentFenceGuardrail` enforces fenced-code formatting on assistant
output, useful when the agent feeds another tool that expects a code
block in a known shape.

### Action budget
`ActionBudgetGuardrail` caps how many times a tool — or all tools
matching a pattern — can be called per run. Set per-pattern budgets
via `.rule("web_fetch", 5)` and a default budget for everything
else.

### Behavioral monitor
`BehavioralMonitorGuardrail` flags suspicious tool-use sequences:
repeated identical calls, write-after-no-read, or any pattern you
encode as a `BehaviorRule`. Bounded windows with a TTL keep memory
flat.

### Tool policy
`ToolPolicyGuardrail` evaluates a list of `ToolRule`s — each is a
tool name, an `InputConstraint` (regex, JSON path equals, allowlist),
and an action. Use it to forbid `bash` from running `rm -rf`, or to
restrict `web_fetch` to your own domains.

### Injection classifier
`InjectionClassifierGuardrail` scores incoming text against known
prompt-injection patterns ("ignore previous instructions", "you are
now…") and acts when the score crosses a threshold.

### Sensor security
The `SensorSecurityGuardrail`, gated behind the `sensor` feature on
the umbrella [heartbit](https://crates.io/crates/heartbit) crate,
validates sensor-pipeline payloads (Telegram, webhook, etc.) before
they reach the agent — replay protection, signature checks, and
per-tenant trust levels.

## Composing multiple guardrails

You attach guardrails by handing the agent a `Vec<Arc<dyn Guardrail>>`.
The order matters because `Deny` short-circuits, so put the cheapest
checks first:

```rust,ignore
let guardrails: Vec<Arc<dyn Guardrail>> = vec![
    Arc::new(SecretScannerGuardrail::builder().build()),
    Arc::new(PiiGuardrail::all_builtin(PiiAction::Redact)),
    Arc::new(LlmJudgeGuardrail::builder(judge).criterion("...").build()?),
];
let agent = AgentRunner::builder(provider).guardrails(guardrails).build()?;
```

## Example: full guardrail stack

The example below wires three guardrails — secret scanner, PII
redactor, LLM judge — onto a single agent and logs any denial through
an `on_event` listener:

```rust,no_run
{{#include ../../../crates/heartbit-core/examples/guardrails.rs}}
```

In production, point the judge guardrail at a smaller, cheaper model
than the main agent's; the cost difference compounds quickly when you
evaluate every response.

## When to write your own

Implement `Guardrail` directly when none of the built-ins capture your
domain rule — say, "deny any response that mentions a competitor's
product" or "redact internal project codenames". The pattern mirrors
the [custom-tool walkthrough in the Tools chapter](../tools/README.md#writing-your-own-tool):
pick the hook(s) you care about, return `Pin<Box<dyn Future<…> + Send>>`
from each, and decide between `Allow`, `Warn`, `Deny`, and `Kill`
based on the input or response. Override `name()` so audit records
attribute denials back to your guardrail. Mutating `pre_llm` or
`post_tool` is the same shape — the request and tool output are
passed mutably so you can redact in place rather than denying outright.
