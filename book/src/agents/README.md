# Agents

An agent in `heartbit-core` is an `AgentRunner` — a configured LLM, a
system prompt, an optional tool set, optional memory, optional
guardrails, and a set of safety limits. You build one with
[`AgentRunnerBuilder`](https://docs.rs/heartbit-core/latest/heartbit_core/struct.AgentRunnerBuilder.html)
and run it by calling `.execute(task).await`. The rest of this chapter
explains what happens between those two calls and how to control it.

## The ReAct loop

Every call to `.execute(...)` runs a [ReAct loop](https://arxiv.org/abs/2210.03629).
Each iteration is one **turn**. On every turn the runner sends the
current message history to the LLM, parses the response, and dispatches
any tool calls the model requested. Tool results become new messages
in the history, and the loop runs again.

When the LLM returns a `tool_use` content block, the runner spawns one
task per tool call on a `tokio::JoinSet` and awaits all of them
together. Independent tools — read three files, fetch two URLs —
therefore execute in parallel, not serially. Tool errors do not crash
the loop: they are converted to `ToolOutput::error` and sent back to
the model, which can self-correct on the next turn.

The loop exits in one of four ways: the model returns a turn with no
tool calls (final answer), `max_turns` is reached, a token limit is
exceeded, or an error bubbles up from a provider or guardrail. The
returned `AgentOutput` carries the result text plus accumulated token
usage and cost.

## Building an agent

A typical agent declares the parts you'll see throughout the rest of
this book — provider, system prompt, tools, turn cap, an event hook:

```rust,no_run
{{#include ../../../crates/heartbit-core/examples/simple_agent.rs}}
```

Two things to notice. First, every method on the builder is optional
except the provider — you only configure what you need, defaults are
sensible. Second, `Arc` is everywhere because the runner internally
shares these values across parallel tool tasks; the type signatures
make this explicit rather than hiding it behind clones.

## System prompts

The system prompt sets the agent's role and ground rules. Keep it
focused: the model is good at staying in character but bad at
following long bullet lists of "always do this, never do that". Two
or three sentences plus any domain conventions usually beats a
five-paragraph essay.

If you need a starting point, the framework ships with 15 reusable
agent archetypes — researcher, coder, reviewer, planner, and so on —
covered in the [templates section](../configuration/README.md#templates)
of the Configuration chapter.

## Token budgets and turn limits

Three independent caps protect you from runaway agents:

- **`max_turns`** (default 10) — how many ReAct iterations the loop
  will perform before giving up.
- **`max_tokens`** — the per-call cap sent to the LLM provider
  (controls response length, not history size).
- **`max_total_tokens`** — a lifetime cap on the sum of input plus
  output tokens across the whole run.

Hitting any of these returns
`Error::WithPartialUsage`, which wraps the underlying error and
preserves the token counter accumulated up to that point — so your
billing telemetry stays accurate even on a mid-run abort. The
[Errors and partial usage](#errors-and-partial-usage) section below
covers the variants.

## Streaming output

Agents that talk to a user benefit from streaming tokens to the screen
as the LLM generates them. Set an `on_text` callback, and the runner
will invoke it for each delta:

```rust,no_run
use std::io::Write;
use std::sync::Arc;

let on_text = Arc::new(|delta: &str| {
    print!("{delta}");
    std::io::stdout().flush().ok();
});
let agent = AgentRunner::builder(provider)
    .on_text(on_text)
    .build()?;
```

See [`OnText`](https://docs.rs/heartbit-core/latest/heartbit_core/llm/type.OnText.html)
for the full signature.

## Events and observability

`on_text` shows you the model's prose; `on_event` shows you everything
else. Set an
[`AgentEvent`](https://docs.rs/heartbit-core/latest/heartbit_core/enum.AgentEvent.html)
listener and the runner emits 29 lifecycle variants — `RunStarted`,
`TurnStarted`, `LlmResponse`, `ToolCallStarted`, `ToolCallCompleted`,
`GuardrailDenied`, `RunCompleted`, and many more.

In dev, log them to stderr as JSON. In production, fan them out to
OpenTelemetry spans, Prometheus counters, or your audit trail; see
[Production Considerations](../production/README.md) for the
recommended OTel wiring.

## Errors and partial usage

`heartbit_core::Error` distinguishes the failure modes that matter:
provider errors (`Anthropic`, `OpenRouter`, `Gemini`, `OpenAiCompat`),
tool errors, guardrail denials, max-turns / max-tokens caps, context
overflow, and configuration errors. The
[error classifier](https://docs.rs/heartbit-core/latest/heartbit_core/llm/error_class/index.html)
sorts provider errors into `RateLimited`, `ServerError`,
`ContextOverflow`, and friends so retry policy can react.

The wrapper variant `Error::WithPartialUsage { source, usage }`
preserves the `TokenUsage` accumulated before the failure. Always
check for it at the outer match arm so your dashboards don't lose the
tokens a failed run already spent.
