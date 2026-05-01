# Hello agent

The smallest useful agent in `heartbit-core` is about thirty lines:
you build a provider, give it to `AgentRunner::builder`, set a system
prompt, and call `.execute(...).await`. Here is the whole file:

```rust,no_run
{{#include ../../../crates/heartbit-core/examples/hello_agent.rs}}
```

## Running it

```bash
ANTHROPIC_API_KEY=sk-... cargo run -p heartbit-core --example hello_agent
```

You should see a paragraph about Rust printed to stdout.

## What's happening

`AnthropicProvider::new(...)` constructs a typed client for Anthropic's
Messages API. `RetryingProvider::with_defaults(...)` wraps it with
exponential backoff on 429 / 5xx / network errors so a transient blip
doesn't kill your run. `BoxedProvider` then erases the concrete type
behind a `dyn`-compatible facade so you can swap providers in
[Configuration](../configuration/README.md) without changing call
sites. The whole stack is wrapped in `Arc` because the agent shares it
across parallel tool invocations.

`AgentRunner::builder(provider).system_prompt(...).build()?` produces
an `AgentRunner` ready to run. The builder is where you'll later add
tools, memory, guardrails, turn caps, and event listeners — the
[Agents chapter](../agents/README.md) walks through every option.

`agent.execute("What is Rust?").await?` runs the
[ReAct loop](../agents/README.md#the-react-loop) until the model
returns a final text response. The returned
[`AgentOutput`](https://docs.rs/heartbit-core/latest/heartbit_core/struct.AgentOutput.html)
contains the text in `.result`, plus `tokens_used`, `tool_calls_made`,
and an estimated `cost_usd`. This minimal agent has no tools, so it
runs exactly one turn.

From here, the next page picks an LLM provider that fits your needs;
after that you'll set up an API key.
