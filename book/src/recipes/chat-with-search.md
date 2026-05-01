# Chat agent with web search

## Goal

A conversational agent that browses the web to answer questions and
returns sources.

## Solution

Build an `AgentRunner` with the default built-in tools. The default set
already includes `web_search` and `web_fetch` alongside the file tools,
so a single call to `builtin_tools(BuiltinToolsConfig::default())` is
enough. Wrap your `LlmProvider` in `RetryingProvider` to absorb 429s,
and attach an `on_event` callback if you want a streaming-style trace
to stderr while the agent works.

For citation-friendly answers, instruct the system prompt to quote the
URLs returned by `web_fetch`. The agent will plan its own
search-then-fetch loop; you do not need to script it. The example below
is a working starting point — change the `system_prompt` to taste, and
extend the user message into a chat loop by feeding follow-up turns
through `agent.execute(...)`.

```rust,no_run
{{#include ../../../crates/heartbit-core/examples/simple_agent.rs}}
```

The example ships with the default tool set, which already exposes
`web_search` and `web_fetch`. If you build a custom `BuiltinToolsConfig`
later, double-check those two flags are on.

## Notes

- The `web_fetch` tool refuses private IPs (loopback, RFC1918,
  link-local) unless `HEARTBIT_ALLOW_PRIVATE_IPS=1` is set. See
  [API keys and environment](../getting-started/env.md) for the opt-out
  and the [Tools chapter](../tools/README.md) for the full SSRF policy.
- For results citation, instruct the system prompt to include source
  URLs in the response.
