# MCP server integration

## Goal

Connect an MCP (Model Context Protocol) server and expose its tools
alongside the built-ins.

## Solution

`McpClient::connect_stdio` spawns the server as a subprocess and
speaks MCP over its stdin/stdout. Once connected, `client.into_tools()`
returns a `Vec<Arc<dyn Tool>>` you can extend onto the built-in set
and pass straight to `AgentRunnerBuilder::tools`. Run with the MCP
command after `--`, for example:
`cargo run -p heartbit-core --example mcp_agent -- npx -y @anthropic-ai/mcp-server-filesystem /tmp`.

```rust,no_run
{{#include ../../../crates/heartbit-core/examples/mcp_agent.rs}}
```

The example fails soft: if no command is supplied — or the connection
errors — the agent runs with the built-in tools only, so the binary
always compiles and runs end to end.

## Built-in MCP presets

Ten ready-to-use server presets ship with `heartbit-core` (GitHub,
Slack, Postgres, …). Reference them by name in your agent config and
heartbit will resolve the URL, env vars, and auth pattern for you. See
[Configuration](../configuration/README.md#mcp-server-presets) for the
full list and the per-preset env-var requirements.

## Notes

- MCP server URLs are validated through the same SSRF defense as
  `web_fetch` (private IPs refused by default; opt out via
  `HEARTBIT_ALLOW_PRIVATE_IPS=1`).
- See [`McpClient`](https://docs.rs/heartbit-core/latest/heartbit_core/struct.McpClient.html)
  for the full API, including `connect_with_auth` for OAuth token
  exchange and `connect` for already-running HTTP servers.
