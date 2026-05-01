# Tools

Tools are how an agent affects the outside world: reading files,
running shell commands, fetching URLs, querying APIs, calling other
agents. The framework ships a curated built-in suite, gives you a
clean trait for your own, and integrates with any
[Model Context Protocol](https://modelcontextprotocol.io) server out
of the box.

## The Tool trait

Every tool is a value that implements two methods:

- `definition() -> ToolDefinition` — the name, human-readable
  description, and a JSON Schema for the input.
- `execute(input: serde_json::Value) -> Pin<Box<dyn Future<Output =
  Result<ToolOutput, Error>> + Send + '_>>` — the async body.

`ToolOutput::success(text)` and `ToolOutput::error(text)` build the
two terminal cases. The error variant is **not** a panic or a runtime
abort: the runner feeds the error string back into the conversation as
a tool result, and the model decides whether to retry, give up, or
ask the user. Tools that *do* panic are caught by the runner and
converted to `ToolOutput::error` for the same reason — a misbehaving
tool must never take down the agent loop.

## Built-in tools

`heartbit-core` includes a focused built-in set covering the cases an
agent needs most:

- `read` — read a file, with mtime tracking for safe edits.
- `write` — write a file, blocked unless a recent `read` exists.
- `edit` — find-and-replace within a file.
- `patch` — apply a unified-diff hunk.
- `bash` — run a shell command, sandboxable via Linux landlock
  (see [Production Considerations](../production/README.md)).
- `list`, `glob`, `grep` — directory traversal and search.
- `todo` — track multi-step task progress across turns.
- `web_fetch` — HTTP GET with default-deny on private IPs.
- `web_search` — query a search backend.
- `image_generate` — call image-generation APIs.
- `tts` — text-to-speech.
- `twitter_post` — post to X/Twitter (per-tenant credentials).
- `skill` — load a named skill from a skill registry.
- `question` — ask the user a structured question
  (interactive sessions).

Build the suite with `builtin_tools(BuiltinToolsConfig::default())`.
Dangerous tools like `bash` are gated behind `dangerous_tools = true`.
For the full list and signatures see
[`heartbit_core::tool::builtins`](https://docs.rs/heartbit-core/latest/heartbit_core/tool/builtins/index.html).

## Writing your own tool

When the built-ins don't cover your domain — pull a record from your
database, hit an internal microservice, render a chart — implement the
trait yourself. The pattern is short:

```rust,no_run
{{#include ../../../crates/heartbit-core/examples/custom_tool.rs}}
```

The `definition()` returns a `ToolDefinition` whose `input_schema` is
a JSON Schema object. The agent's LLM uses the description and schema
to decide when and how to call the tool, so write both for the model:
say what the tool *does*, list each field, and give an example value
where it helps. The `execute` body parses the validated input, does
the work, and returns either `ToolOutput::success` or
`ToolOutput::error`.

## The heartbit_tool macro

The `heartbit_tool` proc-macro in the
[`heartbit-macro`](https://docs.rs/heartbit-macro) crate generates the
trait impl and JSON Schema from the function signature, so simple
tools collapse to a single annotated `async fn`:

```rust,ignore
use heartbit_macro::heartbit_tool;

#[heartbit_tool(description = "Count whitespace-separated words.")]
async fn word_count(text: String) -> Result<ToolOutput, Error> {
    Ok(ToolOutput::success(text.split_whitespace().count().to_string()))
}
```

Reach for the macro when the schema is mechanical. Drop down to the
trait impl when you need custom validation, dynamic schemas, or
request-scoped state.

## Tool input validation

Before dispatching a tool call, the runner validates the LLM-supplied
input against the declared `input_schema` using
`validate_tool_input`. A schema mismatch becomes a tool-result error
the model sees on the next turn — so it self-corrects rather than
crashing the run. This is one of the main payoffs of writing
JSON Schemas: the LLM gets early, structured feedback when it
hallucinates a field name.

## Tool approval (human-in-the-loop)

For high-stakes tools — anything that writes outside a sandbox, calls
a paid API, or posts to a channel — set an `on_approval` callback.
The runner pauses before each tool call, hands you the list of pending
calls, and waits for an `ApprovalDecision`:

- `Allow` — run this call once.
- `Deny` — refuse and tell the model why.
- `AlwaysAllow` — auto-approve this exact call shape for the rest of
  the run.
- `AlwaysDeny` — auto-refuse this shape.

The CLI's `--approve` flag wires this to an interactive prompt. In a
web app you would route the request to a UI and resume on response.

## MCP integration

For tools you don't want to write — Slack, Atlassian, GitHub, or any
of the growing
[MCP server ecosystem](https://modelcontextprotocol.io) — connect an
[`McpClient`](https://docs.rs/heartbit-core/latest/heartbit_core/struct.McpClient.html)
and the server's tools become first-class `Tool` instances you pass
to `AgentRunnerBuilder::tools(...)` alongside the built-ins. The
client speaks MCP Streamable HTTP, handles auth, sampling, and
resource roots, and reuses connection pools across agents. See
[the MCP recipe](../recipes/mcp-integration.md) for an end-to-end
walk-through.
