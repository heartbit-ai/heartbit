# Built-in Tools

14 tools are available by default when running without a config file (env-based mode).

## Tool Reference

| Tool | Description |
|------|-------------|
| `bash` | Execute bash commands. Working directory persists between calls. Default timeout: 120s, max: 600s. |
| `read` | Read a file with line numbers. Detects binary files. Max size: 256 KB. |
| `write` | Write content to a file. Creates parent directories. Read-before-write guard. |
| `edit` | Replace an exact string in a file (must appear exactly once). Read-before-write guard. |
| `patch` | Apply unified diff patches to one or more files. Single-pass hunk application. |
| `glob` | Find files matching a glob pattern. Skips hidden files. |
| `grep` | Search file contents with regex. Uses `rg` when available, falls back to built-in. |
| `list` | List directory contents as an indented tree. Skips common build artifacts. |
| `webfetch` | Fetch content from a URL via HTTP GET. Supports text, markdown, HTML. Max: 5 MB. |
| `websearch` | Search the web via Exa AI. Requires `EXA_API_KEY`. |
| `todowrite` | Write/replace the full todo list. Only 1 item in progress at a time. |
| `todoread` | Read the current todo list. |
| `skill` | Load skill definitions from `SKILL.md` files. |
| `question` | Ask the user structured questions (only when `on_question` callback is set). |

## Cross-Agent Coordination

**Blackboard** — shared `Key -> Value` store for squad agents. Sub-agents get `blackboard_read`, `blackboard_write`, `blackboard_list` tools. After each sub-agent completes, its result is written to `"agent:{name}"`.

## Structured Output

Set `response_schema` (JSON Schema) on an agent. A synthetic `__respond__` tool is injected and `tool_choice` forced to `Any`. The agent calls `__respond__` to produce structured JSON in `AgentOutput::structured`.

## Human-in-the-Loop

`--approve` flag enables interactive approval before each tool execution round. Denied tools receive error results — the LLM can adjust and retry. In Restate path, approval uses per-turn promise keys.

## Streaming

`on_text` callback receives text deltas as they arrive from the LLM. Both Anthropic and OpenRouter providers implement SSE streaming. Sub-agents don't stream — only the orchestrator.

## Agent Events

13 structured `AgentEvent` variants emitted via `OnEvent` callback:

`RunStarted`, `TurnStarted`, `LlmResponse`, `ToolCallStarted`, `ToolCallCompleted`, `ApprovalRequested`, `ApprovalDecision`, `SubAgentsDispatched`, `SubAgentCompleted`, `ContextSummarized`, `RunCompleted`, `GuardrailDenied`, `RunFailed`

Use `--verbose` to emit events as JSON to stderr.

## Cost Tracking

`estimate_cost(model, usage) -> Option<f64>` returns estimated USD cost for known models (Claude 4, 3.5, and 3 generations, including OpenRouter aliases). Accounts for cache read/write token rates. Displayed in CLI output after each run.

## Custom Tools

Implement the `Tool` trait and register with an agent:

```rust
use heartbit::{Tool, ToolDefinition, ToolOutput, Error};
use serde_json::Value;
use std::pin::Pin;
use std::future::Future;

pub struct MyTool;

impl Tool for MyTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new("my_tool", "Does something useful")
            .with_parameter("input", "string", "The input value", true)
    }

    fn execute(
        &self,
        input: Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>> {
        Box::pin(async move {
            let input_str = input["input"].as_str().unwrap_or_default();
            Ok(ToolOutput::text(format!("Processed: {input_str}")))
        })
    }
}

// Register with an agent:
// AgentRunnerBuilder::new(provider).tools(vec![Arc::new(MyTool)]).build()
```
