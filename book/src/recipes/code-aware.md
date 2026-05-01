# Code-aware agent

## Goal

An agent that reads, navigates, and edits a real codebase using file
tools and a language-server-protocol client.

## Solution

The default `builtin_tools(BuiltinToolsConfig::default())` set already
includes the file tools an editor agent needs: `read`, `write`, `edit`,
`patch`, `glob`, and `grep`. They share a `FileTracker` that records
mtimes, so the agent cannot silently overwrite a file that changed under
it.

For richer code intelligence — go-to-definition, find-references, and
diagnostics — attach an `LspManager` from the heartbit umbrella crate's
[`lsp` module](https://docs.rs/heartbit/latest/heartbit/lsp/index.html).
The manager spawns a language server per project and exposes its
capabilities to the agent as additional tools. The example below
illustrates the same composition pattern by attaching a small custom
tool to an `AgentRunner`; swap in the file builtins (or LSP tools) the
same way.

```rust,no_run
{{#include ../../../crates/heartbit-core/examples/custom_tool.rs}}
```

The included example shows a custom price-lookup tool; the same
`AgentRunner::builder(...).tools(...)` pattern composes cleanly with
the file builtins. For LSP, see the umbrella's
[lsp module](https://docs.rs/heartbit/latest/heartbit/lsp/index.html).

## Notes

- The `bash` tool can be sandboxed via Linux landlock (umbrella's
  `sandbox` feature); see
  [Production Considerations](../production/README.md).
- The `read`/`write`/`edit` tools track file mtimes via `FileTracker`
  to prevent stale-write races.
