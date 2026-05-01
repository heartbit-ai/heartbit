# Configuration

`heartbit-core` can be driven entirely from a TOML file via
[`HeartbitConfig`](https://docs.rs/heartbit-core/latest/heartbit_core/config/struct.HeartbitConfig.html).
This is the path you want when the agent topology should be operable
without recompiling — for example, when running `heartbit-cli`'s
`daemon` subcommand, or when a single binary needs to host several
distinct agent setups loaded at startup.

## HeartbitConfig from TOML

`HeartbitConfig::from_toml` parses and validates in one step. It
rejects zero values for `max_turns`/`max_tokens`, enforces unique
agent names, and wires the provider, orchestrator defaults, agents,
and (optionally) Restate, telemetry, daemon, and memory sections into
a typed root struct.

```rust,no_run
{{#include ../../../crates/heartbit-core/examples/from_toml.rs}}
```

The TOML root has four primary tables: `[provider]` (LLM defaults),
`[orchestrator]` (top-level run defaults), one or more `[[agents]]`
entries (each becomes a sub-agent), and optional `[telemetry]`,
`[memory]`, `[daemon]`, and `[restate]` sections wired in by the
respective code paths.

## Provider configuration

The `[provider]` table tells the runtime which LLM to call by
default. Required fields are `name` (e.g. `"anthropic"`,
`"openrouter"`, `"openai_compat"`, `"gemini"`) and `model`. Optional
fields include `base_url` (override the default endpoint — useful for
Azure, self-hosted, or proxy setups), `api_key` (prefer environment
variables in production), and `prompt_caching = true` to enable
Anthropic prompt caching on the system prompt and tool definitions.
Two sub-tables — `[provider.cascade]` and `[provider.retry]` — control
cost and reliability.

## Cascade and retry

`[provider.cascade]` configures
[model cascading](https://docs.rs/heartbit-core/latest/heartbit_core/struct.CascadingProvider.html):
the runtime tries cheap-tier models first and only escalates to the
main `[provider].model` when the confidence gate rejects the cheap
response. Tiers are listed cheapest-first under `[[provider.cascade.tiers]]`.
`[provider.retry]` configures `RetryProviderConfig` — exponential
backoff for transient failures (HTTP 429, 500, 502, 503, 529, network
errors). Defaults are 3 retries, 500 ms base delay, 30 s cap.

## Per-agent overrides

Each `[[agents]]` entry can override the global provider, the
guardrails block, memory namespace, turn/token caps, and the tool
profile. A typical per-agent provider override looks like:

```toml
[[agents]]
name = "fast-router"
description = "Lightweight intent classifier."
system_prompt = "Classify the user request into one of: search, code, chat."

[agents.provider]
name = "openrouter"
model = "google/gemini-2.5-flash"
prompt_caching = false
```

The agent inherits everything else from the top-level `[provider]`
and `[orchestrator]` blocks. Per-agent `guardrails`, `cascade`, and
`response_schema` fields work the same way.

## Templates

Heartbit ships 15 built-in agent templates in
[crates/heartbit-core/templates/](https://github.com/heartbit-ai/heartbit/tree/main/crates/heartbit-core/templates).
Reference one by name on an agent — `template = "researcher"` — and
the template's `system_prompt`, `max_tokens`, `max_turns`, and other
defaults are merged in; user-specified fields on the same agent
override the template. The bundled templates are: `analyst`,
`architect`, `coder`, `customer-support`, `data-scientist`,
`debugger`, `ops`, `orchestrator`, `planner`, `researcher`,
`reviewer`, `security-auditor`, `test-engineer`, `translator`,
`writer`.

## Skills

Skills are reusable system-prompt fragments — domain expertise
bundles that get injected into an agent's prompt at config
resolution. Heartbit ships 10 built-in skill packs in
[crates/heartbit-core/skills/](https://github.com/heartbit-ai/heartbit/tree/main/crates/heartbit-core/skills):
`api-design`, `docker`, `git-expert`, `kubernetes`, `python-expert`,
`rust-expert`, `security`, `sql-expert`, `testing`,
`typescript-expert`. Reference them via `skills = ["rust-expert", "testing"]`
on an `[[agents]]` entry; the runtime auto-injects each skill's
`SKILL.md` at build time. Custom skills can live anywhere on the
filesystem — the bundled names are just resolved first.

## MCP server presets

Connecting to a popular MCP server usually means typing the same URL,
auth headers, and tool allowlist into every config. Heartbit ships 10
named presets in
[crates/heartbit-core/mcp-presets/](https://github.com/heartbit-ai/heartbit/tree/main/crates/heartbit-core/mcp-presets):
`brave-search`, `github`, `gitlab`, `google-calendar`, `jira`,
`linear`, `notion`, `postgresql`, `sentry`, `slack`. Reference one in
an agent's `mcp_servers` list — the preset name resolves to the full
config, and you only need to supply the secrets it requires (typically
through environment variable substitution). When you outgrow a
preset, copy its JSON file into your project and edit freely; preset
resolution falls back to literal config when the name doesn't match.
