# Choosing an LLM provider

`heartbit-core` ships with four LLM providers. Pick one based on what
you're optimizing for:

| Provider | Strengths | Caveats |
|---|---|---|
| Anthropic | Native prompt caching; mature tool use; strong reasoning | Higher per-token latency than some |
| OpenRouter | One API key for 100+ models; cheap A/B testing | Quality varies by underlying model |
| Gemini | Strong on long-context tasks; competitive pricing | Tool-call format quirks |
| OpenAI-compatible | Works with vLLM, Ollama, LM Studio, local LLMs | YMMV on tool-use reliability |

If you don't know yet, start with Anthropic — every example in this
book uses it.

## Swapping providers

Each provider implements the same `LlmProvider` trait, so swapping is
a one-line change. To run [Hello agent](./hello-agent.md) against
OpenRouter instead of Anthropic:

```rust,no_run
use heartbit_core::{BoxedProvider, OpenRouterProvider, RetryingProvider};

let api_key = std::env::var("OPENROUTER_API_KEY").unwrap();
let provider = std::sync::Arc::new(BoxedProvider::new(
    RetryingProvider::with_defaults(
        OpenRouterProvider::new(api_key, "anthropic/claude-sonnet-4"),
    ),
));
```

Replace `OpenRouterProvider` with `GeminiProvider` or
`OpenAiCompatProvider` for the other two.

## Going deeper

See [`heartbit_core::llm`](https://docs.rs/heartbit-core/latest/heartbit_core/llm/index.html)
for the full provider API, including `RetryConfig`, the cascading
provider, and Anthropic prompt caching.
