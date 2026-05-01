# API keys and environment

## Setting an API key

Each provider reads its API key from a conventional environment
variable. Pick the one that matches the provider you chose in
[Choosing an LLM provider](./providers.md):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENROUTER_API_KEY=sk-or-...
export GEMINI_API_KEY=AIza...
export OPENAI_API_KEY=sk-...
```

The `OPENAI_API_KEY` value is also what `OpenAiCompatProvider` reads
when you point it at a local vLLM or Ollama server.

## Local development

For day-to-day development, drop secrets into a `.env` file at the
project root and load them with
[`dotenvy`](https://crates.io/crates/dotenvy):

```rust,no_run
fn main() {
    dotenvy::dotenv().ok();
    // ... rest of main
}
```

Add `.env` to `.gitignore`. Never commit a key to a repository.

## Production

For production, store API keys in the
[vault module](../tools/README.md) (provided by the `heartbit`
umbrella crate's `vault` feature) rather than plain environment
variables. The vault gives you per-tenant key isolation, audit logs,
and rotation hooks that bare env vars can't.

## SSRF protection

The `web_fetch` built-in tool refuses requests to private and loopback
IP ranges by default. This blocks server-side request forgery (SSRF)
attacks where an LLM is tricked into fetching internal endpoints. To
allow internal-network access in development, set:

```bash
HEARTBIT_ALLOW_PRIVATE_IPS=1
```

See the [Tools chapter](../tools/README.md) for the full SSRF policy
and how to scope it down per-agent in production.
