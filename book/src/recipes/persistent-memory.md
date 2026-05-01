# Long-running agent with persistent memory

## Goal

An agent whose memory survives process restarts. Use the heartbit
umbrella crate's
[`PostgresMemoryStore`](https://docs.rs/heartbit/latest/heartbit/struct.PostgresMemoryStore.html)
(behind the `postgres` feature).

## Solution

Provision a Postgres instance, build a `sqlx::PgPool`, and hand the
pool to `PostgresMemoryStore::new`. The schema auto-migrates on first
run, so you don't have to ship migrations alongside your code. Wrap
the store in a `NamespacedMemory` keyed by the agent name (or a tenant
id, for multi-tenant deployments) so different scopes can't read each
other's entries.

Pass the namespaced memory into `AgentRunnerBuilder::with_memory`. The
runner will then expose the five memory tools (`memory_store`,
`memory_recall`, `memory_update`, `memory_forget`, `memory_consolidate`)
and persist every entry to Postgres automatically.

```bash
cargo add heartbit --features postgres
```

```rust,no_run
use std::sync::Arc;

use heartbit::{
    AgentRunner, AnthropicProvider, BoxedProvider, NamespacedMemory,
    PostgresMemoryStore, RetryingProvider,
};
use sqlx::postgres::PgPoolOptions;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("ANTHROPIC_API_KEY")?;
    let provider = Arc::new(BoxedProvider::new(RetryingProvider::with_defaults(
        AnthropicProvider::new(&api_key, "claude-sonnet-4-20250514"),
    )));

    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(&std::env::var("DATABASE_URL")?)
        .await?;

    let store = PostgresMemoryStore::new(pool, "heartbit").await?;
    let memory = Arc::new(NamespacedMemory::new(Arc::new(store), "support-agent"));

    let agent = AgentRunner::builder(provider)
        .system_prompt("You remember the user across sessions.")
        .with_memory(memory)
        .build()?;

    let output = agent.execute("Hi, I'm Alice. Save that.").await?;
    println!("{}", output.result);
    Ok(())
}
```

## Notes

- For multi-tenant deployments, namespace memory per tenant via
  [`NamespacedMemory`](https://docs.rs/heartbit-core/latest/heartbit_core/memory/struct.NamespacedMemory.html).
- See the [Memory chapter](../memory/README.md) for the trait surface
  and the five memory tools.
