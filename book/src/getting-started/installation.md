# Installation

## Add heartbit-core

```bash
cargo add heartbit-core
```

This is the framework only — agents, tools, providers, memory,
guardrails, workflow agents, eval. For Postgres-backed memory, the
Telegram / Discord / Slack chat adapters, fastembed local embeddings,
the secrets vault, multi-tenant daemon mode, and Restate-durable
execution, depend on the umbrella crate instead:

```bash
cargo add heartbit --features postgres,telegram,vault
```

The umbrella re-exports everything in `heartbit_core::*`, so library
code remains import-compatible regardless of which crate you depend on.

## Rust version

`heartbit-core` is on edition 2024 and requires **Rust 1.85 or later**.
A current stable toolchain installed via [rustup](https://rustup.rs)
will work.

## See also

- [crates.io/crates/heartbit-core](https://crates.io/crates/heartbit-core)
- [docs.rs/heartbit-core](https://docs.rs/heartbit-core)
