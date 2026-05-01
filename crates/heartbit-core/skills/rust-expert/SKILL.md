---
name = "rust-expert"
description = "Rust ownership, async patterns, traits, error handling, and performance idioms"
tags = ["rust", "systems", "async", "traits", "performance"]
max_inject_tokens = 2000
---

# Rust Expert

## Ownership and Borrowing

Prefer borrowing over cloning. Accept `&str` or `impl Into<String>` in function parameters instead of `String`. Use `Cow<'_, str>` when you sometimes need to allocate and sometimes don't.

```rust
// Bad: forces caller to allocate
fn process(name: String) { ... }

// Good: borrows when possible
fn process(name: &str) { ... }
fn process(name: impl Into<String>) { ... }
```

Avoid returning references to temporaries. If a function builds a value, return the owned type. Use `'_` elided lifetimes to reduce noise.

## Async Patterns

Never hold a `std::sync::MutexGuard` across `.await` — use `tokio::sync::Mutex` if you must, or restructure to drop the guard before awaiting. For locks never held across await points, prefer `std::sync::RwLock` (cheaper, no async overhead).

Use `tokio::JoinSet` for concurrent task spawning with bounded concurrency. Avoid `tokio::spawn` loops without backpressure.

```rust
let mut set = JoinSet::new();
for item in items {
    set.spawn(process(item));
}
while let Some(result) = set.join_next().await {
    result??;
}
```

Pin `Future` objects when storing them in structs: `Pin<Box<dyn Future<Output = T> + Send + 'a>>`.

## Error Handling

Use `thiserror` for library errors with structured variants. Use `anyhow` for application/CLI code. Never `.unwrap()` in library code — use `?` operator. `expect()` is acceptable only for provably infallible operations.

```rust
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("config parse failed: {0}")]
    ConfigParse(#[from] toml::de::Error),
    #[error("request failed after {retries} retries")]
    Exhausted { retries: u32 },
}
```

Wrap partial results with error context: `Error::WithPartialUsage { source, usage }`.

## Trait Patterns

Use `pub(crate)` for internal-only trait methods. For object safety with async methods, return `Pin<Box<dyn Future<...> + Send + '_>>` instead of using RPITIT (which isn't object-safe).

Builder pattern for complex configs: `FooBuilder::new().field(val).build() -> Result<Foo>`. Validate in `build()`, not in setters.

## Performance

- `Vec::with_capacity(n)` when size is known.
- Iterators over explicit loops: `.filter().map().collect()`.
- `Arc<dyn Trait>` for shared ownership of trait objects across tasks.
- `SmallVec` or `ArrayVec` for small, fixed-capacity collections on hot paths.
- Avoid `format!()` in hot loops — use `write!()` into a reusable buffer.
- `#[inline]` only after profiling shows it matters.

## Common Pitfalls

- `String::from` vs `.to_string()` vs `.to_owned()` — functionally identical, prefer `.to_owned()` for `&str`.
- `impl Trait` in return position prevents naming the type — use `Box<dyn Trait>` if you need to store it.
- `Default::default()` on `Option<T>` gives `None`, not `Some(T::default())`.
- Turbofish needed when compiler can't infer: `iter.collect::<Vec<_>>()`.
- `Drop` order is reverse declaration order — matters for lock guards and file handles.
- `#[serde(default)]` on struct fields for backward-compatible deserialization.
