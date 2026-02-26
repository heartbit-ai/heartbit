# Contributing to Heartbit

## Prerequisites

- **Rust stable** (with `cargo`, `rustfmt`, `clippy`)
- **cmake**
- **libssl-dev** (or equivalent)
- **pkg-config**

These are required for building `rdkafka` and other native dependencies.

### Ubuntu / Debian

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake libssl-dev pkg-config
```

### macOS

```bash
brew install cmake openssl pkg-config
```

### Fedora

```bash
sudo dnf install -y cmake openssl-devel pkg-config gcc
```

## Getting started

```bash
git clone https://github.com/heartbit-ai/heartbit.git
cd heartbit
cargo build
```

## Quality gate

All three commands must pass with zero warnings before any commit or PR:

```bash
cargo fmt -- --check && cargo clippy -- -D warnings && cargo test
```

## TDD mandate

Write tests first, then implementation. No exceptions.

1. **Red** -- write a failing test that defines the expected behavior.
2. **Green** -- write the minimal code to make the test pass.
3. **Refactor** -- clean up while keeping all tests green.

Every public function must have at least one test. `cargo test` must pass before any commit.

## Code style

- `thiserror` for library errors (heartbit crate), `anyhow` for application code (CLI).
- Prefer borrowing over cloning. Use `&str` / `impl Into<String>` for parameters.
- Use `?` operator. Never `.unwrap()` in library code.
- Iterators over loops. `Vec::with_capacity` for known sizes.
- `pub(crate)` for internal APIs. Keep modules focused.
- No premature abstraction -- three similar lines is better than one unused helper.

## Submitting a pull request

1. Fork the repository and create a feature branch from `main`.
2. Write tests first, then implement.
3. Run the quality gate: `cargo fmt -- --check && cargo clippy -- -D warnings && cargo test`
4. Commit with a clear, descriptive message.
5. Push your branch and open a pull request against `main`.
6. Describe what your PR does and why. Link related issues if any.

## Reporting issues

Open a GitHub issue with:
- A clear title and description.
- Steps to reproduce (if applicable).
- Expected vs. actual behavior.
- Rust version (`rustc --version`) and OS.
