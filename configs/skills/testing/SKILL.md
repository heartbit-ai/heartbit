---
name = "testing"
description = "TDD workflow, property-based testing, mocking, integration patterns, and coverage strategies"
tags = ["testing", "tdd", "mocking", "integration", "quality"]
max_inject_tokens = 2000
---

# Testing Expert

## TDD Workflow

Red-Green-Refactor cycle, strictly ordered:

1. **Red**: Write a failing test that describes the desired behavior. Run it. Confirm it fails for the right reason.
2. **Green**: Write the minimum code to make the test pass. No more.
3. **Refactor**: Clean up both test and implementation. Tests must still pass.

Start with the simplest case, then add edge cases. Each test should test one behavior, named to describe that behavior:

```rust
#[test]
fn parse_config_rejects_zero_max_turns() {
    let result = parse_config("max_turns = 0");
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("max_turns"));
}
```

Write the test before the implementation exists. Compile errors are acceptable in the Red phase — the point is to define the interface before building it.

## Property-Based Testing

Test invariants over randomized inputs instead of hand-picked examples. Catches edge cases you'd never think of.

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn roundtrip_serialization(input in ".*") {
        let serialized = serde_json::to_string(&input).unwrap();
        let deserialized: String = serde_json::from_str(&serialized).unwrap();
        assert_eq!(input, deserialized);
    }

    #[test]
    fn parse_never_panics(input in "\\PC*") {
        let _ = parse_config(&input);  // Must not panic
    }
}
```

Good properties: roundtrip (serialize/deserialize), idempotency (applying twice = applying once), invariant preservation (sorted output stays sorted), oracle comparison (fast impl matches reference impl), "no crash" (fuzzing for panics).

## Mocking Strategy

Mock at boundaries (network, filesystem, clock), not internal functions. Use trait-based injection:

```rust
#[cfg_attr(test, mockall::automock)]
trait TimeProvider: Send + Sync {
    fn now(&self) -> DateTime<Utc>;
}

struct RealTime;
impl TimeProvider for RealTime {
    fn now(&self) -> DateTime<Utc> { Utc::now() }
}

// In tests:
let mut mock = MockTimeProvider::new();
mock.expect_now().returning(|| Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap());
```

Prefer fakes (in-memory implementations) over mocks for complex interfaces. Fakes test real behavior; mocks test call patterns. Use mocks when you need to verify interactions (was this called? with what arguments?).

## Integration Patterns

Separate unit tests from integration tests. Unit tests: fast, isolated, in the same file. Integration tests: in `tests/` directory, may need external services.

```rust
// tests/integration_test.rs
#[tokio::test]
async fn full_agent_run_produces_output() {
    let provider = TestProvider::new(vec![mock_response("Hello!")]);
    let runner = AgentRunnerBuilder::new("test", provider).build().unwrap();
    let output = runner.run("Say hello").await.unwrap();
    assert!(!output.text.is_empty());
}
```

Use `testcontainers` for databases/services in integration tests. Each test gets a fresh container — no shared state.

Test fixtures: use builder patterns or factory functions, not shared mutable state. `#[fixture]` (rstest) for parameterized setup.

## Coverage Strategy

Coverage measures executed lines, not tested behavior. 100% line coverage with no assertions is worthless.

Target high coverage on: business logic, error handling paths, boundary conditions, serialization/deserialization. Accept lower coverage on: generated code, trivial getters, platform-specific code, third-party integration wrappers.

Use `cargo llvm-cov` (Rust) or `coverage.py` (Python). Branch coverage over line coverage — catches missed `else` paths.

## Anti-Patterns

- Testing implementation details: tests break on refactors that don't change behavior.
- Shared mutable test state: tests pass alone, fail together. Each test sets up its own state.
- Sleeping in tests: use condition-based waits or test clocks. `tokio::time::pause()` for async.
- Giant test functions: split into arrange/act/assert. One assertion per logical behavior.
- Commenting out failing tests: fix them or delete them. Dead tests are technical debt.
- Over-mocking: if you mock everything, you're testing your mocks, not your code.
