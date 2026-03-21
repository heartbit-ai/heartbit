---
name = "python-expert"
description = "Python type hints, async, packaging, stdlib patterns, and performance"
tags = ["python", "async", "typing", "packaging", "performance"]
max_inject_tokens = 2000
---

# Python Expert

## Type Hints

Use `from __future__ import annotations` for deferred evaluation (PEP 563). Prefer `collections.abc` types over `typing` aliases (`Sequence` over `typing.Sequence`).

```python
from __future__ import annotations
from collections.abc import Sequence, Mapping

def process(items: Sequence[str], config: Mapping[str, int] | None = None) -> list[str]:
    ...
```

Use `TypeVar` with bounds for generic functions. `Protocol` for structural subtyping (duck typing with type safety). `TypeGuard` for narrowing in conditionals. `@overload` for functions with different return types based on input.

Avoid `Any` — use `object` for truly unknown types, `Never` for unreachable code paths.

## Async Patterns

`asyncio.TaskGroup` (3.11+) replaces `gather()` with structured concurrency and proper exception propagation.

```python
async with asyncio.TaskGroup() as tg:
    task1 = tg.create_task(fetch(url1))
    task2 = tg.create_task(fetch(url2))
# Both results available, exceptions propagate correctly
```

Use `asyncio.Semaphore` for bounded concurrency. Never mix `asyncio.run()` calls — one event loop per thread. Use `asyncio.to_thread()` for blocking I/O in async contexts.

Common pitfall: forgetting to `await` a coroutine silently returns the coroutine object. Enable `asyncio` debug mode during development.

## Packaging

Use `pyproject.toml` (PEP 621) exclusively. `setup.py` and `setup.cfg` are legacy.

```toml
[project]
name = "mypackage"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["httpx>=0.27"]

[project.optional-dependencies]
dev = ["pytest", "ruff", "mypy"]
```

Use `uv` for fast dependency resolution and virtual environments. Pin with `uv lock`. Use `src/` layout to prevent accidental imports from project root.

## Virtual Environments

Always isolate: `uv venv && source .venv/bin/activate`. Never install into system Python. Use `.python-version` file for `pyenv`/`uv` auto-detection. CI: `uv sync --frozen` for reproducible installs.

## Stdlib Patterns

- `functools.lru_cache` / `@cache` for memoization (use `maxsize=None` only for bounded input spaces).
- `contextlib.contextmanager` for resource cleanup. `ExitStack` for dynamic context management.
- `dataclasses.dataclass(frozen=True, slots=True)` for immutable value objects with low memory.
- `pathlib.Path` over `os.path` — chainable, type-safe, cross-platform.
- `itertools.batched()` (3.12+) for chunking. `more-itertools` for older Python.
- `enum.StrEnum` (3.11+) for string enums with automatic value.
- `tomllib` (3.11+) for reading TOML — no third-party dep needed.

## Performance Patterns

- List comprehensions are 20-30% faster than equivalent `for` loops with `.append()`.
- `__slots__` on classes reduces memory 40-60% per instance.
- `str.join()` over repeated `+=` concatenation (quadratic vs linear).
- `collections.Counter` over manual dict counting.
- `bisect.insort` for maintaining sorted lists without re-sorting.
- Profile with `py-spy` (sampling) or `cProfile` + `snakeviz`. Never optimize without profiling.
- `struct.pack`/`unpack` for binary data — 10x faster than manual byte manipulation.

## Anti-Patterns

- Mutable default arguments: `def f(items=[])` shares state across calls. Use `None` sentinel.
- Bare `except:` catches `SystemExit` and `KeyboardInterrupt`. Use `except Exception:`.
- `isinstance()` chains signal missing polymorphism — use `match`/`case` (3.10+) or dispatch.
- Global imports of heavy modules: use lazy imports inside functions for CLI startup time.
- `time.time()` for benchmarking: use `time.perf_counter()` or `timeit`.
