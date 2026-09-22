"""Unit tests for post-run Langfuse ingest (stdlib only, no Langfuse server)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from heartbit_tb2.langfuse_ingest import (
    build_job_batch,
    build_trial_batch,
    discover_trials,
    reward_from_result,
    truncate,
)


def _write_trial(root: Path, name: str, *, reward: float = 1.0) -> Path:
    trial = root / name
    agent = trial / "agent"
    agent.mkdir(parents=True)
    (agent / "heartbit-trace.json").write_text(
        json.dumps(
            {
                "model_name": "qwen3.8-27b",
                "result": "done",
                "tool_calls_made": 2,
                "tokens_used": {
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "reasoning_tokens": 5,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                },
                "tool_call_results": [
                    {
                        "tool_name": "bash",
                        "tool_call_id": "c1",
                        "input": "echo hi",
                        "output": "hi",
                        "is_error": False,
                        "duration_ms": 12,
                    },
                    {
                        "tool_name": "read",
                        "tool_call_id": "c2",
                        "input": {"path": "/app/x"},
                        "output": "api_key=sk-secret-should-hide",
                        "is_error": False,
                        "duration_ms": 3,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "terminal-bench/demo",
                "started_at": "2026-09-22T14:00:00Z",
                "finished_at": "2026-09-22T14:05:00Z",
                "config": {"job_id": "job-1"},
                "verifier_result": {"rewards": {"reward": reward}},
            }
        ),
        encoding="utf-8",
    )
    return trial


def test_truncate_redacts_secretish_and_caps_length():
    long = "x" * 10_000
    out = truncate(long, limit=100)
    assert out is not None
    assert len(out) <= 100
    assert "[+" in out
    red = truncate("Authorization: Bearer abcdef and more")
    assert red is not None
    assert "[redacted]" in red
    assert "abcdef" not in red


def test_reward_from_result():
    assert reward_from_result({"verifier_result": {"rewards": {"reward": 1.0}}}) == 1.0
    assert reward_from_result({}) is None
    assert reward_from_result(None) is None


def test_build_trial_batch_shape():
    with tempfile.TemporaryDirectory() as td:
        trial = _write_trial(Path(td), "demo__abc")
        batch = build_trial_batch(trial, session_id="sess")
    types = [e["type"] for e in batch]
    assert "trace-create" in types
    assert "generation-create" in types
    assert types.count("span-create") == 2
    assert "score-create" in types
    score = next(e for e in batch if e["type"] == "score-create")
    assert score["body"]["value"] == 1.0
    assert score["body"]["name"] == "reward"
    read_span = next(
        e for e in batch if e["type"] == "span-create" and e["body"]["name"] == "tool:read"
    )
    assert "[redacted]" in (read_span["body"]["output"] or "")
    assert "sk-secret" not in (read_span["body"]["output"] or "")
    gen = next(e for e in batch if e["type"] == "generation-create")
    assert gen["body"]["model"] == "qwen3.8-27b"
    assert gen["body"]["usage"]["input"] == 100
    assert gen["body"]["usage"]["output"] == 20


def test_discover_and_job_batch():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _write_trial(root, "a__1", reward=1.0)
        _write_trial(root, "b__2", reward=0.0)
        (root / "noise.txt").write_text("x", encoding="utf-8")
        trials = discover_trials(root)
        assert len(trials) == 2
        batch = build_job_batch(root)
        scores = [e for e in batch if e["type"] == "score-create"]
        assert sorted(s["body"]["value"] for s in scores) == [0.0, 1.0]
        traces = [e for e in batch if e["type"] == "trace-create"]
        assert all(t["body"]["sessionId"] == root.name for t in traces)


def test_empty_trial_returns_empty():
    with tempfile.TemporaryDirectory() as td:
        empty = Path(td) / "empty__x"
        (empty / "agent").mkdir(parents=True)
        assert build_trial_batch(empty) == []
