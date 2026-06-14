"""Unit tests for the Harbor-free helpers (run with: python -m pytest, no Harbor needed)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heartbit_tb2.heartbit_io import (  # noqa: E402
    build_heartbit_env,
    is_static_musl,
    parse_model_id,
    parse_stderr_tokens,
    parse_trace_tokens,
    provider_api_key_env,
)


def test_is_static_musl_detects_musl_path_and_flag():
    assert is_static_musl("/repo/target/x86_64-unknown-linux-musl/release/heartbit")
    assert not is_static_musl("/repo/target/release/heartbit")
    assert is_static_musl("/custom/heartbit", static_env="1")
    assert not is_static_musl("/custom/heartbit", static_env="0")


def test_parse_model_id_splits_provider_prefix():
    assert parse_model_id("anthropic/claude-haiku-4-5") == ("anthropic", "claude-haiku-4-5")


def test_parse_model_id_no_prefix_defers_provider():
    assert parse_model_id("claude-haiku-4-5") == (None, "claude-haiku-4-5")


def test_parse_model_id_empty():
    assert parse_model_id(None) == (None, None)
    assert parse_model_id("") == (None, None)


def test_provider_api_key_env_mapping():
    assert provider_api_key_env("anthropic") == "ANTHROPIC_API_KEY"
    assert provider_api_key_env("OpenAI") == "OPENAI_API_KEY"
    assert provider_api_key_env("openrouter") == "OPENROUTER_API_KEY"
    assert provider_api_key_env("unknown") is None
    assert provider_api_key_env(None) is None


def test_build_env_pins_provider_and_forwards_only_that_key():
    env = build_heartbit_env(
        model_name="anthropic/claude-haiku-4-5",
        base_env={
            "ANTHROPIC_API_KEY": "sk-ant-xxx",
            "OPENAI_API_KEY": "sk-oai-should-not-leak",
            "PATH": "/usr/bin",
        },
        workspace="/app",
    )
    assert env["HEARTBIT_PROVIDER"] == "anthropic"
    assert env["HEARTBIT_MODEL"] == "claude-haiku-4-5"
    assert env["ANTHROPIC_API_KEY"] == "sk-ant-xxx"
    assert env["HEARTBIT_API_KEY"] == "sk-ant-xxx"
    # The unrelated provider key must NOT be forwarded when a provider is pinned.
    assert "OPENAI_API_KEY" not in env
    assert env["HEARTBIT_WORKSPACE"] == "/app"
    assert env["HEARTBIT_NONINTERACTIVE"] == "1"
    assert env["HEARTBIT_MAX_TURNS"] == "60"


def test_build_env_codex_proxy_orchestrator_shape():
    # Benchmark the TUI "brain" on a ChatGPT-subscription Codex proxy: provider
    # "codex" is unknown to heartbit, so with a base_url and NO key it uses
    # AuthStyle::None (the only style that permits an http:// gateway URL).
    env = build_heartbit_env(
        model_name="codex/gpt-5.5",
        base_env={
            "HEARTBIT_BASE_URL": "http://172.17.0.1:10531/v1",
            "HEARTBIT_ORCHESTRATOR": "1",
            "HEARTBIT_SUB_AGENT_MAX_TURNS": "200",
            # A stray key for another provider must NOT leak to the codex run.
            "ANTHROPIC_API_KEY": "sk-ant-should-not-leak",
        },
        workspace="/app",
    )
    assert env["HEARTBIT_PROVIDER"] == "codex"
    assert env["HEARTBIT_MODEL"] == "gpt-5.5"
    assert env["HEARTBIT_BASE_URL"] == "http://172.17.0.1:10531/v1"
    assert env["HEARTBIT_ORCHESTRATOR"] == "1"
    assert env["HEARTBIT_SUB_AGENT_MAX_TURNS"] == "200"
    # No key for an unknown provider (AuthStyle::None over http).
    assert "HEARTBIT_API_KEY" not in env
    assert "ANTHROPIC_API_KEY" not in env


def test_build_env_no_prefix_forwards_available_keys_for_autodetect():
    env = build_heartbit_env(
        model_name="claude-haiku-4-5",
        base_env={"ANTHROPIC_API_KEY": "sk-ant"},
        workspace="/work",
    )
    assert "HEARTBIT_PROVIDER" not in env
    assert env["HEARTBIT_MODEL"] == "claude-haiku-4-5"
    assert env["ANTHROPIC_API_KEY"] == "sk-ant"


def test_parse_trace_tokens_extracts_usage_and_cost():
    trace = json.dumps(
        {
            "result": "done" * 400,
            "tool_calls_made": 7,
            "tokens_used": {
                "input_tokens": 1200,
                "output_tokens": 340,
                "cache_read_input_tokens": 800,
                "cache_creation_input_tokens": 200,
                "reasoning_tokens": 50,
            },
            "estimated_cost_usd": 0.0123,
            "model_name": "claude-haiku-4-5-20251001",
            "goal_met": True,
        }
    )
    t = parse_trace_tokens(trace)
    assert t["n_input_tokens"] == 1200
    assert t["n_output_tokens"] == 340
    assert t["n_cache_tokens"] == 1000  # read + creation
    assert t["cost_usd"] == 0.0123
    assert t["metadata"]["model_name"] == "claude-haiku-4-5-20251001"
    assert t["metadata"]["tool_calls_made"] == 7
    assert t["metadata"]["goal_met"] is True
    assert len(t["metadata"]["result_preview"]) <= 500


def test_parse_trace_tokens_rejects_garbage():
    assert parse_trace_tokens("not json") is None


def test_parse_stderr_tokens_scrapes_last_footer():
    stderr = (
        "[event] {...}\n"
        "---\nTokens used: 10 in / 5 out | Tool calls: 1\n"
        "more logs\n"
        "---\nTokens used: 4200 in / 870 out | Tool calls: 12\n"
    )
    t = parse_stderr_tokens(stderr)
    assert t["n_input_tokens"] == 4200  # the LAST footer wins
    assert t["n_output_tokens"] == 870
    assert t["metadata"]["source"] == "stderr-footer"


def test_parse_stderr_tokens_absent_returns_none():
    assert parse_stderr_tokens("no footer here") is None
