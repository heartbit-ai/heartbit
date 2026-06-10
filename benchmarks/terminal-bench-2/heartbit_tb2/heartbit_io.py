"""Pure, Harbor-free helpers for the heartbit Terminal-Bench adapter.

Kept import-light (stdlib only) so they can be unit-tested without installing
Harbor or building heartbit.
"""

from __future__ import annotations

import json
import re
from typing import Any

# Harbor passes model ids as ``provider/model`` (e.g. ``anthropic/claude-haiku-4-5``).
# heartbit selects its provider via HEARTBIT_PROVIDER and the model via
# HEARTBIT_MODEL (no ``provider/`` prefix), reading the key from the provider's
# env var. Map the common providers to their key env names.
_PROVIDER_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "google": "GEMINI_API_KEY",
}


def parse_model_id(model_name: str | None) -> tuple[str | None, str | None]:
    """Split Harbor's ``provider/model`` into ``(provider, model)``.

    No slash -> ``(None, model)`` (let heartbit auto-detect the provider).
    Empty/None -> ``(None, None)`` (heartbit falls back to its own defaults).
    """
    if not model_name:
        return None, None
    if "/" in model_name:
        provider, model = model_name.split("/", 1)
        return provider.strip() or None, model.strip() or None
    return None, model_name.strip() or None


def provider_api_key_env(provider: str | None) -> str | None:
    """The env var holding the API key for a given provider (case-insensitive)."""
    if not provider:
        return None
    return _PROVIDER_KEY_ENV.get(provider.lower())


def build_heartbit_env(
    model_name: str | None,
    base_env: dict[str, str],
    workspace: str,
    max_turns: str = "60",
    tool_timeout: str = "600",
    noninteractive: bool = True,
) -> dict[str, str]:
    """Assemble the env dict for ``heartbit run`` inside the container.

    Forwards only the relevant provider key from ``base_env``; pins the provider
    and workspace so a stray key or the default jail can't take over.
    """
    provider, model = parse_model_id(model_name)
    env: dict[str, str] = {}

    # Preserve PATH/HOME so the binary and its TLS roots resolve.
    for k in ("PATH", "HOME", "LANG", "LC_ALL", "TERM"):
        if k in base_env:
            env[k] = base_env[k]
    env.setdefault("HOME", "/root")

    if provider:
        env["HEARTBIT_PROVIDER"] = provider
        key_env = provider_api_key_env(provider)
        if key_env and key_env in base_env:
            env[key_env] = base_env[key_env]
            # Generic per-run override heartbit also accepts.
            env["HEARTBIT_API_KEY"] = base_env[key_env]
    else:
        # No provider prefix: forward whatever key is present so auto-detect works.
        for key_env in set(_PROVIDER_KEY_ENV.values()):
            if key_env in base_env:
                env[key_env] = base_env[key_env]
    if model:
        env["HEARTBIT_MODEL"] = model

    # Optional OpenAI-compatible proxy base url (e.g. for a gateway).
    if "HEARTBIT_BASE_URL" in base_env:
        env["HEARTBIT_BASE_URL"] = base_env["HEARTBIT_BASE_URL"]

    env["HEARTBIT_WORKSPACE"] = workspace
    env["HEARTBIT_MAX_TURNS"] = str(max_turns)
    env["HEARTBIT_TOOL_TIMEOUT"] = str(tool_timeout)
    if noninteractive:
        env["HEARTBIT_NONINTERACTIVE"] = "1"
    return env


def parse_trace_tokens(trace_text: str) -> dict[str, Any] | None:
    """Parse the ``--trace-file`` AgentOutput JSON into AgentContext fields."""
    try:
        data = json.loads(trace_text)
    except (json.JSONDecodeError, TypeError):
        return None
    usage = data.get("tokens_used") or {}
    return {
        "n_input_tokens": usage.get("input_tokens"),
        "n_output_tokens": usage.get("output_tokens"),
        # heartbit reports cache read + creation separately; report their sum.
        "n_cache_tokens": (usage.get("cache_read_input_tokens", 0) or 0)
        + (usage.get("cache_creation_input_tokens", 0) or 0),
        "cost_usd": data.get("estimated_cost_usd"),
        "metadata": {
            "model_name": data.get("model_name"),
            "tool_calls_made": data.get("tool_calls_made"),
            "goal_met": data.get("goal_met"),
            "reasoning_tokens": usage.get("reasoning_tokens"),
            "result_preview": (data.get("result") or "")[:500],
            "source": "trace-file",
        },
    }


_STDERR_TOKENS_RE = re.compile(r"Tokens used:\s*(\d+)\s*in\s*/\s*(\d+)\s*out")


def parse_stderr_tokens(stderr_text: str) -> dict[str, Any] | None:
    """Fallback: scrape the ``Tokens used: X in / Y out`` footer from stderr.

    Used when ``--trace-file`` is unavailable (e.g. an older binary without the
    benchmark hook). Returns ``None`` if the footer is absent.
    """
    m = None
    for m in _STDERR_TOKENS_RE.finditer(stderr_text or ""):
        pass  # keep the LAST match (the final run's footer)
    if m is None:
        return None
    return {
        "n_input_tokens": int(m.group(1)),
        "n_output_tokens": int(m.group(2)),
        "n_cache_tokens": None,
        "cost_usd": None,
        "metadata": {"source": "stderr-footer"},
    }
