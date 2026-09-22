"""Post-run Langfuse ingest for Heartbit TB2 / TUI traces.

Host-side only (stdlib). Builds a Langfuse batch-ingestion payload from a
Harbor trial directory (``agent/heartbit-trace.json`` + ``result.json`` reward)
so we can tune the harness without shipping OTLP inside the slim musl binary.

API: ``POST {LANGFUSE_HOST}/api/public/ingestion`` (Basic auth public:secret).
"""

from __future__ import annotations

import base64
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# Tool I/O can be huge (bash dumps); keep the UI usable and avoid shipping
# accidental secrets. Live TB2 trial had 82 tools — full outputs blow past
# sensible ingest sizes.
_DEFAULT_IO_CHARS = 4_000
_SECRETISH = re.compile(
    r"(?i)((?:api[_-]?key|secret|token|password|authorization)\s*[:=]\s*)\S+"
)
_BEARER = re.compile(r"(?i)(bearer\s+)\S+")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _new_id() -> str:
    return str(uuid.uuid4())


def truncate(text: str | None, limit: int = _DEFAULT_IO_CHARS) -> str | None:
    if text is None:
        return None
    # Bearer first: otherwise `Authorization: Bearer <tok>` is partially
    # eaten by the key=value pattern and leaves the token behind.
    cleaned = _BEARER.sub(r"\1[redacted]", text)
    cleaned = _SECRETISH.sub(r"\1[redacted]", cleaned)
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 20] + f"…[+{len(cleaned) - limit + 20} chars]"


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def discover_trials(job_dir: Path) -> list[Path]:
    """Harbor job dir → trial subdirs that contain an agent/ folder."""
    if not job_dir.is_dir():
        return []
    trials: list[Path] = []
    for child in sorted(job_dir.iterdir()):
        if child.is_dir() and (child / "agent").is_dir():
            trials.append(child)
    return trials


def reward_from_result(result: dict[str, Any] | None) -> float | None:
    if not result:
        return None
    verifier = result.get("verifier_result") or {}
    rewards = verifier.get("rewards") or {}
    if "reward" in rewards:
        try:
            return float(rewards["reward"])
        except (TypeError, ValueError):
            return None
    return None


def build_trial_batch(
    trial_dir: Path,
    *,
    io_chars: int = _DEFAULT_IO_CHARS,
    session_id: str | None = None,
) -> list[dict[str, Any]]:
    """One Langfuse batch (list of events) for a single Harbor trial."""
    trace_path = trial_dir / "agent" / "heartbit-trace.json"
    result_path = trial_dir / "result.json"
    trace = load_json(trace_path)
    result = load_json(result_path)
    if trace is None and result is None:
        return []

    trial_name = trial_dir.name
    task_name = (result or {}).get("task_name") or trial_name
    model = (trace or {}).get("model_name") or (
        ((result or {}).get("agent_info") or {}).get("model_info") or {}
    ).get("name")
    tokens = (trace or {}).get("tokens_used") or {}
    tools = (trace or {}).get("tool_call_results") or []
    reward = reward_from_result(result)
    started = (result or {}).get("started_at") or _now_iso()
    finished = (result or {}).get("finished_at") or started
    session = session_id or (result or {}).get("config", {}).get("job_id") or trial_dir.parent.name

    trace_id = _new_id()
    events: list[dict[str, Any]] = []

    metadata = {
        "task_name": task_name,
        "trial_name": trial_name,
        "tool_calls_made": (trace or {}).get("tool_calls_made")
        or ((result or {}).get("agent_result") or {}).get("metadata", {}).get("tool_calls_made"),
        "source": "heartbit-tb2-post-run",
        "trace_file": str(trace_path) if trace_path.is_file() else None,
    }

    events.append(
        {
            "id": _new_id(),
            "type": "trace-create",
            "timestamp": started,
            "body": {
                "id": trace_id,
                "name": task_name,
                "sessionId": session,
                "userId": "heartbit-harness",
                "timestamp": started,
                "tags": ["tb2", "heartbit", "post-run"],
                "metadata": metadata,
                "input": {"task": task_name, "model": model},
                "output": {
                    "result_preview": truncate(
                        (trace or {}).get("result")
                        or ((result or {}).get("agent_result") or {})
                        .get("metadata", {})
                        .get("result_preview"),
                        io_chars,
                    ),
                    "reward": reward,
                },
            },
        }
    )

    # Aggregated "generation": we don't have per-LLM-call payloads in
    # --trace-file, so one observation carries the run's token budget.
    gen_id = _new_id()
    events.append(
        {
            "id": _new_id(),
            "type": "generation-create",
            "timestamp": finished,
            "body": {
                "id": gen_id,
                "traceId": trace_id,
                "name": "heartbit-run",
                "startTime": started,
                "endTime": finished,
                "model": model,
                "input": {"task": task_name},
                "output": truncate((trace or {}).get("result"), io_chars),
                "usage": {
                    "input": tokens.get("input_tokens"),
                    "output": tokens.get("output_tokens"),
                    "total": (tokens.get("input_tokens") or 0)
                    + (tokens.get("output_tokens") or 0),
                },
                "metadata": {
                    "reasoning_tokens": tokens.get("reasoning_tokens"),
                    "cache_read_input_tokens": tokens.get("cache_read_input_tokens"),
                    "cache_creation_input_tokens": tokens.get("cache_creation_input_tokens"),
                },
            },
        }
    )

    for i, tool in enumerate(tools):
        if not isinstance(tool, dict):
            continue
        name = tool.get("tool_name") or "tool"
        span_id = _new_id()
        duration_ms = tool.get("duration_ms")
        # Approximate span times from order when absolute times are absent.
        events.append(
            {
                "id": _new_id(),
                "type": "span-create",
                "timestamp": started,
                "body": {
                    "id": span_id,
                    "traceId": trace_id,
                    "name": f"tool:{name}",
                    "startTime": started,
                    "endTime": finished,
                    "input": truncate(
                        tool.get("input")
                        if isinstance(tool.get("input"), str)
                        else json.dumps(tool.get("input"), ensure_ascii=False)
                        if tool.get("input") is not None
                        else None,
                        io_chars,
                    ),
                    "output": truncate(
                        tool.get("output")
                        if isinstance(tool.get("output"), str)
                        else json.dumps(tool.get("output"), ensure_ascii=False)
                        if tool.get("output") is not None
                        else None,
                        io_chars,
                    ),
                    "level": "ERROR" if tool.get("is_error") else "DEFAULT",
                    "statusMessage": "error" if tool.get("is_error") else None,
                    "metadata": {
                        "tool_call_id": tool.get("tool_call_id"),
                        "duration_ms": duration_ms,
                        "index": i,
                    },
                },
            }
        )

    if reward is not None:
        events.append(
            {
                "id": _new_id(),
                "type": "score-create",
                "timestamp": finished,
                "body": {
                    "id": _new_id(),
                    "traceId": trace_id,
                    "name": "reward",
                    "value": reward,
                    "dataType": "NUMERIC",
                    "comment": f"Harbor verifier reward for {task_name}",
                },
            }
        )

    return events


def build_job_batch(
    job_dir: Path,
    *,
    io_chars: int = _DEFAULT_IO_CHARS,
) -> list[dict[str, Any]]:
    """All trials under a Harbor job directory → one flat batch."""
    session = job_dir.name
    batch: list[dict[str, Any]] = []
    for trial in discover_trials(job_dir):
        batch.extend(build_trial_batch(trial, io_chars=io_chars, session_id=session))
    return batch


def post_batch(
    batch: list[dict[str, Any]],
    *,
    host: str,
    public_key: str,
    secret_key: str,
    timeout_secs: float = 60.0,
) -> dict[str, Any]:
    """POST the batch to Langfuse. Returns the parsed JSON response body."""
    if not batch:
        return {"ok": True, "skipped": True, "reason": "empty batch"}
    base = host.rstrip("/")
    url = f"{base}/api/public/ingestion"
    token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    body = json.dumps({"batch": batch}).encode("utf-8")
    req = Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Basic {token}",
            "Content-Type": "application/json",
            "User-Agent": "heartbit-tb2-langfuse-ingest/1",
        },
    )
    try:
        with urlopen(req, timeout=timeout_secs) as resp:
            raw = resp.read().decode("utf-8") or "{}"
            return json.loads(raw)
    except HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Langfuse HTTP {e.code}: {detail}") from e
    except URLError as e:
        raise RuntimeError(f"Langfuse unreachable at {url}: {e}") from e
