# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [2026.2.26] - 2026-02-26

### Added

- Multi-agent runtime with orchestrator and sub-agents (flat hierarchy, parallel dispatch via `tokio::JoinSet`).
- Three execution paths: standalone (in-process), Restate (durable with replay), and daemon (Kafka-backed with HTTP API).
- 14 built-in tools: bash, read, write, edit, patch, glob, grep, list, webfetch, websearch, todowrite, todoread, skill, question.
- MCP Streamable HTTP client (protocol `2025-03-26`) with automatic tool discovery and optional authentication.
- LLM providers: Anthropic and OpenRouter with SSE streaming.
- Retry provider with exponential backoff on 429/5xx errors.
- Cascading provider: tries cheapest model first, escalates on gate rejection or error.
- Prompt caching for Anthropic (cache reads at 10% input rate, writes at 125%).
- Structured output via synthetic `__respond__` tool with JSON Schema validation.
- Context management strategies: unlimited, sliding window, and LLM-generated summarization.
- Memory system with in-memory and PostgreSQL backends (store, recall, update, forget, consolidate).
- Composite recall scoring (recency, importance, relevance, strength) with Ebbinghaus decay.
- Knowledge base with paragraph-aware chunking, keyword search, and file/glob/URL loaders.
- Guardrails: pre/post LLM and tool hooks with allow/deny actions.
- Human-in-the-loop approval for tool execution (`--approve` flag).
- Sensor pipeline with 6 sources, triage, deduplication, and story grouping.
- Telegram bot integration with DM support, streaming responses, and multimodal input (photos, voice, documents).
- Daemon mode: Kafka consumer loop, Axum HTTP API, SSE event streaming, cron scheduler, heartbeat pulse.
- Cross-agent coordination via shared blackboard and memory tools.
- Dynamic task routing based on complexity heuristics.
- Agent workspace with path traversal prevention.
- Cost tracking with per-model pricing for Claude 4, 3.5, and 3 generations.
- 13 structured agent event variants with JSON stderr output (`--verbose`).
- OpenTelemetry tracing via OTLP exporter.
- Interactive chat mode with multi-turn REPL.
- Docker support with multi-stage build.
- Doom loop detection and auto-compaction on context overflow.
- Tool output truncation with UTF-8 safe boundaries.
- Tool name repair via Levenshtein distance matching.
