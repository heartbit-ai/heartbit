# Operating Heartbit

Operational reference for running `heartbit daemon` in production. Focuses on the knobs an operator turns day-to-day (cadence, jitter, kill switches, observability) rather than the full TOML schema — see [`configuration.md`](configuration.md) for that.

For the daemon's overall architecture (Kafka, HTTP API, channels), see [`daemon.md`](daemon.md).

## Pre-flight

Validate a config without starting Kafka, HTTP, or the database:

```bash
heartbit daemon --config heartbit.toml --validate-config
```

Exit code is non-zero with a list of issues if anything would cause startup misbehavior — most importantly:
- A `[[daemon.persona_posts]]` entry whose operator user-id is unresolvable
- A JSONL store with a missing parent directory
- Required path/identifier fields left empty

Run this as part of CI or a deploy hook before rolling out config changes.

## Proactive posting knobs

`[[daemon.persona_posts]]` controls the proactive-post loop. Defaults are listed in [`configuration.md`](configuration.md#daemon).

| Knob | Default | When to change |
|---|---|---|
| `enabled` | `true` | Set to `false` to pause posting for this persona without removing the block. |
| `post_interval_seconds` | `14400` (4h) | Lower for higher-volume personas; minimum is `60`. |
| `interval_jitter_pct` | `25` (±25%) | Lower for stricter cadence (debugging only); higher (up to `50`) to look less bot-like. `0` disables jitter — test only. |
| `active_hours` | unset (24/7) | Set to e.g. `"08:00-22:00"` to restrict to local waking hours. |
| `candidates_per_draft` | `3` | Higher = more LLM cost per tick but better picks. |
| `post_history_store` | `"in_memory"` | Use `"jsonl"` for restart durability. |
| `post_history_path` | required for jsonl | Tilde-expanded; ensure the parent directory exists. |
| `post_history_lookback_days` | `30` | How far back duplicate-topic detection scans. |
| `topic_brief` | unset | Free-form prompt addendum for the topic generator. |

## Engagement-feedback loop

Engagement metrics are refreshed in the background and the top-N engaged posts are injected into the writer as few-shot exemplars.

| Knob | Default | Behavior |
|---|---|---|
| `engagement_refresh_seconds` | `21600` (6h) | Tick interval for the engagement collector. |
| `engagement_top_n` | `5` | Number of top-engaged posts injected as writer exemplars. **Set to `0` to disable injection** — fastest kill switch for the feedback loop. |
| `engagement_min_age_hours` | `24` | Tweets younger than this are skipped (algorithm hasn't fanned out yet). |
| `engagement_max_age_days` | `30` | Tweets older than this are dropped from refresh. |

Engagement metrics live alongside the post history in `.heartbit/engagement/{persona}.jsonl` (jsonl mode) — the file is operator-readable JSONL.

## Mention polling knobs

`[[daemon.persona_mentions]]` controls reactive replies. Two safety layers matter most for ops.

**Thread / bot guards** — leave on unless debugging:
- `enable_thread_depth_guard = true` skips threads this persona already replied to.
- `enable_bot_heuristic_guard = true` evaluates handle patterns, follower ratio, and account age.
- `bot_heuristic_threshold = 2` is the number of signals required to skip.

**Per-conversation cap** — `per_conversation_max_replies = 2` prevents back-and-forth lockup.

**Daily LLM budget** — `daily_token_budget = 100000` (set to `null` to disable) is the safest hard stop when the bot is over-engaging.

## Kill switches

In order of granularity:

1. **Disable one entry**: set `enabled = false` on the specific `[[daemon.persona_posts]]` or `[[daemon.persona_mentions]]` block and reload the daemon.
2. **Disable engagement injection**: `engagement_top_n = 0` keeps the bot posting but removes top-engaged few-shots — useful if the feedback loop is producing degenerate writing.
3. **Pause posting only**: set `active_hours = "00:00-00:01"` (a one-minute window in the past or future). Crude but it doesn't lose any history state.
4. **Stop the daemon**: graceful Ctrl-C / SIGTERM — in-flight ticks complete; queued ticks are dropped.

## Operator user-id resolution

For `[[daemon.persona_posts]]` to function, each enabled entry needs an X user-id. The daemon resolves it in this order at startup:

1. A matching `[[daemon.persona_mentions]]` entry's `user_id` field — preferred (single source of truth in config).
2. The `HEARTBIT_GHOST_OPERATOR_USER_ID` environment variable — kept for backward-compat and quick overrides.
3. **Skip this entry**: the daemon logs an `ERROR ... SKIPPING [[daemon.persona_posts]] entry: ...` banner and increments `heartbit_persona_posts_skipped_total{persona, reason}`. Other personas/entries continue to run; the daemon does **not** crash-loop.

Set an alert on `rate(heartbit_persona_posts_skipped_total[5m]) > 0` so silent skips don't go unnoticed.

The one-off `heartbit persona post <name>` CLI (without `--topic`) does **not** apply this fallback — it errors hard so a misconfigured one-shot run fails fast and visibly to the operator at the terminal.

## Environment variables operators care about

The full list is in [`configuration.md`](configuration.md#environment-variables). The subset commonly tuned at deploy time:

| Variable | Purpose |
|---|---|
| `HEARTBIT_GHOST_OPERATOR_USER_ID` | X user-id fallback for persona_posts (per the resolution order above). |
| `HEARTBIT_GHOST_PERSONAS` | Override persona-config directory (defaults to `~/.heartbit/personas`). |
| `HEARTBIT_GHOST_PROFILES` | Override voice-profile directory. |
| `HEARTBIT_GHOST_CORPORA` | Override corpus directory. |
| `HEARTBIT_TOOL_PROFILE` | Pre-filter tool definitions: `conversational` / `standard` / `full`. |
| `HEARTBIT_AUDIT_RETAIN_DAYS` | Days to keep audit-log rows before pruning. |
| `HEARTBIT_SESSION_PRUNE` | `1` to trim old tool results before each LLM call. |
| `HEARTBIT_TELEGRAM_TOKEN` | Telegram bot token for review-delivery / interactive channel. |

## Observability quick-reference

`/metrics` exposes Prometheus counters. Most useful for ops:

| Metric | Why it matters |
|---|---|
| `heartbit_persona_posts_skipped_total{persona, reason}` | Increments when a persona_posts entry is silently disabled at startup. Alert on `rate > 0`. |
| `heartbit_daemon_tasks_failed_total{tenant}` | Per-tenant task failure counter. |
| `heartbit_llm_cost_usd_total{agent, tenant}` | Running LLM-cost estimate. |
| `heartbit_reliability_doom_loops_detected_total` | Bumps when the doom-loop guard short-circuits a runaway agent. |
| `heartbit_cascade_escalations_total{from_tier, to_tier, reason}` | LLM cascade escalations. |

`/healthz` and `/readyz` return 200 when the daemon is up and ready. Use `/readyz` for load-balancer probes.

## Common operations

**Pause one persona without losing history**: edit the block, set `enabled = false`, restart the daemon. The JSONL store on disk is unchanged.

**Reset engagement few-shots cleanly**: stop the daemon, move `.heartbit/engagement/{persona}.jsonl` aside (don't delete — keep for postmortem), restart. Writer falls back to no exemplars on the next tick.

**Recover from a runaway bot**: bump `daily_token_budget` low (e.g. `10000`), restart. The bot will hit the cap quickly and stop replying; investigate logs without further engagement on X.

**Migrate JSONL stores**: the JSONL files are append-only; copying them between hosts preserves history. Ensure parent dirs exist on the destination, then `--validate-config` to confirm.
