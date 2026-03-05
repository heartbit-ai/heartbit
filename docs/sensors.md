# Sensor Pipeline

Data ingestion pipeline from 7 external sources, through triage and story correlation, to daemon commands.

## Architecture

```
  Sources                  Triage              Stories         Daemon
  -------                  ------              -------         ------
  +-----------+
  |  RSS      |--+
  +----------+  |
  |  Email    |--+      +------------+     +------------+   +---------+
  |  (JMAP/   |  +----->|  Triage     |---->|  Story      |-->| Command  |
  |  Google)  |--+      |  (per-type  |    |  Builder    |   | Producer |
  +----------+  |      |   scoring)  |    |  (dedup,    |   | -> Kafka |
  |  Webhook  |--+      +------------+    |   merge)    |   +---------+
  +----------+  |                         +------------+
  |  Weather  |--+
  +----------+  |
  |  Audio    |--+
  +----------+  |
  |  Image    |--+
  +----------+
```

## Sources

Each source implements the `Sensor` trait with `name()`, `modality()`, and `run()`:

| Source | Modality | Description |
|--------|----------|-------------|
| RSS | Text | Poll RSS/Atom feeds |
| Email (JMAP) | Text | Google Workspace / JMAP email monitoring |
| Webhook | Text | HTTP webhook receiver |
| Weather | Text | Weather data polling |
| Audio | Audio | Audio input processing |
| Image | Image | Image input processing |
| MCP | Text | MCP server events |

## Triage

Per-modality classifiers score urgency and relevance of incoming sensor events.

## Stories

The `StoryCorrelator` aggregates related events:
- **Deduplication** — prevents the same event from triggering multiple actions
- **Merge** — combines related events into a single story
- **Action production** — completed stories produce `DaemonCommand` entries sent to Kafka

## Configuration

```toml
[daemon.sensors]
# Sensor sources are configured per-type
# See daemon configuration for full reference
```

## Feature Flag

Sensors require the `sensor` feature flag (which implies `daemon`):

```bash
cargo build --features sensor
```
