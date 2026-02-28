# Installation

## Pre-built Binaries (Recommended)

```bash
curl -fsSL https://raw.githubusercontent.com/heartbit-ai/heartbit/main/install.sh | bash
```

Installs to `/usr/local/bin/heartbit`. Supported platforms:
- `x86_64-unknown-linux-gnu` (Linux x86_64)
- `x86_64-apple-darwin` (macOS Intel)
- `aarch64-apple-darwin` (macOS Apple Silicon)

Verify:

```bash
heartbit --version
```

## From Source

### Prerequisites

**Rust** (stable, latest):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

**System libraries** (required for rdkafka and OpenSSL):

Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake libssl-dev pkg-config
```

macOS:

```bash
brew install cmake openssl pkg-config
```

Fedora/RHEL:

```bash
sudo dnf install -y cmake openssl-devel pkg-config gcc
```

### Install via cargo

```bash
cargo install --git https://github.com/heartbit-ai/heartbit heartbit-cli
```

### Build from clone

```bash
git clone https://github.com/heartbit-ai/heartbit.git
cd heartbit
cargo build --release -p heartbit-cli
# Binary at target/release/heartbit
```

### With local embeddings

To enable offline ONNX-based text embeddings (no API keys required):

```bash
cargo build --release -p heartbit-cli --features local-embedding
```

## Docker

### Pre-built image

```bash
docker pull ghcr.io/heartbit-ai/heartbit:latest
```

### Run standalone

```bash
docker run --rm \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  ghcr.io/heartbit-ai/heartbit:latest \
  heartbit "Analyze the Rust ecosystem"
```

### Full stack (docker compose)

```bash
docker compose up -d
```

Services included:

| Service | Image | Ports | Purpose |
|---------|-------|-------|---------|
| `restate` | `restatedev/restate:latest` | 8080 (ingress), 9070 (admin) | Durable workflow execution |
| `heartbit` | Built from `./Dockerfile` | 9080 | Restate HTTP worker |
| `postgres` | `pgvector/pgvector:pg17` | 5432 | Persistent memory + task store |
| `kafka` | `apache/kafka:4.0.0` (KRaft) | 9092 | Event streaming for daemon mode |

Configure by mounting your config and setting API keys:

```bash
# .env file (or export directly)
ANTHROPIC_API_KEY=sk-ant-...
OPENROUTER_API_KEY=sk-or-...
```

The compose file mounts `./heartbit.toml` into the container.

### Build image locally

```bash
docker build -t heartbit .
```

The Dockerfile uses a multi-stage build (rust:slim builder → debian:bookworm-slim runtime) and runs as a non-root `heartbit` user.

## Feature Flags

The `heartbit` library crate uses feature flags to keep the default build lightweight. The CLI crate enables `full` by default.

| Feature | What it adds | Extra system deps |
|---------|-------------|-------------------|
| `core` (default) | Agent runner, orchestrator, LLM providers, tools, memory, config | None |
| `kafka` | Kafka consumer/producer | cmake, libssl-dev, pkg-config |
| `daemon` | HTTP API, cron scheduler, metrics (implies `kafka`) | Same as kafka |
| `sensor` | 7 sensor sources, triage pipeline (implies `daemon`) | Same as kafka |
| `restate` | Durable workflows via Restate SDK 0.8 | None |
| `postgres` | PostgreSQL memory + task store with pgvector | None (runtime: PostgreSQL 12+) |
| `a2a` | Agent-to-Agent protocol | None |
| `telegram` | Telegram bot integration | None |
| `local-embedding` | Local ONNX embeddings via fastembed | None (models auto-download ~30MB) |
| `full` | All of the above except `local-embedding` | All of the above |

Build with specific features:

```bash
# Core only (no Kafka, no Postgres — lightweight)
cargo build -p heartbit

# Library with specific features
cargo build -p heartbit --features kafka,postgres

# Full CLI (all features except local-embedding)
cargo build -p heartbit-cli

# Full CLI with local embeddings
cargo build -p heartbit-cli --features local-embedding
```

## Configuration

Heartbit works in two modes:

### Without config file (env vars)

Set an API key and run directly:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
heartbit "Analyze the Rust ecosystem"

# Or with OpenRouter
export OPENROUTER_API_KEY=sk-or-...
heartbit "Analyze the Rust ecosystem"
```

A single agent runs with 14 built-in tools. See the [README](README.md#environment-variables) for all environment variables.

### With config file

Create a `heartbit.toml` for multi-agent orchestration, memory, sensors, etc.:

```bash
heartbit --config heartbit.toml run "your task"
```

See the [README](README.md#configuration) for the full config reference. Example configs ship in the repo:

| File | Purpose |
|------|---------|
| `heartbit.toml` | Multi-agent orchestrator example |
| `daemon-dev.toml` | Daemon with sensors and Telegram |

### Daemon authentication

The daemon supports two authentication modes (can be combined):

**Bearer tokens** (static API keys):

```toml
[daemon.auth]
bearer_tokens = ["your-api-key-1", "your-api-key-2"]  # multiple for key rotation
```

**JWT/JWKS** (multi-tenant with identity provider):

```toml
[daemon.auth]
jwks_url = "https://idp.example.com/.well-known/jwks.json"
issuer = "https://idp.example.com"     # optional: validate iss claim
audience = "heartbit-daemon"           # optional: validate aud claim
user_id_claim = "sub"                  # claim for user ID (default: "sub")
tenant_id_claim = "tid"                # claim for tenant ID (default: "tid")
roles_claim = "roles"                  # claim for roles (default: "roles")
```

When JWT auth is active, the daemon extracts `UserContext` (user_id, tenant_id, roles) from each request. Memory, workspace, and tasks are automatically scoped per user/tenant.

Claim names are configurable to accommodate different identity providers (e.g., `"org_id"` for tenant, `"permissions"` for roles).

## Optional External Services

These are only needed for specific execution paths:

### Kafka (daemon mode)

```bash
# Via docker compose
docker compose up kafka -d

# Or standalone (KRaft mode, no ZooKeeper)
docker run -d --name kafka -p 9092:9092 apache/kafka:4.0.0
```

Required for: `heartbit daemon`

### Restate (durable execution)

```bash
# Via docker compose
docker compose up restate -d

# Or standalone
docker run -d --name restate -p 8080:8080 -p 9070:9070 \
  docker.restate.dev/restatedev/restate:latest
```

Required for: `heartbit serve`, `heartbit submit`

After starting, register the worker:

```bash
curl -X POST http://localhost:9070/deployments \
  -H 'content-type: application/json' \
  -d '{"uri": "http://localhost:9080"}'
```

### PostgreSQL (persistent storage)

```bash
# Via docker compose (includes pgvector)
docker compose up postgres -d

# Or standalone with pgvector
docker run -d --name postgres -p 5432:5432 \
  -e POSTGRES_USER=heartbit \
  -e POSTGRES_PASSWORD=heartbit \
  -e POSTGRES_DB=heartbit \
  pgvector/pgvector:pg17
```

Configure in TOML:

```toml
[memory]
type = "postgres"
database_url = "postgresql://heartbit:heartbit@localhost:5432/heartbit"
```

### MCP servers (via agentgateway)

```bash
cd gateway && ./start.sh
```

Or configure MCP servers directly:

```toml
[[agents]]
mcp_servers = ["http://localhost:8000/mcp"]
```

Or via env var: `HEARTBIT_MCP_SERVERS=http://localhost:8000/mcp`

## Verification

### Quick smoke test

```bash
export ANTHROPIC_API_KEY=sk-ant-...
heartbit "What is 2+2?"
```

### Interactive chat

```bash
heartbit chat
```

### Run tests (development)

```bash
cargo fmt -- --check && cargo clippy -- -D warnings && cargo test
```

2720+ tests should pass.

## Troubleshooting

### rdkafka build fails

**Symptom**: `cmake` not found or OpenSSL errors during `cargo build`.

**Fix**: Install system dependencies:

```bash
# Ubuntu/Debian
sudo apt-get install -y build-essential cmake libssl-dev pkg-config

# macOS
brew install cmake openssl pkg-config
```

### OpenSSL not found on macOS

**Symptom**: `Could not find directory of OpenSSL installation`.

**Fix**:

```bash
brew install openssl
export OPENSSL_DIR=$(brew --prefix openssl)
export PKG_CONFIG_PATH="$OPENSSL_DIR/lib/pkgconfig"
```

### fastembed model download hangs

**Symptom**: First run with `local-embedding` feature takes a long time.

**Fix**: This is expected — the ONNX model (~30MB) downloads on first use. Subsequent runs use the cached model. Default cache: `~/.cache/fastembed/`. Override with:

```toml
[memory.embedding]
provider = "local"
cache_dir = "/tmp/fastembed"
```

### PostgreSQL connection refused

**Symptom**: `connection refused` when using postgres memory store.

**Fix**: Ensure PostgreSQL is running and accessible:

```bash
docker compose ps   # check health
docker compose logs postgres  # check logs
```

Default docker-compose credentials: `heartbit:heartbit@localhost:5432/heartbit`.

### Kafka topics not created

**Symptom**: Daemon fails to consume messages.

**Fix**: Topics are auto-created by the daemon on startup. If using a restricted Kafka cluster, create them manually:

```bash
kafka-topics.sh --create --topic hb.daemon.commands \
  --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
```

### Rust edition 2024 errors

**Symptom**: Compiler errors about unsupported edition.

**Fix**: Update to latest stable Rust:

```bash
rustup update stable
```

The project uses Rust edition 2024, which requires Rust 1.85+.
