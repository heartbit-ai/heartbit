---
name = "docker"
description = "Multi-stage builds, security hardening, compose, layer caching, and debugging"
tags = ["docker", "containers", "security", "compose", "devops"]
max_inject_tokens = 2000
---

# Docker Expert

## Multi-Stage Builds

Separate build and runtime stages. Copy only artifacts, not build tooling.

```dockerfile
FROM rust:1.82-bookworm AS builder
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo "fn main(){}" > src/main.rs && cargo build --release && rm -rf src
COPY src/ src/
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/myapp /usr/local/bin/
USER 1000:1000
ENTRYPOINT ["myapp"]
```

The dummy `cargo build` trick pre-caches dependencies — source changes don't re-download crates.

## Layer Caching

Order instructions from least to most frequently changing. `COPY` invalidates cache for everything after it.

- `COPY package.json` + `RUN npm install` before `COPY . .` — dependency layer cached on source changes.
- Use `.dockerignore` to exclude `.git/`, `target/`, `node_modules/`, `*.md`, test fixtures.
- `--mount=type=cache,target=/root/.cargo/registry` for persistent build caches in BuildKit.

## Security Hardening

- Never run as root: `USER 1000:1000` or create a dedicated user.
- Use distroless or `*-slim` base images. Alpine saves 50MB+ but uses musl (glibc compat issues).
- Pin image digests for reproducibility: `FROM node:20@sha256:abc123...`.
- No secrets in build args or ENV. Use `--mount=type=secret` with BuildKit.
- `HEALTHCHECK CMD curl -f http://localhost:8080/healthz || exit 1` — gives orchestrators signal.
- Scan with `docker scout cves` or `trivy image myapp:latest`.
- Drop capabilities: `--cap-drop=ALL --cap-add=NET_BIND_SERVICE` in compose/run.
- Read-only filesystem: `--read-only --tmpfs /tmp` prevents runtime modification.

## Docker Compose

```yaml
services:
  app:
    build:
      context: .
      target: runtime
    ports: ["8080:8080"]
    depends_on:
      db:
        condition: service_healthy
    environment:
      DATABASE_URL: postgres://user:pass@db:5432/mydb
    deploy:
      resources:
        limits:
          memory: 512M
          cpus: "1.0"

  db:
    image: postgres:16
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user"]
      interval: 5s
      retries: 3
```

Use `depends_on.condition: service_healthy` instead of `sleep` hacks. Named volumes for persistence.

## Debugging

- `docker exec -it <container> /bin/sh` — shell into running container.
- `docker logs -f --since 5m <container>` — tail recent logs.
- `docker inspect <container> | jq '.[0].State'` — check exit code, OOM status.
- `docker stats --no-stream` — one-shot resource usage snapshot.
- `docker history <image>` — see layer sizes and commands.
- `docker compose up --build --force-recreate` — clean rebuild without cache.
- `docker run --rm -it --entrypoint sh <image>` — override entrypoint for debugging.

## Anti-Patterns

- `latest` tag in production: unversioned, unpredictable deploys.
- `apt-get install` without `rm -rf /var/lib/apt/lists/*` wastes 30-100MB per layer.
- `COPY . .` before dependency install: every source change re-installs deps.
- Storing data in containers: use volumes or external storage.
- Running database migrations in Dockerfile: do it at deploy time, not build time.
- `docker-compose` (hyphenated) is legacy. Use `docker compose` (plugin form, v2+).
