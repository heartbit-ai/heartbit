FROM rust:1.85-bookworm AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y cmake libssl-dev pkg-config && rm -rf /var/lib/apt/lists/*
COPY . .
RUN cargo build --release -p heartbit-cli --features daemon,postgres

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libssl3 && rm -rf /var/lib/apt/lists/*
RUN groupadd -r heartbit && useradd -r -g heartbit -s /sbin/nologin heartbit
COPY --from=builder /app/target/release/heartbit /usr/local/bin/heartbit
USER heartbit
ENV RUST_LOG=info
EXPOSE 8081
CMD ["heartbit", "daemon", "--config", "/etc/heartbit/config.toml"]
