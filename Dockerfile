FROM rust:slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y cmake libssl-dev pkg-config && rm -rf /var/lib/apt/lists/*
COPY . .
RUN cargo build --release -p heartbit-cli

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
RUN groupadd -r heartbit && useradd -r -g heartbit -s /sbin/nologin heartbit
COPY --from=builder /app/target/release/heartbit /usr/local/bin/heartbit
USER heartbit
CMD ["heartbit", "serve"]
