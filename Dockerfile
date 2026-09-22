# Multi-stage: build Rust → runtime slim. Sin descargar GGUF en el build.
# GGUF opcional en runtime vía volumen / env GEMMA2_GGUF.

FROM rust:1.85-bookworm AS builder
WORKDIR /src
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config libssl-dev cmake clang \
    && rm -rf /var/lib/apt/lists/*

# Cache de dependencias
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY web ./web
# Bins de archive no se necesitan para agentic_web, pero el árbol src completo
# alimenta la lib. Compilar solo el bin web.
RUN cargo build --release --features web --bin agentic_web

FROM debian:bookworm-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libssl3 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /src/target/release/agentic_web /app/agentic_web
COPY --from=builder /src/web/static /app/web/static
ENV PORT=8080
ENV RUST_LOG=info
EXPOSE 8080
# Railway inyecta PORT; el bin lee 0.0.0.0:$PORT
CMD ["/app/agentic_web"]
