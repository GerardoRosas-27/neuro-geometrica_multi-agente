# Multi-stage: build Rust → runtime slim + GGUF ligero (Gemma 2 2B-it Q3_K_L, ~1.55 GB).
# El GGUF se descarga en el build (DOWNLOAD_GGUF=1, def) desde Hugging Face
# público con sha256 verificado. Si falta en runtime (DOWNLOAD_GGUF=0 o volumen
# vacío), el binario lo descarga al arrancar en segundo plano (GEMMA2_GGUF_URL).

# sysinfo 0.39 / zip 8.x requieren rustc >= 1.95 / 1.88 (no usar 1.85).
FROM rust:bookworm AS builder
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
    ca-certificates libssl3 curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# Modelo ligero (capa aparte, antes del binario, para cachearla entre deploys).
ARG DOWNLOAD_GGUF=1
ARG GEMMA2_GGUF_URL=https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/855f67caed130e1befc571b52bd181be2e858883/gemma-2-2b-it-Q3_K_L.gguf
ARG GEMMA2_GGUF_SHA256=d14b920ed8025a03e0764bdaeca6fe553dfb559569c79b5b83bb5b28fc364984
RUN mkdir -p /app/models && if [ "$DOWNLOAD_GGUF" = "1" ]; then \
      curl -fL --retry 5 --retry-delay 3 -sS -o /app/models/gemma-2-2b-it-Q3_K_L.gguf "$GEMMA2_GGUF_URL" \
      && if [ -n "$GEMMA2_GGUF_SHA256" ]; then \
           echo "$GEMMA2_GGUF_SHA256  /app/models/gemma-2-2b-it-Q3_K_L.gguf" | sha256sum -c -; fi; \
    fi

COPY --from=builder /src/target/release/agentic_web /app/agentic_web
COPY --from=builder /src/web/static /app/web/static
ENV PORT=8080
ENV RUST_LOG=info
# Menos arenas de glibc → RSS estable ~1.8 GB en vez de ~2.3 GB tras generar.
ENV MALLOC_ARENA_MAX=2
ENV GEMMA2_GGUF=/app/models/gemma-2-2b-it-Q3_K_L.gguf
ENV GEMMA2_GGUF_URL=${GEMMA2_GGUF_URL}
ENV GEMMA2_GGUF_SHA256=${GEMMA2_GGUF_SHA256}
EXPOSE 8080
# Railway inyecta PORT; el bin lee 0.0.0.0:$PORT
CMD ["/app/agentic_web"]
