#!/usr/bin/env bash
# Setup en Kaggle (internet ON, requiere teléfono verificado). Instala Rust y compila orbita (CPU).
set -euo pipefail
REPO=${REPO:-https://github.com/GerardoRosas-27/neuro-geometrica_multi-agente.git}
REF=${REF:-main}
W=${W:-/kaggle/working}; SRC=${SRC:-/kaggle/tmp/ngm}   # código en scratch; sólo resultados en /kaggle/working
mkdir -p "$(dirname "$SRC")" "$W/ckpt"
if ! command -v cargo >/dev/null; then
  curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
fi
source "$HOME/.cargo/env"
[ -d "$SRC/.git" ] || git clone --filter=blob:none "$REPO" "$SRC"
cd "$SRC" && git fetch -q origin "$REF" && git checkout -q FETCH_HEAD
git rev-parse HEAD | tee "$W/ckpt/git_sha.txt"
# caché del binario por sha (evita recompilar al reanudar si se adjunta la salida previa como input)
SHA=$(git rev-parse --short HEAD); BIN="$W/ckpt/orbita_$SHA"
if [ -x "$BIN" ]; then mkdir -p orbita/target/release && cp "$BIN" orbita/target/release/orbita
else (cd orbita && RUSTFLAGS="-C target-cpu=native" cargo build --release -q) && cp orbita/target/release/orbita "$BIN"; fi
nproc; free -g | head -2
