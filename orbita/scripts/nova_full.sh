#!/usr/bin/env bash
# ÓRBITA-Nova — entrenamiento completo DEV→CONFIRM (PREREG f64b289).
# Un comando: calibra τ (si falta) → 16 seeds DEV → 16 seeds CONFIRM.
# Reanudable (.done). NO lanza si hay otra carga CPU (bytes/v6/…) salvo NOVA_IGNORE_BUSY=1.
#
# Uso:
#   bash orbita/scripts/nova_full.sh              # full (pesado; pedir OK de CPU)
#   SMOKE=1 bash orbita/scripts/nova_full.sh      # pocos episodios, 1 seed
#   PHASE=dev bash orbita/scripts/nova_full.sh    # sólo DEV
#   NOVA_IGNORE_BUSY=1 bash orbita/scripts/nova_full.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
BIN=${BIN:-$ROOT/orbita/target/release/orbita}
OUT=${OUT:-$ROOT/artifacts/orbita/nova_full}
WDEV=${WDEV:-$ROOT/artifacts/orbita/dev}
WCONF=${WCONF:-$ROOT/artifacts/orbita/confirm}
NICE=${NICE:-10}
JOBS=${JOBS:-1}
SMOKE=${SMOKE:-0}
PHASE=${PHASE:-all}   # all|dev|confirm|calib
LOG="$OUT/nova_full.log"

mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1

echo "=== nova_full $(date -Iseconds) sha=$(git rev-parse --short HEAD) SMOKE=$SMOKE PHASE=$PHASE ==="

# V6_IGNORE-style: no compartir CPU con Investigador / otras ramas
if [ "${NOVA_IGNORE_BUSY:-0}" != 1 ]; then
  if pgrep -af 'bytes_enc\.py|bytes_exp\.py|run_v6 |run_predictive_field' >/dev/null; then
    echo "ABORT: otra carga CPU activa (bytes/v6/…). Esperar OK del Investigador o NOVA_IGNORE_BUSY=1." >&2
    exit 1
  fi
fi
if pgrep -af 'nova-eval --full' >/dev/null; then
  echo "nova_full ya corriendo"; exit 0
fi

if [[ ! -x "$BIN" ]]; then
  echo "building orbita…"
  nice -n "$NICE" cargo build --release --manifest-path orbita/Cargo.toml -j "$JOBS"
fi

DEV_SEEDS=(A400 A401 A402 A403 A404 A405 A406 A407 A408 A409 A40A A40B A40C A40D A40E A40F)
CONF_SEEDS=(B400 B401 B402 B403 B404 B405 B406 B407 B408 B409 B40A B40B B40C B40D B40E B40F)

SMOKE_FLAGS=()
if [[ "$SMOKE" == "1" ]]; then
  SMOKE_FLAGS=(--smoke --n-stream "${N_STREAM:-24}" --consolidate-every "${CONSOLIDATE_EVERY:-8}")
  DEV_SEEDS=(A400)
  CONF_SEEDS=(B400)
fi

# —— 1. Calibración τ (una vez; H DEV fam 1–2) ——
if [[ "$PHASE" == "all" || "$PHASE" == "calib" || "$PHASE" == "dev" ]]; then
  if [[ ! -f "$OUT/tau.json" ]]; then
    if [[ "$SMOKE" == "1" && -n "${TAU:-}" ]]; then
      echo "{\"tau\": $TAU, \"note\": \"smoke override\"}" > "$OUT/tau.json"
    elif [[ "$SMOKE" == "1" ]]; then
      echo '{"tau": 1.0, "note": "smoke default (no calib)"}' > "$OUT/tau.json"
    else
      echo "=== calibrate-tau ==="
      nice -n "$NICE" "$BIN" nova-eval --calibrate-tau --wdir "$WDEV" --out "$OUT"
    fi
  else
    echo "tau.json existe — no recalibrar (PREREG §5)"
  fi
fi
TAU=$(python3 -c "import json; print(json.load(open('$OUT/tau.json'))['tau'])")
echo "tau=$TAU"

run_phase() {
  local phase=$1; shift
  local wdir=$1; shift
  local -a seeds=("$@")
  mkdir -p "$OUT/$phase"
  for s in "${seeds[@]}"; do
    local done="$OUT/$phase/0x${s}.done"
    if [[ -f "$done" ]]; then
      echo "skip 0x$s (.done)"; continue
    fi
    local W="$wdir/weights_0x${s}.json"
    [[ -f "$W" ]] || { echo "missing $W" >&2; exit 1; }
    echo "=== $phase seed 0x$s $(date -Iseconds) ==="
    nice -n "$NICE" "$BIN" nova-eval --full --phase "$phase" \
      --weights "$W" --seed "0x$s" --tau "$TAU" --out "$OUT" --resume \
      "${SMOKE_FLAGS[@]+"${SMOKE_FLAGS[@]}"}"
  done
  local n_done
  n_done=$(ls "$OUT/$phase"/*.done 2>/dev/null | wc -l)
  echo "$phase done_count=$n_done/$(echo "${seeds[@]}" | wc -w)"
  if [[ "$SMOKE" != "1" && "$n_done" -ge 16 ]]; then
    touch "$OUT/$phase.done"
  elif [[ "$SMOKE" == "1" && "$n_done" -ge 1 ]]; then
    touch "$OUT/$phase.smoke.done"
  fi
}

if [[ "$PHASE" == "all" || "$PHASE" == "dev" ]]; then
  run_phase dev "$WDEV" "${DEV_SEEDS[@]}"
fi
if [[ "$PHASE" == "all" || "$PHASE" == "confirm" ]]; then
  # CONFIRM sin reajuste: mismo τ, mismo binario
  run_phase confirm "$WCONF" "${CONF_SEEDS[@]}"
fi

if [[ "$PHASE" == "all" && "$SMOKE" != "1" && -f "$OUT/dev.done" && -f "$OUT/confirm.done" ]]; then
  touch "$OUT/ALL.done"
  echo "ALL.done"
fi
echo "=== nova_full finished $(date -Iseconds) ==="
