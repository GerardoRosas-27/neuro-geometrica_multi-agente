#!/usr/bin/env bash
# ESTELA v0 (docs/estela/PREREG.md). Uso: PHASE=smoke|dev|confirm JOBS=7 bash orbita/scripts/estela.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
BIN=$ROOT/orbita/target/release/orbita
OUT=$ROOT/artifacts/estela
PHASE=${PHASE:-dev}; JOBS=${JOBS:-7}
mkdir -p "$OUT/$PHASE"
LOG=$OUT/$PHASE.log
echo "=== estela $PHASE $(date -Iseconds) sha=$(git rev-parse --short HEAD) ===" >> "$LOG"
case $PHASE in
  smoke) W=artifacts/orbita/dev; SEEDS="A400 A401"; EXTRA="--h-per-fam 4";;
  dev) W=artifacts/orbita/dev; SEEDS="A400 A401 A402 A403 A404 A405 A406 A407 A408 A409 A40A A40B A40C A40D A40E A40F"; EXTRA="";;
  confirm) W=artifacts/orbita/confirm; SEEDS="B400 B401 B402 B403 B404 B405 B406 B407 B408 B409 B40A B40B B40C B40D B40E B40F"; EXTRA="";;
  *) echo "PHASE inválida"; exit 2;;
esac
export BIN W OUT PHASE EXTRA LOG
echo $SEEDS | tr ' ' '\n' | xargs -P "$JOBS" -I{} bash -c \
 'nice -n 10 "$BIN" estela-run --phase "$PHASE" --weights "$W/weights_0x{}.json" --seed 0x{} --out "$OUT" $EXTRA >>"$LOG" 2>&1 && echo "done 0x{} $(date -Iseconds)" >>"$LOG"'
echo "$PHASE done_count=$(ls $OUT/$PHASE/*.done 2>/dev/null | wc -l) $(date -Iseconds)" >> "$LOG"
