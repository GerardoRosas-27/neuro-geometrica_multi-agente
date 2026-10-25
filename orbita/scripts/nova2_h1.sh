#!/usr/bin/env bash
# Nova-2 H1 NOVA_RET (docs/orbita_nova2/PREREG_H1.md). Uso: PHASE=smoke|dev|confirm JOBS=7 bash orbita/scripts/nova2_h1.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
BIN=$ROOT/orbita/target/release/orbita
OUT=$ROOT/artifacts/orbita/nova2_h1
TAU=0.9386641928884383
PHASE=${PHASE:-dev}; JOBS=${JOBS:-7}
mkdir -p "$OUT/$PHASE"
LOG=$OUT/$PHASE.log
echo "=== nova2_h1 $PHASE $(date -Iseconds) sha=$(git rev-parse --short HEAD) ===" >> "$LOG"
case $PHASE in
  smoke) W=artifacts/orbita/dev; SEEDS="A400 A401"; EXTRA="--n-stream 60 --h-per-fam 4 --consolidate-every 20";;
  dev) W=artifacts/orbita/dev; SEEDS="A400 A401 A402 A403 A404 A405 A406 A407 A408 A409 A40A A40B A40C A40D A40E A40F"; EXTRA="";;
  confirm) W=artifacts/orbita/confirm; SEEDS="B400 B401 B402 B403 B404 B405 B406 B407 B408 B409 B40A B40B B40C B40D B40E B40F"; EXTRA="";;
esac
P=$PHASE; [[ $PHASE == smoke ]] && P=smoke
export BIN W TAU OUT P EXTRA LOG
echo $SEEDS | tr ' ' '\n' | xargs -P "$JOBS" -I{} bash -c \
 'nice -n 10 "$BIN" nova2-h1 --phase "$P" --weights "$W/weights_0x{}.json" --seed 0x{} --tau "$TAU" --out "$OUT" $EXTRA >>"$LOG" 2>&1 && echo "done 0x{} $(date -Iseconds)" >>"$LOG"'
echo "$PHASE done_count=$(ls $OUT/$PHASE/*.done | wc -l) $(date -Iseconds)" >> "$LOG"
