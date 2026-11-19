#!/usr/bin/env bash
# METAFE-3S — Control Metacognitivo de Tres Escalas sobre Energía Libre Variacional (docs/metafe3s/PREREG.md).
# Uso: metafe3s_phase.sh tune|dev|confirm [ARMS]
# Emparejamiento exacto de pesos ÓRBITA:
#   tune    (0xC4B0..3) ↔ artifacts/orbita/dev/weights_0xA40{0..3}.json
#   dev     (0xA400..F) ↔ artifacts/orbita/dev/weights_0xA40{0..F}.json
#   confirm (0xB400..F) ↔ artifacts/orbita/confirm/weights_0xB40{0..F}.json
set -uo pipefail
PH=${1:?fase (tune|dev|confirm)}
ARMS=${2:-""}
JOBS=${JOBS:-2}
R=artifacts/metafe3s
mkdir -p "$R/$PH"
export MF3_B="$PWD/orbita/target/release/orbita"
export MF3_R="$R"
export MF3_PH="$PH"
export MF3_ARMS="$ARMS"

case "$PH" in
  tune)    L="0 1 2 3"; S0=0xC4B; WD=dev; WP=A ;;
  dev)     L="0 1 2 3 4 5 6 7 8 9 A B C D E F"; S0=0xA40; WD=dev; WP=A ;;
  confirm) L="0 1 2 3 4 5 6 7 8 9 A B C D E F"; S0=0xB40; WD=confirm; WP=B ;;
  *) echo "fase invalida: $PH" >&2; exit 1 ;;
esac

for i in $L; do
  S="$S0$i"
  [ -f "$R/$PH/$S.done" ] && continue
  echo "$S artifacts/orbita/$WD/weights_0x${WP}40$i.json"
done | xargs -r -P "$JOBS" -n 2 bash -c '
  S=$1; W=$2
  L="$MF3_R/$MF3_PH/log_$S.txt"
  echo "START $S $(date -Is)" > "$L"
  if [ -n "$MF3_ARMS" ]; then
    nice -n 10 "$MF3_B" metafe3s-run --phase "$MF3_PH" --seed "$S" --weights "$W" --out "$MF3_R" --arms "$MF3_ARMS" >> "$L" 2>&1
  else
    nice -n 10 "$MF3_B" metafe3s-run --phase "$MF3_PH" --seed "$S" --weights "$W" --out "$MF3_R" >> "$L" 2>&1
  fi
  rc=$?
  echo "$rc" > "$MF3_R/$MF3_PH/rc_$S.txt"
  echo "END $S rc=$rc $(date -Is)" >> "$L"
' _

echo "$PH $(ls "$R/$PH"/*.done 2>/dev/null | wc -l)"
