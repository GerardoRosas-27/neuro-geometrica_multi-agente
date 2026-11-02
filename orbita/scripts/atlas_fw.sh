#!/usr/bin/env bash
# ATLAS-FW (docs/atlas_fw/PREREG.md). Uso (desde la raíz): atlas_fw.sh dev|confirm [ARMS]
# Un proceso por semilla; v4 se entrena una vez y todos los brazos lo reutilizan. JOBS=7, nice 10.
# Pesos ÓRBITA-v1: semilla 0x{A,B}48i ↔ artifacts/orbita/{dev,confirm}/weights_0x{A,B}40i.json
# Cada semilla escribe rc_0xS.txt (código de salida) y log_0xS.txt. Sin interpolación en sh -c.
set -uo pipefail
PH=${1:?dev|confirm}; ARMS=${2:-v4,M1,M2,M3,M4,M5,M6,ALL}; JOBS=${JOBS:-7}
case $PH in dev) P=A;; confirm) P=B;; *) exit 1;; esac
R=artifacts/atlas_fw; mkdir -p $R/$PH
export ATLAS_B=$PWD/orbita/target/release/orbita ATLAS_R=$R ATLAS_PH=$PH ATLAS_ARMS=$ARMS
for i in 0 1 2 3 4 5 6 7 8 9 A B C D E F; do S=0x${P}48$i; [ -f $R/$PH/$S.done ] && continue; echo "$S artifacts/orbita/$PH/weights_0x${P}40$i.json"; done |
  xargs -r -P "$JOBS" -n 2 bash -c 'S=$1; W=$2; L="$ATLAS_R/$ATLAS_PH/log_$S.txt"; echo "START $S $(date -Is)" > "$L"; nice -n 10 "$ATLAS_B" atlas-run --phase "$ATLAS_PH" --seed "$S" --weights "$W" --out "$ATLAS_R" --arms "$ATLAS_ARMS" >> "$L" 2>&1; rc=$?; echo "$rc" > "$ATLAS_R/$ATLAS_PH/rc_$S.txt"; echo "END $S rc=$rc $(date -Is)" >> "$L"' _
echo "$PH $(ls $R/$PH/*.done 2>/dev/null | wc -l)/16"
