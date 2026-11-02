#!/usr/bin/env bash
# Ablaciones ATLAS-FW (docs/atlas_fw/PREREG_ABLACIONES.md). Uso (desde la raíz): atlas_fw_abl.sh abl_dev|abl_confirm ARMS
# Semillas 0x{A,B}49i ↔ pesos ÓRBITA-v1 artifacts/orbita/{dev,confirm}/weights_0x{A,B}40i.json. JOBS=7, nice 10.
set -uo pipefail
PH=${1:?abl_dev|abl_confirm}; ARMS=${2:?ARMS}; JOBS=${JOBS:-7}
case $PH in abl_dev) P=A; W=dev;; abl_confirm) P=B; W=confirm;; *) exit 1;; esac
R=artifacts/atlas_fw/ablaciones; mkdir -p $R/$PH
export ATLAS_B=$PWD/orbita/target/release/orbita ATLAS_R=$R ATLAS_PH=$PH ATLAS_ARMS=$ARMS
for i in 0 1 2 3 4 5 6 7 8 9 A B C D E F; do S=0x${P}49$i; [ -f $R/$PH/$S.done ] && continue; echo "$S artifacts/orbita/$W/weights_0x${P}40$i.json"; done |
  xargs -r -P "$JOBS" -n 2 bash -c 'S=$1; W=$2; L="$ATLAS_R/$ATLAS_PH/log_$S.txt"; echo "START $S $(date -Is)" > "$L"; nice -n 10 "$ATLAS_B" atlas-run --phase "$ATLAS_PH" --seed "$S" --weights "$W" --out "$ATLAS_R" --arms "$ATLAS_ARMS" >> "$L" 2>&1; rc=$?; echo "$rc" > "$ATLAS_R/$ATLAS_PH/rc_$S.txt"; echo "END $S rc=$rc $(date -Is)" >> "$L"' _
echo "$PH $(ls $R/$PH/*.done 2>/dev/null | wc -l)/16"
