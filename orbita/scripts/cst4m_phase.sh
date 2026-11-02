#!/usr/bin/env bash
# Uso (desde la raíz): cst4m_phase.sh dev|confirm "v4 m1 ..." — reanudable vía .done (un job por semilla×brazo)
set -uo pipefail
PH=${1:?dev|confirm}; ARMS=${2:?brazos}; JOBS=${JOBS:-7}
case $PH in dev) P=A;; confirm) P=B;; *) exit 1;; esac
OUT=artifacts/constelacion_v4_mejoras/$PH; mkdir -p $OUT
FZ=docs/constelacion_v4_mejoras/frozen.json; B=orbita/target/release/orbita
for a in $ARMS; do for i in 0 1 2 3 4 5 6 7 8 9 A B C D E F; do S=0x${P}40$i; [ -f $OUT/${S}_$a.done ] || echo "$S $a"; done; done |
  xargs -r -P $JOBS -n 2 sh -c 'nice -n 10 '$B' cst4m-run --phase '$PH' --seed "$0" --arm "$1" --weights artifacts/orbita/'$PH'/weights_$0.json --frozen '$FZ' --out artifacts/constelacion_v4_mejoras >> '$OUT'/log.txt 2>&1 || echo "$0 $1 FAIL" >> '$OUT'/fails.txt'
ls $OUT/*.done 2>/dev/null | wc -l
