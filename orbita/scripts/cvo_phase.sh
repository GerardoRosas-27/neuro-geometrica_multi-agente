#!/usr/bin/env bash
# Uso (desde la raíz): cvo_phase.sh dev|confirm — reanudable vía .done
set -uo pipefail
PH=${1:?dev|confirm}; JOBS=${JOBS:-7}
case $PH in dev) P=A;; confirm) P=B;; *) exit 1;; esac
OUT=artifacts/cst_vs_orbita/$PH; mkdir -p $OUT
for i in 0 1 2 3 4 5 6 7 8 9 A B C D E F; do S=0x${P}40$i; [ -f $OUT/$S.done ] || echo $S; done |
  xargs -r -P $JOBS -I{} sh -c 'nice -n 10 orbita/scripts/cvo_seed.sh '$PH' {} >> '$OUT'/log_{}.txt 2>&1 || echo "{} FAIL" >> '$OUT'/fails.txt'
ls $OUT/*.done 2>/dev/null | wc -l
