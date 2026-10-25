#!/usr/bin/env bash
# Uso (desde la raíz): setsid nohup orbita/scripts/eco.sh dev|confirm &  — reanudable vía .done
set -euo pipefail
PH=${1:?dev|confirm}; JOBS=${JOBS:-7}; B=orbita/target/release/orbita
case $PH in dev) P=A;; confirm) P=B;; *) exit 1;; esac
OUT=artifacts/eco/$PH; mkdir -p $OUT
for i in 0 1 2 3 4 5 6 7 8 9 A B C D E F; do S=0x${P}40$i
  [ -f $OUT/$S.done ] && continue
  echo "$S artifacts/orbita/$PH/weights_$S.json"
done | xargs -r -P $JOBS -n 2 sh -c 'nice -n 10 '$B' eco-run --phase '$PH' --weights "$1" --seed "$0" --out '$OUT' >> '$OUT'/log.txt 2>&1'
ls $OUT/*.done | wc -l
