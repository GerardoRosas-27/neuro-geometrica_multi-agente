#!/usr/bin/env bash
# Uso (desde la raíz): cst_campo_solo_phase.sh dev|confirm — un job por semilla (brazos cst,field_only,no_orb), reanudable vía .done
set -uo pipefail
PH=${1:?dev|confirm}; JOBS=${JOBS:-7}; N=${N:-16}
case $PH in dev) P=A;; confirm) P=B;; *) exit 1;; esac
OUT=artifacts/cst_campo_solo; B=orbita/target/release/orbita; [ -x $B ] || { echo "falta binario $B"; exit 1; }
mkdir -p $OUT/$PH
for i in $(printf '%X\n' $(seq 0 $((N-1)))); do S=0x${P}40$i; [ -f $OUT/$PH/$S.done ] || echo $S; done |
  xargs -r -P $JOBS -n 1 sh -c 'nice -n 10 '$B' cst4-run --phase '$PH' --seed "$0" --arms cst,field_only,no_orb --weights artifacts/orbita/'$PH'/weights_$0.json --out '$OUT' >> '$OUT/$PH'/log.txt 2>&1 || echo "$0 FAIL" >> '$OUT/$PH'/fails.txt'
ls $OUT/$PH/*.done 2>/dev/null | wc -l
