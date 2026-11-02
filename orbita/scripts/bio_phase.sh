#!/usr/bin/env bash
# bio_phase.sh dev|confirm "v4 hyb hyb_nobio ref" — reanudable vía .done (un job por semilla×brazo)
set -uo pipefail
cd /workspace/ngm-bio
PH=${1:?}; ARMS=${2:?}; JOBS=${JOBS:-3}
case $PH in dev) P=A;; confirm) P=B;; *) exit 1;; esac
OUT=artifacts/bio_constelacion/$PH; mkdir -p $OUT; FZ=docs/bio_constelacion/frozen.json; B=orbita/target/release/orbita
[ -x $B ] || { echo "FALTA BINARIO $B" >&2; exit 2; }
for a in $ARMS; do for i in 0 1 2 3 4 5 6 7 8 9 A B C D E F; do S=0x${P}40$i; [ -f $OUT/${S}_$a.done ] || echo "$S $a"; done; done |
  xargs -r -P $JOBS -n 2 sh -c 'if [ "$1" = v4 ]; then C=cst4m-run; else C=bio-run; fi; nice -n 10 '$B' $C --phase '$PH' --seed "$0" --arm "$1" --weights artifacts/orbita/'$PH'/weights_$0.json --frozen '$FZ' --out artifacts/bio_constelacion >> '$OUT'/log.txt 2>&1 || echo "$0 $1 FAIL" >> '$OUT'/fails.txt'
ls $OUT/*.done 2>/dev/null | wc -l
