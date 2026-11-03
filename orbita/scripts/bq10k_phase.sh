#!/usr/bin/env bash
# BQ10 aislado de v4 + barrido k (docs/bq10_aislado_k/PREREG.md). Uso: bq10k_phase.sh bq_tune|bq_dev|bq_confirm ARMS KS [LISTA]
set -uo pipefail
PH=${1:?fase}; ARMS=${2:?arms}; KS=${3:?ks}; JOBS=${JOBS:-6}
R=artifacts/bq10_aislado_k; mkdir -p $R/$PH
export BQ_B=$PWD/orbita/target/release/orbita BQ_R=$R BQ_PH=$PH BQ_ARMS=$ARMS BQ_KS=$KS
case $PH in bq_tune) L=${4:-0}; S0=0xC4B; WD=dev; WP=A;; bq_dev) L="0 1 2 3 4 5 6 7 8 9 A B C D E F"; S0=0xA40; WD=dev; WP=A;; bq_confirm) L="0 1 2 3 4 5 6 7 8 9 A B C D E F"; S0=0xB40; WD=confirm; WP=B;; *) exit 1;; esac
for i in $L; do S=$S0$i; [ -f $R/$PH/$S.done ] && continue; echo "$S artifacts/orbita/$WD/weights_0x${WP}40$i.json"; done |
  xargs -r -P "$JOBS" -n 2 bash -c 'S=$1; W=$2; L="$BQ_R/$BQ_PH/log_$S.txt"; echo "START $S $(date -Is)" >> "$L"; nice -n 10 "$BQ_B" atlas-run --phase "$BQ_PH" --seed "$S" --weights "$W" --out "$BQ_R" --arms "$BQ_ARMS" --ks "$BQ_KS" >> "$L" 2>&1; rc=$?; echo "$rc" > "$BQ_R/$BQ_PH/rc_$S.txt"; echo "END $S rc=$rc $(date -Is)" >> "$L"' _
echo "$PH $(ls $R/$PH/*.done 2>/dev/null | wc -l)"
