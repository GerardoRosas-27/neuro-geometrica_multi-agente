#!/usr/bin/env bash
# CONSTELACIÓN v4-ESCALADA (docs/constelacion_v4_escalada/PREREG.md)
# Uso (desde la raíz): setsid nohup orbita/scripts/constelacion_v4_escalada.sh dev|confirm [ARMS] &
#   ARMS por defecto: "PAT A T P v4" (más largo primero). En CONFIRM, ARMS = v4 + variantes que pasaron el gate DEV (PREREG §Gate).
# Un proceso por (semilla, brazo); reanudable vía .done; JOBS=7, nice 10.
# Pesos ÓRBITA-v1 congelados: semilla nueva 0x{A,B}45i ↔ artifacts/orbita/{dev,confirm}/weights_0x{A,B}40i.json (PREREG §Datos).
set -uo pipefail
PH=${1:?dev|confirm}; ARMS=${2:-"PAT A T P v4"}; JOBS=${JOBS:-7}; B=orbita/target/release/orbita
case $PH in dev) P=A;; confirm) P=B;; *) exit 1;; esac
R=artifacts/constelacion_v4_escalada
for arm in $ARMS; do mkdir -p $R/$arm/$PH; for i in 0 1 2 3 4 5 6 7 8 9 A B C D E F; do S=0x${P}45$i
  [ -f $R/$arm/$PH/$S.done ] && continue
  echo "$S $arm artifacts/orbita/$PH/weights_0x${P}40$i.json"
done; done | xargs -r -P $JOBS -n 3 sh -c 'nice -n 10 '$B' cst4esc-run --phase '$PH' --seed "$0" --arms "$1" --weights "$2" --out '$R'/"$1" >> '$R'/"$1"/'$PH'/log.txt 2>&1'
for arm in $ARMS; do echo "$arm $(ls $R/$arm/$PH/*.done 2>/dev/null | wc -l)/16"; done
