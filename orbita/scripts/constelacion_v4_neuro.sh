#!/usr/bin/env bash
# CONSTELACIÓN v4-NEURO (docs/constelacion_v4_neuro/PREREG.md). Uso (desde la raíz): constelacion_v4_neuro.sh p1|p2 dev|confirm [ARMS] [MECH]
# Un proceso por semilla (todos los brazos dentro: en p1 el entrenamiento v4 del flujo A se hace una vez y se reutiliza); reanudable vía .done; JOBS=6, nice 10.
# ÓRBITA-v1 congelada: semilla 0x{A,B}4{6,7}i ↔ artifacts/orbita/{dev,confirm}/weights_0x{A,B}40i.json
set -uo pipefail
ST=${1:?p1|p2}; PH=${2:?dev|confirm}; ARMS=${3:-}; MECH=${4:-}; JOBS=${JOBS:-6}; B=orbita/target/release/orbita
case $PH in dev) P=A;; confirm) P=B;; *) exit 1;; esac; case $ST in p1) D=6;; p2) D=7;; *) exit 1;; esac
R=artifacts/constelacion_v4_neuro/$ST; mkdir -p $R/$PH
EX=""; [ -n "$ARMS" ] && EX="$EX --arms $ARMS"; [ -n "$MECH" ] && EX="$EX --mech $MECH"
for i in 0 1 2 3 4 5 6 7 8 9 A B C D E F; do S=0x${P}4${D}$i; [ -f $R/$PH/$S.done ] && continue; echo "$S artifacts/orbita/$PH/weights_0x${P}40$i.json"; done |
  xargs -r -P $JOBS -n 2 sh -c "nice -n 10 $B cst4n-run --stage $ST --phase $PH --seed \"\$0\" --weights \"\$1\" --out $R$EX >> $R/$PH/log.txt 2>&1"
echo "$ST $PH $(ls $R/$PH/*.done 2>/dev/null | wc -l)/16"
