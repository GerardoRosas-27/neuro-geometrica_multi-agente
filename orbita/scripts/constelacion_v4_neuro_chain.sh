#!/bin/bash
# Cadena única: P1 DEV → gate → P1 CONFIRM (v4 + brazos del gate) → veredicto P1 → [si algún mecanismo pasa] P2 DEV → gate → P2 CONFIRM → veredicto P2.
# Reanudable (los .done se saltan). Hace commit+push normal tras cada etapa. Escribe CHAIN_DONE / CHAIN_STOP al final.
cd "$(dirname "$0")/../.." || exit 1
A=artifacts/constelacion_v4_neuro; S=orbita/scripts/constelacion_v4_neuro.sh; AN=scripts/analyze_constelacion_v4_neuro.py
push() { git add $A && git commit -qm "$1" && git push -q -u origin exp/constelacion-v4-neuro; }
echo "[$(date)] P1 DEV"; $S p1 dev
python3 $AN p1 dev > $A/p1/analysis_dev.json || { echo "análisis P1 DEV falló; alto"; exit 1; }
G=$(python3 $AN p1 dev --gate) || exit 1; echo "gate: $G" > $A/p1/gate.txt; push "constelacion-v4-neuro: P1 DEV + gate ($G)"
[ "$G" = "v4" ] && { echo "gate P1: ningún brazo pasa; sin CONFIRM ni fase 2"; echo CHAIN_STOP; exit 0; }
echo "[$(date)] P1 CONFIRM $G"; $S p1 confirm "$(echo $G | tr ' ' ,)"
python3 $AN p1 dev confirm > $A/p1/analysis_dev_confirm.json || { echo "análisis P1 CONFIRM falló"; exit 1; }
MECH=$(python3 $AN p1 dev confirm --mech); echo "fase2_mech: ${MECH:-<ninguno>}" > $A/p1/fase2_mech.txt
push "constelacion-v4-neuro: P1 CONFIRM + veredicto (fase2_mech=${MECH:-ninguno})"
[ -z "$MECH" ] && { echo "P1: nada pasa tras CONFIRM; sin fase 2"; echo CHAIN_STOP; exit 0; }
echo "[$(date)] P2 DEV mech=$MECH"; $S p2 dev "" "$MECH"
python3 $AN p2 dev > $A/p2/analysis_dev.json || exit 1
G2=$(python3 $AN p2 dev --gate) || exit 1; echo "gate: $G2" > $A/p2/gate.txt; push "constelacion-v4-neuro: P2 DEV + gate ($G2)"
[ "$G2" = "v4" ] && { echo "gate P2: NEU no pasa"; echo CHAIN_STOP; exit 0; }
echo "[$(date)] P2 CONFIRM"; $S p2 confirm "" "$MECH"
python3 $AN p2 dev confirm > $A/p2/analysis_dev_confirm.json || exit 1
push "constelacion-v4-neuro: P2 CONFIRM + veredicto"
echo CHAIN_DONE
