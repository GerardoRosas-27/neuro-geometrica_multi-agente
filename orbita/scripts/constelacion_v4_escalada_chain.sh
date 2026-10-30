#!/bin/bash
# Cadena DEV → gate → CONFIRM (sólo brazos del gate) → análisis. NO se lanza en la fase de plan.
cd "$(dirname "$0")/../.." || exit 1
R=artifacts/constelacion_v4_escalada
for t in 1; do orbita/scripts/constelacion_v4_escalada.sh dev; done
python3 scripts/analyze_constelacion_v4_escalada.py dev > $R/analysis_dev.json
ARMS=$(python3 scripts/analyze_constelacion_v4_escalada.py dev --gate); echo "gate: $ARMS" > $R/gate.txt
git add $R && git commit -qm "constelacion-v4-escalada: DEV + gate ($ARMS)" && git push -q
[ "$ARMS" = "v4" ] && { echo "gate: ninguna variante pasa; sin CONFIRM"; exit 0; }
for t in 1; do orbita/scripts/constelacion_v4_escalada.sh confirm "$ARMS"; done
python3 scripts/analyze_constelacion_v4_escalada.py dev confirm > $R/analysis_dev_confirm.json
git add $R && git commit -qm "constelacion-v4-escalada: CONFIRM + análisis" && git push -q
echo CHAIN_DONE
