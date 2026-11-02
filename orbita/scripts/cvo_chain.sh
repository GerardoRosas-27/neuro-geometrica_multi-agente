#!/usr/bin/env bash
# Cadena DEV → (regla de paro) → CONFIRM → análisis. Uso: cd /workspace/ngm-mejor && setsid nohup orbita/scripts/cvo_chain.sh > /workspace/cvo-data/chain.log 2>&1 &
set -uo pipefail
cd /workspace/ngm-mejor; A=artifacts/cst_vs_orbita
n() { ls $A/$1/*.done 2>/dev/null | wc -l; }
for t in 1 2 3; do [ $(n dev) -lt 16 ] && orbita/scripts/cvo_phase.sh dev; done
git add $A/dev && git commit -qm "cst-vs-orbita: DEV $(n dev)/16" && git push -q origin HEAD
python3 scripts/analyze_cst_vs_orbita.py --stopcheck dev > $A/stopcheck_dev.json; rc=$?
if [ $rc -ne 0 ] || [ $(n dev) -lt 16 ]; then git add $A && git commit -qm "cst-vs-orbita: PARO tras DEV" && git push -q origin HEAD; echo CHAIN_STOPPED; exit 1; fi
for t in 1 2 3; do [ $(n confirm) -lt 16 ] && orbita/scripts/cvo_phase.sh confirm; done
python3 scripts/analyze_cst_vs_orbita.py > $A/analysis.json; python3 scripts/analyze_cst_vs_orbita.py --md > $A/RESULTADOS.md
git add $A && git commit -qm "cst-vs-orbita: CONFIRM $(n confirm)/16 + análisis" && git push -q origin HEAD
echo CHAIN_DONE
