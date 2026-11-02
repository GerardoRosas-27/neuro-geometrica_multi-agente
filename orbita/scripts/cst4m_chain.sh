#!/bin/bash
# Cadena CONSTELACIÓN v4 mejoras: DEV → compuertas → (ALL en DEV) → CONFIRM → análisis → RESULTADOS.md. Reanudable.
cd /workspace/ngm-v4-mejoras
P=orbita/scripts/cst4m_phase.sh; A=artifacts/constelacion_v4_mejoras
M2=$(python3 -c "import json;print('m2' if json.load(open('docs/constelacion_v4_mejoras/frozen.json'))['m2']['accepted'] else '')")
DEVARMS="v4 m1 m1_ctrl $M2 m3 m3_rf m3_orb"; NDEV=$(( $(echo $DEVARMS | wc -w) * 16 ))
cnt(){ ls $A/$1/*.done 2>/dev/null | wc -l; }
for t in 1 2 3; do [ $(cnt dev) -lt $NDEV ] && $P dev "$DEVARMS"; done
git add $A/dev && git commit -qm "constelacion-v4-mejoras: DEV $(cnt dev)/$NDEV" && git push -q
python3 scripts/analyze_constelacion_v4_mejoras.py dev > $A/gates_dev.json
read PASS GO <<<$(python3 -c "
import json;c=json.load(open('$A/gates_dev.json')).get('cmp',{})
p=[a for a in ['m1','m2','m3'] if a in c and c[a]['pass']];g=[a for a in ['m1','m2','m3'] if a in c and not c[a]['stop']]
print(','.join(p) or '-', ','.join(g) or '-')")
echo "gates: pass=$PASS go=$GO"
CARMS="v4"; for a in ${GO//,/ }; do [ "$a" != "-" ] && CARMS="$CARMS $a"; done
[[ " $CARMS " == *" m1 "* ]] && CARMS="$CARMS m1_ctrl"
if [ "$PASS" != "-" ]; then
  for t in 1 2 3; do $P dev "all" >/dev/null; done
  git add $A/dev && git commit -qm "constelacion-v4-mejoras: DEV brazo ALL (exploratorio)" && git push -q
  python3 scripts/analyze_constelacion_v4_mejoras.py dev > $A/gates_dev.json
  python3 -c "import json;c=json.load(open('$A/gates_dev.json'))['cmp'];exit(0 if 'all' in c and not c['all']['stop'] else 1)" && CARMS="$CARMS all"
fi
git add $A/gates_dev.json && git commit -qm "constelacion-v4-mejoras: compuertas DEV (pass=$PASS, a CONFIRM: $CARMS)" && git push -q
NC=$(( $(echo $CARMS | wc -w) * 16 ))
for t in 1 2 3; do [ $(cnt confirm) -lt $NC ] && $P confirm "$CARMS"; done
git add $A/confirm && git commit -qm "constelacion-v4-mejoras: CONFIRM $(cnt confirm)/$NC" && git push -q
python3 scripts/analyze_constelacion_v4_mejoras.py final > $A/verdicts.json
git add $A docs/constelacion_v4_mejoras/RESULTADOS.md && git commit -qm "constelacion-v4-mejoras: análisis DEV+CONFIRM y RESULTADOS.md" && git push -q
echo CHAIN_DONE
