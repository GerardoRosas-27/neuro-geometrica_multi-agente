#!/bin/sh
# Test de humo del pipeline end-to-end (docs/preregistro_integracion_e2e.md): parser + un texto por la cadena completa.
set -e
cd "$(dirname "$0")"; PY=${PY:-/workspace/orbita-venv/bin/python}; export PYTHONPATH=$(dirname "$PWD")
$PY test_numnorm.py > /tmp/e2e_numnorm.log || { cat /tmp/e2e_numnorm.log; echo "FALLO parser"; exit 1; }
OUT=$($PY -W ignore e2e.py run --phase dev --seed 0xA400 --idx 1 --text "Transforma la figura con una rotación: el primer valor es cero coma sesenta y nueve y el segundo es 21,95 grados.")
echo "$OUT"
echo "$OUT" | $PY -c "
import json,sys; r=json.loads(sys.stdin.read())
assert r['parse_ok'] and abs(r['p_hat'][0]-0.69)<1e-9 and abs(r['p_hat'][1]-0.383099)<1e-4, r['p_hat']
assert sum(r['c_hat'][:8])==1.0 and len(r['z16'])>0 and 0<=r['cdt_fam_hat']<8 and r['cdt_residual16']==r['cdt_residual16']
assert r['queries'].endswith('queries=0'), r['queries']
print('SMOKE OK')"
