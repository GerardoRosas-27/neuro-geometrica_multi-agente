#!/bin/sh
# Test de humo de la demo (sin Gemma generativa): comprueba lectura, re-identificación y campos de salida.
set -e
R=${R:-$(cd "$(dirname "$0")/../.." && pwd)}; PY=${PY:-/workspace/orbita-venv/bin/python}; export PYTHONPATH=$R/orbita
$PY $R/orbita/llm/demo.py --text "Aplica una rotación con parámetros 0,5 y -0,2." --obs 1 --simulate 1,0.5,-0.2 --no-explain --json | $PY -c "
import sys, json; d = json.load(sys.stdin)
assert d['familia_leida'] == 'rotación' and d['p_leido'] == [0.5, -0.2] and d['p_fuente'] == 'numnorm'
assert len(d['trayectoria_h1_h16']) == 16 and len(d['re_observacion']['trayectoria_h1_h16']) == 16
assert 0 <= d['yo']['confianza_acierto_h16'] <= 1 and d['re_observacion']['k'] == 1
print('smoke demo OK')"
