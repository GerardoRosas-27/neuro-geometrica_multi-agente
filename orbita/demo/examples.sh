#!/bin/sh
# Ejemplos reales de la demo; guarda salidas en artifacts/orbita/demo/
set -e
R=/workspace/ngm-orbita; PY=/workspace/orbita-venv/bin/python; export PYTHONPATH=$R/orbita; D=$R/artifacts/orbita/demo; mkdir -p $D
run() { n=$1; shift; $PY $R/orbita/llm/demo.py "$@" > $D/$n.json; echo "$n ok"; }
run 01_es_rotacion_obs1   --text "Aplica una rotación con parámetros 0,5 y -0,2." --obs 1 --simulate 1,0.5,-0.2
run 02_en_reflexion_obs0  --text "Transform the shape with a reflection: the first value is minus three tenths and the second is 0.4."
run 03_fr_spirale_obs2    --text "Transforme la figure avec une spirale : la première valeur est 0,6 et la seconde est -0,5." --obs 2 --simulate 9,0.6,-0.5
run 04_pt_rotacao_obs1    --text "Aplique uma rotação com parâmetros 0,3 e 0,7." --obs 1 --simulate 1,0.3,0.7
run 05_it_scalatura_obs2  --text "Applica una scalatura con parametri -0,4 e 0,2." --obs 2 --simulate 3,-0.4,0.2
