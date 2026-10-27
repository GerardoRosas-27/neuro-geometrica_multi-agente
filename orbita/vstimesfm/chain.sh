#!/usr/bin/env bash
# chain.sh PHASE — TimesFM + VAR2_OLS por seed sobre latentes dinlat cacheados (2 en paralelo × 4 hilos) → analyze.
# CONFIRM: si faltan embeddings/res de dinlat para la fase, primero corre dinlat/chain.sh PHASE (Gemma + ÓRBITA).
set -eu
PH=$1; H=$(cd "$(dirname "$0")" && pwd); DD=/workspace/orbita-data/dinlat/$PH; O=/workspace/orbita-data/timesfm/$PH
P=/workspace/timesfm-venv/bin/python; PO=/workspace/orbita-venv/bin/python
mkdir -p $O; date -Is > $O/start
[ -f $DD/done ] || bash $H/../dinlat/chain.sh $PH
sed "s#\$# $O/tfm_SEED.json#" $DD/jobs | awk '{s=$1; sub("SEED",s,$6); print $1,$2,$3,$4,$5,$6}' > $O/jobs
xargs -P 2 -L 1 sh -c 'TFM_THREADS=4 OMP_NUM_THREADS=4 '$P' '$H'/run_tfm.py "$0" "$@" > '$O'/log_$0.txt 2>&1 || echo FALLO $0' < $O/jobs
$PO $H/analyze.py $PH $DD $O > $O/analysis.txt
date -Is > $O/done
