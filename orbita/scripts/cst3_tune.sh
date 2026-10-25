#!/usr/bin/env bash
# uso: cst3_tune.sh NAME SEED "flags"  (sólo semillas de ajuste 0xC4xx/0xD4xx; smoke)
N=$1; S=$2; F=$3; O=artifacts/constelacion_v3/tune/$N; mkdir -p $O
nice -n 5 orbita/target/release/orbita cst3-run --phase smoke --seed $S --weights artifacts/orbita/dev/weights_0xA400.json --out $O --arms ${ARMS:-cst} --na ${NA:-192} --nb ${NB:-16} $F > $O/log_$S.txt 2>&1
