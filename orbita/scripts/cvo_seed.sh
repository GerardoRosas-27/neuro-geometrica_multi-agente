#!/usr/bin/env bash
# Una semilla CST vs ÓRBITA-eq (docs/cst_vs_orbita/PREREG.md + ENMIENDA_1). Uso: cvo_seed.sh dev|confirm 0xA400
set -euo pipefail
PH=$1; S=$2; B=orbita/target/release/orbita; PY=/workspace/orbita-venv/bin/python
A=artifacts/cst_vs_orbita; D=/workspace/cvo-data/$PH/$S; W=$A/$PH/weights_eq_$S.json
mkdir -p $A/$PH $D
[ -f $A/$PH/$S.done ] && exit 0
[ -f $D/train.bin ] || $B cvo-export --seed $S --out $D
[ -f $W ] || $PY orbita/train_orbita.py --data $D --out $W --seed $S --steps 3840 --batch 32 --lr 3e-3 --lr_final 3e-5 --lam_h 0.01 --mh_x 1
$B parity --weights $W --probe $W.probe.json > $A/$PH/parity_eq_$S.json
python3 -c "import json,sys;d=json.load(open('$A/$PH/parity_eq_$S.json'));sys.exit(0 if d['max_l2_diff']<1e-9 else 1)"
$B cvo-eval --phase $PH --seed $S --weights $W --orig artifacts/orbita/$PH/weights_$S.json --out $A
