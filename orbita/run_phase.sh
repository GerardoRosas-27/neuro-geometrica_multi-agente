#!/usr/bin/env bash
# Uso: orbita/run_phase.sh dev|confirm [JOBS]   (reanudable: salta seeds con métricas y reanuda checkpoints)
set -u
PH=$1; J=${2:-8}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
B=$ROOT/orbita/target/release/orbita
P=${ORBITA_PY:-/workspace/orbita-venv/bin/python}
D=${ORBITA_DATA:-/workspace/orbita-data}/$PH
OUT=$ROOT/artifacts/orbita/$PH
mkdir -p $D $OUT
if [ "$PH" = dev ]; then SEEDS=$(for i in $(seq 0 15); do printf "0xA4%02X " $i; done); else SEEDS=$(for i in $(seq 0 15); do printf "0xB4%02X " $i; done); fi
one() {
  s=$1
  [ -f $OUT/metrics_$s.csv ] && return 0
  [ -f $D/$s/meta.json ] || $B export --seed $s --out $D/$s > /dev/null
  [ -f $D/w_$s.json ] || OMP_NUM_THREADS=1 $P -W ignore $ROOT/orbita/train_orbita.py --data $D/$s --out $D/w_$s.json --seed $s \
      --steps 256000 --lam_h 0.01 --mh_x 1 > $D/log_$s.txt 2>&1
  $B parity --weights $D/w_$s.json --probe $D/w_$s.json.probe.json > $OUT/parity_$s.json
  $B eval --seed $s --weights $D/w_$s.json --meta $D/$s/meta.json --out $OUT --label ORBITA >> $D/log_$s.txt 2>&1
  cp $D/w_$s.json $OUT/weights_$s.json; cp $D/$s/meta.json $OUT/meta_$s.json
}
export -f one; export B P D OUT ROOT
echo $SEEDS | tr ' ' '\n' | grep . | xargs -P $J -I{} bash -c 'one {}'
echo "phase $PH done"
