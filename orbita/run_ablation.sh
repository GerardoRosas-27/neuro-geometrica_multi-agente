#!/usr/bin/env bash
# Uso: orbita/run_ablation.sh A|B|C dev|confirm [JOBS]  (reanudable; usa los datos exportados de ÓRBITA)
set -u
V=$1; PH=$2; J=${3:-8}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
B=$ROOT/orbita/target/release/orbita
P=${ORBITA_PY:-/workspace/orbita-venv/bin/python}
D=${ORBITA_DATA:-/workspace/orbita-data}/$PH
W=${ORBITA_DATA:-/workspace/orbita-data}/abl_$V/$PH
OUT=$ROOT/artifacts/orbita/ablations/$V/$PH
mkdir -p $W $OUT
if [ "$PH" = dev ]; then SEEDS=$(for i in $(seq 0 15); do printf "0xA4%02X " $i; done); else SEEDS=$(for i in $(seq 0 15); do printf "0xB4%02X " $i; done); fi
one() {
  s=$1
  [ -f $OUT/metrics_$s.csv ] && return 0
  [ -f $D/$s/meta.json ] || $B export --seed $s --out $D/$s > /dev/null
  if [ ! -f $W/w_$s.json ]; then
    /usr/bin/time -f "wall_s=%e" -o $W/time_$s.txt env OMP_NUM_THREADS=1 nice -n 10 $P -W ignore $ROOT/orbita/train_orbita.py --data $D/$s --out $W/w_$s.json --seed $s \
      --steps 256000 --lam_h 0.01 --mh_x 1 --variant $V > $W/log_$s.txt 2>&1
  fi
  [ -x $B ] || return 0     # evaluación diferida si el binario aún no existe
  $B parity --weights $W/w_$s.json --probe $W/w_$s.json.probe.json > $OUT/parity_$s.json
  $B eval --seed $s --weights $W/w_$s.json --meta $D/$s/meta.json --out $OUT --label ABL_$V >> $W/log_$s.txt 2>&1
  cp $W/w_$s.json $OUT/weights_$s.json; cp $W/time_$s.txt $OUT/ 2>/dev/null
}
export -f one; export B P D W OUT ROOT V
echo $SEEDS | tr ' ' '\n' | grep . | xargs -P $J -I{} bash -c 'one {}'
echo "ablation $V $PH done"
